#!/usr/bin/env python3
"""
Hidden-interpolation FID with NO pooling in nearest-neighbour search.

Protocol:
    image -> VAE -> noisy SiT input -> layer-depth hidden h [B,T,D]
    -> build a deterministic signed-hash sketch from the FULL flattened hidden
       (all T*D coordinates contribute; no mean/max pooling)
    -> exact cosine search in sketch space for top-M candidates
    -> exact full-hidden cosine/L2 reranking over those candidates
    -> interpolate the selected full hidden tensors
    -> mean-pool ONLY after NN selection/interpolation, for Frechet distance

This is still an approximate global NN method because the candidate shortlist is
obtained from a compressed full-hidden sketch. It is designed for 100k/200k
reference x 50k query, where exact exhaustive full-hidden search is impractical.
Use audit_fullhidden_sketch_recall.py on a sampled bank to measure recall@M.

Requires eval_sit_hidden_ifid_100k50k_v3.py in the same directory.
v7 performs exhaustive exact full-hidden nearest-neighbor search over every
reference item. The reference hidden bank stays on an 80 GiB A800; search
streams contiguous reference chunks and never pools before NN/interpolation.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
import eval_sit_hidden_ifid_100k50k_v3 as base




@dataclass
class FullReferenceCache:
    full_path: str
    labels_path: str
    metadata_path: str
    n: int
    t: int
    d: int

    def open_full(self, mode: str = "r") -> np.memmap:
        return np.memmap(
            self.full_path, mode=mode, dtype=np.float16, shape=(self.n, self.t, self.d)
        )

    def open_labels(self, mmap_mode: str = "r") -> np.ndarray:
        return np.load(self.labels_path, mmap_mode=mmap_mode)


def full_cache_paths(cache_dir: str) -> Tuple[str, str, str]:
    return (
        os.path.join(cache_dir, "reference_hidden_fp16.mmap"),
        os.path.join(cache_dir, "reference_labels_i64.npy"),
        os.path.join(cache_dir, "metadata.json"),
    )


@torch.no_grad()
def build_reference_cache_no_pool(
    args: argparse.Namespace,
    loaded: base.LoadedModels,
    ref_loader: DataLoader,
    n_ref: int,
    device: torch.device,
) -> FullReferenceCache:
    """Build/reuse full hidden + labels only. No pooled NN descriptor is created."""
    os.makedirs(args.cache_dir, exist_ok=True)
    full_path, labels_path, metadata_path = full_cache_paths(args.cache_dir)
    expected_meta = {
        "exp_path": os.path.abspath(args.exp_path),
        "train_steps": str(args.train_steps),
        "ckpt_key": args.ckpt_key,
        "depth": args.depth,
        "t_value": args.t,
        "seed": args.seed,
        "path_type": loaded.path_type,
        "n": n_ref,
        "tokens": loaded.token_count,
        "dim": loaded.hidden_dim,
    }

    if args.reuse_cache and all(
        os.path.exists(x) for x in (full_path, labels_path, metadata_path)
    ):
        try:
            old_meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except Exception:
            old_meta = None
        if old_meta == expected_meta:
            print(f"Reusing full-hidden reference cache: {args.cache_dir}")
            return FullReferenceCache(
                full_path, labels_path, metadata_path,
                n_ref, loaded.token_count, loaded.hidden_dim,
            )
        print("Existing full-hidden cache metadata differs; rebuilding cache.")

    full_mm = np.memmap(
        full_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_ref, loaded.token_count, loaded.hidden_dim),
    )
    label_mm = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int64, shape=(n_ref,)
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 100_003)
    offset = 0
    pbar = tqdm(total=n_ref, desc="Extract reference full h")
    for images, labels, _names in ref_loader:
        if offset >= n_ref:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        remaining = n_ref - offset
        if images.shape[0] > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        with base.make_autocast(device, args.amp_dtype):
            z = base.encode_vae(
                loaded.vae, images, labels,
                vae_1d=getattr(loaded.sit, "vae_1d", False),
            )
            h = base.extract_hidden_from_raw_latent(
                loaded.sit,
                z,
                labels,
                depth=args.depth,
                t_value=args.t,
                path_type=loaded.path_type,
                generator=generator,
            )

        b = h.shape[0]
        full_mm[offset:offset+b] = h.float().cpu().numpy().astype(np.float16)
        label_mm[offset:offset+b] = labels.cpu().numpy().astype(np.int64)
        offset += b
        pbar.update(b)
        del images, labels, z, h
    pbar.close()

    if offset != n_ref:
        raise RuntimeError(f"Reference loader yielded {offset}, expected {n_ref}")
    full_mm.flush(); label_mm.flush()
    del full_mm, label_mm
    Path(metadata_path).write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return FullReferenceCache(
        full_path, labels_path, metadata_path,
        n_ref, loaded.token_count, loaded.hidden_dim,
    )


class StructuredFullHiddenSketch:
    """Deterministic signed feature-hashing projection of full [T,D] hidden.

    Every original coordinate participates exactly once. We independently
    permute token/channel axes, multiply by fixed Rademacher signs, flatten, and
    sum coordinates into `sketch_dim` buckets. This is NOT spatial pooling.
    """

    def __init__(
        self,
        tokens: int,
        hidden_dim: int,
        sketch_dim: int,
        seed: int,
        device: torch.device,
    ) -> None:
        self.tokens = int(tokens)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = self.tokens * self.hidden_dim
        self.sketch_dim = int(sketch_dim)
        self.device = device

        if self.sketch_dim <= 0:
            raise ValueError("sketch_dim must be positive")
        if self.sketch_dim > self.feature_dim:
            raise ValueError(
                f"sketch_dim={self.sketch_dim} exceeds full feature dim={self.feature_dim}"
            )

        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.token_perm = torch.randperm(self.tokens, generator=g).to(device)
        self.channel_perm = torch.randperm(self.hidden_dim, generator=g).to(device)
        signs = torch.randint(
            0,
            2,
            (self.tokens, self.hidden_dim),
            generator=g,
            dtype=torch.int8,
        )
        signs = signs.mul_(2).sub_(1)
        self.signs = signs.to(device=device, dtype=torch.float16)
        self.n_buckets_per_output = math.ceil(self.feature_dim / self.sketch_dim)
        self.padded_dim = self.n_buckets_per_output * self.sketch_dim

    @torch.no_grad()
    def transform(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got {tuple(hidden.shape)}")
        if hidden.shape[1] != self.tokens or hidden.shape[2] != self.hidden_dim:
            raise ValueError(
                f"Shape mismatch: expected [B,{self.tokens},{self.hidden_dim}], "
                f"got {tuple(hidden.shape)}"
            )

        # Do not normalize tokens independently: full-hidden cosine in the final
        # reranker is computed after flattening the raw residual stream.
        x = hidden.to(dtype=torch.float16)
        x = x.index_select(1, self.token_perm).index_select(2, self.channel_perm)
        x = x * self.signs.unsqueeze(0)
        flat = x.reshape(x.shape[0], -1)
        if self.padded_dim != self.feature_dim:
            flat = F.pad(flat, (0, self.padded_dim - self.feature_dim))
        sketch = flat.reshape(flat.shape[0], self.n_buckets_per_output, self.sketch_dim)
        sketch = sketch.sum(dim=1) / math.sqrt(float(self.n_buckets_per_output))
        return F.normalize(sketch.float(), dim=1)


class TorchSketchIndex:
    """Exact cosine search over compressed full-hidden sketches."""

    def __init__(
        self,
        reference_sketch: np.ndarray,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device
        ref_np = np.asarray(reference_sketch, dtype=np.float16)
        ref = torch.from_numpy(ref_np).to(device=device, dtype=dtype)
        ref = F.normalize(ref.float(), dim=1).to(dtype)
        self.reference_t = ref.transpose(0, 1).contiguous()
        gib = self.reference_t.numel() * self.reference_t.element_size() / (1024 ** 3)
        print(
            "Full-hidden sketch index: exact cosine in sketch space; "
            f"reference={(ref.shape[0], ref.shape[1])}, resident={gib:.2f} GiB"
        )
        del ref

    @torch.no_grad()
    def search_tensor(self, query_sketch: torch.Tensor, k: int) -> torch.Tensor:
        """Return shortlist indices directly on the index GPU."""
        q = F.normalize(query_sketch.float(), dim=1).to(self.reference_t.dtype)
        scores = q @ self.reference_t
        return torch.topk(scores, k=k, dim=1, largest=True).indices

    @torch.no_grad()
    def search(self, query_sketch: torch.Tensor, k: int) -> np.ndarray:
        inds = self.search_tensor(query_sketch, k)
        return inds.cpu().numpy().astype(np.int64, copy=False)


def sketch_paths(cache_dir: str, sketch_dim: int, sketch_seed: int) -> Tuple[str, str]:
    stem = f"reference_fullsketch_d{sketch_dim}_seed{sketch_seed}"
    return (
        os.path.join(cache_dir, stem + "_f16.npy"),
        os.path.join(cache_dir, stem + "_metadata.json"),
    )


@torch.no_grad()
def build_or_load_reference_sketch(
    args: argparse.Namespace,
    cache: FullReferenceCache,
    sketcher: StructuredFullHiddenSketch,
    device: torch.device,
) -> np.ndarray:
    sketch_path, meta_path = sketch_paths(args.cache_dir, args.sketch_dim, args.sketch_seed)
    expected = {
        "exp_path": os.path.abspath(args.exp_path),
        "train_steps": str(args.train_steps),
        "ckpt_key": args.ckpt_key,
        "depth": args.depth,
        "t_value": args.t,
        "seed": args.seed,
        "path_type": args.path_type,
        "n": cache.n,
        "tokens": cache.t,
        "hidden_dim": cache.d,
        "sketch_dim": args.sketch_dim,
        "sketch_seed": args.sketch_seed,
        "protocol": "signed_hash_of_full_hidden_no_pool",
    }

    if args.reuse_cache and os.path.exists(sketch_path) and os.path.exists(meta_path):
        try:
            old = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except Exception:
            old = None
        if old == expected:
            print(f"Reusing full-hidden sketch cache: {sketch_path}")
            return np.load(sketch_path, mmap_mode="r")

    reference_full = cache.open_full("r")
    sketch_mm = np.lib.format.open_memmap(
        sketch_path,
        mode="w+",
        dtype=np.float16,
        shape=(cache.n, args.sketch_dim),
    )

    pbar = tqdm(total=cache.n, desc="Build full-hidden sketch")
    for start in range(0, cache.n, args.sketch_build_batch):
        end = min(start + args.sketch_build_batch, cache.n)
        block_np = np.asarray(reference_full[start:end], dtype=np.float16).copy()
        block = torch.from_numpy(block_np).to(device=device, non_blocking=True)
        sketch = sketcher.transform(block)
        sketch_mm[start:end] = sketch.cpu().numpy().astype(np.float16)
        pbar.update(end - start)
        del block_np, block, sketch
    pbar.close()
    sketch_mm.flush()
    del sketch_mm
    Path(meta_path).write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return np.load(sketch_path, mmap_mode="r")



def open_reference_for_queries(
    args: argparse.Namespace,
    cache: FullReferenceCache,
    device: torch.device,
):
    """Open or preload the full reference bank for query-time reranking.

    Modes:
      * GPU: fastest; recommended for 50k reference on an 80 GiB A800.
      * RAM: avoids random disk I/O but still transfers shortlisted candidates.
      * mmap: lowest memory, usually I/O-bound.
    """
    full_mm = cache.open_full("r")
    labels_mm = cache.open_labels("r")

    if args.preload_reference_gpu:
        if device.type != "cuda":
            raise ValueError("--preload-reference-gpu requires a CUDA device")
        gib = cache.n * cache.t * cache.d * np.dtype(np.float16).itemsize / (1024 ** 3)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / (1024 ** 3)
        total_gib = total_bytes / (1024 ** 3)
        reserve_gib = float(args.gpu_memory_reserve_gib)
        print(
            f"GPU full-reference preload requested: bank={gib:.2f} GiB, "
            f"free={free_gib:.2f}/{total_gib:.2f} GiB, reserve={reserve_gib:.2f} GiB"
        )
        if free_gib < gib + reserve_gib:
            raise RuntimeError(
                f"Not enough free GPU memory: need roughly {gib + reserve_gib:.2f} GiB, "
                f"but only {free_gib:.2f} GiB is free. Reduce batch size/reserve or "
                "disable --preload-reference-gpu."
            )

        full_gpu = torch.empty(
            (cache.n, cache.t, cache.d),
            dtype=torch.float16,
            device=device,
        )
        pbar = tqdm(total=cache.n, desc="Preload reference h to GPU")
        for start_i in range(0, cache.n, args.preload_chunk):
            end_i = min(start_i + args.preload_chunk, cache.n)
            block_np = np.asarray(full_mm[start_i:end_i], dtype=np.float16)
            if not block_np.flags.c_contiguous:
                block_np = np.ascontiguousarray(block_np)
            block_cpu = torch.from_numpy(block_np)
            full_gpu[start_i:end_i].copy_(block_cpu, non_blocking=False)
            pbar.update(end_i - start_i)
        pbar.close()
        labels_gpu = torch.from_numpy(
            np.asarray(labels_mm, dtype=np.int64).copy()
        ).to(device=device)
        del full_mm, labels_mm
        torch.cuda.synchronize(device)
        allocated_gib = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"Reference bank is GPU-resident; total torch allocated={allocated_gib:.2f} GiB")
        return full_gpu, labels_gpu

    if not args.preload_reference_ram:
        try:
            full_mm._mmap.madvise(mmap.MADV_RANDOM)
        except Exception:
            pass
        return full_mm, labels_mm

    gib = cache.n * cache.t * cache.d * np.dtype(np.float16).itemsize / (1024 ** 3)
    print(
        f"Preloading full reference hidden into system RAM: "
        f"shape=({cache.n},{cache.t},{cache.d}), about {gib:.2f} GiB"
    )
    full_ram = np.empty((cache.n, cache.t, cache.d), dtype=np.float16)
    pbar = tqdm(total=cache.n, desc="Preload reference h to RAM")
    for start_i in range(0, cache.n, args.preload_chunk):
        end_i = min(start_i + args.preload_chunk, cache.n)
        full_ram[start_i:end_i] = full_mm[start_i:end_i]
        pbar.update(end_i - start_i)
    pbar.close()
    labels_ram = np.asarray(labels_mm, dtype=np.int64).copy()
    del full_mm, labels_mm
    return full_ram, labels_ram


@torch.no_grad()
def rerank_full_hidden_fast(
    query_h: torch.Tensor,
    candidate_indices,
    reference_full,
    metric: str,
    device: torch.device,
):
    """Exact full-hidden reranking within the shortlist.

    If the reference bank is GPU-resident, both candidate gathering and
    distance computation stay entirely on GPU.
    """
    if isinstance(reference_full, torch.Tensor):
        if isinstance(candidate_indices, np.ndarray):
            candidate_indices_t = torch.from_numpy(candidate_indices).to(
                device=device, dtype=torch.long
            )
        else:
            candidate_indices_t = candidate_indices.to(device=device, dtype=torch.long)
        candidates = reference_full[candidate_indices_t]
        candidate_indices_np = None
    else:
        if isinstance(candidate_indices, torch.Tensor):
            candidate_indices_np = candidate_indices.cpu().numpy().astype(
                np.int64, copy=False
            )
        else:
            candidate_indices_np = candidate_indices
        candidate_np = np.asarray(
            reference_full[candidate_indices_np], dtype=np.float16
        )
        if not candidate_np.flags.c_contiguous:
            candidate_np = np.ascontiguousarray(candidate_np)
        candidates = torch.from_numpy(candidate_np).to(device=device)
        candidate_indices_t = None

    q = query_h.float().reshape(query_h.shape[0], -1)
    c = candidates.float().reshape(candidates.shape[0], candidates.shape[1], -1)

    if metric == "cosine":
        # Avoid materializing normalized copies of the very large candidate tensor.
        dot = torch.einsum("bd,bkd->bk", q, c)
        q_norm = q.norm(dim=-1, keepdim=True)
        c_norm = c.norm(dim=-1)
        scores = dot / (q_norm * c_norm + 1e-8)
        best_pos = scores.argmax(dim=1)
    elif metric == "l2":
        scores = ((c - q[:, None, :]) ** 2).mean(dim=-1)
        best_pos = scores.argmin(dim=1)
    else:
        raise ValueError(metric)

    rows = torch.arange(query_h.shape[0], device=device)
    best_h = candidates[rows, best_pos]
    if candidate_indices_t is not None:
        best_global_idx_t = candidate_indices_t[rows, best_pos]
        return best_h, best_global_idx_t

    best_pos_cpu = best_pos.cpu().numpy()
    best_global_idx = candidate_indices_np[
        np.arange(query_h.shape[0]), best_pos_cpu
    ]
    return best_h, best_global_idx.astype(np.int64, copy=False)



@torch.no_grad()
def build_reference_search_stats(
    reference_full: torch.Tensor,
    metric: str,
    chunk_size: int,
) -> torch.Tensor:
    """Precompute one scalar norm/statistic per reference item."""
    if not isinstance(reference_full, torch.Tensor) or reference_full.device.type != "cuda":
        raise ValueError("Exact full-NN mode requires a GPU-resident reference bank")

    ref_flat = reference_full.reshape(reference_full.shape[0], -1)
    stats = torch.empty(reference_full.shape[0], dtype=torch.float32, device=reference_full.device)
    desc = "Reference cosine norms" if metric == "cosine" else "Reference squared norms"
    pbar = tqdm(total=reference_full.shape[0], desc=desc)

    for start in range(0, reference_full.shape[0], chunk_size):
        end = min(start + chunk_size, reference_full.shape[0])
        block = ref_flat[start:end].float()
        if metric == "cosine":
            stats[start:end] = block.norm(dim=1)
        elif metric == "l2":
            stats[start:end] = (block * block).sum(dim=1)
        else:
            raise ValueError(metric)
        pbar.update(end - start)
        del block
    pbar.close()
    return stats


@torch.no_grad()
def exact_full_hidden_nn_gpu(
    query_h: torch.Tensor,
    reference_full: torch.Tensor,
    reference_stats: torch.Tensor,
    metric: str,
    ref_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search every reference using the complete unpooled T x D hidden tensor.

    Only the current reference chunk's scores are materialized. The function
    maintains one best score and index per query, so setting the candidate
    count to the full bank never creates [B, N, T, D].
    """
    if reference_full.device != query_h.device:
        raise ValueError("query and reference must be on the same GPU")

    b = query_h.shape[0]
    q = query_h.reshape(b, -1).float()
    ref_flat = reference_full.reshape(reference_full.shape[0], -1)

    if metric == "cosine":
        q_stat = q.norm(dim=1)
        best_score = torch.full((b,), -float("inf"), dtype=torch.float32, device=q.device)
    elif metric == "l2":
        q_stat = (q * q).sum(dim=1)
        best_score = torch.full((b,), float("inf"), dtype=torch.float32, device=q.device)
    else:
        raise ValueError(metric)

    best_index = torch.zeros((b,), dtype=torch.long, device=q.device)

    for start in range(0, reference_full.shape[0], ref_chunk_size):
        end = min(start + ref_chunk_size, reference_full.shape[0])

        # FP32/TF32 GEMM avoids FP16-range overflow for long flattened vectors.
        r = ref_flat[start:end].float()
        dot = q @ r.transpose(0, 1)

        if metric == "cosine":
            scores = dot / (q_stat[:, None] * reference_stats[start:end][None, :] + 1e-8)
            chunk_score, chunk_pos = scores.max(dim=1)
            update = chunk_score > best_score
        else:
            scores = (
                q_stat[:, None]
                + reference_stats[start:end][None, :]
                - 2.0 * dot
            )
            chunk_score, chunk_pos = scores.min(dim=1)
            update = chunk_score < best_score

        best_score = torch.where(update, chunk_score, best_score)
        best_index = torch.where(update, chunk_pos + start, best_index)
        del r, dot, scores, chunk_score, chunk_pos, update

    nn_h = reference_full[best_index]
    return nn_h, best_index


@torch.no_grad()
def evaluate_queries(
    args: argparse.Namespace,
    loaded: base.LoadedModels,
    val_loader: DataLoader,
    n_val: int,
    cache: FullReferenceCache,
    device: torch.device,
) -> Dict[str, Any]:
    reference_full, reference_labels = open_reference_for_queries(args, cache, device)
    if not isinstance(reference_full, torch.Tensor):
        raise RuntimeError(
            "Exact full-NN mode requires --preload-reference-gpu; mmap/RAM modes are disabled."
        )
    reference_stats = build_reference_search_stats(
        reference_full, args.nn_metric, args.exact_ref_chunk
    )

    arrays: Dict[str, List[np.ndarray]] = {
        "src_raw": [], "src_ln": [],
        "main_raw": [], "main_ln": [],
        "a0_raw": [], "a0_ln": [],
        "a1_raw": [], "a1_ln": [],
        "rand_raw": [], "rand_ln": [],
    }
    selected_indices: List[np.ndarray] = []
    same_class_total = 0
    seen = 0
    full_l2_sum = 0.0
    pooled_l2_sum = 0.0

    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 200_003)
    random_generator = np.random.default_rng(args.seed + 300_003)

    pbar = tqdm(total=n_val, desc="Evaluate query h")
    for images, labels, _names in val_loader:
        if seen >= n_val:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        remaining = n_val - seen
        if images.shape[0] > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        with base.make_autocast(device, args.amp_dtype):
            z = base.encode_vae(
                loaded.vae,
                images,
                labels,
                vae_1d=getattr(loaded.sit, "vae_1d", False),
            )
            query_h = base.extract_hidden_from_raw_latent(
                loaded.sit,
                z,
                labels,
                depth=args.depth,
                t_value=args.t,
                path_type=loaded.path_type,
                generator=noise_generator,
            )

        query_h = query_h.float()

        # Exhaustive NN over ALL references using the full unpooled hidden.
        nn_h, nn_indices = exact_full_hidden_nn_gpu(
            query_h,
            reference_full,
            reference_stats,
            args.nn_metric,
            args.exact_ref_chunk,
        )
        nn_h = nn_h.float()
        if isinstance(nn_indices, torch.Tensor):
            selected_indices.append(nn_indices.detach().cpu().numpy())
            nn_labels = reference_labels[nn_indices]
        else:
            selected_indices.append(nn_indices)
            nn_labels = torch.from_numpy(
                np.asarray(reference_labels[nn_indices], dtype=np.int64)
            ).to(device)
        same_class_total += int((nn_labels == labels).sum().item())

        main_h = (1.0 - args.alpha) * query_h + args.alpha * nn_h
        a0_h = query_h
        a1_h = nn_h

        random_indices = random_generator.integers(
            0, cache.n, size=query_h.shape[0], endpoint=False
        )
        if isinstance(reference_full, torch.Tensor):
            random_indices_t = torch.from_numpy(random_indices).to(
                device=device, dtype=torch.long
            )
            random_h = reference_full[random_indices_t].float()
        else:
            random_np = np.asarray(reference_full[random_indices], dtype=np.float16)
            random_h = torch.from_numpy(random_np).to(device=device).float()
        rand_intp_h = 0.5 * query_h + 0.5 * random_h

        # Pooling starts only here, after NN selection and full-hidden interpolation.
        arrays["src_raw"].append(base.pooled_raw(query_h).cpu().numpy())
        arrays["src_ln"].append(base.pooled_token_layernorm(query_h).cpu().numpy())
        arrays["main_raw"].append(base.pooled_raw(main_h).cpu().numpy())
        arrays["main_ln"].append(base.pooled_token_layernorm(main_h).cpu().numpy())
        arrays["a0_raw"].append(base.pooled_raw(a0_h).cpu().numpy())
        arrays["a0_ln"].append(base.pooled_token_layernorm(a0_h).cpu().numpy())
        arrays["a1_raw"].append(base.pooled_raw(a1_h).cpu().numpy())
        arrays["a1_ln"].append(base.pooled_token_layernorm(a1_h).cpu().numpy())
        arrays["rand_raw"].append(base.pooled_raw(rand_intp_h).cpu().numpy())
        arrays["rand_ln"].append(base.pooled_token_layernorm(rand_intp_h).cpu().numpy())

        full_l2_sum += float(query_h.flatten(1).norm(dim=1).sum().item())
        pooled_l2_sum += float(base.pooled_raw(query_h).norm(dim=1).sum().item())

        b = query_h.shape[0]
        seen += b
        pbar.update(b)
        del (
            images, labels, z, query_h, nn_h,
            main_h, a0_h, a1_h, random_h, rand_intp_h, nn_indices, nn_labels,
        )
    pbar.close()

    if seen != n_val:
        raise RuntimeError(f"Validation loader yielded {seen}, expected {n_val}")

    features = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    src_raw = features["src_raw"]
    src_ln = features["src_ln"]
    half = n_val // 2

    metrics = {
        "main_hidden_fid_raw": base.frechet_from_features(src_raw, features["main_raw"]),
        "main_hidden_fid_token_ln": base.frechet_from_features(src_ln, features["main_ln"]),
        "sanity_alpha0_fid_raw": base.frechet_from_features(src_raw, features["a0_raw"]),
        "sanity_alpha0_fid_token_ln": base.frechet_from_features(src_ln, features["a0_ln"]),
        "sanity_alpha1_fid_raw": base.frechet_from_features(src_raw, features["a1_raw"]),
        "sanity_alpha1_fid_token_ln": base.frechet_from_features(src_ln, features["a1_ln"]),
        "random_neighbor_alpha05_fid_raw": base.frechet_from_features(src_raw, features["rand_raw"]),
        "random_neighbor_alpha05_fid_token_ln": base.frechet_from_features(src_ln, features["rand_ln"]),
        "nn_same_class_rate": same_class_total / float(n_val),
        "query_hidden_mean_full_l2": full_l2_sum / float(n_val),
        "query_hidden_mean_pooled_l2": pooled_l2_sum / float(n_val),
        "query_pooled_cov_trace_raw": base.covariance_trace(src_raw),
        "query_pooled_cov_trace_token_ln": base.covariance_trace(src_ln),
        "split_half_fid_raw": base.frechet_from_features(src_raw[:half], src_raw[half:2 * half]),
        "split_half_fid_token_ln": base.frechet_from_features(src_ln[:half], src_ln[half:2 * half]),
    }

    if abs(metrics["sanity_alpha0_fid_raw"]) > args.alpha0_tolerance:
        raise RuntimeError(
            f"alpha=0 raw FID should be ~0, got {metrics['sanity_alpha0_fid_raw']}"
        )

    if args.save_pooled_features:
        np.savez_compressed(
            os.path.join(args.output_dir, "pooled_features.npz"),
            **features,
            selected_reference_indices=np.concatenate(selected_indices),
        )

    return {
        "settings": {
            "exp_path": os.path.abspath(args.exp_path),
            "train_steps": str(args.train_steps),
            "ckpt_key": args.ckpt_key,
            "depth": args.depth,
            "t_input_before_shift": args.t,
            "path_type": loaded.path_type,
            "alpha": args.alpha,
            "nn_metric_full_hidden": args.nn_metric,
            "retrieval_protocol": "exhaustive_exact_full_hidden_all_references",
            "coarse_representation": None,
            "coarse_topk": None,
            "sketch_dim": None,
            "sketch_seed": None,
            "exact_reference_chunk": args.exact_ref_chunk,
            "n_reference": cache.n,
            "n_query": n_val,
            "tokens": cache.t,
            "hidden_dim": cache.d,
            "seed": args.seed,
            "preload_reference_ram": bool(args.preload_reference_ram),
            "preload_reference_gpu": bool(args.preload_reference_gpu),
            "query_batch_size": args.batch_size,
            "dataloader_prefetch_factor": args.prefetch_factor,
        },
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-path", required=True)
    p.add_argument("--train-steps", default="400000")
    p.add_argument("--ckpt-key", choices=["ema", "model"], default="ema")
    p.add_argument("--dataset-ref", required=True)
    p.add_argument("--dataset-val", required=True)
    p.add_argument("--small-ref", type=int, default=200_000)
    p.add_argument("--small-val", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--t", type=float, default=0.1)
    p.add_argument("--path-type", choices=["linear", "cosine"], default="linear")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--nn-metric", choices=["cosine", "l2"], default="cosine")
    p.add_argument("--coarse-topk", type=int, default=64, help="Ignored in exact-full-NN mode")
    p.add_argument(
        "--exact-ref-chunk",
        type=int,
        default=4096,
        help="Number of GPU-resident references processed per exact-search GEMM chunk.",
    )
    p.add_argument("--sketch-dim", type=int, default=4096)
    p.add_argument("--sketch-seed", type=int, default=9187)
    p.add_argument("--sketch-build-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp-dtype", choices=["no", "fp16", "bf16"], default="no")
    p.add_argument("--expected-tokens", type=int, default=0)
    p.add_argument("--expected-dim", type=int, default=1152)
    p.add_argument("--cache-dir", required=True)
    p.add_argument(
        "--preload-reference-ram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy full reference hidden from memmap into system RAM before query evaluation.",
    )
    p.add_argument(
        "--preload-reference-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the complete FP16 reference hidden bank resident on GPU.",
    )
    p.add_argument(
        "--gpu-memory-reserve-gib",
        type=float,
        default=20.0,
        help="Required free GPU memory left after loading the reference bank.",
    )
    p.add_argument("--preload-chunk", type=int, default=512)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-pooled-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--alpha0-tolerance", type=float, default=1e-6)

    # Legacy options required by base.load_models/build_reference_cache.
    p.add_argument("--coarse-backend", default="unused-fullsketch")
    p.add_argument("--faiss-gpu", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--allow-torch-coarse-fallback", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--coarse-chunk", type=int, default=20_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.t <= 1.0):
        raise ValueError("--t must be in [0,1]")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")
    if args.small_ref <= 0 or args.small_val <= 1:
        raise ValueError("--small-ref and --small-val must be positive")
    if not args.preload_reference_gpu:
        raise ValueError("Exact full-NN mode requires --preload-reference-gpu")
    if args.preload_reference_gpu and args.preload_reference_ram:
        raise ValueError(
            "--preload-reference-gpu and --preload-reference-ram are mutually exclusive"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ref_set = ImageNetValDataset(args.dataset_ref, transform, args.small_ref)
    val_set = ImageNetValDataset(args.dataset_val, transform, args.small_val)
    n_ref = min(args.small_ref, len(ref_set))
    n_val = min(args.small_val, len(val_set))

    loader_extra = {}
    if args.num_workers > 0:
        loader_extra["persistent_workers"] = True
        loader_extra["prefetch_factor"] = args.prefetch_factor

    ref_loader = DataLoader(
        ref_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        **loader_extra,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        **loader_extra,
    )

    loaded = base.load_models(args, device)
    cache = build_reference_cache_no_pool(args, loaded, ref_loader, n_ref, device)

    result = evaluate_queries(
        args,
        loaded,
        val_loader,
        n_val,
        cache,
        device,
    )

    result_path = os.path.join(args.output_dir, "hidden_ifid_metrics.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Hidden interpolation metrics (exhaustive exact full-hidden NN) ===")
    for key, value in result["metrics"].items():
        print(f"{key}={value:.8f}")
    print(f"Saved: {result_path}")

    if not args.keep_cache:
        shutil.rmtree(args.cache_dir)
        print(f"Removed scratch cache: {args.cache_dir}")


if __name__ == "__main__":
    main()
