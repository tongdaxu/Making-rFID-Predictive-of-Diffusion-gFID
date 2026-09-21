#!/usr/bin/env python3
"""
First-pass evaluation for SiT layer-8 hidden interpolation geometry.

Main experiment (keeps the teacher's requested flow):
    image -> VAE latent -> SiT training-time normalization/noising
    -> hidden after `depth` blocks
    -> nearest neighbour selected with full, unpooled hidden features
    -> full-hidden linear interpolation
    -> mean pool only for Frechet-distance computation

Scalability:
    Exact full-hidden search over 100k x 50k x (256*1152) is impractical.
    This script therefore uses:
      1) pooled features only to retrieve coarse top-M candidates;
      2) full unpooled [T,D] hidden features to rerank those candidates.
    The final NN is determined by the unpooled hidden feature.

Extra diagnostics do not change the main pipeline:
    - LayerNorm-robust hidden FID
    - NN same-class rate
    - alpha=0 / alpha=1 sanity checks
    - random-neighbour alpha=0.5 baseline
    - hidden norms / covariance trace / split-half finite-sample FID

Token count is inferred dynamically from the actual VAE latent size and SiT patch size.
Use --expected-tokens N only when you intentionally want to assert a fixed N.

Expected repository layout/imports:
    from ifid.dataset import ImageNetValDataset
    from ifid.sit.sit import SiT_models
    from ifid.vae.utils import instantiate_from_config
    from ifid.fid.fid import calculate_frechet_distance

The model/EMA loading follows the project's generate.py pattern. REPA projector tensors are ignored because the metric reads the pre-projector backbone hidden state.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.fid.fid import calculate_frechet_distance
from ifid.sit.sit import SiT_models
from ifid.vae.utils import instantiate_from_config


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def load_json_namespace(path: str) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        return SimpleNamespace(**json.load(f))


def cfg_get(config: SimpleNamespace, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def canonicalize_sit_model_name(name: Any) -> str:
    """Map historical args.json spellings to keys used by SiT_models.

    Some experiments recorded names such as ``SiT_XL/1D`` while the current
    registry uses ``SiT-XL/1D``.  They denote the same architecture.
    """
    value = str(name)
    aliases = {
        "SiT_XL/": "SiT-XL/",
        "SiT_L/": "SiT-L/",
        "SiT_B/": "SiT-B/",
        "SiT_S/": "SiT-S/",
    }
    for old, new in aliases.items():
        if value.startswith(old):
            return new + value[len(old):]
    return value


def unwrap_vae_latents(z: Any) -> torch.Tensor:
    """Normalize common VAE encode outputs to a Tensor."""
    if isinstance(z, (tuple, list)):
        if not z:
            raise RuntimeError("vae.encode returned an empty tuple/list")
        z = z[0]
    if hasattr(z, "latent_dist"):
        z = z.latent_dist.sample()
    if hasattr(z, "sample") and not torch.is_tensor(z):
        sample = z.sample
        z = sample() if callable(sample) else sample
    if not torch.is_tensor(z):
        raise TypeError(f"Unsupported VAE encode output type: {type(z)}")
    return z


def prepare_latents_for_sit(z: Any, *, vae_1d: bool) -> torch.Tensor:
    """Adapt a VAE wrapper output to the actual SiT variant recorded in args.json.

    Spatial SiT expects [B,C,H,W]. Token/1D SiT expects [B,N,C].
    This matters for DETOK/RAE-like wrappers and prevents silently reshaping a
    true 1D-token experiment into a spatial experiment.
    """
    z = unwrap_vae_latents(z)

    if vae_1d:
        if z.ndim == 3:  # expected [B,N,C]
            return z.contiguous()
        if z.ndim == 4:  # [B,C,H,W] -> [B,H*W,C]
            b, c, h, w = z.shape
            return z.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        raise RuntimeError(f"Unsupported 1D-SiT VAE latent shape: {tuple(z.shape)}")

    if z.ndim == 4:
        return z.contiguous()
    if z.ndim == 3:  # [B,N,C] -> [B,C,sqrt(N),sqrt(N)]
        b, n, c = z.shape
        side = int(math.sqrt(n))
        if side * side != n:
            raise RuntimeError(
                f"Token latent length N={n} is not square; shape={tuple(z.shape)}"
            )
        return z.transpose(1, 2).contiguous().view(b, c, side, side)
    raise RuntimeError(f"Unsupported spatial-SiT VAE latent shape: {tuple(z.shape)}")


@torch.no_grad()
def encode_vae(
    vae: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    vae_1d: bool,
) -> torch.Tensor:
    try:
        z = vae.encode(images, labels)
    except TypeError:
        z = vae.encode(images)
    return prepare_latents_for_sit(z, vae_1d=vae_1d)


def make_autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "no":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


# -----------------------------------------------------------------------------
# Load the project's VAE + vanilla SiT exactly like generate.py
# -----------------------------------------------------------------------------


@dataclass
class LoadedModels:
    vae: torch.nn.Module
    sit: torch.nn.Module
    config: SimpleNamespace
    path_type: str
    token_count: int
    hidden_dim: int


@torch.no_grad()
def load_models(args: argparse.Namespace, device: torch.device) -> LoadedModels:
    args_json = os.path.join(args.exp_path, "args.json")
    config = load_json_namespace(args_json)

    vae_config_path = cfg_get(config, "vae_config")
    if not vae_config_path:
        raise RuntimeError(f"vae_config is missing in {args_json}")

    vae_cfg = OmegaConf.load(vae_config_path)
    vae = instantiate_from_config(vae_cfg).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    model_name_raw = cfg_get(config, "model")
    model_name = canonicalize_sit_model_name(model_name_raw)
    if model_name not in SiT_models:
        raise KeyError(
            f"Unknown SiT model {model_name_raw!r} (canonical={model_name!r}); "
            f"available={list(SiT_models)}"
        )
    if model_name != str(model_name_raw):
        print(f"Canonicalized SiT model name: {model_name_raw!r} -> {model_name!r}")
    vae_1d = model_name.endswith("/1D")

    resolution = int(cfg_get(config, "resolution", 256))
    fake_image = torch.zeros(1, 3, resolution, resolution, device=device)
    fake_label = torch.zeros(1, dtype=torch.long, device=device)
    fake_z = encode_vae(vae, fake_image, fake_label, vae_1d=vae_1d)[0]

    if vae_1d:
        if fake_z.ndim != 2:  # [N,C]
            raise RuntimeError(f"Expected prepared 1D latent [N,C], got {tuple(fake_z.shape)}")
        latent_size = int(fake_z.shape[0])
        in_channels = int(fake_z.shape[1])
    else:
        if fake_z.ndim != 3:  # [C,H,W]
            raise RuntimeError(f"Expected prepared spatial latent [C,H,W], got {tuple(fake_z.shape)}")
        in_channels = int(fake_z.shape[0])
        if fake_z.shape[-2] != fake_z.shape[-1]:
            raise RuntimeError(f"Only square spatial latents are supported: {tuple(fake_z.shape)}")
        latent_size = int(fake_z.shape[-1])

    tshift = float(cfg_get(config, "tshift", math.sqrt(float(fake_z.numel()) / 4096.0)))
    block_kwargs = {
        "fused_attn": bool(cfg_get(config, "fused_attn", True)),
        "qk_norm": bool(cfg_get(config, "qk_norm", False)),
    }

    sit = SiT_models[model_name](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=int(cfg_get(config, "num_classes", 1000)),
        class_dropout_prob=float(cfg_get(config, "cfg_prob", 0.1)),
        bn_momentum=float(cfg_get(config, "bn_momentum", 0.1)),
        tshift=tshift,
        **block_kwargs,
    ).to(device)

    ckpt_name = str(args.train_steps).zfill(7) + ".pt"
    ckpt_path = os.path.join(args.exp_path, "checkpoints", ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if args.ckpt_key not in checkpoint:
        raise KeyError(f"Checkpoint has no key {args.ckpt_key!r}; keys={list(checkpoint)}")

    state_dict = checkpoint[args.ckpt_key]
    projector_keys = [k for k in state_dict if k.startswith("projectors.")]
    if projector_keys:
        # The metric uses the backbone residual stream immediately before the
        # REPA projector.  Projector weights are therefore intentionally not
        # instantiated or loaded, while every backbone key remains strict.
        print(
            f"Checkpoint contains {len(projector_keys)} REPA projector tensors; "
            "ignoring projectors.* and loading the SiT backbone strictly."
        )
        state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith("projectors.")
        }

    msg = sit.load_state_dict(state_dict, strict=False)
    if msg.unexpected_keys:
        raise RuntimeError(f"Unexpected non-projector checkpoint keys: {msg.unexpected_keys[:20]}")
    if msg.missing_keys:
        raise RuntimeError(f"Missing SiT backbone checkpoint keys: {msg.missing_keys[:20]}")
    del state_dict, checkpoint

    sit.eval()  # disables class-label dropout
    for p in sit.parameters():
        p.requires_grad_(False)

    # Probe the requested layer shape using the actual training-time input path.
    probe_label = torch.zeros(1, dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    probe_h = extract_hidden_from_raw_latent(
        sit,
        fake_z.unsqueeze(0),
        probe_label,
        depth=args.depth,
        t_value=args.t,
        path_type=cfg_get(config, "path_type", args.path_type),
        generator=generator,
    )
    token_count, hidden_dim = int(probe_h.shape[1]), int(probe_h.shape[2])

    if args.expected_tokens > 0 and token_count != args.expected_tokens:
        raise RuntimeError(
            f"Expected {args.expected_tokens} tokens, got {token_count}. "
            "Check latent size and SiT patch size."
        )
    if args.expected_dim > 0 and hidden_dim != args.expected_dim:
        raise RuntimeError(f"Expected hidden dim {args.expected_dim}, got {hidden_dim}")

    print(
        f"Loaded {model_name}; fake_z={tuple(fake_z.shape)}; tshift={tshift:.6f}; "
        f"layer{args.depth}={tuple(probe_h.shape)}; path={cfg_get(config, 'path_type', args.path_type)}"
    )
    del probe_h
    gc.collect()
    torch.cuda.empty_cache()

    return LoadedModels(
        vae=vae,
        sit=sit,
        config=config,
        path_type=str(cfg_get(config, "path_type", args.path_type)),
        token_count=token_count,
        hidden_dim=hidden_dim,
    )


# -----------------------------------------------------------------------------
# Exact extraction point corresponding to official REPA encoder_depth=8
# -----------------------------------------------------------------------------


@torch.no_grad()
def extract_hidden_from_raw_latent(
    model: torch.nn.Module,
    raw_z: torch.Tensor,
    labels: torch.Tensor,
    *,
    depth: int,
    t_value: float,
    path_type: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Reproduce this project's SiT.forward pre-processing, then return the raw
    residual-stream output after `depth` blocks, before any REPA projector.

    This is intentionally not just model.forward_feats(raw_z, ...):
    forward_feats expects an already normalized/noised model input.
    """
    if not (1 <= depth <= len(model.blocks)):
        raise ValueError(f"depth={depth} outside [1,{len(model.blocks)}]")

    if getattr(model, "vae_1d", False):
        normalized_z = model.bn(raw_z.transpose(1, 2)).transpose(1, 2)
        time_shape = (raw_z.shape[0], 1, 1)
    else:
        normalized_z = model.bn(raw_z)
        time_shape = (raw_z.shape[0], 1, 1, 1)

    raw_t = torch.full(
        time_shape,
        float(t_value),
        device=normalized_z.device,
        dtype=normalized_z.dtype,
    )
    shifted_t = model.shift_time(raw_t)
    alpha_t, sigma_t, _, _ = model.interpolant(shifted_t, path_type=path_type)
    noise = torch.randn(
        normalized_z.shape,
        generator=generator,
        device=normalized_z.device,
        dtype=normalized_z.dtype,
    )
    model_input = alpha_t * normalized_z + sigma_t * noise

    # forward_feats returns x after block number `depth` because it checks i+1.
    hidden = model.forward_feats(
        model_input,
        shifted_t.flatten(),
        labels,
        depth=depth,
    )
    if hidden is None or hidden.ndim != 3:
        raise RuntimeError(f"Unexpected hidden output: {None if hidden is None else hidden.shape}")
    return hidden


# -----------------------------------------------------------------------------
# Reference feature cache
# -----------------------------------------------------------------------------


@dataclass
class ReferenceCache:
    full_path: str
    pool_path: str
    labels_path: str
    metadata_path: str
    n: int
    t: int
    d: int

    def open_full(self, mode: str = "r") -> np.memmap:
        return np.memmap(self.full_path, mode=mode, dtype=np.float16, shape=(self.n, self.t, self.d))

    def open_pool(self, mmap_mode: str = "r") -> np.ndarray:
        return np.load(self.pool_path, mmap_mode=mmap_mode)

    def open_labels(self, mmap_mode: str = "r") -> np.ndarray:
        return np.load(self.labels_path, mmap_mode=mmap_mode)



def cache_paths(cache_dir: str) -> Tuple[str, str, str, str]:
    return (
        os.path.join(cache_dir, "reference_hidden_fp16.mmap"),
        os.path.join(cache_dir, "reference_pool_f32.npy"),
        os.path.join(cache_dir, "reference_labels_i64.npy"),
        os.path.join(cache_dir, "metadata.json"),
    )


@torch.no_grad()
def build_reference_cache(
    args: argparse.Namespace,
    loaded: LoadedModels,
    ref_loader: DataLoader,
    n_ref: int,
    device: torch.device,
) -> ReferenceCache:
    os.makedirs(args.cache_dir, exist_ok=True)
    full_path, pool_path, labels_path, metadata_path = cache_paths(args.cache_dir)

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

    if args.reuse_cache and all(os.path.exists(p) for p in (full_path, pool_path, labels_path, metadata_path)):
        with open(metadata_path, "r", encoding="utf-8") as f:
            old_meta = json.load(f)
        if old_meta == expected_meta:
            print(f"Reusing reference cache: {args.cache_dir}")
            return ReferenceCache(
                full_path, pool_path, labels_path, metadata_path,
                n_ref, loaded.token_count, loaded.hidden_dim,
            )
        print("Existing cache metadata differs; rebuilding cache.")

    full_mm = np.memmap(
        full_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_ref, loaded.token_count, loaded.hidden_dim),
    )
    pool_mm = np.lib.format.open_memmap(
        pool_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_ref, loaded.hidden_dim),
    )
    label_mm = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int64,
        shape=(n_ref,),
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 100_003)
    offset = 0
    pbar = tqdm(total=n_ref, desc="Extract reference h")
    for images, labels, _names in ref_loader:
        if offset >= n_ref:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        remaining = n_ref - offset
        if images.shape[0] > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        with make_autocast(device, args.amp_dtype):
            z = encode_vae(loaded.vae, images, labels, vae_1d=getattr(loaded.sit, "vae_1d", False))
            h = extract_hidden_from_raw_latent(
                loaded.sit,
                z,
                labels,
                depth=args.depth,
                t_value=args.t,
                path_type=loaded.path_type,
                generator=generator,
            )

        b = h.shape[0]
        h_f32 = h.float()
        full_mm[offset:offset+b] = h_f32.cpu().numpy().astype(np.float16)
        pool_mm[offset:offset+b] = h_f32.mean(dim=1).cpu().numpy().astype(np.float32)
        label_mm[offset:offset+b] = labels.cpu().numpy().astype(np.int64)
        offset += b
        pbar.update(b)
        del images, labels, z, h, h_f32
    pbar.close()

    if offset != n_ref:
        raise RuntimeError(f"Reference loader yielded {offset}, expected {n_ref}")

    full_mm.flush(); pool_mm.flush(); label_mm.flush()
    del full_mm, pool_mm, label_mm
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2)

    return ReferenceCache(
        full_path, pool_path, labels_path, metadata_path,
        n_ref, loaded.token_count, loaded.hidden_dim,
    )


# -----------------------------------------------------------------------------
# Coarse pooled retrieval, then full-unpooled reranking
# -----------------------------------------------------------------------------


class CoarseIndex:
    def search(self, query_pool: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError


class FaissCoarseIndex(CoarseIndex):
    def __init__(self, reference_pool: np.ndarray, use_gpu: bool, gpu_id: int):
        import faiss  # optional dependency

        ref = np.asarray(reference_pool, dtype=np.float32)
        ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
        cpu_index = faiss.IndexFlatIP(ref.shape[1])
        if use_gpu:
            resources = faiss.StandardGpuResources()
            self._resources = resources
            self.index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)
        else:
            self.index = cpu_index
        self.index.add(ref)

    def search(self, query_pool: np.ndarray, k: int) -> np.ndarray:
        q = np.asarray(query_pool, dtype=np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        _dist, inds = self.index.search(q, k)
        return inds.astype(np.int64, copy=False)


class TorchCoarseIndex(CoarseIndex):
    """Exact cosine coarse retrieval implemented in PyTorch.

    On CUDA, the normalized 100k x 1152 reference pool is kept resident in
    fp16 (about 220 MiB) and each query batch is searched with one matrix
    multiplication.  This is a practical replacement when FAISS is absent.
    CPU mode retains chunked float32 search.
    """

    def __init__(self, reference_pool: np.ndarray, device: torch.device, chunk: int):
        self.device = device
        self.chunk = int(chunk)
        self.reference_pool_cpu = reference_pool
        self.reference_t = None

        if device.type == "cuda":
            ref_np = np.asarray(reference_pool, dtype=np.float32)
            ref = torch.from_numpy(ref_np).to(device=device, dtype=torch.float16)
            ref = F.normalize(ref, dim=1)
            # Transpose once so search is q @ [D,N].
            self.reference_t = ref.transpose(0, 1).contiguous()
            mib = self.reference_t.numel() * self.reference_t.element_size() / (1024 ** 2)
            print(
                f"Torch coarse index: exact CUDA cosine search; "
                f"reference={tuple(ref.shape)}, resident={mib:.1f} MiB"
            )
            del ref
        else:
            print("Torch coarse index: exact chunked CPU cosine search")

    @torch.no_grad()
    def search(self, query_pool: np.ndarray, k: int) -> np.ndarray:
        if self.reference_t is not None:
            q = torch.from_numpy(np.asarray(query_pool, dtype=np.float32)).to(
                self.device, dtype=self.reference_t.dtype
            )
            q = F.normalize(q, dim=1)
            scores = q @ self.reference_t
            inds = torch.topk(scores, k=k, dim=1, largest=True).indices
            return inds.cpu().numpy().astype(np.int64, copy=False)

        q = torch.from_numpy(np.asarray(query_pool, dtype=np.float32)).to(self.device)
        q = F.normalize(q, dim=1)
        best_scores = torch.full((q.shape[0], k), -float("inf"), device=self.device)
        best_indices = torch.full((q.shape[0], k), -1, dtype=torch.long, device=self.device)
        n = self.reference_pool_cpu.shape[0]
        for start in range(0, n, self.chunk):
            end = min(start + self.chunk, n)
            r_np = np.asarray(self.reference_pool_cpu[start:end], dtype=np.float32)
            r = F.normalize(torch.from_numpy(r_np).to(self.device), dim=1)
            scores = q @ r.T
            local_k = min(k, scores.shape[1])
            s, idx = torch.topk(scores, k=local_k, dim=1)
            idx = idx + start
            merged_s = torch.cat([best_scores, s], dim=1)
            merged_i = torch.cat([best_indices, idx], dim=1)
            best_scores, pos = torch.topk(merged_s, k=k, dim=1)
            best_indices = torch.gather(merged_i, 1, pos)
            del r, scores, s, idx, merged_s, merged_i, pos
        return best_indices.cpu().numpy().astype(np.int64, copy=False)


def build_coarse_index(
    args: argparse.Namespace,
    reference_pool: np.ndarray,
    device: torch.device,
) -> CoarseIndex:
    backend = args.coarse_backend
    errors: List[str] = []

    if backend in ("auto", "faiss"):
        if args.faiss_gpu and device.type == "cuda":
            try:
                index = FaissCoarseIndex(
                    reference_pool,
                    use_gpu=True,
                    gpu_id=(device.index or 0),
                )
                print("Coarse index: FAISS GPU cosine IndexFlatIP")
                return index
            except Exception as exc:
                errors.append(f"FAISS GPU: {type(exc).__name__}: {exc}")
                print(f"FAISS GPU unavailable: {exc}")

        try:
            index = FaissCoarseIndex(reference_pool, use_gpu=False, gpu_id=0)
            print("Coarse index: FAISS CPU cosine IndexFlatIP")
            return index
        except Exception as exc:
            errors.append(f"FAISS CPU: {type(exc).__name__}: {exc}")
            print(f"FAISS CPU unavailable: {exc}")

        if backend == "faiss" and not args.allow_torch_coarse_fallback:
            raise RuntimeError("FAISS coarse index failed: " + " | ".join(errors))

    if backend == "torch" or args.allow_torch_coarse_fallback or backend == "auto":
        if errors:
            print("Falling back to exact PyTorch coarse search.")
        return TorchCoarseIndex(reference_pool, device, args.coarse_chunk)

    raise RuntimeError(
        "No coarse-search backend available. "
        + (" | ".join(errors) if errors else "")
    )


@torch.no_grad()
def rerank_full_hidden(
    query_h: torch.Tensor,
    candidate_indices: np.ndarray,
    reference_full: np.memmap,
    metric: str,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Final neighbour selection uses the full, unpooled [T,D] hidden."""
    candidate_np = np.asarray(reference_full[candidate_indices], dtype=np.float16).copy()
    candidates = torch.from_numpy(candidate_np).to(device=device)
    q = query_h.float().reshape(query_h.shape[0], -1)
    c = candidates.float().reshape(candidates.shape[0], candidates.shape[1], -1)

    if metric == "cosine":
        q = F.normalize(q, dim=-1)
        c = F.normalize(c, dim=-1)
        scores = torch.einsum("bd,bkd->bk", q, c)
        best_pos = scores.argmax(dim=1)
    elif metric == "l2":
        scores = ((c - q[:, None, :]) ** 2).mean(dim=-1)
        best_pos = scores.argmin(dim=1)
    else:
        raise ValueError(metric)

    rows = torch.arange(query_h.shape[0], device=device)
    best_h = candidates[rows, best_pos]
    best_global_idx = candidate_indices[np.arange(query_h.shape[0]), best_pos.cpu().numpy()]
    return best_h, best_global_idx.astype(np.int64, copy=False)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def pooled_raw(h: torch.Tensor) -> torch.Tensor:
    return h.float().mean(dim=1)


def pooled_token_layernorm(h: torch.Tensor) -> torch.Tensor:
    h32 = h.float()
    return F.layer_norm(h32, (h32.shape[-1],)).mean(dim=1)


def frechet_from_features(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"Bad feature shapes: {a.shape}, {b.shape}")
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    mu_a, mu_b = a64.mean(axis=0), b64.mean(axis=0)
    cov_a = np.cov(a64, rowvar=False)
    cov_b = np.cov(b64, rowvar=False)
    return float(calculate_frechet_distance(mu_a, cov_a, mu_b, cov_b))


def covariance_trace(x: np.ndarray) -> float:
    # trace(cov) equals the sum of per-dimension sample variances.
    return float(np.var(np.asarray(x, dtype=np.float64), axis=0, ddof=1).sum())


@torch.no_grad()
def evaluate_queries(
    args: argparse.Namespace,
    loaded: LoadedModels,
    val_loader: DataLoader,
    n_val: int,
    cache: ReferenceCache,
    coarse_index: CoarseIndex,
    device: torch.device,
) -> Dict[str, Any]:
    reference_full = cache.open_full("r")
    reference_labels = cache.open_labels("r")

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

        with make_autocast(device, args.amp_dtype):
            z = encode_vae(loaded.vae, images, labels, vae_1d=getattr(loaded.sit, "vae_1d", False))
            query_h = extract_hidden_from_raw_latent(
                loaded.sit,
                z,
                labels,
                depth=args.depth,
                t_value=args.t,
                path_type=loaded.path_type,
                generator=noise_generator,
            )

        query_h = query_h.float()
        query_pool_np = pooled_raw(query_h).cpu().numpy().astype(np.float32)
        coarse_candidates = coarse_index.search(query_pool_np, args.coarse_topk)
        nn_h, nn_indices = rerank_full_hidden(
            query_h,
            coarse_candidates,
            reference_full,
            args.nn_metric,
            device,
        )
        nn_h = nn_h.float()
        selected_indices.append(nn_indices)

        nn_labels = torch.from_numpy(np.asarray(reference_labels[nn_indices], dtype=np.int64)).to(device)
        same_class_total += int((nn_labels == labels).sum().item())

        # Main flow: interpolate full hidden first, pool only afterwards.
        main_h = (1.0 - args.alpha) * query_h + args.alpha * nn_h
        a0_h = query_h
        a1_h = nn_h

        random_indices = random_generator.integers(0, cache.n, size=query_h.shape[0], endpoint=False)
        random_np = np.asarray(reference_full[random_indices], dtype=np.float16).copy()
        random_h = torch.from_numpy(random_np).to(device=device).float()
        rand_intp_h = 0.5 * query_h + 0.5 * random_h

        arrays["src_raw"].append(pooled_raw(query_h).cpu().numpy())
        arrays["src_ln"].append(pooled_token_layernorm(query_h).cpu().numpy())
        arrays["main_raw"].append(pooled_raw(main_h).cpu().numpy())
        arrays["main_ln"].append(pooled_token_layernorm(main_h).cpu().numpy())
        arrays["a0_raw"].append(pooled_raw(a0_h).cpu().numpy())
        arrays["a0_ln"].append(pooled_token_layernorm(a0_h).cpu().numpy())
        arrays["a1_raw"].append(pooled_raw(a1_h).cpu().numpy())
        arrays["a1_ln"].append(pooled_token_layernorm(a1_h).cpu().numpy())
        arrays["rand_raw"].append(pooled_raw(rand_intp_h).cpu().numpy())
        arrays["rand_ln"].append(pooled_token_layernorm(rand_intp_h).cpu().numpy())

        full_l2_sum += float(query_h.flatten(1).norm(dim=1).sum().item())
        pooled_l2_sum += float(pooled_raw(query_h).norm(dim=1).sum().item())

        b = query_h.shape[0]
        seen += b
        pbar.update(b)
        del (
            images, labels, z, query_h, nn_h, main_h, a0_h, a1_h,
            random_h, rand_intp_h, coarse_candidates, nn_indices, nn_labels,
        )
    pbar.close()

    if seen != n_val:
        raise RuntimeError(f"Validation loader yielded {seen}, expected {n_val}")

    features = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    if n_val <= loaded.hidden_dim:
        print(
            f"WARNING: n_val={n_val} <= hidden_dim={loaded.hidden_dim}; covariance is rank-deficient. "
            "Use at least 2k-5k queries for a meaningful pilot."
        )

    src_raw = features["src_raw"]
    src_ln = features["src_ln"]
    half = n_val // 2

    metrics = {
        "main_hidden_fid_raw": frechet_from_features(src_raw, features["main_raw"]),
        "main_hidden_fid_token_ln": frechet_from_features(src_ln, features["main_ln"]),
        "sanity_alpha0_fid_raw": frechet_from_features(src_raw, features["a0_raw"]),
        "sanity_alpha0_fid_token_ln": frechet_from_features(src_ln, features["a0_ln"]),
        "sanity_alpha1_fid_raw": frechet_from_features(src_raw, features["a1_raw"]),
        "sanity_alpha1_fid_token_ln": frechet_from_features(src_ln, features["a1_ln"]),
        "random_neighbor_alpha05_fid_raw": frechet_from_features(src_raw, features["rand_raw"]),
        "random_neighbor_alpha05_fid_token_ln": frechet_from_features(src_ln, features["rand_ln"]),
        "nn_same_class_rate": same_class_total / float(n_val),
        "query_hidden_mean_full_l2": full_l2_sum / float(n_val),
        "query_hidden_mean_pooled_l2": pooled_l2_sum / float(n_val),
        "query_pooled_cov_trace_raw": covariance_trace(src_raw),
        "query_pooled_cov_trace_token_ln": covariance_trace(src_ln),
        "split_half_fid_raw": frechet_from_features(src_raw[:half], src_raw[half:2*half]),
        "split_half_fid_token_ln": frechet_from_features(src_ln[:half], src_ln[half:2*half]),
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
            "coarse_topk": args.coarse_topk,
            "coarse_backend": args.coarse_backend,
            "n_reference": cache.n,
            "n_query": n_val,
            "tokens": cache.t,
            "hidden_dim": cache.d,
            "seed": args.seed,
        },
        "metrics": metrics,
    }


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-path", required=True)
    p.add_argument("--train-steps", default="400000")
    p.add_argument("--ckpt-key", choices=["ema", "model"], default="ema")
    p.add_argument("--dataset-ref", required=True)
    p.add_argument("--dataset-val", required=True)
    p.add_argument("--small-ref", type=int, default=100_000)
    p.add_argument("--small-val", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--t", type=float, default=0.1)
    p.add_argument("--path-type", choices=["linear", "cosine"], default="linear")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--nn-metric", choices=["cosine", "l2"], default="cosine")
    p.add_argument("--coarse-topk", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp-dtype", choices=["no", "fp16", "bf16"], default="no")
    p.add_argument(
        "--expected-tokens",
        type=int,
        default=0,
        help=(
            "Optional token-count assertion. 0 disables the assertion because "
            "different VAE latent resolutions / SiT patch sizes legitimately "
            "produce different token counts (for example 64 or 256)."
        ),
    )
    p.add_argument("--expected-dim", type=int, default=1152)

    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-pooled-features", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument(
        "--coarse-backend",
        choices=["auto", "faiss", "torch"],
        default="auto",
        help="Coarse pooled-NN backend. 'auto' tries FAISS and then exact PyTorch.",
    )
    p.add_argument("--faiss-gpu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-torch-coarse-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coarse-chunk", type=int, default=20_000)
    p.add_argument("--alpha0-tolerance", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.t <= 1.0):
        raise ValueError("--t must be in [0,1]")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")
    if args.small_ref <= 0 or args.small_val <= 1:
        raise ValueError("--small-ref and --small-val must be positive")

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

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

    ref_loader = DataLoader(
        ref_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    loaded = load_models(args, device)
    cache = build_reference_cache(args, loaded, ref_loader, n_ref, device)
    reference_pool = cache.open_pool("r")
    coarse_index = build_coarse_index(args, reference_pool, device)
    result = evaluate_queries(args, loaded, val_loader, n_val, cache, coarse_index, device)

    result_path = os.path.join(args.output_dir, "hidden_ifid_metrics.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Hidden interpolation metrics ===")
    for key, value in result["metrics"].items():
        print(f"{key}={value:.8f}")
    print(f"Saved: {result_path}")

    if not args.keep_cache:
        shutil.rmtree(args.cache_dir)
        print(f"Removed scratch cache: {args.cache_dir}")


if __name__ == "__main__":
    main()
