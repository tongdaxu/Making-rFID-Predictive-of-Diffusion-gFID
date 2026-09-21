#!/usr/bin/env python3
"""
25-VAE Local-Score rFID evaluator.

Goal
----
For each ImageNet validation image x:
    z_raw = VAE.encode(x)
    z     = channel-normalize(z_raw) using statistics from a cross-class
            ImageNet-train reference bank
    z_t   = alpha(t) * z + sigma(t) * eps
    zhat  = Local-Score posterior mean E[z0 | z_t(Omega_x)]
    xhat  = VAE.decode(denormalize(zhat))

Then compute:
    LS-rFID(t,P) = FID({x}, {xhat})

Default requested settings:
    raw t=0.2, P=1
    raw t=0.4, P=3

Reference/query protocol:
    50k ImageNet train cross-class references
    50k ImageNet val queries

IMPORTANT COMPUTATIONAL NOTE
----------------------------
The exact LS posterior over all 50k references at every spatial location for
all 50k queries is prohibitively expensive for the 25-VAE table.

This evaluator therefore uses a top-K TRUNCATED Local Score:
  1) retrieve K nearest clean normalized reference latents globally;
  2) compute the Local Score posterior exactly over those K candidates.

The candidate pool is shared by t=0.2 and t=0.4 for each query.
This is an approximation to the exact 50k-reference Local Score, and the
metrics JSON records that fact explicitly.

The code is deliberately diffusion-model-free:
- no SiT checkpoint
- no learned score network
- latent mean/std come only from the VAE reference latent bank

Time convention:
- default shifted_raw to match the benchmark SiT time-shift convention:
      tshift = sqrt(latent_numel / 4096)
      t_eff  = tshift*t / (1 + (tshift-1)*t)
- use --time-mode fixed_effective to make effective corruption exactly t
  for every VAE.

Latent layouts:
- native BCHW is used directly;
- native [B,N,C] token latents are spatialized only for Local Score, then
  converted back to [B,N,C] before VAE.decode().
- non-square token sequences require --token-grid H W.
"""

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.fid.fid import InceptionV3, calculate_frechet_distance
from ifid.vae.utils import instantiate_from_config


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_setting(s: str) -> Tuple[float, int]:
    try:
        t_s, p_s = s.split(":")
        t = float(t_s)
        p = int(p_s)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Bad setting {s!r}; expected t:P, e.g. 0.2:1"
        ) from e
    if not (0.0 < t < 1.0):
        raise argparse.ArgumentTypeError(f"t must be in (0,1), got {t}")
    if p <= 0 or p % 2 != 1:
        raise argparse.ArgumentTypeError(f"P must be positive odd, got {p}")
    return t, p


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--vae-config", type=str, required=True)
    p.add_argument("--exp-name", type=str, required=True)

    p.add_argument("--dataset", type=str, default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", type=str, default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--ref-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=50000)
    p.add_argument("--resolution", type=int, default=256)

    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--ref-bs", type=int, default=32,
                   help="Reference VAE encode batch size; independent of query batch size.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--query-block-batches", type=int, default=4)

    p.add_argument(
        "--settings",
        type=parse_setting,
        nargs="+",
        default=[(0.2, 1), (0.4, 1)],
        help="Local-Score settings t:P. Default: 0.2:1 0.4:1",
    )
    p.add_argument(
        "--time-mode",
        choices=["shifted_raw", "fixed_effective"],
        default="shifted_raw",
    )
    p.add_argument("--path-type", choices=["linear"], default="linear")
    p.add_argument("--noise-seed", type=int, default=12345)

    p.add_argument(
        "--candidate-topk",
        type=int,
        default=512,
        help="Global candidate pool size before exact Local-Score weighting.",
    )
    p.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=32,
        help="Candidate chunk size for online Local-Score softmax.",
    )
    p.add_argument(
        "--metric-chunk-size",
        type=int,
        default=256,
        help="Reference chunk size during global top-K retrieval.",
    )
    p.add_argument(
        "--metric-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
    )
    p.add_argument(
        "--gpu-bank-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help=(
            "Dtype of the normalized 50k reference bank kept resident on GPU. "
            "auto uses fp32 when it fits comfortably and bf16 otherwise."
        ),
    )

    p.add_argument(
        "--token-grid",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="For native [B,N,C] latents with non-square N.",
    )

    p.add_argument(
        "--latent-cache-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/cache",
    )
    p.add_argument("--cache-group-batches", type=int, default=20)
    p.add_argument(
        "--cache-dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
    )
    p.add_argument(
        "--reuse-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete reference cache when available; otherwise rebuild.",
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/results",
    )
    p.add_argument("--save-preview", type=int, default=0)
    p.add_argument("--keep-features", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


# ---------------------------------------------------------------------
# VAE wrappers
# ---------------------------------------------------------------------

def unwrap_latent(z):
    if isinstance(z, (tuple, list)):
        z = z[0]

    if torch.is_tensor(z):
        return z

    if hasattr(z, "latent_dist"):
        ld = z.latent_dist
        if hasattr(ld, "mode") and callable(ld.mode):
            return ld.mode()
        if hasattr(ld, "mean") and torch.is_tensor(ld.mean):
            return ld.mean
        if hasattr(ld, "sample") and callable(ld.sample):
            return ld.sample()

    if hasattr(z, "mode") and callable(z.mode):
        return z.mode()

    if hasattr(z, "sample"):
        if torch.is_tensor(z.sample):
            return z.sample
        if callable(z.sample):
            out = z.sample()
            if torch.is_tensor(out):
                return out

    raise TypeError(f"Unsupported VAE encode output type: {type(z)}")


@torch.no_grad()
def encode_vae(vae, x, y=None):
    try:
        out = vae.encode(x, y) if y is not None else vae.encode(x)
    except TypeError:
        out = vae.encode(x)
    z = unwrap_latent(out)
    if z.ndim not in (3, 4):
        raise RuntimeError(
            f"Local Score expects native [B,C,H,W] or [B,N,C]; got {tuple(z.shape)}"
        )
    return z


@torch.no_grad()
def decode_vae(vae, z, y=None):
    try:
        out = vae.decode(z, y) if y is not None else vae.decode(z)
    except TypeError:
        out = vae.decode(z)

    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out):
        return out
    if hasattr(out, "sample") and torch.is_tensor(out.sample):
        return out.sample
    raise TypeError(f"Unsupported VAE decode output type: {type(out)}")


# ---------------------------------------------------------------------
# Latent layout / normalization
# ---------------------------------------------------------------------

def latent_channel_sum_sumsq_count(z: torch.Tensor):
    z = z.float()
    if z.ndim == 4:      # [B,C,H,W]
        dims = (0, 2, 3)
        count = z.shape[0] * z.shape[2] * z.shape[3]
    elif z.ndim == 3:    # [B,N,C]
        dims = (0, 1)
        count = z.shape[0] * z.shape[1]
    else:
        raise ValueError(tuple(z.shape))
    return z.sum(dims), (z * z).sum(dims), int(count)


def normalize_native(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    if z.ndim == 4:
        # [B,C,H,W] OR [B,K,N,C] for candidate tensor.
        if z.shape[1] == mean.numel():
            return (z - mean[None, :, None, None]) / std[None, :, None, None]
        if z.shape[-1] == mean.numel():
            return (z - mean[None, None, None, :]) / std[None, None, None, :]
    elif z.ndim == 5:
        # [B,K,C,H,W]
        return (
            z - mean[None, None, :, None, None]
        ) / std[None, None, :, None, None]
    elif z.ndim == 3:
        # [B,N,C]
        return (z - mean[None, None, :]) / std[None, None, :]
    raise ValueError(f"Cannot normalize shape {tuple(z.shape)} with C={mean.numel()}")


def denormalize_native(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    if z.ndim == 4:
        return z * std[None, :, None, None] + mean[None, :, None, None]
    if z.ndim == 3:
        return z * std[None, None, :] + mean[None, None, :]
    raise ValueError(tuple(z.shape))


def infer_token_grid(n: int, token_grid: Optional[Sequence[int]]) -> Tuple[int, int]:
    if token_grid is not None:
        h, w = int(token_grid[0]), int(token_grid[1])
        if h * w != n:
            raise ValueError(
                f"--token-grid {h} {w} has product {h*w}, but token length N={n}"
            )
        return h, w

    h = int(math.isqrt(n))
    if h * h != n:
        raise ValueError(
            f"Token latent N={n} is not square. Pass --token-grid H W."
        )
    return h, h


def native_to_spatial(z: torch.Tensor, token_grid=None) -> torch.Tensor:
    """
    Query:
      [B,C,H,W] -> [B,C,H,W]
      [B,N,C]   -> [B,C,H,W]

    Candidate:
      [B,K,C,H,W] -> itself
      [B,K,N,C]   -> [B,K,C,H,W]
    """
    if z.ndim == 5:
        return z.contiguous()

    if z.ndim == 4:
        # Ambiguous: query BCHW or candidate B,K,N,C.
        # Candidate B,K,N,C is recognized because last dim is channel dim
        # and dimensions 1/2 are K/N.
        if token_grid is not None:
            h, w = int(token_grid[0]), int(token_grid[1])
            if z.shape[-2] == h * w:
                # [B,K,N,C]
                b, k, n, c = z.shape
                return (
                    z.permute(0, 1, 3, 2)
                    .reshape(b, k, c, h, w)
                    .contiguous()
                )
        # Native spatial query.
        return z.contiguous()

    if z.ndim == 3:
        b, n, c = z.shape
        h, w = infer_token_grid(n, token_grid)
        return z.permute(0, 2, 1).reshape(b, c, h, w).contiguous()

    raise ValueError(f"Unsupported latent shape: {tuple(z.shape)}")


def spatial_to_native(z4: torch.Tensor, native_shape: Sequence[int], token_grid=None):
    """
    z4 is [B,C,H,W]. native_shape excludes batch.
    """
    if len(native_shape) == 3:
        # Native BCHW.
        return z4

    if len(native_shape) == 2:
        # Native [N,C].
        n, c = int(native_shape[0]), int(native_shape[1])
        h, w = infer_token_grid(n, token_grid)
        if tuple(z4.shape[1:]) != (c, h, w):
            raise ValueError(
                f"Spatial/native mismatch: spatial={tuple(z4.shape)}, native={native_shape}"
            )
        return z4.reshape(z4.shape[0], c, n).permute(0, 2, 1).contiguous()

    raise ValueError(f"Unsupported native shape: {native_shape}")


# ---------------------------------------------------------------------
# Time / path
# ---------------------------------------------------------------------

def effective_t(t_raw: float, tshift: float, time_mode: str) -> float:
    if time_mode == "fixed_effective":
        return float(t_raw)
    t = float(t_raw)
    return float(tshift * t / (1.0 + (tshift - 1.0) * t))


def linear_interpolant(t_eff: float):
    return 1.0 - t_eff, t_eff


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------

def dtype_from_name(name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def cache_key(args):
    cfg = Path(args.vae_config).stem
    return f"{cfg}_ref{args.ref_size}_res{args.resolution}_seed{args.seed}"


def cache_paths(args):
    root = Path(args.latent_cache_dir) / cache_key(args)
    return root, root / "stats.pt"


def discover_chunks(root: Path):
    chunks = sorted(root.glob("chunk*.pt"))
    meta = []
    total = 0
    latent_shape = None
    for p in chunks:
        pack = torch.load(p, map_location="cpu")
        z = pack["z"]
        start = int(pack["start"])
        end = int(pack["end"])
        if start != total:
            raise RuntimeError(f"Non-contiguous cache at {p}: start={start}, expected={total}")
        total = end
        if latent_shape is None:
            latent_shape = tuple(z.shape[1:])
        elif tuple(z.shape[1:]) != latent_shape:
            raise RuntimeError("Latent shape changed across cache chunks.")
        meta.append({"path": str(p), "start": start, "end": end})
        del pack, z
    return meta, latent_shape, total


@torch.no_grad()
def build_or_load_reference_cache(args, vae, train_loader, device):
    root, stats_path = cache_paths(args)

    if args.reuse_cache and root.is_dir() and stats_path.exists():
        try:
            # Fast reuse path: stats.pt already records ref_size and latent_shape.
            # Do NOT torch.load every chunk merely to rediscover start/end metadata;
            # the GPU-bank loader will read each chunk exactly once and obtain
            # start/end from that same load.
            stats = torch.load(stats_path, map_location="cpu")
            chunks_on_disk = sorted(root.glob("chunk*.pt"))
            if (
                int(stats["ref_size"]) == args.ref_size
                and len(chunks_on_disk) > 0
            ):
                latent_shape = tuple(stats["latent_shape"])
                chunks = [{"path": str(x)} for x in chunks_on_disk]
                print(
                    f"[cache] reuse {root} ({len(chunks)} chunks; "
                    "fast manifest path, no pre-scan)"
                )
                return (
                    chunks,
                    latent_shape,
                    stats["mean"].float(),
                    stats["std"].float(),
                )
            print("[cache] stats/chunks incomplete or mismatched; rebuilding")
        except Exception as e:
            print(f"[cache] reuse failed ({e!r}); rebuilding")

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    store_dtype = dtype_from_name(args.cache_dtype)
    buf_z = []
    buf_y = []
    chunks = []
    local_start = 0
    group_id = 0

    stat_sum = None
    stat_sumsq = None
    stat_count = 0
    latent_shape = None

    def flush():
        nonlocal local_start, group_id, buf_z, buf_y
        if not buf_z:
            return
        zcat = torch.cat(buf_z, 0).contiguous()
        ycat = torch.cat(buf_y, 0).contiguous()
        end = local_start + zcat.shape[0]
        p = root / f"chunk{group_id:05d}.pt"
        torch.save(
            {"z": zcat, "y": ycat, "start": local_start, "end": end},
            p,
        )
        chunks.append({"path": str(p), "start": local_start, "end": end})
        local_start = end
        group_id += 1
        buf_z, buf_y = [], []

    for batch_idx, (img, y, _name) in enumerate(
        tqdm(train_loader, desc=f"encode ref {Path(args.vae_config).stem}")
    ):
        img = img.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        z = encode_vae(vae, img, y_dev)

        if latent_shape is None:
            latent_shape = tuple(z.shape[1:])
        elif tuple(z.shape[1:]) != latent_shape:
            raise RuntimeError("VAE latent shape changed across reference batches.")

        s, ss, cnt = latent_channel_sum_sumsq_count(z)
        if stat_sum is None:
            stat_sum = torch.zeros_like(s, dtype=torch.float64, device=device)
            stat_sumsq = torch.zeros_like(ss, dtype=torch.float64, device=device)
        stat_sum += s.double()
        stat_sumsq += ss.double()
        stat_count += cnt

        buf_z.append(z.detach().cpu().to(store_dtype))
        buf_y.append(y.detach().cpu())

        if len(buf_z) >= args.cache_group_batches:
            flush()

    flush()

    if local_start != args.ref_size:
        raise RuntimeError(
            f"Reference cache contains {local_start} samples, expected {args.ref_size}"
        )

    count_t = float(stat_count)
    mean = (stat_sum / count_t).float().cpu()
    var = (stat_sumsq / count_t).float().cpu() - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-6))

    torch.save(
        {
            "mean": mean,
            "std": std,
            "latent_shape": latent_shape,
            "ref_size": args.ref_size,
            "vae_config": args.vae_config,
            "normalization": "reference-bank per-channel mean/std",
        },
        stats_path,
    )

    print(f"[cache] built {root}")
    print(f"[cache] latent_shape={latent_shape}, C={mean.numel()}")
    return chunks, latent_shape, mean, std


# ---------------------------------------------------------------------
# Global candidate retrieval
# ---------------------------------------------------------------------

@torch.no_grad()
def retrieve_topk_blocked(
    cache_files,
    query_records,
    mean,
    std,
    K,
    device,
    metric_chunk_size,
    metric_dtype,
):
    """
    Cross-class top-K in CLEAN channel-normalized latent space.

    Returns one [B,K] CPU LongTensor per query record.
    """
    mean_d = mean.to(device)
    std_d = std.to(device)

    states = []
    for rec in query_records:
        z = rec["z"].to(device)
        zn = normalize_native(z.float(), mean_d, std_d)
        q = zn.reshape(zn.shape[0], -1).to(metric_dtype)
        states.append(
            {
                "q": q,
                "q_sq": (q * q).sum(-1),
                "best_cost": None,
                "best_idx": None,
            }
        )

    for meta in cache_files:
        pack = torch.load(meta["path"], map_location="cpu")
        z_all = pack["z"]
        base = int(pack["start"])

        for s in range(0, z_all.shape[0], metric_chunk_size):
            e = min(s + metric_chunk_size, z_all.shape[0])
            cand = z_all[s:e].to(device=device, dtype=torch.float32)
            cand = normalize_native(cand, mean_d, std_d)
            cf = cand.reshape(cand.shape[0], -1).to(metric_dtype)
            c_sq = (cf * cf).sum(-1)

            for st in states:
                # Squared Euclidean distance; common scaling is irrelevant to top-K.
                cost = (
                    c_sq[:, None]
                    + st["q_sq"][None, :]
                    - 2.0 * (cf @ st["q"].T)
                ).float()

                k_now = min(K, cost.shape[0])
                cst, idx = torch.topk(cost, k=k_now, dim=0, largest=False)
                idx = idx + base + s

                if st["best_cost"] is None:
                    st["best_cost"] = cst
                    st["best_idx"] = idx
                else:
                    mc = torch.cat([st["best_cost"], cst], dim=0)
                    mi = torch.cat([st["best_idx"], idx], dim=0)
                    st["best_cost"], pos = torch.topk(
                        mc, k=min(K, mc.shape[0]), dim=0, largest=False
                    )
                    st["best_idx"] = torch.gather(mi, 0, pos)

                del cost, cst, idx

            del cand, cf, c_sq

        del pack, z_all

    out = []
    for st in states:
        # [K,B] -> [B,K]
        out.append(st["best_idx"].T.contiguous().cpu())
    return out


@torch.no_grad()
def fetch_candidate_block(cache_files, index_batches, latent_shape, cache_dtype):
    """
    index_batches: list of [B,K] global reference indices.
    Returns CPU tensors [B,K,*latent_shape].
    Scans each cache file at most once for the whole query block.
    """
    outputs = []
    all_idx = []
    all_bid = []
    all_flat = []

    for bid, idx2 in enumerate(index_batches):
        B, K = idx2.shape
        out = torch.empty(
            B, K, *latent_shape, dtype=cache_dtype, device="cpu"
        )
        outputs.append(out)

        flat_idx = idx2.reshape(-1).long()
        flat_pos = torch.arange(B * K, dtype=torch.long)
        all_idx.append(flat_idx)
        all_bid.append(torch.full_like(flat_idx, bid))
        all_flat.append(flat_pos)

    all_idx = torch.cat(all_idx)
    all_bid = torch.cat(all_bid)
    all_flat = torch.cat(all_flat)

    order = torch.argsort(all_idx)
    all_idx = all_idx[order]
    all_bid = all_bid[order]
    all_flat = all_flat[order]

    filled = [torch.zeros(x.shape[0] * x.shape[1], dtype=torch.bool)
              for x in index_batches]

    for meta in cache_files:
        start, end = int(meta["start"]), int(meta["end"])
        lo = int(torch.searchsorted(all_idx, torch.tensor(start), right=False))
        hi = int(torch.searchsorted(all_idx, torch.tensor(end), right=False))
        if lo >= hi:
            continue

        pack = torch.load(meta["path"], map_location="cpu")
        z_all = pack["z"]

        idx_here = all_idx[lo:hi] - start
        selected = z_all[idx_here]
        bids = all_bid[lo:hi]
        flats = all_flat[lo:hi]

        for bid in torch.unique(bids).tolist():
            m = bids == bid
            flat_out = outputs[bid].view(
                outputs[bid].shape[0] * outputs[bid].shape[1],
                *latent_shape,
            )
            flat_out[flats[m]] = selected[m].to(cache_dtype)
            filled[bid][flats[m]] = True

        del pack, z_all, selected

    for bid, f in enumerate(filled):
        if not bool(f.all()):
            raise RuntimeError(
                f"Candidate fetch incomplete for block item {bid}: "
                f"{int((~f).sum())} missing"
            )
    return outputs



# ---------------------------------------------------------------------
# Fast GPU-resident candidate retrieval
# ---------------------------------------------------------------------

@torch.no_grad()
def choose_gpu_bank_dtype(args, total: int, latent_shape, device):
    numel = int(total) * int(np.prod(latent_shape))
    free_bytes, _ = torch.cuda.mem_get_info(device)

    if args.gpu_bank_dtype != "auto":
        dtype = dtype_from_name(args.gpu_bank_dtype)
    else:
        # Prefer fp32 for numerical agreement with the original implementation.
        fp32_bytes = numel * 4
        # Leave substantial headroom for the VAE, Inception, query matrices,
        # top-k buffers, and Local-Score candidate chunks.
        if fp32_bytes <= int(0.62 * free_bytes):
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

    need = numel * torch.tensor([], dtype=dtype).element_size()
    print(
        f"[gpu-bank] dtype={dtype}, estimated={need / 2**30:.2f} GiB, "
        f"free_before={free_bytes / 2**30:.2f} GiB"
    )
    return dtype


@torch.no_grad()
def load_normalized_reference_bank_gpu(
    cache_files,
    total,
    latent_shape,
    mean,
    std,
    device,
    bank_dtype,
):
    """
    Load the cached reference latents exactly ONCE, normalize them, and keep the
    full 50k bank resident on the A800. This removes the repeated torch.load()
    scans that dominated the previous implementation.
    """
    bank = torch.empty(
        (int(total), *tuple(latent_shape)),
        dtype=bank_dtype,
        device=device,
    )
    mean_d = mean.to(device=device, dtype=torch.float32)
    std_d = std.to(device=device, dtype=torch.float32)

    pbar = tqdm(cache_files, desc="load ref bank -> GPU")
    for meta in pbar:
        pack = torch.load(meta["path"], map_location="cpu")
        raw = pack["z"].to(device=device, dtype=torch.float32, non_blocking=False)
        norm = normalize_native(raw, mean_d, std_d)
        s = int(meta.get("start", pack["start"]))
        e = int(meta.get("end", pack["end"]))
        bank[s:e].copy_(norm.to(bank_dtype))
        del pack, raw, norm

    return bank


@torch.no_grad()
def reference_sq_norms(ref_bank_norm: torch.Tensor, chunk_size: int = 4096):
    """
    Compute ||r||^2 once. Chunking avoids creating a second tensor as large as
    the complete reference bank.
    """
    n = ref_bank_norm.shape[0]
    out = torch.empty(n, device=ref_bank_norm.device, dtype=torch.float32)
    flat = ref_bank_norm.reshape(n, -1)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        x = flat[s:e].float()
        out[s:e] = (x * x).sum(dim=1)
        del x
    return out


@torch.no_grad()
def retrieve_topk_gpu_bank(
    ref_bank_norm,
    ref_sq,
    query_records,
    mean,
    std,
    K,
    device,
    metric_chunk_size,
):
    """
    Same global clean-normalized squared-L2 top-K as retrieve_topk_blocked(),
    but:
      1) reference latents are already resident on GPU;
      2) all queries in the block are concatenated into one GEMM.

    This changes the implementation, not the metric definition.
    Returns one GPU [B,K] LongTensor for each original validation batch.
    """
    mean_d = mean.to(device=device, dtype=torch.float32)
    std_d = std.to(device=device, dtype=torch.float32)

    batch_sizes = []
    q_parts = []
    for rec in query_records:
        z = rec["z"].float()
        zn = normalize_native(z, mean_d, std_d)
        q_parts.append(zn.reshape(zn.shape[0], -1))
        batch_sizes.append(int(zn.shape[0]))

    q = torch.cat(q_parts, dim=0)
    q_compute = q.to(dtype=ref_bank_norm.dtype)
    # Compute norms in fp32 from the same normalized query values.
    q_sq = (q * q).sum(dim=1)
    del q, q_parts

    nref = ref_bank_norm.shape[0]
    ref_flat = ref_bank_norm.reshape(nref, -1)

    best_cost = None
    best_idx = None

    # Large reference chunks turn thousands of tiny GEMMs into a few large GEMMs.
    for s in range(0, nref, metric_chunk_size):
        e = min(s + metric_chunk_size, nref)
        rf = ref_flat[s:e]

        # [R,Q]. Accumulate the distance expression in fp32.
        cross = rf @ q_compute.T
        cost = (
            ref_sq[s:e, None]
            + q_sq[None, :]
            - 2.0 * cross.float()
        )

        k_now = min(K, e - s)
        cst, idx = torch.topk(cost, k=k_now, dim=0, largest=False)
        idx = idx + s

        if best_cost is None:
            best_cost, best_idx = cst, idx
        else:
            mc = torch.cat([best_cost, cst], dim=0)
            mi = torch.cat([best_idx, idx], dim=0)
            best_cost, pos = torch.topk(
                mc, k=min(K, mc.shape[0]), dim=0, largest=False
            )
            best_idx = torch.gather(mi, 0, pos)

        del rf, cross, cost, cst, idx

    # [K,Q] -> [Q,K], then split back to the original validation batches.
    all_idx = best_idx.T.contiguous()
    out = []
    off = 0
    for b in batch_sizes:
        out.append(all_idx[off:off+b])
        off += b
    return out


@torch.no_grad()
def local_score_mu_topk_gpu_bank(
    z_t: torch.Tensor,
    candidate_indices: torch.Tensor,
    ref_bank_norm: torch.Tensor,
    alpha: float,
    sigma: float,
    patch_size: int,
    token_grid,
    candidate_chunk_size: int,
):
    """
    Same truncated Local-Score posterior as local_score_mu_topk(), except
    candidates are gathered directly from the normalized GPU-resident reference
    bank instead of being repeatedly fetched from disk/CPU.
    """
    B, K = candidate_indices.shape
    _, C, H, W = z_t.shape
    native_shape = tuple(ref_bank_norm.shape[1:])

    running_max = torch.full(
        (B, H, W), -float("inf"), device=z_t.device, dtype=torch.float32
    )
    running_den = torch.zeros(
        (B, H, W), device=z_t.device, dtype=torch.float32
    )
    running_num = torch.zeros(
        (B, C, H, W), device=z_t.device, dtype=torch.float32
    )

    denom_const = 2.0 * float(sigma) * float(sigma)

    for s in range(0, K, candidate_chunk_size):
        e = min(s + candidate_chunk_size, K)
        ids = candidate_indices[:, s:e].reshape(-1)

        cand_native = ref_bank_norm.index_select(0, ids)
        cand_native = cand_native.reshape(
            B, e - s, *native_shape
        ).float()
        cand_sp = native_to_spatial(cand_native, token_grid=token_grid)

        if tuple(cand_sp.shape[2:]) != (C, H, W):
            raise RuntimeError(
                f"Candidate/query spatial mismatch: cand={tuple(cand_sp.shape)}, "
                f"query={tuple(z_t.shape)}"
            )

        diff2 = (
            (float(alpha) * cand_sp - z_t[:, None]) ** 2
        ).sum(dim=2)
        local_dist = box_sum_2d(diff2, patch_size)
        logits = -local_dist / denom_const

        chunk_max = logits.max(dim=1).values
        new_max = torch.maximum(running_max, chunk_max)
        old_scale = torch.exp(running_max - new_max)
        old_scale = torch.where(
            torch.isfinite(running_max), old_scale, torch.zeros_like(old_scale)
        )
        w = torch.exp(logits - new_max[:, None])

        running_den = running_den * old_scale + w.sum(dim=1)
        running_num = (
            running_num * old_scale[:, None]
            + (w[:, :, None] * cand_sp).sum(dim=1)
        )
        running_max = new_max

        del ids, cand_native, cand_sp, diff2, local_dist, logits, w

    return running_num / running_den.clamp_min(1e-20)[:, None]



# ---------------------------------------------------------------------
# Local Score
# ---------------------------------------------------------------------

def box_sum_2d(x: torch.Tensor, p: int):
    """
    x [B,K,H,W] -> zero-padded P×P sum, output [B,K,H,W].
    Matches the single-image Local Score prototype: padding contributes zero
    squared distance, equivalent to a smaller observed patch at boundaries.
    """
    if p == 1:
        return x
    if p <= 0 or p % 2 != 1:
        raise ValueError(p)

    b, k, h, w = x.shape
    y = x.reshape(b * k, 1, h, w)
    r = p // 2
    yp = F.pad(y, (r, r, r, r), mode="constant", value=0.0)
    ii = F.pad(yp, (1, 0, 1, 0), mode="constant", value=0.0)
    ii = ii.cumsum(-2).cumsum(-1)
    out = (
        ii[..., p:, p:]
        - ii[..., :-p, p:]
        - ii[..., p:, :-p]
        + ii[..., :-p, :-p]
    )
    return out.reshape(b, k, h, w)


@torch.no_grad()
def local_score_mu_topk(
    z_t: torch.Tensor,
    candidate_native_cpu: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    alpha: float,
    sigma: float,
    patch_size: int,
    token_grid,
    candidate_chunk_size: int,
    device,
):
    """
    Exact Local-Score posterior mean over the supplied K candidate pool.

    z_t: [B,C,H,W], normalized noisy query.
    candidate_native_cpu: [B,K,*native_shape], RAW (unnormalized) clean latents.

    Uses an online log-sum-exp accumulator so high-dimensional 25-VAE models
    do not require all K normalized candidates to live on GPU simultaneously.
    """
    B, K = candidate_native_cpu.shape[:2]
    _, C, H, W = z_t.shape

    mean_d = mean.to(device)
    std_d = std.to(device)

    running_max = torch.full(
        (B, H, W), -float("inf"), device=device, dtype=torch.float32
    )
    running_den = torch.zeros(
        (B, H, W), device=device, dtype=torch.float32
    )
    running_num = torch.zeros(
        (B, C, H, W), device=device, dtype=torch.float32
    )

    denom_const = 2.0 * float(sigma) * float(sigma)

    for s in range(0, K, candidate_chunk_size):
        e = min(s + candidate_chunk_size, K)

        cand_native = candidate_native_cpu[:, s:e].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        cand_norm_native = normalize_native(cand_native, mean_d, std_d)
        cand_sp = native_to_spatial(cand_norm_native, token_grid=token_grid)
        # [B,k,C,H,W]

        if tuple(cand_sp.shape[2:]) != (C, H, W):
            raise RuntimeError(
                f"Candidate/query spatial mismatch: cand={tuple(cand_sp.shape)}, "
                f"query={tuple(z_t.shape)}"
            )

        diff2 = (
            (float(alpha) * cand_sp - z_t[:, None]) ** 2
        ).sum(dim=2)  # [B,k,H,W]
        local_dist = box_sum_2d(diff2, patch_size)
        logits = -local_dist / denom_const

        chunk_max = logits.max(dim=1).values
        new_max = torch.maximum(running_max, chunk_max)

        old_scale = torch.exp(running_max - new_max)
        old_scale = torch.where(
            torch.isfinite(running_max), old_scale, torch.zeros_like(old_scale)
        )
        w = torch.exp(logits - new_max[:, None])

        running_den = running_den * old_scale + w.sum(dim=1)
        running_num = (
            running_num * old_scale[:, None]
            + (w[:, :, None] * cand_sp).sum(dim=1)
        )
        running_max = new_max

        del cand_native, cand_norm_native, cand_sp, diff2, local_dist, logits, w

    mu = running_num / running_den.clamp_min(1e-20)[:, None]
    return mu


# ---------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------

def inception_features(inception, x01):
    pred = inception(x01)[0]
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, (1, 1))
    return pred.squeeze(-1).squeeze(-1)


def fid_from_features(a: np.ndarray, b: np.ndarray):
    return float(
        calculate_frechet_distance(
            a.mean(0),
            np.cov(a, rowvar=False),
            b.mean(0),
            np.cov(b, rowvar=False),
        )
    )


class OnlineFeatureStats:
    """
    Streaming mean/covariance for Inception features.

    This is algebraically equivalent to numpy.cov(..., rowvar=False) with the
    default unbiased normalization (N-1), but avoids keeping all 50k x 2048
    features in host RAM.

    Per distribution, host RAM is ~32 MiB for the 2048x2048 float64 second
    moment instead of ~400 MiB for 50k float32 features.
    """
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sum_outer = np.zeros((self.dim, self.dim), dtype=np.float64)

    def update(self, x):
        if torch.is_tensor(x):
            x = x.detach().float().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected [N,{self.dim}] features, got {x.shape}")
        self.n += int(x.shape[0])
        self.sum += x.sum(axis=0)
        self.sum_outer += x.T @ x

    def mean_cov(self):
        if self.n < 2:
            raise RuntimeError(f"Need at least 2 features, got {self.n}")
        mean = self.sum / float(self.n)
        cov = (
            self.sum_outer
            - float(self.n) * np.outer(mean, mean)
        ) / float(self.n - 1)
        # Numerical symmetry cleanup.
        cov = (cov + cov.T) * 0.5
        return mean, cov


def fid_from_stats(a: OnlineFeatureStats, b: OnlineFeatureStats):
    if a.n != b.n:
        raise RuntimeError(f"Feature-count mismatch: {a.n} vs {b.n}")
    ma, ca = a.mean_cov()
    mb, cb = b.mean_cov()
    return float(calculate_frechet_distance(ma, ca, mb, cb))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@torch.no_grad()
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    if args.candidate_topk <= 0 or args.candidate_topk > args.ref_size:
        raise ValueError("--candidate-topk must be in [1, ref-size]")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    # ImageNetValDataset is the existing IFID repository dataset wrapper.
    train_set = ImageNetValDataset(args.dataset_ref, transform, args.ref_size)
    val_set = ImageNetValDataset(args.dataset, transform, args.val_size)

    train_loader = DataLoader(
        train_set,
        batch_size=args.ref_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.num_workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.num_workers > 0),
        drop_last=False,
    )

    vae = instantiate_from_config(OmegaConf.load(args.vae_config)).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Probe latent layout.
    fake_x = torch.zeros(1, 3, args.resolution, args.resolution, device=device)
    fake_y = torch.zeros(1, dtype=torch.long, device=device)
    fake_z = encode_vae(vae, fake_x, fake_y)
    native_shape = tuple(fake_z.shape[1:])
    latent_numel = int(np.prod(native_shape))
    tshift = math.sqrt(float(latent_numel) / 4096.0)

    if len(native_shape) == 2:
        infer_token_grid(int(native_shape[0]), args.token_grid)
    elif args.token_grid is not None:
        print("[WARN] --token-grid ignored because this VAE already returns BCHW.")

    cache_files, cache_shape, latent_mean, latent_std = build_or_load_reference_cache(
        args, vae, train_loader, device
    )
    if tuple(cache_shape) != native_shape:
        raise RuntimeError(
            f"Cache/probe latent shape mismatch: cache={cache_shape}, probe={native_shape}"
        )

    metric_dtype = dtype_from_name(args.metric_dtype)
    cache_dtype = dtype_from_name(args.cache_dtype)

    # Keep the normalized 50k reference bank resident on the A800. The old
    # implementation repeatedly torch.load()-ed every cache file for every
    # tiny query block, which left the GPU almost completely idle.
    bank_dtype = choose_gpu_bank_dtype(
        args, args.ref_size, native_shape, device
    )
    ref_bank_norm = load_normalized_reference_bank_gpu(
        cache_files=cache_files,
        total=args.ref_size,
        latent_shape=native_shape,
        mean=latent_mean,
        std=latent_std,
        device=device,
        bank_dtype=bank_dtype,
    )
    ref_sq = reference_sq_norms(
        ref_bank_norm, chunk_size=max(1024, args.metric_chunk_size)
    )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    setting_meta = []
    for t_raw, p in args.settings:
        te = effective_t(t_raw, tshift, args.time_mode)
        alpha, sigma = linear_interpolant(te)
        setting_meta.append(
            {
                "t_raw": float(t_raw),
                "patch_size": int(p),
                "t_effective": float(te),
                "alpha": float(alpha),
                "sigma": float(sigma),
                "key": f"t{float(t_raw):g}_p{int(p)}",
            }
        )

    print("=" * 72)
    print("Local-Score rFID (25-VAE version)")
    print("vae_config       :", args.vae_config)
    print("exp_name         :", args.exp_name)
    print("native_shape     :", native_shape)
    print("token_grid       :", args.token_grid)
    print("latent_numel     :", latent_numel)
    print("tshift           :", tshift)
    print("time_mode        :", args.time_mode)
    print("ref / val        :", args.ref_size, "/", args.val_size)
    print("candidate_topk   :", args.candidate_topk)
    print("query bs / ref bs:", args.bs, "/", args.ref_bs)
    print("query block imgs :", args.bs * args.query_block_batches)
    print("GPU bank dtype   :", ref_bank_norm.dtype)
    print("settings         :", [(x["t_raw"], x["patch_size"], x["t_effective"]) for x in setting_meta])
    print("normalization    : reference-bank channel mean/std")
    print("IMPORTANT        : top-K truncated Local Score, NOT exact all-50k LS")
    print("=" * 72)

    # Streaming FID moments: dramatically lower host/cgroup RAM.
    feat_dim = 2048
    src_stats = OnlineFeatureStats(feat_dim)
    rec_stats = OnlineFeatureStats(feat_dim)
    ls_stats: Dict[str, OnlineFeatureStats] = {
        m["key"]: OnlineFeatureStats(feat_dim) for m in setting_meta
    }

    # Only retain raw features when explicitly requested.
    src_feats = [] if args.keep_features else None
    rec_feats = [] if args.keep_features else None
    ls_feats: Dict[str, List[np.ndarray]] = (
        {m["key"]: [] for m in setting_meta} if args.keep_features else {}
    )

    preview_left = int(args.save_preview)
    preview_dir = out_dir / "preview"
    if preview_left > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(args.noise_seed)

    val_iter = iter(val_loader)
    total_batches = len(val_loader)
    done = 0

    pbar = tqdm(total=total_batches, desc=f"LS-rFID {args.exp_name}")

    while done < total_batches:
        n_block = min(args.query_block_batches, total_batches - done)
        query_records = []

        for _ in range(n_block):
            img, y, name = next(val_iter)
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = encode_vae(vae, img, y)
            query_records.append({"img": img, "y": y, "name": name, "z": z})

        # Exact same GLOBAL clean-normalized top-K prefilter, now done using one
        # GPU-resident reference bank and one concatenated query matrix.
        topk_index_batches = retrieve_topk_gpu_bank(
            ref_bank_norm=ref_bank_norm,
            ref_sq=ref_sq,
            query_records=query_records,
            mean=latent_mean,
            std=latent_std,
            K=args.candidate_topk,
            device=device,
            metric_chunk_size=args.metric_chunk_size,
        )

        for rec, cand_idx in zip(query_records, topk_index_batches):
            img, y, names = rec["img"], rec["y"], rec["name"]
            z_raw = rec["z"].float()

            mean_d = latent_mean.to(device)
            std_d = latent_std.to(device)
            z_norm_native = normalize_native(z_raw, mean_d, std_d)
            z_norm_sp = native_to_spatial(z_norm_native, args.token_grid)

            eps = torch.randn(
                z_norm_sp.shape,
                generator=noise_gen,
                device=device,
                dtype=torch.float32,
            )

            # Source + ordinary reconstruction.
            img_src01 = (img.float() + 1.0) / 2.0
            img_rec = decode_vae(vae, z_raw, y)
            img_rec01 = (img_rec.float() + 1.0) / 2.0

            src_f = inception_features(inception, img_src01).float()
            rec_f = inception_features(inception, img_rec01).float()
            src_stats.update(src_f)
            rec_stats.update(rec_f)
            if args.keep_features:
                src_feats.append(src_f.cpu().numpy())
                rec_feats.append(rec_f.cpu().numpy())
            del src_f, rec_f

            preview_imgs = {}

            for meta in setting_meta:
                alpha, sigma = meta["alpha"], meta["sigma"]
                z_t = alpha * z_norm_sp + sigma * eps

                mu_sp = local_score_mu_topk_gpu_bank(
                    z_t=z_t,
                    candidate_indices=cand_idx,
                    ref_bank_norm=ref_bank_norm,
                    alpha=alpha,
                    sigma=sigma,
                    patch_size=meta["patch_size"],
                    token_grid=args.token_grid,
                    candidate_chunk_size=args.candidate_chunk_size,
                )

                mu_native_norm = spatial_to_native(
                    mu_sp,
                    native_shape=native_shape,
                    token_grid=args.token_grid,
                )
                mu_native_raw = denormalize_native(
                    mu_native_norm,
                    mean_d,
                    std_d,
                )

                img_hat = decode_vae(vae, mu_native_raw, y)
                img_hat01 = (img_hat.float() + 1.0) / 2.0

                ls_f = inception_features(inception, img_hat01).float()
                ls_stats[meta["key"]].update(ls_f)
                if args.keep_features:
                    ls_feats[meta["key"]].append(ls_f.cpu().numpy())
                del ls_f

                if preview_left > 0:
                    preview_imgs[meta["key"]] = img_hat01.detach().cpu()

                del z_t, mu_sp, mu_native_norm, mu_native_raw, img_hat, img_hat01

            if preview_left > 0:
                src_cpu = img_src01.detach().cpu()
                rec_cpu = img_rec01.detach().cpu()
                nsave = min(preview_left, src_cpu.shape[0])
                for j in range(nsave):
                    from PIL import Image
                    def save_tensor(t, path):
                        arr = (
                            torch.clamp(t, 0, 1)
                            .mul(255).round().to(torch.uint8)
                            .permute(1, 2, 0).numpy()
                        )
                        Image.fromarray(arr).save(path)

                    stem = Path(str(names[j])).stem
                    save_tensor(src_cpu[j], preview_dir / f"{stem}_src.png")
                    save_tensor(rec_cpu[j], preview_dir / f"{stem}_recon.png")
                    for key, batch_imgs in preview_imgs.items():
                        save_tensor(batch_imgs[j], preview_dir / f"{stem}_{key}_ls.png")
                preview_left -= nsave

            del img_rec, img_src01, img_rec01, z_raw, z_norm_native, z_norm_sp, eps, cand_idx
            done += 1
            pbar.update(1)

        del query_records, topk_index_batches
        torch.cuda.empty_cache()

    pbar.close()

    if src_stats.n != args.val_size:
        raise RuntimeError(
            f"Expected {args.val_size} source features, got {src_stats.n}"
        )

    metrics = {
        "rfid": fid_from_stats(src_stats, rec_stats),
    }
    for meta in setting_meta:
        key = meta["key"]
        metrics[f"ls_rfid_{key}"] = fid_from_stats(src_stats, ls_stats[key])

    if args.keep_features:
        src = np.concatenate(src_feats, axis=0)
        rec = np.concatenate(rec_feats, axis=0)
        np.save(out_dir / "features_source.npy", src)
        np.save(out_dir / "features_recon.npy", rec)
        for meta in setting_meta:
            key = meta["key"]
            pred = np.concatenate(ls_feats[key], axis=0)
            np.save(out_dir / f"features_{key}.npy", pred)

    result = {
        "vae_config": args.vae_config,
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "dataset_ref": args.dataset_ref,
        "ref_size": int(args.ref_size),
        "val_size": int(args.val_size),
        "native_latent_shape": list(native_shape),
        "token_grid": args.token_grid,
        "latent_numel": latent_numel,
        "tshift": float(tshift),
        "time_mode": args.time_mode,
        "path_type": args.path_type,
        "normalization": "reference-bank per-channel mean/std",
        "candidate_prefilter": "global top-K squared L2 on clean normalized latent",
        "candidate_topk": int(args.candidate_topk),
        "retrieval_impl": "GPU-resident normalized reference bank; batched exact global top-K over the same cached 50k refs",
        "gpu_bank_dtype": str(ref_bank_norm.dtype),
        "query_block_images": int(args.bs * args.query_block_batches),
        "fid_feature_accumulation": "streaming float64 sum + second moment; no 50k feature retention unless --keep-features",
        "local_score_exactness": (
            "exact posterior over candidate_topk only; truncated approximation "
            "to all-50k Local Score"
        ),
        "same_noise_across_settings_per_query": True,
        "noise_seed": int(args.noise_seed),
        "settings": setting_meta,
        "metrics": metrics,
    }

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        fieldnames = ["metric", "value"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k, v in metrics.items():
            w.writerow({"metric": k, "value": f"{v:.10f}"})

    print("\n" + "=" * 72)
    print("DONE:", args.exp_name)
    for k, v in metrics.items():
        print(f"{k:24s} = {v:.6f}")
    print("metrics:", out_dir / "metrics.json")
    print("=" * 72)


if __name__ == "__main__":
    main(parse_args())
