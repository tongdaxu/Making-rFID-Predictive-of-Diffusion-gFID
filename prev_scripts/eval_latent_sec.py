#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute SEC / Spectral Energy Concentration for IFID VAE latents.

This implementation follows KlingAIResearch/diffusing-right-space exactly
for the SEC metric definition:

  latents = latents.float()
  latent_dct = torch_dct.dct_2d(latents, norm="ortho")
  energy = latent_dct ** 2
  avg_energy = energy.mean(dim=(0, 1))
  dist_map = u + v for Manhattan, or sqrt(u^2+v^2) for Euclidean
  high_freq_mask = dist_map > threshold * max_dist
  SEC@threshold = high_energy / total_energy

It also follows their per_channel_norm before SEC:

  mean = latents.mean(dim=[0, 2, 3], keepdim=True)
  std  = latents.std(dim=[0, 2, 3], keepdim=True)  # torch default correction=1
  latents = (latents - mean) / std

Because 50k latents can be large, this script computes the same quantities
with a two-pass/cache pipeline:
  1. encode all images, cache spatial latents, and compute global channel mean/std
  2. stream normalized latents through DCT and accumulate sum of energy over B,C

The streamed avg_energy is mathematically equivalent to original
energy.mean(dim=(0, 1)) on the full normalized tensor.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from omegaconf import OmegaConf

try:
    import torch_dct as dct
except ImportError as e:
    raise ImportError(
        "SEC requires torch-dct to match the original implementation. "
        "Install it with: pip install torch-dct"
    ) from e

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config


# ---------------- VAE helpers ----------------
def unwrap_vae_latents(z: Any) -> torch.Tensor:
    if isinstance(z, torch.Tensor):
        return z
    if isinstance(z, (tuple, list)):
        z = z[0]
        if isinstance(z, torch.Tensor):
            return z
    if isinstance(z, dict):
        for key in ["z", "latent", "latents", "sample", "moments"]:
            if key in z:
                z = z[key]
                break
        if isinstance(z, torch.Tensor):
            return z
    if hasattr(z, "mode") and callable(z.mode):
        z = z.mode()
    elif hasattr(z, "sample") and callable(z.sample):
        z = z.sample()
    if isinstance(z, torch.Tensor):
        return z
    raise TypeError(f"Unsupported VAE encode output type: {type(z)}")


@torch.no_grad()
def encode_vae(vae, images, labels=None):
    try:
        if labels is not None:
            z = vae.encode(images, labels)
        else:
            z = vae.encode(images)
    except TypeError:
        z = vae.encode(images)
    return unwrap_vae_latents(z)


def infer_hw_from_tokens(n: int, token_grid: Optional[List[int]]) -> Tuple[int, int]:
    if token_grid is not None:
        h, w = int(token_grid[0]), int(token_grid[1])
        if h * w != n:
            raise ValueError(f"--token-grid {h} {w} product != N={n}")
        return h, w
    h = int(math.isqrt(n))
    if h * h != n:
        raise ValueError(f"Token length N={n} is not square. Pass --token-grid H W.")
    return h, h


def to_spatial_latent(z: torch.Tensor, token_grid: Optional[List[int]] = None) -> torch.Tensor:
    if z.ndim == 4:
        return z
    if z.ndim == 3:
        # Assume [B,N,C], same convention as eval_latent_viv_cds_srss.py.
        b, n, c = z.shape
        h, w = infer_hw_from_tokens(n, token_grid)
        return z.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
    raise ValueError(f"Expected [B,C,H,W] or [B,N,C], got {tuple(z.shape)}")


def latent_channel_sum_sumsq_count(z4: torch.Tensor):
    z = z4.float()
    s = z.sum(dim=(0, 2, 3))
    ss = (z * z).sum(dim=(0, 2, 3))
    cnt = z.shape[0] * z.shape[2] * z.shape[3]
    return s, ss, cnt


def normalize_per_channel(z4: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-12):
    mean = mean.to(z4.device, dtype=torch.float32).view(1, -1, 1, 1)
    std = std.to(z4.device, dtype=torch.float32).view(1, -1, 1, 1).clamp_min(eps)
    return (z4.float() - mean) / std


# ---------------- Original SEC implementation ----------------
@torch.no_grad()
def spectral_energy_concentration_original(
    latents: torch.Tensor,
    thresholds: List[float] = [0.25, 0.5],
    dist_type: str = "Manhattan",
):
    """
    Direct implementation of KlingAIResearch/diffusing-right-space metrics/sec.py.
    Use this for small tensors or validation. For 50k, the main path below streams
    the exact same avg_energy calculation batch-by-batch.
    """
    assert dist_type in ["Euclidean", "Manhattan"]
    latents = latents.float()
    B, C, H, W = latents.shape

    latent_dct = dct.dct_2d(latents, norm="ortho")
    energy = latent_dct ** 2
    avg_energy = energy.mean(dim=(0, 1))

    u = torch.arange(H, device=latents.device).view(-1, 1).expand(H, W).float()
    v = torch.arange(W, device=latents.device).view(1, -1).expand(H, W).float()

    if dist_type == "Manhattan":
        dist_map = u + v
        max_dist = (H - 1) + (W - 1)
    else:
        dist_map = torch.sqrt(u ** 2 + v ** 2)
        max_dist = math.sqrt((H - 1) ** 2 + (W - 1) ** 2)

    total_energy = avg_energy.sum()
    results = {}

    for th in thresholds:
        actual_th = th * max_dist
        high_freq_mask = dist_map > actual_th
        high_energy = (avg_energy * high_freq_mask).sum()
        ratio = (high_energy / total_energy).item() if total_energy > 0 else 0.0
        results[float(th)] = float(ratio)

    return results


def make_dist_map(H: int, W: int, device: torch.device, dist_type: str):
    assert dist_type in ["Euclidean", "Manhattan"]
    u = torch.arange(H, device=device).view(-1, 1).expand(H, W).float()
    v = torch.arange(W, device=device).view(1, -1).expand(H, W).float()

    if dist_type == "Manhattan":
        dist_map = u + v
        max_dist = (H - 1) + (W - 1)
    else:
        dist_map = torch.sqrt(u ** 2 + v ** 2)
        max_dist = math.sqrt((H - 1) ** 2 + (W - 1) ** 2)

    return dist_map, float(max_dist)


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--token-grid", nargs=2, type=int, default=None)
    parser.add_argument("--store-dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--dist-type", type=str, default="Manhattan", choices=["Manhattan", "Euclidean"])
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--save-avg-energy", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = ImageNetValDataset(args.dataset, transform, args.num_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    meta_path = out_dir / "latent_cache_meta.json"
    labels_path = out_dir / "labels.npy"
    names_path = out_dir / "names.txt"

    if not (args.reuse_cache and meta_path.exists()):
        vae_config = OmegaConf.load(args.vae_config)
        vae = instantiate_from_config(vae_config).to(device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)

        print("Encoding latents and computing channel statistics...")
        pos = 0
        latent_shape = None
        spatial_shape = None
        C = H = W = D = None
        dtype_np = np.float32 if args.store_dtype == "fp32" else np.float16
        mmap = None
        labels_all = np.empty((len(dataset),), dtype=np.int64)
        names_all = []

        stat_sum = None
        stat_sumsq = None
        stat_count = 0
        cache_path = out_dir / f"latents_spatial_{args.store_dtype}.dat"

        for img, y, names in tqdm(loader, desc="Encode"):
            img = img.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True).long()
            z = encode_vae(vae, img, y_dev)
            z4 = to_spatial_latent(z, token_grid=args.token_grid)

            if spatial_shape is None:
                latent_shape = list(z.shape[1:])
                _, C, H, W = z4.shape
                spatial_shape = [int(C), int(H), int(W)]
                D = int(C * H * W)
                mmap = np.memmap(cache_path, dtype=dtype_np, mode="w+", shape=(len(dataset), D))
                stat_sum = torch.zeros(C, dtype=torch.float64, device=device)
                stat_sumsq = torch.zeros(C, dtype=torch.float64, device=device)

            s, ss, cnt = latent_channel_sum_sumsq_count(z4)
            stat_sum += s.to(torch.float64)
            stat_sumsq += ss.to(torch.float64)
            stat_count += int(cnt)

            z_np = z4.reshape(z4.shape[0], -1).detach().float().cpu().numpy().astype(dtype_np)
            b = z_np.shape[0]
            mmap[pos:pos + b] = z_np
            labels_all[pos:pos + b] = y.detach().cpu().numpy().astype(np.int64)
            names_all.extend([str(n) for n in names])
            pos += b

            del img, y_dev, z, z4, z_np
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mmap.flush()

        # Match torch.std(..., correction=1) in diffusing-right-space metrics/norm.py.
        n = float(stat_count)
        mean_t = stat_sum / n
        if stat_count > 1:
            var_unbiased = (stat_sumsq - (stat_sum * stat_sum) / n) / (n - 1.0)
        else:
            var_unbiased = torch.zeros_like(stat_sumsq)
        std_t = torch.sqrt(var_unbiased.clamp_min(1e-12))

        mean = mean_t.detach().cpu().float().numpy()
        std = std_t.detach().cpu().float().numpy()

        np.save(labels_path, labels_all[:pos])
        with open(names_path, "w") as f:
            for nme in names_all:
                f.write(nme + "\n")

        meta = {
            "vae_config": args.vae_config,
            "dataset": args.dataset,
            "num_images": int(pos),
            "latent_shape": latent_shape,
            "spatial_shape": spatial_shape,
            "D": int(D),
            "store_dtype": args.store_dtype,
            "cache_path": str(cache_path),
            "labels_path": str(labels_path),
            "names_path": str(names_path),
            "channel_mean": mean.tolist(),
            "channel_std": std.tolist(),
            "std_correction": 1,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    else:
        print(f"Reusing cache from {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    N = int(meta["num_images"])
    C, H, W = map(int, meta["spatial_shape"])
    D = int(meta["D"])
    dtype_np = np.float32 if meta["store_dtype"] == "fp32" else np.float16
    latents_mm = np.memmap(meta["cache_path"], dtype=dtype_np, mode="r", shape=(N, D))

    mean_t = torch.tensor(meta["channel_mean"], dtype=torch.float32)
    std_t = torch.tensor(meta["channel_std"], dtype=torch.float32)

    print("Computing SEC with original torch_dct.dct_2d(norm='ortho') definition...")
    energy_sum = torch.zeros((H, W), dtype=torch.float64, device=device)
    count_bc = 0

    for start in tqdm(range(0, N, args.batch_size), desc="SEC"):
        end = min(start + args.batch_size, N)
        z_np = np.asarray(latents_mm[start:end])
        z4 = torch.from_numpy(z_np).reshape(end - start, C, H, W).to(device=device, dtype=torch.float32)
        z4 = normalize_per_channel(z4, mean_t, std_t)

        latent_dct = dct.dct_2d(z4.float(), norm="ortho")
        energy = latent_dct ** 2
        energy_sum += energy.sum(dim=(0, 1)).to(torch.float64)
        count_bc += int(z4.shape[0] * z4.shape[1])

        del z4, latent_dct, energy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_energy = (energy_sum / float(count_bc)).to(torch.float32)
    dist_map, max_dist = make_dist_map(H, W, device, args.dist_type)

    total_energy = avg_energy.sum()
    sec_results = {}
    for th in args.thresholds:
        th = float(th)
        actual_th = th * max_dist
        high_freq_mask = dist_map > actual_th
        high_energy = (avg_energy * high_freq_mask).sum()
        ratio = (high_energy / total_energy).item() if total_energy > 0 else 0.0
        sec_results[th] = float(ratio)

    results = {
        "vae_config": args.vae_config,
        "dataset": args.dataset,
        "num_images": N,
        "latent_shape": meta["latent_shape"],
        "spatial_shape": meta["spatial_shape"],
        "store_dtype": meta["store_dtype"],
        "std_correction": meta.get("std_correction", None),
        "metric": "SEC",
        "sec_definition": "KlingAIResearch/diffusing-right-space metrics/sec.py",
        "normalization": "per_channel_norm from metrics/norm.py, torch.std correction=1 over [0,2,3]",
        "thresholds": [float(x) for x in args.thresholds],
        "dist_type": args.dist_type,
        "max_dist": float(max_dist),
        "SEC": {str(k): v for k, v in sec_results.items()},
    }

    for k, v in sec_results.items():
        # Convenient flat keys for CSV/table collection.
        results[f"SEC@{k:g}"] = float(v)

    if args.save_avg_energy:
        np.save(out_dir / "sec_avg_energy.npy", avg_energy.detach().cpu().numpy())
        np.save(out_dir / "sec_dist_map.npy", dist_map.detach().cpu().numpy())

    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
