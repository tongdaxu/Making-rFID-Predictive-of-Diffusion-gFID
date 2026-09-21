#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute VIV / CDS / SRSS for IFID VAE latents.

Strict changes in v2:
  1. cache dtype defaults to fp32.
  2. per-channel std uses correction=1, matching torch.std default.
  3. SRSS requires masks; if requested without masks, this script errors.
  4. mask downsampling follows iREPA style: binary mask + max_pool2d.
  5. flat masks [B,T] are accepted directly and not mistaken for 2D images.
  6. SRSS mask filtering supports:
     - adaptive: pos/neg patch counts >= ceil(frac * T), default frac=0.25
     - strict: original iREPA 16x16 comparison threshold 64/64
     - none: no mask-count filtering

Sources followed:
  - KlingAIResearch/diffusing-right-space metrics/viv.py and metrics/norm.py
  - End2End-Diffusion/iREPA metrics/spatial_metrics.py and spatial_metrics_comparison.py
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from omegaconf import OmegaConf

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
        # assume [B,N,C]
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


# ---------------- iREPA-style spatial metrics ----------------
class GridCache:
    def __init__(self):
        self._dist = {}

    def dist(self, H: int, W: int, device: torch.device):
        key = (H, W, str(device))
        if key not in self._dist:
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )
            coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
            self._dist[key] = (coords[:, None, :] - coords[None, :, :]).abs().sum(-1)
        return self._dist[key]


_GRID = GridCache()


@torch.no_grad()
def metric_cds_from_z4(z4: torch.Tensor, dmax: int = 8) -> torch.Tensor:
    B, C, H, W = z4.shape
    u = z4.flatten(2).transpose(1, 2).contiguous()
    u = F.normalize(u.float(), dim=-1, eps=1e-8)
    G = torch.einsum("btd,bsd->bts", u, u)
    dist = _GRID.dist(H, W, z4.device)

    sims, ds = [], []
    for d in range(1, dmax + 1):
        m = dist == d
        if m.any():
            sims.append(G[:, m].mean(dim=1))
            ds.append(d)
    if not sims:
        return torch.zeros(B, device=z4.device)

    S = torch.stack(sims, dim=1)
    x = torch.tensor(ds, device=z4.device, dtype=S.dtype)
    x0 = x - x.mean()
    denom = (x0 @ x0).clamp(min=1e-12)
    slope = ((S - S.mean(dim=1, keepdim=True)) @ x0) / denom
    return -slope


@torch.no_grad()
def metric_srss_from_z4(
    z4: torch.Tensor,
    masks_flat: torch.Tensor,
    d_pos: int = 1,
    d_pos_min: int = 1,
    d_neg: int = 6,
) -> torch.Tensor:
    B, C, H, W = z4.shape
    T = H * W
    if masks_flat.shape != (B, T):
        raise ValueError(f"masks_flat must be {(B,T)}, got {tuple(masks_flat.shape)}")

    u = z4.flatten(2).transpose(1, 2).contiguous()
    u = F.normalize(u.float(), dim=-1, eps=1e-8)
    G = u @ u.transpose(1, 2)
    dist = _GRID.dist(H, W, z4.device)

    pos_dist_mask = (dist <= d_pos) & (dist >= d_pos_min)
    neg_dist_mask = dist >= d_neg

    out = []
    for b in range(B):
        fg = masks_flat[b].to(torch.bool)
        if fg.sum() < 2 or (~fg).sum() < 1:
            out.append(torch.tensor(float("nan"), device=z4.device))
            continue

        pos_mask = (fg[:, None] & fg[None, :]) & pos_dist_mask
        neg_mask = (fg[:, None] & (~fg)[None, :]) & neg_dist_mask

        Gb = G[b]
        pos_counts = pos_mask.sum(dim=1)
        neg_counts = neg_mask.sum(dim=1)
        pos_means = (Gb * pos_mask).sum(dim=1) / pos_counts.clamp_min(1)
        neg_means = (Gb * neg_mask).sum(dim=1) / neg_counts.clamp_min(1)

        valid = fg & (pos_counts > 0) & (neg_counts > 0)
        if valid.any():
            out.append((pos_means[valid] - neg_means[valid]).mean())
        else:
            out.append(torch.tensor(float("nan"), device=z4.device))
    return torch.stack(out, dim=0)


# ---------------- VIV ----------------
@torch.no_grad()
def viv_one_class(z4: torch.Tensor, num_iters: int = 5) -> float:
    z4 = z4.float()
    N, C, H, W = z4.shape
    S_dim = H * W

    z = z4.reshape(N, -1)
    v_total = z.var(dim=0, unbiased=False).sum()

    z_mat = z4.reshape(N, C, S_dim)
    z_mat = z_mat - z_mat.mean(dim=0, keepdim=True)

    W_S = None
    for _ in range(num_iters):
        if W_S is None:
            z_C = z_mat.permute(0, 2, 1).reshape(-1, C)
        else:
            z_whitened_S = torch.matmul(z_mat, W_S)
            z_C = z_whitened_S.permute(0, 2, 1).reshape(-1, C)

        _, S_C, Vh_C = torch.linalg.svd(z_C, full_matrices=False)
        eig_C = (S_C ** 2) / max(z_C.shape[0] - 1, 1)
        W_C = Vh_C.T * (1.0 / torch.sqrt(eig_C + 1e-12)).unsqueeze(0)

        z_mat_perm = z_mat.permute(0, 2, 1)
        z_whitened_C = torch.matmul(z_mat_perm, W_C).permute(0, 2, 1)
        z_S = z_whitened_C.reshape(-1, S_dim)

        _, S_S, Vh_S = torch.linalg.svd(z_S, full_matrices=False)
        eig_S = (S_S ** 2) / max(z_S.shape[0] - 1, 1)
        W_S = Vh_S.T * (1.0 / torch.sqrt(eig_S + 1e-12)).unsqueeze(0)

    eig_vals = (eig_C.unsqueeze(1) * eig_S.unsqueeze(0)).flatten()
    eig_vals = torch.sort(eig_vals, descending=True).values
    c_scale = v_total / (eig_vals.sum() + 1e-12)
    lambda_d = c_scale * eig_vals
    viv = ((math.pi / 2.0) * torch.sqrt(lambda_d.clamp_min(0.0))).mean().item()
    return float(viv)


# ---------------- mask loading/downsampling ----------------
def load_mask_npz(path: Optional[str]):
    if path is None:
        return None
    obj = np.load(path)
    if "masks" in obj:
        return obj["masks"]
    if "arr_0" in obj:
        return obj["arr_0"]
    raise KeyError(f"{path} must contain key 'masks' or 'arr_0'")


def load_mask_from_dir(mask_dir: str, name: str) -> np.ndarray:
    root = Path(mask_dir)
    name_path = Path(str(name))
    stem = name_path.stem
    candidates = [
        root / f"{stem}.png",
        root / f"{stem}.jpg",
        root / f"{stem}.jpeg",
        root / str(name),
        root / name_path.name,
    ]
    for p in candidates:
        if p.exists():
            return np.asarray(Image.open(p).convert("L"))
    raise FileNotFoundError(f"Cannot find mask for name={name} under {mask_dir}")


def masks_to_flat_irepa(mask_np: np.ndarray, H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Convert masks to flat [B,H*W] bool.

    Accepted:
      [B,T] where T == H*W: already flat iREPA-style masks, returned directly.
      [B,H,W]: already token-grid masks.
      [B,H0,W0]: high-res masks. Binarize and max_pool2d to HxW.
      [H0,W0]: single mask, treated as [1,H0,W0].
    """
    arr = np.asarray(mask_np)

    if arr.ndim == 1:
        if arr.shape[0] != H * W:
            raise ValueError(f"Flat mask length {arr.shape[0]} != H*W={H*W}")
        arr = arr[None, :]

    if arr.ndim == 2:
        # Could be [B,T] or a single high-res [H0,W0].
        if arr.shape[1] == H * W:
            return torch.from_numpy(arr > 0).to(device=device, dtype=torch.bool)
        if arr.shape == (H, W):
            arr = arr[None, :, :]
        else:
            arr = arr[None, :, :]

    if arr.ndim != 3:
        raise ValueError(f"Expected mask [B,T], [B,H,W], or [B,H0,W0], got {arr.shape}")

    B, H0, W0 = arr.shape
    if H0 == H and W0 == W:
        return torch.from_numpy((arr > 0).reshape(B, H * W)).to(device=device, dtype=torch.bool)

    if H0 % H != 0 or W0 % W != 0:
        raise ValueError(
            f"Cannot strict max_pool mask from {(H0,W0)} to {(H,W)}: dimensions are not divisible. "
            "Preprocess masks to a divisible resolution or provide flat [B,H*W] masks."
        )

    kh, kw = H0 // H, W0 // W
    masks = torch.from_numpy(arr).float().unsqueeze(1)  # [B,1,H0,W0]
    masks = (masks > 0).float()
    pooled = F.max_pool2d(masks, kernel_size=(kh, kw), stride=(kh, kw))
    if pooled.shape[-2:] != (H, W):
        raise RuntimeError(f"max_pool2d produced {pooled.shape[-2:]}, expected {(H,W)}")
    return (pooled[:, 0] > 0).reshape(B, H * W).to(device=device, dtype=torch.bool)


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
    parser.add_argument("--metrics", nargs="+", default=["viv", "cds"], choices=["viv", "cds", "srss"])
    parser.add_argument("--token-grid", nargs=2, type=int, default=None)
    parser.add_argument("--store-dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("--cds-dmax", type=int, default=8)
    parser.add_argument("--srss-d-pos", type=int, default=1)
    parser.add_argument("--srss-d-neg", type=int, default=6)
    parser.add_argument("--srss-filter-mode", type=str, default="adaptive", choices=["adaptive", "strict", "none"],
                        help=(
                            "SRSS mask-count filtering mode. "
                            "adaptive: require ceil(frac*T) foreground/background tokens; "
                            "strict: require fixed patch counts, default 64/64 like iREPA 16x16 comparison; "
                            "none: disable count filtering."
                        ))
    parser.add_argument("--srss-min-pos-frac", type=float, default=0.25,
                        help="Adaptive SRSS filter foreground fraction. 0.25 gives 64 for a 16x16 grid.")
    parser.add_argument("--srss-min-neg-frac", type=float, default=0.25,
                        help="Adaptive SRSS filter background fraction. 0.25 gives 64 for a 16x16 grid.")
    parser.add_argument("--srss-min-pos-patches", type=int, default=64,
                        help="Strict SRSS filter foreground patch count, used when --srss-filter-mode strict.")
    parser.add_argument("--srss-min-neg-patches", type=int, default=64,
                        help="Strict SRSS filter background patch count, used when --srss-filter-mode strict.")
    parser.add_argument("--no-srss-mask-filter", action="store_true",
                        help="Deprecated alias for --srss-filter-mode none.")
    parser.add_argument("--mask-npz", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--viv-num-iters", type=int, default=5)
    parser.add_argument("--viv-max-classes", type=int, default=-1)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    if args.no_srss_mask_filter:
        args.srss_filter_mode = "none"

    if "srss" in args.metrics and args.mask_npz is None and args.mask_dir is None:
        raise ValueError(
            "SRSS requested but no mask source provided. "
            "Pass --mask-npz with [N,T] / [N,H,W] / [N,H0,W0] masks, or --mask-dir. "
            "This is intentional: official SRSS is mask-aware."
        )

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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

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
            torch.cuda.empty_cache()

        mmap.flush()

        # Match torch.std(..., correction=1) used by per_channel_norm.
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
    labels = np.load(meta["labels_path"])[:N]
    with open(meta["names_path"], "r") as f:
        names = [line.strip() for line in f.readlines()[:N]]

    mean_t = torch.tensor(meta["channel_mean"], dtype=torch.float32)
    std_t = torch.tensor(meta["channel_std"], dtype=torch.float32)

    results: Dict[str, Any] = {
        "vae_config": args.vae_config,
        "dataset": args.dataset,
        "num_images": N,
        "latent_shape": meta["latent_shape"],
        "spatial_shape": meta["spatial_shape"],
        "store_dtype": meta["store_dtype"],
        "std_correction": meta.get("std_correction", None),
        "metrics_requested": args.metrics,
        "srss_filter_mode": args.srss_filter_mode if "srss" in args.metrics else None,
        "srss_min_pos_frac": float(args.srss_min_pos_frac),
        "srss_min_neg_frac": float(args.srss_min_neg_frac),
        "srss_strict_min_pos_patches": int(args.srss_min_pos_patches),
        "srss_strict_min_neg_patches": int(args.srss_min_neg_patches),
    }

    mask_npz_arr = load_mask_npz(args.mask_npz)

    # CDS / SRSS batchwise
    if "cds" in args.metrics or "srss" in args.metrics:
        cds_vals = []
        srss_vals = []
        srss_keep_flags = []
        srss_pos_counts_all = []
        srss_neg_counts_all = []

        for start in tqdm(range(0, N, args.batch_size), desc="CDS/SRSS"):
            end = min(start + args.batch_size, N)
            z_np = np.asarray(latents_mm[start:end])
            z4 = torch.from_numpy(z_np).reshape(end - start, C, H, W).to(device=device, dtype=torch.float32)
            z4 = normalize_per_channel(z4, mean_t, std_t)

            if "cds" in args.metrics:
                cds = metric_cds_from_z4(z4, dmax=args.cds_dmax)
                cds_vals.append(cds.detach().cpu())

            if "srss" in args.metrics:
                if mask_npz_arr is not None:
                    masks_np = mask_npz_arr[start:end]
                else:
                    masks_np = np.stack([load_mask_from_dir(args.mask_dir, n) for n in names[start:end]], axis=0)
                masks_flat = masks_to_flat_irepa(masks_np, H, W, device)

                # iREPA comparison pipeline filters masks after max-pool downsampling.
                # Original iREPA comparison used 64/64 on a 16x16 grid (T=256), i.e. 25%/25%.
                # For non-16x16 latent grids, adaptive mode preserves this ratio.
                pos_counts = masks_flat.sum(dim=1).to(torch.int64)
                neg_counts = (masks_flat.shape[1] - pos_counts).to(torch.int64)
                T_mask = int(masks_flat.shape[1])
                if args.srss_filter_mode == "none":
                    min_pos = 0
                    min_neg = 0
                    keep = torch.ones_like(pos_counts, dtype=torch.bool)
                elif args.srss_filter_mode == "strict":
                    min_pos = int(args.srss_min_pos_patches)
                    min_neg = int(args.srss_min_neg_patches)
                    keep = (pos_counts >= min_pos) & (neg_counts >= min_neg)
                elif args.srss_filter_mode == "adaptive":
                    min_pos = max(1, int(math.ceil(float(args.srss_min_pos_frac) * T_mask)))
                    min_neg = max(1, int(math.ceil(float(args.srss_min_neg_frac) * T_mask)))
                    keep = (pos_counts >= min_pos) & (neg_counts >= min_neg)
                else:
                    raise ValueError(f"Unknown srss_filter_mode: {args.srss_filter_mode}")

                srss_keep_flags.append(keep.detach().cpu())
                srss_pos_counts_all.append(pos_counts.detach().cpu())
                srss_neg_counts_all.append(neg_counts.detach().cpu())

                if keep.any():
                    srss = metric_srss_from_z4(
                        z4[keep],
                        masks_flat[keep],
                        d_pos=args.srss_d_pos,
                        d_neg=args.srss_d_neg,
                    )
                    srss_vals.append(srss.detach().cpu())

            del z4
            torch.cuda.empty_cache()

        if cds_vals:
            cds_all = torch.cat(cds_vals)
            valid = ~torch.isnan(cds_all)
            results["CDS"] = float(cds_all[valid].mean().item()) if valid.any() else None
            results["CDS_std"] = float(cds_all[valid].std().item()) if valid.sum() > 1 else 0.0
            np.save(out_dir / "cds_per_image.npy", cds_all.numpy())

        if "srss" in args.metrics:
            if srss_keep_flags:
                keep_all = torch.cat(srss_keep_flags)
                pos_counts_all = torch.cat(srss_pos_counts_all)
                neg_counts_all = torch.cat(srss_neg_counts_all)
                np.save(out_dir / "srss_keep_mask.npy", keep_all.numpy())
                np.save(out_dir / "srss_pos_counts.npy", pos_counts_all.numpy())
                np.save(out_dir / "srss_neg_counts.npy", neg_counts_all.numpy())
                results["SRSS_filter_keep_ratio"] = float(keep_all.float().mean().item())
                results["SRSS_num_kept_after_mask_filter"] = int(keep_all.sum().item())
                results["SRSS_num_total_masks"] = int(keep_all.numel())
                results["SRSS_filter_note"] = (
                    "adaptive uses ceil(frac*T) per latent grid; strict uses fixed 64/64-style thresholds"
                )

            if srss_vals:
                srss_all = torch.cat(srss_vals)
                valid = ~torch.isnan(srss_all)
                results["SRSS"] = float(srss_all[valid].mean().item()) if valid.any() else None
                results["SRSS_std"] = float(srss_all[valid].std().item()) if valid.sum() > 1 else 0.0
                results["SRSS_valid_ratio_after_filter"] = float(valid.float().mean().item())
                np.save(out_dir / "srss_per_kept_image.npy", srss_all.numpy())
            else:
                results["SRSS"] = None
                results["SRSS_note"] = "No samples remained after SRSS mask filtering, or all SRSS values were invalid."

    # VIV by class
    if "viv" in args.metrics:
        label_to_indices = defaultdict(list)
        for idx, lab in enumerate(labels.tolist()):
            label_to_indices[int(lab)].append(idx)

        classes = sorted(label_to_indices.keys())
        if args.viv_max_classes is not None and args.viv_max_classes > 0:
            classes = classes[: args.viv_max_classes]

        viv_values = {}
        for cls in tqdm(classes, desc="VIV by class"):
            idxs = label_to_indices[cls]
            if len(idxs) < 2:
                continue
            z_np = np.asarray(latents_mm[np.asarray(idxs)])
            z4 = torch.from_numpy(z_np).reshape(len(idxs), C, H, W).to(device=device, dtype=torch.float32)
            z4 = normalize_per_channel(z4, mean_t, std_t)
            viv = viv_one_class(z4, num_iters=args.viv_num_iters)
            viv_values[str(cls)] = viv
            del z4
            torch.cuda.empty_cache()

        results["VIV"] = float(np.mean(list(viv_values.values()))) if viv_values else None
        results["VIV_num_classes"] = len(viv_values)
        with open(out_dir / "viv_per_class.json", "w") as f:
            json.dump(viv_values, f, indent=2)

    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
