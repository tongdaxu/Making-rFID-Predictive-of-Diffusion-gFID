#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute latent mask-cos metrics for IFID VAE/tokenizer configs.

Metric definition requested by user:
  for t in [0.1, 0.4, 0.6, 0.9]:
    latent = vae.encode(x)  # converted to [B,C,H,W]
    noise = randn_like(latent)
    latent_plus = random_spatial_mask(latent, mask_ratio=0.75)
    latent_t  = t * latent      + (1-t) * noise
    latent_pt = t * latent_plus + (1-t) * noise
    fs        = flatten spatial to [B,T,C]
    fs_tilde  = flatten spatial to [B,T,C]
    align_t   = mean_flat(cos(fs, fs_tilde))
    entropy_t = mean_flat(cross-sample cos(fs, fs))

Outputs:
  output-dir/metrics.json
  output-dir/per_batch_metrics.csv

Notes:
  - This is an evaluation metric, not a training loss.
  - [B,N,C] token latents are converted to [B,C,H,W]. If N is not square,
    pass --token-grid H W.
  - For fairness, use the same batch size, data order, t-values, mask-ratio,
    normalize mode, and seed across all VAEs.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config


# ------------------------- IFID VAE helpers -------------------------

def unwrap_vae_latents(z: Any) -> torch.Tensor:
    """Robustly unwrap common VAE encode outputs into a Tensor."""
    if isinstance(z, torch.Tensor):
        return z
    if isinstance(z, (tuple, list)):
        z0 = z[0]
        if isinstance(z0, torch.Tensor):
            return z0
        z = z0
    if isinstance(z, dict):
        for key in ["z", "latent", "latents", "sample", "moments", "mean"]:
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
def encode_vae(vae, images: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Call vae.encode(images, labels) when supported, otherwise vae.encode(images)."""
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
            raise ValueError(f"--token-grid {h} {w} product != token length N={n}")
        return h, w
    h = int(math.isqrt(n))
    if h * h != n:
        raise ValueError(
            f"Token latent length N={n} is not square. Pass --token-grid H W."
        )
    return h, h


def to_spatial_latent(z: torch.Tensor, token_grid: Optional[List[int]] = None) -> torch.Tensor:
    """Convert latent to [B,C,H,W]. Accepts [B,C,H,W] or [B,N,C]."""
    if z.ndim == 4:
        return z.contiguous()
    if z.ndim == 3:
        # assume [B,N,C]
        b, n, c = z.shape
        h, w = infer_hw_from_tokens(n, token_grid)
        return z.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
    raise ValueError(f"Expected latent [B,C,H,W] or [B,N,C], got {tuple(z.shape)}")


# ------------------------- metric functions -------------------------

def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Mean over non-batch dimensions, matching common diffusion mean_flat."""
    if x.ndim <= 1:
        return x
    return x.mean(dim=tuple(range(1, x.ndim)))


def random_spatial_mask(x: torch.Tensor, mask_ratio: float = 0.75) -> torch.Tensor:
    """Randomly set spatial latent tokens to 0, same mask for all channels."""
    if x.ndim != 4:
        raise ValueError(f"random_spatial_mask expects [B,C,H,W], got {tuple(x.shape)}")
    b, c, h, w = x.shape
    mask = torch.rand(b, h, w, device=x.device) < mask_ratio
    return x.masked_fill(mask.unsqueeze(1), 0.0)


def cal_align_cos(fs: torch.Tensor, fs_tilde: torch.Tensor) -> torch.Tensor:
    """Return per-sample align values, shape [B]. Inputs [B,T,C]."""
    fs_tilde = F.normalize(fs_tilde, dim=-1)
    fs = F.normalize(fs, dim=-1)
    align = mean_flat((fs * fs_tilde).sum(dim=-1))
    return align


def cal_entropy_cos(fs: torch.Tensor) -> torch.Tensor:
    """Return per-sample entropy/cross-sample cosine values, shape [B].

    This intentionally follows the user's code:
      sim = bmm(fs.permute(1,0,2), fs.permute(1,2,0)).permute(1,2,0)
      diag over sample dimension is zeroed.
    """
    fs = F.normalize(fs, dim=-1)
    # [T,B,C] @ [T,C,B] -> [T,B,B] -> [B,B,T]
    sim = torch.bmm(fs.permute(1, 0, 2), fs.permute(1, 2, 0)).permute(1, 2, 0)
    idx = torch.arange(sim.size(0), device=sim.device)
    sim[idx, idx, :] = 0.0
    entropy = mean_flat(sim)
    return entropy


def z4_to_btc(z4: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] -> [B,T,C]."""
    b, c, h, w = z4.shape
    return z4.reshape(b, c, h * w).permute(0, 2, 1).contiguous()


class RunningStat:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, values: torch.Tensor):
        v = values.detach().float()
        finite = torch.isfinite(v)
        if finite.any():
            v = v[finite]
            self.count += int(v.numel())
            self.sum += float(v.sum().item())
            self.sumsq += float((v * v).sum().item())

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.sum / self.count

    def std(self) -> Optional[float]:
        if self.count <= 1:
            return None
        m = self.mean()
        var = max(0.0, (self.sumsq - self.count * m * m) / (self.count - 1))
        return math.sqrt(var)

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {"mean": self.mean(), "std": self.std(), "count": self.count}


# ------------------------- optional normalization -------------------------

@torch.no_grad()
def normalize_z4(z4: torch.Tensor, mode: str, eps: float = 1e-8) -> torch.Tensor:
    """Normalize latent before applying noise/mask metric.

    Default mode is none because the proposed metric formula uses raw encoded latents.
    per_sample_channel can be useful for ablations when latent scale differs strongly across VAEs.
    """
    if mode == "none":
        return z4.float()
    if mode == "per_sample_channel":
        mean = z4.float().mean(dim=(2, 3), keepdim=True)
        std = z4.float().std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
        return (z4.float() - mean) / std
    if mode == "per_sample_all":
        mean = z4.float().mean(dim=(1, 2, 3), keepdim=True)
        std = z4.float().std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(eps)
        return (z4.float() - mean) / std
    raise ValueError(f"Unknown normalize mode: {mode}")


# ------------------------- main -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-config", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-images", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--t-values", nargs="+", type=float, default=[0.1, 0.4, 0.6, 0.9])
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--token-grid", nargs=2, type=int, default=None)
    p.add_argument("--normalize-latent", type=str, default="none", choices=["none", "per_sample_channel", "per_sample_all"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--drop-last", action="store_true", help="Drop last incomplete batch. Makes entropy batch size fixed.")
    p.add_argument("--save-per-batch", action="store_true", help="Save one row per batch/t to per_batch_metrics.csv")
    p.add_argument("--no-labels", action="store_true", help="Do not pass class labels to vae.encode, even if available.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
        drop_last=args.drop_last,
    )

    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    stats: Dict[str, RunningStat] = {}
    for t in args.t_values:
        key = f"{t:g}"
        stats[f"align_{key}"] = RunningStat()
        stats[f"entropy_{key}"] = RunningStat()

    per_batch_rows = []
    latent_shape = None
    spatial_shape = None
    num_seen = 0
    num_batches = 0

    pbar = tqdm(loader, desc=f"mask-cos {Path(args.vae_config).name}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # ImageNetValDataset in this repo usually returns (image, label, name)
            if isinstance(batch, (tuple, list)):
                images = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            else:
                images = batch
                labels = None

            images = images.to(device, non_blocking=True)
            labels_dev = None
            if labels is not None and not args.no_labels:
                labels_dev = labels.to(device, non_blocking=True).long()

            z = encode_vae(vae, images, labels_dev)
            z4 = to_spatial_latent(z, token_grid=args.token_grid)
            z4 = normalize_z4(z4, args.normalize_latent)

            if latent_shape is None:
                latent_shape = list(z.shape[1:])
                spatial_shape = list(z4.shape[1:])
                print(f"latent_shape={latent_shape}, spatial_shape={spatial_shape}")

            b = z4.shape[0]
            num_seen += int(b)
            num_batches += 1

            for t in args.t_values:
                t_float = float(t)
                t_key = f"{t_float:g}"

                # Follow the proposed metric: fresh random mask/noise for each t.
                latent_plus = random_spatial_mask(z4, mask_ratio=args.mask_ratio)
                noise = torch.randn_like(z4)

                latent_t = t_float * z4 + (1.0 - t_float) * noise
                latent_pt = t_float * latent_plus + (1.0 - t_float) * noise

                fs = z4_to_btc(latent_t)
                fs_tilde = z4_to_btc(latent_pt)

                align_vals = cal_align_cos(fs, fs_tilde)
                entropy_vals = cal_entropy_cos(fs)

                stats[f"align_{t_key}"].update(align_vals)
                stats[f"entropy_{t_key}"].update(entropy_vals)

                if args.save_per_batch:
                    per_batch_rows.append({
                        "batch_idx": batch_idx,
                        "batch_size": int(b),
                        "t": t_float,
                        "align_mean": float(align_vals.mean().item()),
                        "entropy_mean": float(entropy_vals.mean().item()),
                        "align_std": float(align_vals.float().std(unbiased=False).item()) if align_vals.numel() > 1 else 0.0,
                        "entropy_std": float(entropy_vals.float().std(unbiased=False).item()) if entropy_vals.numel() > 1 else 0.0,
                    })

            del images, labels_dev, z, z4
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Flatten final metrics for easy table use.
    metrics = {}
    metric_details = {}
    for t in args.t_values:
        t_key = f"{float(t):g}"
        for prefix in ["align", "entropy"]:
            name = f"{prefix}_{t_key}"
            d = stats[name].to_dict()
            metrics[name] = d["mean"]
            metrics[f"{name}_std"] = d["std"]
            metric_details[name] = d

    result = {
        "vae_config": args.vae_config,
        "dataset": args.dataset,
        "num_images_requested": int(args.num_images),
        "num_images_used": int(num_seen),
        "num_batches": int(num_batches),
        "batch_size": int(args.batch_size),
        "drop_last": bool(args.drop_last),
        "resolution": int(args.resolution),
        "t_values": [float(x) for x in args.t_values],
        "mask_ratio": float(args.mask_ratio),
        "normalize_latent": args.normalize_latent,
        "seed": int(args.seed),
        "latent_shape": latent_shape,
        "spatial_shape": spatial_shape,
        "token_grid": args.token_grid,
        "metrics": metrics,
        "metric_details": metric_details,
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved metrics: {metrics_path}")

    if args.save_per_batch:
        csv_path = out_dir / "per_batch_metrics.csv"
        with csv_path.open("w", newline="") as f:
            fields = ["batch_idx", "batch_size", "t", "align_mean", "entropy_mean", "align_std", "entropy_std"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_batch_rows)
        print(f"Saved per-batch metrics: {csv_path}")

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
