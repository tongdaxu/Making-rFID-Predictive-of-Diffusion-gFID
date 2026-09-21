import os
import argparse
import json
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--sample-dir", type=str, default="./stats")
    parser.add_argument("--image-size", type=int, default=256)

    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=-1)

    parser.add_argument(
        "--stats-mode",
        type=str,
        default="token",
        choices=["token", "channel"],
        help=(
            "token: save mean/std with shape [1,T,D] for [B,T,D] latents. "
            "channel: save mean/std with shape [1,1,D]."
        ),
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
    )

    parser.add_argument("--tf32", action="store_true")

    return parser.parse_args()


def get_dtype(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def disable_latent_norm_if_exists(model):
    """
    统计 stats 时必须统计 raw latent，不能用已经 normalize 过的 latent。
    """
    if hasattr(model, "normalize_latents"):
        model.normalize_latents = False

    if hasattr(model, "latents_mean"):
        model.latents_mean = None

    if hasattr(model, "latents_std"):
        model.latents_std = None


def canonicalize_latent(z: torch.Tensor) -> torch.Tensor:
    """
    Return z as either:
      [B, T, D] or [B, C, H, W]

    Do not flatten by default. Different stats modes handle it later.
    """
    if isinstance(z, (tuple, list)):
        z = z[0]

    if not torch.is_tensor(z):
        raise TypeError(f"vae.encode should return Tensor, got {type(z)}")

    if z.ndim not in [3, 4]:
        raise RuntimeError(f"Unsupported latent shape: {tuple(z.shape)}")

    return z


def update_stats_for_3d(
    z: torch.Tensor,
    stats_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    z: [B, T, D]
    """
    z = z.to(torch.float64)

    if stats_mode == "token":
        # mean/std per token and channel: [T, D]
        cur_sum = z.sum(dim=0)
        cur_sumsq = (z * z).sum(dim=0)
        cur_count = torch.tensor(z.shape[0], device=z.device, dtype=torch.float64)

    elif stats_mode == "channel":
        # mean/std per channel: [D]
        z_flat = z.reshape(-1, z.shape[-1])
        cur_sum = z_flat.sum(dim=0)
        cur_sumsq = (z_flat * z_flat).sum(dim=0)
        cur_count = torch.tensor(z_flat.shape[0], device=z.device, dtype=torch.float64)

    else:
        raise ValueError(stats_mode)

    return cur_sum, cur_sumsq, cur_count


def update_stats_for_4d(
    z: torch.Tensor,
    stats_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    z: [B, C, H, W]
    """
    z = z.to(torch.float64)

    if stats_mode == "token":
        # RAE-style spatial stats: [C, H, W]
        cur_sum = z.sum(dim=0)
        cur_sumsq = (z * z).sum(dim=0)
        cur_count = torch.tensor(z.shape[0], device=z.device, dtype=torch.float64)

    elif stats_mode == "channel":
        # per-channel stats: [C]
        cur_sum = z.sum(dim=(0, 2, 3))
        cur_sumsq = (z * z).sum(dim=(0, 2, 3))
        cur_count = torch.tensor(
            z.shape[0] * z.shape[2] * z.shape[3],
            device=z.device,
            dtype=torch.float64,
        )

    else:
        raise ValueError(stats_mode)

    return cur_sum, cur_sumsq, cur_count


def main(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator()
    device = accelerator.device
    set_seed(args.seed + accelerator.process_index)

    dtype = get_dtype(args.precision)

    if accelerator.is_main_process:
        os.makedirs(args.sample_dir, exist_ok=True)
        print("============================================================")
        print("[calculate_stat.py]")
        print("config:", args.config)
        print("data_path:", args.data_path)
        print("sample_dir:", args.sample_dir)
        print("image_size:", args.image_size)
        print("batch_size:", args.batch_size)
        print("max_samples:", args.max_samples)
        print("stats_mode:", args.stats_mode)
        print("precision:", args.precision)
        print("============================================================")

    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ]
    )

    dataset = ImageNetValDataset(
        args.data_path,
        transform,
        args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    cfg = OmegaConf.load(args.config)
    vae = instantiate_from_config(cfg).to(device).eval()
    disable_latent_norm_if_exists(vae)

    for _, p in vae.named_parameters():
        p.requires_grad_(False)

    loader, vae = accelerator.prepare(loader, vae)

    local_sum = None
    local_sumsq = None
    local_count = torch.tensor(0.0, device=device, dtype=torch.float64)

    latent_shape = None
    raw_vae = accelerator.unwrap_model(vae)

    for img, y, name in tqdm(loader, disable=not accelerator.is_local_main_process):
        img = img.to(device, non_blocking=True)

        with torch.no_grad():
            if dtype != torch.float32:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    z = raw_vae.encode(img)
            else:
                z = raw_vae.encode(img)

        z = canonicalize_latent(z)

        if latent_shape is None:
            latent_shape = tuple(z.shape[1:])
            if accelerator.is_main_process:
                print("[DEBUG] latent shape:", latent_shape)

        if z.ndim == 3:
            cur_sum, cur_sumsq, cur_count = update_stats_for_3d(
                z,
                args.stats_mode,
            )
        elif z.ndim == 4:
            cur_sum, cur_sumsq, cur_count = update_stats_for_4d(
                z,
                args.stats_mode,
            )
        else:
            raise RuntimeError(f"Unsupported latent shape: {tuple(z.shape)}")

        if local_sum is None:
            local_sum = torch.zeros_like(cur_sum, dtype=torch.float64, device=device)
            local_sumsq = torch.zeros_like(cur_sumsq, dtype=torch.float64, device=device)

        local_sum += cur_sum
        local_sumsq += cur_sumsq
        local_count += cur_count

    if local_sum is None:
        raise RuntimeError("No samples were processed. Check --data-path and --max-samples.")

    global_sum = accelerator.reduce(local_sum, reduction="sum")
    global_sumsq = accelerator.reduce(local_sumsq, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum")

    if accelerator.is_main_process:
        mean = global_sum / global_count
        var = global_sumsq / global_count - mean * mean
        var = torch.clamp(var, min=1e-12)
        std = torch.sqrt(var)

        # Save broadcast-friendly shapes.
        if len(latent_shape) == 2:
            # z: [B, T, D]
            if args.stats_mode == "token":
                # [T,D] -> [1,T,D]
                mean_save = mean.float().unsqueeze(0).cpu()
                std_save = std.float().unsqueeze(0).cpu()
            else:
                # [D] -> [1,1,D]
                mean_save = mean.float().view(1, 1, -1).cpu()
                std_save = std.float().view(1, 1, -1).cpu()

        elif len(latent_shape) == 3:
            # z: [B, C, H, W]
            if args.stats_mode == "token":
                # [C,H,W] -> [1,C,H,W]
                mean_save = mean.float().unsqueeze(0).cpu()
                std_save = std.float().unsqueeze(0).cpu()
            else:
                # [C] -> [1,C,1,1]
                mean_save = mean.float().view(1, -1, 1, 1).cpu()
                std_save = std.float().view(1, -1, 1, 1).cpu()

        else:
            raise RuntimeError(f"Unsupported latent_shape: {latent_shape}")

        stat_path = os.path.join(args.sample_dir, "stat.pt")

        payload = {
            "mean": mean_save,
            "std": std_save,
            "var": (std_save * std_save),
            "count": int(global_count.item()),
            "latent_shape": latent_shape,
            "stats_mode": args.stats_mode,
            "config": args.config,
            "data_path": args.data_path,
            "image_size": args.image_size,
            "max_samples": args.max_samples,
            "precision": args.precision,
        }

        torch.save(payload, stat_path)

        meta_path = os.path.join(args.sample_dir, "stat_meta.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "stat_path": stat_path,
                    "count": payload["count"],
                    "latent_shape": list(latent_shape),
                    "mean_shape": list(mean_save.shape),
                    "std_shape": list(std_save.shape),
                    "stats_mode": args.stats_mode,
                    "config": args.config,
                    "data_path": args.data_path,
                    "image_size": args.image_size,
                    "max_samples": args.max_samples,
                    "precision": args.precision,
                },
                f,
                indent=2,
            )

        print("============================================================")
        print("[DONE]")
        print("saved:", stat_path)
        print("meta:", meta_path)
        print("count:", payload["count"])
        print("latent_shape:", latent_shape)
        print("mean shape:", tuple(mean_save.shape))
        print("std shape:", tuple(std_save.shape))
        print("mean abs avg:", mean_save.abs().mean().item())
        print("std avg:", std_save.mean().item())
        print("std min:", std_save.min().item())
        print("std max:", std_save.max().item())
        print("============================================================")


if __name__ == "__main__":
    args = parse_args()
    main(args)