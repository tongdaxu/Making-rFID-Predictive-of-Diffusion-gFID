#!/usr/bin/env python3
"""
One-step diffusion-aware reconstruction FID.

For each ImageNet validation image x:
    z0_raw = VAE.encode(x)
    z0     = normalize(z0_raw) using the paired SiT EMA BN statistics
    z_t    = alpha_t * z0 + sigma_t * eps
    v_hat  = SiT(z_t, t, y)                      # exactly one diffusion forward
    z0_hat = (dot_sigma_t * z_t - sigma_t * v_hat)
             / (alpha_t * dot_sigma_t - sigma_t * dot_alpha_t)
    x_hat  = VAE.decode(denormalize(z0_hat))

Then compute FID(source images, x_hat) for each requested t.

Important:
- t-values are canonical/pre-shift timesteps. We apply model.shift_time(t), matching
  the benchmark SiT time-shift convention.
- This evaluator assumes the paired SiT checkpoint uses velocity ("v") prediction.
- No sampler / reverse trajectory / CFG is used.
- The same noise realization is reused for all t values within each image batch.
"""

import argparse
import gc
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from dictdot import dictdot
from omegaconf import OmegaConf
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.fid.fid import InceptionV3, calculate_frechet_distance
from ifid.sit.sit import SiT_models
from ifid.vae.utils import instantiate_from_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-path", type=str, required=True,
                   help="Paired SiT experiment directory containing args.json and checkpoints/.")
    p.add_argument("--train-steps", type=int, required=True)
    p.add_argument("--dataset", type=str, default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--num-images", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=25, help="Per-GPU batch size.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--t-values", type=float, nargs="+", default=[0.1, 0.4, 0.8])
    p.add_argument("--path-type", type=str, default=None, choices=["linear", "cosine"],
                   help="Default: read from args.json (fallback: linear).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument("--out-dir", type=str,
                   default="/share/xutongda/REPA-E/exps_interpolate/one_step_rfid")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-rfid", action=argparse.BooleanOptionalAction, default=True,
                   help="Also compute ordinary VAE rFID as a sanity check.")
    p.add_argument("--keep-features", action="store_true")
    return p.parse_args()


def unwrap_encode(vae, img, y):
    try:
        z = vae.encode(img, y)
    except TypeError:
        z = vae.encode(img)
    if isinstance(z, (tuple, list)):
        z = z[0]
    if not torch.is_tensor(z):
        if hasattr(z, "latent_dist"):
            z = z.latent_dist.sample()
        elif hasattr(z, "sample") and torch.is_tensor(z.sample):
            z = z.sample
        else:
            raise TypeError(f"Unsupported VAE encode output type: {type(z)}")
    return z


def unwrap_decode(vae, z):
    out = vae.decode(z)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        if hasattr(out, "sample") and torch.is_tensor(out.sample):
            out = out.sample
        else:
            raise TypeError(f"Unsupported VAE decode output type: {type(out)}")
    return out


def inception_features(inception, x01):
    pred = inception(x01)[0]
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    return pred.squeeze(-1).squeeze(-1)


def broadcast_shape(t_flat, z):
    return t_flat.view(t_flat.shape[0], *([1] * (z.ndim - 1)))


@torch.no_grad()
def predict_clean_from_velocity(model, z0_norm, y, eps, t_raw, path_type):
    """
    General stochastic-interpolant inversion:
        z_t = alpha z0 + sigma eps
        v   = d_alpha z0 + d_sigma eps
    =>  z0 = (d_sigma z_t - sigma v) / (alpha d_sigma - sigma d_alpha)
    """
    b = z0_norm.shape[0]
    t0 = torch.full((b,), float(t_raw), device=z0_norm.device, dtype=z0_norm.dtype)
    t_eff = model.shift_time(t0)
    t_bc = broadcast_shape(t_eff, z0_norm)

    alpha, sigma, d_alpha, d_sigma = model.interpolant(t_bc, path_type=path_type)
    zt = alpha * z0_norm + sigma * eps
    v_hat = model.inference(zt, t_eff, y)

    det = alpha * d_sigma - sigma * d_alpha
    if torch.any(det.abs() < 1e-8):
        raise RuntimeError(f"Degenerate interpolant determinant at t={t_raw}")

    z0_hat_norm = (d_sigma * zt - sigma * v_hat) / det
    return z0_hat_norm


def fid_from_features(src, pred):
    mu_src = src.mean(axis=0)
    cov_src = np.cov(src, rowvar=False)
    mu_pred = pred.mean(axis=0)
    cov_pred = np.cov(pred, rowvar=False)
    return float(calculate_frechet_distance(mu_src, cov_src, mu_pred, cov_pred))


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available()

    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device)

    if args.num_images % world_size != 0:
        raise ValueError(
            f"--num-images={args.num_images} must be divisible by world_size={world_size} "
            "to avoid DistributedSampler padding/duplicates."
        )

    torch.manual_seed(args.seed + rank)

    with open(os.path.join(args.exp_path, "args.json"), "r") as f:
        config = dictdot(json.load(f))

    configured_prediction = getattr(config, "prediction", "v")
    if configured_prediction != "v":
        raise ValueError(
            f"This evaluator derives clean latents from velocity prediction, "
            f"but args.json says prediction={configured_prediction!r}."
        )

    path_type = args.path_type or getattr(config, "path_type", "linear")

    # VAE
    vae_config = OmegaConf.load(config.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    resolution = int(getattr(config, "resolution", 256))
    fake_in = torch.zeros((1, 3, resolution, resolution), device=device)
    fake_y = torch.zeros((1,), dtype=torch.long, device=device)
    fake_z_batch = unwrap_encode(vae, fake_in, fake_y)
    fake_z = fake_z_batch[0]

    if fake_z.ndim == 3:
        latent_size = fake_z.shape[-1]
        in_channels = fake_z.shape[0]
        vae_1d = False
    elif fake_z.ndim == 2:
        latent_size = fake_z.shape[0]
        in_channels = fake_z.shape[-1]
        vae_1d = True
    else:
        raise ValueError(f"Unexpected latent shape: {tuple(fake_z.shape)}")

    tshift = math.sqrt(float(fake_z.numel()) / 4096.0)

    # paired SiT
    block_kwargs = {
        "fused_attn": getattr(config, "fused_attn", True),
        "qk_norm": getattr(config, "qk_norm", False),
    }
    model_name = config.model
    if model_name not in SiT_models:
        alt = model_name.replace("SiT_", "SiT-")
        if alt in SiT_models:
            model_name = alt
        else:
            raise KeyError(f"Unknown model name {config.model}; available={list(SiT_models)}")

    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=config.num_classes,
        class_dropout_prob=config.cfg_prob,
        bn_momentum=config.bn_momentum,
        tshift=tshift,
        **block_kwargs,
    ).to(device)

    ckpt_path = os.path.join(args.exp_path, "checkpoints", f"{int(args.train_steps):07d}.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict["ema"], strict=False)
    model.eval()

    if vae_1d:
        latents_scale = state_dict["ema"]["bn.running_var"].rsqrt().view(1, 1, in_channels).to(device)
        latents_bias = state_dict["ema"]["bn.running_mean"].view(1, 1, in_channels).to(device)
    else:
        latents_scale = state_dict["ema"]["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
        latents_bias = state_dict["ema"]["bn.running_mean"].view(1, in_channels, 1, 1).to(device)

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    # Inception
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    # Data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = ImageNetValDataset(args.dataset, transform, args.num_images)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    exp_name = os.path.basename(os.path.abspath(args.exp_path))
    run_name = f"{exp_name}_step{int(args.train_steps):07d}_{path_type}_t" + "-".join(f"{t:g}" for t in args.t_values)
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"exp_path       : {args.exp_path}", flush=True)
        print(f"checkpoint     : {ckpt_path}", flush=True)
        print(f"model          : {model_name}", flush=True)
        print(f"latent shape   : {tuple(fake_z.shape)}", flush=True)
        print(f"tshift         : {tshift:.6f}", flush=True)
        print(f"path_type      : {path_type}", flush=True)
        print("prediction     : v", flush=True)
        print(f"t raw          : {args.t_values}", flush=True)
        print(f"num_images     : {args.num_images}", flush=True)
        print(f"world_size     : {world_size}", flush=True)
        print(f"batch/GPU      : {args.batch_size}", flush=True)
        print(f"out_dir        : {out_dir}", flush=True)

    src_feats = []
    rec_feats = [] if args.include_rfid else None
    pred_feats = {float(t): [] for t in args.t_values}

    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(args.noise_seed + rank)

    pbar = tqdm(loader, disable=(rank != 0), desc="one-step rFID")
    with torch.no_grad():
        for img, y, _name in pbar:
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            z_raw = unwrap_encode(vae, img, y)
            # Match the SiT training forward exactly: normalize clean VAE latents
            # with the model BN before constructing z_t.
            if vae_1d:
                z0_norm = model.bn(z_raw.transpose(1, 2)).transpose(1, 2)
            else:
                z0_norm = model.bn(z_raw)

            eps = torch.randn(z0_norm.shape, dtype=z0_norm.dtype, device=device, generator=noise_gen)

            x01 = (img + 1.0) / 2.0
            src_feats.append(inception_features(inception, x01).float().cpu().numpy())

            if args.include_rfid:
                x_rec = (unwrap_decode(vae, z_raw) + 1.0) / 2.0
                rec_feats.append(inception_features(inception, x_rec).float().cpu().numpy())

            for t in args.t_values:
                z0_hat_norm = predict_clean_from_velocity(
                    model=model,
                    z0_norm=z0_norm,
                    y=y,
                    eps=eps,
                    t_raw=float(t),
                    path_type=path_type,
                )
                z0_hat_raw = z0_hat_norm / latents_scale + latents_bias
                x_hat = (unwrap_decode(vae, z0_hat_raw) + 1.0) / 2.0
                pred_feats[float(t)].append(inception_features(inception, x_hat).float().cpu().numpy())

    np.save(out_dir / f"src_rank{rank:03d}.npy", np.concatenate(src_feats, axis=0))
    if args.include_rfid:
        np.save(out_dir / f"rec_rank{rank:03d}.npy", np.concatenate(rec_feats, axis=0))
    for t in args.t_values:
        np.save(out_dir / f"t{float(t):g}_rank{rank:03d}.npy", np.concatenate(pred_feats[float(t)], axis=0))

    dist.barrier()

    if rank == 0:
        src_all = np.concatenate([np.load(out_dir / f"src_rank{r:03d}.npy") for r in range(world_size)], axis=0)
        if src_all.shape[0] != args.num_images:
            raise RuntimeError(f"Expected {args.num_images} source features, got {src_all.shape[0]}")

        result = {
            "exp_path": args.exp_path,
            "train_steps": int(args.train_steps),
            "model": model_name,
            "vae_config": str(config.vae_config),
            "num_images": int(args.num_images),
            "world_size": int(world_size),
            "batch_size_per_gpu": int(args.batch_size),
            "path_type": path_type,
            "prediction": "v",
            "tshift": float(tshift),
            "t_values_raw": [float(t) for t in args.t_values],
            "noise_seed": int(args.noise_seed),
            "metrics": {},
        }

        if args.include_rfid:
            rec_all = np.concatenate([np.load(out_dir / f"rec_rank{r:03d}.npy") for r in range(world_size)], axis=0)
            result["metrics"]["rfid"] = fid_from_features(src_all, rec_all)

        for t in args.t_values:
            pred_all = np.concatenate([
                np.load(out_dir / f"t{float(t):g}_rank{r:03d}.npy") for r in range(world_size)
            ], axis=0)
            result["metrics"][f"one_step_rfid_t{float(t):g}"] = fid_from_features(src_all, pred_all)

        out_json = out_dir / "metrics.json"
        out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(json.dumps(result["metrics"], indent=2, sort_keys=True), flush=True)
        print(f"Saved: {out_json}", flush=True)

        if not args.keep_features:
            for r in range(world_size):
                (out_dir / f"src_rank{r:03d}.npy").unlink(missing_ok=True)
                if args.include_rfid:
                    (out_dir / f"rec_rank{r:03d}.npy").unlink(missing_ok=True)
                for t in args.t_values:
                    (out_dir / f"t{float(t):g}_rank{r:03d}.npy").unlink(missing_ok=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
