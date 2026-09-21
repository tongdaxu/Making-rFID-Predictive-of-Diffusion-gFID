#!/usr/bin/env python3
"""
Safe one-step diffusion-aware reconstruction FID for the legacy IFID SiT baselines.

Scope (intentional):
  - legacy IFID checkpoints trained with external prediction='v'
  - NO prediction_internal branch in args.json
  - latent normalization via the SiT EMA BatchNorm running statistics
  - one diffusion-network forward per requested t; no sampler / reverse trajectory / CFG

For each ImageNet validation image x:
  z_raw   = VAE.encode(x)
  z0      = SiT-BN(z_raw)
  z_t     = alpha(t_eff) * z0 + sigma(t_eff) * eps
  v_hat   = SiT.inference(z_t, t_eff, y)
  z0_hat  = invert stochastic interpolant from (z_t, v_hat)
  z_raw^  = legacy IFID denormalization using EMA BN stats
  x_hat   = VAE.decode(z_raw^)

Then compute FID(source images, x_hat) for each requested t.

Two timestep conventions are supported:
  shifted_raw     : t_eff = model.shift_time(t_raw), matching the legacy IFID
                    training/sampling time-shift convention.
  fixed_effective : t_eff = t_raw, so the requested t values are the actual
                    interpolant/noise coefficients (for linear path).

The default is shifted_raw so it is directly comparable to the first XL run.
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
    p.add_argument("--exp-path", type=str, required=True)
    p.add_argument("--train-steps", type=int, required=True)
    p.add_argument("--dataset", type=str, default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--num-images", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--t-values", type=float, nargs="+", default=[0.1, 0.4, 0.8])
    p.add_argument(
        "--time-mode",
        type=str,
        default="shifted_raw",
        choices=["shifted_raw", "fixed_effective"],
        help=(
            "shifted_raw: t_eff=model.shift_time(t); "
            "fixed_effective: t_eff=t"
        ),
    )
    p.add_argument(
        "--path-type",
        type=str,
        default=None,
        choices=["linear", "cosine"],
        help="Default: use args.json. If supplied, it must match args.json.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument(
        "--out-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/one_step_rfid_safe",
    )
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--include-rfid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute ordinary VAE rFID as a control.",
    )
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


def legacy_denormalize(z_norm, latents_scale, latents_bias):
    # Match the existing IFID generation code: z_raw = z_norm / scale + bias,
    # with scale = running_var.rsqrt().
    return z_norm / latents_scale + latents_bias


@torch.no_grad()
def predict_clean_from_velocity(
    model,
    z0_norm,
    y,
    eps,
    t_value,
    path_type,
    time_mode,
):
    b = z0_norm.shape[0]
    t_requested = torch.full(
        (b,), float(t_value), device=z0_norm.device, dtype=z0_norm.dtype
    )
    if time_mode == "shifted_raw":
        t_eff = model.shift_time(t_requested)
    elif time_mode == "fixed_effective":
        t_eff = t_requested
    else:
        raise ValueError(time_mode)

    t_bc = broadcast_shape(t_eff, z0_norm)
    alpha, sigma, d_alpha, d_sigma = model.interpolant(t_bc, path_type=path_type)
    zt = alpha * z0_norm + sigma * eps

    # Exactly one diffusion network forward. No sampler.
    v_hat = model.inference(zt, t_eff, y)

    # General inversion of:
    #   z_t = alpha z0 + sigma eps
    #   v   = d_alpha z0 + d_sigma eps
    det = alpha * d_sigma - sigma * d_alpha
    if torch.any(det.abs() < 1e-8):
        raise RuntimeError(f"Degenerate interpolant determinant at t={t_value}")
    z0_hat_norm = (d_sigma * zt - sigma * v_hat) / det
    return z0_hat_norm


def fid_from_features(src, pred):
    mu_src = src.mean(axis=0)
    cov_src = np.cov(src, rowvar=False)
    mu_pred = pred.mean(axis=0)
    cov_pred = np.cov(pred, rowvar=False)
    return float(calculate_frechet_distance(mu_src, cov_src, mu_pred, cov_pred))


def effective_t_value(t, tshift, time_mode):
    t = float(t)
    if time_mode == "fixed_effective":
        return t
    return float(tshift * t / (1.0 + (tshift - 1.0) * t))


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
            f"--num-images={args.num_images} must be divisible by world_size={world_size}"
        )

    torch.manual_seed(args.seed + rank)

    args_json = os.path.join(args.exp_path, "args.json")
    with open(args_json, "r") as f:
        raw_config = json.load(f)
    config = dictdot(raw_config)

    # ------------------------------------------------------------------
    # Hard safety checks: this evaluator is deliberately scoped to the
    # legacy IFID pure-v + BN baseline family. Do not silently mix the
    # newer prediction_internal / per-sample-normalization branch.
    # ------------------------------------------------------------------
    configured_prediction = str(raw_config.get("prediction", "v")).lower()
    if configured_prediction != "v":
        raise ValueError(
            f"Unsupported external prediction={configured_prediction!r}; expected legacy v"
        )

    if "prediction_internal" in raw_config:
        raise ValueError(
            "args.json contains prediction_internal="
            f"{raw_config.get('prediction_internal')!r}. This evaluator intentionally "
            "refuses the newer internal-prediction branch because its prediction and "
            "normalization semantics must be handled separately."
        )

    configured_path = str(raw_config.get("path_type", "linear")).lower()
    if args.path_type is not None and args.path_type != configured_path:
        raise ValueError(
            f"Requested --path-type={args.path_type}, but args.json path_type={configured_path}"
        )
    path_type = configured_path if args.path_type is None else args.path_type

    # VAE
    vae_config = OmegaConf.load(config.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    resolution = int(raw_config.get("resolution", 256))
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

    # Paired SiT
    block_kwargs = {
        "fused_attn": raw_config.get("fused_attn", True),
        "qk_norm": raw_config.get("qk_norm", False),
    }
    model_name = str(config.model)
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

    # Catch a silent 1D/2D architecture mismatch before loading weights.
    model_vae_1d = bool(getattr(model, "vae_1d", False))
    if model_vae_1d != vae_1d:
        raise RuntimeError(
            f"VAE latent says vae_1d={vae_1d}, but model {model_name} has "
            f"vae_1d={model_vae_1d}. Check args.json model name / 1D factory."
        )

    ckpt_path = os.path.join(
        args.exp_path, "checkpoints", f"{int(args.train_steps):07d}.pt"
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    if "ema" not in state_dict:
        raise KeyError(f"Checkpoint has no 'ema': {ckpt_path}")
    ema_state = state_dict["ema"]

    incompat = model.load_state_dict(ema_state, strict=False)
    missing_keys = list(incompat.missing_keys)
    unexpected_keys = list(incompat.unexpected_keys)
    # Missing inference keys are dangerous; fail. Unexpected extras (e.g. training-only
    # projector heads) do not alter the loaded inference network, so report but allow.
    if missing_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch: missing keys while loading inference model:\n  "
            + "\n  ".join(missing_keys[:100])
        )
    model.eval()

    required_bn = ["bn.running_mean", "bn.running_var"]
    for k in required_bn:
        if k not in ema_state:
            raise KeyError(f"Legacy BN checkpoint expected key {k!r}, but it is absent")

    if vae_1d:
        latents_scale = ema_state["bn.running_var"].rsqrt().view(1, 1, in_channels).to(device)
        latents_bias = ema_state["bn.running_mean"].view(1, 1, in_channels).to(device)
    else:
        latents_scale = ema_state["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
        latents_bias = ema_state["bn.running_mean"].view(1, in_channels, 1, 1).to(device)

    del state_dict, ema_state
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
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
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
    t_tag = "-".join(f"{t:g}" for t in args.t_values)
    run_name = (
        f"{exp_name}_step{int(args.train_steps):07d}_"
        f"{path_type}_{args.time_mode}_t{t_tag}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t_eff_map = {
        f"{float(t):g}": effective_t_value(t, tshift, args.time_mode)
        for t in args.t_values
    }

    if rank == 0:
        print(f"exp_path              : {args.exp_path}", flush=True)
        print(f"checkpoint            : {ckpt_path}", flush=True)
        print(f"model                 : {model_name}", flush=True)
        print(f"latent shape          : {tuple(fake_z.shape)}", flush=True)
        print(f"vae_1d                : {vae_1d}", flush=True)
        print(f"tshift                : {tshift:.6f}", flush=True)
        print(f"time_mode             : {args.time_mode}", flush=True)
        print(f"t requested           : {args.t_values}", flush=True)
        print(f"t effective           : {t_eff_map}", flush=True)
        print(f"path_type             : {path_type}", flush=True)
        print("prediction            : legacy v", flush=True)
        print("prediction_internal   : absent (required)", flush=True)
        print(f"unexpected ckpt keys  : {len(unexpected_keys)}", flush=True)
        if unexpected_keys:
            print(f"unexpected first 20   : {unexpected_keys[:20]}", flush=True)
        print(f"num_images            : {args.num_images}", flush=True)
        print(f"world_size            : {world_size}", flush=True)
        print(f"batch/GPU             : {args.batch_size}", flush=True)
        print(f"out_dir               : {out_dir}", flush=True)

    src_feats = []
    rec_feats = [] if args.include_rfid else None
    pred_feats = {float(t): [] for t in args.t_values}

    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(args.noise_seed + rank)

    roundtrip_rel_mse = None
    roundtrip_max_abs = None

    pbar = tqdm(loader, disable=(rank != 0), desc="one-step rFID")
    with torch.no_grad():
        for batch_idx, (img, y, _name) in enumerate(pbar):
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            z_raw = unwrap_encode(vae, img, y)
            if vae_1d:
                z0_norm = model.bn(z_raw.transpose(1, 2)).transpose(1, 2)
            else:
                z0_norm = model.bn(z_raw)

            if batch_idx == 0:
                z_roundtrip = legacy_denormalize(z0_norm, latents_scale, latents_bias)
                mse = torch.mean((z_roundtrip.float() - z_raw.float()) ** 2)
                denom = torch.mean(z_raw.float() ** 2).clamp_min(1e-12)
                roundtrip_rel_mse = float((mse / denom).item())
                roundtrip_max_abs = float(
                    (z_roundtrip.float() - z_raw.float()).abs().max().item()
                )
                if rank == 0:
                    print(
                        f"BN roundtrip sanity    : rel_mse={roundtrip_rel_mse:.3e}, "
                        f"max_abs={roundtrip_max_abs:.3e}",
                        flush=True,
                    )

            eps = torch.randn(
                z0_norm.shape,
                dtype=z0_norm.dtype,
                device=device,
                generator=noise_gen,
            )

            x01 = (img + 1.0) / 2.0
            src_feats.append(
                inception_features(inception, x01).float().cpu().numpy()
            )

            if args.include_rfid:
                x_rec = (unwrap_decode(vae, z_raw) + 1.0) / 2.0
                rec_feats.append(
                    inception_features(inception, x_rec).float().cpu().numpy()
                )

            for t in args.t_values:
                z0_hat_norm = predict_clean_from_velocity(
                    model=model,
                    z0_norm=z0_norm,
                    y=y,
                    eps=eps,
                    t_value=float(t),
                    path_type=path_type,
                    time_mode=args.time_mode,
                )
                z0_hat_raw = legacy_denormalize(
                    z0_hat_norm, latents_scale, latents_bias
                )
                x_hat = (unwrap_decode(vae, z0_hat_raw) + 1.0) / 2.0
                pred_feats[float(t)].append(
                    inception_features(inception, x_hat).float().cpu().numpy()
                )

    np.save(out_dir / f"src_rank{rank:03d}.npy", np.concatenate(src_feats, axis=0))
    if args.include_rfid:
        np.save(out_dir / f"rec_rank{rank:03d}.npy", np.concatenate(rec_feats, axis=0))
    for t in args.t_values:
        np.save(
            out_dir / f"t{float(t):g}_rank{rank:03d}.npy",
            np.concatenate(pred_feats[float(t)], axis=0),
        )

    dist.barrier()

    if rank == 0:
        src_all = np.concatenate(
            [np.load(out_dir / f"src_rank{r:03d}.npy") for r in range(world_size)],
            axis=0,
        )
        if src_all.shape[0] != args.num_images:
            raise RuntimeError(
                f"Expected {args.num_images} source features, got {src_all.shape[0]}"
            )

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
            "prediction_internal": None,
            "normalization": "legacy_ifid_ema_batchnorm",
            "tshift": float(tshift),
            "time_mode": args.time_mode,
            "t_values_requested": [float(t) for t in args.t_values],
            "t_values_effective": t_eff_map,
            "noise_seed": int(args.noise_seed),
            "checkpoint_load": {
                "missing_keys": missing_keys,
                "unexpected_keys_count": len(unexpected_keys),
                "unexpected_keys_first_50": unexpected_keys[:50],
            },
            "normalization_roundtrip_rank0_first_batch": {
                "relative_mse": roundtrip_rel_mse,
                "max_abs": roundtrip_max_abs,
            },
            "metrics": {},
        }

        if args.include_rfid:
            rec_all = np.concatenate(
                [np.load(out_dir / f"rec_rank{r:03d}.npy") for r in range(world_size)],
                axis=0,
            )
            result["metrics"]["rfid"] = fid_from_features(src_all, rec_all)

        for t in args.t_values:
            pred_all = np.concatenate(
                [
                    np.load(out_dir / f"t{float(t):g}_rank{r:03d}.npy")
                    for r in range(world_size)
                ],
                axis=0,
            )
            key = f"one_step_rfid_t{float(t):g}"
            result["metrics"][key] = fid_from_features(src_all, pred_all)

        # Exploratory transforms only. FID is not additive, so these are not a
        # decomposition of diffusion error; they are merely alternative predictors.
        if "rfid" in result["metrics"]:
            base = result["metrics"]["rfid"]
            for t in args.t_values:
                key = f"one_step_rfid_t{float(t):g}"
                val = result["metrics"][key]
                result["metrics"][f"delta_rfid_t{float(t):g}"] = val - base
                result["metrics"][f"ratio_rfid_t{float(t):g}"] = (
                    val / base if base != 0 else None
                )

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
