#!/usr/bin/env python3
"""
Evaluate PSNR and LPIPS for the diffusion-aware reconstruction proxies.

Supported settings
------------------
  recon
      ordinary VAE reconstruction

  direct:<t>
      previous "one step rFID" construction:
        z0 -> z_t -> one SiT forward -> x0 prediction -> decode

  ode:<t>
      teacher's step-xpred construction:
        z0 -> z_t -> one reverse Euler ODE step from raw t to raw t/2
        -> SiT forward at midpoint -> x0 prediction -> decode

  sde:<t>
      same, but the reverse step is one Euler-Maruyama SDE step.

Example:
  --settings direct:0.1,direct:0.4,direct:0.8,ode:0.8,sde:0.8

Metric convention
-----------------
This intentionally matches the existing IFID VAE evaluation code:
  get_psnr(source_raw, pred_raw, zero_mean=True)
  get_lpips(source_raw, pred_raw, zero_mean=True, network_type="alex")

The tensors passed to PSNR/LPIPS stay in the VAE's raw [-1, 1] image scale.
They are NOT converted to [0,1] and are NOT uint8-quantized before these metrics.

Time convention
---------------
Default --time-mode shifted_raw:
  raw t is shifted with model.shift_time(t).
For ode/sde, raw midpoint = t/2 and BOTH raw grid points are shifted
independently, matching the repository sampler convention.

Noise convention
----------------
- one forward-noise realization is reused across every selected setting for
  the same image batch;
- one independent Brownian realization is reused across all selected SDE
  settings for the same image batch;
- default seeds match the prior one-step / step-xpred evaluators.

No CFG; true ImageNet class label y is used.
"""

import argparse
import gc
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from dictdot import dictdot
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.fid.psnr import get_psnr
from ifid.fid.lpips import get_lpips
from ifid.sit.sit import SiT_models
from ifid.sit.samplers import get_score_from_velocity, compute_diffusion
from ifid.vae.utils import instantiate_from_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-path", type=str, required=True)
    p.add_argument("--train-steps", type=int, default=400000)
    p.add_argument("--dataset", type=str, default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--num-images", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU batch size")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--settings",
        type=str,
        required=True,
        help=(
            "Comma-separated list, e.g. "
            "direct:0.1,direct:0.4,direct:0.8,ode:0.8,sde:0.8. "
            "Optional ordinary reconstruction baseline: recon"
        ),
    )
    p.add_argument("--mid-frac", type=float, default=0.5)
    p.add_argument(
        "--time-mode",
        type=str,
        default="shifted_raw",
        choices=["shifted_raw", "fixed_effective"],
    )
    p.add_argument(
        "--path-type",
        type=str,
        default=None,
        choices=["linear", "cosine"],
        help="Default: use args.json. If supplied, it must match args.json.",
    )
    p.add_argument("--lpips-network", type=str, default="alex")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forward-noise-seed", type=int, default=12345)
    p.add_argument("--sde-noise-seed", type=int, default=54321)
    p.add_argument(
        "--out-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/proxy_psnr_lpips",
    )
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def setting_key(kind, t):
    if kind == "recon":
        return "recon"
    return f"{kind}_t{float(t):g}"


def parse_settings(spec):
    items = []
    seen = set()
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw == "recon":
            item = ("recon", None)
        else:
            if ":" not in raw:
                raise ValueError(
                    f"Bad setting {raw!r}; expected recon or kind:t, e.g. ode:0.8"
                )
            kind, t_str = raw.split(":", 1)
            kind = kind.strip().lower()
            if kind not in {"direct", "ode", "sde"}:
                raise ValueError(f"Unsupported setting kind {kind!r}")
            t = float(t_str)
            if not (0.0 < t <= 1.0):
                raise ValueError(f"t must be in (0,1], got {t}")
            item = (kind, t)
        key = setting_key(*item)
        if key not in seen:
            items.append(item)
            seen.add(key)
    if not items:
        raise ValueError("No settings selected")
    return items


def settings_tag(settings):
    return "__".join(setting_key(k, t).replace(".", "p") for k, t in settings)


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
    # Keep exactly the same decode convention as the rFID proxy evaluators.
    out = vae.decode(z)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        if hasattr(out, "sample") and torch.is_tensor(out.sample):
            out = out.sample
        else:
            raise TypeError(f"Unsupported VAE decode output type: {type(out)}")
    return out


def broadcast_shape(t_flat, z):
    return t_flat.view(t_flat.shape[0], *([1] * (z.ndim - 1)))


def legacy_denormalize(z_norm, latents_scale, latents_bias):
    return z_norm / latents_scale + latents_bias


def resolve_time(model, raw_t, batch, device, dtype, time_mode):
    t = torch.full((batch,), float(raw_t), device=device, dtype=dtype)
    if time_mode == "shifted_raw":
        return model.shift_time(t)
    if time_mode == "fixed_effective":
        return t
    raise ValueError(time_mode)


@torch.no_grad()
def clean_from_state_velocity(model, state, v_hat, t_eff, path_type):
    t_bc = broadcast_shape(t_eff, state)
    alpha, sigma, d_alpha, d_sigma = model.interpolant(t_bc, path_type=path_type)
    det = alpha * d_sigma - sigma * d_alpha
    det_t = torch.as_tensor(det, device=state.device)
    if torch.any(det_t.abs() < 1e-8):
        raise RuntimeError("Degenerate interpolant determinant")
    return (d_sigma * state - sigma * v_hat) / det


@torch.no_grad()
def direct_xpred(model, z0_norm, y, forward_eps, t_raw, path_type, time_mode):
    b = z0_norm.shape[0]
    t_eff = resolve_time(model, t_raw, b, z0_norm.device, z0_norm.dtype, time_mode)
    t_bc = broadcast_shape(t_eff, z0_norm)
    alpha, sigma, _d_alpha, _d_sigma = model.interpolant(t_bc, path_type=path_type)
    z_t = alpha * z0_norm + sigma * forward_eps
    v_hat = model.inference(z_t, t_eff, y)
    return clean_from_state_velocity(model, z_t, v_hat, t_eff, path_type)


@torch.no_grad()
def step_then_xpred(
    model,
    z0_norm,
    y,
    forward_eps,
    sde_eps,
    t_start_raw,
    mid_frac,
    path_type,
    time_mode,
    step_mode,
):
    if step_mode not in {"ode", "sde"}:
        raise ValueError(step_mode)
    if not (0.0 < mid_frac < 1.0):
        raise ValueError(f"mid_frac must be in (0,1), got {mid_frac}")

    b = z0_norm.shape[0]
    t_mid_raw = float(t_start_raw) * float(mid_frac)
    t_start = resolve_time(model, t_start_raw, b, z0_norm.device, z0_norm.dtype, time_mode)
    t_mid = resolve_time(model, t_mid_raw, b, z0_norm.device, z0_norm.dtype, time_mode)

    ts_bc = broadcast_shape(t_start, z0_norm)
    alpha_s, sigma_s, _da, _ds = model.interpolant(ts_bc, path_type=path_type)
    z_start = alpha_s * z0_norm + sigma_s * forward_eps

    # NFE #1
    v_start = model.inference(z_start, t_start, y)
    dt = t_mid - t_start
    dt_bc = broadcast_shape(dt, z0_norm)

    if step_mode == "ode":
        z_mid = z_start + dt_bc * v_start
    else:
        score_start = get_score_from_velocity(
            v_start, z_start, t_start, path_type=path_type
        )
        diffusion = compute_diffusion(t_start)
        diffusion_bc = broadcast_shape(diffusion, z0_norm)
        drift = v_start - 0.5 * diffusion_bc * score_start
        dW = sde_eps * torch.sqrt(torch.abs(dt_bc))
        z_mid = (
            z_start
            + drift * dt_bc
            + torch.sqrt(diffusion_bc.clamp_min(0.0)) * dW
        )

    if not torch.isfinite(z_mid).all():
        raise FloatingPointError(
            f"Non-finite midpoint latent for {step_mode} t={t_start_raw}"
        )

    # NFE #2
    v_mid = model.inference(z_mid, t_mid, y)
    z0_hat = clean_from_state_velocity(model, z_mid, v_mid, t_mid, path_type)
    if not torch.isfinite(z0_hat).all():
        raise FloatingPointError(
            f"Non-finite x prediction for {step_mode} t={t_start_raw}"
        )
    return z0_hat


def scalar_shift(t, tshift):
    t = float(t)
    return float(tshift * t / (1.0 + (tshift - 1.0) * t))


def metric_time_description(settings, tshift, time_mode, mid_frac):
    out = {}
    for kind, t in settings:
        key = setting_key(kind, t)
        if kind == "recon":
            out[key] = {"kind": "recon"}
            continue
        t_eff = scalar_shift(t, tshift) if time_mode == "shifted_raw" else float(t)
        item = {"kind": kind, "t_raw": float(t), "t_effective": t_eff}
        if kind in {"ode", "sde"}:
            mid_raw = float(t) * float(mid_frac)
            mid_eff = scalar_shift(mid_raw, tshift) if time_mode == "shifted_raw" else mid_raw
            item["mid_raw"] = mid_raw
            item["mid_effective"] = mid_eff
        out[key] = item
    return out


def add_batch_metric(accum, key, source_raw, pred_raw, lpips_network):
    """Match old code: mean(metric per batch), then mean over batches."""
    psnr = get_psnr(source_raw, pred_raw, zero_mean=True)
    lpips = get_lpips(
        source_raw,
        pred_raw,
        zero_mean=True,
        network_type=lpips_network,
    )
    accum[key]["psnr_sum_batchmeans"] += torch.mean(psnr).double()
    accum[key]["lpips_sum_batchmeans"] += torch.mean(lpips).double()
    accum[key]["num_batches"] += 1.0


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    settings = parse_settings(args.settings)

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

    with open(os.path.join(args.exp_path, "args.json"), "r") as f:
        raw_config = json.load(f)
    config = dictdot(raw_config)

    prediction = str(raw_config.get("prediction", "v")).lower()
    if prediction != "v":
        raise ValueError(f"Expected legacy prediction=v, got {prediction}")
    if "prediction_internal" in raw_config:
        raise ValueError("Refusing newer prediction_internal checkpoint")

    configured_path = str(raw_config.get("path_type", "linear")).lower()
    if args.path_type is not None and args.path_type != configured_path:
        raise ValueError(
            f"--path-type={args.path_type} != args.json path_type={configured_path}"
        )
    path_type = configured_path if args.path_type is None else args.path_type

    vae_config = OmegaConf.load(config.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    resolution = int(raw_config.get("resolution", 256))
    fake_in = torch.zeros((1, 3, resolution, resolution), device=device)
    fake_y = torch.zeros((1,), dtype=torch.long, device=device)
    fake_z = unwrap_encode(vae, fake_in, fake_y)[0]

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
            raise KeyError(f"Unknown model name {config.model}")

    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=config.num_classes,
        class_dropout_prob=config.cfg_prob,
        bn_momentum=config.bn_momentum,
        tshift=tshift,
        **block_kwargs,
    ).to(device)

    model_vae_1d = bool(getattr(model, "vae_1d", False))
    if model_vae_1d != vae_1d:
        raise RuntimeError(
            f"VAE says vae_1d={vae_1d}, model {model_name} says {model_vae_1d}"
        )

    ckpt_path = os.path.join(
        args.exp_path, "checkpoints", f"{int(args.train_steps):07d}.pt"
    )
    state = torch.load(ckpt_path, map_location=device)
    if "ema" not in state:
        raise KeyError(f"Checkpoint has no ema: {ckpt_path}")
    ema_state = state["ema"]

    incompat = model.load_state_dict(ema_state, strict=False)
    if incompat.missing_keys:
        raise RuntimeError(
            "Missing checkpoint keys:\n" + "\n".join(incompat.missing_keys[:100])
        )
    unexpected_keys = list(incompat.unexpected_keys)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for k in ["bn.running_mean", "bn.running_var"]:
        if k not in ema_state:
            raise KeyError(f"Missing legacy BN key: {k}")

    if vae_1d:
        latents_scale = ema_state["bn.running_var"].rsqrt().view(1, 1, in_channels).to(device)
        latents_bias = ema_state["bn.running_mean"].view(1, 1, in_channels).to(device)
    else:
        latents_scale = ema_state["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
        latents_bias = ema_state["bn.running_mean"].view(1, in_channels, 1, 1).to(device)

    del state, ema_state
    gc.collect()
    torch.cuda.empty_cache()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = ImageNetValDataset(args.dataset, transform, args.num_images)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
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
    tag = settings_tag(settings)
    run_name = f"{exp_name}_step{int(args.train_steps):07d}_{args.time_mode}_{tag}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    time_desc = metric_time_description(settings, tshift, args.time_mode, args.mid_frac)

    if rank == 0:
        print(f"exp_path             : {args.exp_path}", flush=True)
        print(f"model                : {model_name}", flush=True)
        print(f"latent shape         : {tuple(fake_z.shape)}", flush=True)
        print(f"vae_1d               : {vae_1d}", flush=True)
        print(f"tshift               : {tshift:.6f}", flush=True)
        print(f"time_mode            : {args.time_mode}", flush=True)
        print(f"path_type            : {path_type}", flush=True)
        print(f"mid_frac             : {args.mid_frac}", flush=True)
        print(f"settings             : {[setting_key(*s) for s in settings]}", flush=True)
        print(f"time_description     : {json.dumps(time_desc)}", flush=True)
        print("PSNR convention      : raw [-1,1], zero_mean=True", flush=True)
        print(
            f"LPIPS convention     : raw [-1,1], zero_mean=True, net={args.lpips_network}",
            flush=True,
        )
        print("CFG                   : none; true class labels", flush=True)
        print(f"unexpected ckpt keys : {len(unexpected_keys)}", flush=True)
        print(f"num_images           : {args.num_images}", flush=True)
        print(f"world_size           : {world_size}", flush=True)
        print(f"batch/GPU            : {args.batch_size}", flush=True)
        print(f"out_dir              : {out_dir}", flush=True)

    accum = {}
    for kind, t in settings:
        key = setting_key(kind, t)
        accum[key] = {
            "psnr_sum_batchmeans": torch.zeros((), device=device, dtype=torch.float64),
            "lpips_sum_batchmeans": torch.zeros((), device=device, dtype=torch.float64),
            "num_batches": torch.zeros((), device=device, dtype=torch.float64),
        }

    fwd_gen = torch.Generator(device=device)
    fwd_gen.manual_seed(args.forward_noise_seed + rank)
    sde_gen = torch.Generator(device=device)
    sde_gen.manual_seed(args.sde_noise_seed + rank)

    pbar = tqdm(loader, disable=(rank != 0), desc="PSNR/LPIPS proxy sweep")

    with torch.no_grad():
        for img, y, _name in pbar:
            img = img.to(device, non_blocking=True)  # raw [-1,1]
            y = y.to(device, non_blocking=True)

            z_raw = unwrap_encode(vae, img, y)
            if vae_1d:
                z0_norm = model.bn(z_raw.transpose(1, 2)).transpose(1, 2)
            else:
                z0_norm = model.bn(z_raw)

            forward_eps = torch.randn(
                z0_norm.shape,
                dtype=z0_norm.dtype,
                device=device,
                generator=fwd_gen,
            )
            sde_eps = torch.randn(
                z0_norm.shape,
                dtype=z0_norm.dtype,
                device=device,
                generator=sde_gen,
            )

            if any(k == "recon" for k, _ in settings):
                recon_raw = unwrap_decode(vae, z_raw)
                add_batch_metric(accum, "recon", img, recon_raw, args.lpips_network)

            for kind, t in settings:
                if kind == "recon":
                    continue

                if kind == "direct":
                    z0_hat_norm = direct_xpred(
                        model=model,
                        z0_norm=z0_norm,
                        y=y,
                        forward_eps=forward_eps,
                        t_raw=float(t),
                        path_type=path_type,
                        time_mode=args.time_mode,
                    )
                else:
                    z0_hat_norm = step_then_xpred(
                        model=model,
                        z0_norm=z0_norm,
                        y=y,
                        forward_eps=forward_eps,
                        sde_eps=sde_eps,
                        t_start_raw=float(t),
                        mid_frac=float(args.mid_frac),
                        path_type=path_type,
                        time_mode=args.time_mode,
                        step_mode=kind,
                    )

                z0_hat_raw = legacy_denormalize(z0_hat_norm, latents_scale, latents_bias)
                pred_raw = unwrap_decode(vae, z0_hat_raw)
                add_batch_metric(
                    accum,
                    setting_key(kind, t),
                    img,
                    pred_raw,
                    args.lpips_network,
                )

                del z0_hat_norm, z0_hat_raw, pred_raw

    for key in accum:
        for field in ("psnr_sum_batchmeans", "lpips_sum_batchmeans", "num_batches"):
            dist.all_reduce(accum[key][field], op=dist.ReduceOp.SUM)

    if rank == 0:
        metrics = {}
        for kind, t in settings:
            key = setting_key(kind, t)
            n = float(accum[key]["num_batches"].item())
            if n <= 0:
                raise RuntimeError(f"No metric batches for {key}")
            metrics[key] = {
                "psnr": float((accum[key]["psnr_sum_batchmeans"] / n).item()),
                "lpips": float((accum[key]["lpips_sum_batchmeans"] / n).item()),
                "num_batch_means": int(n),
            }

        result = {
            "exp_path": args.exp_path,
            "checkpoint": ckpt_path,
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
            "time_mode": args.time_mode,
            "mid_frac": float(args.mid_frac),
            "time_description": time_desc,
            "forward_noise_seed": int(args.forward_noise_seed),
            "sde_noise_seed": int(args.sde_noise_seed),
            "psnr": {
                "input_scale": "raw_vae_image_scale_[-1,1]",
                "zero_mean": True,
            },
            "lpips": {
                "input_scale": "raw_vae_image_scale_[-1,1]",
                "zero_mean": True,
                "network_type": args.lpips_network,
            },
            "aggregation": "mean_of_batch_means_matching_previous_ifid_eval",
            "metrics": metrics,
        }

        out_json = out_dir / "metrics.json"
        out_json.write_text(json.dumps(result, indent=2, sort_keys=True))

        print("\n===== RESULTS =====", flush=True)
        for kind, t in settings:
            key = setting_key(kind, t)
            m = metrics[key]
            print(
                f"{key:16s}  PSNR={m['psnr']:.6f}  LPIPS={m['lpips']:.6f}",
                flush=True,
            )
        print(f"Saved: {out_json}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
