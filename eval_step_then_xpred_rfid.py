#!/usr/bin/env python3

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
from ifid.sit.samplers import get_score_from_velocity, compute_diffusion
from ifid.vae.utils import instantiate_from_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-path", type=str, required=True)
    p.add_argument("--train-steps", type=int, required=True)
    p.add_argument("--dataset", type=str, default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--num-images", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--t-start-values", type=float, nargs="+", default=[0.8])
    p.add_argument("--mid-frac", type=float, default=0.5)
    p.add_argument("--step-mode", type=str, choices=["ode", "sde"], required=True)
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

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forward-noise-seed", type=int, default=12345)
    p.add_argument("--sde-noise-seed", type=int, default=54321)
    p.add_argument(
        "--out-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/step_then_xpred_rfid",
    )
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-rfid", action=argparse.BooleanOptionalAction, default=True)
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
    return z_norm / latents_scale + latents_bias


def resolve_time(model, raw_t, batch, device, dtype, time_mode):
    t = torch.full((batch,), float(raw_t), device=device, dtype=dtype)
    if time_mode == "shifted_raw":
        return model.shift_time(t)
    if time_mode == "fixed_effective":
        return t
    raise ValueError(time_mode)


@torch.no_grad()
def xpred_from_state_velocity(model, state, v_hat, t_eff, path_type):
    """
    Convert a state x_t and velocity prediction v_t into the implied clean x0.

    For linear path this reduces exactly to:
        x0_hat = x_t - t * v_hat
    """
    t_bc = broadcast_shape(t_eff, state)
    alpha, sigma, d_alpha, d_sigma = model.interpolant(t_bc, path_type=path_type)
    det = alpha * d_sigma - sigma * d_alpha
    if torch.any(det.abs() < 1e-8):
        raise RuntimeError("Degenerate interpolant determinant")
    return (d_sigma * state - sigma * v_hat) / det


@torch.no_grad()
def one_reverse_step_then_xpred(
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
    """
    1) Construct the exact forward-noised state at raw t_start.
    2) Take one reverse Euler ODE / Euler-Maruyama SDE step to raw t_start*mid_frac.
    3) Run the model again at the midpoint and convert velocity -> x prediction.
    """
    if not (0.0 < mid_frac < 1.0):
        raise ValueError(f"mid_frac must be in (0,1), got {mid_frac}")

    b = z0_norm.shape[0]
    t_mid_raw = float(t_start_raw) * float(mid_frac)

    t_start = resolve_time(
        model, float(t_start_raw), b, z0_norm.device, z0_norm.dtype, time_mode
    )
    t_mid = resolve_time(
        model, t_mid_raw, b, z0_norm.device, z0_norm.dtype, time_mode
    )

    ts_bc = broadcast_shape(t_start, z0_norm)
    tm_bc = broadcast_shape(t_mid, z0_norm)

    alpha_s, sigma_s, _, _ = model.interpolant(ts_bc, path_type=path_type)
    z_start = alpha_s * z0_norm + sigma_s * forward_eps

    # NFE #1: direction/drift at start time.
    v_start = model.inference(z_start, t_start, y)
    dt = t_mid - t_start
    dt_bc = broadcast_shape(dt, z0_norm)

    if step_mode == "ode":
        # Exactly the local Euler update used by the repository's euler_sampler:
        # x_next = x_cur + (t_next - t_cur) * v_cur
        z_mid = z_start + dt_bc * v_start

    elif step_mode == "sde":
        # Match the repository's Euler-Maruyama local update.
        # The stock sampler uses shifted t_cur in compute_diffusion(t_cur).
        score_start = get_score_from_velocity(
            v_start, z_start, t_start, path_type=path_type
        )
        diffusion = compute_diffusion(t_start)
        diffusion_bc = broadcast_shape(diffusion, z0_norm)

        drift = v_start - 0.5 * diffusion_bc * score_start
        brownian = sde_eps * torch.sqrt(torch.abs(dt_bc))
        z_mid = (
            z_start
            + drift * dt_bc
            + torch.sqrt(diffusion_bc.clamp_min(0.0)) * brownian
        )
    else:
        raise ValueError(step_mode)

    # NFE #2: x-prediction at the midpoint.
    v_mid = model.inference(z_mid, t_mid, y)
    z0_hat = xpred_from_state_velocity(
        model, z_mid, v_mid, t_mid, path_type=path_type
    )

    return z0_hat


def fid_from_features(src, pred):
    mu_src = src.mean(axis=0)
    cov_src = np.cov(src, rowvar=False)
    mu_pred = pred.mean(axis=0)
    cov_pred = np.cov(pred, rowvar=False)
    return float(calculate_frechet_distance(mu_src, cov_src, mu_pred, cov_pred))


def scalar_shift(t, tshift):
    t = float(t)
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

    with open(os.path.join(args.exp_path, "args.json"), "r") as f:
        raw_config = json.load(f)
    config = dictdot(raw_config)

    configured_prediction = str(raw_config.get("prediction", "v")).lower()
    if configured_prediction != "v":
        raise ValueError(f"Expected legacy prediction=v, got {configured_prediction}")
    if "prediction_internal" in raw_config:
        raise ValueError(
            "This evaluator only supports legacy checkpoints without prediction_internal"
        )

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
    state_dict = torch.load(ckpt_path, map_location=device)
    if "ema" not in state_dict:
        raise KeyError(f"Checkpoint has no ema: {ckpt_path}")
    ema_state = state_dict["ema"]

    incompat = model.load_state_dict(ema_state, strict=False)
    if incompat.missing_keys:
        raise RuntimeError(
            "Missing checkpoint keys:\n" + "\n".join(incompat.missing_keys[:100])
        )
    unexpected_keys = list(incompat.unexpected_keys)
    model.eval()

    for k in ["bn.running_mean", "bn.running_var"]:
        if k not in ema_state:
            raise KeyError(f"Missing legacy BN key: {k}")

    if vae_1d:
        latents_scale = (
            ema_state["bn.running_var"].rsqrt().view(1, 1, in_channels).to(device)
        )
        latents_bias = (
            ema_state["bn.running_mean"].view(1, 1, in_channels).to(device)
        )
    else:
        latents_scale = (
            ema_state["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
        )
        latents_bias = (
            ema_state["bn.running_mean"].view(1, in_channels, 1, 1).to(device)
        )

    del state_dict, ema_state
    gc.collect()
    torch.cuda.empty_cache()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
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
    t_tag = "-".join(f"{t:g}" for t in args.t_start_values)
    run_name = (
        f"{exp_name}_step{int(args.train_steps):07d}_"
        f"{args.step_mode}_{path_type}_{args.time_mode}_"
        f"mid{args.mid_frac:g}_t{t_tag}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    time_pairs = {}
    for t in args.t_start_values:
        mid = float(t) * float(args.mid_frac)
        if args.time_mode == "shifted_raw":
            se = scalar_shift(t, tshift)
            me = scalar_shift(mid, tshift)
        else:
            se = float(t)
            me = mid
        time_pairs[f"{float(t):g}"] = {
            "start_raw": float(t),
            "mid_raw": mid,
            "start_effective": se,
            "mid_effective": me,
        }

    if rank == 0:
        print(f"exp_path             : {args.exp_path}", flush=True)
        print(f"model                : {model_name}", flush=True)
        print(f"latent shape         : {tuple(fake_z.shape)}", flush=True)
        print(f"tshift               : {tshift:.6f}", flush=True)
        print(f"step_mode            : {args.step_mode}", flush=True)
        print(f"time_mode            : {args.time_mode}", flush=True)
        print(f"mid_frac             : {args.mid_frac}", flush=True)
        print(f"time_pairs           : {json.dumps(time_pairs)}", flush=True)
        print("NFE                   : 2", flush=True)
        print("CFG                   : none; true class labels", flush=True)
        print(f"unexpected ckpt keys : {len(unexpected_keys)}", flush=True)
        print(f"out_dir              : {out_dir}", flush=True)

    src_feats = []
    rec_feats = [] if args.include_rfid else None
    pred_feats = {float(t): [] for t in args.t_start_values}

    fwd_gen = torch.Generator(device=device)
    fwd_gen.manual_seed(args.forward_noise_seed + rank)
    sde_gen = torch.Generator(device=device)
    sde_gen.manual_seed(args.sde_noise_seed + rank)

    pbar = tqdm(loader, disable=(rank != 0), desc=f"{args.step_mode} step -> xpred rFID")
    with torch.no_grad():
        for img, y, _name in pbar:
            img = img.to(device, non_blocking=True)
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
            # Reuse one independent Brownian-noise realization across all requested t
            # within this image batch. ODE ignores this tensor.
            sde_eps = torch.randn(
                z0_norm.shape,
                dtype=z0_norm.dtype,
                device=device,
                generator=sde_gen,
            )

            x01 = (img + 1.0) / 2.0
            src_feats.append(inception_features(inception, x01).float().cpu().numpy())

            if args.include_rfid:
                x_rec = (unwrap_decode(vae, z_raw) + 1.0) / 2.0
                rec_feats.append(
                    inception_features(inception, x_rec).float().cpu().numpy()
                )

            for t in args.t_start_values:
                z0_hat_norm = one_reverse_step_then_xpred(
                    model=model,
                    z0_norm=z0_norm,
                    y=y,
                    forward_eps=forward_eps,
                    sde_eps=sde_eps,
                    t_start_raw=float(t),
                    mid_frac=float(args.mid_frac),
                    path_type=path_type,
                    time_mode=args.time_mode,
                    step_mode=args.step_mode,
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
        np.save(
            out_dir / f"rec_rank{rank:03d}.npy",
            np.concatenate(rec_feats, axis=0),
        )
    for t in args.t_start_values:
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
            "step_mode": args.step_mode,
            "nfe": 2,
            "time_mode": args.time_mode,
            "mid_frac": float(args.mid_frac),
            "time_pairs": time_pairs,
            "forward_noise_seed": int(args.forward_noise_seed),
            "sde_noise_seed": int(args.sde_noise_seed),
            "metrics": {},
        }

        if args.include_rfid:
            rec_all = np.concatenate(
                [np.load(out_dir / f"rec_rank{r:03d}.npy") for r in range(world_size)],
                axis=0,
            )
            result["metrics"]["rfid"] = fid_from_features(src_all, rec_all)

        for t in args.t_start_values:
            pred_all = np.concatenate(
                [
                    np.load(out_dir / f"t{float(t):g}_rank{r:03d}.npy")
                    for r in range(world_size)
                ],
                axis=0,
            )
            mid = float(t) * float(args.mid_frac)
            key = (
                f"{args.step_mode}_step_then_xpred_rfid_"
                f"t{float(t):g}_to_t{mid:g}"
            )
            result["metrics"][key] = fid_from_features(src_all, pred_all)

        out_json = out_dir / "metrics.json"
        out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(json.dumps(result["metrics"], indent=2, sort_keys=True), flush=True)
        print(f"Saved: {out_json}", flush=True)

        if not args.keep_features:
            for r in range(world_size):
                (out_dir / f"src_rank{r:03d}.npy").unlink(missing_ok=True)
                if args.include_rfid:
                    (out_dir / f"rec_rank{r:03d}.npy").unlink(missing_ok=True)
                for t in args.t_start_values:
                    (out_dir / f"t{float(t):g}_rank{r:03d}.npy").unlink(
                        missing_ok=True
                    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
