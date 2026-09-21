#!/usr/bin/env python3
"""
DINOv2 cosine similarity for the same decoded proxy images used by the
direct-xpred / step-xpred experiments.

For each source image x and each selected proxy setting:
    x -> proxy decoded image x_hat
    DINOv2(x) and DINOv2(x_hat)
    -> cosine similarity

The repository's existing encoder path is reused:
    load_encoders("dinov2-vit-b", ...)
and the DINOv2 representation is:
    forward_features(...)[ "x_norm_patchtokens" ]

Two cosine summaries are reported at essentially no extra model cost:

1) dino_patch_cos  [PRIMARY]
   Corresponding DINO patch tokens are L2-normalized along feature dimension;
   cosine is computed per patch and averaged over patches and images.
   This matches the token-wise cosine convention already used by REPA/REPA-E
   projector alignment in the codebase.

2) dino_global_cos
   DINO patch tokens are mean-pooled per image first, then cosine is computed.
   This is a more global image-level semantic similarity.

For convenient correlation with gFID/rFID (where lower is better), the JSON
also reports:
    dino_patch_dist  = 1 - dino_patch_cos
    dino_global_dist = 1 - dino_global_cos

Supported settings:
    recon
    direct:<t>
    ode:<t>
    sde:<t>

Examples:
    --settings ode:0.8,sde:0.8,ode:0.4,sde:0.4,ode:0.2,sde:0.2
    --settings direct:0.1,direct:0.4,direct:0.8

Time/noise/diffusion conventions are identical to the previous proxy
evaluators:
- legacy v-prediction SiT checkpoints only
- EMA BatchNorm latent normalization
- shifted_raw by default
- for ODE/SDE, raw midpoint = raw t * mid_frac (default t/2), then each raw
  grid point is shifted independently
- no CFG; true ImageNet class label
- shared forward-noise realization across selected settings in a batch
- shared independent Brownian realization across selected SDE settings

DINO preprocessing follows the repository's dinov2 branch:
    image -> [0,1]
    ImageNet mean/std normalization
    bicubic resize 256 -> 224
Decoder outputs are clamped to valid [0,1] image range before DINO.
"""

import argparse
import gc
import importlib
import json
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from dictdot import dictdot
from omegaconf import OmegaConf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import Normalize
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
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
            "Comma-separated list: recon, direct:t, ode:t, sde:t. "
            "Example: ode:0.8,sde:0.8,ode:0.4,sde:0.4"
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
        help="Default: args.json. Explicit value must match args.json.",
    )

    # Existing REPA/REPA-E encoder convention.
    p.add_argument("--dino-enc-type", type=str, default="dinov2-vit-b")
    p.add_argument(
        "--encoder-code-root",
        type=str,
        default="/share/xutongda/REPA-E",
        help=(
            "Repository root containing utils.py/load_encoders. "
            "The evaluator also tries imports already available on PYTHONPATH."
        ),
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forward-noise-seed", type=int, default=12345)
    p.add_argument("--sde-noise-seed", type=int, default=54321)
    p.add_argument(
        "--out-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/proxy_dinocos",
    )
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def setting_key(kind, t):
    if kind == "recon":
        return "recon"
    return f"{kind}_t{float(t):g}"


def parse_settings(spec):
    out = []
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
                    f"Bad setting {raw!r}; expected recon or kind:t"
                )
            kind, t_str = raw.split(":", 1)
            kind = kind.strip().lower()
            if kind not in {"direct", "ode", "sde"}:
                raise ValueError(f"Unsupported setting kind: {kind}")
            t = float(t_str)
            if not (0.0 < t <= 1.0):
                raise ValueError(f"t must be in (0,1], got {t}")
            item = (kind, t)

        key = setting_key(*item)
        if key not in seen:
            out.append(item)
            seen.add(key)

    if not out:
        raise ValueError("No settings selected")
    return out


def settings_tag(settings):
    return "__".join(
        setting_key(kind, t).replace(".", "p")
        for kind, t in settings
    )


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
            raise TypeError(f"Unsupported VAE encode output: {type(z)}")
    return z


def unwrap_decode(vae, z):
    out = vae.decode(z)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        if hasattr(out, "sample") and torch.is_tensor(out.sample):
            out = out.sample
        else:
            raise TypeError(f"Unsupported VAE decode output: {type(out)}")
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
    alpha, sigma, d_alpha, d_sigma = model.interpolant(
        t_bc, path_type=path_type
    )
    det = alpha * d_sigma - sigma * d_alpha
    if torch.any(torch.as_tensor(det, device=state.device).abs() < 1e-8):
        raise RuntimeError("Degenerate interpolant determinant")
    return (d_sigma * state - sigma * v_hat) / det


@torch.no_grad()
def direct_xpred(
    model,
    z0_norm,
    y,
    forward_eps,
    t_raw,
    path_type,
    time_mode,
):
    b = z0_norm.shape[0]
    t_eff = resolve_time(
        model, t_raw, b, z0_norm.device, z0_norm.dtype, time_mode
    )
    t_bc = broadcast_shape(t_eff, z0_norm)
    alpha, sigma, _, _ = model.interpolant(t_bc, path_type=path_type)
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

    t_start = resolve_time(
        model, t_start_raw, b, z0_norm.device, z0_norm.dtype, time_mode
    )
    t_mid = resolve_time(
        model, t_mid_raw, b, z0_norm.device, z0_norm.dtype, time_mode
    )

    ts_bc = broadcast_shape(t_start, z0_norm)
    alpha_s, sigma_s, _, _ = model.interpolant(
        ts_bc, path_type=path_type
    )
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
    z0_hat = clean_from_state_velocity(
        model, z_mid, v_mid, t_mid, path_type
    )
    if not torch.isfinite(z0_hat).all():
        raise FloatingPointError(
            f"Non-finite clean prediction for {step_mode} t={t_start_raw}"
        )
    return z0_hat


def scalar_shift(t, tshift):
    t = float(t)
    return float(tshift * t / (1.0 + (tshift - 1.0) * t))


def time_description(settings, tshift, time_mode, mid_frac):
    desc = {}
    for kind, t in settings:
        key = setting_key(kind, t)
        if kind == "recon":
            desc[key] = {"kind": "recon"}
            continue

        eff = (
            scalar_shift(t, tshift)
            if time_mode == "shifted_raw"
            else float(t)
        )
        item = {
            "kind": kind,
            "t_raw": float(t),
            "t_effective": eff,
        }
        if kind in {"ode", "sde"}:
            mid_raw = float(t) * float(mid_frac)
            mid_eff = (
                scalar_shift(mid_raw, tshift)
                if time_mode == "shifted_raw"
                else mid_raw
            )
            item["mid_raw"] = mid_raw
            item["mid_effective"] = mid_eff
        desc[key] = item
    return desc


def import_load_encoders(encoder_code_root):
    """
    Reuse the repository's existing load_encoders rather than constructing
    another DINO implementation in this evaluator.
    """
    if encoder_code_root:
        root = str(Path(encoder_code_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    tried = []
    for module_name in ("utils", "ifid.utils", "ifid.sit.utils"):
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, "load_encoders", None)
            if fn is not None:
                return fn, module_name
            tried.append(f"{module_name}: imported but no load_encoders")
        except Exception as e:
            tried.append(f"{module_name}: {type(e).__name__}: {e}")

    raise ImportError(
        "Could not import repository load_encoders.\n"
        f"encoder_code_root={encoder_code_root!r}\n"
        + "\n".join(tried)
        + "\nLocate it with: grep -R \"def load_encoders\" -n "
          "/video_ssd/kongzishang/xutongda/git/IFID "
          "/share/xutongda/REPA-E 2>/dev/null | head -20"
    )


def load_dino_encoder(enc_type, device, resolution, encoder_code_root):
    load_encoders, module_name = import_load_encoders(encoder_code_root)

    encoders, encoder_types, architectures = load_encoders(
        enc_type, device, resolution
    )
    if not isinstance(encoders, (list, tuple)):
        encoders = [encoders]
        encoder_types = [encoder_types]
        architectures = [architectures]

    matches = []
    for encoder, encoder_type, arch in zip(
        encoders, encoder_types, architectures
    ):
        if "dinov2" in str(encoder_type).lower():
            matches.append((encoder, str(encoder_type), str(arch)))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one DINOv2 encoder from enc_type={enc_type!r}, "
            f"got {len(matches)}. encoder_types={list(encoder_types)}"
        )

    encoder, encoder_type, arch = matches[0]
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    return encoder, encoder_type, arch, module_name


_DINO_NORM = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)


def preprocess_dinov2_from_m11(x_m11):
    """
    Equivalent to the repository dinov2 preprocessing branch for 256px images:
      x/255 -> ImageNet Normalize -> bicubic resize to 224.

    Here x_m11 is the IFID evaluator image convention [-1,1], so first map to
    [0,1]. Clamp because decoded predictions can slightly exceed the nominal
    image range.
    """
    resolution = x_m11.shape[-1]
    x = ((x_m11.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    x = _DINO_NORM(x)
    target = 224 * (resolution // 256)
    if target <= 0:
        raise ValueError(f"Unexpected image resolution {resolution}")
    x = F.interpolate(x, target, mode="bicubic")
    return x


@torch.no_grad()
def extract_dino_patchtokens(encoder, x_m11, encoder_type):
    x = preprocess_dinov2_from_m11(x_m11)
    feats = encoder.forward_features(x)

    # This is the exact representation used in the repository's DINOv2 path.
    if isinstance(feats, dict):
        if "x_norm_patchtokens" not in feats:
            raise KeyError(
                "DINOv2 forward_features returned a dict without "
                "'x_norm_patchtokens'. Keys="
                + str(list(feats.keys()))
            )
        tokens = feats["x_norm_patchtokens"]
    else:
        raise TypeError(
            "Expected repository DINOv2 forward_features to return a dict "
            f"with x_norm_patchtokens, got {type(feats)} "
            f"(encoder_type={encoder_type})"
        )

    if tokens.ndim != 3:
        raise RuntimeError(
            f"Expected DINO patch tokens [B,N,C], got {tuple(tokens.shape)}"
        )
    return tokens.float()


@torch.no_grad()
def dino_cosines(source_tokens, pred_tokens):
    if source_tokens.shape != pred_tokens.shape:
        raise RuntimeError(
            f"DINO token shape mismatch: source={tuple(source_tokens.shape)} "
            f"pred={tuple(pred_tokens.shape)}"
        )

    # Primary: same patchwise cosine convention as REPA projector alignment.
    s_patch = F.normalize(source_tokens, dim=-1)
    p_patch = F.normalize(pred_tokens, dim=-1)
    patch_cos_per_image = (s_patch * p_patch).sum(dim=-1).mean(dim=-1)

    # Secondary: mean-pool patch tokens, then image-level cosine.
    s_global = F.normalize(source_tokens.mean(dim=1), dim=-1)
    p_global = F.normalize(pred_tokens.mean(dim=1), dim=-1)
    global_cos_per_image = (s_global * p_global).sum(dim=-1)

    return patch_cos_per_image, global_cos_per_image


def new_accum(device):
    return {
        "patch_sum": torch.zeros((), device=device, dtype=torch.float64),
        "patch_sumsq": torch.zeros((), device=device, dtype=torch.float64),
        "global_sum": torch.zeros((), device=device, dtype=torch.float64),
        "global_sumsq": torch.zeros((), device=device, dtype=torch.float64),
        "count": torch.zeros((), device=device, dtype=torch.float64),
    }


def update_accum(acc, patch_cos, global_cos):
    p = patch_cos.double()
    g = global_cos.double()
    acc["patch_sum"] += p.sum()
    acc["patch_sumsq"] += (p * p).sum()
    acc["global_sum"] += g.sum()
    acc["global_sumsq"] += (g * g).sum()
    acc["count"] += float(p.numel())


def mean_std(sum_v, sumsq_v, n):
    mean = sum_v / n
    var = torch.clamp(sumsq_v / n - mean * mean, min=0.0)
    return float(mean.item()), float(torch.sqrt(var).item())


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
            f"--num-images={args.num_images} must be divisible by "
            f"world_size={world_size}"
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

    # VAE
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
            f"VAE says vae_1d={vae_1d}, model says {model_vae_1d}"
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
            "Missing checkpoint keys:\n"
            + "\n".join(incompat.missing_keys[:100])
        )
    unexpected_keys = list(incompat.unexpected_keys)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for k in ["bn.running_mean", "bn.running_var"]:
        if k not in ema_state:
            raise KeyError(f"Missing legacy BN key: {k}")

    if vae_1d:
        latents_scale = (
            ema_state["bn.running_var"]
            .rsqrt()
            .view(1, 1, in_channels)
            .to(device)
        )
        latents_bias = (
            ema_state["bn.running_mean"]
            .view(1, 1, in_channels)
            .to(device)
        )
    else:
        latents_scale = (
            ema_state["bn.running_var"]
            .rsqrt()
            .view(1, in_channels, 1, 1)
            .to(device)
        )
        latents_bias = (
            ema_state["bn.running_mean"]
            .view(1, in_channels, 1, 1)
            .to(device)
        )

    del state, ema_state
    gc.collect()
    torch.cuda.empty_cache()

    # Reuse repository DINO encoder path.
    dino, dino_type, dino_arch, loader_module = load_dino_encoder(
        args.dino_enc_type,
        device,
        resolution,
        args.encoder_code_root,
    )

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
    tag = settings_tag(settings)
    run_name = (
        f"{exp_name}_step{int(args.train_steps):07d}_"
        f"{args.time_mode}_{tag}_dinocos"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tdesc = time_description(
        settings, tshift, args.time_mode, args.mid_frac
    )

    if rank == 0:
        print(f"exp_path             : {args.exp_path}", flush=True)
        print(f"model                : {model_name}", flush=True)
        print(f"latent shape         : {tuple(fake_z.shape)}", flush=True)
        print(f"tshift               : {tshift:.6f}", flush=True)
        print(f"time_mode            : {args.time_mode}", flush=True)
        print(f"mid_frac             : {args.mid_frac}", flush=True)
        print(f"settings             : {[setting_key(*s) for s in settings]}", flush=True)
        print(f"time_description     : {json.dumps(tdesc)}", flush=True)
        print(f"DINO enc_type        : {args.dino_enc_type}", flush=True)
        print(f"DINO encoder_type    : {dino_type}", flush=True)
        print(f"DINO architecture    : {dino_arch}", flush=True)
        print(f"load_encoders module : {loader_module}", flush=True)
        print("DINO features        : x_norm_patchtokens", flush=True)
        print("primary metric       : mean corresponding-patch cosine", flush=True)
        print("CFG                   : none; true class labels", flush=True)
        print(f"unexpected ckpt keys : {len(unexpected_keys)}", flush=True)
        print(f"num_images            : {args.num_images}", flush=True)
        print(f"world_size            : {world_size}", flush=True)
        print(f"batch/GPU             : {args.batch_size}", flush=True)
        print(f"out_dir               : {out_dir}", flush=True)

    accum = {
        setting_key(kind, t): new_accum(device)
        for kind, t in settings
    }

    fwd_gen = torch.Generator(device=device)
    fwd_gen.manual_seed(args.forward_noise_seed + rank)

    sde_gen = torch.Generator(device=device)
    sde_gen.manual_seed(args.sde_noise_seed + rank)

    pbar = tqdm(
        loader,
        disable=(rank != 0),
        desc="DINO cosine proxy sweep",
    )

    with torch.no_grad():
        for img, y, _name in pbar:
            img = img.to(device, non_blocking=True)  # [-1,1]
            y = y.to(device, non_blocking=True)

            # Original DINO features once per batch, reused for every setting.
            src_tokens = extract_dino_patchtokens(
                dino, img, dino_type
            )

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

            for kind, t in settings:
                key = setting_key(kind, t)

                if kind == "recon":
                    pred_raw = unwrap_decode(vae, z_raw)
                else:
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

                    z0_hat_raw = legacy_denormalize(
                        z0_hat_norm,
                        latents_scale,
                        latents_bias,
                    )
                    pred_raw = unwrap_decode(vae, z0_hat_raw)

                pred_tokens = extract_dino_patchtokens(
                    dino, pred_raw, dino_type
                )
                patch_cos, global_cos = dino_cosines(
                    src_tokens, pred_tokens
                )
                update_accum(
                    accum[key], patch_cos, global_cos
                )

                if kind != "recon":
                    del z0_hat_norm, z0_hat_raw
                del pred_raw, pred_tokens, patch_cos, global_cos

    # Exact global aggregation over images/ranks.
    for key in accum:
        for field in accum[key]:
            dist.all_reduce(accum[key][field], op=dist.ReduceOp.SUM)

    if rank == 0:
        metrics = {}
        for kind, t in settings:
            key = setting_key(kind, t)
            a = accum[key]
            n = float(a["count"].item())
            if int(n) != args.num_images:
                raise RuntimeError(
                    f"{key}: expected {args.num_images} images, got {n}"
                )

            patch_mean, patch_std = mean_std(
                a["patch_sum"], a["patch_sumsq"], n
            )
            global_mean, global_std = mean_std(
                a["global_sum"], a["global_sumsq"], n
            )

            metrics[key] = {
                "dino_patch_cos": patch_mean,
                "dino_patch_cos_std": patch_std,
                "dino_patch_dist": 1.0 - patch_mean,
                "dino_global_cos": global_mean,
                "dino_global_cos_std": global_std,
                "dino_global_dist": 1.0 - global_mean,
                "num_images": int(n),
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
            "time_description": tdesc,
            "forward_noise_seed": int(args.forward_noise_seed),
            "sde_noise_seed": int(args.sde_noise_seed),
            "dino": {
                "requested_enc_type": args.dino_enc_type,
                "encoder_type": dino_type,
                "architecture": dino_arch,
                "load_encoders_module": loader_module,
                "features": "x_norm_patchtokens",
                "preprocess": (
                    "[-1,1] -> clamp [0,1] -> ImageNet mean/std "
                    "-> bicubic 224"
                ),
                "primary": "mean_corresponding_patch_cosine",
            },
            "metrics": metrics,
        }

        out_json = out_dir / "metrics.json"
        out_json.write_text(json.dumps(result, indent=2, sort_keys=True))

        print("\n===== RESULTS =====", flush=True)
        for kind, t in settings:
            key = setting_key(kind, t)
            m = metrics[key]
            print(
                f"{key:16s}  "
                f"DINO_patch_cos={m['dino_patch_cos']:.8f}  "
                f"DINO_global_cos={m['dino_global_cos']:.8f}",
                flush=True,
            )
        print(f"Saved: {out_json}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
