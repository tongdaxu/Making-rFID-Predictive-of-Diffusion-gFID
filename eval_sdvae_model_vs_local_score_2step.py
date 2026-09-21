#!/usr/bin/env python3
"""
Two-step single-image SD-VAE experiment:
start from t=0.8, do a reverse ODE step to t=0.4, then do x-pred at t=0.4.

Compare three branches:
A) Model (SiT)
B) Local Score (ours)
C) Ideal Score (control)

Default setting matches the teacher's request:
- one ImageNet val image
- cross-class / unconditional bank
- 50k ImageNet-train references
- t_start = 0.8, t_mid = 0.4
- Local patch schedule fixed manually:
    P(0.8)=7, P(0.4)=3

Outputs:
- one-step decoded x-preds at t=0.8
- intermediate decoded states at t=0.4 (optional but useful)
- final two-step decoded x-preds
- comparison contact sheet
- metrics.json / metrics.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.sit.sit import SiT_models
from ifid.vae.utils import instantiate_from_config


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--exp-path",
        type=str,
        default="/share/xutongda/REPA-E/exps/sit-xl-sdvae-400k",
    )
    p.add_argument("--train-steps", type=int, default=400000)

    p.add_argument("--dataset", type=str, default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", type=str, default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--val-index", type=int, default=0)

    # Default = teacher requested cross-class 50k
    g = p.add_mutually_exclusive_group()
    g.add_argument("--unconditional-bank", dest="unconditional_bank", action="store_true")
    g.add_argument("--same-class-bank", dest="unconditional_bank", action="store_false")
    p.set_defaults(unconditional_bank=True)

    p.add_argument(
        "--unconditional-ref-size",
        type=int,
        default=50000,
        help="Used only when --unconditional-bank is active.",
    )
    p.add_argument(
        "--ref-per-class",
        type=int,
        default=1000,
        help="Used only when --same-class-bank is active. <=0 means all.",
    )

    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--t-start", type=float, default=0.8)
    p.add_argument("--t-mid", type=float, default=0.4)

    # fixed Local Score patch schedule
    p.add_argument("--p-start", type=int, default=7)
    p.add_argument("--p-mid", type=int, default=3)

    p.add_argument(
        "--time-mode",
        choices=["shifted_raw", "fixed_effective"],
        default="shifted_raw",
    )
    p.add_argument("--seed", type=int, default=12345)

    p.add_argument(
        "--cache-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_single/cache",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_single/sdvae_model_vs_local_2step_crossclass50k",
    )
    p.add_argument(
        "--force-reencode",
        action="store_true",
        help="Ignore existing reference-latent cache.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# VAE/model helpers
# ---------------------------------------------------------------------

def unwrap_encode(vae, img, y=None):
    try:
        z = vae.encode(img, y) if y is not None else vae.encode(img)
    except TypeError:
        z = vae.encode(img)

    if isinstance(z, (tuple, list)):
        z = z[0]
    if torch.is_tensor(z):
        return z
    if hasattr(z, "latent_dist"):
        ld = z.latent_dist
        if hasattr(ld, "mode"):
            return ld.mode()
        return ld.sample()
    if hasattr(z, "mode") and callable(z.mode):
        return z.mode()
    if hasattr(z, "sample") and torch.is_tensor(z.sample):
        return z.sample
    raise TypeError(f"Unsupported VAE encode output: {type(z)}")


def unwrap_decode(vae, z):
    out = vae.decode(z)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out):
        return out
    if hasattr(out, "sample") and torch.is_tensor(out.sample):
        return out.sample
    raise TypeError(f"Unsupported VAE decode output: {type(out)}")


def normalize_latent(z_raw, scale, bias):
    return (z_raw - bias) * scale


def denormalize_latent(z_norm, scale, bias):
    return z_norm / scale + bias


def resolve_time(model, t_raw, device, dtype, time_mode):
    t = torch.tensor([float(t_raw)], device=device, dtype=dtype)
    if time_mode == "shifted_raw":
        return model.shift_time(t)
    return t


def broadcast_time(t, z):
    return t.view(t.shape[0], *([1] * (z.ndim - 1)))


@torch.no_grad()
def model_clean_from_velocity(model, z_t, v, t_eff, path_type):
    t_bc = broadcast_time(t_eff, z_t)
    alpha, sigma, d_alpha, d_sigma = model.interpolant(t_bc, path_type=path_type)
    det = alpha * d_sigma - sigma * d_alpha
    if torch.any(det.abs() < 1e-8):
        raise RuntimeError("Degenerate interpolant determinant.")
    z0_hat = (d_sigma * z_t - sigma * v) / det
    return z0_hat, alpha, sigma, d_alpha, d_sigma


def induced_velocity_from_xpred(z_t, z0_hat, alpha, sigma, d_alpha, d_sigma):
    if torch.any(sigma.abs() < 1e-8):
        raise RuntimeError("sigma too small for velocity conversion.")
    eps_hat = (z_t - alpha * z0_hat) / sigma
    return d_alpha * z0_hat + d_sigma * eps_hat


def induced_score_from_xpred(z_t, z0_hat, alpha, sigma):
    return (alpha * z0_hat - z_t) / (sigma * sigma)


# ---------------------------------------------------------------------
# Exact Ideal / Local Score posterior means
# ---------------------------------------------------------------------

def box_sum_2d(x, kernel_size):
    p = int(kernel_size)
    if p == 1:
        return x
    if p <= 0 or p % 2 != 1:
        raise ValueError(f"Patch size must be positive odd, got {p}")

    r = p // 2
    xp = F.pad(x, (r, r, r, r), mode="constant", value=0.0)
    ii = F.pad(xp, (1, 0, 1, 0), mode="constant", value=0.0)
    ii = ii.cumsum(dim=-2).cumsum(dim=-1)
    out = (
        ii[..., p:, p:]
        - ii[..., :-p, p:]
        - ii[..., p:, :-p]
        + ii[..., :-p, :-p]
    )
    return out


@torch.no_grad()
def analytic_posteriors(bank, z_t, alpha, sigma, patch_sizes):
    if bank.ndim != 4 or z_t.ndim != 4:
        raise ValueError("Expected BCHW latents.")

    alpha_s = float(alpha.flatten()[0].item())
    sigma_s = float(sigma.flatten()[0].item())
    if sigma_s <= 0:
        raise ValueError(f"sigma must be >0, got {sigma_s}")

    diff2 = ((alpha_s * bank - z_t) ** 2).sum(dim=1, keepdim=True)  # [N,1,H,W]

    # Ideal
    global_dist = diff2.flatten(1).sum(dim=1)
    global_logits = -global_dist / (2.0 * sigma_s * sigma_s)
    global_w = torch.softmax(global_logits, dim=0)
    ideal_mu = (global_w[:, None, None, None] * bank).sum(dim=0, keepdim=True)

    # Local
    locals_by_p = {}
    for p in patch_sizes:
        local_dist = box_sum_2d(diff2, p)[:, 0]  # [N,H,W]
        local_logits = -local_dist / (2.0 * sigma_s * sigma_s)
        local_w = torch.softmax(local_logits, dim=0)
        mu = (local_w[:, None, :, :] * bank).sum(dim=0, keepdim=True)
        locals_by_p[int(p)] = mu

    return ideal_mu, locals_by_p


# ---------------------------------------------------------------------
# Metrics / image output
# ---------------------------------------------------------------------

def flat_cos(a, b):
    return float(
        F.cosine_similarity(
            a.float().flatten(1),
            b.float().flatten(1),
            dim=1,
            eps=1e-8,
        ).mean().item()
    )


def rel_mse(a, b, eps=1e-8):
    num = (a.float() - b.float()).pow(2).flatten(1).mean(dim=1)
    den = a.float().pow(2).flatten(1).mean(dim=1)
    return float((num / (den + eps)).mean().item())


def image_psnr(a, b, eps=1e-10):
    a01 = ((a.float() + 1.0) / 2.0).clamp(0, 1)
    b01 = ((b.float() + 1.0) / 2.0).clamp(0, 1)
    mse = (a01 - b01).pow(2).mean()
    return float((-10.0 * torch.log10(mse + eps)).item())


def tensor_to_pil(x):
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    x = ((x + 1.0) / 2.0).clamp(0, 1)
    arr = (
        (x * 255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(arr)


def labeled_cell(img, label, width=256, label_h=28):
    img = img.resize((width, width), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, width + label_h), "white")
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), label, fill="black")
    return canvas


def make_contact_sheet(rows, out_path, cell=256):
    label_h = 28
    ncols = max(len(r) for r in rows)
    nrows = len(rows)
    sheet = Image.new("RGB", (ncols * cell, nrows * (cell + label_h)), "white")
    for ri, row in enumerate(rows):
        for ci, (img, label) in enumerate(row):
            cell_img = labeled_cell(img, label, width=cell, label_h=label_h)
            sheet.paste(cell_img, (ci * cell, ri * (cell + label_h)))
    sheet.save(out_path)


# ---------------------------------------------------------------------
# Reference latent bank
# ---------------------------------------------------------------------

def choose_reference_indices(
    train_set,
    query_label,
    class_conditioned,
    ref_per_class,
    unconditional_ref_size,
):
    if class_conditioned:
        idx = [
            i for i, (_path, y) in enumerate(train_set.samples)
            if int(y) == int(query_label)
        ]
        if ref_per_class > 0:
            idx = idx[:ref_per_class]
        if not idx:
            raise RuntimeError(f"No training samples found for class {query_label}")
        return idx

    idx = list(range(len(train_set.samples)))
    if unconditional_ref_size > 0 and unconditional_ref_size < len(idx):
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(idx), generator=g)[:unconditional_ref_size]
        idx = perm.tolist()
    return idx


@torch.no_grad()
def build_or_load_bank(
    args,
    vae,
    train_set,
    ref_indices,
    query_label,
    scale,
    bias,
    device,
):
    exp_name = Path(args.exp_path).name
    cond_tag = (
        f"class{int(query_label):04d}"
        if not args.unconditional_bank
        else "unconditional"
    )
    nref = len(ref_indices)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{exp_name}_{cond_tag}_n{nref}_legacyBN_fp16.pt"

    if cache_path.exists() and not args.force_reencode:
        cache = torch.load(cache_path, map_location="cpu")
        bank = cache["latents"].float()
        if tuple(bank.shape[1:]) != tuple(cache["latent_shape"]):
            raise RuntimeError("Corrupt latent bank cache.")
        print(f"[cache] loaded {cache_path}")
        return bank, cache_path

    subset = Subset(train_set, ref_indices)
    loader = DataLoader(
        subset,
        batch_size=args.encode_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    parts = []
    for img, y, _name in tqdm(loader, desc="Encoding reference latents"):
        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z_raw = unwrap_encode(vae, img, y)
        if z_raw.ndim != 4:
            raise RuntimeError(f"Expected BCHW latent, got {tuple(z_raw.shape)}")
        z_norm = normalize_latent(z_raw, scale, bias)
        parts.append(z_norm.detach().cpu().to(torch.float16))

    bank = torch.cat(parts, dim=0)
    cache = {
        "latents": bank,
        "latent_shape": tuple(bank.shape[1:]),
        "num_refs": int(bank.shape[0]),
        "query_label": int(query_label),
        "class_conditioned": not args.unconditional_bank,
        "exp_path": args.exp_path,
        "normalization": "SiT EMA BatchNorm running mean/var",
    }
    torch.save(cache, cache_path)
    print(f"[cache] saved {cache_path}")
    return bank.float(), cache_path


# ---------------------------------------------------------------------
# Reverse step helper
# ---------------------------------------------------------------------

@torch.no_grad()
def reverse_euler_step(state_t, velocity_t, t_eff_cur, t_eff_next):
    dt = float((t_eff_next - t_eff_cur).item())
    return state_t + dt * velocity_t


def save_image(vae, z_norm_or_xraw, out_path, is_norm_latent, scale, bias):
    if is_norm_latent:
        z_raw = denormalize_latent(z_norm_or_xraw, scale, bias)
        img = unwrap_decode(vae, z_raw)
    else:
        img = z_norm_or_xraw
    tensor_to_pil(img).save(out_path)
    return img


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    if not (0.0 < args.t_mid < args.t_start < 1.0):
        raise ValueError("Require 0 < t_mid < t_start < 1.")

    for p in [args.p_start, args.p_mid]:
        if p <= 0 or p % 2 != 1:
            raise ValueError(f"Patch size must be positive odd, got {p}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    # ---------------------------------------------------------
    # Load experiment configuration / VAE / SiT
    # ---------------------------------------------------------
    args_json = Path(args.exp_path) / "args.json"
    raw_cfg = json.loads(args_json.read_text())

    prediction = str(raw_cfg.get("prediction", "v")).lower()
    if prediction != "v":
        raise RuntimeError(f"Expected prediction=v, got {prediction}")
    if "prediction_internal" in raw_cfg:
        raise RuntimeError("Refusing checkpoint with prediction_internal.")

    path_type = str(raw_cfg.get("path_type", "linear")).lower()
    if path_type != "linear":
        raise RuntimeError(f"Expected benchmark linear path, got {path_type}")

    cfg = type("Cfg", (), raw_cfg)

    vae_cfg = OmegaConf.load(cfg.vae_config)
    vae = instantiate_from_config(vae_cfg).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    resolution = int(raw_cfg.get("resolution", 256))
    fake = torch.zeros(1, 3, resolution, resolution, device=device)
    fake_z = unwrap_encode(vae, fake)[0]
    if fake_z.ndim != 3:
        raise RuntimeError(f"Expected CHW latent, got {tuple(fake_z.shape)}")
    C, H, W = fake_z.shape
    if H != W:
        raise RuntimeError(f"Expected square latent, got {H}x{W}")
    if args.p_start > H or args.p_mid > H:
        raise ValueError(f"Patch size exceeds latent size {H}")

    tshift = math.sqrt(float(fake_z.numel()) / 4096.0)

    model_name = str(cfg.model)
    if model_name not in SiT_models:
        alt = model_name.replace("SiT_", "SiT-")
        if alt in SiT_models:
            model_name = alt
        else:
            raise KeyError(f"Unknown SiT model name: {cfg.model}")

    model = SiT_models[model_name](
        input_size=H,
        in_channels=C,
        num_classes=cfg.num_classes,
        class_dropout_prob=cfg.cfg_prob,
        bn_momentum=cfg.bn_momentum,
        tshift=tshift,
        fused_attn=raw_cfg.get("fused_attn", True),
        qk_norm=raw_cfg.get("qk_norm", False),
    ).to(device)

    ckpt_path = Path(args.exp_path) / "checkpoints" / f"{int(args.train_steps):07d}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    ema = ckpt["ema"]

    incompat = model.load_state_dict(ema, strict=False)
    if incompat.missing_keys:
        raise RuntimeError("Missing model keys:\n" + "\n".join(incompat.missing_keys[:100]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    running_mean = ema["bn.running_mean"].view(1, C, 1, 1).float()
    running_var = ema["bn.running_var"].view(1, C, 1, 1).float()
    scale = running_var.rsqrt().to(device)
    bias = running_mean.to(device)

    del ckpt, ema

    # ---------------------------------------------------------
    # Dataset + one query image
    # ---------------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    val_set = ImageNetValDataset(args.dataset, transform, -1)
    train_set = ImageNetValDataset(args.dataset_ref, transform, -1)

    if val_set.class_to_idx != train_set.class_to_idx:
        raise RuntimeError("Train/val class_to_idx mappings differ.")

    if not (0 <= args.val_index < len(val_set)):
        raise IndexError(f"--val-index={args.val_index} outside valid range")

    img, query_label, query_name = val_set[args.val_index]
    img = img.unsqueeze(0).to(device)
    y = torch.tensor([query_label], device=device, dtype=torch.long)

    class_name = next(k for k, v in val_set.class_to_idx.items() if int(v) == int(query_label))

    z0_raw = unwrap_encode(vae, img, y)
    z0 = normalize_latent(z0_raw, scale, bias)
    recon = unwrap_decode(vae, z0_raw)

    # fixed epsilon
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    eps = torch.randn(z0.shape, generator=g, device=device, dtype=z0.dtype)

    # ---------------------------------------------------------
    # Build reference bank
    # ---------------------------------------------------------
    ref_indices = choose_reference_indices(
        train_set=train_set,
        query_label=query_label,
        class_conditioned=(not args.unconditional_bank),
        ref_per_class=args.ref_per_class,
        unconditional_ref_size=args.unconditional_ref_size,
    )

    bank_cpu, cache_path = build_or_load_bank(
        args=args,
        vae=vae,
        train_set=train_set,
        ref_indices=ref_indices,
        query_label=query_label,
        scale=scale,
        bias=bias,
        device=device,
    )

    bank = bank_cpu.to(device=device, dtype=torch.float32)
    z0 = z0.float()
    z0_raw = z0_raw.float()
    eps = eps.float()

    # ---------------------------------------------------------
    # Output setup
    # ---------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor_to_pil(img).save(out_dir / "original.png")
    tensor_to_pil(recon).save(out_dir / "vae_reconstruction.png")

    print("============================================================")
    print("Two-step Model vs Local Score experiment")
    print(f"exp_path              : {args.exp_path}")
    print(f"checkpoint            : {ckpt_path}")
    print(f"model                 : {model_name}")
    print(f"query                 : {query_name}")
    print(f"query class           : {query_label} ({class_name})")
    print(f"latent shape          : {(C, H, W)}")
    print(f"time_mode             : {args.time_mode}")
    print(f"t_start -> t_mid      : {args.t_start} -> {args.t_mid}")
    print(f"Local patch schedule  : P({args.t_start})={args.p_start}, P({args.t_mid})={args.p_mid}")
    print(f"class conditioned     : {not args.unconditional_bank}")
    print(f"reference count       : {bank.shape[0]}")
    print(f"reference cache       : {cache_path}")
    print("============================================================")

    # ---------------------------------------------------------
    # Time setup
    # ---------------------------------------------------------
    t_start_eff = resolve_time(model, args.t_start, device, z0.dtype, args.time_mode)
    t_mid_eff = resolve_time(model, args.t_mid, device, z0.dtype, args.time_mode)

    t_start_bc = broadcast_time(t_start_eff, z0)
    alpha_start, sigma_start, d_alpha_start, d_sigma_start = model.interpolant(
        t_start_bc, path_type=path_type
    )

    t_mid_bc = broadcast_time(t_mid_eff, z0)
    alpha_mid, sigma_mid, d_alpha_mid, d_sigma_mid = model.interpolant(
        t_mid_bc, path_type=path_type
    )

    # forward corruption to z_{0.8}
    z_start = alpha_start * z0 + sigma_start * eps

    # ---------------------------------------------------------
    # ONE-STEP at t=0.8
    # ---------------------------------------------------------
    # Model
    v_model_1 = model.inference(z_start, t_start_eff, y)
    model_xpred_1, alpha1m, sigma1m, da1m, ds1m = model_clean_from_velocity(
        model, z_start, v_model_1, t_start_eff, path_type
    )
    model_score_1 = induced_score_from_xpred(z_start, model_xpred_1, alpha1m, sigma1m)

    # Ideal / Local from same z_start
    ideal_xpred_1, local_dict_1 = analytic_posteriors(
        bank=bank,
        z_t=z_start,
        alpha=alpha_start,
        sigma=sigma_start,
        patch_sizes=[args.p_start],
    )
    local_xpred_1 = local_dict_1[args.p_start]

    ideal_v_1 = induced_velocity_from_xpred(z_start, ideal_xpred_1, alpha1m, sigma1m, da1m, ds1m)
    ideal_score_1 = induced_score_from_xpred(z_start, ideal_xpred_1, alpha1m, sigma1m)

    local_v_1 = induced_velocity_from_xpred(z_start, local_xpred_1, alpha1m, sigma1m, da1m, ds1m)
    local_score_1 = induced_score_from_xpred(z_start, local_xpred_1, alpha1m, sigma1m)

    # decode one-step x-preds
    model_img_1 = save_image(vae, model_xpred_1, out_dir / "t0p8_model_xpred_1step.png", True, scale, bias)
    local_img_1 = save_image(vae, local_xpred_1, out_dir / f"t0p8_local_P{args.p_start:02d}_xpred_1step.png", True, scale, bias)
    ideal_img_1 = save_image(vae, ideal_xpred_1, out_dir / "t0p8_ideal_xpred_1step.png", True, scale, bias)

    # ---------------------------------------------------------
    # STEP 1: 0.8 -> 0.4
    # ---------------------------------------------------------
    z_model_mid = reverse_euler_step(z_start, v_model_1, t_start_eff, t_mid_eff)
    z_local_mid = reverse_euler_step(z_start, local_v_1, t_start_eff, t_mid_eff)
    z_ideal_mid = reverse_euler_step(z_start, ideal_v_1, t_start_eff, t_mid_eff)

    # decode intermediate states
    model_mid_img = save_image(vae, z_model_mid, out_dir / "t0p4_model_state_after_step1.png", True, scale, bias)
    local_mid_img = save_image(vae, z_local_mid, out_dir / "t0p4_local_state_after_step1.png", True, scale, bias)
    ideal_mid_img = save_image(vae, z_ideal_mid, out_dir / "t0p4_ideal_state_after_step1.png", True, scale, bias)

    # ---------------------------------------------------------
    # STEP 2 / final x-pred at t=0.4
    # ---------------------------------------------------------
    # Model branch
    v_model_2 = model.inference(z_model_mid, t_mid_eff, y)
    model_xpred_2, alpha2m, sigma2m, da2m, ds2m = model_clean_from_velocity(
        model, z_model_mid, v_model_2, t_mid_eff, path_type
    )
    model_img_2 = save_image(vae, model_xpred_2, out_dir / "t0p8_to_t0p4_model_xpred_2step_final.png", True, scale, bias)

    # Local branch
    ideal_from_local_mid_dummy, local_dict_2 = analytic_posteriors(
        bank=bank,
        z_t=z_local_mid,
        alpha=alpha_mid,
        sigma=sigma_mid,
        patch_sizes=[args.p_mid],
    )
    local_xpred_2 = local_dict_2[args.p_mid]
    local_v_2 = induced_velocity_from_xpred(z_local_mid, local_xpred_2, alpha2m, sigma2m, da2m, ds2m)
    local_img_2 = save_image(
        vae,
        local_xpred_2,
        out_dir / f"t0p8_to_t0p4_local_P{args.p_mid:02d}_xpred_2step_final.png",
        True,
        scale,
        bias,
    )

    # Ideal branch
    ideal_xpred_2, _ = analytic_posteriors(
        bank=bank,
        z_t=z_ideal_mid,
        alpha=alpha_mid,
        sigma=sigma_mid,
        patch_sizes=[args.p_mid],
    )
    ideal_v_2 = induced_velocity_from_xpred(z_ideal_mid, ideal_xpred_2, alpha2m, sigma2m, da2m, ds2m)
    ideal_img_2 = save_image(vae, ideal_xpred_2, out_dir / "t0p8_to_t0p4_ideal_xpred_2step_final.png", True, scale, bias)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    metrics_rows = []

    row = {
        "phase": "one_step_t0p8",
        "method": f"local_P{args.p_start}",
        "velocity_cos_model": flat_cos(v_model_1, local_v_1),
        "score_cos_model": flat_cos(model_score_1, local_score_1),
        "xpred_cos_model": flat_cos(model_xpred_1, local_xpred_1),
        "xpred_rel_mse_model": rel_mse(model_xpred_1, local_xpred_1),
        "decode_psnr_model": image_psnr(model_img_1, local_img_1),
        "state_cos_model": "",
        "state_rel_mse_model": "",
    }
    metrics_rows.append(row)

    row = {
        "phase": "one_step_t0p8",
        "method": "ideal",
        "velocity_cos_model": flat_cos(v_model_1, ideal_v_1),
        "score_cos_model": flat_cos(model_score_1, ideal_score_1),
        "xpred_cos_model": flat_cos(model_xpred_1, ideal_xpred_1),
        "xpred_rel_mse_model": rel_mse(model_xpred_1, ideal_xpred_1),
        "decode_psnr_model": image_psnr(model_img_1, ideal_img_1),
        "state_cos_model": "",
        "state_rel_mse_model": "",
    }
    metrics_rows.append(row)

    row = {
        "phase": "mid_state_t0p4_after_step1",
        "method": f"local_P{args.p_start}",
        "velocity_cos_model": "",
        "score_cos_model": "",
        "xpred_cos_model": "",
        "xpred_rel_mse_model": "",
        "decode_psnr_model": image_psnr(model_mid_img, local_mid_img),
        "state_cos_model": flat_cos(z_model_mid, z_local_mid),
        "state_rel_mse_model": rel_mse(z_model_mid, z_local_mid),
    }
    metrics_rows.append(row)

    row = {
        "phase": "mid_state_t0p4_after_step1",
        "method": "ideal",
        "velocity_cos_model": "",
        "score_cos_model": "",
        "xpred_cos_model": "",
        "xpred_rel_mse_model": "",
        "decode_psnr_model": image_psnr(model_mid_img, ideal_mid_img),
        "state_cos_model": flat_cos(z_model_mid, z_ideal_mid),
        "state_rel_mse_model": rel_mse(z_model_mid, z_ideal_mid),
    }
    metrics_rows.append(row)

    row = {
        "phase": "two_step_final_xpred_at_t0p4",
        "method": f"local_P{args.p_mid}",
        "velocity_cos_model": flat_cos(v_model_2, local_v_2),
        "score_cos_model": "",
        "xpred_cos_model": flat_cos(model_xpred_2, local_xpred_2),
        "xpred_rel_mse_model": rel_mse(model_xpred_2, local_xpred_2),
        "decode_psnr_model": image_psnr(model_img_2, local_img_2),
        "state_cos_model": "",
        "state_rel_mse_model": "",
    }
    metrics_rows.append(row)

    row = {
        "phase": "two_step_final_xpred_at_t0p4",
        "method": "ideal",
        "velocity_cos_model": flat_cos(v_model_2, ideal_v_2),
        "score_cos_model": "",
        "xpred_cos_model": flat_cos(model_xpred_2, ideal_xpred_2),
        "xpred_rel_mse_model": rel_mse(model_xpred_2, ideal_xpred_2),
        "decode_psnr_model": image_psnr(model_img_2, ideal_img_2),
        "state_cos_model": "",
        "state_rel_mse_model": "",
    }
    metrics_rows.append(row)

    # ---------------------------------------------------------
    # Contact sheets
    # ---------------------------------------------------------
    original_pil = tensor_to_pil(img)
    recon_pil = tensor_to_pil(recon)

    one_step_sheet = [
        [
            (original_pil, f"Original ({class_name})"),
            (recon_pil, "VAE recon"),
            (tensor_to_pil(model_img_1), "Model xpred t=0.8"),
            (tensor_to_pil(local_img_1), f"Local xpred t=0.8 P={args.p_start}"),
            (tensor_to_pil(ideal_img_1), "Ideal xpred t=0.8"),
        ]
    ]
    make_contact_sheet(one_step_sheet, out_dir / "comparison_one_step_t0p8.png", cell=256)

    mid_sheet = [
        [
            (original_pil, f"Original ({class_name})"),
            (recon_pil, "VAE recon"),
            (tensor_to_pil(model_mid_img), "State after step1 @ t=0.4 (Model)"),
            (tensor_to_pil(local_mid_img), "State after step1 @ t=0.4 (Local)"),
            (tensor_to_pil(ideal_mid_img), "State after step1 @ t=0.4 (Ideal)"),
        ]
    ]
    make_contact_sheet(mid_sheet, out_dir / "comparison_mid_state_t0p4.png", cell=256)

    two_step_sheet = [
        [
            (original_pil, f"Original ({class_name})"),
            (recon_pil, "VAE recon"),
            (tensor_to_pil(model_img_2), "2-step Model final"),
            (tensor_to_pil(local_img_2), f"2-step Local final P={args.p_mid}"),
            (tensor_to_pil(ideal_img_2), "2-step Ideal final"),
        ]
    ]
    make_contact_sheet(two_step_sheet, out_dir / "comparison_two_step_final.png", cell=256)

    full_sheet = [
        [
            (original_pil, f"Original ({class_name})"),
            (recon_pil, "VAE recon"),
            (tensor_to_pil(model_img_1), "1-step Model"),
            (tensor_to_pil(local_img_1), f"1-step Local P={args.p_start}"),
            (tensor_to_pil(ideal_img_1), "1-step Ideal"),
        ],
        [
            (original_pil, f"Original ({class_name})"),
            (recon_pil, "VAE recon"),
            (tensor_to_pil(model_img_2), "2-step Model"),
            (tensor_to_pil(local_img_2), f"2-step Local P={args.p_mid}"),
            (tensor_to_pil(ideal_img_2), "2-step Ideal"),
        ],
    ]
    full_sheet_path = out_dir / "comparison_1step_vs_2step.png"
    make_contact_sheet(full_sheet, full_sheet_path, cell=256)

    # ---------------------------------------------------------
    # CSV / JSON
    # ---------------------------------------------------------
    csv_path = out_dir / "metrics.csv"
    fields = [
        "phase",
        "method",
        "velocity_cos_model",
        "score_cos_model",
        "xpred_cos_model",
        "xpred_rel_mse_model",
        "decode_psnr_model",
        "state_cos_model",
        "state_rel_mse_model",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metrics_rows)

    result = {
        "experiment": "single_image_2step_model_vs_local_score",
        "paper": "Kamb & Ganguli, arXiv:2412.20292",
        "exp_path": args.exp_path,
        "checkpoint": str(ckpt_path),
        "model": model_name,
        "vae_config": str(cfg.vae_config),
        "query": {
            "val_index": int(args.val_index),
            "filename": query_name,
            "class_index": int(query_label),
            "class_name": class_name,
        },
        "latent_shape": [int(C), int(H), int(W)],
        "normalization": "SiT EMA BatchNorm running statistics",
        "path_type": path_type,
        "time_mode": args.time_mode,
        "tshift": float(tshift),
        "schedule": {
            "t_start": float(args.t_start),
            "t_mid": float(args.t_mid),
            "local_p_start": int(args.p_start),
            "local_p_mid": int(args.p_mid),
        },
        "seed": int(args.seed),
        "reference_bank": {
            "class_conditioned": not args.unconditional_bank,
            "count": int(bank.shape[0]),
            "cache": str(cache_path),
        },
        "outputs": {
            "comparison_one_step": str(out_dir / "comparison_one_step_t0p8.png"),
            "comparison_mid_state": str(out_dir / "comparison_mid_state_t0p4.png"),
            "comparison_two_step": str(out_dir / "comparison_two_step_final.png"),
            "comparison_full": str(full_sheet_path),
            "metrics_csv": str(csv_path),
        },
        "rows": metrics_rows,
    }

    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(result, indent=2))

    print("\n============================================================")
    print("DONE")
    print(f"comparison full  : {full_sheet_path}")
    print(f"metrics CSV      : {csv_path}")
    print(f"metrics JSON     : {json_path}")
    print("============================================================")


if __name__ == "__main__":
    main(parse_args())