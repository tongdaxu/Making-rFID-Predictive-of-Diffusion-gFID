#!/usr/bin/env python3
"""
Single-image SD-VAE experiment:
compare one-step SiT model x-pred with the analytic Local Score (LS) x-pred
adapted from Kamb & Ganguli, arXiv:2412.20292.

Goal
----
For one ImageNet validation image and fixed Gaussian noise eps, evaluate
t = 0.8, 0.6, 0.4, 0.2 (configurable):

    z0 --add noise--> z_t

Then compare:

A) Model:
    z_t -> SiT velocity prediction -> model x_pred

B) Ideal Score (control):
    exact empirical Bayes posterior over reference training latents
    -> ideal x_pred = posterior mean E[z0 | z_t]

C) Local Score (ours):
    for every latent spatial location x, form an independent posterior using
    only the co-located P x P latent neighborhood Omega_x
    -> local x_pred(x) = E[z0(x) | z_t(Omega_x)]

For each t and each P, the script reports:
    - cosine(model velocity, local analytic velocity)   [paper-style calibration]
    - cosine(model score, local score)
    - cosine(model x_pred, local x_pred)
    - relative MSE(model x_pred, local x_pred)

It decodes model / ideal / local x-preds with the same SD-VAE and saves:
    - individual PNGs
    - one contact sheet for all t using the best P at each t
    - one contact sheet per t showing all locality scales
    - metrics.json / metrics.csv

Important conventions
---------------------
1. This comparison intentionally uses the SiT checkpoint's EMA BatchNorm
   normalization, so model and analytic LS live in exactly the same normalized
   latent space. This is for validating "can Local Score approximate the trained
   one-step model?". A future fully diffusion-free metric should replace this
   normalization with statistics computed from the VAE training latent bank.

2. The analytic posterior is class-conditioned by default because the SiT being
   modeled is class-conditional. The paper explicitly states that its (E)LS
   machine inherits the class-conditionality of the neural network it models.
   For ImageNet this means the reference bank contains training examples from
   the same class as the query image.

3. We use the same Gaussian noise realization eps for every t, matching the
   previous direct-xpred sweep convention.

4. For the SiT linear path:
       alpha(t)=1-t, sigma(t)=t
   but the implementation uses model.interpolant() and supports shifted_raw.
   For SD-VAE tshift=1, raw and effective t are identical.

5. Local Score is the LS machine, not ELS:
   candidate patches are co-located in the training latent, exactly as Eq. (7-8)
   in the paper. No translation search over other spatial locations is done.
"""

import argparse
import csv
import json
import math
import os
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

    # Existing trained SD-VAE + SiT experiment used only as the comparison model.
    p.add_argument(
        "--exp-path",
        type=str,
        default="/share/xutongda/REPA-E/exps/sit-xl-sdvae-400k",
    )
    p.add_argument("--train-steps", type=int, default=400000)

    p.add_argument(
        "--dataset",
        type=str,
        default="/video_ssd/lpm/ImageNet/val",
    )
    p.add_argument(
        "--dataset-ref",
        type=str,
        default="/video_ssd/lpm/ImageNet/train",
    )
    p.add_argument("--val-index", type=int, default=0)

    # Use same-class ImageNet train examples by default.
    p.add_argument(
        "--ref-per-class",
        type=int,
        default=1000,
        help="Number of same-class training latents. <=0 means all available.",
    )
    p.add_argument(
        "--unconditional-bank",
        action="store_true",
        help="Use an unconditional bank instead of same-class references.",
    )
    p.add_argument(
        "--unconditional-ref-size",
        type=int,
        default=50000,
        help="Only used with --unconditional-bank.",
    )
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0.8, 0.6, 0.4, 0.2],
    )
    p.add_argument(
        "--patch-sizes",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9, 13, 17, 25, 31],
        help="Odd P for P x P Local Score neighborhoods in the 32x32 SD-VAE latent.",
    )
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
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_single/sdvae",
    )
    p.add_argument(
        "--force-reencode",
        action="store_true",
        help="Ignore an existing reference-latent cache.",
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
    alpha, sigma, d_alpha, d_sigma = model.interpolant(
        t_bc, path_type=path_type
    )
    det = alpha * d_sigma - sigma * d_alpha
    if torch.any(det.abs() < 1e-8):
        raise RuntimeError("Degenerate interpolant determinant.")
    z0_hat = (d_sigma * z_t - sigma * v) / det
    return z0_hat, alpha, sigma, d_alpha, d_sigma


def induced_velocity_from_xpred(
    z_t, z0_hat, alpha, sigma, d_alpha, d_sigma
):
    """
    Given a clean estimate z0_hat at state z_t, infer eps_hat from
        z_t = alpha z0_hat + sigma eps_hat
    and convert it to the corresponding velocity
        v_hat = d_alpha z0_hat + d_sigma eps_hat.
    """
    if torch.any(sigma.abs() < 1e-8):
        raise RuntimeError("sigma too small for velocity conversion.")
    eps_hat = (z_t - alpha * z0_hat) / sigma
    return d_alpha * z0_hat + d_sigma * eps_hat


def induced_score_from_xpred(z_t, z0_hat, alpha, sigma):
    # Score of a Gaussian-corruption posterior:
    # s = (alpha E[z0|zt] - zt) / sigma^2
    return (alpha * z0_hat - z_t) / (sigma * sigma)


# ---------------------------------------------------------------------
# Exact Ideal / Local Score posterior means
# ---------------------------------------------------------------------

def box_sum_2d(x, kernel_size):
    """
    Exact zero-padded P x P box SUM using integral images.
    x: [N,1,H,W]
    Returns [N,1,H,W].

    Zero-padding contributes zero squared distance. For LS this is equivalent
    to a smaller observed neighborhood at image boundaries.
    """
    p = int(kernel_size)
    if p == 1:
        return x
    if p <= 0 or p % 2 != 1:
        raise ValueError(f"Patch size must be positive odd, got {p}")

    r = p // 2
    xp = F.pad(x, (r, r, r, r), mode="constant", value=0.0)
    # leading zero row/column for rectangle-sum indexing
    ii = F.pad(xp, (1, 0, 1, 0), mode="constant", value=0.0)
    ii = ii.cumsum(dim=-2).cumsum(dim=-1)

    # Output spatial size is exactly original H,W.
    out = (
        ii[..., p:, p:]
        - ii[..., :-p, p:]
        - ii[..., p:, :-p]
        + ii[..., :-p, :-p]
    )
    return out


@torch.no_grad()
def analytic_posteriors(
    bank,
    z_t,
    alpha,
    sigma,
    patch_sizes,
):
    """
    bank: [N,C,H,W] normalized clean latents
    z_t : [1,C,H,W]

    Returns:
      ideal_mu: [1,C,H,W]
      local_mu_by_p: dict[P] -> [1,C,H,W]

    The local posterior is exact over all candidates in `bank`:
       w_i(x) proportional to
       exp(-|| z_t[Omega_x] - alpha z_i[Omega_x] ||^2 / (2 sigma^2))
    and
       mu_local(x) = sum_i w_i(x) z_i(x)
    """
    if bank.ndim != 4 or z_t.ndim != 4:
        raise ValueError("This single-image prototype expects 2D BCHW latents.")

    alpha_s = float(alpha.flatten()[0].item())
    sigma_s = float(sigma.flatten()[0].item())
    if sigma_s <= 0:
        raise ValueError(f"sigma must be >0, got {sigma_s}")

    # Per-coordinate squared residual between noisy query and every shrunken
    # clean reference latent. [N,1,H,W]
    diff2 = ((alpha_s * bank - z_t) ** 2).sum(dim=1, keepdim=True)

    # Ideal Score / global posterior.
    global_dist = diff2.flatten(1).sum(dim=1)  # [N]
    global_logits = -global_dist / (2.0 * sigma_s * sigma_s)
    global_w = torch.softmax(global_logits, dim=0)
    ideal_mu = (
        global_w[:, None, None, None] * bank
    ).sum(dim=0, keepdim=True)

    # Local Score / location-wise posterior.
    locals_by_p = {}
    for p in patch_sizes:
        local_dist = box_sum_2d(diff2, p)[:, 0]  # [N,H,W]
        local_logits = -local_dist / (2.0 * sigma_s * sigma_s)
        local_w = torch.softmax(local_logits, dim=0)  # posterior over N
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
    # "a" is reference/model prediction.
    num = (a.float() - b.float()).pow(2).flatten(1).mean(dim=1)
    den = a.float().pow(2).flatten(1).mean(dim=1)
    return float((num / (den + eps)).mean().item())


def image_psnr(a, b, eps=1e-10):
    # Just for model-vs-local agreement diagnostics, not as the proposed metric.
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
    """
    rows: list[list[(PIL.Image, label)]]
    """
    label_h = 28
    ncols = max(len(r) for r in rows)
    nrows = len(rows)
    sheet = Image.new(
        "RGB",
        (ncols * cell, nrows * (cell + label_h)),
        "white",
    )
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
            raise RuntimeError(
                f"No ImageNet training samples found for class {query_label}"
            )
        return idx

    idx = list(range(len(train_set.samples)))
    # Deterministic spread over the dataset rather than relying on dataset.small's
    # shuffled prefix, so cache identity is explicit.
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
    cache_path = cache_dir / (
        f"{exp_name}_{cond_tag}_n{nref}_legacyBN_fp16.pt"
    )

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
            raise RuntimeError(
                f"Expected SD-VAE BCHW latent, got {tuple(z_raw.shape)}"
            )
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
# Main
# ---------------------------------------------------------------------

@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    for p in args.patch_sizes:
        if p <= 0 or p % 2 != 1:
            raise ValueError(f"All patch sizes must be positive odd, got {p}")
    for t in args.times:
        if not (0.0 < t < 1.0):
            raise ValueError(f"All t must be in (0,1), got {t}")

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
        raise RuntimeError(
            f"This prototype is intended for the benchmark linear path, got {path_type}"
        )

    cfg = type("Cfg", (), raw_cfg)

    vae_cfg = OmegaConf.load(cfg.vae_config)
    vae = instantiate_from_config(vae_cfg).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    resolution = int(raw_cfg.get("resolution", 256))
    fake = torch.zeros(1, 3, resolution, resolution, device=device)
    fake_z = unwrap_encode(vae, fake)[0]
    if fake_z.ndim != 3:
        raise RuntimeError(
            f"This SD-VAE prototype expects CHW latent, got {tuple(fake_z.shape)}"
        )

    C, H, W = fake_z.shape
    if H != W:
        raise RuntimeError(f"Expected square latent, got {H}x{W}")
    for p in args.patch_sizes:
        if p > H:
            raise ValueError(
                f"Patch P={p} exceeds latent spatial size {H}. "
                "Choose P <= latent size."
            )

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

    ckpt_path = (
        Path(args.exp_path)
        / "checkpoints"
        / f"{int(args.train_steps):07d}.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    ema = ckpt["ema"]

    incompat = model.load_state_dict(ema, strict=False)
    if incompat.missing_keys:
        raise RuntimeError(
            "Missing model keys:\n" + "\n".join(incompat.missing_keys[:100])
        )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Exact legacy normalization used by the trained SiT.
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
        raise RuntimeError(
            "Train/val class_to_idx mappings differ; cannot safely class-condition."
        )

    if not (0 <= args.val_index < len(val_set)):
        raise IndexError(
            f"--val-index={args.val_index} outside [0,{len(val_set)-1}]"
        )

    img, query_label, query_name = val_set[args.val_index]
    img = img.unsqueeze(0).to(device)
    y = torch.tensor([query_label], device=device, dtype=torch.long)

    class_name = next(
        k for k, v in val_set.class_to_idx.items()
        if int(v) == int(query_label)
    )

    z0_raw = unwrap_encode(vae, img, y)
    z0 = normalize_latent(z0_raw, scale, bias)
    recon = unwrap_decode(vae, z0_raw)

    # Fixed epsilon reused for every t.
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    eps = torch.randn(
        z0.shape,
        generator=g,
        device=device,
        dtype=z0.dtype,
    )

    # ---------------------------------------------------------
    # Build exact reference bank
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

    original_pil = tensor_to_pil(img)
    recon_pil = tensor_to_pil(recon)
    original_pil.save(out_dir / "original.png")
    recon_pil.save(out_dir / "vae_reconstruction.png")

    all_rows = []
    metrics_rows = []
    best_by_t = {}

    print("============================================================")
    print("Single-image Model vs Local Score experiment")
    print(f"exp_path            : {args.exp_path}")
    print(f"checkpoint          : {ckpt_path}")
    print(f"model               : {model_name}")
    print(f"query               : {query_name}")
    print(f"query class         : {query_label} ({class_name})")
    print(f"latent shape        : {(C, H, W)}")
    print(f"tshift              : {tshift:.6f}")
    print(f"time_mode           : {args.time_mode}")
    print(f"times               : {args.times}")
    print(f"patch_sizes         : {args.patch_sizes}")
    print(f"class conditioned   : {not args.unconditional_bank}")
    print(f"reference count     : {bank.shape[0]}")
    print(f"reference cache     : {cache_path}")
    print("same epsilon across t: True")
    print("============================================================")

    # ---------------------------------------------------------
    # Main t sweep
    # ---------------------------------------------------------
    for t_raw in args.times:
        t_eff = resolve_time(
            model, t_raw, device, z0.dtype, args.time_mode
        )
        t_bc = broadcast_time(t_eff, z0)

        alpha, sigma, d_alpha, d_sigma = model.interpolant(
            t_bc, path_type=path_type
        )

        # Same forward-corruption convention as prior direct-xpred evaluator.
        z_t = alpha * z0 + sigma * eps

        # Neural one-step model prediction.
        v_model = model.inference(z_t, t_eff, y)
        model_xpred, alpha2, sigma2, da2, ds2 = model_clean_from_velocity(
            model, z_t, v_model, t_eff, path_type
        )
        model_score = induced_score_from_xpred(
            z_t, model_xpred, alpha2, sigma2
        )

        # Exact Ideal and Local Score analytic posterior means.
        ideal_xpred, local_by_p = analytic_posteriors(
            bank=bank,
            z_t=z_t,
            alpha=alpha2,
            sigma=sigma2,
            patch_sizes=args.patch_sizes,
        )

        ideal_v = induced_velocity_from_xpred(
            z_t, ideal_xpred, alpha2, sigma2, da2, ds2
        )
        ideal_score = induced_score_from_xpred(
            z_t, ideal_xpred, alpha2, sigma2
        )

        # Decode model + ideal.
        model_raw = denormalize_latent(
            model_xpred, scale, bias
        )
        ideal_raw = denormalize_latent(
            ideal_xpred, scale, bias
        )

        model_img = unwrap_decode(vae, model_raw)
        ideal_img = unwrap_decode(vae, ideal_raw)

        ttag = f"{float(t_raw):g}".replace(".", "p")
        model_pil = tensor_to_pil(model_img)
        ideal_pil = tensor_to_pil(ideal_img)

        model_pil.save(out_dir / f"t{ttag}_model_xpred.png")
        ideal_pil.save(out_dir / f"t{ttag}_ideal_xpred.png")

        # Ideal control.
        ideal_row = {
            "t_raw": float(t_raw),
            "t_effective": float(t_eff.item()),
            "patch_size": "ideal",
            "velocity_cos_model": flat_cos(v_model, ideal_v),
            "score_cos_model": flat_cos(model_score, ideal_score),
            "xpred_cos_model": flat_cos(model_xpred, ideal_xpred),
            "xpred_rel_mse_model": rel_mse(model_xpred, ideal_xpred),
            "decode_psnr_model": image_psnr(model_img, ideal_img),
        }
        metrics_rows.append(ideal_row)

        local_cells = [
            (model_pil, f"Model xpred t={t_raw:g}"),
            (ideal_pil, "Ideal"),
        ]

        best_p = None
        best_vcos = -float("inf")
        best_local_pil = None
        best_metrics = None

        print(f"\n===== t={t_raw:g} (effective={t_eff.item():.6f}) =====")
        print(
            f"{'P':>5s}  {'v_cos':>10s}  {'score_cos':>10s}  "
            f"{'xpred_cos':>10s}  {'relMSE':>10s}  {'decPSNR':>10s}"
        )

        for p in args.patch_sizes:
            local_xpred = local_by_p[int(p)]

            local_v = induced_velocity_from_xpred(
                z_t, local_xpred, alpha2, sigma2, da2, ds2
            )
            local_score = induced_score_from_xpred(
                z_t, local_xpred, alpha2, sigma2
            )

            local_raw = denormalize_latent(
                local_xpred, scale, bias
            )
            local_img = unwrap_decode(vae, local_raw)
            local_pil = tensor_to_pil(local_img)
            local_pil.save(
                out_dir / f"t{ttag}_local_P{int(p):02d}.png"
            )

            row = {
                "t_raw": float(t_raw),
                "t_effective": float(t_eff.item()),
                "patch_size": int(p),
                "velocity_cos_model": flat_cos(v_model, local_v),
                "score_cos_model": flat_cos(model_score, local_score),
                "xpred_cos_model": flat_cos(model_xpred, local_xpred),
                "xpred_rel_mse_model": rel_mse(model_xpred, local_xpred),
                "decode_psnr_model": image_psnr(model_img, local_img),
            }
            metrics_rows.append(row)

            print(
                f"{p:5d}  "
                f"{row['velocity_cos_model']:10.6f}  "
                f"{row['score_cos_model']:10.6f}  "
                f"{row['xpred_cos_model']:10.6f}  "
                f"{row['xpred_rel_mse_model']:10.6f}  "
                f"{row['decode_psnr_model']:10.4f}"
            )

            local_cells.append((local_pil, f"Local P={p}"))

            # Following the paper's scale-calibration idea:
            # choose P that maximizes cosine between neural and analytic
            # denoising prediction. Our SiT predicts velocity, so v-cos is
            # the primary scale-selection statistic.
            if row["velocity_cos_model"] > best_vcos:
                best_vcos = row["velocity_cos_model"]
                best_p = int(p)
                best_local_pil = local_pil.copy()
                best_metrics = row.copy()

            del local_xpred, local_v, local_score, local_raw, local_img

        best_by_t[str(float(t_raw))] = {
            "patch_size": best_p,
            **best_metrics,
        }

        print(
            f"BEST P={best_p} by velocity cosine "
            f"({best_vcos:.6f})"
        )

        # Per-t sheet: model, ideal, all local scales.
        make_contact_sheet(
            [local_cells],
            out_dir / f"t{ttag}_all_local_scales.png",
            cell=256,
        )

        # Main one-image comparison row.
        all_rows.append(
            [
                (original_pil, f"Original ({class_name})"),
                (recon_pil, "VAE recon"),
                (model_pil, f"Model xpred t={t_raw:g}"),
                (best_local_pil, f"Local best P={best_p}"),
                (ideal_pil, "Ideal"),
            ]
        )

        del (
            z_t,
            v_model,
            model_xpred,
            model_score,
            ideal_xpred,
            ideal_v,
            ideal_score,
            model_raw,
            ideal_raw,
            model_img,
            ideal_img,
            local_by_p,
        )
        torch.cuda.empty_cache()

    # One contact sheet requested for qualitative inspection.
    comparison_path = out_dir / "comparison_all_t_best_local.png"
    make_contact_sheet(all_rows, comparison_path, cell=256)

    # CSV
    csv_path = out_dir / "metrics.csv"
    fields = [
        "t_raw",
        "t_effective",
        "patch_size",
        "velocity_cos_model",
        "score_cos_model",
        "xpred_cos_model",
        "xpred_rel_mse_model",
        "decode_psnr_model",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metrics_rows)

    # JSON
    result = {
        "paper": "Kamb & Ganguli, arXiv:2412.20292",
        "experiment": "single_image_model_vs_local_score",
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
        "times": [float(x) for x in args.times],
        "patch_sizes": [int(x) for x in args.patch_sizes],
        "same_forward_noise_across_times": True,
        "seed": int(args.seed),
        "reference_bank": {
            "class_conditioned": not args.unconditional_bank,
            "count": int(bank.shape[0]),
            "cache": str(cache_path),
        },
        "scale_selection": (
            "For each t choose P maximizing cosine between SiT velocity "
            "prediction and Local-Score-induced velocity."
        ),
        "best_by_t": best_by_t,
        "rows": metrics_rows,
        "comparison_image": str(comparison_path),
    }

    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(result, indent=2))

    print("\n============================================================")
    print("DONE")
    print(f"comparison image : {comparison_path}")
    print(f"metrics CSV      : {csv_path}")
    print(f"metrics JSON     : {json_path}")
    print("Best locality scale by t:")
    for t, m in best_by_t.items():
        print(
            f"  t={t}: P={m['patch_size']}, "
            f"v_cos={m['velocity_cos_model']:.6f}, "
            f"xpred_cos={m['xpred_cos_model']:.6f}, "
            f"decode_PSNR={m['decode_psnr_model']:.3f}"
        )
    print("============================================================")


if __name__ == "__main__":
    main(parse_args())
