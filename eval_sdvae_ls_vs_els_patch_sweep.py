#!/usr/bin/env python3
"""
Single-image SD-VAE comparison:
Model x_pred vs LS x_pred vs ELS x_pred.

This is the direct follow-up to eval_sdvae_model_vs_local_score.py.

LS:
  for query position x, posterior candidates are the CO-LOCATED patches
  from every reference latent.

ELS:
  for query position x, posterior candidates are ALL valid P x P patches
  from every reference latent, regardless of their original location.
  The posterior mean returns the CENTER latent vector of the selected patch.

For P=1, ELS is especially simple:
  every output latent location can borrow from every latent location in every
  reference image.

The ELS computation is exact over the supplied reference bank, but streamed in
reference chunks to avoid materializing the enormous [query_pos, all_patches]
matrix at once.

Default experiment requested here:
  - SD-VAE + SiT-XL 400k
  - val-index 0
  - 50k cross-class ImageNet-train reference bank
  - t = 0.4, 0.2
  - P = 1
  - same noise seed = 12345
  - compare decoded Model / LS / ELS x_preds
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms

from ifid.dataset import ImageNetValDataset
from ifid.sit.sit import SiT_models
from ifid.vae.utils import instantiate_from_config

from eval_sdvae_model_vs_local_score import (
    unwrap_encode,
    unwrap_decode,
    normalize_latent,
    denormalize_latent,
    resolve_time,
    broadcast_time,
    model_clean_from_velocity,
    induced_velocity_from_xpred,
    induced_score_from_xpred,
    analytic_posteriors,
    flat_cos,
    rel_mse,
    image_psnr,
    tensor_to_pil,
    make_contact_sheet,
    choose_reference_indices,
    build_or_load_bank,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--exp-path",
        default="/share/xutongda/REPA-E/exps/sit-xl-sdvae-400k",
    )
    p.add_argument("--train-steps", type=int, default=400000)

    p.add_argument("--dataset", default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--val-index", type=int, default=0)

    p.add_argument("--unconditional-bank", action="store_true", default=True)
    p.add_argument("--unconditional-ref-size", type=int, default=50000)
    p.add_argument("--ref-per-class", type=int, default=1000)

    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--times", type=float, nargs="+", default=[0.4, 0.2])
    p.add_argument("--patch-sizes", type=int, nargs="+", default=[1])

    p.add_argument(
        "--time-mode",
        choices=["shifted_raw", "fixed_effective"],
        default="shifted_raw",
    )
    p.add_argument("--seed", type=int, default=12345)

    p.add_argument(
        "--els-ref-chunk",
        type=int,
        default=32,
        help="Number of reference images processed per ELS streaming chunk.",
    )

    p.add_argument(
        "--cache-dir",
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_single/cache",
    )
    p.add_argument(
        "--out-dir",
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_single/"
                "sdvae_model_vs_ls_vs_els_crossclass50k_p1",
    )
    p.add_argument(
        "--ls-els-p-sweep-only",
        action="store_true",
        help="Compare LS vs exact ELS directly across several patch sizes and exit.",
    )
    p.add_argument(
        "--p-sweep-time",
        type=float,
        default=0.4,
        help="Noise time used for the LS-vs-ELS patch-size sweep.",
    )
    p.add_argument(
        "--p-sweep-ref-size",
        type=int,
        default=1000,
        help="Use only the first N reference latents for the patch-size sweep.",
    )
    p.add_argument(
        "--p-sweep-patches",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7],
        help="Odd patch sizes to compare.",
    )
    p.add_argument("--force-reencode", action="store_true")
    return p.parse_args()


@torch.no_grad()
def extract_query_patches_circular(z_t: torch.Tensor, p: int):
    """
    z_t: [1,C,H,W]
    returns query patches [H*W, C*p*p], centered at every spatial location.

    Following the paper's ELS implementation, query patches use circular
    padding so the operator is translation equivariant.
    """
    if z_t.ndim != 4 or z_t.shape[0] != 1:
        raise ValueError(f"Expected [1,C,H,W], got {tuple(z_t.shape)}")

    if p == 1:
        return z_t[0].permute(1, 2, 0).reshape(-1, z_t.shape[1]).contiguous()

    d = p // 2
    zp = F.pad(z_t, (d, d, d, d), mode="circular")
    # [1, C*p*p, H*W] -> [H*W, C*p*p]
    q = F.unfold(zp, kernel_size=p, stride=1, padding=0)
    return q[0].T.contiguous()


@torch.no_grad()
def extract_reference_patches_and_centers(bank_chunk: torch.Tensor, p: int):
    """
    bank_chunk: [B,C,H,W], clean normalized references.

    Returns:
      patches : [M, C*p*p]
      centers : [M, C]

    For p>1, use every VALID P x P patch of each reference image and return
    its center pixel. This mirrors Kamb & Ganguli's LocalEquivScoreModule:
    translated patches from every possible valid reference location.

    For p=1, every spatial latent vector is a candidate.
    """
    B, C, H, W = bank_chunk.shape

    if p == 1:
        # [B,H,W,C] -> [B*H*W,C]
        centers = bank_chunk.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        return centers, centers

    d = p // 2
    # [B,C*p*p,Lr]
    patches = F.unfold(bank_chunk, kernel_size=p, stride=1, padding=0)
    Lr = patches.shape[-1]
    patches = patches.transpose(1, 2).reshape(B * Lr, C * p * p).contiguous()

    # Centers aligned with valid P x P windows.
    centers = bank_chunk[:, :, d:H-d, d:W-d]
    centers = centers.permute(0, 2, 3, 1).reshape(B * Lr, C).contiguous()

    return patches, centers


@torch.no_grad()
def els_posterior_mean(
    bank: torch.Tensor,
    z_t: torch.Tensor,
    alpha,
    sigma,
    p: int,
    ref_chunk: int = 32,
):
    """
    Exact ELS posterior mean over ALL translated valid patches in `bank`.

    At each query location q:
      w_j(q) ∝ exp(
          - || qpatch - alpha * refpatch_j ||^2 / (2 sigma^2)
      )
      mu(q) = Σ_j w_j(q) * ref_center_j

    The candidate dimension j runs over:
      reference image × valid spatial patch location.

    Streaming online log-sum-exp keeps memory bounded.
    """
    if bank.ndim != 4 or z_t.ndim != 4:
        raise ValueError("Expected BCHW bank/query")
    if z_t.shape[0] != 1:
        raise ValueError("This qualitative script expects one query image.")
    if p <= 0 or p % 2 != 1:
        raise ValueError(f"P must be positive odd, got {p}")

    alpha_s = float(alpha.flatten()[0].item())
    sigma_s = float(sigma.flatten()[0].item())
    if sigma_s <= 0:
        raise ValueError(f"sigma must be >0, got {sigma_s}")

    _, C, H, W = z_t.shape
    if p > min(H, W):
        raise ValueError(f"P={p} exceeds latent size {H}x{W}")

    # Lq = H*W query patch descriptors.
    qpatch = extract_query_patches_circular(z_t, p).float()
    Lq, D = qpatch.shape
    q_sq = (qpatch * qpatch).sum(dim=1)  # [Lq]

    running_max = torch.full(
        (Lq,), -float("inf"), device=z_t.device, dtype=torch.float32
    )
    running_den = torch.zeros(
        (Lq,), device=z_t.device, dtype=torch.float32
    )
    running_num = torch.zeros(
        (Lq, C), device=z_t.device, dtype=torch.float32
    )

    denom = 2.0 * sigma_s * sigma_s

    for s in range(0, bank.shape[0], ref_chunk):
        e = min(s + ref_chunk, bank.shape[0])
        refs = bank[s:e].float()

        rpatch, rcenter = extract_reference_patches_and_centers(refs, p)
        if rpatch.shape[1] != D:
            raise RuntimeError(
                f"Patch dim mismatch q={D}, ref={rpatch.shape[1]}"
            )

        # ||q - alpha*r||^2
        r_sq = (rpatch * rpatch).sum(dim=1)
        cross = qpatch @ rpatch.T
        dist = (
            q_sq[:, None]
            - 2.0 * alpha_s * cross
            + (alpha_s * alpha_s) * r_sq[None, :]
        ).clamp_min_(0.0)

        logits = -dist / denom

        chunk_max = logits.max(dim=1).values
        new_max = torch.maximum(running_max, chunk_max)

        old_scale = torch.exp(running_max - new_max)
        old_scale = torch.where(
            torch.isfinite(running_max),
            old_scale,
            torch.zeros_like(old_scale),
        )

        w = torch.exp(logits - new_max[:, None])

        running_den = running_den * old_scale + w.sum(dim=1)
        running_num = (
            running_num * old_scale[:, None]
            + w @ rcenter
        )
        running_max = new_max

        del refs, rpatch, rcenter, r_sq, cross, dist, logits, chunk_max, new_max, old_scale, w

    mu = running_num / running_den.clamp_min(1e-30)[:, None]
    # [H*W,C] -> [1,C,H,W]
    mu = mu.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0).contiguous()
    return mu


def tensor_pair_metrics(a: torch.Tensor, b: torch.Tensor):
    a = a.float()
    b = b.float()
    d = a - b
    denom = a.square().mean().clamp_min(1e-30)
    mse = d.square().mean()
    return {
        "max_abs": float(d.abs().max().item()),
        "mean_abs": float(d.abs().mean().item()),
        "rmse": float(mse.sqrt().item()),
        "rel_mse": float((mse / denom).item()),
        "cos": float(flat_cos(a, b)),
    }


@torch.no_grad()
def run_ls_els_patch_sweep(
    *,
    bank: torch.Tensor,
    z_t: torch.Tensor,
    alpha,
    sigma,
    vae,
    scale,
    bias,
    patch_sizes,
    ref_chunk,
    ref_size,
    out_dir: Path,
):
    n = min(int(ref_size), int(bank.shape[0]))
    if n <= 0:
        raise ValueError(f"p-sweep-ref-size must be > 0, got {ref_size}")
    b = bank[:n].contiguous()

    print("\n" + "=" * 80)
    print("LS vs EXACT ELS PATCH-SIZE SWEEP")
    print(f"references : {n}")
    print(f"latent     : {tuple(z_t.shape[1:])}")
    print(f"P values   : {patch_sizes}")
    print("=" * 80)

    # Compute all LS posteriors in one call.
    _ideal, ls_by_p = analytic_posteriors(
        bank=b,
        z_t=z_t,
        alpha=alpha,
        sigma=sigma,
        patch_sizes=patch_sizes,
    )

    rows = []
    sheet_rows = []
    for p in patch_sizes:
        p = int(p)
        print(f"\n[P={p}] exact ELS ...", flush=True)
        ls_x = ls_by_p[p]
        els_x = els_posterior_mean(
            bank=b,
            z_t=z_t,
            alpha=alpha,
            sigma=sigma,
            p=p,
            ref_chunk=ref_chunk,
        )

        latent = tensor_pair_metrics(ls_x, els_x)

        ls_img = unwrap_decode(
            vae, denormalize_latent(ls_x, scale, bias)
        )
        els_img = unwrap_decode(
            vae, denormalize_latent(els_x, scale, bias)
        )
        image = tensor_pair_metrics(ls_img, els_img)
        decode_psnr = image_psnr(ls_img, els_img)

        row = {
            "patch_size": p,
            "latent_ls_vs_els": latent,
            "decoded_ls_vs_els": image,
            "decode_psnr": float(decode_psnr),
        }
        rows.append(row)

        print(
            f"P={p}: "
            f"latent rel_mse={latent['rel_mse']:.6e}, "
            f"latent cos={latent['cos']:.9f}, "
            f"decode PSNR={decode_psnr:.3f} dB, "
            f"decode rel_mse={image['rel_mse']:.6e}"
        )

        ls_pil = tensor_to_pil(ls_img)
        els_pil = tensor_to_pil(els_img)
        ls_pil.save(out_dir / f"P{p:02d}_LS.png")
        els_pil.save(out_dir / f"P{p:02d}_ELS.png")
        sheet_rows.append([
            (ls_pil, f"LS P={p}"),
            (els_pil, f"ELS P={p}"),
        ])

        del ls_x, els_x, ls_img, els_img
        torch.cuda.empty_cache()

    sheet_path = out_dir / "comparison_LS_vs_ELS_patch_sweep.png"
    make_contact_sheet(sheet_rows, sheet_path, cell=256)

    return {
        "reference_count": n,
        "patch_sizes": [int(p) for p in patch_sizes],
        "rows": rows,
        "comparison_image": str(sheet_path),
    }


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    for p in args.patch_sizes:
        if p <= 0 or p % 2 != 1:
            raise ValueError(f"All P must be positive odd, got {p}")

    # ------------------------------------------------------------
    # Load VAE + trained SiT only for qualitative comparison.
    # ------------------------------------------------------------
    raw_cfg = json.loads((Path(args.exp_path) / "args.json").read_text())
    if str(raw_cfg.get("prediction", "v")).lower() != "v":
        raise RuntimeError("Expected legacy prediction=v")
    if "prediction_internal" in raw_cfg:
        raise RuntimeError("Refusing prediction_internal checkpoint")

    path_type = str(raw_cfg.get("path_type", "linear")).lower()
    if path_type != "linear":
        raise RuntimeError(f"Expected linear path, got {path_type}")

    cfg = type("Cfg", (), raw_cfg)

    vae = instantiate_from_config(OmegaConf.load(cfg.vae_config)).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    resolution = int(raw_cfg.get("resolution", 256))
    fake = torch.zeros(1, 3, resolution, resolution, device=device)
    fake_z = unwrap_encode(vae, fake)[0]
    if fake_z.ndim != 3:
        raise RuntimeError(f"Expected CHW latent, got {tuple(fake_z.shape)}")

    C, H, W = fake_z.shape
    tshift = math.sqrt(float(fake_z.numel()) / 4096.0)

    model_name = str(cfg.model)
    if model_name not in SiT_models:
        alt = model_name.replace("SiT_", "SiT-")
        if alt in SiT_models:
            model_name = alt
        else:
            raise KeyError(model_name)

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

    ckpt_path = Path(args.exp_path) / "checkpoints" / f"{args.train_steps:07d}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    ema = ckpt["ema"]

    incompat = model.load_state_dict(ema, strict=False)
    if incompat.missing_keys:
        raise RuntimeError("Missing model keys:\n" + "\n".join(incompat.missing_keys[:100]))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    running_mean = ema["bn.running_mean"].view(1, C, 1, 1).float()
    running_var = ema["bn.running_var"].view(1, C, 1, 1).float()
    scale = running_var.rsqrt().to(device)
    bias = running_mean.to(device)
    del ckpt, ema

    # ------------------------------------------------------------
    # Query + bank.
    # ------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    val_set = ImageNetValDataset(args.dataset, transform, -1)
    train_set = ImageNetValDataset(args.dataset_ref, transform, -1)

    img, query_label, query_name = val_set[args.val_index]
    img = img.unsqueeze(0).to(device)
    y = torch.tensor([query_label], device=device, dtype=torch.long)

    class_name = next(
        k for k, v in val_set.class_to_idx.items()
        if int(v) == int(query_label)
    )

    z0_raw = unwrap_encode(vae, img, y)
    z0 = normalize_latent(z0_raw, scale, bias).float()
    recon = unwrap_decode(vae, z0_raw)

    g = torch.Generator(device=device).manual_seed(args.seed)
    eps = torch.randn(
        z0.shape, generator=g, device=device, dtype=z0.dtype
    ).float()

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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.ls_els_p_sweep_only:
        t_raw = float(args.p_sweep_time)
        t_eff = resolve_time(model, t_raw, device, z0.dtype, args.time_mode)
        t_bc = broadcast_time(t_eff, z0)
        alpha, sigma, _da, _ds = model.interpolant(t_bc, path_type=path_type)
        z_t = alpha * z0 + sigma * eps

        result = run_ls_els_patch_sweep(
            bank=bank,
            z_t=z_t,
            alpha=alpha,
            sigma=sigma,
            vae=vae,
            scale=scale,
            bias=bias,
            patch_sizes=args.p_sweep_patches,
            ref_chunk=args.els_ref_chunk,
            ref_size=args.p_sweep_ref_size,
            out_dir=out,
        )
        result.update({
            "t_raw": t_raw,
            "t_effective": float(t_eff.item()),
            "cache": str(cache_path),
        })
        path = out / "ls_vs_els_patch_sweep.json"
        path.write_text(json.dumps(result, indent=2))
        print("\nsaved:", path)
        print("comparison:", result["comparison_image"])
        return

    original_pil = tensor_to_pil(img)
    recon_pil = tensor_to_pil(recon)
    original_pil.save(out / "original.png")
    recon_pil.save(out / "vae_reconstruction.png")

    print("=" * 72)
    print("Model vs LS vs ELS")
    print("query             :", query_name)
    print("class             :", query_label, class_name)
    print("latent            :", (C, H, W))
    print("reference count   :", bank.shape[0])
    print("cross-class       :", args.unconditional_bank)
    print("times             :", args.times)
    print("P                 :", args.patch_sizes)
    print("ELS ref chunk     :", args.els_ref_chunk)
    print("cache             :", cache_path)
    print("=" * 72)

    metrics = []
    sheet_rows = []

    for t_raw in args.times:
        t_eff = resolve_time(model, t_raw, device, z0.dtype, args.time_mode)
        t_bc = broadcast_time(t_eff, z0)
        alpha, sigma, da, ds = model.interpolant(t_bc, path_type=path_type)

        z_t = alpha * z0 + sigma * eps

        # Model.
        v_model = model.inference(z_t, t_eff, y)
        model_xpred, alpha2, sigma2, da2, ds2 = model_clean_from_velocity(
            model, z_t, v_model, t_eff, path_type
        )
        model_score = induced_score_from_xpred(
            z_t, model_xpred, alpha2, sigma2
        )
        model_img = unwrap_decode(
            vae, denormalize_latent(model_xpred, scale, bias)
        )
        model_pil = tensor_to_pil(model_img)

        # LS baseline for direct visual comparison.
        _ideal, ls_by_p = analytic_posteriors(
            bank=bank,
            z_t=z_t,
            alpha=alpha2,
            sigma=sigma2,
            patch_sizes=args.patch_sizes,
        )

        for p in args.patch_sizes:
            ls_xpred = ls_by_p[int(p)]

            # Exact ELS over all translated patches in the 50k bank.
            els_xpred = els_posterior_mean(
                bank=bank,
                z_t=z_t,
                alpha=alpha2,
                sigma=sigma2,
                p=int(p),
                ref_chunk=args.els_ref_chunk,
            )

            ls_img = unwrap_decode(
                vae, denormalize_latent(ls_xpred, scale, bias)
            )
            els_img = unwrap_decode(
                vae, denormalize_latent(els_xpred, scale, bias)
            )

            ls_pil = tensor_to_pil(ls_img)
            els_pil = tensor_to_pil(els_img)

            ttag = f"{float(t_raw):g}".replace(".", "p")
            model_pil.save(out / f"t{ttag}_model_xpred.png")
            ls_pil.save(out / f"t{ttag}_LS_P{p:02d}_xpred.png")
            els_pil.save(out / f"t{ttag}_ELS_P{p:02d}_xpred.png")

            ls_v = induced_velocity_from_xpred(
                z_t, ls_xpred, alpha2, sigma2, da2, ds2
            )
            els_v = induced_velocity_from_xpred(
                z_t, els_xpred, alpha2, sigma2, da2, ds2
            )
            ls_score = induced_score_from_xpred(
                z_t, ls_xpred, alpha2, sigma2
            )
            els_score = induced_score_from_xpred(
                z_t, els_xpred, alpha2, sigma2
            )

            row = {
                "t_raw": float(t_raw),
                "t_effective": float(t_eff.item()),
                "patch_size": int(p),

                "ls_velocity_cos_model": flat_cos(v_model, ls_v),
                "els_velocity_cos_model": flat_cos(v_model, els_v),

                "ls_score_cos_model": flat_cos(model_score, ls_score),
                "els_score_cos_model": flat_cos(model_score, els_score),

                "ls_xpred_cos_model": flat_cos(model_xpred, ls_xpred),
                "els_xpred_cos_model": flat_cos(model_xpred, els_xpred),

                "ls_rel_mse_model": rel_mse(model_xpred, ls_xpred),
                "els_rel_mse_model": rel_mse(model_xpred, els_xpred),

                "ls_decode_psnr_model": image_psnr(model_img, ls_img),
                "els_decode_psnr_model": image_psnr(model_img, els_img),
            }
            metrics.append(row)

            print(
                f"t={t_raw:g} P={p}: "
                f"vcos LS={row['ls_velocity_cos_model']:.6f}, "
                f"ELS={row['els_velocity_cos_model']:.6f}; "
                f"xpred_cos LS={row['ls_xpred_cos_model']:.6f}, "
                f"ELS={row['els_xpred_cos_model']:.6f}"
            )

            sheet_rows.append([
                (original_pil, "Original"),
                (recon_pil, "VAE recon"),
                (model_pil, f"decode(model x_pred) t={t_raw:g}"),
                (ls_pil, f"decode(LS x_pred) P={p}"),
                (els_pil, f"decode(ELS x_pred) P={p}"),
            ])

            del ls_xpred, els_xpred, ls_img, els_img
            torch.cuda.empty_cache()

        del z_t, v_model, model_xpred, model_img, ls_by_p

    sheet_path = out / "comparison_model_LS_ELS.png"
    make_contact_sheet(sheet_rows, sheet_path, cell=256)

    result = {
        "paper": "Kamb & Ganguli, ICML 2025 / arXiv:2412.20292",
        "experiment": "single_image_model_vs_LS_vs_ELS",
        "query": {
            "val_index": args.val_index,
            "filename": query_name,
            "class_index": int(query_label),
            "class_name": class_name,
        },
        "reference_bank": {
            "class_conditioned": not args.unconditional_bank,
            "count": int(bank.shape[0]),
            "cache": str(cache_path),
        },
        "els_definition": (
            "For every query location, posterior candidates are all valid P x P "
            "patches at all spatial positions from all reference latents; the "
            "posterior mean returns the candidate patch center."
        ),
        "els_boundary": (
            "Query patches use circular padding; reference candidate patches "
            "are valid windows, matching the paper's LocalEquivScoreModule."
        ),
        "times": args.times,
        "patch_sizes": args.patch_sizes,
        "seed": args.seed,
        "metrics": metrics,
        "comparison_image": str(sheet_path),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print("\nDONE")
    print("comparison:", sheet_path)
    print("metrics:", out / "metrics.json")


if __name__ == "__main__":
    main(parse_args())
