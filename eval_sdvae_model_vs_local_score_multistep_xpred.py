#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision import transforms

from ifid.dataset import ImageNetValDataset
from ifid.sit.sit import SiT_models
from ifid.vae.utils import instantiate_from_config

# Reuse the already-tested single-step implementation.
from eval_sdvae_model_vs_local_score import (
    analytic_posteriors, build_or_load_bank, choose_reference_indices,
    denormalize_latent, flat_cos, image_psnr, induced_velocity_from_xpred,
    make_contact_sheet, model_clean_from_velocity, normalize_latent,
    rel_mse, resolve_time, tensor_to_pil, unwrap_decode, unwrap_encode,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-path", default="/share/xutongda/REPA-E/exps/sit-xl-sdvae-400k")
    p.add_argument("--train-steps", type=int, default=400000)
    p.add_argument("--dataset", default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--val-index", type=int, default=0)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--unconditional-bank", dest="unconditional_bank", action="store_true")
    g.add_argument("--same-class-bank", dest="unconditional_bank", action="store_false")
    p.set_defaults(unconditional_bank=True)

    p.add_argument("--unconditional-ref-size", type=int, default=50000)
    p.add_argument("--ref-per-class", type=int, default=1000)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--t-start", type=float, default=0.8)
    p.add_argument("--nfe", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--patch-sizes", type=int, nargs="+",
                   default=[1, 3, 5, 7, 9, 13, 17, 25, 31])
    p.add_argument("--p-mode", choices=["oracle", "fixed"], default="oracle")

    # Fixed schedule, only used with --p-mode fixed.
    p.add_argument("--fixed-high-threshold", type=float, default=0.7)
    p.add_argument("--fixed-low-threshold", type=float, default=0.3)
    p.add_argument("--fixed-high-p", type=int, default=7)
    p.add_argument("--fixed-mid-p", type=int, default=3)
    p.add_argument("--fixed-low-p", type=int, default=1)

    p.add_argument("--time-mode", choices=["shifted_raw", "fixed_effective"],
                   default="shifted_raw")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--cache-dir",
                   default="/share/xutongda/REPA-E/exps_interpolate/local_score_single/cache")
    p.add_argument("--out-dir",
                   default="/share/xutongda/REPA-E/exps_interpolate/local_score_single/"
                           "sdvae_model_vs_local_multistep_crossclass50k_oracle")
    p.add_argument("--force-reencode", action="store_true")
    p.add_argument("--save-intermediate", action="store_true")
    return p.parse_args()


def raw_grid(t_start, nfe):
    # 0.8,0.4 for NFE=2; 0.8,0.6,0.4,0.2 for NFE=4; etc.
    return [float(t_start) * (nfe - k) / nfe for k in range(nfe)]


def fixed_p(t, a):
    if t >= a.fixed_high_threshold:
        return a.fixed_high_p
    if t >= a.fixed_low_threshold:
        return a.fixed_mid_p
    return a.fixed_low_p


@torch.no_grad()
def decode_norm(vae, z, scale, bias):
    return unwrap_decode(vae, denormalize_latent(z, scale, bias))


@torch.no_grad()
def local_choice(model, bank, z, t_eff, y, alpha, sigma, da, ds,
                 patch_sizes, mode, p_fixed):
    _, by_p = analytic_posteriors(
        bank=bank, z_t=z, alpha=alpha, sigma=sigma, patch_sizes=patch_sizes
    )

    # Proper oracle target: SiT velocity at the SAME Local trajectory state.
    v_target = model.inference(z, t_eff, y)

    v_by_p, cos_by_p = {}, {}
    for p in patch_sizes:
        xp = by_p[int(p)]
        vv = induced_velocity_from_xpred(z, xp, alpha, sigma, da, ds)
        v_by_p[int(p)] = vv
        cos_by_p[int(p)] = flat_cos(v_target, vv)

    if mode == "oracle":
        p = max(cos_by_p, key=cos_by_p.get)
    else:
        p = int(p_fixed)
        if p not in by_p:
            raise ValueError(f"fixed P={p} not in patch sizes {patch_sizes}")

    return p, by_p[p], v_by_p[p], cos_by_p


@torch.no_grad()
def main(a):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    nfes = sorted(set(int(x) for x in a.nfe))
    if any(n < 2 for n in nfes):
        raise ValueError("NFE must be >=2")

    ps = [int(x) for x in a.patch_sizes]
    if any(p <= 0 or p % 2 == 0 for p in ps):
        raise ValueError("patch sizes must be positive odd integers")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(a.seed)

    # ----- load VAE / SiT -----
    raw = json.loads((Path(a.exp_path) / "args.json").read_text())
    if str(raw.get("prediction", "v")).lower() != "v":
        raise RuntimeError("expected prediction=v")
    if "prediction_internal" in raw:
        raise RuntimeError("refusing checkpoint with prediction_internal")
    path_type = str(raw.get("path_type", "linear")).lower()
    if path_type != "linear":
        raise RuntimeError(f"expected linear path, got {path_type}")

    cfg = type("Cfg", (), raw)
    vae = instantiate_from_config(OmegaConf.load(cfg.vae_config)).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    resolution = int(raw.get("resolution", 256))
    fake_z = unwrap_encode(vae, torch.zeros(1, 3, resolution, resolution, device=device))[0]
    C, H, W = fake_z.shape
    if H != W or max(ps) > H:
        raise RuntimeError(f"bad latent/patch sizes: latent={C,H,W}, maxP={max(ps)}")

    tshift = math.sqrt(float(fake_z.numel()) / 4096.0)
    model_name = str(cfg.model)
    if model_name not in SiT_models:
        alt = model_name.replace("SiT_", "SiT-")
        if alt in SiT_models:
            model_name = alt
        else:
            raise KeyError(model_name)

    model = SiT_models[model_name](
        input_size=H, in_channels=C, num_classes=cfg.num_classes,
        class_dropout_prob=cfg.cfg_prob, bn_momentum=cfg.bn_momentum,
        tshift=tshift, fused_attn=raw.get("fused_attn", True),
        qk_norm=raw.get("qk_norm", False),
    ).to(device)

    ckpt_path = Path(a.exp_path) / "checkpoints" / f"{a.train_steps:07d}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    ema = ckpt["ema"]
    incompat = model.load_state_dict(ema, strict=False)
    if incompat.missing_keys:
        raise RuntimeError("missing keys:\n" + "\n".join(incompat.missing_keys[:100]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    scale = ema["bn.running_var"].view(1, C, 1, 1).float().rsqrt().to(device)
    bias = ema["bn.running_mean"].view(1, C, 1, 1).float().to(device)
    del ckpt, ema

    # ----- data / query -----
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val = ImageNetValDataset(a.dataset, tfm, -1)
    train = ImageNetValDataset(a.dataset_ref, tfm, -1)
    if val.class_to_idx != train.class_to_idx:
        raise RuntimeError("train/val class mappings differ")

    img, label, name = val[a.val_index]
    img = img.unsqueeze(0).to(device)
    y = torch.tensor([label], device=device, dtype=torch.long)
    class_name = next(k for k, v in val.class_to_idx.items() if int(v) == int(label))

    z0_raw = unwrap_encode(vae, img, y)
    z0 = normalize_latent(z0_raw, scale, bias).float()
    recon = unwrap_decode(vae, z0_raw)

    gen = torch.Generator(device=device).manual_seed(a.seed)
    eps = torch.randn(z0.shape, generator=gen, device=device, dtype=z0.dtype).float()

    # ----- reference bank -----
    refs = choose_reference_indices(
        train_set=train, query_label=label,
        class_conditioned=(not a.unconditional_bank),
        ref_per_class=a.ref_per_class,
        unconditional_ref_size=a.unconditional_ref_size,
    )
    bank_cpu, cache_path = build_or_load_bank(
        args=a, vae=vae, train_set=train, ref_indices=refs,
        query_label=label, scale=scale, bias=bias, device=device,
    )
    bank = bank_cpu.to(device=device, dtype=torch.float32)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    original_pil, recon_pil = tensor_to_pil(img), tensor_to_pil(recon)
    original_pil.save(out / "original.png")
    recon_pil.save(out / "vae_reconstruction.png")

    print("="*60)
    print("Multi-step Model vs Local Score")
    print("query:", name, label, class_name)
    print("NFE:", nfes)
    print("P mode:", a.p_mode, "| P candidates:", ps)
    print("cross-class:", a.unconditional_bank, "| refs:", bank.shape[0])
    print("cache:", cache_path)
    print("="*60)

    # Common z_0.8 for every NFE.
    t0_eff = resolve_time(model, a.t_start, device, z0.dtype, a.time_mode)
    t0_bc = t0_eff.view(t0_eff.shape[0], *([1]*(z0.ndim-1)))
    alpha0, sigma0, _, _ = model.interpolant(t0_bc, path_type=path_type)
    z_start = alpha0 * z0 + sigma0 * eps

    rows, summaries, summary_sheet = [], {}, []

    for nfe in nfes:
        times = raw_grid(a.t_start, nfe)
        z_m, z_l = z_start.clone(), z_start.clone()
        p_path, traj_sheet = [], []

        print("\nNFE", nfe, ":", " -> ".join(f"{t:g}" for t in times))

        for k, t in enumerate(times):
            te = resolve_time(model, t, device, z0.dtype, a.time_mode)
            tbc = te.view(te.shape[0], *([1]*(z0.ndim-1)))
            alpha, sigma, da, ds = model.interpolant(tbc, path_type=path_type)

            # Model trajectory.
            v_m = model.inference(z_m, te, y)
            xp_m, _, _, _, _ = model_clean_from_velocity(
                model, z_m, v_m, te, path_type
            )

            # Local trajectory.
            pfix = fixed_p(t, a)
            chosen_p, xp_l, v_l, all_cos = local_choice(
                model, bank, z_l, te, y, alpha, sigma, da, ds,
                ps, a.p_mode, pfix
            )
            p_path.append(chosen_p)

            r = {
                "nfe": nfe, "step_index": k, "t_raw": t,
                "next_t_raw": "", "is_final": k == len(times)-1,
                "p_mode": a.p_mode, "chosen_p": chosen_p,
                "oracle_same_state_vcos": all_cos[chosen_p],
                "state_cos_before": flat_cos(z_m, z_l),
                "state_rel_mse_before": rel_mse(z_m, z_l),
                "velocity_cos_cross_state": flat_cos(v_m, v_l),
                "xpred_cos_cross_state": flat_cos(xp_m, xp_l),
                "state_cos_after": "", "state_rel_mse_after": "",
                "all_p_vcos": json.dumps(
                    {str(p): float(c) for p, c in all_cos.items()},
                    sort_keys=True
                ),
            }

            if k == len(times)-1:
                final_xp_m, final_xp_l = xp_m, xp_l
                rows.append(r)
                break

            next_t = times[k+1]
            next_te = resolve_time(model, next_t, device, z0.dtype, a.time_mode)
            dt = float((next_te - te).item())
            z_m = z_m + dt * v_m
            z_l = z_l + dt * v_l

            r["next_t_raw"] = next_t
            r["state_cos_after"] = flat_cos(z_m, z_l)
            r["state_rel_mse_after"] = rel_mse(z_m, z_l)
            rows.append(r)

            print(
                f"  {t:g}->{next_t:g}  P={chosen_p}  "
                f"same-state-vcos={all_cos[chosen_p]:.6f}  "
                f"state-cos={r['state_cos_after']:.6f}"
            )

            if a.save_intermediate:
                # IMPORTANT: visualize x-pred, NOT the current noisy/intermediate state.
                # After the Euler step we are at next_t. Evaluate both methods there,
                # convert to clean x-pred, and decode those clean predictions.
                next_tbc = next_te.view(next_te.shape[0], *([1]*(z0.ndim-1)))
                next_alpha, next_sigma, next_da, next_ds = model.interpolant(
                    next_tbc, path_type=path_type
                )

                next_v_m = model.inference(z_m, next_te, y)
                next_xp_m, _, _, _, _ = model_clean_from_velocity(
                    model, z_m, next_v_m, next_te, path_type
                )

                next_pfix = fixed_p(next_t, a)
                next_chosen_p, next_xp_l, _next_v_l, _next_all_cos = local_choice(
                    model, bank, z_l, next_te, y,
                    next_alpha, next_sigma, next_da, next_ds,
                    ps, a.p_mode, next_pfix
                )

                im_m = decode_norm(vae, next_xp_m, scale, bias)
                im_l = decode_norm(vae, next_xp_l, scale, bias)
                pil_m, pil_l = tensor_to_pil(im_m), tensor_to_pil(im_l)

                tag = f"{next_t:g}".replace(".", "p")
                pil_m.save(out / f"nfe{nfe}_t{tag}_decode_xpred.png")
                pil_l.save(out / f"nfe{nfe}_t{tag}_decode_ours_xpred.png")
                traj_sheet.append([
                    (pil_m, f"decode(x_pred) @t={next_t:g}"),
                    (pil_l, f"decode(ours x_pred) @t={next_t:g} | P={next_chosen_p}"),
                ])

        final_im_m = decode_norm(vae, final_xp_m, scale, bias)
        final_im_l = decode_norm(vae, final_xp_l, scale, bias)
        final_pil_m, final_pil_l = tensor_to_pil(final_im_m), tensor_to_pil(final_im_l)
        final_pil_m.save(out / f"nfe{nfe}_decode_xpred.png")
        final_pil_l.save(out / f"nfe{nfe}_decode_ours_xpred.png")

        ptxt = "->".join(map(str, p_path))
        fcos = flat_cos(final_xp_m, final_xp_l)
        fmse = rel_mse(final_xp_m, final_xp_l)
        fpsnr = image_psnr(final_im_m, final_im_l)

        summaries[str(nfe)] = {
            "times": times, "p_path": p_path,
            "final_xpred_cos": fcos,
            "final_xpred_rel_mse": fmse,
            "final_decode_psnr": fpsnr,
        }

        print(f"  FINAL: P path={ptxt}, xpred_cos={fcos:.6f}, relMSE={fmse:.6f}, PSNR={fpsnr:.3f}")

        summary_sheet.append([
            (original_pil, f"Original ({class_name})"),
            (recon_pil, "VAE recon"),
            (final_pil_m, f"decode(x_pred) | NFE={nfe}"),
            (final_pil_l, f"decode(ours x_pred) | NFE={nfe} | P:{ptxt}"),
        ])

        if a.save_intermediate:
            traj_sheet.append([
                (final_pil_m, f"decode(x_pred) FINAL | NFE={nfe}"),
                (final_pil_l, f"decode(ours x_pred) FINAL | NFE={nfe} | P:{ptxt}"),
            ])
            make_contact_sheet(traj_sheet, out / f"trajectory_nfe{nfe}.png", cell=256)

        torch.cuda.empty_cache()

    summary_path = out / "comparison_decode_xpred_nfe_2_4_8.png"
    make_contact_sheet(summary_sheet, summary_path, cell=256)

    csv_path = out / "metrics.csv"
    fields = [
        "nfe","step_index","t_raw","next_t_raw","is_final","p_mode","chosen_p",
        "oracle_same_state_vcos","state_cos_before","state_rel_mse_before",
        "state_cos_after","state_rel_mse_after","velocity_cos_cross_state",
        "xpred_cos_cross_state","all_p_vcos",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    result = {
        "experiment": "single_image_multistep_model_vs_local_score",
        "query": {"val_index": a.val_index, "filename": name,
                  "class_index": int(label), "class_name": class_name},
        "reference_bank": {"class_conditioned": not a.unconditional_bank,
                           "count": int(bank.shape[0]), "cache": str(cache_path)},
        "t_start": a.t_start, "nfe": nfes, "p_mode": a.p_mode,
        "patch_sizes": ps, "same_forward_noise_across_nfe": True,
        "runs": summaries, "rows": rows,
        "comparison_image": str(summary_path),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print("\nDONE")
    print("summary:", summary_path)
    print("metrics:", csv_path, out / "metrics.json")


if __name__ == "__main__":
    main(parse_args())
