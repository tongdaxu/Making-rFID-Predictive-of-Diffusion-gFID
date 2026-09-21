#!/usr/bin/env python3
"""Qualitative SD-VAE Local Score sanity check, P=1 only.

Pipeline:
    ImageNet train -> latent bank
    one ImageNet val image -> z0 -> fixed Gaussian noise at t
    -> analytic Local posterior mean with P=1
    -> SD-VAE decode

No Ideal Score is computed.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-config", default="./configs/SDVAE.yaml")
    p.add_argument("--dataset-ref", default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--dataset", default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--t-values", type=float, nargs="+", default=[0.8, 0.4, 0.2])
    p.add_argument("--path-type", choices=["linear", "cosine"], default="linear")
    p.add_argument("--train-size", type=int, default=100000)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--ref-chunk-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--latent-norm", choices=["channel", "global", "none"], default="channel")
    p.add_argument("--stats-eps", type=float, default=1e-6)
    p.add_argument("--cache-dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--cache-path", default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--out-dir", default="./local_p1_sdvae_single_100k")
    p.add_argument("--velocity-start", type=float, default=0.8)
    p.add_argument("--velocity-steps", type=int, default=3, choices=[2, 3],
                   help="Run 2 or 3 Euler denoising steps from velocity-start to 0.")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


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
        return ld.mode() if hasattr(ld, "mode") else ld.sample()
    if hasattr(z, "mode") and callable(z.mode):
        return z.mode()
    if hasattr(z, "mean") and torch.is_tensor(z.mean):
        return z.mean
    raise TypeError(type(z))


def unwrap_decode(vae, z):
    out = vae.decode(z)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out):
        return out
    if hasattr(out, "sample") and torch.is_tensor(out.sample):
        return out.sample
    raise TypeError(type(out))


def interpolant_scalar(t, path_type):
    if path_type == "linear":
        return 1.0 - float(t), float(t)
    return math.cos(float(t) * math.pi / 2), math.sin(float(t) * math.pi / 2)


def xpred_to_vpred(x_cur, x_pred, t_cur, path_type="linear"):
    """Convert clean-latent x prediction to the velocity used by the Euler sampler.

    Path convention: x_t = alpha(t) * x_0 + sigma(t) * eps, sampling t: high -> low.
    For linear path alpha=1-t, sigma=t:
        eps_pred = (x_t - (1-t) * x_pred) / t
        v_pred   = eps_pred - x_pred
                 = (x_t - x_pred) / t
    """
    t = float(t_cur)
    if t <= 0:
        raise ValueError("x-pred -> v-pred conversion requires t > 0")
    if path_type == "linear":
        eps_pred = (x_cur - (1.0 - t) * x_pred) / t
        v_pred = eps_pred - x_pred
        return v_pred, eps_pred
    elif path_type == "cosine":
        w = math.pi / 2.0
        alpha = math.cos(w * t)
        sigma = math.sin(w * t)
        da = -w * math.sin(w * t)
        ds =  w * math.cos(w * t)
        eps_pred = (x_cur - alpha * x_pred) / sigma
        v_pred = da * x_pred + ds * eps_pred
        return v_pred, eps_pred
    else:
        raise NotImplementedError(path_type)


@torch.no_grad()
def local_velocity_euler_sampler(
    z_start, bank_cpu, mean, std, vae, device,
    t_start=0.8, num_steps=3, path_type="linear", ref_chunk_size=2048,
    save_dir=None,
):
    """Mirror the provided euler_sampler, except model.inference is replaced by:
       Local P=1 x_pred -> v_pred.

    Returns the final latent. Optionally saves actual decoded images for each
    x_pred and state, but the primary output is one final decoded image.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if not (0.0 < t_start <= 1.0):
        raise ValueError("t_start must be in (0,1]")

    t_steps = torch.linspace(float(t_start), 0.0, num_steps + 1, dtype=torch.float64)
    x_next = z_start.to(device=device, dtype=torch.float32)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, (t_cur_t, t_next_t) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        t_cur = float(t_cur_t.item())
        t_next = float(t_next_t.item())
        x_cur = x_next

        alpha, sigma = interpolant_scalar(t_cur, path_type)
        x_pred = local_mean_p1(
            x_cur, bank_cpu, alpha, sigma, device, ref_chunk_size
        )
        v_pred, eps_pred = xpred_to_vpred(x_cur, x_pred, t_cur, path_type)

        # EXACT same Euler update pattern as the user's sampler:
        # x_next = x_cur + (t_next - t_cur) * d_cur
        x_next = x_cur + (t_next - t_cur) * v_pred

        rec = {
            "step": i, "t_cur": t_cur, "t_next": t_next,
            "xpred_rms": float(x_pred.square().mean().sqrt().item()),
            "vpred_rms": float(v_pred.square().mean().sqrt().item()),
            "state_rms": float(x_next.square().mean().sqrt().item()),
        }
        records.append(rec)

        if save_dir is not None:
            # These are actual decoded images, not a visualization of v itself.
            to_pil(unwrap_decode(vae, denormalize(x_pred, mean, std))).save(
                save_dir / f"step{i+1}_t{t_cur:.3f}_xpred.png"
            )
            to_pil(unwrap_decode(vae, denormalize(x_next, mean, std))).save(
                save_dir / f"step{i+1}_to_t{t_next:.3f}_state.png"
            )

    return x_next, records


def normalize(z, mean, std):
    return (z - mean.to(z.device, z.dtype)) / std.to(z.device, z.dtype)


def denormalize(z, mean, std):
    return z * std.to(z.device, z.dtype) + mean.to(z.device, z.dtype)


def resolve_cache_path(args):
    if args.cache_path:
        return Path(args.cache_path)
    if args.train_size % 1000 == 0:
        size_tag = f"{args.train_size // 1000}k"
    else:
        size_tag = str(args.train_size)
    # Same naming convention as the full Ideal+Local script, so both scripts
    # can share one encoded latent bank.
    return Path(f"./ideal_local_sdvae_train{size_tag}_cache.pt")


def cache_mismatch_reason(obj, args):
    checks = [
        (int(obj.get("train_size", -1)) == args.train_size, "train_size"),
        (obj.get("latent_norm") == args.latent_norm, "latent_norm"),
        (obj.get("cache_dtype", args.cache_dtype) == args.cache_dtype, "cache_dtype"),
        (obj.get("vae_config", args.vae_config) == args.vae_config, "vae_config"),
    ]
    bad = [name for ok, name in checks if not ok]
    return ", ".join(bad) if bad else None


@torch.no_grad()
def build_or_load_bank(args, vae, transform, device):
    cp = resolve_cache_path(args)
    if cp.exists() and not args.rebuild_cache:
        obj = torch.load(cp, map_location="cpu")
        reason = cache_mismatch_reason(obj, args)
        if reason is None:
            print("[cache] loaded", cp)
            return obj
        print(f"[cache] incompatible ({reason}); rebuilding: {cp}")

    ds = ImageNetValDataset(args.dataset_ref, transform, args.train_size)
    loader = DataLoader(
        ds,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    raw_chunks = []
    sum_c = sumsq_c = None
    total_spatial = 0
    total_scalar = 0
    sum_all = 0.0
    sumsq_all = 0.0

    for img, y, _ in tqdm(loader, desc="Encode reference bank"):
        img, y = img.to(device), y.to(device)
        z = unwrap_encode(vae, img, y)
        if z.ndim != 4:
            raise RuntimeError(f"Expected BCHW latent, got {tuple(z.shape)}")

        z = z.float().cpu()
        raw_chunks.append(z.to(torch.float16))

        if args.latent_norm == "channel":
            cur_sum = z.sum((0, 2, 3), keepdim=True, dtype=torch.float64)
            zd = z.double()
            cur_sumsq = (zd * zd).sum((0, 2, 3), keepdim=True)
            sum_c = cur_sum if sum_c is None else sum_c + cur_sum
            sumsq_c = cur_sumsq if sumsq_c is None else sumsq_c + cur_sumsq
            total_spatial += z.shape[0] * z.shape[2] * z.shape[3]
        elif args.latent_norm == "global":
            zd = z.double()
            sum_all += zd.sum().item()
            sumsq_all += (zd * zd).sum().item()
            total_scalar += z.numel()

    train_raw = torch.cat(raw_chunks, 0)[: args.train_size].contiguous()

    if args.latent_norm == "none":
        C = train_raw.shape[1]
        mean = torch.zeros((1, C, 1, 1), dtype=torch.float32)
        std = torch.ones_like(mean)
    elif args.latent_norm == "channel":
        mean64 = sum_c / total_spatial
        var64 = sumsq_c / total_spatial - mean64 * mean64
        mean = mean64.float()
        std = torch.sqrt(var64.clamp_min(args.stats_eps).float())
    else:
        mean64 = sum_all / total_scalar
        var64 = sumsq_all / total_scalar - mean64 * mean64
        mean = torch.tensor(mean64, dtype=torch.float32).view(1, 1, 1, 1)
        std = torch.sqrt(
            torch.tensor(max(var64, args.stats_eps), dtype=torch.float32)
        ).view(1, 1, 1, 1)

    out_dtype = torch.float16 if args.cache_dtype == "fp16" else torch.float32
    norm_chunks = []
    cpu_chunk = max(args.train_batch_size * 8, 256)
    for s in tqdm(range(0, len(train_raw), cpu_chunk), desc="Normalize reference bank"):
        x = train_raw[s : s + cpu_chunk].float()
        norm_chunks.append(normalize(x, mean, std).to(out_dtype))
    train_norm = torch.cat(norm_chunks, 0).contiguous()

    obj = dict(
        vae_config=args.vae_config,
        train_size=args.train_size,
        latent_shape=list(train_norm.shape[1:]),
        latent_norm=args.latent_norm,
        cache_dtype=args.cache_dtype,
        mean=mean.cpu(),
        std=std.cpu(),
        train_norm=train_norm,
    )
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, cp)
    print("[cache] saved", cp)
    return obj


@torch.no_grad()
def local_mean_p1(z_t, bank_cpu, alpha, sigma, device, chunk):
    """Exact P=1 Local posterior mean.

    At each spatial position x, the posterior only sees the C-dimensional
    latent vector at that same position. This is the P=1 specialization of
    the Local Score and avoids unfold(), making 100k-bank evaluation cheaper.
    """
    _, C, H, W = z_t.shape
    L = H * W

    q = z_t[0].reshape(C, L).float()  # [C, L]
    qsq = (q * q).sum(0)             # [L]

    m = torch.full((L,), float("-inf"), device=device)
    den = torch.zeros(L, device=device)
    num = torch.zeros(C, L, device=device)
    scale = 2.0 * sigma * sigma

    for s in tqdm(range(0, len(bank_cpu), chunk), desc="Local P=1", leave=False):
        ref = bank_cpu[s : s + chunk].to(device=device, dtype=torch.float32)
        B = ref.shape[0]
        r = ref.reshape(B, C, L)      # [B, C, L]

        rsq = (r * r).sum(1)         # [B, L]
        qr = (r * q[None]).sum(1)    # [B, L]
        dist2 = (qsq[None] + alpha * alpha * rsq - 2.0 * alpha * qr).clamp_min(0)
        logits = -dist2 / scale

        # Streaming log-sum-exp across reference-bank chunks.
        cm = logits.max(0).values
        nm = torch.maximum(m, cm)
        old = torch.where(torch.isfinite(m), torch.exp(m - nm), torch.zeros_like(nm))
        w = torch.exp(logits - nm[None])

        num = num * old[None] + torch.einsum("bl,bcl->cl", w, r)
        den = den * old + w.sum(0)
        m = nm

    return (num / den.clamp_min(1e-30)[None]).reshape(1, C, H, W)


def to_pil(x):
    x = ((x.detach().float().cpu()[0] + 1) / 2).clamp(0, 1)
    arr = x.mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def contact_sheet(rows, headers, labels, path, cell=256):
    left, top = 82, 34
    canvas = Image.new("RGB", (left + cell * len(headers), top + cell * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for j, h in enumerate(headers):
        draw.text((left + j * cell + 4, 10), h, fill="black", font=font)
    for i, label in enumerate(labels):
        draw.text((6, top + i * cell + 6), label, fill="black", font=font)
        for j, im in enumerate(rows[i]):
            canvas.paste(
                im.resize((cell, cell), Image.Resampling.LANCZOS),
                (left + j * cell, top + i * cell),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    vae = instantiate_from_config(OmegaConf.load(args.vae_config)).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    cache = build_or_load_bank(args, vae, transform, device)
    bank = cache["train_norm"].contiguous()
    mean, std = cache["mean"].float(), cache["std"].float()
    print("bank:", tuple(bank.shape), bank.dtype, "norm=", cache["latent_norm"])

    val_ds = ImageNetValDataset(args.dataset, transform, max(args.val_index + 1, 1))
    img, y, name = val_ds[args.val_index]
    img = img[None].to(device)
    y = torch.as_tensor([y], device=device)

    z0_raw = unwrap_encode(vae, img, y).float()
    z0 = normalize(z0_raw, mean, std)
    recon = unwrap_decode(vae, z0_raw)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    # Same eps is reused for all t values, so visual changes mainly reflect t.
    eps = torch.randn(z0.shape, device=device, dtype=z0.dtype, generator=gen)

    original_pil = to_pil(img)
    recon_pil = to_pil(recon)
    headers = ["Original", "VAE recon", "Noisy decode", "Local P=1"]
    rows, labels = [], []

    report = {
        "vae_config": args.vae_config,
        "train_size": args.train_size,
        "val_index": args.val_index,
        "val_name": str(name),
        "seed": args.seed,
        "path_type": args.path_type,
        "latent_norm": args.latent_norm,
        "t_values": args.t_values,
        "local_patch": 1,
        "results": {},
    }

    for t in args.t_values:
        alpha, sigma = interpolant_scalar(t, args.path_type)
        if sigma <= 0:
            raise ValueError("sigma must be > 0 for Local Score")

        print(f"\n=== t={t:g} alpha={alpha:.6f} sigma={sigma:.6f} ===")
        zt = alpha * z0 + sigma * eps

        noisy_img = unwrap_decode(vae, denormalize(zt, mean, std))
        mu_l = local_mean_p1(
            zt,
            bank,
            alpha,
            sigma,
            device,
            args.ref_chunk_size,
        )
        local_img = unwrap_decode(vae, denormalize(mu_l, mean, std))

        noisy_pil = to_pil(noisy_img)
        local_pil = to_pil(local_img)
        row = [original_pil.copy(), recon_pil.copy(), noisy_pil, local_pil]
        rows.append(row)
        labels.append(f"t={t:g}")

        t_tag = str(t).replace(".", "p")
        noisy_pil.save(out / f"t{t_tag}_noisy.png")
        local_pil.save(out / f"t{t_tag}_local_p1.png")
        report["results"][f"t{t:g}"] = {"alpha": alpha, "sigma": sigma}

    # Advisor-requested experiment: start exactly from z_t at t=0.8,
    # convert Local x-pred to v-pred, and run only 2-3 Euler steps.
    t0 = float(args.velocity_start)
    a0, s0 = interpolant_scalar(t0, args.path_type)
    z_start = a0 * z0 + s0 * eps
    vel_dir = out / f"velocity_from_t{str(t0).replace('.', 'p')}_{args.velocity_steps}step"
    z_final, vel_records = local_velocity_euler_sampler(
        z_start=z_start, bank_cpu=bank, mean=mean, std=std, vae=vae, device=device,
        t_start=t0, num_steps=args.velocity_steps, path_type=args.path_type,
        ref_chunk_size=args.ref_chunk_size, save_dir=vel_dir,
    )
    final_img = to_pil(unwrap_decode(vae, denormalize(z_final, mean, std)))
    final_path = vel_dir / "FINAL_denoised.png"
    final_img.save(final_path)
    print("[velocity] final decoded image:", final_path)
    for r in vel_records:
        print(f"[velocity] step {r['step']+1}: {r['t_cur']:.4f} -> {r['t_next']:.4f} "
              f"xpred_rms={r['xpred_rms']:.4f} vpred_rms={r['vpred_rms']:.4f} state_rms={r['state_rms']:.4f}")

    comparison = out / "comparison_local_p1.png"
    contact_sheet(rows, headers, labels, comparison, args.resolution)
    (out / "run.json").write_text(json.dumps(report, indent=2))
    original_pil.save(out / "original.png")
    recon_pil.save(out / "vae_recon.png")

    print("\ncomparison:", comparison)
    print("run info:", out / "run.json")


if __name__ == "__main__":
    main(parse_args())
