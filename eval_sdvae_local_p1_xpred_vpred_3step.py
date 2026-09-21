#!/usr/bin/env python3
"""Qualitative SD-VAE Local Score sanity check, P=1 only.

Pipeline:
    ImageNet train -> latent bank
    one ImageNet val image -> z0 -> fixed Gaussian noise at t
    -> analytic Local posterior mean with P=1
    -> SD-VAE decode

No Ideal Score is computed.

Additionally, this version can convert Local P=1 x-prediction to the
corresponding path velocity and integrate a multi-step trajectory from t=0.8
to t=0. This is intended to distinguish an endpoint failure from a failure of
the whole denoising trajectory.
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
    p.add_argument("--t-values", type=float, nargs="+", default=[0.8, 0.6, 0.4, 0.2])
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
    p.add_argument("--device", default="cuda:0")
    # Multi-step trajectory experiment requested by the advisor.
    p.add_argument("--trajectory-start", type=float, default=0.8)
    p.add_argument("--trajectory-steps", type=int, default=3,
                   help="Number of coarse Euler denoising steps from trajectory-start to 0 (default: 3).")
    p.add_argument("--no-trajectory", action="store_true",
                   help="Disable the t=0.8 multi-step velocity trajectory experiment.")
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


def interpolant_with_derivatives(t, path_type):
    """Return alpha(t), sigma(t), d alpha/dt, d sigma/dt."""
    t = float(t)
    if path_type == "linear":
        return 1.0 - t, t, -1.0, 1.0
    w = math.pi / 2.0
    alpha = math.cos(w * t)
    sigma = math.sin(w * t)
    alpha_dot = -w * math.sin(w * t)
    sigma_dot = w * math.cos(w * t)
    return alpha, sigma, alpha_dot, sigma_dot


def xpred_to_velocity(z_t, x_pred, t, path_type):
    """Convert a clean-latent prediction x_pred into path velocity.

    z_t = alpha(t) * x + sigma(t) * eps
    eps_pred = (z_t - alpha * x_pred) / sigma
    v_pred = alpha_dot * x_pred + sigma_dot * eps_pred

    For the default linear path this simplifies to
        v_pred = (z_t - x_pred) / t.
    """
    alpha, sigma, alpha_dot, sigma_dot = interpolant_with_derivatives(t, path_type)
    if sigma <= 0:
        raise ValueError(f"sigma(t) must be > 0 when converting x-pred to velocity, got t={t}")
    eps_pred = (z_t - alpha * x_pred) / sigma
    v_pred = alpha_dot * x_pred + sigma_dot * eps_pred
    return v_pred, eps_pred


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


@torch.no_grad()
def run_velocity_trajectory(
    z_start,
    bank_cpu,
    mean,
    std,
    vae,
    device,
    path_type,
    t_start,
    num_steps,
    ref_chunk_size,
    out_dir,
):
    """Integrate dz/dt = v_theta(z,t) backward from t_start to 0.

    At every step:
      1) Local P=1 gives x_pred(z_t, t).
      2) x_pred is converted analytically to velocity.
      3) One backward Euler step updates z_t.

    The final step is allowed to land exactly at t=0; Local Score is never
    evaluated at t=0, so there is no sigma=0 division.
    """
    if not (0.0 < t_start <= 1.0):
        raise ValueError("trajectory-start must be in (0, 1]")
    if num_steps < 1:
        raise ValueError("trajectory-steps must be >= 1")

    traj_dir = Path(out_dir) / "xpred_to_vpred_t0p8"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # E.g. 8 steps: 0.8, 0.7, ..., 0.1, 0.0.
    times = torch.linspace(t_start, 0.0, num_steps + 1).tolist()
    z = z_start.clone()
    records = []
    state_pils = []
    xpred_pils = []

    # Save the starting noisy state.
    state_img = unwrap_decode(vae, denormalize(z, mean, std))
    state_pil = to_pil(state_img)
    state_pil.save(traj_dir / f"state_t{times[0]:.3f}.png")
    state_pils.append(state_pil)

    for k in range(num_steps):
        t_cur = float(times[k])
        t_next = float(times[k + 1])
        dt = t_next - t_cur  # negative for denoising toward t=0

        alpha, sigma = interpolant_scalar(t_cur, path_type)
        print(
            f"[trajectory] step {k+1:02d}/{num_steps}: "
            f"t={t_cur:.4f} -> {t_next:.4f}, dt={dt:.4f}"
        )

        x_pred = local_mean_p1(
            z, bank_cpu, alpha, sigma, device, ref_chunk_size
        )
        v_pred, eps_pred = xpred_to_velocity(z, x_pred, t_cur, path_type)

        # Explicit Euler integration backward in t.
        z_next = z + dt * v_pred

        # Decode both the local x-pred at the current point and the new state.
        xpred_img = unwrap_decode(vae, denormalize(x_pred, mean, std))
        xpred_pil = to_pil(xpred_img)
        xpred_pil.save(traj_dir / f"xpred_at_t{t_cur:.3f}.png")
        xpred_pils.append(xpred_pil)

        next_img = unwrap_decode(vae, denormalize(z_next, mean, std))
        next_pil = to_pil(next_img)
        next_pil.save(traj_dir / f"state_t{t_next:.3f}.png")
        state_pils.append(next_pil)

        records.append({
            "step": k + 1,
            "t": t_cur,
            "t_next": t_next,
            "dt": dt,
            "alpha": alpha,
            "sigma": sigma,
            "velocity_rms": float(v_pred.square().mean().sqrt().item()),
            "eps_pred_rms": float(eps_pred.square().mean().sqrt().item()),
            "x_pred_rms": float(x_pred.square().mean().sqrt().item()),
            "state_rms": float(z.square().mean().sqrt().item()),
        })
        z = z_next

    # Build a compact sheet that explicitly shows what the advisor asked for:
    # current state z_t -> Local x-pred -> converted v-pred -> updated state.
    # v_pred is a latent vector, so for visualization we decode the one-step
    # Euler proposal z_t + dt * v_pred rather than decoding v_pred itself.
    labels = []
    sheet_rows = []
    z = z_start.clone()
    for k in range(num_steps):
        t_cur = float(times[k])
        t_next = float(times[k + 1])
        dt = t_next - t_cur
        alpha, sigma = interpolant_scalar(t_cur, path_type)
        x_pred = local_mean_p1(z, bank_cpu, alpha, sigma, device, ref_chunk_size)
        v_pred, _ = xpred_to_velocity(z, x_pred, t_cur, path_type)
        z_next = z + dt * v_pred

        state_img = to_pil(unwrap_decode(vae, denormalize(z, mean, std)))
        xpred_img = to_pil(unwrap_decode(vae, denormalize(x_pred, mean, std)))
        proposal_img = to_pil(unwrap_decode(vae, denormalize(z_next, mean, std)))
        sheet_rows.append([state_img, xpred_img, proposal_img])
        labels.append(f"t={t_cur:.2f} -> {t_next:.2f}")
        z = z_next

    sheet_path = traj_dir / "xpred_to_vpred_steps.png"
    contact_sheet(
        sheet_rows,
        ["Current state z_t", "Local x_pred", "After v_pred Euler step"],
        labels,
        sheet_path,
        cell=256,
    )

    return {
        "t_start": t_start,
        "num_steps": num_steps,
        "times": times,
        "records": records,
        "final_state_path": str(traj_dir / f"state_t{times[-1]:.3f}.png"),
        "trajectory_sheet": str(sheet_path),
    }


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

    # Multi-step denoising from t=trajectory_start using x-pred -> velocity.
    # We deliberately build the start state from the SAME z0 and SAME eps as
    # the single-step experiment, so t=0.8 is directly comparable.
    if not args.no_trajectory:
        t_start = float(args.trajectory_start)
        alpha_start, sigma_start = interpolant_scalar(t_start, args.path_type)
        z_start = alpha_start * z0 + sigma_start * eps
        report["velocity_trajectory"] = run_velocity_trajectory(
            z_start=z_start,
            bank_cpu=bank,
            mean=mean,
            std=std,
            vae=vae,
            device=device,
            path_type=args.path_type,
            t_start=t_start,
            num_steps=args.trajectory_steps,
            ref_chunk_size=args.ref_chunk_size,
            out_dir=out,
        )

    comparison = out / "comparison_local_p1.png"
    contact_sheet(rows, headers, labels, comparison, args.resolution)
    (out / "run.json").write_text(json.dumps(report, indent=2))
    original_pil.save(out / "original.png")
    recon_pil.save(out / "vae_recon.png")

    print("\ncomparison:", comparison)
    print("run info:", out / "run.json")


if __name__ == "__main__":
    main(parse_args())
