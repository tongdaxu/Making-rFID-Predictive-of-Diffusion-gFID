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
    p.add_argument("--stage-t-values", type=float, nargs="+", default=[0.8, 0.4, 0.2],
                   help="Sequential re-noising schedule, e.g. 0.8 0.4 0.2.")
    p.add_argument("--fresh-noise-each-stage", action="store_true",
                   help="Use fresh Gaussian noise after each x_pred. Default reuses the same eps.")
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
    """
    Teacher-requested staged procedure:

        z0
          -> add noise at t=0.8 -> Local P=1 -> z_pred_0
          -> add noise at t=0.4 to z_pred_0 -> Local P=1 -> z_pred_1
          -> add noise at t=0.2 to z_pred_1 -> Local P=1 -> z_pred_2
          -> decode z_pred_2

    This is NOT Euler / velocity integration.
    Each stage treats the previous clean prediction as the new clean latent,
    corrupts it again at a lower t, then predicts clean latent again.
    """
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

    # Controlled default: reuse one epsilon at every stage.
    # Set --fresh-noise-each-stage if the teacher wants an independent epsilon
    # for each re-noising operation.
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    shared_eps = torch.randn(z0.shape, device=device, dtype=z0.dtype, generator=gen)

    original_pil = to_pil(img)
    recon_pil = to_pil(recon)
    original_pil.save(out / "original.png")
    recon_pil.save(out / "vae_recon.png")

    current_clean = z0
    rows = []
    labels = []
    report = {
        "vae_config": args.vae_config,
        "train_size": args.train_size,
        "val_index": args.val_index,
        "val_name": str(name),
        "seed": args.seed,
        "path_type": args.path_type,
        "latent_norm": args.latent_norm,
        "stage_t_values": args.stage_t_values,
        "fresh_noise_each_stage": args.fresh_noise_each_stage,
        "local_patch": 1,
        "stages": [],
    }

    for stage_idx, t in enumerate(args.stage_t_values, start=1):
        alpha, sigma = interpolant_scalar(t, args.path_type)
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        if args.fresh_noise_each_stage:
            eps = torch.randn(
                current_clean.shape,
                device=device,
                dtype=current_clean.dtype,
                generator=gen,
            )
        else:
            eps = shared_eps

        # Important:
        # this stage corrupts the PREVIOUS z_pred, not the original z0.
        z_noisy = alpha * current_clean + sigma * eps

        print(
            f"\n=== stage {stage_idx}: t={t:g}, "
            f"alpha={alpha:.6f}, sigma={sigma:.6f} ==="
        )

        z_pred = local_mean_p1(
            z_noisy,
            bank,
            alpha,
            sigma,
            device,
            args.ref_chunk_size,
        )

        noisy_img = unwrap_decode(vae, denormalize(z_noisy, mean, std))
        pred_img = unwrap_decode(vae, denormalize(z_pred, mean, std))

        noisy_pil = to_pil(noisy_img)
        pred_pil = to_pil(pred_img)

        t_tag = str(t).replace(".", "p")
        noisy_path = out / f"stage{stage_idx}_t{t_tag}_input_noisy.png"
        pred_path = out / f"stage{stage_idx}_t{t_tag}_zpred.png"
        noisy_pil.save(noisy_path)
        pred_pil.save(pred_path)

        rows.append([noisy_pil, pred_pil])
        labels.append(f"stage {stage_idx}, t={t:g}")

        report["stages"].append(
            {
                "stage": stage_idx,
                "t": float(t),
                "alpha": float(alpha),
                "sigma": float(sigma),
                "input_noisy": str(noisy_path),
                "z_pred": str(pred_path),
            }
        )

        # The prediction becomes the clean latent for the next stage.
        current_clean = z_pred

    final_img = unwrap_decode(vae, denormalize(current_clean, mean, std))
    final_pil = to_pil(final_img)
    final_path = out / "FINAL_zpred_decoded.png"
    final_pil.save(final_path)

    comparison = out / "staged_local_p1.png"
    contact_sheet(
        rows,
        ["re-noised input", "Local P=1 z_pred"],
        labels,
        comparison,
        args.resolution,
    )

    (out / "run.json").write_text(json.dumps(report, indent=2))

    print("\nFinal decoded image:", final_path)
    print("Stage comparison:", comparison)
    print("Run info:", out / "run.json")


if __name__ == "__main__":
    main(parse_args())
