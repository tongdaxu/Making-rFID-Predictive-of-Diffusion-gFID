#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

import torch
import torch.nn.functional as F
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
    p.add_argument("--local-patches", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--path-type", choices=["linear", "cosine"], default="linear")
    p.add_argument("--train-size", type=int, default=10000)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--ref-chunk-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--val-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--latent-norm", choices=["channel", "global", "none"], default="channel")
    p.add_argument("--stats-eps", type=float, default=1e-6)
    p.add_argument("--cache-dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--cache-path", default="./ideal_local_sdvae_train10k_cache.pt")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--out-dir", default="./ideal_local_sdvae_single")
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


def compute_stats(train_raw, mode, eps):
    if mode == "none":
        mean = torch.zeros((1, train_raw.shape[1], 1, 1))
        std = torch.ones_like(mean)
    elif mode == "channel":
        mean = train_raw.mean((0, 2, 3), keepdim=True)
        var = train_raw.var((0, 2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var.clamp_min(eps))
    else:
        mean = train_raw.mean().view(1, 1, 1, 1)
        var = train_raw.var(unbiased=False).view(1, 1, 1, 1)
        std = torch.sqrt(var.clamp_min(eps))
    return mean.float(), std.float()


def normalize(z, mean, std):
    return (z - mean.to(z.device, z.dtype)) / std.to(z.device, z.dtype)


def denormalize(z, mean, std):
    return z * std.to(z.device, z.dtype) + mean.to(z.device, z.dtype)


@torch.no_grad()
def build_or_load_bank(args, vae, transform, device):
    cp = Path(args.cache_path)
    if cp.exists() and not args.rebuild_cache:
        obj = torch.load(cp, map_location="cpu")
        if int(obj["train_size"]) != args.train_size:
            raise RuntimeError("cache train-size mismatch; use another cache-path or --rebuild-cache")
        return obj

    ds = ImageNetValDataset(args.dataset_ref, transform, args.train_size)
    loader = DataLoader(ds, batch_size=args.train_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    chunks = []
    for img, y, _ in tqdm(loader, desc="Encode reference bank"):
        img, y = img.to(device), y.to(device)
        z = unwrap_encode(vae, img, y)
        if z.ndim != 4:
            raise RuntimeError(f"Expected BCHW latent, got {tuple(z.shape)}")
        chunks.append(z.float().cpu())
    train_raw = torch.cat(chunks, 0)[:args.train_size].contiguous()
    mean, std = compute_stats(train_raw, args.latent_norm, args.stats_eps)
    train_norm = normalize(train_raw, mean, std)
    train_norm = train_norm.to(torch.float16 if args.cache_dtype == "fp16" else torch.float32)

    obj = dict(
        vae_config=args.vae_config, train_size=args.train_size,
        latent_shape=list(train_norm.shape[1:]), latent_norm=args.latent_norm,
        mean=mean.cpu(), std=std.cpu(), train_norm=train_norm.contiguous()
    )
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, cp)
    print("[cache] saved", cp)
    return obj


@torch.no_grad()
def ideal_mean(z_t, bank_cpu, alpha, sigma, device, chunk):
    q = z_t.reshape(1, -1).float()
    D = q.shape[1]
    m = torch.tensor(float("-inf"), device=device)
    den = torch.zeros((), device=device)
    num = torch.zeros(D, device=device)
    scale = 2.0 * sigma * sigma

    for s in range(0, len(bank_cpu), chunk):
        ref = bank_cpu[s:s+chunk].to(device=device, dtype=torch.float32)
        rf = ref.reshape(ref.shape[0], -1)
        logits = -((q - alpha * rf) ** 2).sum(1) / scale
        cm = logits.max()
        nm = torch.maximum(m, cm)
        old = torch.where(torch.isfinite(m), torch.exp(m - nm), torch.zeros_like(nm))
        w = torch.exp(logits - nm)
        num = num * old + torch.einsum("b,bd->d", w, rf)
        den = den * old + w.sum()
        m = nm
    return (num / den.clamp_min(1e-30)).reshape_as(z_t)


@torch.no_grad()
def local_mean(z_t, bank_cpu, alpha, sigma, P, device, chunk):
    if P < 1 or P % 2 == 0:
        raise ValueError("local patch sizes must be positive odd integers")
    _, C, H, W = z_t.shape
    L = H * W
    pad = P // 2
    qp = F.unfold(z_t.float(), kernel_size=P, padding=pad)[0]  # [Dpatch,L]
    qsq = (qp * qp).sum(0)

    m = torch.full((L,), float("-inf"), device=device)
    den = torch.zeros(L, device=device)
    num = torch.zeros(C, L, device=device)
    scale = 2.0 * sigma * sigma

    for s in range(0, len(bank_cpu), chunk):
        ref = bank_cpu[s:s+chunk].to(device=device, dtype=torch.float32)
        B = ref.shape[0]
        rp = F.unfold(ref, kernel_size=P, padding=pad)          # [B,Dpatch,L]
        rsq = (rp * rp).sum(1)
        qr = torch.einsum("bdl,dl->bl", rp, qp)
        dist2 = (qsq[None] + alpha*alpha*rsq - 2.0*alpha*qr).clamp_min(0)
        logits = -dist2 / scale

        cm = logits.max(0).values
        nm = torch.maximum(m, cm)
        old = torch.where(torch.isfinite(m), torch.exp(m - nm), torch.zeros_like(nm))
        w = torch.exp(logits - nm[None])

        centers = ref.reshape(B, C, L)
        num = num * old[None] + torch.einsum("bl,bcl->cl", w, centers)
        den = den * old + w.sum(0)
        m = nm

    return (num / den.clamp_min(1e-30)[None]).reshape(1, C, H, W)


def score(z_t, mu, alpha, sigma):
    return (alpha * mu - z_t) / (sigma * sigma)


def score_stats(local_s, ideal_s):
    a, b = local_s.flatten(1).float(), ideal_s.flatten(1).float()
    cos = F.cosine_similarity(a, b, dim=1).item()
    relmse = ((a-b).pow(2).mean() / (b.pow(2).mean() + 1e-8)).item()
    return cos, relmse


def to_pil(x):
    x = ((x.detach().float().cpu()[0] + 1) / 2).clamp(0, 1)
    arr = x.mul(255).round().byte().permute(1,2,0).numpy()
    return Image.fromarray(arr)


def contact_sheet(rows, headers, labels, path, cell=256):
    left, top = 82, 34
    canvas = Image.new("RGB", (left + cell*len(headers), top + cell*len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for j,h in enumerate(headers):
        draw.text((left+j*cell+4, 10), h, fill="black", font=font)
    for i,label in enumerate(labels):
        draw.text((6, top+i*cell+6), label, fill="black", font=font)
        for j,im in enumerate(rows[i]):
            canvas.paste(im.resize((cell,cell), Image.Resampling.LANCZOS),
                         (left+j*cell, top+i*cell))
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

    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    vae = instantiate_from_config(OmegaConf.load(args.vae_config)).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    cache = build_or_load_bank(args, vae, transform, device)
    bank = cache["train_norm"].contiguous()
    mean, std = cache["mean"].float(), cache["std"].float()
    print("bank:", tuple(bank.shape), bank.dtype, "norm=", cache["latent_norm"])

    val_ds = ImageNetValDataset(args.dataset, transform, max(args.val_index+1, 1))
    img, y, name = val_ds[args.val_index]
    img = img[None].to(device)
    y = torch.as_tensor([y], device=device)

    z0_raw = unwrap_encode(vae, img, y).float()
    z0 = normalize(z0_raw, mean, std)
    recon = unwrap_decode(vae, z0_raw)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    eps = torch.randn(z0.shape, device=device, dtype=z0.dtype, generator=gen)

    original_pil, recon_pil = to_pil(img), to_pil(recon)
    headers = ["Original", "VAE recon", "Noisy decode", "Ideal"] + [f"Local P={p}" for p in args.local_patches]
    rows, labels = [], []
    report = {
        "vae_config": args.vae_config, "train_size": args.train_size,
        "val_index": args.val_index, "val_name": str(name),
        "seed": args.seed, "path_type": args.path_type,
        "latent_norm": args.latent_norm,
        "t_values": args.t_values, "local_patches": args.local_patches,
        "results": {},
    }

    for t in args.t_values:
        alpha, sigma = interpolant_scalar(t, args.path_type)
        print(f"\n=== t={t:g} alpha={alpha:.6f} sigma={sigma:.6f} ===")
        zt = alpha*z0 + sigma*eps

        noisy_img = unwrap_decode(vae, denormalize(zt, mean, std))
        mu_i = ideal_mean(zt, bank, alpha, sigma, device, args.ref_chunk_size)
        s_i = score(zt, mu_i, alpha, sigma)
        ideal_img = unwrap_decode(vae, denormalize(mu_i, mean, std))

        row = [original_pil.copy(), recon_pil.copy(), to_pil(noisy_img), to_pil(ideal_img)]
        rr = {"alpha": alpha, "sigma": sigma, "locals": {}}

        for P in args.local_patches:
            mu_l = local_mean(zt, bank, alpha, sigma, P, device, args.ref_chunk_size)
            s_l = score(zt, mu_l, alpha, sigma)
            cos, relmse = score_stats(s_l, s_i)
            local_img = unwrap_decode(vae, denormalize(mu_l, mean, std))
            row.append(to_pil(local_img))
            rr["locals"][str(P)] = {
                "score_cos_to_ideal": cos,
                "score_rel_mse_to_ideal": relmse,
            }
            print(f"P={P}: cos(Local,Ideal)={cos:.6f} relMSE={relmse:.6f}")

        rows.append(row)
        labels.append(f"t={t:g}")
        report["results"][f"t{t:g}"] = rr

    comparison = out / "comparison.png"
    contact_sheet(rows, headers, labels, comparison, args.resolution)
    (out/"metrics.json").write_text(json.dumps(report, indent=2))
    original_pil.save(out/"original.png")
    recon_pil.save(out/"vae_recon.png")
    print("\ncomparison:", comparison)
    print("metrics:", out/"metrics.json")


if __name__ == "__main__":
    main(parse_args())
