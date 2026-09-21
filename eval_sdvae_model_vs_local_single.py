import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from omegaconf import OmegaConf

from ifid.dataset import ImageNetValDataset
from ifid.sit.sit import SiT_models
from ifid.vae.utils import instantiate_from_config


def unwrap_vae_latents(z):
    if isinstance(z, (tuple, list)):
        z = z[0]
    if not torch.is_tensor(z):
        raise TypeError(f"VAE encode returned {type(z)}")
    return z


def prepare_latents_for_sit(z):
    z = unwrap_vae_latents(z)

    if z.ndim == 4:
        return z.contiguous()

    if z.ndim == 3:
        b, n, c = z.shape
        side = int(math.sqrt(n))
        if side * side != n:
            raise RuntimeError(
                f"Token latent length N={n} is not square; shape={tuple(z.shape)}"
            )
        return z.transpose(1, 2).contiguous().view(b, c, side, side)

    raise RuntimeError(f"Unsupported latent shape for SiT: {tuple(z.shape)}")


def encode_vae(vae, img, y=None):
    try:
        z = vae.encode(img, y)
    except TypeError:
        z = vae.encode(img)
    return prepare_latents_for_sit(z).float()


def decode_vae(vae, z, y=None):
    try:
        x = vae.decode(z, y)
    except TypeError:
        x = vae.decode(z)
    if isinstance(x, (tuple, list)):
        x = x[0]
    return x.float()


def normalize_latents(z, scale, bias):
    return (z - bias) * scale


def denormalize_latents(z, scale, bias):
    return z / scale + bias


def tensor_to_pil(x):
    x = x.detach().float()
    if x.ndim == 4:
        x = x[0]
    x = ((x + 1.0) / 2.0).clamp(0, 1)
    x = (x.permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(x)


def get_model_dtype(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def make_cache_path(args):
    if args.cache_path:
        return Path(args.cache_path)
    ckpt_tag = Path(args.ckpt).stem
    return Path(args.out_dir) / f"local_bank_train{args.train_size}_{ckpt_tag}.pt"


@torch.no_grad()
def load_models(args, device):
    vae_cfg = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_cfg).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    fake_img = torch.zeros(
        1, 3, args.resolution, args.resolution, device=device, dtype=torch.float32
    )
    fake_z = encode_vae(vae, fake_img)
    _, in_channels, latent_h, latent_w = fake_z.shape
    if latent_h != latent_w:
        raise RuntimeError(f"Expected square spatial latent, got {tuple(fake_z.shape)}")
    latent_size = latent_h
    tshift = math.sqrt(float(fake_z[0].numel()) / 4096.0)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt["args"]

    model_name = getattr(train_args, "model")
    num_classes = getattr(train_args, "num_classes", 1000)
    cfg_prob = getattr(train_args, "cfg_prob", 0.1)
    bn_momentum = getattr(train_args, "bn_momentum", 0.1)
    fused_attn = getattr(train_args, "fused_attn", True)
    qk_norm = getattr(train_args, "qk_norm", False)

    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=cfg_prob,
        bn_momentum=bn_momentum,
        tshift=tshift,
        fused_attn=fused_attn,
        qk_norm=qk_norm,
    )
    state_key = "ema" if args.use_ema else "model"
    model.load_state_dict(ckpt[state_key], strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    stats = model.extract_latents_stats()
    scale = stats["latents_scale"].float().to(device).view(1, in_channels, 1, 1)
    bias = stats["latents_bias"].float().to(device).view(1, in_channels, 1, 1)

    print("[load] model:", model_name)
    print("[load] latent shape:", tuple(fake_z.shape))
    print("[load] tshift:", tshift)
    print("[load] latent scale:", scale.flatten().cpu().tolist())
    print("[load] latent bias :", bias.flatten().cpu().tolist())

    return model, vae, scale, bias


@torch.no_grad()
def build_or_load_bank(args, vae, scale, bias, transform, device):
    cache_path = make_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not args.rebuild_cache:
        obj = torch.load(cache_path, map_location="cpu")
        expected = int(args.train_size)
        if int(obj["train_size"]) == expected:
            print(f"[cache] loaded {cache_path}")
            bank = obj["train_norm"].float().contiguous()
            return bank
        print("[cache] train-size mismatch, rebuilding:", cache_path)

    ds = ImageNetValDataset(args.dataset_ref, transform, args.train_size)
    bank_chunks = []

    for idx in range(len(ds)):
        img, y, _ = ds[idx]
        img = img.unsqueeze(0).to(device)
        y = torch.as_tensor([y], device=device, dtype=torch.long)

        z_raw = encode_vae(vae, img, y)
        z_norm = normalize_latents(z_raw, scale, bias)
        bank_chunks.append(z_norm.squeeze(0).cpu().to(torch.float16))

        if (idx + 1) % 500 == 0 or (idx + 1) == len(ds):
            print(f"[cache] encoded {idx+1}/{len(ds)}")

    bank = torch.stack(bank_chunks, dim=0).contiguous()

    torch.save(
        {
            "train_size": int(args.train_size),
            "train_norm": bank,
            "ckpt": str(args.ckpt),
            "dataset_ref": str(args.dataset_ref),
        },
        cache_path,
    )
    print(f"[cache] saved {cache_path}")
    return bank.float()


@torch.no_grad()
def local_mean_p1(query, bank_cpu, alpha, sigma, device, ref_chunk_size=4096):
    """
    query:    [1, C, H, W] normalized latent
    bank_cpu: [N, C, H, W] normalized latent on CPU
    returns:  [1, C, H, W] posterior mean using P=1 local weights
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    _, C, H, W = query.shape
    query_hw_c = query[0].permute(1, 2, 0).contiguous().view(H * W, C)   # [HW, C]
    out_sum = torch.zeros(H * W, C, device=device, dtype=torch.float32)
    denom = torch.zeros(H * W, device=device, dtype=torch.float32)

    # stabilize softmax in chunks using running max
    running_max = torch.full((H * W,), -float("inf"), device=device, dtype=torch.float32)

    # first pass: find per-position max logit
    for start in range(0, bank_cpu.shape[0], ref_chunk_size):
        end = min(start + ref_chunk_size, bank_cpu.shape[0])
        ref = bank_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        ref_hw_c = ref.permute(0, 2, 3, 1).contiguous().view(end - start, H * W, C)  # [B,HW,C]
        diff = query_hw_c.unsqueeze(0) - alpha * ref_hw_c
        logits = -diff.square().sum(dim=-1) / (2.0 * sigma * sigma)                   # [B,HW]
        running_max = torch.maximum(running_max, logits.max(dim=0).values)
        del ref, ref_hw_c, diff, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # second pass: accumulate normalized weighted sum
    for start in range(0, bank_cpu.shape[0], ref_chunk_size):
        end = min(start + ref_chunk_size, bank_cpu.shape[0])
        ref = bank_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        ref_hw_c = ref.permute(0, 2, 3, 1).contiguous().view(end - start, H * W, C)
        diff = query_hw_c.unsqueeze(0) - alpha * ref_hw_c
        logits = -diff.square().sum(dim=-1) / (2.0 * sigma * sigma)                   # [B,HW]
        w = torch.exp(logits - running_max.unsqueeze(0))                               # [B,HW]

        denom += w.sum(dim=0)
        out_sum += (w.unsqueeze(-1) * ref_hw_c).sum(dim=0)

        del ref, ref_hw_c, diff, logits, w
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_hw_c = out_sum / denom.clamp_min(1e-12).unsqueeze(-1)
    out = out_hw_c.view(H, W, C).permute(2, 0, 1).unsqueeze(0).contiguous()
    return out


@torch.no_grad()
def model_output_xpred(model, z0_norm, eps, y, t_raw_value, model_dtype):
    """
    Keep consistency with prior diagnostic:
    use shifted time both for constructing z_t and for x_pred = z_t - t * v_pred
    """
    device = z0_norm.device
    t_raw = torch.tensor([float(t_raw_value)], device=device, dtype=torch.float64)
    t_shifted = model.shift_time(t_raw)
    t = t_shifted.to(dtype=z0_norm.dtype)
    t_view = t.view(t.shape[0], *([1] * (z0_norm.ndim - 1)))

    z_t = (1.0 - t_view) * z0_norm + t_view * eps
    v_pred = model.inference(
        z_t.to(dtype=model_dtype),
        t_shifted.to(device=device, dtype=model_dtype),
        y=y,
    ).float()
    x_pred = z_t.float() - t_view.float() * v_pred

    return {
        "t_raw": float(t_raw_value),
        "t_shifted": float(t_shifted.item()),
        "alpha": float(1.0 - t_shifted.item()),
        "sigma": float(t_shifted.item()),
        "z_t": z_t.float(),
        "v_pred": v_pred,
        "x_pred": x_pred,
    }


def contact_sheet(rows, col_titles, row_titles, save_path, cell=256, pad=24):
    n_rows = len(rows)
    n_cols = len(col_titles)
    W = n_cols * cell + (n_cols + 1) * pad
    H = n_rows * cell + (n_rows + 1) * pad + 30
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    for j, title in enumerate(col_titles):
        x = pad + j * (cell + pad)
        draw.text((x + 6, 6), title, fill="black")

    for i, title in enumerate(row_titles):
        y = 30 + pad + i * (cell + pad)
        draw.text((6, y + 6), title, fill="black")
        for j in range(n_cols):
            img = rows[i][j].resize((cell, cell))
            x = pad + j * (cell + pad)
            canvas.paste(img, (x, y))

    canvas.save(save_path)


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = get_model_dtype(args.dtype)

    model, vae, scale, bias = load_models(args, device)

    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    bank = build_or_load_bank(args, vae, scale, bias, transform, device)

    val_set = ImageNetValDataset(args.dataset, transform, max(args.val_index + 1, 1))
    img, y, name = val_set[args.val_index]
    img = img.unsqueeze(0).to(device)
    y = torch.as_tensor([y], device=device, dtype=torch.long)

    z0_raw = encode_vae(vae, img, y)
    z0_norm = normalize_latents(z0_raw, scale, bias)
    eps = torch.randn_like(z0_norm)

    original_pil = tensor_to_pil(img)
    recon_pil = tensor_to_pil(decode_vae(vae, z0_raw, y))
    original_pil.save(out_dir / "original.png")
    recon_pil.save(out_dir / "vae_reconstruction.png")

    rows = []
    row_titles = []
    summary = {
        "ckpt": str(args.ckpt),
        "dataset_ref": str(args.dataset_ref),
        "train_size": int(args.train_size),
        "dataset": str(args.dataset),
        "val_index": int(args.val_index),
        "name": str(name),
        "t_values": [float(x) for x in args.t_values],
        "images": {},
    }

    for t_raw in args.t_values:
        model_res = model_output_xpred(
            model=model,
            z0_norm=z0_norm,
            eps=eps,
            y=y,
            t_raw_value=t_raw,
            model_dtype=model_dtype,
        )

        alpha = model_res["alpha"]
        sigma = model_res["sigma"]
        z_t = model_res["z_t"]
        x_model = model_res["x_pred"]

        x_local = local_mean_p1(
            query=z_t,
            bank_cpu=bank,
            alpha=alpha,
            sigma=sigma,
            device=device,
            ref_chunk_size=args.ref_chunk_size,
        )

        noisy_img = tensor_to_pil(decode_vae(vae, denormalize_latents(z_t, scale, bias), y))
        model_img = tensor_to_pil(decode_vae(vae, denormalize_latents(x_model, scale, bias), y))
        local_img = tensor_to_pil(decode_vae(vae, denormalize_latents(x_local, scale, bias), y))

        tag = str(t_raw).replace(".", "p")
        noisy_path = out_dir / f"t{tag}_noisy_decode.png"
        model_path = out_dir / f"t{tag}_model_output_xpred.png"
        local_path = out_dir / f"t{tag}_local_p1_xpred.png"

        noisy_img.save(noisy_path)
        model_img.save(model_path)
        local_img.save(local_path)

        torch.save(
            {
                "t_raw": model_res["t_raw"],
                "t_shifted": model_res["t_shifted"],
                "alpha": alpha,
                "sigma": sigma,
                "z_t": z_t.cpu(),
                "model_xpred": x_model.cpu(),
                "local_xpred": x_local.cpu(),
            },
            out_dir / f"t{tag}_debug.pt",
        )

        print(
            f"[t={t_raw}] shifted_t={model_res['t_shifted']:.6f} "
            f"alpha={alpha:.6f} sigma={sigma:.6f} | "
            f"z_t_rms={z_t.square().mean().sqrt().item():.4f} | "
            f"model_xpred_rms={x_model.square().mean().sqrt().item():.4f} | "
            f"local_xpred_rms={x_local.square().mean().sqrt().item():.4f}"
        )

        rows.append([original_pil, recon_pil, noisy_img, model_img, local_img])
        row_titles.append(f"t={t_raw}")
        summary["images"][str(t_raw)] = {
            "noisy_decode": str(noisy_path),
            "model_output_xpred": str(model_path),
            "local_p1_xpred": str(local_path),
        }

    sheet_path = out_dir / "comparison_model_vs_local.png"
    contact_sheet(
        rows=rows,
        col_titles=["Original", "VAE recon", "Noisy decode", "Model output", "Local P=1"],
        row_titles=row_titles,
        save_path=sheet_path,
        cell=args.resolution,
    )
    summary["sheet"] = str(sheet_path)

    (out_dir / "run.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved comparison sheet:", sheet_path)
    print("Saved run info:", out_dir / "run.json")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--vae-config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset-ref", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--val-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--train-size", type=int, default=10000)
    p.add_argument("--ref-chunk-size", type=int, default=4096)
    p.add_argument("--cache-path", type=str, default="")
    p.add_argument("--rebuild-cache", action="store_true")

    p.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=[0.8, 0.6, 0.4, 0.2],
    )

    p.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )

    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--out-dir", type=str, default="./model_vs_local_check")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
