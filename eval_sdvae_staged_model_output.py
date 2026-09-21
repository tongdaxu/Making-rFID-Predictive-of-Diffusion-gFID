import argparse
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image
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

    raise RuntimeError(f"Unsupported latent shape: {tuple(z.shape)}")


def normalize_latents(z, scale, bias):
    return (z - bias) * scale


def denormalize_latents(z, scale, bias):
    return z / scale + bias


def encode_vae(vae, img, y=None):
    try:
        z = vae.encode(img, y)
    except TypeError:
        z = vae.encode(img)
    return prepare_latents_for_sit(z)


def decode_vae(vae, z, y=None):
    try:
        x = vae.decode(z, y)
    except TypeError:
        x = vae.decode(z)
    if isinstance(x, (tuple, list)):
        x = x[0]
    return x


def tensor_to_pil(x):
    # VAE output is expected in [-1, 1]
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


@torch.no_grad()
def load_models(args, device):
    # -----------------------
    # VAE
    # -----------------------
    vae_cfg = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_cfg).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # infer latent shape exactly as training code does
    fake_img = torch.zeros(
        1, 3, args.resolution, args.resolution,
        device=device, dtype=torch.float32
    )
    fake_z = encode_vae(vae, fake_img)
    _, in_channels, latent_h, latent_w = fake_z.shape
    if latent_h != latent_w:
        raise RuntimeError(f"Expected square spatial latent, got {fake_z.shape}")
    latent_size = latent_h

    tshift = math.sqrt(float(fake_z[0].numel()) / 4096.0)

    # -----------------------
    # checkpoint
    # -----------------------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "args" not in ckpt:
        raise KeyError("Checkpoint does not contain ckpt['args']")
    train_args = ckpt["args"]

    # args can be Namespace-like
    model_name = getattr(train_args, "model")
    num_classes = getattr(train_args, "num_classes", 1000)
    cfg_prob = getattr(train_args, "cfg_prob", 0.1)
    bn_momentum = getattr(train_args, "bn_momentum", 0.1)
    fused_attn = getattr(train_args, "fused_attn", True)
    qk_norm = getattr(train_args, "qk_norm", False)

    print("[load] model:", model_name)
    print("[load] latent shape:", tuple(fake_z.shape))
    print("[load] tshift:", tshift)

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
    if state_key not in ckpt:
        raise KeyError(f"Checkpoint does not contain ckpt['{state_key}']")
    model.load_state_dict(ckpt[state_key], strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Same latent normalization stats used by sampling code.
    stats = model.extract_latents_stats()

    scale = stats["latents_scale"].float().to(device)
    bias = stats["latents_bias"].float().to(device)

    if fake_z.ndim == 4:
        scale = scale.view(1, in_channels, 1, 1)
        bias = bias.view(1, in_channels, 1, 1)
    else:
        raise RuntimeError("This diagnostic script currently expects spatial 4D latent")

    print("[load] latent scale:", scale.flatten().cpu().tolist())
    print("[load] latent bias :", bias.flatten().cpu().tolist())

    return model, vae, scale, bias


@torch.no_grad()
def get_model_output_xpred(
    model,
    z0_norm,
    eps,
    y,
    t_raw_value,
    model_dtype,
):
    """
    Check REAL SiT model output at one time.

    The existing sampler does:
        t_steps = model.shift_time(t_steps)
        model.inference(x_t, t_shifted, ...)

    To keep the state and model time consistent here, this diagnostic uses
    the shifted time BOTH for constructing x_t and for v -> x_pred.

    Linear path:
        x_t = (1-t) x0 + t eps
        v   = eps - x0
        x0  = x_t - t v
    """
    device = z0_norm.device

    t_raw = torch.tensor([float(t_raw_value)], device=device, dtype=torch.float64)
    t_shifted = model.shift_time(t_raw)

    # broadcast shifted physical time over latent
    t = t_shifted.to(dtype=z0_norm.dtype)
    t_view = t.view(t.shape[0], *([1] * (z0_norm.ndim - 1)))

    # construct the noisy latent in normalized SiT space
    z_t = (1.0 - t_view) * z0_norm + t_view * eps

    # REAL model output: velocity
    v_pred = model.inference(
        z_t.to(dtype=model_dtype),
        t_shifted.to(device=device, dtype=model_dtype),
        y=y,
    ).float()

    # velocity -> clean x prediction
    x_pred = z_t.float() - t_view.float() * v_pred

    return {
        "t_raw": float(t_raw_value),
        "t_shifted": float(t_shifted.item()),
        "z_t": z_t.float(),
        "v_pred": v_pred,
        "x_pred": x_pred,
    }


@torch.no_grad()
def main(args):
    """
    Rewritten version of the previously buggy staged figure.

    IMPORTANT:
    - No Local / Ideal / latent bank.
    - Each stage uses the REAL SiT model output.
    - The model predicts velocity; we convert it to x_pred.
    - The previous stage's x_pred is treated as the clean latent for the next stage.

    Flow:
        z0
          -> add t=0.8 noise -> SiT v_pred -> x_pred_1
          -> add t=0.4 noise to x_pred_1 -> SiT v_pred -> x_pred_2
          -> add t=0.2 noise to x_pred_2 -> SiT v_pred -> x_pred_3
          -> decode x_pred_3
    """
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

    val_set = ImageNetValDataset(
        args.dataset,
        transform,
        max(args.val_index + 1, 1),
    )
    img, y, name = val_set[args.val_index]
    img = img.unsqueeze(0).to(device)
    y = torch.as_tensor([y], device=device, dtype=torch.long)

    # VAE raw latent -> SiT normalized latent
    z0_raw = encode_vae(vae, img, y)
    z0_norm = normalize_latents(z0_raw, scale, bias)

    tensor_to_pil(img).save(out_dir / "original.png")
    tensor_to_pil(decode_vae(vae, z0_raw, y)).save(
        out_dir / "vae_reconstruction.png"
    )

    # Controlled default: reuse the same epsilon at all stages.
    # Use --fresh-noise-each-stage if desired.
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    shared_eps = torch.randn(
        z0_norm.shape,
        device=device,
        dtype=z0_norm.dtype,
        generator=gen,
    )

    current_clean = z0_norm

    from PIL import ImageDraw

    rows = []
    row_titles = []
    records = []

    for stage_idx, t_raw in enumerate(args.stage_t_values, start=1):
        # Same time treatment as the validated model-output script.
        t_raw_tensor = torch.tensor(
            [float(t_raw)],
            device=device,
            dtype=torch.float64,
        )
        t_shifted = model.shift_time(t_raw_tensor)
        t = t_shifted.to(dtype=current_clean.dtype)
        t_view = t.view(
            t.shape[0],
            *([1] * (current_clean.ndim - 1)),
        )

        if args.fresh_noise_each_stage:
            eps = torch.randn(
                current_clean.shape,
                device=device,
                dtype=current_clean.dtype,
                generator=gen,
            )
        else:
            eps = shared_eps

        # Re-noise PREVIOUS x_pred using the current stage t.
        z_t = (1.0 - t_view) * current_clean + t_view * eps

        # REAL MODEL OUTPUT = velocity
        v_pred = model.inference(
            z_t.to(dtype=model_dtype),
            t_shifted.to(device=device, dtype=model_dtype),
            y=y,
        ).float()

        # linear path: x_pred = x_t - t * v_pred
        x_pred = z_t.float() - t_view.float() * v_pred

        noisy_raw = denormalize_latents(z_t, scale, bias)
        xpred_raw = denormalize_latents(x_pred, scale, bias)

        noisy_img = decode_vae(vae, noisy_raw, y)
        xpred_img = decode_vae(vae, xpred_raw, y)

        noisy_pil = tensor_to_pil(noisy_img)
        xpred_pil = tensor_to_pil(xpred_img)

        tag = str(t_raw).replace(".", "p")
        noisy_path = out_dir / f"stage{stage_idx}_t{tag}_renoised_input.png"
        pred_path = out_dir / f"stage{stage_idx}_t{tag}_model_output_xpred.png"

        noisy_pil.save(noisy_path)
        xpred_pil.save(pred_path)

        rows.append([noisy_pil, xpred_pil])
        row_titles.append(f"stage {stage_idx}, t={t_raw:g}")

        records.append({
            "stage": stage_idx,
            "t_raw": float(t_raw),
            "t_shifted": float(t_shifted.item()),
            "renoised_input": str(noisy_path),
            "model_output_xpred": str(pred_path),
            "zt_rms": float(z_t.square().mean().sqrt().item()),
            "vpred_rms": float(v_pred.square().mean().sqrt().item()),
            "xpred_rms": float(x_pred.square().mean().sqrt().item()),
        })

        print(
            f"[stage {stage_idx}] raw_t={t_raw:.4f} "
            f"shifted_t={t_shifted.item():.6f} | "
            f"zt_rms={records[-1]['zt_rms']:.4f} | "
            f"vpred_rms={records[-1]['vpred_rms']:.4f} | "
            f"xpred_rms={records[-1]['xpred_rms']:.4f}"
        )

        # IMPORTANT: use the model x_pred as the next stage's clean latent.
        current_clean = x_pred

    # final model prediction decode
    final_raw = denormalize_latents(current_clean, scale, bias)
    final_img = decode_vae(vae, final_raw, y)
    final_path = out_dir / "FINAL_model_output.png"
    tensor_to_pil(final_img).save(final_path)

    # Recreate the old two-column figure, but correctly:
    # left = re-noised input, right = REAL model output.
    cell = args.resolution
    pad = 28
    top = 32
    n_rows = len(rows)
    n_cols = 2
    W = n_cols * cell + (n_cols + 1) * pad
    H = n_rows * cell + (n_rows + 1) * pad + top

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    col_titles = ["re-noised input", "Model output"]

    for j, title in enumerate(col_titles):
        x = pad + j * (cell + pad)
        draw.text((x + 6, 8), title, fill="black")

    for i, title in enumerate(row_titles):
        y0 = top + pad + i * (cell + pad)
        draw.text((6, y0 + 6), title, fill="black")
        for j in range(2):
            img_j = rows[i][j].resize((cell, cell))
            x0 = pad + j * (cell + pad)
            canvas.paste(img_j, (x0, y0))

    sheet_path = out_dir / "staged_model_output.png"
    canvas.save(sheet_path)

    report = {
        "ckpt": str(args.ckpt),
        "dataset": str(args.dataset),
        "val_index": int(args.val_index),
        "name": str(name),
        "stage_t_values": [float(x) for x in args.stage_t_values],
        "fresh_noise_each_stage": bool(args.fresh_noise_each_stage),
        "records": records,
        "final": str(final_path),
        "sheet": str(sheet_path),
    }
    (out_dir / "run.json").write_text(json.dumps(report, indent=2))

    print("\nFinal:", final_path)
    print("Comparison:", sheet_path)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--vae-config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--val-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--stage-t-values",
        type=float,
        nargs="+",
        default=[0.8, 0.4, 0.2],
    )

    p.add_argument(
        "--fresh-noise-each-stage",
        action="store_true",
        help="Use independent Gaussian noise at each stage. Default reuses one eps.",
    )

    p.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )

    p.add_argument(
        "--use-ema",
        action="store_true",
        help="Load ckpt['ema'] instead of ckpt['model']",
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default="./staged_model_output_check",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
