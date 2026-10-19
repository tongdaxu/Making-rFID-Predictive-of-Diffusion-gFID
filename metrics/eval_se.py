# eval_se.py
"""
VAE-only SE (Scale-Equivariance) eval on REAL images.

Current config-based interface:

nohup env CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port 29505 eval_se.py \
  --base_path /video_ssd/kongzishang/imagenet_val \
  --vae-config ./configs/FLUX2VAE.yaml \
  > /video_ssd/kongzishang/xutongda/git/IFID/losses/eval_se_flux2vae.out 2>&1 &
"""

import os
import math
import time
import argparse
import random
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

import lpips


# -------------------------
# config instantiate
# -------------------------
def load_config(path: str):
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)


def get_obj_from_str(string: str):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Config must contain field `target`.")

    target = config["target"]
    params = config.get("params", {}) or {}
    cls = get_obj_from_str(target)
    return cls(**params)


# -------------------------
# preprocess
# -------------------------
def preprocess_imgs_vae(imgs: torch.Tensor) -> torch.Tensor:
    # imgs: B,C,H,W in [0,255] -> [-1,1]
    return imgs.float() / 127.5 - 1.0


def _vae_forward_get_posterior_and_z(vae, processed_image: torch.Tensor):
    """
    Compatible with current Flux2VAE interface:
        z = vae.encode(x)

    Also supports:
        encode(x).latent_dist
        encode(x).sample()
        forward outputs like (posterior, z)
    """
    if hasattr(vae, "encode") and callable(getattr(vae, "encode")):
        out = vae.encode(processed_image)

        # Flux2VAE path: encode(x) returns z directly.
        if torch.is_tensor(out):
            return None, out

        # diffusers style: encode(x).latent_dist
        if hasattr(out, "latent_dist"):
            posterior = out.latent_dist
            if hasattr(posterior, "sample") and callable(getattr(posterior, "sample")):
                z = posterior.sample()
            elif hasattr(posterior, "mean") and torch.is_tensor(posterior.mean):
                z = posterior.mean
            else:
                raise RuntimeError("Cannot obtain z from out.latent_dist.")
            return posterior, z

        # posterior directly
        if hasattr(out, "sample") and callable(getattr(out, "sample")):
            posterior = out
            z = posterior.sample()
            return posterior, z

        if hasattr(out, "mean") and torch.is_tensor(out.mean):
            posterior = out
            z = posterior.mean
            return posterior, z

        # tuple/list from encode
        if isinstance(out, (tuple, list)):
            if len(out) >= 2 and torch.is_tensor(out[1]):
                return out[0], out[1]
            if len(out) >= 1 and torch.is_tensor(out[0]):
                return None, out[0]

        raise RuntimeError("vae.encode(x) returned unsupported output type.")

    # fallback: old forward interface
    try:
        out = vae(processed_image, return_recon=False)
    except TypeError:
        out = vae(processed_image)

    if isinstance(out, (tuple, list)):
        if len(out) >= 3 and torch.is_tensor(out[2]):
            return out[1], out[2]
        if len(out) >= 2 and torch.is_tensor(out[1]):
            return out[0], out[1]

    raise RuntimeError("Unexpected VAE output; cannot obtain posterior and z.")


def _vae_decode_from_z(vae, z: torch.Tensor) -> torch.Tensor:
    """
    Compatible with:
      1. Flux2VAE.decode(z) -> Tensor
      2. diffusers decode(z).sample
      3. old vae.decoder(z)
    """
    if hasattr(vae, "decode") and callable(getattr(vae, "decode")):
        out = vae.decode(z)

        if torch.is_tensor(out):
            return out

        if hasattr(out, "sample") and torch.is_tensor(out.sample):
            return out.sample

        if isinstance(out, dict) and "sample" in out and torch.is_tensor(out["sample"]):
            return out["sample"]

        if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0]

    if hasattr(vae, "decoder") and callable(getattr(vae, "decoder")):
        out = vae.decoder(z)

        if torch.is_tensor(out):
            return out

        if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0]

    raise RuntimeError("Cannot decode from z. Please adapt _vae_decode_from_z().")


def _downsample_image(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return x

    h, w = x.shape[-2:]
    return F.interpolate(
        x,
        size=(h // scale, w // scale),
        mode="bilinear",
        align_corners=False,
    )


def is_rae_like_vae(vae) -> bool:
    """
    RAE / ScaleRAE usually has fixed latent_mean / latent_var spatial stats.
    Resizing its latent may crash decode or make the metric invalid.
    """
    cls_name = vae.__class__.__name__.lower()
    module_name = vae.__class__.__module__.lower()

    if hasattr(vae, "rae"):
        return True
    if hasattr(vae, "model") and hasattr(vae.model, "rae"):
        return True
    if "rae" in cls_name:
        return True
    if ".rae" in module_name or "rae_module" in module_name:
        return True

    return False


def _downsample_latent_for_decode(
    vae,
    z: torch.Tensor,
    scale: int,
    *,
    allow_token_latent_downsample: bool = False,
):
    """
    Safe latent downsample for SE eval.

    Returns:
      z_tilde, applied_latent_scale

    Rules:
      - 4D B,C,H,W latents: downsample normally.
      - RAE-like latents: never downsample, because decode often expects fixed H,W stats.
      - 3D token latents B,L,C / B,C,L: default to identity because many token
        decoders require fixed token count. Set --allow-token-latent-downsample 1
        to use the existing square-token-grid downsample path.
      - 2D/global latents: identity fallback.
    """
    if scale == 1:
        return z, True

    if is_rae_like_vae(vae):
        return z, False

    if z.dim() == 4:
        return _downsample_latent(z, scale), True

    if z.dim() == 3 and allow_token_latent_downsample:
        try:
            return _downsample_latent(z, scale), True
        except Exception:
            return z, False

    return z, False


def _downsample_latent(z: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Support latents in:
      - B,C,H,W
      - B,C,L
      - B,L,C
    """
    if scale == 1:
        return z

    if z.dim() == 4:
        h, w = z.shape[-2:]
        if h // scale < 1 or w // scale < 1:
            raise ValueError(f"scale={scale} too large for latent shape {tuple(z.shape)}")

        return F.interpolate(
            z,
            size=(h // scale, w // scale),
            mode="bilinear",
            align_corners=False,
        )

    if z.dim() != 3:
        raise ValueError(f"Unsupported latent dim={z.dim()} with shape={tuple(z.shape)}")

    B0, D1, D2 = z.shape

    def is_square(n: int) -> bool:
        r = int(math.isqrt(n))
        return r * r == n

    if is_square(D1) and is_square(D2):
        # Treat larger square as token count.
        if D1 >= D2:
            layout = "BLC"
            L, C = D1, D2
            z_bcl = z.permute(0, 2, 1).contiguous()
        else:
            layout = "BCL"
            C, L = D1, D2
            z_bcl = z
    elif is_square(D1):
        layout = "BLC"
        L, C = D1, D2
        z_bcl = z.permute(0, 2, 1).contiguous()
    elif is_square(D2):
        layout = "BCL"
        C, L = D1, D2
        z_bcl = z
    else:
        raise ValueError(f"Cannot infer token grid from latent shape {tuple(z.shape)}.")

    h = int(math.isqrt(L))
    w = h

    if h // scale < 1 or w // scale < 1:
        raise ValueError(f"scale={scale} too large for latent token grid {h}x{w}.")

    z2d = z_bcl.view(B0, C, h, w)
    z2d_ds = F.interpolate(
        z2d,
        size=(h // scale, w // scale),
        mode="bilinear",
        align_corners=False,
    )

    z_bcl_ds = z2d_ds.flatten(2)

    if layout == "BCL":
        return z_bcl_ds
    else:
        return z_bcl_ds.permute(0, 2, 1).contiguous()


@torch.no_grad()
def distance(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    net_lpips: nn.Module,
    lambda_l2: float,
    lambda_lpips: float,
):
    pred_f = pred.float()
    tgt_f = tgt.float()

    l2 = F.mse_loss(pred_f, tgt_f, reduction="mean")
    lp = net_lpips(pred_f, tgt_f).mean()
    loss = l2 * lambda_l2 + lp * lambda_lpips

    return loss, l2, lp


@torch.no_grad()
def try_kl_from_posterior(posterior):
    if posterior is None:
        return None

    if hasattr(posterior, "kl") and callable(getattr(posterior, "kl")):
        try:
            kl = posterior.kl()
            if torch.is_tensor(kl):
                return kl.mean()
        except Exception:
            pass

    if hasattr(posterior, "kl") and torch.is_tensor(getattr(posterior, "kl")):
        return getattr(posterior, "kl").mean()

    for mean_key in ("mean", "mu"):
        if hasattr(posterior, mean_key) and hasattr(posterior, "logvar"):
            mu = getattr(posterior, mean_key)
            logvar = getattr(posterior, "logvar")

            if torch.is_tensor(mu) and torch.is_tensor(logvar):
                kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
                return kl.flatten(1).sum(dim=1).mean()

    return None


# -------------------------
# dataset
# -------------------------
class ImageNetFolderDataset(Dataset):
    """
    ImageNet-style folder:
        root/class_name/*.jpg
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        self.samples = []
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())

        if len(classes) == 0:
            raise RuntimeError(f"No class folders found under: {root}")

        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(exts):
                    path = os.path.join(cls_dir, fname)
                    label = class_to_idx[cls_name]
                    self.samples.append((path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {root}")

        print(f"[Dataset] root       : {root}")
        print(f"[Dataset] num classes: {len(classes)}")
        print(f"[Dataset] num images : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# -------------------------
# VAE load
# -------------------------
def load_vae(args, device, accelerator=None):
    cfg = load_config(args.vae_config)

    if (accelerator is None) or accelerator.is_main_process:
        print(f"[VAE] config: {args.vae_config}")
        print(f"[VAE] target: {cfg.get('target')}")

    vae = instantiate_from_config(cfg)
    vae = vae.to(device)
    vae.eval()

    for p in vae.parameters():
        p.requires_grad_(False)

    return vae


# -------------------------
# eval
# -------------------------
@torch.no_grad()
def eval_se_over_dataset(image_root, vae, net_lpips, args, device, accelerator: Accelerator):
    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    ds = ImageNetFolderDataset(root=image_root, transform=transform)

    loader = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    loader = accelerator.prepare(loader)

    scales = list(args.scales)

    n_total = torch.tensor(0.0, device=device)

    sum_recon = torch.tensor(0.0, device=device)
    sum_se_raw = torch.tensor(0.0, device=device)
    sum_obj_no_kl = torch.tensor(0.0, device=device)
    sum_total = torch.tensor(0.0, device=device)
    sum_kl = torch.tensor(0.0, device=device)

    sum_recon_l2 = torch.tensor(0.0, device=device)
    sum_recon_lp = torch.tensor(0.0, device=device)
    sum_se_l2 = torch.tensor(0.0, device=device)
    sum_se_lp = torch.tensor(0.0, device=device)

    per_scale_n = {s: torch.tensor(0.0, device=device) for s in scales}
    per_scale_sum = {s: torch.tensor(0.0, device=device) for s in scales}

    warned_kl = False

    pbar = tqdm(
        loader,
        desc="Eval SE",
        disable=not accelerator.is_local_main_process,
    )

    for img01, _ in pbar:
        img01 = img01.to(device, non_blocking=True)

        raw255 = img01 * 255.0
        x = preprocess_imgs_vae(raw255)

        posterior, z = _vae_forward_get_posterior_and_z(vae, x)

        # recon: Dec(z)
        x_hat = _vae_decode_from_z(vae, z)

        if x_hat.shape[-2:] != x.shape[-2:]:
            x_hat = F.interpolate(
                x_hat,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if args.clamp_decode:
            x_hat = x_hat.clamp(-1.0, 1.0)

        recon, recon_l2, recon_lp = distance(
            x_hat,
            x,
            net_lpips,
            args.lambda_l2,
            args.lambda_lpips,
        )

        # pick scale
        scale = scales[0]
        if args.random_scale and len(scales) > 1:
            idx = torch.randint(low=0, high=len(scales), size=(1,), device=device).item()
            scale = scales[idx]

        # SE: x_tilde vs Dec(z_tilde)
        # For 1D/token/global latents, changing token count is often not decodable.
        # In that case we fall back to identity so the script stays compatible.
        z_tilde, applied_latent_scale = _downsample_latent_for_decode(
            vae,
            z,
            scale,
            allow_token_latent_downsample=bool(args.allow_token_latent_downsample),
        )

        if applied_latent_scale:
            x_tilde = _downsample_image(x, scale)
        else:
            # compatibility fallback: this SE term degenerates to recon-style loss.
            x_tilde = x

        if accelerator.is_main_process and int(n_total.item()) == 0:
            print("z.shape              =", tuple(z.shape))
            print("z_tilde.shape        =", tuple(z_tilde.shape))
            print("x.shape              =", tuple(x.shape))
            print("x_tilde.shape        =", tuple(x_tilde.shape))
            print("applied_latent_scale =", applied_latent_scale)

        x_hat_tilde = _vae_decode_from_z(vae, z_tilde)

        if x_hat_tilde.shape[-2:] != x_tilde.shape[-2:]:
            x_hat_tilde = F.interpolate(
                x_hat_tilde,
                size=x_tilde.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if args.clamp_decode:
            x_hat_tilde = x_hat_tilde.clamp(-1.0, 1.0)

        se_raw, se_l2, se_lp = distance(
            x_hat_tilde,
            x_tilde,
            net_lpips,
            args.lambda_l2,
            args.lambda_lpips,
        )

        se_weighted = se_raw * args.alpha
        objective_no_kl = recon + se_weighted

        kl = torch.tensor(0.0, device=device)
        if args.beta > 0:
            kl_val = try_kl_from_posterior(posterior)

            if kl_val is None:
                if (not warned_kl) and accelerator.is_main_process:
                    print("[Warn] beta>0 but KL is unavailable from posterior; KL will be treated as 0.")
                    warned_kl = True
            else:
                kl = kl_val.to(device)

        total_loss = objective_no_kl + args.beta * kl

        bs = x.size(0)
        bs_t = torch.tensor(float(bs), device=device)

        n_total += bs_t

        sum_recon += recon * bs
        sum_se_raw += se_raw * bs
        sum_obj_no_kl += objective_no_kl * bs
        sum_total += total_loss * bs
        sum_kl += kl * bs

        sum_recon_l2 += recon_l2 * bs
        sum_recon_lp += recon_lp * bs
        sum_se_l2 += se_l2 * bs
        sum_se_lp += se_lp * bs

        per_scale_n[scale] += bs_t
        per_scale_sum[scale] += se_raw * bs

        mean_recon = (sum_recon / torch.clamp(n_total, min=1.0)).item()
        mean_se_raw = (sum_se_raw / torch.clamp(n_total, min=1.0)).item()
        mean_obj = (sum_obj_no_kl / torch.clamp(n_total, min=1.0)).item()
        mean_total = (sum_total / torch.clamp(n_total, min=1.0)).item()

        pbar.set_postfix(
            {
                "recon": f"{mean_recon:.6f}",
                "se_raw": f"{mean_se_raw:.6f}",
                "obj_no_kl": f"{mean_obj:.6f}",
                "total": f"{mean_total:.6f}",
                "scale": scale,
                "n": int(n_total.item()),
            }
        )

        if args.num_real > 0 and int(n_total.item()) >= args.num_real:
            break

    # reduce across processes
    n_total = accelerator.reduce(n_total, reduction="sum")

    sum_recon = accelerator.reduce(sum_recon, reduction="sum")
    sum_se_raw = accelerator.reduce(sum_se_raw, reduction="sum")
    sum_obj_no_kl = accelerator.reduce(sum_obj_no_kl, reduction="sum")
    sum_total = accelerator.reduce(sum_total, reduction="sum")
    sum_kl = accelerator.reduce(sum_kl, reduction="sum")

    sum_recon_l2 = accelerator.reduce(sum_recon_l2, reduction="sum")
    sum_recon_lp = accelerator.reduce(sum_recon_lp, reduction="sum")
    sum_se_l2 = accelerator.reduce(sum_se_l2, reduction="sum")
    sum_se_lp = accelerator.reduce(sum_se_lp, reduction="sum")

    for s in scales:
        per_scale_n[s] = accelerator.reduce(per_scale_n[s], reduction="sum")
        per_scale_sum[s] = accelerator.reduce(per_scale_sum[s], reduction="sum")

    n = max(1.0, float(n_total.item()))

    stats = {
        "n_total": int(n_total.item()),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "recon_mean": float((sum_recon / n).item()),
        "se_raw_mean": float((sum_se_raw / n).item()),
        "se_weighted_mean": float((sum_se_raw * args.alpha / n).item()),
        "objective_no_kl_mean": float((sum_obj_no_kl / n).item()),
        "kl_mean": float((sum_kl / n).item()),
        "total_loss_mean": float((sum_total / n).item()),
        "recon_l2_mean": float((sum_recon_l2 / n).item()),
        "recon_lpips_mean": float((sum_recon_lp / n).item()),
        "se_l2_mean": float((sum_se_l2 / n).item()),
        "se_lpips_mean": float((sum_se_lp / n).item()),
    }

    for s in scales:
        n_s = float(per_scale_n[s].item())
        if n_s > 0:
            stats[f"se_raw_mean_s{s}"] = float((per_scale_sum[s] / n_s).item())

    return stats


# -------------------------
# argparse + main
# -------------------------
def build_parser():
    p = argparse.ArgumentParser(
        "VAE-only SE eval. d = L2 + LPIPS, config-based VAE interface."
    )

    p.add_argument("--base_path", "--base-path", dest="base_path", type=str, required=True)
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", type=str, default="./losses/se")

    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=8)

    # VAE config
    p.add_argument("--vae-config", type=str, required=True)

    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num_real", "--num-real", dest="num_real", type=int, default=50000)

    # SE setup
    p.add_argument("--scales", type=int, nargs="+", default=[2, 4])
    p.add_argument(
        "--random_scale",
        "--random-scale",
        dest="random_scale",
        type=int,
        default=0,
        help="1: random pick from scales per batch; 0: always use scales[0]",
    )

    # objective weights
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--beta", type=float, default=0.0)

    # distance = lambda_l2 * MSE + lambda_lpips * LPIPS
    p.add_argument("--lambda_l2", "--lambda-l2", dest="lambda_l2", type=float, default=1.0)
    p.add_argument("--lambda_lpips", "--lambda-lpips", dest="lambda_lpips", type=float, default=1.0)
    p.add_argument("--lpips_net", "--lpips-net", dest="lpips_net", type=str, default="vgg", choices=["vgg", "alex", "squeeze"])

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clamp_decode", "--clamp-decode", dest="clamp_decode", type=int, default=0)
    p.add_argument(
        "--allow_token_latent_downsample",
        "--allow-token-latent-downsample",
        dest="allow_token_latent_downsample",
        type=int,
        default=0,
        help="1: allow downsampling square token latents B,L,C/B,C,L; 0: use identity fallback for token/global latents.",
    )

    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print("[Device]", device)

    set_seed(args.seed + accelerator.process_index)
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)

    accelerator.wait_for_everyone()

    vae = load_vae(args, device, accelerator)

    net_lpips = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    for p in net_lpips.parameters():
        p.requires_grad_(False)

    t0 = time.time()

    stats = eval_se_over_dataset(
        image_root=args.base_path,
        vae=vae,
        net_lpips=net_lpips,
        args=args,
        device=device,
        accelerator=accelerator,
    )

    dt = time.time() - t0

    if accelerator.is_main_process:
        print("\n================ SE EVAL RESULT ================")
        print(f"vae_config           : {args.vae_config}")
        print(f"recon_mean           : {stats['recon_mean']:.8f}")
        print(f"se_raw_mean          : {stats['se_raw_mean']:.8f}")
        print(f"se_weighted_mean     : {stats['se_weighted_mean']:.8f}")
        print(f"objective_no_kl_mean : {stats['objective_no_kl_mean']:.8f}")
        print(f"total_loss_mean      : {stats['total_loss_mean']:.8f}")

        print(f"alpha                : {stats['alpha']:.4f}")
        print(f"beta                 : {stats['beta']:.4f}")
        print(f"kl_mean              : {stats['kl_mean']:.8f}")

        print(f"recon_l2_mean        : {stats['recon_l2_mean']:.8f}")
        print(f"recon_lpips_mean     : {stats['recon_lpips_mean']:.8f}")
        print(f"se_l2_mean           : {stats['se_l2_mean']:.8f}")
        print(f"se_lpips_mean        : {stats['se_lpips_mean']:.8f}")

        for k in sorted([k for k in stats.keys() if k.startswith("se_raw_mean_s")]):
            print(f"{k:20s}: {stats[k]:.8f}")

        print(f"n_total              : {stats['n_total']}")
        print(f"time                 : {dt:.2f}s")
        print("================================================\n")

    if accelerator.is_main_process:
        print("SE evaluation complete!")


if __name__ == "__main__":
    main()