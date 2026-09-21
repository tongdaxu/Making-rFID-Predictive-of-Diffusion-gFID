import os
import time
import argparse
import random
import importlib
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

import lpips
import math


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


def _vae_get_posterior_and_z(vae, x: torch.Tensor):
    """
    Current interface:
        Flux2VAE.encode(x) -> z

    Also compatible with:
        diffusers AutoencoderKL.encode(x).latent_dist
        old VAE forward outputs
    """
    if hasattr(vae, "encode") and callable(getattr(vae, "encode")):
        out = vae.encode(x)

        # Flux2VAE path: encode(x) returns z directly.
        if torch.is_tensor(out):
            return None, out

        # diffusers AutoencoderKL path: encode(x) returns object with latent_dist.
        if hasattr(out, "latent_dist"):
            posterior = out.latent_dist
            if hasattr(posterior, "sample") and callable(getattr(posterior, "sample")):
                z = posterior.sample()
            elif hasattr(posterior, "mean") and torch.is_tensor(posterior.mean):
                z = posterior.mean
            else:
                raise RuntimeError("Cannot obtain z from out.latent_dist.")
            return posterior, z

        # Some custom VAEs return posterior directly.
        if hasattr(out, "sample") and callable(getattr(out, "sample")):
            posterior = out
            z = posterior.sample()
            return posterior, z

        if hasattr(out, "mean") and torch.is_tensor(out.mean):
            posterior = out
            z = out.mean
            return posterior, z

        raise RuntimeError("vae.encode(x) returned unsupported type.")

    # fallback: parse forward outputs
    try:
        out = vae(x, return_recon=False)
    except TypeError:
        out = vae(x)

    if isinstance(out, (tuple, list)):
        # common:
        #   (posterior, z)
        #   (recon, posterior, z)
        if len(out) >= 3 and torch.is_tensor(out[2]):
            return out[1], out[2]
        if len(out) >= 2 and torch.is_tensor(out[1]):
            return out[0], out[1]

    raise RuntimeError("Cannot obtain (posterior, z).")


def _vae_decode_from_z(vae, z: torch.Tensor):
    if not (hasattr(vae, "decode") and callable(getattr(vae, "decode"))):
        raise RuntimeError("VAE has no .decode(z).")

    out = vae.decode(z)

    # Flux2VAE path: decode(z) returns tensor.
    if torch.is_tensor(out):
        return out

    # diffusers path: decode(z).sample
    if hasattr(out, "sample") and torch.is_tensor(out.sample):
        return out.sample

    if isinstance(out, dict) and "sample" in out and torch.is_tensor(out["sample"]):
        return out["sample"]

    raise RuntimeError(f"vae.decode(z) returned {type(out)}; cannot extract recon tensor.")


# -------------------------
# EQ transform tau
# -------------------------
def sample_eq_transform(*, p_alpha: float, anisotropic: bool, allow_scale: bool):
    if random.random() < p_alpha:
        return 1.0, 0

    angle = random.choice([1, 2, 3])

    if not allow_scale:
        return 1.0, angle

    choices = [s / 32.0 for s in range(8, 32)]

    if anisotropic:
        scale = (random.choice(choices), random.choice(choices))
    else:
        scale = random.choice(choices)

    return scale, angle


def apply_tau(x: torch.Tensor, scale, angle: int, interp: str):
    """
    Apply tau = scale + rot90 on BCHW tensor.
    """
    if scale != 1 and scale != 1.0:
        x = F.interpolate(x, scale_factor=scale, mode=interp, align_corners=False)

    if angle != 0:
        x = torch.rot90(x, k=angle, dims=[-1, -2]).contiguous()

    return x

def _is_identity_scale(scale):
    return scale == 1 or scale == 1.0


def _square_size(n: int):
    s = int(round(math.sqrt(n)))
    if s * s == n:
        return s
    return None


def infer_latent_tau_kind(z: torch.Tensor):
    """
    Return:
      - "bchw"     : spatial latent, shape B,C,H,W
      - "blc_grid" : token latent, shape B,L,C, and L=S*S
      - "none"     : no safe spatial tau can be applied
    """
    if z.dim() == 4:
        return "bchw"

    if z.dim() == 3:
        # common token latent: B,L,C
        L = z.shape[1]
        s = _square_size(L)
        if s is not None:
            return "blc_grid"

    return "none"


def apply_tau_to_latent(z: torch.Tensor, scale, angle: int, interp: str):
    """
    Apply tau to latent safely.

    Supported:
      - B,C,H,W: scale + rot90
      - B,L,C with L=S*S: rot90 only, no scale

    Unsupported:
      - return None
    """
    kind = infer_latent_tau_kind(z)

    if kind == "bchw":
        return apply_tau(z, scale=scale, angle=angle, interp=interp)

    if kind == "blc_grid":
        # token latents cannot safely change token count, so scale must be identity
        if not _is_identity_scale(scale):
            return None

        B, L, C = z.shape
        S = _square_size(L)
        if S is None:
            return None

        z_grid = z.transpose(1, 2).reshape(B, C, S, S).contiguous()
        z_grid = apply_tau(z_grid, scale=1.0, angle=angle, interp=interp)
        z_t = z_grid.flatten(2).transpose(1, 2).contiguous()
        return z_t

    return None


# -------------------------
# dataset
# -------------------------
class ImageNetFolderDataset(Dataset):
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
# core loss
# EQ term only, no GAN, no KL
# -------------------------
@torch.no_grad()
def compute_eq_lpips_no_gan_no_kl(
    vae,
    lpips_model,
    img01: torch.Tensor,
    perceptual_weight: float,
    logvar: float,
    device,
    *,
    p_alpha: float,
    anisotropic: bool,
    interp: str,
):
    # img01: [0,1] -> [-1,1]
    raw255 = img01 * 255.0
    x = preprocess_imgs_vae(raw255).to(device)

    # encode original x
    _, z = _vae_get_posterior_and_z(vae, x)

    latent_kind = infer_latent_tau_kind(z)

    # RAE / ScaleRAE has fixed latent_mean / latent_var spatial statistics.
    # Do NOT resize or rotate its latent; otherwise decode may crash or become invalid.
    if is_rae_like_vae(vae):
        scale, angle = 1.0, 0
        z_t = z
        x_t = x
    else:
        # B,C,H,W latent can support scale + rot90.
        # B,L,C token latent can only support rot90 if L=S*S.
        # Other 1D latent has no safe spatial tau.
        allow_scale = latent_kind == "bchw"

        scale, angle = sample_eq_transform(
            p_alpha=p_alpha,
            anisotropic=anisotropic,
            allow_scale=allow_scale,
        )

        z_t = apply_tau_to_latent(z, scale=scale, angle=angle, interp=interp)

        # If latent has no reliable spatial layout, fall back to identity.
        if z_t is None:
            scale, angle = 1.0, 0
            z_t = z
            x_t = x
        else:
            x_t = apply_tau(x, scale=scale, angle=angle, interp=interp)

    xrec_t = _vae_decode_from_z(vae, z_t)

    # safety: decoder output size may differ after latent scaling
    if xrec_t.shape[-2:] != x_t.shape[-2:]:
        xrec_t = F.interpolate(
            xrec_t,
            size=x_t.shape[-2:],
            mode=interp,
            align_corners=False,
        )

    # Lrec = L1 + w * LPIPS
    rec_map = (x_t.contiguous() - xrec_t.contiguous()).abs()

    if perceptual_weight > 0:
        p = lpips_model(x_t.contiguous(), xrec_t.contiguous())
        p_mean = p.mean()

        if p.dim() == 4:
            p_map = p
        elif p.dim() == 2:
            p_map = p.view(p.size(0), 1, 1, 1)
        elif p.dim() == 1:
            p_map = p.view(-1, 1, 1, 1)
        else:
            p_map = p.view(p.size(0), 1, 1, 1)

        rec_map = rec_map + perceptual_weight * p_map
    else:
        p_mean = torch.tensor(0.0, device=device)

    B = x_t.size(0)
    logvar_t = torch.tensor(logvar, device=device, dtype=x_t.dtype)

    nll_map = rec_map / torch.exp(logvar_t) + logvar_t
    nll_loss = torch.sum(nll_map) / B
    rec_loss_mean = rec_map.mean()
    total = nll_loss

    return {
        "total_loss": total.detach().float(),
        "nll_loss": nll_loss.detach().float(),
        "rec_loss_mean": rec_loss_mean.detach().float(),
        "p_loss_mean": p_mean.detach().float(),
        "angle": torch.tensor(float(angle), device=device),
        "scale_is_tuple": torch.tensor(float(isinstance(scale, tuple)), device=device),
    }

def is_rae_like_vae(vae) -> bool:
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

# -------------------------
# main eval loop
# -------------------------
def evaluate_vae(args):
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print("[Device]", device)

    set_seed(args.seed + accelerator.process_index)
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    ds = ImageNetFolderDataset(root=args.base_path, transform=transform)

    loader = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    loader = accelerator.prepare(loader)

    vae = load_vae(args, device, accelerator)
    vae.eval()

    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    sums = {
        "total_loss": 0.0,
        "nll_loss": 0.0,
        "rec_loss_mean": 0.0,
        "p_loss_mean": 0.0,
    }

    n_seen = 0
    t0 = time.time()

    pbar = tqdm(
        loader,
        desc="EvalEQ",
        disable=not accelerator.is_local_main_process,
    )

    for img01, _ in pbar:
        img01 = img01.to(device, non_blocking=True)

        m = compute_eq_lpips_no_gan_no_kl(
            vae=vae,
            lpips_model=lpips_model,
            img01=img01,
            perceptual_weight=args.perceptual_weight,
            logvar=args.logvar,
            device=device,
            p_alpha=args.p_alpha,
            anisotropic=args.anisotropic,
            interp=args.interp,
        )

        b = img01.size(0)
        n_seen += b

        for k in sums:
            sums[k] += float(m[k].detach().cpu()) * b

        if args.num_images > 0 and n_seen >= args.num_images:
            break

        pbar.set_postfix(
            {
                "eq_total": sums["total_loss"] / max(1, n_seen),
                "eq_rec": sums["rec_loss_mean"] / max(1, n_seen),
            }
        )

    # single-process default path
    for k in sums:
        sums[k] /= max(1, n_seen)

    if accelerator.is_main_process:
        print("\n================== FINAL EQ LOSSES (Eq.5 first term) ==================")
        print(f"vae_config        : {args.vae_config}")
        print(f"num_images        : {n_seen}")
        print(f"resolution        : {args.resolution}")
        print(f"perceptual_weight : {args.perceptual_weight}")
        print(f"logvar            : {args.logvar}")
        print(f"p_alpha(identity) : {args.p_alpha}")
        print(f"anisotropic       : {args.anisotropic}")
        print(f"interp            : {args.interp}")
        print("------------------------------------------")
        print(f"eq_total_loss     : {sums['total_loss']:.6f}")
        print(f"eq_nll_loss       : {sums['nll_loss']:.6f}")
        print(f"eq_rec_loss_mean  : {sums['rec_loss_mean']:.6f}")
        print(f"eq_p_loss_mean    : {sums['p_loss_mean']:.6f}")
        print(f"time (s)          : {time.time() - t0:.2f}")
        print("======================================================================\n")


def build_parser():
    p = argparse.ArgumentParser("EQ-VAE Eq.5 term eval, config-based VAE interface")

    p.add_argument(
        "--base_path",
        "--base-path",
        dest="base_path",
        type=str,
        required=True,
        help="Image root, ImageNet-style folder",
    )

    # loader
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument(
        "--num_images",
        "--num-images",
        dest="num_images",
        type=int,
        default=50000,
        help="<=0 means eval all",
    )

    # VAE config
    p.add_argument("--vae-config", type=str, required=True)

    # loss params
    p.add_argument("--perceptual_weight", "--perceptual-weight", dest="perceptual_weight", type=float, default=1.0)
    p.add_argument("--logvar", type=float, default=0.0)

    # EQ transform params
    p.add_argument(
        "--p_alpha",
        "--p-alpha",
        dest="p_alpha",
        type=float,
        default=0.0,
        help="probability to use identity tau=I",
    )
    p.add_argument("--anisotropic", action="store_true")
    p.add_argument("--interp", type=str, default="bilinear", choices=["bicubic", "bilinear"])

    # misc
    p.add_argument("--seed", type=int, default=0)

    return p


def main():
    args = build_parser().parse_args()
    evaluate_vae(args)


if __name__ == "__main__":
    main()