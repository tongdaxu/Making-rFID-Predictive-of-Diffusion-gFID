"""
Evaluate VF loss (MDMS + MCS) for a config-based VAE by calibrating W: z -> 1024.

Current interface:
  --vae-config ./configs/FLUX2VAE.yaml

Flux2VAE assumption:
  vae.encode(x) -> z
  z is 2D latent map: B,C,H,W

Example:
nohup env CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port 29505 eval_vf.py \
  --base_path /video_ssd/kongzishang/imagenet_val \
  --vae-config ./configs/FLUX2VAE.yaml \
  > /video_ssd/kongzishang/xutongda/git/IFID/losses/eval_vf_flux2vae.out 2>&1 &
"""

import os
import time
import argparse
import random
import math
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from einops import rearrange


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


def _is_square(n: int) -> bool:
    if n <= 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


def to_bchw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert VAE latent / feature to BCHW for Conv2d-based VF evaluation.

    Supported:
      - B,C,H,W: unchanged
      - B,L,C where L=S*S: -> B,C,S,S
      - B,C,L where L=S*S: -> B,C,S,S
      - B,L,C or B,C,L non-square: -> B,C,1,L using a size heuristic
      - B,D global latent: -> B,D,1,1

    Note: non-spatial/global fallback is only for compatibility; VF values are
    less comparable to true spatial latents.
    """
    if x.dim() == 4:
        return x.contiguous()

    if x.dim() == 2:
        return x[:, :, None, None].contiguous()

    if x.dim() == 3:
        B, D1, D2 = x.shape

        d1_square = _is_square(D1)
        d2_square = _is_square(D2)

        # Square-token cases. If both are square, treat the larger dim as token count.
        if d1_square and ((not d2_square) or D1 >= D2):
            # B,L,C -> B,C,S,S
            S = int(math.isqrt(D1))
            return x.permute(0, 2, 1).reshape(B, D2, S, S).contiguous()

        if d2_square:
            # B,C,L -> B,C,S,S
            S = int(math.isqrt(D2))
            return x.reshape(B, D1, S, S).contiguous()

        # Non-square sequence fallback. Use the smaller dim as channel dim.
        if D1 <= D2:
            # likely B,C,L
            return x.reshape(B, D1, 1, D2).contiguous()

        # likely B,L,C
        return x.permute(0, 2, 1).reshape(B, D2, 1, D1).contiguous()

    raise RuntimeError(f"Unsupported latent/feature shape={tuple(x.shape)}")


# -------------------------
# dataset
# -------------------------
class ImageNetFolderDataset(Dataset):
    """ImageNet-style folder: root/class_name/*.jpg"""

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
                    self.samples.append((os.path.join(cls_dir, fname), class_to_idx[cls_name]))

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


@torch.no_grad()
def get_posterior_and_z(
    vae,
    x_m11: torch.Tensor,
    sample_posterior: bool,
    assume_encode_returns_moments: bool,
):
    """
    Current preferred path:
        Flux2VAE.encode(x) -> z tensor, B,C,H,W

    Also supports:
        encode(x).latent_dist
        encode(x).sample()
        old forward outputs like (posterior, z)
    """

    # -------- 1) preferred current interface: encode() --------
    if hasattr(vae, "encode") and callable(getattr(vae, "encode")):
        out = vae.encode(x_m11)

        # Flux2VAE path: encode(x) returns z directly.
        if torch.is_tensor(out):
            t = out

            # Do NOT enable this for Flux2VAE.
            # Flux2VAE returns B,128,H,W, and 128 is even, so blindly splitting would be wrong.
            if assume_encode_returns_moments:
                if t.dim() == 4 and t.shape[1] % 2 == 0:
                    mean, logvar = torch.chunk(t, 2, dim=1)
                    if sample_posterior:
                        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
                    else:
                        z = mean
                    return t, z

            return None, t

        # diffusers style: encode(x).latent_dist
        if hasattr(out, "latent_dist"):
            posterior = out.latent_dist

            if sample_posterior and hasattr(posterior, "sample") and callable(getattr(posterior, "sample")):
                z = posterior.sample()
            elif hasattr(posterior, "mode") and callable(getattr(posterior, "mode")):
                z = posterior.mode()
            elif hasattr(posterior, "mean") and torch.is_tensor(posterior.mean):
                z = posterior.mean
            elif hasattr(posterior, "sample") and callable(getattr(posterior, "sample")):
                z = posterior.sample()
            else:
                raise RuntimeError("Cannot obtain z from out.latent_dist.")

            if not torch.is_tensor(z):
                raise RuntimeError(f"posterior returned non-tensor z: {type(z)}")

            return posterior, z

        # posterior directly
        if hasattr(out, "sample") and callable(getattr(out, "sample")):
            posterior = out
            if sample_posterior:
                z = posterior.sample()
            elif hasattr(posterior, "mode") and callable(getattr(posterior, "mode")):
                z = posterior.mode()
            elif hasattr(posterior, "mean") and torch.is_tensor(posterior.mean):
                z = posterior.mean
            else:
                z = posterior.sample()

            if not torch.is_tensor(z):
                raise RuntimeError(f"posterior.sample/mode returned non-tensor: {type(z)}")

            return posterior, z

        if hasattr(out, "mean") and torch.is_tensor(out.mean):
            return out, out.mean

        if isinstance(out, (tuple, list)):
            if len(out) >= 2 and torch.is_tensor(out[1]):
                return out[0], out[1]
            if len(out) >= 1 and torch.is_tensor(out[0]):
                return None, out[0]

    # -------- 2) fallback: old forward interface --------
    out = None
    try:
        out = vae(x_m11, return_recon=False)
    except Exception:
        try:
            out = vae(x_m11)
        except Exception:
            out = None

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        posterior = out[0]
        z = out[1]
        if torch.is_tensor(z):
            return posterior, z

    raise RuntimeError("Cannot extract z from VAE. Please check VAE encode/forward signature.")


# -------------------------
# foundation model
# -------------------------
try:
    import timm
except Exception:
    timm = None


def get_mae_encoder():
    model = timm.create_model(
        "hf-hub:timm/vit_large_patch16_224.mae",
        pretrained=True,
        dynamic_img_size=True,
    )
    model.requires_grad_(False)
    return model


def get_dinov2_encoder():
    model = timm.create_model(
        "hf-hub:timm/vit_large_patch14_dinov2.lvd142m",
        pretrained=True,
        dynamic_img_size=True,
    )
    model.requires_grad_(False)
    return model


def create_foundation_model(model_type):
    assert model_type in ["mae", "dinov2"], f"Unsupported foundation model type: {model_type}"

    if model_type == "mae":
        return get_mae_encoder(), 1024

    if model_type == "dinov2":
        return get_dinov2_encoder(), 1024


class aux_foundation_model(nn.Module):
    def __init__(self, model_type):
        super().__init__()

        if timm is None:
            raise RuntimeError("timm is not available. Please install timm.")

        self.model, feature_dim = create_foundation_model(model_type)
        self.type = model_type
        self.feature_dim = feature_dim

    def forward_mae(self, x):
        b, c, h, w = x.shape
        tokens = self.model.forward_features(x)[:, 1:]
        H, W = h // 16, w // 16
        return tokens.reshape(b, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def forward_dinov2(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        feats = self.model.forward_features(x)
        tokens = feats[:, 1:]

        B, N, C = tokens.shape
        gs = int(N ** 0.5)
        assert gs * gs == N, f"tokens N={N} is not a square number"

        return tokens.reshape(B, gs, gs, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        with torch.no_grad():
            if self.type == "mae":
                return self.forward_mae(x)
            if self.type == "dinov2":
                return self.forward_dinov2(x)
            raise NotImplementedError(self.type)


# -------------------------
# VF loss pieces
# -------------------------
def compute_mdms_chunked(z_proj, aux, distmat_margin: float, chunk: int):
    """
    vf_loss_1 / MDMS:
        relu(|S(z') - S(aux)| - margin).mean()
    """
    z_proj = to_bchw(z_proj)
    aux = to_bchw(aux)

    if z_proj.shape[-2:] != aux.shape[-2:]:
        aux = F.interpolate(aux, size=z_proj.shape[-2:], mode="bilinear", align_corners=False)

    z_flat = rearrange(z_proj, "b c h w -> b c (h w)")
    a_flat = rearrange(aux, "b c h w -> b c (h w)")

    z_norm = F.normalize(z_flat, dim=1)
    a_norm = F.normalize(a_flat, dim=1)

    B, C, N = z_norm.shape

    if chunk <= 0:
        z_cos = torch.einsum("bci,bcj->bij", z_norm, z_norm)
        a_cos = torch.einsum("bci,bcj->bij", a_norm, a_norm)
        diff = (z_cos - a_cos).abs()
        return F.relu(diff - distmat_margin).mean()

    zT = z_norm.transpose(1, 2).contiguous()
    aT = a_norm.transpose(1, 2).contiguous()

    total = 0.0
    count = 0

    for j0 in range(0, N, chunk):
        j1 = min(N, j0 + chunk)

        z_blk = z_norm[:, :, j0:j1]
        a_blk = a_norm[:, :, j0:j1]

        z_sim = torch.bmm(zT, z_blk)
        a_sim = torch.bmm(aT, a_blk)

        relu = F.relu((z_sim - a_sim).abs() - distmat_margin)

        total += relu.sum()
        count += relu.numel()

    return total / max(1, count)


def compute_mcs(z_proj, aux, cos_margin: float):
    """
    vf_loss_2 / MCS:
        relu(1 - cos_margin - cos(aux, z')).mean()
    """
    z_proj = to_bchw(z_proj)
    aux = to_bchw(aux)

    if z_proj.shape[-2:] != aux.shape[-2:]:
        aux = F.interpolate(aux, size=z_proj.shape[-2:], mode="bilinear", align_corners=False)

    if z_proj.shape[1] != aux.shape[1]:
        raise RuntimeError(f"MCS requires same C. z' C={z_proj.shape[1]} aux C={aux.shape[1]}")

    return F.relu(1.0 - cos_margin - F.cosine_similarity(aux, z_proj, dim=1)).mean()


# -------------------------
# split
# -------------------------
def make_split_indices(n: int, test_ratio: float, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)

    perm = torch.randperm(n, generator=g).tolist()

    n_test = int(round(n * test_ratio))
    n_test = max(1, min(n - 1, n_test))

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return train_idx, test_idx


# -------------------------
# main eval
# -------------------------
def evaluate_vf(args):
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
    train_idx, test_idx = make_split_indices(len(ds), args.test_ratio, args.split_seed)

    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, test_idx)

    if accelerator.is_main_process:
        print(
            f"[Split] total={len(ds)} | train={len(train_ds)} | "
            f"test={len(test_ds)} | test_ratio={args.test_ratio}"
        )

    # load models
    vae = load_vae(args, device, accelerator).eval()

    fm = aux_foundation_model(args.use_vf).to(device).eval()
    for p in fm.parameters():
        p.requires_grad_(False)

    # -------------------------
    # 1) infer Cz from one batch
    # -------------------------
    infer_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    infer_loader = accelerator.prepare(infer_loader)

    img01, _ = next(iter(infer_loader))
    img01 = img01.to(device, non_blocking=True)
    x_m11 = preprocess_imgs_vae(img01 * 255.0)

    with torch.no_grad():
        _, z_raw = get_posterior_and_z(
            vae=vae,
            x_m11=x_m11,
            sample_posterior=args.sample_posterior,
            assume_encode_returns_moments=args.assume_encode_returns_moments,
        )
        z = to_bchw(z_raw)

    Cz = z.shape[1]

    if accelerator.is_main_process:
        print(f"[Init] raw z shape={tuple(z_raw.shape)} | BCHW z shape={tuple(z.shape)} => Cz={Cz}")

    # paper-like projection: z(Cz) -> 1024
    proj = nn.Conv2d(Cz, 1024, kernel_size=1, bias=False).to(device)
    nn.init.normal_(proj.weight, mean=0.0, std=args.proj_init_std)

    # -------------------------
    # 2) calibration on TRAIN, rank0 only
    # -------------------------
    if accelerator.is_main_process:
        proj.train()

        opt = torch.optim.AdamW(
            proj.parameters(),
            lr=args.calib_lr,
            weight_decay=args.calib_wd,
        )

        calib_loader = DataLoader(
            train_ds,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        n_seen = 0
        steps = 0
        t_calib0 = time.time()

        pbar = tqdm(calib_loader, desc="Calibrating W on TRAIN", leave=False)

        for epoch in range(max(1, args.calib_epochs)):
            for img01, _ in pbar:
                img01 = img01.to(device, non_blocking=True)
                x_m11 = preprocess_imgs_vae(img01 * 255.0)

                with torch.no_grad():
                    _, z = get_posterior_and_z(
                        vae=vae,
                        x_m11=x_m11,
                        sample_posterior=args.sample_posterior,
                        assume_encode_returns_moments=args.assume_encode_returns_moments,
                    )
                    z = to_bchw(z).float()
                    aux = fm(x_m11).float()

                z_proj = proj(z)

                if args.calib_objective == "mcs":
                    loss = compute_mcs(z_proj, aux, args.cos_margin)
                else:
                    mdms = compute_mdms_chunked(
                        z_proj,
                        aux,
                        args.distmat_margin,
                        args.distmat_chunk,
                    )
                    mcs = compute_mcs(z_proj, aux, args.cos_margin)
                    loss = mdms * args.distmat_weight + mcs * args.cos_weight

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                b = img01.size(0)
                n_seen += b
                steps += 1

                pbar.set_postfix(
                    {
                        "loss": float(loss.detach().cpu()),
                        "seen": n_seen,
                        "steps": steps,
                    }
                )

                if args.calib_steps > 0 and steps >= args.calib_steps:
                    break
                if args.calib_images > 0 and n_seen >= args.calib_images:
                    break

            if args.calib_steps > 0 and steps >= args.calib_steps:
                break
            if args.calib_images > 0 and n_seen >= args.calib_images:
                break

        pbar.close()
        proj.eval()

        print(
            f"[Calib] done on TRAIN: seen={n_seen}, steps={steps}, "
            f"time={time.time() - t_calib0:.2f}s"
        )

    # broadcast proj weights to all ranks
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        for p in proj.parameters():
            torch.distributed.broadcast(p.data, src=0)
        torch.distributed.barrier()

    proj.eval()

    # -------------------------
    # 3) evaluation on TEST
    # -------------------------
    test_loader = DataLoader(
        test_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    test_loader = accelerator.prepare(test_loader)

    sums = {
        "vf": 0.0,
        "mdms": 0.0,
        "mcs": 0.0,
    }

    n_eval = 0
    t0 = time.time()

    pbar = tqdm(
        test_loader,
        desc=f"Eval VF on TEST ({args.use_vf})",
        disable=not accelerator.is_local_main_process,
    )

    for img01, _ in pbar:
        img01 = img01.to(device, non_blocking=True)
        x_m11 = preprocess_imgs_vae(img01 * 255.0)

        with torch.no_grad():
            _, z = get_posterior_and_z(
                vae=vae,
                x_m11=x_m11,
                sample_posterior=args.sample_posterior,
                assume_encode_returns_moments=args.assume_encode_returns_moments,
            )
            z = to_bchw(z).float()
            aux = fm(x_m11).float()

            z_proj = proj(z)

            mdms = compute_mdms_chunked(
                z_proj,
                aux,
                args.distmat_margin,
                args.distmat_chunk,
            )
            mcs = compute_mcs(z_proj, aux, args.cos_margin)
            vf = mdms * args.distmat_weight + mcs * args.cos_weight

        b = img01.size(0)
        n_eval += b

        sums["vf"] += float(vf.detach().cpu()) * b
        sums["mdms"] += float(mdms.detach().cpu()) * b
        sums["mcs"] += float(mcs.detach().cpu()) * b

        if args.num_images > 0 and n_eval >= args.num_images:
            break

        pbar.set_postfix(
            {
                "vf": sums["vf"] / max(1, n_eval),
                "mdms": sums["mdms"] / max(1, n_eval),
                "mcs": sums["mcs"] / max(1, n_eval),
                "n": n_eval,
            }
        )

    # reduce across processes
    pack = torch.tensor(
        [sums["vf"], sums["mdms"], sums["mcs"], float(n_eval)],
        device=device,
    )

    pack = accelerator.reduce(pack, reduction="sum")
    vf_sum, mdms_sum, mcs_sum, n_sum = [x.item() for x in pack]

    vf_avg = vf_sum / max(1.0, n_sum)
    mdms_avg = mdms_sum / max(1.0, n_sum)
    mcs_avg = mcs_sum / max(1.0, n_sum)

    if accelerator.is_main_process:
        print("\n================== FINAL VF (TEST) ==================")
        print(f"vae_config      : {args.vae_config}")
        print(f"use_vf          : {args.use_vf}")
        print(f"resolution      : {args.resolution}")
        print(f"sample_posterior: {args.sample_posterior}")
        print(f"split_seed      : {args.split_seed}")
        print(f"test_ratio      : {args.test_ratio}")
        print(f"test_images_eval: {int(n_sum)}")
        print("------------------------------------------")
        print("[Calibration on TRAIN]")
        print(f"calib_objective : {args.calib_objective}")
        print(f"calib_images    : {args.calib_images}")
        print(f"calib_steps     : {args.calib_steps}")
        print(f"calib_epochs    : {args.calib_epochs}")
        print(f"calib_lr        : {args.calib_lr}")
        print(f"calib_wd        : {args.calib_wd}")
        print(f"proj_init_std   : {args.proj_init_std}")
        print("------------------------------------------")
        print("[VF params]")
        print(f"distmat_margin  : {args.distmat_margin}")
        print(f"cos_margin      : {args.cos_margin}")
        print(f"distmat_weight  : {args.distmat_weight}")
        print(f"cos_weight      : {args.cos_weight}")
        print(f"distmat_chunk   : {args.distmat_chunk}")
        print("------------------------------------------")
        print(f"vf_total_mean   : {vf_avg:.6f}")
        print(f"mdms_mean       : {mdms_avg:.6f}")
        print(f"mcs_mean        : {mcs_avg:.6f}")
        print(f"time (s)        : {time.time() - t0:.2f}")
        print("=====================================================\n")


def build_parser():
    p = argparse.ArgumentParser(
        "Eval VF for config-based 2D VAE by calibrating W on train split."
    )

    # data
    p.add_argument("--base_path", "--base-path", dest="base_path", type=str, required=True)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument(
        "--num_images",
        "--num-images",
        dest="num_images",
        type=int,
        default=10000,
        help="TEST eval images; <=0 means all test split",
    )

    # split
    p.add_argument("--test_ratio", "--test-ratio", dest="test_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", "--split-seed", dest="split_seed", type=int, default=42)

    # VAE
    p.add_argument("--vae-config", type=str, required=True)

    # foundation model
    p.add_argument("--use_vf", "--use-vf", dest="use_vf", type=str, default="dinov2", choices=["mae", "dinov2"])

    # z sampling
    p.add_argument("--sample_posterior", "--sample-posterior", dest="sample_posterior", action="store_true")
    p.add_argument("--no_sample_posterior", "--no-sample-posterior", dest="sample_posterior", action="store_false")
    p.set_defaults(sample_posterior=True)

    # VF params
    p.add_argument("--cos_margin", "--cos-margin", dest="cos_margin", type=float, default=0.5)
    p.add_argument("--distmat_margin", "--distmat-margin", dest="distmat_margin", type=float, default=0.25)
    p.add_argument("--distmat_weight", "--distmat-weight", dest="distmat_weight", type=float, default=1.0)
    p.add_argument("--cos_weight", "--cos-weight", dest="cos_weight", type=float, default=1.0)
    p.add_argument("--distmat_chunk", "--distmat-chunk", dest="distmat_chunk", type=int, default=256)

    # calibration
    p.add_argument("--calib_images", "--calib-images", dest="calib_images", type=int, default=40000)
    p.add_argument("--calib_steps", "--calib-steps", dest="calib_steps", type=int, default=0)
    p.add_argument("--calib_epochs", "--calib-epochs", dest="calib_epochs", type=int, default=1)
    p.add_argument("--calib_lr", "--calib-lr", dest="calib_lr", type=float, default=1e-3)
    p.add_argument("--calib_wd", "--calib-wd", dest="calib_wd", type=float, default=0.0)
    p.add_argument("--proj_init_std", "--proj-init-std", dest="proj_init_std", type=float, default=0.02)
    p.add_argument(
        "--calib_objective",
        "--calib-objective",
        dest="calib_objective",
        type=str,
        default="vf",
        choices=["vf", "mcs"],
    )

    # misc
    p.add_argument("--seed", type=int, default=0)

    # For Flux2VAE this should stay False.
    p.add_argument(
        "--assume_encode_returns_moments",
        "--assume-encode-returns-moments",
        dest="assume_encode_returns_moments",
        action="store_true",
        default=False,
        help="Only enable for old VAEs whose encode(x) returns B,2C,H,W moments. Do not enable for Flux2VAE.",
    )

    return p


def main():
    args = build_parser().parse_args()
    evaluate_vf(args)


if __name__ == "__main__":
    main()