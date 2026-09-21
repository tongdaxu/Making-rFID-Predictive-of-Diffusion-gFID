"""
Evaluate LDS loss for config-based VAE.

LDS:
    near cosine similarity - far cosine similarity

Interface follows eval_vf.py style:

Example:
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port 29700 eval_lds.py \
  --base_path /video_ssd/kongzishang/imagenet_val \
  --vae-config ./configs/SDVAE.yaml \
  --bs 8 \
  --num_images 50000 \
  --out_dir /video_ssd/kongzishang/xutongda/git/IFID/losses/lds/sdvae
"""

import os
import json
import csv
import time
import math
import argparse
import random
import importlib
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed


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
    Convert latent to BCHW for LDS.

    Supported:
      - B,C,H,W: unchanged
      - B,L,C where L=S*S: -> B,C,S,S
      - B,C,L where L=S*S: -> B,C,S,S
      - B,L,C or B,C,L non-square: -> B,C,1,L
      - B,D global latent: -> B,D,1,1

    Note:
      LDS is most meaningful for true 2D spatial latents.
      For B,C,1,L fallback, LDS becomes a 1D sequence locality metric.
      For B,D,1,1, LDS is undefined because there are no pairs.
    """
    if x.dim() == 4:
        return x.contiguous()

    if x.dim() == 2:
        return x[:, :, None, None].contiguous()

    if x.dim() == 3:
        B, D1, D2 = x.shape

        d1_square = _is_square(D1)
        d2_square = _is_square(D2)

        if d1_square and ((not d2_square) or D1 >= D2):
            # B,L,C -> B,C,S,S
            S = int(math.isqrt(D1))
            return x.permute(0, 2, 1).reshape(B, D2, S, S).contiguous()

        if d2_square:
            # B,C,L -> B,C,S,S
            S = int(math.isqrt(D2))
            return x.reshape(B, D1, S, S).contiguous()

        if D1 <= D2:
            # likely B,C,L
            return x.reshape(B, D1, 1, D2).contiguous()

        # likely B,L,C
        return x.permute(0, 2, 1).reshape(B, D2, 1, D1).contiguous()

    raise RuntimeError(f"Unsupported latent shape={tuple(x.shape)}")


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

        return img, label, path


def make_eval_indices(n: int, num_images: int, seed: int):
    """
    Same spirit as eval_vf.py: deterministic subset if num_images > 0.
    If num_images <= 0, use all.
    """
    if num_images <= 0 or num_images >= n:
        return list(range(n))

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    return perm[:num_images]


# -------------------------
# VAE load / encode
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
    Copied interface style from eval_vf.py.

    Supports:
      - encode(x) -> tensor
      - encode(x).latent_dist
      - encode(x) posterior with .sample() / .mode()
      - encode(x) tuple/list
      - old forward fallback
    """
    if hasattr(vae, "encode") and callable(getattr(vae, "encode")):
        out = vae.encode(x_m11)

        if torch.is_tensor(out):
            t = out

            if assume_encode_returns_moments:
                if t.dim() == 4 and t.shape[1] % 2 == 0:
                    mean, logvar = torch.chunk(t, 2, dim=1)
                    if sample_posterior:
                        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
                    else:
                        z = mean
                    return t, z

            return None, t

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
# LDS
# -------------------------

@torch.no_grad()
def compute_lds_paper_with_details(
    x: torch.Tensor,
    far_dist: int = 6,
    dist_type: str = "manhattan",
    far_chunk: int = 256,
):
    """
    Paper-style LDS / iREPA-style LDS.

    Input:
        x: [B, C, H, W]

    Definition:
        local = mean cosine similarity of 4-neighbor spatial pairs
        far   = mean cosine similarity of pairs with Manhattan distance >= far_dist
        LDS   = local - far

    This is different from the earlier teacher-style:
        near = dist < H//2
        far  = dist >= H//2
    """
    B, C, H, W = x.shape
    T = H * W

    if T <= 1:
        nan = torch.full((B,), float("nan"), device=x.device, dtype=torch.float32)
        return nan, nan, nan

    # token-wise L2 normalize for cosine
    x = x.float()
    x_norm = F.normalize(x, dim=1, eps=1e-8)  # [B, C, H, W]

    # ------------------------------------------------------------
    # local similarity: 4-neighbor, use right/down undirected pairs
    # ------------------------------------------------------------
    local_sims = []

    if H > 1:
        sim_v = F.cosine_similarity(
            x_norm[:, :, :-1, :],
            x_norm[:, :, 1:, :],
            dim=1,
            eps=1e-8,
        )  # [B, H-1, W]
        local_sims.append(sim_v.reshape(B, -1))

    if W > 1:
        sim_h = F.cosine_similarity(
            x_norm[:, :, :, :-1],
            x_norm[:, :, :, 1:],
            dim=1,
            eps=1e-8,
        )  # [B, H, W-1]
        local_sims.append(sim_h.reshape(B, -1))

    if len(local_sims) == 0:
        nan = torch.full((B,), float("nan"), device=x.device, dtype=torch.float32)
        return nan, nan, nan

    local_all = torch.cat(local_sims, dim=1)
    local_mean = local_all.mean(dim=1)  # [B]

    # ------------------------------------------------------------
    # far similarity: all pairs with spatial distance >= far_dist
    # chunked to avoid big [B,T,T] memory
    # ------------------------------------------------------------
    feats = x_norm.flatten(2).transpose(1, 2).contiguous()  # [B, T, C]

    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing="ij",
    )
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # [T, 2]

    if dist_type == "manhattan":
        dist = (coords[:, None, :] - coords[None, :, :]).abs().sum(-1)  # [T,T]
    elif dist_type == "euclidean":
        dist = torch.sqrt(
            ((coords[:, None, :] - coords[None, :, :]).float() ** 2).sum(-1)
        )
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    far_mask_full = dist >= far_dist  # [T,T]

    if far_mask_full.sum() == 0:
        nan = torch.full((B,), float("nan"), device=x.device, dtype=torch.float32)
        return nan, nan, nan

    far_sum = torch.zeros(B, device=x.device, dtype=torch.float32)
    far_count = 0

    far_chunk = max(1, far_chunk)

    for s in range(0, T, far_chunk):
        e = min(s + far_chunk, T)

        # [B, chunk, T]
        sim_blk = feats[:, s:e, :] @ feats.transpose(1, 2)

        mask_blk = far_mask_full[s:e, :]  # [chunk, T]
        cnt = int(mask_blk.sum().item())

        if cnt > 0:
            far_sum += sim_blk[:, mask_blk].reshape(B, -1).sum(dim=1)
            far_count += cnt

        del sim_blk

    far_mean = far_sum / max(1, far_count)

    lds = local_mean - far_mean
    return lds, local_mean, far_mean

@torch.no_grad()
def latent_channel_sum_sumsq_count_bchw(z: torch.Tensor):
    """
    z: [B, C, H, W]
    Dataset-level per-channel statistics.
    """
    z = z.float()
    reduce_dims = (0, 2, 3)
    count = z.shape[0] * z.shape[2] * z.shape[3]

    s = z.sum(dim=reduce_dims)
    ss = (z ** 2).sum(dim=reduce_dims)

    return s, ss, count


@torch.no_grad()
def compute_channel_bn_stats(
    vae,
    loader,
    device,
    sample_posterior: bool,
    assume_encode_returns_moments: bool,
):
    """
    Compute dataset-level per-channel mean/std after converting latent to BCHW.

    This is our LDS-channelBN ablation:
        z = (z - mean_c) / std_c
    """
    stat_sum = None
    stat_sumsq = None
    stat_count = 0
    bchw_shape = None

    for batch in tqdm(loader, desc="Computing channel BN stats"):
        img01, label, path = batch

        img01 = img01.to(device, non_blocking=True)
        x_m11 = preprocess_imgs_vae(img01 * 255.0)

        with torch.no_grad():
            _, z_raw = get_posterior_and_z(
                vae=vae,
                x_m11=x_m11,
                sample_posterior=sample_posterior,
                assume_encode_returns_moments=assume_encode_returns_moments,
            )
            z = to_bchw(z_raw).float()

        if bchw_shape is None:
            bchw_shape = tuple(z.shape[1:])

        s, ss, cnt = latent_channel_sum_sumsq_count_bchw(z)

        if stat_sum is None:
            stat_sum = torch.zeros_like(s, dtype=torch.float64, device=device)
            stat_sumsq = torch.zeros_like(ss, dtype=torch.float64, device=device)

        stat_sum += s.to(dtype=torch.float64)
        stat_sumsq += ss.to(dtype=torch.float64)
        stat_count += cnt

        del img01, x_m11, z_raw, z
        torch.cuda.empty_cache()

    stat_count_t = torch.tensor(float(stat_count), device=device, dtype=torch.float64)

    mean = stat_sum / stat_count_t
    var = stat_sumsq / stat_count_t - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-6))

    print(f"[Channel BN] bchw_shape={bchw_shape}, count={stat_count}")

    return mean.float(), std.float(), bchw_shape


def apply_channel_bn_bchw(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """
    z: [B, C, H, W]
    mean/std: [C]
    """
    return (z - mean[None, :, None, None]) / std[None, :, None, None]

# -------------------------
# main eval
# -------------------------
def evaluate_lds(args):
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print("[Device]", device)

    set_seed(args.seed + accelerator.process_index)
    random.seed(args.seed + accelerator.process_index)
    np.random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)

    os.makedirs(args.out_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    ds = ImageNetFolderDataset(root=args.base_path, transform=transform)
    eval_idx = make_eval_indices(len(ds), args.num_images, args.seed)
    eval_ds = Subset(ds, eval_idx)

    if accelerator.is_main_process:
        print(f"[Eval subset] total={len(ds)} | eval={len(eval_ds)} | num_images={args.num_images}")

    loader = DataLoader(
        eval_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    loader = accelerator.prepare(loader)

    vae = load_vae(args, device, accelerator).eval()

    # -------------------------
    # Optional channel BN stats for LDS-channelBN
    # -------------------------
    channel_bn_mean = None
    channel_bn_std = None

    if args.lds_norm == "channel_bn":
        bn_base_path = args.bn_base_path if args.bn_base_path is not None else args.base_path

        if accelerator.is_main_process:
            print("[LDS Norm] channel_bn")
            print(f"[LDS Norm] BN base path : {bn_base_path}")
            print(f"[LDS Norm] BN num images: {args.bn_num_images}")

        bn_ds = ImageNetFolderDataset(root=bn_base_path, transform=transform)
        bn_idx = make_eval_indices(len(bn_ds), args.bn_num_images, args.seed)
        bn_ds = Subset(bn_ds, bn_idx)

        bn_loader = DataLoader(
            bn_ds,
            batch_size=args.bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        # Your shell uses --num_processes=1, so this is fine.
        channel_bn_mean, channel_bn_std, _ = compute_channel_bn_stats(
            vae=vae,
            loader=bn_loader,
            device=device,
            sample_posterior=args.sample_posterior,
            assume_encode_returns_moments=args.assume_encode_returns_moments,
        )

    else:
        if accelerator.is_main_process:
            print("[LDS Norm] none")

    sums = {
        "lds": 0.0,
        "near": 0.0,
        "far": 0.0,
    }
    sq_sums = {
        "lds": 0.0,
        "near": 0.0,
        "far": 0.0,
    }
    n_valid = 0
    n_total = 0

    all_rows = []
    all_lds_local = []

    raw_shape = None
    bchw_shape = None

    t0 = time.time()

    pbar = tqdm(
        loader,
        desc="Eval LDS",
        disable=not accelerator.is_local_main_process,
    )

    for batch in pbar:
        img01, label, path = batch
        img01 = img01.to(device, non_blocking=True)
        x_m11 = preprocess_imgs_vae(img01 * 255.0)

        with torch.no_grad():
            _, z_raw = get_posterior_and_z(
                vae=vae,
                x_m11=x_m11,
                sample_posterior=args.sample_posterior,
                assume_encode_returns_moments=args.assume_encode_returns_moments,
            )

            if raw_shape is None:
                raw_shape = tuple(z_raw.shape[1:])

            z = to_bchw(z_raw).float()

            if bchw_shape is None:
                bchw_shape = tuple(z.shape[1:])

            if args.lds_norm == "channel_bn":
                z = apply_channel_bn_bchw(
                    z,
                    channel_bn_mean.to(device),
                    channel_bn_std.to(device),
                )
            elif args.lds_norm == "none":
                pass
            else:
                raise ValueError(args.lds_norm)

            lds, near_mean, far_mean = compute_lds_paper_with_details(
                z,
                far_dist=args.far_dist,
                dist_type=args.dist_type,
                far_chunk=args.far_chunk,
            )

        valid_mask = torch.isfinite(lds)
        n_batch_total = int(lds.numel())
        n_batch_valid = int(valid_mask.sum().item())

        n_total += n_batch_total

        if n_batch_valid > 0:
            lds_v = lds[valid_mask]
            near_v = near_mean[valid_mask]
            far_v = far_mean[valid_mask]

            sums["lds"] += float(lds_v.sum().detach().cpu())
            sums["near"] += float(near_v.sum().detach().cpu())
            sums["far"] += float(far_v.sum().detach().cpu())

            sq_sums["lds"] += float((lds_v ** 2).sum().detach().cpu())
            sq_sums["near"] += float((near_v ** 2).sum().detach().cpu())
            sq_sums["far"] += float((far_v ** 2).sum().detach().cpu())

            n_valid += n_batch_valid

            all_lds_local.append(lds_v.detach().cpu().float().numpy())

        if args.save_per_sample and accelerator.is_main_process:
            lds_cpu = lds.detach().cpu().float().numpy()
            near_cpu = near_mean.detach().cpu().float().numpy()
            far_cpu = far_mean.detach().cpu().float().numpy()
            labels_cpu = label.detach().cpu().numpy().tolist()

            if isinstance(path, (list, tuple)):
                paths = list(path)
            else:
                paths = [str(p) for p in path]

            for i in range(len(lds_cpu)):
                all_rows.append({
                    "path": paths[i],
                    "label": int(labels_cpu[i]),
                    "lds": float(lds_cpu[i]),
                    "near_mean": float(near_cpu[i]),
                    "far_mean": float(far_cpu[i]),
                })

        if n_valid > 0:
            pbar.set_postfix(
                {
                    "lds": sums["lds"] / max(1, n_valid),
                    "near": sums["near"] / max(1, n_valid),
                    "far": sums["far"] / max(1, n_valid),
                    "n": n_valid,
                }
            )

        del img01, x_m11, z_raw, z, lds, near_mean, far_mean
        torch.cuda.empty_cache()

    pack = torch.tensor(
        [
            sums["lds"], sums["near"], sums["far"],
            sq_sums["lds"], sq_sums["near"], sq_sums["far"],
            float(n_valid), float(n_total)
        ],
        device=device,
        dtype=torch.float64,
    )

    pack = accelerator.reduce(pack, reduction="sum")
    (
        lds_sum, near_sum, far_sum,
        lds_sq_sum, near_sq_sum, far_sq_sum,
        n_valid_sum, n_total_sum
    ) = [x.item() for x in pack]

    n_valid_sum = int(n_valid_sum)
    n_total_sum = int(n_total_sum)

    def mean_std(total, sq_total, n):
        if n <= 0:
            return float("nan"), float("nan")
        mean = total / n
        var = max(0.0, sq_total / n - mean * mean)
        return mean, math.sqrt(var)

    lds_mean, lds_std = mean_std(lds_sum, lds_sq_sum, n_valid_sum)
    near_mean, near_std = mean_std(near_sum, near_sq_sum, n_valid_sum)
    far_mean, far_std = mean_std(far_sum, far_sq_sum, n_valid_sum)

    if accelerator.is_main_process:
        lds_all = None
        if len(all_lds_local) > 0:
            lds_all = np.concatenate(all_lds_local, axis=0)
        else:
            lds_all = np.array([], dtype=np.float32)

        metrics = {
            "vae_config": args.vae_config,
            "base_path": args.base_path,
            "resolution": args.resolution,
            "num_images_arg": args.num_images,
            "num_images_total": n_total_sum,
            "num_images_valid": n_valid_sum,
            "bs": args.bs,
            "sample_posterior": args.sample_posterior,
            "assume_encode_returns_moments": args.assume_encode_returns_moments,
            "raw_latent_shape": raw_shape,
            "bchw_latent_shape": bchw_shape,
            "dist_type": args.dist_type,
            "lds_mean": lds_mean,
            "lds_std": lds_std,
            "far_mean": far_mean,
            "far_std": far_std,
            "time_sec": time.time() - t0,
            "lds_variant": args.lds_variant,
            "lds_norm": args.lds_norm,
            "far_dist": args.far_dist,
            "far_chunk": args.far_chunk,
            "bn_base_path": args.bn_base_path,
            "bn_num_images": args.bn_num_images,
            "local_mean": near_mean,
            "local_std": near_std,
            "near_mean_legacy": near_mean,
            "near_std_legacy": near_std,
        }

        if lds_all.size > 0:
            metrics.update({
                "lds_p05": float(np.percentile(lds_all, 5)),
                "lds_p25": float(np.percentile(lds_all, 25)),
                "lds_p50": float(np.percentile(lds_all, 50)),
                "lds_p75": float(np.percentile(lds_all, 75)),
                "lds_p95": float(np.percentile(lds_all, 95)),
            })

        metrics_path = os.path.join(args.out_dir, "lds_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n================== FINAL LDS ==================")
        print(json.dumps(metrics, indent=2))
        print("================================================\n")

        if args.save_per_sample:
            csv_path = os.path.join(args.out_dir, "lds_per_sample.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["path", "label", "lds", "near_mean", "far_mean"],
                )
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row)
            print("Saved per-sample CSV:", csv_path)

        if lds_all.size > 0:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 5))
                plt.hist(lds_all, bins=args.hist_bins)
                plt.title("LDS distribution")
                plt.xlabel("LDS = near cosine similarity - far cosine similarity")
                plt.ylabel("number of samples")
                plt.tight_layout()
                fig_path = os.path.join(args.out_dir, "hist_lds.png")
                plt.savefig(fig_path, dpi=200)
                plt.close()
                print("Saved histogram:", fig_path)
            except Exception as e:
                print("[Warning] Failed to plot LDS histogram:", repr(e))


def build_parser():
    p = argparse.ArgumentParser("Eval LDS for config-based VAE.")

    p.add_argument("--base_path", "--base-path", dest="base_path", type=str, required=True)
    p.add_argument("--vae-config", type=str, required=True)
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", type=str, required=True)

    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument(
        "--num_images",
        "--num-images",
        dest="num_images",
        type=int,
        default=50000,
        help="Eval images; <=0 means all images.",
    )

    p.add_argument("--dist_type", "--dist-type", dest="dist_type", type=str, default="manhattan",
                   choices=["manhattan", "euclidean"])

    p.add_argument("--sample_posterior", "--sample-posterior", dest="sample_posterior", action="store_true")
    p.add_argument("--no_sample_posterior", "--no-sample-posterior", dest="sample_posterior", action="store_false")
    p.set_defaults(sample_posterior=False)

    p.add_argument(
        "--assume_encode_returns_moments",
        "--assume-encode-returns-moments",
        dest="assume_encode_returns_moments",
        action="store_true",
        default=False,
        help="Only enable for old VAEs whose encode(x) returns B,2C,H,W moments. Do not enable for Flux2VAE.",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hist_bins", "--hist-bins", dest="hist_bins", type=int, default=100)
    p.add_argument("--save_per_sample", "--save-per-sample", dest="save_per_sample", action="store_true")

    p.add_argument(
        "--lds_variant",
        "--lds-variant",
        dest="lds_variant",
        type=str,
        default="paper",
        choices=["paper"],
        help="LDS variant. paper = 4-neighbor local similarity - far_dist similarity.",
    )

    p.add_argument(
        "--lds_norm",
        "--lds-norm",
        dest="lds_norm",
        type=str,
        default="none",
        choices=["none", "channel_bn"],
        help="Normalization before computing paper-style LDS.",
    )

    p.add_argument(
        "--far_dist",
        "--far-dist",
        dest="far_dist",
        type=int,
        default=6,
        help="Paper-style LDS far distance threshold. Default follows iREPA-style LDS.",
    )

    p.add_argument(
        "--far_chunk",
        "--far-chunk",
        dest="far_chunk",
        type=int,
        default=256,
        help="Chunk size for far-pair cosine computation.",
    )

    p.add_argument(
        "--bn_base_path",
        "--bn-base-path",
        dest="bn_base_path",
        type=str,
        default=None,
        help="Dataset path used to compute channel BN stats. Default: same as --base_path.",
    )

    p.add_argument(
        "--bn_num_images",
        "--bn-num-images",
        dest="bn_num_images",
        type=int,
        default=50000,
        help="Number of images used to estimate channel BN stats. <=0 means all.",
    )

    return p


def main():
    args = build_parser().parse_args()
    evaluate_lds(args)


if __name__ == "__main__":
    main()