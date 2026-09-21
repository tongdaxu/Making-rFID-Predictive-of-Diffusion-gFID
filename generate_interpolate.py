#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from omegaconf import OmegaConf
from torchvision import transforms

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        global_idx = self.indices[local_idx]
        item = self.dataset[global_idx]
        if len(item) == 3:
            x, y, name = item
        elif len(item) == 2:
            x, y = item
            name = str(global_idx)
        else:
            raise RuntimeError(f"Unexpected dataset item length: {len(item)}")
        return global_idx, x, y, name


def rank_indices(n, rank, world_size):
    return list(range(rank, n, world_size))


def encode_vae(vae, x):
    out = vae.encode(x)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def decode_vae(vae, z, y=None):
    try:
        return vae.decode(z, y)
    except TypeError:
        return vae.decode(z)


def slerp(z0, z1, alpha, eps=1e-7):
    orig_shape = z0.shape
    bsz = z0.shape[0]

    z0f = z0.reshape(bsz, -1)
    z1f = z1.reshape(bsz, -1)

    z0n = F.normalize(z0f, dim=1)
    z1n = F.normalize(z1f, dim=1)

    dot = (z0n * z1n).sum(dim=1).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    alpha_t = torch.full_like(omega, float(alpha))

    coeff0 = torch.sin((1.0 - alpha_t) * omega) / so
    coeff1 = torch.sin(alpha_t * omega) / so

    out = coeff0[:, None] * z0f + coeff1[:, None] * z1f
    lin = (1.0 - float(alpha)) * z0f + float(alpha) * z1f

    bad = so.abs() < eps
    if bad.any():
        out[bad] = lin[bad]

    return out.reshape(orig_shape)


def interpolate_latents(z, z_nn, mode, alpha):
    if mode == "linear":
        return (1.0 - alpha) * z + alpha * z_nn
    if mode == "slerp":
        return slerp(z, z_nn, alpha)
    if mode == "mask":
        # Spatial/channel mask interpolation. Mostly for compatibility.
        mask = torch.rand_like(z[:, :1]) < alpha
        return torch.where(mask, z_nn, z)
    raise ValueError(f"Unknown interpolation mode: {mode}")


def images_to_uint8(x):
    # Same convention as REPA-E generate.py: VAE decode output is expected in [-1, 1].
    x = (x + 1.0) / 2.0
    x = torch.clamp(255.0 * x, 0, 255)
    x = x.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return x


def build_npz_from_images(image_dir: Path, npz_path: Path, num_images: int, image_size: int):
    arr = np.empty((num_images, image_size, image_size, 3), dtype=np.uint8)

    for i in tqdm(range(num_images), desc="Building interpolate .npz"):
        p = image_dir / f"{i:06d}.png"
        if not p.exists():
            raise FileNotFoundError(f"Missing interpolate image: {p}")
        img = Image.open(p).convert("RGB")
        if img.size != (image_size, image_size):
            raise RuntimeError(f"Image size mismatch: {p}, got {img.size}, expected {(image_size, image_size)}")
        arr[i] = np.asarray(img, dtype=np.uint8)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, arr_0=arr)
    print(f"[NPZ] saved {npz_path} shape={arr.shape} dtype={arr.dtype}", flush=True)


def all_gather_cat_first_dim(x, world_size):
    if world_size == 1:
        return x
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=0)


def assert_same_batch_size(bsz, device, world_size):
    if world_size == 1:
        return
    t = torch.tensor([bsz], dtype=torch.long, device=device)
    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    vals = [int(x.item()) for x in gathered]
    if len(set(vals)) != 1:
        raise RuntimeError(
            f"Different batch sizes across ranks: {vals}. "
            "Use a small_val divisible by world_size * bs."
        )


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed + rank)

    out_root = Path(args.sample_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    image_dir = out_root / f"{args.exp_name}_images"
    npz_path = out_root / f"{args.exp_name}.npz"

    if npz_path.exists() and npz_path.stat().st_size > 0 and not args.overwrite:
        if accelerator.is_main_process:
            print(f"[SKIP] npz already exists: {npz_path}", flush=True)
        return

    if accelerator.is_main_process:
        print("=" * 100, flush=True)
        print("[generate_interpolate.py]", flush=True)
        print(f"world_size     : {world_size}", flush=True)
        print(f"vae_config     : {args.vae_config}", flush=True)
        print(f"dataset_ref    : {args.dataset_ref}", flush=True)
        print(f"dataset_val    : {args.dataset}", flush=True)
        print(f"small_ref      : {args.small}", flush=True)
        print(f"small_val      : {args.small_val}", flush=True)
        print(f"bs             : {args.bs}", flush=True)
        print(f"top            : {args.top}", flush=True)
        print(f"intp           : {args.intp}", flush=True)
        print(f"alpha          : {args.alpha}", flush=True)
        print(f"nn_sample      : {args.nn_sample}", flush=True)
        print(f"pclass         : {args.pclass}", flush=True)
        print(f"image_dir      : {image_dir}", flush=True)
        print(f"npz_path       : {npz_path}", flush=True)
        print("=" * 100, flush=True)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    ref_set_full = ImageNetValDataset(args.dataset_ref, transform, args.small)
    val_set_full = ImageNetValDataset(args.dataset, transform, args.small_val)

    ref_indices = rank_indices(len(ref_set_full), rank, world_size)
    val_indices = rank_indices(len(val_set_full), rank, world_size)

    if len(val_indices) % args.bs != 0:
        raise RuntimeError(
            f"Rank {rank}: local val size {len(val_indices)} is not divisible by bs={args.bs}. "
            f"Use small_val divisible by world_size * bs. "
            f"small_val={len(val_set_full)}, world_size={world_size}, bs={args.bs}"
        )

    ref_set = IndexedSubset(ref_set_full, ref_indices)
    val_set = IndexedSubset(val_set_full, val_indices)

    ref_loader = DataLoader(
        ref_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Encode local reference/train latents.
    local_zs = []
    local_ys = []

    ref_iter = ref_loader
    if accelerator.is_main_process:
        ref_iter = tqdm(ref_iter, desc="Encoding local ref/train latents")

    for _, x, y, _ in ref_iter:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = encode_vae(vae, x).detach().to(torch.float32)

        local_zs.append(z)
        local_ys.append(y.detach())

    zs = torch.cat(local_zs, dim=0).contiguous()
    ys = torch.cat(local_ys, dim=0).contiguous()

    z_shape = zs.shape[1:]
    zs_flat = zs.reshape(zs.shape[0], -1).contiguous()
    zs_norm = (zs_flat * zs_flat).sum(dim=1, keepdim=True)

    if accelerator.is_main_process:
        image_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    image_dir.mkdir(parents=True, exist_ok=True)

    val_iter = val_loader
    if accelerator.is_main_process:
        val_iter = tqdm(val_iter, desc="Generating interpolate images")

    for idx, x, y, _ in val_iter:
        assert_same_batch_size(len(idx), device, world_size)

        idx = idx.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = encode_vae(vae, x).detach().to(torch.float32)
        bsz = z.shape[0]
        z_flat = z.reshape(bsz, -1).contiguous()

        # Squared L2 cost: [local_ref, batch]
        z_norm = (z_flat * z_flat).sum(dim=1, keepdim=True).T
        cost = zs_norm + z_norm - 2.0 * (zs_flat @ z_flat.T)
        cost = torch.clamp(cost, min=0.0)

        if args.pclass:
            cost = cost + 1e9 * (ys[:, None] != y[None, :]).to(cost.dtype)

        local_k = min(args.top, cost.shape[0])
        local_vals, local_idx = torch.topk(cost, k=local_k, dim=0, largest=False)

        local_cands = zs[local_idx.reshape(-1)]
        local_cands = local_cands.reshape(local_k, bsz, *z_shape).contiguous()

        all_vals = all_gather_cat_first_dim(local_vals.contiguous(), world_size)
        all_cands = all_gather_cat_first_dim(local_cands.contiguous(), world_size)

        global_k = min(args.top, all_vals.shape[0])
        top_vals, top_order = torch.topk(all_vals, k=global_k, dim=0, largest=False)

        batch_arange = torch.arange(bsz, device=device)
        top_cands = all_cands[top_order, batch_arange[None, :]]

        if args.nn_sample == "nearest":
            pick = torch.zeros((bsz,), dtype=torch.long, device=device)
        elif args.nn_sample == "uniform":
            pick = torch.randint(0, global_k, (bsz,), device=device)
        elif args.nn_sample == "categorical":
            logits = -top_vals.T
            logits = logits - logits.max(dim=1, keepdim=True).values
            pick = torch.distributions.Categorical(logits=logits).sample()
        else:
            raise ValueError(f"Unknown nn_sample: {args.nn_sample}")

        z_nn = top_cands[pick, batch_arange]
        z_intp = interpolate_latents(z, z_nn, args.intp, args.alpha)

        img = decode_vae(vae, z_intp, y)
        img_np = images_to_uint8(img)

        for one_img, one_idx in zip(img_np, idx.detach().cpu().tolist()):
            Image.fromarray(one_img).save(image_dir / f"{int(one_idx):06d}.png")

        del x, y, z, z_flat, cost, local_vals, local_idx, local_cands
        del all_vals, all_cands, top_vals, top_order, top_cands, z_nn, z_intp, img
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        expected = len(val_set_full)
        print(f"[CHECK] expected images: {expected}", flush=True)

        missing = []
        for i in range(expected):
            if not (image_dir / f"{i:06d}.png").exists():
                missing.append(i)
                if len(missing) >= 20:
                    break

        if missing:
            raise RuntimeError(f"Missing images, first examples: {missing}")

        if args.create_npz:
            build_npz_from_images(image_dir, npz_path, expected, args.image_size)

        if args.create_npz and not args.keep_samples:
            print(f"[CLEAN] removing image folder: {image_dir}", flush=True)
            shutil.rmtree(image_dir)

        print("=" * 100, flush=True)
        print("[DONE]", flush=True)
        print(f"npz_path={npz_path}", flush=True)
        print("=" * 100, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vae-config", type=str, required=True)

    parser.add_argument("--dataset", type=str, default="/video_ssd/lpm/ImageNet/val")
    parser.add_argument("--dataset-ref", type=str, default="/video_ssd/lpm/ImageNet/train")

    parser.add_argument("--small", type=int, default=200000)
    parser.add_argument("--small-val", type=int, default=50000)

    parser.add_argument("--sample-dir", type=str, default="/share/xutongda/REPA-E/exps_interpolate")
    parser.add_argument("--exp-name", type=str, required=True)

    parser.add_argument("--bs", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)

    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--intp", type=str, default="slerp", choices=["linear", "slerp", "mask"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--nn-sample", type=str, default="categorical", choices=["nearest", "uniform", "categorical"])

    parser.add_argument("--pclass", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--create-npz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)
