#!/usr/bin/env python3
"""
Patch-iFID sweep: exact same-position hard-NN retrieval with P in {1,3,5,7}
and original whole-latent interpolation.

For every query latent z, spatial/token position x, and patch size P:

    j*_P(x) = argmax_j cos(
        z[Omega_P(x)],
        z_j[Omega_P(x)]
    )

where j ranges over ALL reference images and Omega_P(x) is the centered
same-position P x P patch. At image/latent boundaries, out-of-range entries
are treated as zero for BOTH query and reference; for cosine/L2 this is
equivalent to comparing only the valid clipped neighborhood.

Only the CENTER latent vector of the winning reference patch is used:

    z_NN,P(x) = z_{j*_P(x)}(x)

All selected center vectors are assembled into one composite NN latent.
Then the ORIGINAL iFID interpolation geometry is preserved:

    z_hat_P = whole_latent_SLERP(z, z_NN,P, alpha)

with alpha=0.5 by default.

Important:
- exact ALL-reference same-position search; NO global top-K prefilter;
- hard top-1 NN;
- same reference spatial/token position only; NO translation search;
- P is the only retrieval-context variable in the sweep;
- P=1 is exactly the earlier local P=1 hard cosine-NN definition;
- multiple P values share the expensive query-reference channel dot products,
  so P=1/3/5/7 are evaluated in one reference scan.

For native [B,N,C] token latents, the supplied token-grid determines P>1
neighborhoods. P=1 is grid-factorization invariant; P>1 assumes the token
ordering has the corresponding 2D spatial meaning.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.fid.fid import InceptionV3, calculate_frechet_distance
from ifid.vae.utils import instantiate_from_config

# Reuse the already-tested 25-VAE encode/decode/cache/layout helpers.
from eval_local_score_rfid_25_fast import (
    encode_vae,
    decode_vae,
    native_to_spatial,
    spatial_to_native,
    infer_token_grid,
    build_or_load_reference_cache,
    inception_features,
    dtype_from_name,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--vae-config", type=str, required=True)
    p.add_argument("--exp-name", type=str, required=True)

    p.add_argument("--dataset", type=str, default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", type=str, default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--ref-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=50000)
    p.add_argument("--resolution", type=int, default=256)

    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--ref-bs", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument(
        "--patch-sizes",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7],
        help="Odd same-position patch sizes. Default: 1 3 5 7.",
    )

    p.add_argument(
        "--intp",
        choices=["slerp", "linear"],
        default="slerp",
    )
    p.add_argument("--alpha", type=float, default=0.5)

    p.add_argument(
        "--token-grid",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="For native [B,N,C] latents when N is represented as a 2D token grid.",
    )

    p.add_argument(
        "--ref-chunk-size",
        type=int,
        default=1024,
        help=(
            "Reference images per exact patch-NN chunk. This controls temporary "
            "[B,R,H,W] similarity memory; it does NOT truncate the reference bank."
        ),
    )
    p.add_argument(
        "--metric-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Formal run: fp32.",
    )
    p.add_argument(
        "--gpu-bank-dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Storage dtype for the GPU-resident raw reference bank.",
    )

    p.add_argument(
        "--latent-cache-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/cache",
    )
    p.add_argument("--cache-group-batches", type=int, default=20)
    p.add_argument(
        "--cache-dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
    )
    p.add_argument(
        "--reuse-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/patch_ifid_sweep_25",
    )
    p.add_argument("--save-preview", type=int, default=0)
    p.add_argument("--keep-features", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


def validate_patch_sizes(patch_sizes: Iterable[int]) -> Tuple[int, ...]:
    vals = tuple(int(x) for x in patch_sizes)
    if not vals:
        raise ValueError("Need at least one --patch-sizes value.")
    if len(set(vals)) != len(vals):
        raise ValueError(f"Duplicate patch sizes: {vals}")
    for x in vals:
        if x <= 0 or x % 2 != 1:
            raise ValueError(
                f"Patch sizes must be positive odd integers; got P={x}"
            )
    return vals


def box_sum_2d(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    Centered P x P sum with zero padding.

    Zero-padding BOTH query and reference patch statistics is equivalent to
    cosine over the clipped valid neighborhood because padded coordinates are 0.
    """
    if p == 1:
        return x
    shape = x.shape
    if x.ndim != 4:
        raise ValueError(f"box_sum_2d expects NCHW, got {shape}")
    return F.avg_pool2d(
        x,
        kernel_size=p,
        stride=1,
        padding=p // 2,
        count_include_pad=True,
    ) * float(p * p)


@torch.no_grad()
def load_raw_reference_bank_gpu(
    cache_files,
    total: int,
    latent_shape: Sequence[int],
    device,
    bank_dtype,
):
    bank = torch.empty(
        (total, *latent_shape),
        device=device,
        dtype=bank_dtype,
    )

    seen = 0
    for meta in tqdm(cache_files, desc="load raw ref bank to GPU"):
        pack = torch.load(meta["path"], map_location="cpu")
        z = pack["z"]
        start = int(pack["start"])
        end = int(pack["end"])

        if start != seen:
            raise RuntimeError(
                f"Non-contiguous cache: start={start}, expected={seen}"
            )
        if end - start != z.shape[0]:
            raise RuntimeError(
                f"Bad cache metadata in {meta['path']}: "
                f"{end-start} vs tensor {z.shape[0]}"
            )
        if tuple(z.shape[1:]) != tuple(latent_shape):
            raise RuntimeError(
                f"Reference shape mismatch: {tuple(z.shape[1:])} "
                f"vs {tuple(latent_shape)}"
            )

        bank[start:end].copy_(
            z.to(device=device, dtype=bank_dtype, non_blocking=True)
        )
        seen = end
        del pack, z

    if seen != total:
        raise RuntimeError(f"Loaded {seen} refs, expected {total}")

    return bank


def whole_latent_slerp(
    z1: torch.Tensor,
    z2: torch.Tensor,
    t: float,
    eps: float = 1e-7,
):
    """
    Original iFID interpolation geometry:
    flatten the ENTIRE latent per image, compute one angle, interpolate once.
    """
    B = z1.shape[0]
    z1f = z1.float().reshape(B, -1)
    z2f = z2.float().reshape(B, -1)

    z1n = z1f / (z1f.norm(dim=1, keepdim=True) + eps)
    z2n = z2f / (z2f.norm(dim=1, keepdim=True) + eps)

    dot = (z1n * z2n).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w2 = torch.sin(t * theta) / (sin_theta + eps)

    zt = w1[:, None] * z1f + w2[:, None] * z2f

    small = sin_theta < 1e-6
    if small.any():
        zt[small] = (1.0 - t) * z1f[small] + t * z2f[small]

    return zt.reshape_as(z1)


def whole_latent_interpolate(
    query_native: torch.Tensor,
    nn_native: torch.Tensor,
    intp: str,
    alpha: float,
):
    if intp == "linear":
        return (
            (1.0 - alpha) * query_native.float()
            + alpha * nn_native.float()
        )
    if intp == "slerp":
        return whole_latent_slerp(query_native, nn_native, alpha)
    raise ValueError(intp)


@torch.no_grad()
def exact_same_position_patch_nn_multi_p(
    query_native: torch.Tensor,
    ref_bank_native: torch.Tensor,
    native_shape,
    token_grid,
    patch_sizes: Sequence[int],
    ref_chunk_size: int,
    metric_dtype,
):
    """
    Exact all-reference hard cosine NN for multiple same-position patch sizes.

    Efficient identity:
      patch dot(q,r,x) =
        box_sum_x( sum_c q_c(x) * r_c(x) )

    Therefore all P values share the expensive local channel dot product
    [B,R,H,W]; each P only adds box sums and a top-1 reduction.

    Returns:
      nn_native_by_p: dict[P] -> composite raw NN latent, native layout
      stats_by_p:
        - mean selected patch cosine over all query positions
        - mean whole-latent cosine(query, composite NN)
        - mean relative-L2(query, composite NN)
    """
    q_sp = native_to_spatial(query_native.float(), token_grid=token_grid)
    B, C, H, W = q_sp.shape
    L = H * W
    N = int(ref_bank_native.shape[0])

    q_metric = q_sp.to(metric_dtype)

    # Local center-vector norms and patch norms.
    q_sq_map = (q_sp.float() * q_sp.float()).sum(dim=1, keepdim=True)
    q_patch_sq = {
        p: box_sum_2d(q_sq_map, p)[:, 0]  # [B,H,W]
        for p in patch_sizes
    }

    best_sim: Dict[int, torch.Tensor] = {
        p: torch.full(
            (B, H, W),
            -float("inf"),
            device=q_sp.device,
            dtype=torch.float32,
        )
        for p in patch_sizes
    }

    best_vec: Dict[int, torch.Tensor] = {
        p: torch.empty(
            (B, C, H, W),
            device=q_sp.device,
            dtype=torch.float32,
        )
        for p in patch_sizes
    }

    # Position index used to gather the winning RAW center vector.
    pos_flat = (
        torch.arange(L, device=q_sp.device)
        .view(1, L)
        .expand(B, L)
    )

    rstep = max(1, int(ref_chunk_size))

    for rs in range(0, N, rstep):
        re = min(N, rs + rstep)
        R = re - rs

        r_native = ref_bank_native[rs:re]
        r_sp = native_to_spatial(r_native, token_grid=token_grid)

        if tuple(r_sp.shape[1:]) != (C, H, W):
            raise RuntimeError(
                f"Query/ref spatial mismatch: query={(C,H,W)}, "
                f"ref={tuple(r_sp.shape[1:])}"
            )

        r_metric = r_sp.to(metric_dtype)

        # [L,B,C] x [L,C,R] -> [L,B,R] -> [B,R,H,W]
        q_lbc = (
            q_metric.permute(2, 3, 0, 1)
            .reshape(L, B, C)
            .contiguous()
        )
        r_lrc = (
            r_metric.permute(2, 3, 0, 1)
            .reshape(L, R, C)
            .contiguous()
        )

        dot_lbr = torch.bmm(
            q_lbc,
            r_lrc.transpose(1, 2),
        ).float()

        dot_map = (
            dot_lbr.permute(1, 2, 0)
            .reshape(B, R, H, W)
            .contiguous()
        )

        r_sq_map = (
            (r_sp.float() * r_sp.float())
            .sum(dim=1, keepdim=True)
        )  # [R,1,H,W]

        # Raw reference centers as [R,L,C] for winner gather.
        r_rlc_raw = (
            r_sp.float()
            .permute(0, 2, 3, 1)
            .reshape(R, L, C)
            .contiguous()
        )

        for p in patch_sizes:
            # Numerator: [B,R,H,W].
            dot_patch = box_sum_2d(
                dot_map.reshape(B * R, 1, H, W),
                p,
            ).reshape(B, R, H, W)

            # Reference patch squared norms: [R,H,W].
            r_patch_sq = box_sum_2d(r_sq_map, p)[:, 0]

            denom = torch.sqrt(
                q_patch_sq[p][:, None]
                * r_patch_sq[None]
            ).clamp_min_(1e-12)

            sim = dot_patch / denom

            chunk_sim, chunk_idx = sim.max(dim=1)  # [B,H,W]
            improve = chunk_sim > best_sim[p]

            # Gather the RAW center vector from the winning reference image
            # independently at every same position.
            idx_flat = chunk_idx.reshape(B, L)
            selected_blc = r_rlc_raw[
                idx_flat,
                pos_flat,
            ]  # [B,L,C]

            selected = (
                selected_blc.reshape(B, H, W, C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            best_sim[p] = torch.where(
                improve,
                chunk_sim,
                best_sim[p],
            )
            best_vec[p] = torch.where(
                improve[:, None],
                selected,
                best_vec[p],
            )

            del (
                dot_patch,
                r_patch_sq,
                denom,
                sim,
                chunk_sim,
                chunk_idx,
                improve,
                idx_flat,
                selected_blc,
                selected,
            )

        del (
            r_native,
            r_sp,
            r_metric,
            q_lbc,
            r_lrc,
            dot_lbr,
            dot_map,
            r_sq_map,
            r_rlc_raw,
        )

    nn_native_by_p = {}
    stats_by_p = {}

    q_flat = query_native.float().reshape(B, -1)
    q_norm = torch.linalg.vector_norm(q_flat, dim=1).clamp_min(1e-12)

    for p in patch_sizes:
        nn_native = spatial_to_native(
            best_vec[p],
            native_shape=native_shape,
            token_grid=token_grid,
        ).float()

        n_flat = nn_native.reshape(B, -1)

        whole_cos = F.cosine_similarity(
            q_flat,
            n_flat,
            dim=1,
        )
        rel_l2 = (
            torch.linalg.vector_norm(q_flat - n_flat, dim=1)
            / q_norm
        )

        nn_native_by_p[p] = nn_native
        stats_by_p[p] = {
            "patch_cos_sum": float(best_sim[p].sum().item()),
            "patch_cos_count": int(best_sim[p].numel()),
            "whole_cos_sum": float(whole_cos.sum().item()),
            "whole_rel_l2_sum": float(rel_l2.sum().item()),
            "image_count": int(B),
        }

    return nn_native_by_p, stats_by_p


def fid_from_features(src: np.ndarray, pred: np.ndarray) -> float:
    mu_src = src.mean(axis=0)
    cov_src = np.cov(src, rowvar=False)
    mu_pred = pred.mean(axis=0)
    cov_pred = np.cov(pred, rowvar=False)
    return float(
        calculate_frechet_distance(
            mu_src, cov_src, mu_pred, cov_pred
        )
    )


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    patch_sizes = validate_patch_sizes(args.patch_sizes)

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")
    if args.ref_chunk_size <= 0:
        raise ValueError("--ref-chunk-size must be > 0")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    train_set = ImageNetValDataset(
        args.dataset_ref, transform, args.ref_size
    )
    val_set = ImageNetValDataset(
        args.dataset, transform, args.val_size
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.ref_bs,
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

    vae = instantiate_from_config(
        OmegaConf.load(args.vae_config)
    ).to(device).eval()

    for param in vae.parameters():
        param.requires_grad_(False)

    fake_x = torch.zeros(
        1, 3, args.resolution, args.resolution, device=device
    )
    fake_y = torch.zeros(
        1, dtype=torch.long, device=device
    )
    fake_z = encode_vae(vae, fake_x, fake_y)
    native_shape = tuple(fake_z.shape[1:])

    if len(native_shape) == 2:
        infer_token_grid(int(native_shape[0]), args.token_grid)
    elif args.token_grid is not None:
        print(
            "[WARN] --token-grid ignored because VAE already returns BCHW."
        )

    cache_files, cache_shape, _mean, _std = (
        build_or_load_reference_cache(
            args, vae, train_loader, device
        )
    )

    if tuple(cache_shape) != native_shape:
        raise RuntimeError(
            f"Cache/probe latent mismatch: "
            f"{cache_shape} vs {native_shape}"
        )

    bank_dtype = dtype_from_name(args.gpu_bank_dtype)
    metric_dtype = dtype_from_name(args.metric_dtype)

    ref_bank = load_raw_reference_bank_gpu(
        cache_files=cache_files,
        total=args.ref_size,
        latent_shape=native_shape,
        device=device,
        bank_dtype=bank_dtype,
    )

    # Determine spatial size once so impossible P values fail early.
    fake_sp = native_to_spatial(
        fake_z.float(), token_grid=args.token_grid
    )
    _, C, H, W = fake_sp.shape

    for p in patch_sizes:
        if p > max(H, W) * 2 - 1:
            raise ValueError(
                f"P={p} is unreasonable for latent grid {H}x{W}."
            )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    print("=" * 88)
    print("Patch-iFID exact same-position hard-NN sweep")
    print("vae_config        :", args.vae_config)
    print("exp_name          :", args.exp_name)
    print("native_shape      :", native_shape)
    print("spatial_shape     :", (C, H, W))
    print("token_grid        :", args.token_grid)
    print("ref / val         :", args.ref_size, "/", args.val_size)
    print("patch_sizes       :", patch_sizes)
    print("NN metric         : patch cosine")
    print("NN scope          : ALL refs, SAME position")
    print("global prefilter  : NONE")
    print("translation search: NONE")
    print(
        "boundary          : zero-padded both sides "
        "(equiv. clipped valid patch)"
    )
    print(
        "interpolation     : WHOLE-LATENT",
        args.intp,
        "alpha=",
        args.alpha,
    )
    print("GPU bank dtype    :", ref_bank.dtype)
    print("metric dtype      :", metric_dtype)
    print("ref chunk         :", args.ref_chunk_size)
    print("=" * 88)

    src_feats = []
    pred_feats: Dict[int, list] = {
        p: [] for p in patch_sizes
    }

    stat_acc = {
        p: {
            "patch_cos_sum": 0.0,
            "patch_cos_count": 0,
            "whole_cos_sum": 0.0,
            "whole_rel_l2_sum": 0.0,
            "image_count": 0,
        }
        for p in patch_sizes
    }

    preview_left = int(args.save_preview)
    preview_dir = out_dir / "preview"
    if preview_left > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for img, y, names in tqdm(
        val_loader,
        desc=f"patch-iFID {args.exp_name}",
    ):
        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = encode_vae(vae, img, y).float()

        nn_by_p, batch_stats = (
            exact_same_position_patch_nn_multi_p(
                query_native=z,
                ref_bank_native=ref_bank,
                native_shape=native_shape,
                token_grid=args.token_grid,
                patch_sizes=patch_sizes,
                ref_chunk_size=args.ref_chunk_size,
                metric_dtype=metric_dtype,
            )
        )

        img_src01 = (img.float() + 1.0) / 2.0
        src_feats.append(
            inception_features(
                inception, img_src01
            ).float().cpu().numpy()
        )

        preview_src_cpu = (
            img_src01.detach().cpu()
            if preview_left > 0
            else None
        )

        for p in patch_sizes:
            for k in stat_acc[p]:
                stat_acc[p][k] += batch_stats[p][k]

            z_intp = whole_latent_interpolate(
                query_native=z,
                nn_native=nn_by_p[p],
                intp=args.intp,
                alpha=args.alpha,
            )

            img_hat = decode_vae(vae, z_intp, y)
            img_hat01 = (img_hat.float() + 1.0) / 2.0

            pred_feats[p].append(
                inception_features(
                    inception, img_hat01
                ).float().cpu().numpy()
            )

            if preview_left > 0:
                from PIL import Image

                out_cpu = img_hat01.detach().cpu()
                nsave = min(
                    preview_left,
                    out_cpu.shape[0],
                )

                def save_tensor(t, path):
                    arr = (
                        torch.clamp(t, 0, 1)
                        .mul(255)
                        .round()
                        .to(torch.uint8)
                        .permute(1, 2, 0)
                        .numpy()
                    )
                    Image.fromarray(arr).save(path)

                for j in range(nsave):
                    stem = Path(str(names[j])).stem
                    if p == patch_sizes[0]:
                        save_tensor(
                            preview_src_cpu[j],
                            preview_dir / f"{stem}_src.png",
                        )
                    save_tensor(
                        out_cpu[j],
                        preview_dir
                        / f"{stem}_patch_ifid_p{p}.png",
                    )

            del z_intp, img_hat, img_hat01

        if preview_left > 0:
            preview_left -= min(
                preview_left, img.shape[0]
            )

        del (
            z,
            nn_by_p,
            batch_stats,
            img_src01,
            preview_src_cpu,
        )

    src = np.concatenate(src_feats, axis=0)

    if src.shape[0] != args.val_size:
        raise RuntimeError(
            f"Expected {args.val_size} source features, "
            f"got {src.shape[0]}"
        )

    metrics = {}
    diagnostics = {}

    for p in patch_sizes:
        pred = np.concatenate(pred_feats[p], axis=0)
        if pred.shape[0] != args.val_size:
            raise RuntimeError(
                f"P={p}: expected {args.val_size} pred features, "
                f"got {pred.shape[0]}"
            )

        print(f"[FID] patch-iFID P={p} ...", flush=True)
        score = fid_from_features(src, pred)
        metrics[f"patch_ifid_p{p}"] = score

        acc = stat_acc[p]
        diagnostics[f"p{p}"] = {
            "mean_selected_patch_cosine": (
                acc["patch_cos_sum"]
                / max(acc["patch_cos_count"], 1)
            ),
            "mean_composite_whole_latent_cosine": (
                acc["whole_cos_sum"]
                / max(acc["image_count"], 1)
            ),
            "mean_composite_relative_l2": (
                acc["whole_rel_l2_sum"]
                / max(acc["image_count"], 1)
            ),
        }

        if args.keep_features:
            np.save(
                out_dir / f"features_patch_ifid_p{p}.npy",
                pred,
            )

    if args.keep_features:
        np.save(out_dir / "features_source.npy", src)

    result = {
        "vae_config": args.vae_config,
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "dataset_ref": args.dataset_ref,
        "ref_size": int(args.ref_size),
        "val_size": int(args.val_size),
        "native_latent_shape": list(native_shape),
        "spatial_latent_shape": [C, H, W],
        "token_grid": args.token_grid,
        "patch_sizes": list(patch_sizes),
        "nn_scope": (
            "all reference images at the same spatial/token position"
        ),
        "nn_topk": 1,
        "nn_metric": "cosine over centered same-position P x P patch",
        "boundary": (
            "zero padding for both query/reference; "
            "equivalent to clipped valid neighborhood"
        ),
        "global_image_prefilter": False,
        "translation_search": False,
        "center_vector_only": True,
        "interpolation": f"whole_latent_{args.intp}",
        "alpha": float(args.alpha),
        "raw_latent_matching": True,
        "gpu_bank_dtype": str(ref_bank.dtype),
        "metric_dtype": str(metric_dtype),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2)
    )

    with (out_dir / "metrics.csv").open(
        "w", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "patch_size",
                "patch_ifid",
                "mean_selected_patch_cosine",
                "mean_composite_whole_latent_cosine",
                "mean_composite_relative_l2",
            ]
        )
        for p in patch_sizes:
            d = diagnostics[f"p{p}"]
            w.writerow(
                [
                    p,
                    metrics[f"patch_ifid_p{p}"],
                    d["mean_selected_patch_cosine"],
                    d[
                        "mean_composite_whole_latent_cosine"
                    ],
                    d["mean_composite_relative_l2"],
                ]
            )

    print("\n" + "=" * 88)
    print("DONE:", args.exp_name)
    for p in patch_sizes:
        d = diagnostics[f"p{p}"]
        print(
            f"P={p:<2d} "
            f"iFID={metrics[f'patch_ifid_p{p}']:.6f}  "
            f"patch_cos={d['mean_selected_patch_cosine']:.6f}  "
            f"whole_cos="
            f"{d['mean_composite_whole_latent_cosine']:.6f}  "
            f"relL2={d['mean_composite_relative_l2']:.6f}"
        )
    print("metrics:", out_dir / "metrics.json")
    print("=" * 88)


if __name__ == "__main__":
    main(parse_args())
