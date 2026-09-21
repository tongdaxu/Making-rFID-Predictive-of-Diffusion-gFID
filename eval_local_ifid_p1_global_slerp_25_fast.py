#!/usr/bin/env python3
"""
Exact same-position local-NN iFID, P=1, with ORIGINAL whole-latent interpolation.

Ablation goal
-------------
Change ONLY nearest-neighbor retrieval locality relative to iFID:

Original-style geometry:
    global NN retrieval -> whole-latent SLERP

This experiment:
    same-position P=1 local NN retrieval -> assemble one composite NN latent
    -> whole-latent SLERP

For each query latent z and each spatial/token position x independently:

    j*(x) = argmin_j d(z(x), z_j(x))

where j ranges over ALL reference images and the reference position is the SAME x.
There is:
- no whole-image/global top-K prefilter;
- no use of other query positions when choosing j*(x);
- no translation search over other reference positions.

The selected local vectors are assembled into one composite nearest-neighbor latent:

    z_LNN(x) = z_{j*(x)}(x)

Then interpolation is performed ONCE over the entire flattened latent, exactly in
the geometry of the original iFID SLERP:

    z_hat = SLERP(z, z_LNN, alpha)

with alpha=0.5 by default.

For slerp mode, local NN retrieval uses cosine distance, matching the original
iFID cosine retrieval convention. For linear mode, retrieval uses squared L2.

This script reuses the raw reference-latent cache from the recent 25-VAE LSM run.
For native token latents [B,N,C], P=1 same-position means the same token index n.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

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

# Reuse the already-tested 25-VAE wrappers/cache/layout helpers.
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
        "--intp",
        choices=["slerp", "linear"],
        default="slerp",
        help=(
            "Interpolation mode. slerp uses same-position cosine NN; "
            "linear uses same-position squared-L2 NN."
        ),
    )
    p.add_argument("--alpha", type=float, default=0.5)

    p.add_argument(
        "--token-grid",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Only for native [B,N,C] latents with non-square N.",
    )

    p.add_argument(
        "--ref-chunk-size",
        type=int,
        default=4096,
        help="Reference images per exact local-NN search chunk.",
    )
    p.add_argument(
        "--position-chunk-size",
        type=int,
        default=64,
        help="Same-position locations/tokens processed together.",
    )
    p.add_argument(
        "--query-chunk-size",
        type=int,
        default=64,
        help="Query images processed together inside the local-NN kernel.",
    )
    p.add_argument(
        "--metric-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Arithmetic dtype for local NN. Formal run: fp32.",
    )
    p.add_argument(
        "--gpu-bank-dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help=(
            "Storage dtype for the raw reference bank resident on GPU. "
            "The LSM cache was fp16, so fp16 is the natural reuse setting."
        ),
    )

    p.add_argument(
        "--latent-cache-dir",
        type=str,
        default="/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/cache",
        help="Reuse the recent LSM raw-reference cache when ref-size matches.",
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
        default="/share/xutongda/REPA-E/exps_interpolate/local_ifid_p1_25",
    )
    p.add_argument("--save-preview", type=int, default=0)
    p.add_argument("--keep-features", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


@torch.no_grad()
def load_raw_reference_bank_gpu(
    cache_files,
    total: int,
    latent_shape: Sequence[int],
    device,
    bank_dtype,
):
    """
    Load cached RAW latents to one GPU-resident bank.

    No channel normalization is applied: this is iFID-style raw latent matching.
    """
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


def whole_latent_slerp(z1: torch.Tensor, z2: torch.Tensor, t: float, eps=1e-7):
    """
    Original iFID SLERP geometry:
    flatten the ENTIRE latent per image, compute one angle, interpolate once,
    then reshape back.
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
    """
    Keep interpolation geometry matched to original iFID.
    """
    if intp == "linear":
        return (1.0 - alpha) * query_native.float() + alpha * nn_native.float()
    if intp == "slerp":
        return whole_latent_slerp(query_native, nn_native, alpha)
    raise ValueError(intp)


@torch.no_grad()
def exact_same_position_nn_p1(
    query_native: torch.Tensor,
    ref_bank_native: torch.Tensor,
    native_shape,
    token_grid,
    intp: str,
    ref_chunk_size: int,
    position_chunk_size: int,
    query_chunk_size: int,
    metric_dtype,
):
    """
    Hard top-1 exact nearest neighbor over ALL reference images, independently
    for every same spatial/token position.

    Returns a RAW native-layout nearest-neighbor latent with the same shape as
    query_native.
    """
    q_sp = native_to_spatial(query_native.float(), token_grid=token_grid)
    B, C, H, W = q_sp.shape
    L = H * W
    N = int(ref_bank_native.shape[0])

    # [B,L,C], same position index L for query and references.
    q_blc = (
        q_sp.permute(0, 2, 3, 1)
        .reshape(B, L, C)
        .contiguous()
    )

    out_blc = torch.empty_like(q_blc, dtype=torch.float32)

    qstep = max(1, int(query_chunk_size))
    rstep = max(1, int(ref_chunk_size))
    pstep = max(1, int(position_chunk_size))

    for qb0 in range(0, B, qstep):
        qb1 = min(B, qb0 + qstep)
        q = q_blc[qb0:qb1].to(metric_dtype)
        bq = qb1 - qb0

        best_cost = torch.full(
            (bq, L),
            float("inf"),
            device=q.device,
            dtype=torch.float32,
        )
        best_vec = torch.empty(
            (bq, L, C),
            device=q.device,
            dtype=torch.float32,
        )

        # Pre-normalize query only for cosine-NN / slerp mode.
        if intp == "slerp":
            q_metric = F.normalize(q.float(), dim=-1).to(metric_dtype)
            q_sq = None
        else:
            q_metric = q
            q_sq = (q.float() * q.float()).sum(dim=-1)

        for rs in range(0, N, rstep):
            re = min(N, rs + rstep)

            r_native = ref_bank_native[rs:re]
            r_sp = native_to_spatial(r_native, token_grid=token_grid)
            if tuple(r_sp.shape[1:]) != (C, H, W):
                raise RuntimeError(
                    f"Query/ref spatial mismatch: query={(C,H,W)}, "
                    f"ref={tuple(r_sp.shape[1:])}"
                )

            # [R,L,C]
            r_rlc = (
                r_sp.permute(0, 2, 3, 1)
                .reshape(re - rs, L, C)
                .contiguous()
            )

            for ps in range(0, L, pstep):
                pe = min(L, ps + pstep)
                P = pe - ps

                # Batched over positions:
                # q_pbc: [P,Bq,C]
                # r_prc: [P,R,C]
                q_pbc = q_metric[:, ps:pe].permute(1, 0, 2).contiguous()

                if intp == "slerp":
                    r_prc = F.normalize(
                        r_rlc[:, ps:pe].float(), dim=-1
                    ).permute(1, 0, 2).contiguous().to(metric_dtype)

                    # [P,Bq,R], lower cost = higher cosine.
                    sim = torch.bmm(
                        q_pbc,
                        r_prc.transpose(1, 2),
                    )
                    cost = -sim.float()
                    del sim
                else:
                    r_prc = (
                        r_rlc[:, ps:pe]
                        .permute(1, 0, 2)
                        .contiguous()
                        .to(metric_dtype)
                    )
                    cross = torch.bmm(
                        q_pbc,
                        r_prc.transpose(1, 2),
                    ).float()
                    r_sq = (
                        r_prc.float() * r_prc.float()
                    ).sum(dim=-1)  # [P,R]
                    qq = q_sq[:, ps:pe].T.contiguous()  # [P,Bq]
                    cost = (
                        qq[:, :, None]
                        + r_sq[:, None, :]
                        - 2.0 * cross
                    )
                    del cross, r_sq, qq

                # [P,Bq]
                chunk_cost, chunk_idx = cost.min(dim=2)

                # Gather the RAW reference vector, not normalized metric vector.
                raw_prc = (
                    r_rlc[:, ps:pe]
                    .permute(1, 0, 2)
                    .contiguous()
                    .float()
                )
                pidx = torch.arange(P, device=q.device)[:, None].expand(P, bq)
                selected_pbc = raw_prc[pidx, chunk_idx]  # [P,Bq,C]

                old = best_cost[:, ps:pe].T  # [P,Bq]
                improve = chunk_cost < old

                new_cost = torch.where(improve, chunk_cost, old)
                old_vec = best_vec[:, ps:pe].permute(1, 0, 2)
                new_vec = torch.where(
                    improve[:, :, None],
                    selected_pbc,
                    old_vec,
                )

                best_cost[:, ps:pe] = new_cost.T
                best_vec[:, ps:pe] = new_vec.permute(1, 0, 2)

                del (
                    q_pbc,
                    r_prc,
                    cost,
                    chunk_cost,
                    chunk_idx,
                    raw_prc,
                    pidx,
                    selected_pbc,
                    old,
                    improve,
                    new_cost,
                    old_vec,
                    new_vec,
                )

            del r_native, r_sp, r_rlc

        out_blc[qb0:qb1] = best_vec
        del q, q_metric, q_sq, best_cost, best_vec

    out_sp = (
        out_blc.reshape(B, H, W, C)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    out_native = spatial_to_native(
        out_sp,
        native_shape=native_shape,
        token_grid=token_grid,
    )
    return out_native


def fid_from_features(src: np.ndarray, pred: np.ndarray) -> float:
    mu_src = src.mean(0)
    cov_src = np.cov(src, rowvar=False)
    mu_pred = pred.mean(0)
    cov_pred = np.cov(pred, rowvar=False)
    return float(
        calculate_frechet_distance(mu_src, cov_src, mu_pred, cov_pred)
    )


@torch.no_grad()
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")
    if args.ref_chunk_size <= 0:
        raise ValueError("--ref-chunk-size must be > 0")
    if args.position_chunk_size <= 0:
        raise ValueError("--position-chunk-size must be > 0")
    if args.query_chunk_size <= 0:
        raise ValueError("--query-chunk-size must be > 0")

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

    train_set = ImageNetValDataset(args.dataset_ref, transform, args.ref_size)
    val_set = ImageNetValDataset(args.dataset, transform, args.val_size)

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

    vae = instantiate_from_config(OmegaConf.load(args.vae_config)).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    fake_x = torch.zeros(
        1, 3, args.resolution, args.resolution, device=device
    )
    fake_y = torch.zeros(1, dtype=torch.long, device=device)
    fake_z = encode_vae(vae, fake_x, fake_y)
    native_shape = tuple(fake_z.shape[1:])

    if len(native_shape) == 2:
        infer_token_grid(int(native_shape[0]), args.token_grid)
    elif args.token_grid is not None:
        print("[WARN] --token-grid ignored because VAE already returns BCHW.")

    cache_files, cache_shape, _mean, _std = build_or_load_reference_cache(
        args, vae, train_loader, device
    )
    if tuple(cache_shape) != native_shape:
        raise RuntimeError(
            f"Cache/probe latent mismatch: {cache_shape} vs {native_shape}"
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

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    print("=" * 80)
    print("Same-position P=1 local-NN iFID + ORIGINAL whole-latent interpolation")
    print("vae_config      :", args.vae_config)
    print("exp_name        :", args.exp_name)
    print("native_shape    :", native_shape)
    print("token_grid      :", args.token_grid)
    print("ref / val       :", args.ref_size, "/", args.val_size)
    print("interpolation   : WHOLE-LATENT", args.intp, "alpha=", args.alpha)
    print(
        "NN metric       :",
        "cosine" if args.intp == "slerp" else "squared L2",
    )
    print("NN candidates   : ALL refs at SAME position")
    print("global prefilter: NONE")
    print("hard NN         : top-1")
    print("GPU bank dtype  :", ref_bank.dtype)
    print("metric dtype    :", metric_dtype)
    print("ref chunk       :", args.ref_chunk_size)
    print("position chunk  :", args.position_chunk_size)
    print("query chunk     :", args.query_chunk_size)
    print("=" * 80)

    src_feats = []
    pred_feats = []

    preview_left = int(args.save_preview)
    preview_dir = out_dir / "preview"
    if preview_left > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for img, y, names in tqdm(
        val_loader,
        desc=f"localNN-P1+globalSLERP {args.exp_name}",
    ):
        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = encode_vae(vae, img, y).float()

        z_nn = exact_same_position_nn_p1(
            query_native=z,
            ref_bank_native=ref_bank,
            native_shape=native_shape,
            token_grid=args.token_grid,
            intp=args.intp,
            ref_chunk_size=args.ref_chunk_size,
            position_chunk_size=args.position_chunk_size,
            query_chunk_size=args.query_chunk_size,
            metric_dtype=metric_dtype,
        )

        # IMPORTANT: local retrieval, but ORIGINAL whole-latent interpolation.
        z_intp = whole_latent_interpolate(
            query_native=z,
            nn_native=z_nn,
            intp=args.intp,
            alpha=args.alpha,
        )

        img_hat = decode_vae(vae, z_intp, y)

        img_src01 = (img.float() + 1.0) / 2.0
        img_hat01 = (img_hat.float() + 1.0) / 2.0

        src_feats.append(
            inception_features(inception, img_src01).float().cpu().numpy()
        )
        pred_feats.append(
            inception_features(inception, img_hat01).float().cpu().numpy()
        )

        if preview_left > 0:
            from PIL import Image

            src_cpu = img_src01.detach().cpu()
            out_cpu = img_hat01.detach().cpu()
            nsave = min(preview_left, src_cpu.shape[0])

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
                save_tensor(src_cpu[j], preview_dir / f"{stem}_src.png")
                save_tensor(
                    out_cpu[j],
                    preview_dir / f"{stem}_localNN_p1_globalSLERP.png",
                )
            preview_left -= nsave

        del z, z_nn, z_intp, img_hat, img_src01, img_hat01

    src = np.concatenate(src_feats, axis=0)
    pred = np.concatenate(pred_feats, axis=0)

    if src.shape[0] != args.val_size:
        raise RuntimeError(
            f"Expected {args.val_size} features, got {src.shape[0]}"
        )

    print("[FID] local-iFID P=1 ...", flush=True)
    score = fid_from_features(src, pred)

    metric_key = f"localnn_p1_global_{args.intp}_a{args.alpha:g}"
    metrics = {metric_key: score}

    if args.keep_features:
        np.save(out_dir / "features_source.npy", src)
        np.save(out_dir / "features_local_ifid_p1.npy", pred)

    result = {
        "vae_config": args.vae_config,
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "dataset_ref": args.dataset_ref,
        "ref_size": int(args.ref_size),
        "val_size": int(args.val_size),
        "native_latent_shape": list(native_shape),
        "token_grid": args.token_grid,
        "patch_size": 1,
        "same_position_only": True,
        "translation_search": False,
        "global_image_prefilter": False,
        "nn_scope": "all reference images at the same spatial/token position",
        "nn_topk": 1,
        "nn_metric": "cosine" if args.intp == "slerp" else "squared_l2",
        "interpolation": f"whole_latent_{args.intp}",
        "alpha": float(args.alpha),
        "raw_latent_matching": True,
        "local_nn_assembled_then_whole_latent_interpolation": True,
        "gpu_bank_dtype": str(ref_bank.dtype),
        "metric_dtype": str(metric_dtype),
        "metrics": metrics,
    }

    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, f"{v:.10f}"])

    print("\n" + "=" * 80)
    print("DONE:", args.exp_name)
    print(f"{metric_key:36s} = {score:.6f}")
    print("metrics:", out_dir / "metrics.json")
    print("=" * 80)


if __name__ == "__main__":
    main(parse_args())
