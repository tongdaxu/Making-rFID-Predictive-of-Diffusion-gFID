#!/usr/bin/env python3
"""
Hybrid DINO-filtered / VAE-retrieved iFID.

Six retrieval settings are evaluated together:

    K=1
    K=10
    K=50
    K=200
    K=1000
    GLOBAL = exact VAE-latent cosine hard top-1 over all 50k refs

For DINO K:

    C_i^(K) = DINO top-K reference image indices
    j*_K(i) = argmax_{j in C_i^(K)} cos(E(x_i), E(x_j))

For GLOBAL:

    j*_G(i) = argmax_j cos(E(x_i), E(x_j))

Then every setting uses exactly the same interpolation:

    z_hat = whole-latent SLERP(E(x_i), E(x_j*), alpha=0.5)

and the same decoder + Inception FID.

The DINO candidate lists are shared by all VAEs. DINO only FILTERS candidates;
the final nearest-neighbor choice for K>1 is made by the current VAE latent.

4090-oriented implementation
----------------------------
- Reuses the existing 50k VAE reference latent cache from the LSM experiments.
- Keeps the raw reference bank in CPU RAM (fp16/bf16 as cached).
- If that bank is small enough, also mirrors it on GPU automatically.
- Processes many validation queries together for retrieval, while VAE encode/
  decode still uses a small micro-batch (default 16).
- Exact GLOBAL hard top-1 scans the 50k reference bank once per QUERY BLOCK,
  not once per decoder micro-batch.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

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

from eval_local_score_rfid_25_fast import (
    encode_vae,
    decode_vae,
    build_or_load_reference_cache,
    inception_features,
)


K_LIST = (1, 10, 50, 200, 1000)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--vae-config", required=True)
    p.add_argument("--exp-name", required=True)

    p.add_argument("--dataset", default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--ref-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=50000)
    p.add_argument("--resolution", type=int, default=256)

    p.add_argument(
        "--dino-cache",
        default=(
            "/share/xutongda/REPA-E/exps_interpolate/"
            "dino_ifid_25/dinov2_vitb_global_ref50000_val50000"
        ),
    )
    p.add_argument("--dino-topk", type=int, default=1000)

    # VAE compute micro-batch. 4090 default requested by user.
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--ref-bs", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)

    # Number of queries sharing retrieval work.
    p.add_argument("--query-block-size", type=int, default=256)

    # Number of DINO candidates materialized at once.
    p.add_argument("--candidate-chunk", type=int, default=16)

    # GLOBAL 50k hard-NN reference scan chunk.
    p.add_argument("--global-ref-chunk", type=int, default=512)

    # Small banks are mirrored to GPU, large banks remain CPU-resident.
    p.add_argument("--gpu-bank-max-gb", type=float, default=8.0)

    p.add_argument(
        "--latent-cache-dir",
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

    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument(
        "--intp",
        choices=["slerp", "linear"],
        default="slerp",
    )

    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Formal retrieval default: FP32 matmul without TF32.",
    )
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--out-dir",
        default=(
            "/share/xutongda/REPA-E/exps_interpolate/"
            "hybrid_dino_vae_ifid_25/"
            "dinoK1_10_50_200_1000_globalhard_50k50k"
        ),
    )

    return p.parse_args()


def whole_latent_slerp(z1, z2, t, eps=1e-7):
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

    out = w1[:, None] * z1f + w2[:, None] * z2f

    small = sin_theta < 1e-6
    if small.any():
        out[small] = (
            (1.0 - t) * z1f[small] + t * z2f[small]
        )
    return out.reshape_as(z1)


def interpolate(zq, zr, mode, alpha):
    if mode == "slerp":
        return whole_latent_slerp(zq, zr, alpha)
    if mode == "linear":
        return (1.0 - alpha) * zq.float() + alpha * zr.float()
    raise ValueError(mode)


def fid_from_features(src, pred):
    mu_src = src.mean(axis=0)
    cov_src = np.cov(src, rowvar=False)
    mu_pred = pred.mean(axis=0)
    cov_pred = np.cov(pred, rowvar=False)
    return float(
        calculate_frechet_distance(
            mu_src, cov_src, mu_pred, cov_pred
        )
    )


def load_dino_topk(args):
    root = Path(args.dino_cache)
    idx_path = root / f"nn_top{args.dino_topk}_indices.npy"
    val_names_path = root / "val_names.txt"
    ref_names_path = root / "ref_names.txt"

    for p in (idx_path, val_names_path, ref_names_path):
        if not p.exists():
            raise FileNotFoundError(p)

    idx = np.load(idx_path, mmap_mode="r")
    if idx.ndim != 2 or idx.shape[1] != args.dino_topk:
        raise RuntimeError(
            f"DINO top-K shape mismatch: {idx.shape}; "
            f"expected second dimension {args.dino_topk}"
        )
    if idx.shape[0] < args.val_size:
        raise RuntimeError(
            f"DINO top-K cache has only {idx.shape[0]} queries, "
            f"but evaluator needs {args.val_size}"
        )

    for k in K_LIST:
        if k > args.dino_topk:
            raise RuntimeError(
                f"Need K={k}, but DINO cache only has top-{args.dino_topk}"
            )

    val_names = val_names_path.read_text(
        encoding="utf-8"
    ).splitlines()
    ref_names = ref_names_path.read_text(
        encoding="utf-8"
    ).splitlines()

    if len(val_names) < args.val_size:
        raise RuntimeError(
            f"DINO val_names has {len(val_names)} entries, "
            f"need at least {args.val_size}"
        )
    if len(ref_names) != args.ref_size:
        raise RuntimeError("DINO ref_names length mismatch")

    return idx, val_names, ref_names


@torch.inference_mode()
def load_reference_bank_cpu(
    cache_files,
    total,
    latent_shape,
):
    """
    Sequentially load the existing chunked VAE latent cache once into CPU RAM.
    This avoids reopening hundreds of cache chunks for every query.
    """
    bank = None
    seen = 0

    for meta in tqdm(cache_files, desc="load VAE ref bank to CPU"):
        pack = torch.load(meta["path"], map_location="cpu")
        z = pack["z"].contiguous()
        start = int(pack["start"])
        end = int(pack["end"])

        if start != seen:
            raise RuntimeError(
                f"Non-contiguous cache: start={start}, expected={seen}"
            )
        if end - start != z.shape[0]:
            raise RuntimeError("Bad reference cache metadata")
        if tuple(z.shape[1:]) != tuple(latent_shape):
            raise RuntimeError(
                f"Reference latent shape mismatch: "
                f"{tuple(z.shape[1:])} vs {tuple(latent_shape)}"
            )

        if bank is None:
            bank = torch.empty(
                (total, *latent_shape),
                dtype=z.dtype,
                device="cpu",
            )

        bank[start:end].copy_(z)
        seen = end

        del pack, z

    if bank is None or seen != total:
        raise RuntimeError(
            f"Loaded {seen} reference latents, expected {total}"
        )

    return bank


def bank_gb(bank):
    return (
        bank.numel() * bank.element_size()
        / (1024.0 ** 3)
    )


@torch.inference_mode()
def hybrid_best_indices(
    q_norm: torch.Tensor,
    dino_idx_block: np.ndarray,
    ref_cpu: torch.Tensor,
    ref_gpu: torch.Tensor,
    candidate_chunk: int,
):
    """
    Exact VAE cosine selection inside nested DINO top-K candidate sets.

    Returns dict K -> GLOBAL reference index [Q].
    """
    Q = q_norm.shape[0]
    Kmax = dino_idx_block.shape[1]
    sims = torch.empty(
        (Q, Kmax),
        device=q_norm.device,
        dtype=torch.float32,
    )

    for s in range(0, Kmax, candidate_chunk):
        e = min(Kmax, s + candidate_chunk)
        idx_np = np.asarray(
            dino_idx_block[:, s:e],
            dtype=np.int64,
        )
        idx = torch.from_numpy(idx_np).to(
            device=q_norm.device,
            dtype=torch.long,
        )

        flat_idx = idx.reshape(-1)

        if ref_gpu is not None:
            r = ref_gpu.index_select(0, flat_idx)
        else:
            r_cpu = ref_cpu.index_select(
                0, flat_idx.cpu()
            )
            r = r_cpu.to(
                q_norm.device,
                non_blocking=False,
            )
            del r_cpu

        r = r.float().reshape(
            Q, e - s, -1
        )
        r = F.normalize(r, dim=2)

        sims[:, s:e] = (
            r * q_norm[:, None, :]
        ).sum(dim=2)

        del idx, flat_idx, r

    out = {}
    dino_idx_t = torch.from_numpy(
        np.asarray(dino_idx_block, dtype=np.int64)
    ).to(q_norm.device)

    for k in K_LIST:
        pos = sims[:, :k].argmax(dim=1)
        chosen = dino_idx_t[
            torch.arange(Q, device=q_norm.device),
            pos,
        ]
        out[k] = chosen.cpu().long()

    return out


@torch.inference_mode()
def global_hard_best_indices(
    q_norm: torch.Tensor,
    ref_cpu: torch.Tensor,
    ref_gpu: torch.Tensor,
    ref_chunk: int,
):
    """
    Exact global hard top-1 cosine NN over ALL reference VAE latents.
    """
    Q = q_norm.shape[0]
    device = q_norm.device

    best_sim = torch.full(
        (Q,),
        -float("inf"),
        device=device,
        dtype=torch.float32,
    )
    best_idx = torch.zeros(
        (Q,),
        device=device,
        dtype=torch.long,
    )

    N = ref_cpu.shape[0]

    for s in range(0, N, ref_chunk):
        e = min(N, s + ref_chunk)

        if ref_gpu is not None:
            r = ref_gpu[s:e]
        else:
            r = ref_cpu[s:e].to(
                device,
                non_blocking=False,
            )

        r = F.normalize(
            r.float().reshape(e - s, -1),
            dim=1,
        )

        sim = q_norm @ r.T
        vals, pos = sim.max(dim=1)

        improve = vals > best_sim
        best_sim = torch.where(
            improve, vals, best_sim
        )
        best_idx = torch.where(
            improve, pos + s, best_idx
        )

        del r, sim, vals, pos, improve

    return best_idx.cpu().long(), best_sim.cpu()


@torch.inference_mode()
def decode_setting_features(
    vae,
    inception,
    zq_cpu,
    y_cpu,
    ref_indices_cpu,
    ref_cpu,
    device,
    bs,
    intp,
    alpha,
):
    """
    Decode one retrieval setting for one query block.
    """
    feats = []
    Q = zq_cpu.shape[0]

    selected_ref_cpu = ref_cpu.index_select(
        0, ref_indices_cpu.long()
    )

    for s in range(0, Q, bs):
        e = min(Q, s + bs)

        zq = zq_cpu[s:e].to(
            device=device,
            dtype=torch.float32,
        )
        zr = selected_ref_cpu[s:e].to(
            device=device,
            dtype=torch.float32,
        )
        y = y_cpu[s:e].to(
            device=device,
            non_blocking=True,
        )

        zhat = interpolate(
            zq, zr, intp, alpha
        )
        xhat = decode_vae(vae, zhat, y)
        xhat01 = (xhat.float() + 1.0) / 2.0

        feats.append(
            inception_features(
                inception, xhat01
            ).float().cpu().numpy()
        )

        del zq, zr, y, zhat, xhat, xhat01

    del selected_ref_cpu
    return feats


@torch.inference_mode()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.ref_size != 50000:
        print(
            "[WARN] This experiment was designed as a matched 50k-ref sweep."
        )
    if args.query_block_size < args.bs:
        raise ValueError(
            "--query-block-size must be >= --bs"
        )
    if args.candidate_chunk <= 0:
        raise ValueError("--candidate-chunk must be > 0")
    if args.global_ref_chunk <= 0:
        raise ValueError("--global-ref-chunk must be > 0")
    if not (0 <= args.alpha <= 1):
        raise ValueError("--alpha must be in [0,1]")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    try:
        torch.set_float32_matmul_precision(
            "high" if args.tf32 else "highest"
        )
    except Exception:
        pass

    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dino_topk, dino_val_names, dino_ref_names = (
        load_dino_topk(args)
    )

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    train_set = ImageNetValDataset(
        args.dataset_ref,
        transform,
        args.ref_size,
    )
    val_set = ImageNetValDataset(
        args.dataset,
        transform,
        args.val_size,
    )

    # Cheap ordering sanity against DINO cache.
    for i in (0, args.val_size // 2, args.val_size - 1):
        _x, _y, name = val_set[i]
        if str(name) != dino_val_names[i]:
            raise RuntimeError(
                f"Validation ordering mismatch at {i}: "
                f"{name!r} vs {dino_val_names[i]!r}"
            )

    for i in (0, args.ref_size // 2, args.ref_size - 1):
        _x, _y, name = train_set[i]
        if str(name) != dino_ref_names[i]:
            raise RuntimeError(
                f"Reference ordering mismatch at {i}: "
                f"{name!r} vs {dino_ref_names[i]!r}"
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
        persistent_workers=(args.num_workers > 0),
    )

    vae = instantiate_from_config(
        OmegaConf.load(args.vae_config)
    ).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Probe latent shape.
    fake_x = torch.zeros(
        1, 3, args.resolution, args.resolution,
        device=device,
    )
    fake_y = torch.zeros(
        1, dtype=torch.long, device=device
    )
    fake_z = encode_vae(
        vae, fake_x, fake_y
    ).float()
    latent_shape = tuple(fake_z.shape[1:])
    del fake_x, fake_y, fake_z

    # Reuse existing 50k reference cache.
    cache_files, cache_shape, _mean, _std = (
        build_or_load_reference_cache(
            args, vae, train_loader, device
        )
    )
    if tuple(cache_shape) != latent_shape:
        raise RuntimeError(
            f"cache/probe mismatch: "
            f"{cache_shape} vs {latent_shape}"
        )

    ref_cpu = load_reference_bank_cpu(
        cache_files,
        args.ref_size,
        latent_shape,
    )
    cpu_bank_gb = bank_gb(ref_cpu)

    ref_gpu = None
    if (
        args.gpu_bank_max_gb > 0
        and cpu_bank_gb <= args.gpu_bank_max_gb
    ):
        try:
            ref_gpu = ref_cpu.to(device)
            print(
                f"[bank] mirrored {cpu_bank_gb:.2f} GB reference bank to GPU"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ref_gpu = None
            print(
                "[bank] GPU mirror OOM; falling back to CPU-resident bank"
            )
    else:
        print(
            f"[bank] CPU-only reference bank: {cpu_bank_gb:.2f} GB"
        )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3(
        [block_idx]
    ).to(device).eval()

    metric_names = {
        1: "hybrid_ifid_dinoK1_vae_cos_top1",
        10: "hybrid_ifid_dinoK10_vae_cos_top1",
        50: "hybrid_ifid_dinoK50_vae_cos_top1",
        200: "hybrid_ifid_dinoK200_vae_cos_top1",
        1000: "hybrid_ifid_dinoK1000_vae_cos_top1",
        "global": "global_hard_ifid_vae_cos_top1_50k",
    }

    src_feats: List[np.ndarray] = []
    pred_feats: Dict[object, List[np.ndarray]] = {
        k: [] for k in list(K_LIST) + ["global"]
    }

    hit_counts = {k: 0 for k in K_LIST}
    total_queries = 0

    # Build retrieval blocks from ordinary small VAE batches.
    z_buf = []
    y_buf = []
    index_buf = []

    def process_block(zq_cpu, y_cpu, query_indices):
        nonlocal total_queries

        Q = zq_cpu.shape[0]

        zq_gpu = zq_cpu.to(
            device=device,
            dtype=torch.float32,
        )
        q_norm = F.normalize(
            zq_gpu.reshape(Q, -1),
            dim=1,
        )

        idx_np = np.asarray(
            dino_topk[
                query_indices[0]:
                query_indices[-1] + 1,
                :args.dino_topk,
            ],
            dtype=np.int64,
        )

        # Blocks are assembled in strict contiguous dataset order.
        if idx_np.shape[0] != Q:
            raise RuntimeError(
                "Internal DINO block slicing mismatch"
            )

        hybrid_idx = hybrid_best_indices(
            q_norm=q_norm,
            dino_idx_block=idx_np,
            ref_cpu=ref_cpu,
            ref_gpu=ref_gpu,
            candidate_chunk=args.candidate_chunk,
        )

        global_idx, _global_sim = (
            global_hard_best_indices(
                q_norm=q_norm,
                ref_cpu=ref_cpu,
                ref_gpu=ref_gpu,
                ref_chunk=args.global_ref_chunk,
            )
        )

        # How often would the unrestricted VAE hard NN survive DINO filtering?
        global_np = global_idx.numpy()
        for k in K_LIST:
            hit = (
                idx_np[:, :k]
                == global_np[:, None]
            ).any(axis=1)
            hit_counts[k] += int(hit.sum())

        total_queries += Q

        # Decode all six retrieval settings.
        for k in K_LIST:
            pred_feats[k].extend(
                decode_setting_features(
                    vae=vae,
                    inception=inception,
                    zq_cpu=zq_cpu,
                    y_cpu=y_cpu,
                    ref_indices_cpu=hybrid_idx[k],
                    ref_cpu=ref_cpu,
                    device=device,
                    bs=args.bs,
                    intp=args.intp,
                    alpha=args.alpha,
                )
            )

        pred_feats["global"].extend(
            decode_setting_features(
                vae=vae,
                inception=inception,
                zq_cpu=zq_cpu,
                y_cpu=y_cpu,
                ref_indices_cpu=global_idx,
                ref_cpu=ref_cpu,
                device=device,
                bs=args.bs,
                intp=args.intp,
                alpha=args.alpha,
            )
        )

        del (
            zq_gpu,
            q_norm,
            hybrid_idx,
            global_idx,
            _global_sim,
        )

    global_pos = 0

    print("=" * 96)
    print("Hybrid DINO-filtered / VAE-retrieved iFID")
    print("model                 :", args.exp_name)
    print("latent shape          :", latent_shape)
    print("ref / val             :", args.ref_size, "/", args.val_size)
    print("DINO K               :", K_LIST)
    print("final retrieval       : VAE whole-latent cosine hard top-1")
    print("endpoint              : exact global VAE hard top-1 over 50k")
    print("interpolation         :", args.intp, "alpha=", args.alpha)
    print("VAE micro-batch       :", args.bs)
    print("query retrieval block :", args.query_block_size)
    print("candidate chunk       :", args.candidate_chunk)
    print("global ref chunk      :", args.global_ref_chunk)
    print("CPU ref bank GB       :", f"{cpu_bank_gb:.2f}")
    print("GPU ref bank          :", ref_gpu is not None)
    print("=" * 96)

    for img, y, names in tqdm(
        val_loader,
        desc=f"hybrid-iFID {args.exp_name}",
    ):
        B = img.shape[0]

        # Exact order sanity for every query.
        for j, name in enumerate(names):
            expected = dino_val_names[global_pos + j]
            if str(name) != expected:
                raise RuntimeError(
                    f"Validation order mismatch at {global_pos+j}: "
                    f"{name!r} vs {expected!r}"
                )

        img = img.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)

        z = encode_vae(
            vae, img, y_dev
        ).float()

        # Source Inception features are shared by all six metrics.
        img01 = (img.float() + 1.0) / 2.0
        src_feats.append(
            inception_features(
                inception, img01
            ).float().cpu().numpy()
        )

        z_buf.append(z.cpu())
        y_buf.append(y.cpu())
        index_buf.extend(
            range(global_pos, global_pos + B)
        )

        global_pos += B

        del img, y_dev, z, img01

        buffered = sum(x.shape[0] for x in z_buf)
        if buffered >= args.query_block_size:
            zq_cpu = torch.cat(z_buf, dim=0)
            y_cpu = torch.cat(y_buf, dim=0)
            qidx = list(index_buf)

            # Because val_loader batches are contiguous, this should hold.
            if qidx != list(
                range(qidx[0], qidx[0] + len(qidx))
            ):
                raise RuntimeError(
                    "Query block is not contiguous"
                )

            process_block(
                zq_cpu, y_cpu, qidx
            )

            z_buf.clear()
            y_buf.clear()
            index_buf.clear()

    if z_buf:
        zq_cpu = torch.cat(z_buf, dim=0)
        y_cpu = torch.cat(y_buf, dim=0)
        qidx = list(index_buf)
        process_block(zq_cpu, y_cpu, qidx)

    if global_pos != args.val_size:
        raise RuntimeError(
            f"Processed {global_pos} queries, expected {args.val_size}"
        )

    src = np.concatenate(
        src_feats, axis=0
    )
    if src.shape[0] != args.val_size:
        raise RuntimeError("Source feature count mismatch")

    metrics = {}
    suffix = f"{args.intp}_a{args.alpha:g}"

    for k in K_LIST:
        pred = np.concatenate(
            pred_feats[k], axis=0
        )
        if pred.shape[0] != args.val_size:
            raise RuntimeError(
                f"K={k} pred feature count mismatch"
            )
        key = (
            f"{metric_names[k]}_{suffix}"
        )
        print(f"[FID] K={k}", flush=True)
        metrics[key] = fid_from_features(
            src, pred
        )

    pred = np.concatenate(
        pred_feats["global"], axis=0
    )
    if pred.shape[0] != args.val_size:
        raise RuntimeError(
            "global pred feature count mismatch"
        )
    global_key = (
        f"{metric_names['global']}_{suffix}"
    )
    print("[FID] GLOBAL hard top-1", flush=True)
    metrics[global_key] = fid_from_features(
        src, pred
    )

    hit_rates = {
        f"global_hard_nn_in_dino_top{k}": (
            hit_counts[k] / max(total_queries, 1)
        )
        for k in K_LIST
    }

    result = {
        "vae_config": args.vae_config,
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "dataset_ref": args.dataset_ref,
        "ref_size": int(args.ref_size),
        "val_size": int(args.val_size),
        "latent_shape": list(latent_shape),
        "dino_cache": str(
            Path(args.dino_cache).resolve()
        ),
        "dino_candidate_sizes": list(K_LIST),
        "candidate_filter": "DINO global cosine top-K",
        "final_nn_metric": "VAE whole-latent cosine hard top-1",
        "global_endpoint": (
            "exact VAE whole-latent cosine hard top-1 over all refs"
        ),
        "interpolation": f"whole_latent_{args.intp}",
        "alpha": float(args.alpha),
        "reference_latent_storage_dtype": str(ref_cpu.dtype),
        "gpu_reference_bank": ref_gpu is not None,
        "metrics": metrics,
        "diagnostics": hit_rates,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2)
    )

    with (out_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["setting", "ifid"])
        for k in K_LIST:
            key = (
                f"{metric_names[k]}_{suffix}"
            )
            w.writerow([f"DINO_top{k}", metrics[key]])
        w.writerow(
            ["GLOBAL_VAE_hard_top1_50k", metrics[global_key]]
        )

        w.writerow([])
        w.writerow(["diagnostic", "value"])
        for key, value in hit_rates.items():
            w.writerow([key, value])

    print("\n" + "=" * 96)
    print("DONE:", args.exp_name)
    for k in K_LIST:
        key = f"{metric_names[k]}_{suffix}"
        print(
            f"K={k:<4d}  iFID={metrics[key]:.6f}  "
            f"global-hit={hit_rates[f'global_hard_nn_in_dino_top{k}']:.4f}"
        )
    print(
        "GLOBAL hard top1  iFID="
        f"{metrics[global_key]:.6f}"
    )
    print("metrics:", out_dir / "metrics.json")
    print("=" * 96)


if __name__ == "__main__":
    main(parse_args())
