#!/usr/bin/env python3
"""
Sanity check for the P=1 local-iFID hypothesis.

Compare, on the SAME query images and SAME 50k reference bank:

1) Exact same-position P=1 hard cosine NN:
       each local vector chooses its own reference image.
2) Exact whole-latent hard cosine NN:
       one reference image is chosen for the whole latent.

The purpose is to test whether P=1 local retrieval is "too easy":
local selected vectors should have cosine similarity extremely close to 1,
while the best whole-latent NN should be substantially farther away.

No decoding and no FID are computed.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config

from eval_local_score_rfid_25_fast import (
    encode_vae,
    native_to_spatial,
    infer_token_grid,
    build_or_load_reference_cache,
    dtype_from_name,
)

from eval_local_ifid_p1_global_slerp_25_fast import (
    load_raw_reference_bank_gpu,
    exact_same_position_nn_p1,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-config", required=True)
    p.add_argument("--exp-name", required=True)

    p.add_argument("--dataset", default="/video_ssd/lpm/ImageNet/val")
    p.add_argument("--dataset-ref", default="/video_ssd/lpm/ImageNet/train")
    p.add_argument("--ref-size", type=int, default=50000)
    p.add_argument("--val-size", type=int, default=128)
    p.add_argument("--resolution", type=int, default=256)

    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--ref-bs", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--token-grid", type=int, nargs=2, default=None)

    p.add_argument("--ref-chunk-size", type=int, default=4096)
    p.add_argument("--position-chunk-size", type=int, default=64)
    p.add_argument("--query-chunk-size", type=int, default=32)
    p.add_argument(
        "--global-ref-chunk-size",
        type=int,
        default=4096,
    )

    p.add_argument(
        "--metric-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
    )
    p.add_argument(
        "--gpu-bank-dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
    )

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

    p.add_argument(
        "--out-dir",
        default="/share/xutongda/REPA-E/exps_interpolate/local_ifid_p1_sanity",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def exact_global_hard_cosine_nn(
    query_native,
    ref_bank_native,
    ref_chunk_size,
    metric_dtype,
):
    """
    Exact hard top-1 cosine NN over the whole flattened latent.
    Returns raw reference latents.
    """
    B = query_native.shape[0]
    qf = query_native.float().reshape(B, -1)
    qn = F.normalize(qf, dim=1).to(metric_dtype)

    best_sim = torch.full(
        (B,),
        -float("inf"),
        device=query_native.device,
        dtype=torch.float32,
    )
    best_idx = torch.zeros(
        (B,),
        device=query_native.device,
        dtype=torch.long,
    )

    N = ref_bank_native.shape[0]
    step = max(1, int(ref_chunk_size))

    for s in range(0, N, step):
        e = min(N, s + step)
        rf = ref_bank_native[s:e].float().reshape(e - s, -1)
        rn = F.normalize(rf, dim=1).to(metric_dtype)

        sim = (qn @ rn.T).float()
        chunk_sim, chunk_idx = sim.max(dim=1)
        improve = chunk_sim > best_sim

        best_sim = torch.where(improve, chunk_sim, best_sim)
        best_idx = torch.where(improve, chunk_idx + s, best_idx)

        del rf, rn, sim, chunk_sim, chunk_idx, improve

    nn = ref_bank_native.index_select(0, best_idx).float()
    return nn, best_sim


def flat_cos_per_image(a, b, eps=1e-12):
    af = a.float().reshape(a.shape[0], -1)
    bf = b.float().reshape(b.shape[0], -1)
    return F.cosine_similarity(af, bf, dim=1, eps=eps)


def rel_l2_per_image(a, b, eps=1e-12):
    af = a.float().reshape(a.shape[0], -1)
    bf = b.float().reshape(b.shape[0], -1)
    num = torch.linalg.vector_norm(af - bf, dim=1)
    den = torch.linalg.vector_norm(af, dim=1).clamp_min(eps)
    return num / den


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

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

    fake_x = torch.zeros(1, 3, args.resolution, args.resolution, device=device)
    fake_y = torch.zeros(1, dtype=torch.long, device=device)
    fake_z = encode_vae(vae, fake_x, fake_y)
    native_shape = tuple(fake_z.shape[1:])

    if len(native_shape) == 2:
        infer_token_grid(int(native_shape[0]), args.token_grid)

    cache_files, cache_shape, _mean, _std = build_or_load_reference_cache(
        args, vae, train_loader, device
    )
    if tuple(cache_shape) != native_shape:
        raise RuntimeError(
            f"Cache/probe mismatch: {cache_shape} vs {native_shape}"
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

    # Accumulators.
    local_position_cos = []
    local_whole_cos = []
    local_whole_rel_l2 = []
    global_whole_cos = []
    global_whole_rel_l2 = []

    local_gt_099 = 0
    local_gt_0999 = 0
    local_gt_09999 = 0
    local_count = 0

    for img, y, _names in tqdm(
        val_loader, desc=f"sanity {args.exp_name}"
    ):
        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = encode_vae(vae, img, y).float()

        # Exact P=1 same-position hard cosine NN.
        z_local = exact_same_position_nn_p1(
            query_native=z,
            ref_bank_native=ref_bank,
            native_shape=native_shape,
            token_grid=args.token_grid,
            intp="slerp",
            ref_chunk_size=args.ref_chunk_size,
            position_chunk_size=args.position_chunk_size,
            query_chunk_size=args.query_chunk_size,
            metric_dtype=metric_dtype,
        )

        z_sp = native_to_spatial(z, args.token_grid)
        l_sp = native_to_spatial(z_local, args.token_grid)

        # One cosine per spatial/token position.
        pos_cos = F.cosine_similarity(
            z_sp.float(), l_sp.float(), dim=1
        ).flatten()

        local_position_cos.extend(pos_cos.cpu().tolist())
        local_count += pos_cos.numel()
        local_gt_099 += int((pos_cos > 0.99).sum().item())
        local_gt_0999 += int((pos_cos > 0.999).sum().item())
        local_gt_09999 += int((pos_cos > 0.9999).sum().item())

        local_whole_cos.extend(
            flat_cos_per_image(z, z_local).cpu().tolist()
        )
        local_whole_rel_l2.extend(
            rel_l2_per_image(z, z_local).cpu().tolist()
        )

        # Exact whole-latent hard cosine NN, same 50k bank.
        z_global, gsim = exact_global_hard_cosine_nn(
            query_native=z,
            ref_bank_native=ref_bank,
            ref_chunk_size=args.global_ref_chunk_size,
            metric_dtype=metric_dtype,
        )
        global_whole_cos.extend(gsim.cpu().tolist())
        global_whole_rel_l2.extend(
            rel_l2_per_image(z, z_global).cpu().tolist()
        )

        del z, z_local, z_sp, l_sp, pos_cos, z_global, gsim

    result = {
        "vae_config": args.vae_config,
        "exp_name": args.exp_name,
        "ref_size": args.ref_size,
        "val_size": args.val_size,
        "native_latent_shape": list(native_shape),
        "token_grid": args.token_grid,
        "metric": "hard cosine NN on raw latents",
        "local_definition": "P=1 exact same-position over all refs",
        "global_definition": "exact whole-latent hard top-1 over all refs",
        "local_position_cosine": summarize(local_position_cos),
        "local_position_cosine_fraction_gt_0.99": local_gt_099 / local_count,
        "local_position_cosine_fraction_gt_0.999": local_gt_0999 / local_count,
        "local_position_cosine_fraction_gt_0.9999": local_gt_09999 / local_count,
        "local_composite_whole_latent_cosine": summarize(local_whole_cos),
        "local_composite_relative_l2": summarize(local_whole_rel_l2),
        "global_nn_whole_latent_cosine": summarize(global_whole_cos),
        "global_nn_relative_l2": summarize(global_whole_rel_l2),
    }

    path = out_dir / "sanity_stats.json"
    path.write_text(json.dumps(result, indent=2))

    csv_path = out_dir / "sanity_stats.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stat", "value"])
        rows = [
            ("local_position_cos_mean", result["local_position_cosine"]["mean"]),
            ("local_position_cos_median", result["local_position_cosine"]["median"]),
            ("local_position_cos_p05", result["local_position_cosine"]["p05"]),
            ("local_pos_fraction_cos_gt_0.99", result["local_position_cosine_fraction_gt_0.99"]),
            ("local_pos_fraction_cos_gt_0.999", result["local_position_cosine_fraction_gt_0.999"]),
            ("local_pos_fraction_cos_gt_0.9999", result["local_position_cosine_fraction_gt_0.9999"]),
            ("local_composite_whole_cos_mean", result["local_composite_whole_latent_cosine"]["mean"]),
            ("local_composite_rel_l2_mean", result["local_composite_relative_l2"]["mean"]),
            ("global_nn_whole_cos_mean", result["global_nn_whole_latent_cosine"]["mean"]),
            ("global_nn_rel_l2_mean", result["global_nn_relative_l2"]["mean"]),
        ]
        w.writerows(rows)

    print("\n" + "=" * 76)
    print("P=1 LOCAL NN vs WHOLE-LATENT NN SANITY")
    print("model:", args.exp_name)
    print("latent:", native_shape)
    print("queries / refs:", args.val_size, "/", args.ref_size)
    print("-" * 76)
    print(
        "local position cosine : "
        f"mean={result['local_position_cosine']['mean']:.6f}  "
        f"median={result['local_position_cosine']['median']:.6f}  "
        f"p05={result['local_position_cosine']['p05']:.6f}"
    )
    print(
        "local cos fractions   : "
        f">.99={result['local_position_cosine_fraction_gt_0.99']:.4f}  "
        f">.999={result['local_position_cosine_fraction_gt_0.999']:.4f}  "
        f">.9999={result['local_position_cosine_fraction_gt_0.9999']:.4f}"
    )
    print(
        "local composite whole : "
        f"cos={result['local_composite_whole_latent_cosine']['mean']:.6f}  "
        f"relL2={result['local_composite_relative_l2']['mean']:.6f}"
    )
    print(
        "global NN whole       : "
        f"cos={result['global_nn_whole_latent_cosine']['mean']:.6f}  "
        f"relL2={result['global_nn_relative_l2']['mean']:.6f}"
    )
    print("-" * 76)
    print("saved:", path)
    print("=" * 76)


if __name__ == "__main__":
    main(parse_args())
