#!/usr/bin/env python3
"""
Build exact DINO top-K candidate indices from the already-built repository-DINO
feature cache.

Input cache was produced by build_repo_dinov2_nn_cache.py and contains:
    ref_features.npy  [N_ref, D], float32, L2-normalized
    val_features.npy  [N_val, D], float32, L2-normalized
    nn_indices.npy    optional existing exact top-1

This script computes exact cosine top-K over ALL reference images:
    C_i^(K) = TopK_j  f_val[i]^T f_ref[j]

No DINO model is loaded again.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-dir",
        required=True,
        help="Existing DINO feature cache directory.",
    )
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--query-chunk", type=int, default=128)
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--save-cosine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save top-K cosine values (large; not needed by hybrid-iFID).",
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


@torch.inference_mode()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.query_chunk <= 0:
        raise ValueError("--query-chunk must be > 0")

    root = Path(args.cache_dir)
    ref_path = root / "ref_features.npy"
    val_path = root / "val_features.npy"
    meta_path = root / "meta.json"

    for p in (ref_path, val_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(p)

    out_idx = root / f"nn_top{args.topk}_indices.npy"
    out_cos = root / f"nn_top{args.topk}_cosine.npy"

    ref_np = np.load(ref_path, mmap_mode="r")
    val_np = np.load(val_path, mmap_mode="r")

    if ref_np.ndim != 2 or val_np.ndim != 2:
        raise RuntimeError(
            f"Expected 2D feature arrays, got {ref_np.shape} and {val_np.shape}"
        )
    if ref_np.shape[1] != val_np.shape[1]:
        raise RuntimeError("DINO feature dimensions do not match")
    if args.topk > ref_np.shape[0]:
        raise ValueError(
            f"topk={args.topk} > number of references={ref_np.shape[0]}"
        )

    if out_idx.exists() and not args.force:
        idx = np.load(out_idx, mmap_mode="r")
        if idx.shape == (val_np.shape[0], args.topk):
            print(f"[cache] reuse {out_idx}")
            return

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    try:
        torch.set_float32_matmul_precision(
            "high" if args.tf32 else "highest"
        )
    except Exception:
        pass

    # 50k x 768/1024 is small enough to keep on a 4090.
    ref = torch.from_numpy(
        np.asarray(ref_np, dtype=np.float32)
    ).to(device=device, dtype=torch.float32).contiguous()

    n_val = int(val_np.shape[0])
    # int32 is sufficient for 50k refs and halves disk/RAM vs int64.
    idx_mm = np.lib.format.open_memmap(
        out_idx,
        mode="w+",
        dtype=np.int32,
        shape=(n_val, args.topk),
    )
    cos_mm = None
    if args.save_cosine:
        cos_mm = np.lib.format.open_memmap(
            out_cos,
            mode="w+",
            dtype=np.float32,
            shape=(n_val, args.topk),
        )

    for qs in tqdm(
        range(0, n_val, args.query_chunk),
        desc=f"exact DINO top-{args.topk}",
    ):
        qe = min(n_val, qs + args.query_chunk)
        q = torch.from_numpy(
            np.asarray(val_np[qs:qe], dtype=np.float32)
        ).to(device=device, dtype=torch.float32).contiguous()

        sim = q @ ref.T
        vals, idx = torch.topk(
            sim,
            k=args.topk,
            dim=1,
            largest=True,
            sorted=True,
        )

        idx_mm[qs:qe] = (
            idx.cpu().numpy().astype(np.int32, copy=False)
        )
        if cos_mm is not None:
            cos_mm[qs:qe] = (
                vals.cpu().numpy().astype(np.float32, copy=False)
            )

        del q, sim, vals, idx

    idx_mm.flush()
    if cos_mm is not None:
        cos_mm.flush()

    # The first top-K element must reproduce the previous exact DINO top-1.
    old_top1 = root / "nn_indices.npy"
    if old_top1.exists():
        old = np.load(old_top1)
        new = np.load(out_idx, mmap_mode="r")[:, 0].astype(np.int64)
        mismatch = int(np.count_nonzero(old.astype(np.int64) != new))
        if mismatch:
            raise RuntimeError(
                f"DINO top-1 verification FAILED: {mismatch}/{n_val} indices differ"
            )
        print("[VERIFY PASS] top-K[:,0] exactly matches existing DINO top-1")

    meta = json.loads(meta_path.read_text())
    meta[f"top{args.topk}_cache"] = {
        "indices": str(out_idx),
        "cosine": str(out_cos) if args.save_cosine else None,
        "metric": "exact cosine",
        "sorted": True,
        "dtype_indices": "int32",
        "tf32": bool(args.tf32),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("=" * 80)
    print(f"DINO exact top-{args.topk} cache done")
    print("indices:", out_idx)
    print("shape  :", (n_val, args.topk))
    print("=" * 80)


if __name__ == "__main__":
    main(parse_args())
