#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def first_int_key(path: Path):
    m = re.search(r"\d+", path.stem)
    if m:
        return int(m.group(0))
    return 10**18


def find_intp_dir(sample_root: Path, exp_name: str, subdir_name: str):
    candidates = []

    for d in sample_root.iterdir():
        if not d.is_dir():
            continue
        if d.name == exp_name or d.name.startswith(exp_name):
            intp = d / subdir_name
            if intp.is_dir():
                n = sum(1 for p in intp.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
                candidates.append((n, intp))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    print("=" * 100)
    print(f"[SEARCH] sample_root={sample_root}")
    print(f"[SEARCH] exp_name={exp_name}")
    print(f"[SEARCH] subdir_name={subdir_name}")
    print("[CANDIDATES]")
    for n, d in candidates:
        print(f"  count={n:6d}  {d}")
    print("=" * 100)

    if not candidates:
        raise RuntimeError(f"No candidate {subdir_name} folder found for exp_name={exp_name}")

    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--subdir-name", type=str, default="intp_dist")
    parser.add_argument("--expected", type=int, default=50000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sample_root = Path(args.sample_root)
    out_npz = Path(args.out_npz)

    if out_npz.exists() and out_npz.stat().st_size > 0 and not args.overwrite:
        print(f"[SKIP] npz already exists: {out_npz}")
        return

    count, intp_dir = find_intp_dir(sample_root, args.exp_name, args.subdir_name)

    if count < args.expected:
        raise RuntimeError(
            f"Not enough images in {intp_dir}: found {count}, expected {args.expected}"
        )

    files = sorted(
        [p for p in intp_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: (first_int_key(p), str(p)),
    )[:args.expected]

    print(f"[SELECTED] intp_dir={intp_dir}")
    print(f"[SELECTED] files={len(files)}")
    print(f"[SELECTED] first={files[0]}")
    print(f"[SELECTED] last ={files[-1]}")

    arr = np.empty((args.expected, args.image_size, args.image_size, 3), dtype=np.uint8)

    for i, p in enumerate(tqdm(files, desc="Packing intp_dist images")):
        img = Image.open(p).convert("RGB")
        if img.size != (args.image_size, args.image_size):
            raise RuntimeError(f"Image size mismatch: {p}, got={img.size}, expected={(args.image_size, args.image_size)}")
        arr[i] = np.asarray(img, dtype=np.uint8)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, arr_0=arr)

    print(f"[OK] saved {out_npz}")
    print(f"shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
