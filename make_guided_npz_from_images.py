#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a folder of generated images to OpenAI guided-diffusion evaluator .npz format.

Strict default:
  - no resize
  - no crop
  - preserve generated uint8 RGB images as-is

The evaluator expects:
  arr_0: uint8 NHWC images in [0,255]

Example:
  python make_guided_npz_from_images.py \
    --image-dir samples/sit-xl-sdvae_400000_cfg1.0 \
    --out-npz npz/sdvae_50k.npz \
    --limit 50000
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not files:
        raise RuntimeError(f"No images found under {image_dir}")
    return files


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def load_image(path: Path, resize: int = -1, center_crop: bool = False) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if center_crop:
        img = center_crop_square(img)
    if resize is not None and resize > 0:
        img = img.resize((resize, resize), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"Bad image shape for {path}: {arr.shape}")
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--resize", type=int, default=-1, help="-1 means no resize, strict guided-diffusion style.")
    parser.add_argument("--center-crop", action="store_true", help="Off by default; do not use for strict eval.")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--strict-same-shape", action="store_true", default=True)
    args = parser.parse_args()

    files = list_images(args.image_dir)
    if args.limit is not None and args.limit > 0:
        files = files[: args.limit]

    arrays = []
    first_shape = None
    for p in tqdm(files, desc=f"Loading {args.image_dir}"):
        arr = load_image(p, resize=args.resize, center_crop=args.center_crop)
        if first_shape is None:
            first_shape = arr.shape
        elif args.strict_same_shape and arr.shape != first_shape:
            raise RuntimeError(
                f"Image shapes differ. First shape={first_shape}, but {p} has {arr.shape}. "
                "For strict eval, generated samples should already have one resolution. "
                "If you intentionally want post-resize, pass --resize 256."
            )
        arrays.append(arr)

    arr = np.stack(arrays, axis=0).astype(np.uint8)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.compressed:
        np.savez_compressed(out, arr_0=arr)
    else:
        np.savez(out, arr_0=arr)

    print(f"Saved {out}")
    print(f"arr_0 shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
    if args.resize is not None and args.resize > 0:
        print(f"WARNING: images were resized to {args.resize}. This is not strict guided-diffusion eval.")


if __name__ == "__main__":
    main()
