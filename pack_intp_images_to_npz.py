#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", str(path))
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--expected", type=int, default=50000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_npz = Path(args.out_npz)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    # 50k × 256 × 256 × 3 uint8 约 9.15 GiB。
    # 已存在的大文件默认认为已经打包完成。
    if (
        out_npz.exists()
        and out_npz.stat().st_size > 1_000_000_000
        and not args.overwrite
    ):
        print(
            f"[SKIP] NPZ already exists: {out_npz} "
            f"size={out_npz.stat().st_size / 1024**3:.2f} GiB"
        )
        return

    images = sorted(
        [
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ],
        key=natural_key,
    )

    print("=" * 100)
    print(f"IMAGE_DIR: {image_dir}")
    print(f"FOUND:     {len(images)}")
    print(f"EXPECTED:  {args.expected}")
    print(f"OUT_NPZ:   {out_npz}")
    print("=" * 100)

    if len(images) != args.expected:
        raise RuntimeError(
            f"Image count must be exactly {args.expected}, "
            f"but found {len(images)} in {image_dir}"
        )

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    tmp_npy = Path(str(out_npz) + ".tmp.npy")
    tmp_npz = Path(str(out_npz) + ".tmp")

    for path in (tmp_npy, tmp_npz):
        if path.exists():
            path.unlink()

    array = np.lib.format.open_memmap(
        tmp_npy,
        mode="w+",
        dtype=np.uint8,
        shape=(
            args.expected,
            args.image_size,
            args.image_size,
            3,
        ),
    )

    try:
        for index, path in enumerate(tqdm(images, desc="Packing images")):
            with Image.open(path) as image:
                image = image.convert("RGB")

                if image.size != (args.image_size, args.image_size):
                    raise RuntimeError(
                        f"Unexpected image size: {path}; "
                        f"got={image.size}, "
                        f"expected={(args.image_size, args.image_size)}"
                    )

                array[index] = np.asarray(image, dtype=np.uint8)

        array.flush()
        del array

        mmap_array = np.load(tmp_npy, mmap_mode="r")

        # guided-diffusion evaluator 读取 arr_0。
        with tmp_npz.open("wb") as file:
            np.savez(file, arr_0=mmap_array)

        del mmap_array

        os.replace(tmp_npz, out_npz)
        tmp_npy.unlink()

    except Exception:
        for path in (tmp_npy, tmp_npz):
            if path.exists():
                path.unlink()
        raise

    print(
        f"[OK] Saved: {out_npz} "
        f"size={out_npz.stat().st_size / 1024**3:.2f} GiB"
    )


if __name__ == "__main__":
    main()
