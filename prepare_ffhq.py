#!/usr/bin/env python3

import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


PARQUET_DIR = Path("/mnt/task_runtime/dataset/ffhq/data")
OUTPUT_DIR = Path("/mnt/task_runtime/dataset/ffhq/imgs")


def decode_image(obj):
    """Decode one image stored in a HuggingFace parquet file."""

    obj = obj.as_py() if hasattr(obj, "as_py") else obj

    if isinstance(obj, dict):
        # {"bytes": ..., "path": ...}
        return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")

    if isinstance(obj, bytes):
        return Image.open(io.BytesIO(obj)).convert("RGB")

    if isinstance(obj, Image.Image):
        return obj.convert("RGB")

    raise TypeError(f"Unsupported image type: {type(obj)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))

    if len(parquet_files) == 0:
        raise RuntimeError(f"No parquet files found in {PARQUET_DIR}")

    img_idx = 0

    for parquet_file in parquet_files:
        print(f"Processing {parquet_file.name}")

        table = pq.read_table(parquet_file)

        if "image" not in table.column_names:
            raise RuntimeError(
                f"'image' column not found.\nColumns: {table.column_names}"
            )

        images = table["image"]

        for image in tqdm(images, leave=False):
            image = decode_image(image)
            image.save(OUTPUT_DIR / f"{img_idx:06d}.png")
            img_idx += 1

    print(f"\nFinished! Saved {img_idx} images to")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()