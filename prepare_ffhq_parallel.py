#!/usr/bin/env python3

import argparse
import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def decode_image(obj):
    obj = obj.as_py() if hasattr(obj, "as_py") else obj

    if isinstance(obj, dict):
        return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")

    if isinstance(obj, bytes):
        return Image.open(io.BytesIO(obj)).convert("RGB")

    if isinstance(obj, Image.Image):
        return obj.convert("RGB")

    raise TypeError(f"Unsupported image type: {type(obj)}")


def count_rows(parquet_path: str) -> int:
    pf = pq.ParquetFile(parquet_path)
    return pf.metadata.num_rows


def process_one_parquet(args):
    parquet_path, output_dir, offset, batch_size, overwrite, compress_level = args

    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)

    pf = pq.ParquetFile(parquet_path)

    written = 0
    skipped = 0
    local_idx = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=["image"]):
        images = batch.column(0)

        for image_obj in images:
            global_idx = offset + local_idx
            out_path = output_dir / f"{global_idx:06d}.png"

            if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
                skipped += 1
                local_idx += 1
                continue

            img = decode_image(image_obj)
            tmp_path = out_path.with_suffix(".png.tmp")

            # PNG 是无损格式，compress_level 只影响文件大小和写入速度，不改变图像内容。
            img.save(tmp_path, format="PNG", compress_level=compress_level)
            os.replace(tmp_path, out_path)

            written += 1
            local_idx += 1

    return {
        "parquet": parquet_path.name,
        "rows": local_idx,
        "written": written,
        "skipped": skipped,
        "offset": offset,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default="/video_ssd/kongzishang/xutongda/git/IFID/datasets/ffhq/data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/video_ssd/kongzishang/xutongda/git/IFID/datasets/ffhq/imgs",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compress-level", type=int, default=1)
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))

    if len(parquet_files) == 0:
        raise RuntimeError(f"No parquet files found in {parquet_dir}")

    print("=" * 100)
    print("[FFHQ parallel prepare]")
    print("parquet_dir:", parquet_dir)
    print("output_dir :", output_dir)
    print("num_workers:", args.num_workers)
    print("batch_size :", args.batch_size)
    print("overwrite  :", args.overwrite)
    print("compress   :", args.compress_level)
    print("num parquet:", len(parquet_files))
    print("=" * 100)

    print("[1/3] Counting rows...")
    row_counts = []
    total = 0
    for p in tqdm(parquet_files):
        n = count_rows(str(p))
        row_counts.append(n)
        total += n

    print(f"Total rows/images: {total}")

    print("[2/3] Building tasks...")
    tasks = []
    offset = 0
    for p, n in zip(parquet_files, row_counts):
        tasks.append(
            (
                str(p),
                str(output_dir),
                offset,
                args.batch_size,
                args.overwrite,
                args.compress_level,
            )
        )
        offset += n

    print("[3/3] Decoding in parallel...")
    results = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(process_one_parquet, t) for t in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            res = fut.result()
            results.append(res)
            print(
                f"[DONE] {res['parquet']} "
                f"offset={res['offset']} rows={res['rows']} "
                f"written={res['written']} skipped={res['skipped']}"
            )

    written = sum(r["written"] for r in results)
    skipped = sum(r["skipped"] for r in results)
    rows = sum(r["rows"] for r in results)

    print("=" * 100)
    print("Finished.")
    print("rows   :", rows)
    print("written:", written)
    print("skipped:", skipped)
    print("output :", output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()
