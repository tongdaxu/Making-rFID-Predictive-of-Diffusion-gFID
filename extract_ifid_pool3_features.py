#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract ONLY OpenAI guided-diffusion Inception-v3 pool_3 features from the
already-packed original-iFID NPZs.

Why this script exists:
- guided-diffusion Evaluator.read_activations() computes both pool_3 and spatial
  features. For iFID / iPrec / iRec / iFID-R we only need pool_3.
- This script runs only `evaluator.pool_features`, so each 50k-image NPZ becomes
  ~390 MiB float32 instead of keeping a ~9.2 GiB image NPZ.
- Output is .npy [50000, 2048], float32, written incrementally with open_memmap.

The feature extractor is exactly the pool_3 graph created by the repository's
OpenAI guided-diffusion evaluations/evaluator.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm


MODELS = [
    ("SD-VAE", "sdvae"),
    ("SD3-VAE", "sd3vae"),
    ("FLUX-VAE", "fluxvae"),
    ("DC-AE", "dcae"),
    ("SOFT-VQ", "softvq"),
    ("VA-VAE", "vavae"),
    ("VA-VAE-64", "vavae64"),
    ("EQ-VAE", "eqvae"),
    ("MAE-TOK", "maetok"),
    ("IN-VAE", "invae"),
    ("REPAE-SDVAE", "repae-sdvae"),
    ("DE-TOK", "detok"),
    ("QwenImg-VAE", "qwenimgvae"),
    ("RAE", "rae"),
    ("SVG", "svg"),
    ("FLUX2-VAE", "flux2"),
    ("SVG T2I", "svg-t2i-p256"),
    ("DM-VAE", "dmvae"),
    ("UAE", "uae"),
    ("VTP-S", "vtps"),
    ("VTP-B", "vtpb"),
    ("VTP-L", "vtpl"),
    ("RAE T2I SigLIP-2", "scalerae-siglip2"),
    ("RAE T2I WebSSL", "scalerae-webssl"),
    ("PAE DINOv2", "pae-dino"),
]

SAMPLE_SUFFIX = "-slerp-a05-top10-200000x50000.npz"
REFERENCE_FILENAME = "imagenet-val-src_dist-50000.npz"


def import_evaluator(guided_diffusion_dir: str):
    eval_dir = Path(guided_diffusion_dir).resolve() / "evaluations"
    evaluator_file = eval_dir / "evaluator.py"
    if not evaluator_file.exists():
        raise FileNotFoundError(
            f"Cannot find guided-diffusion evaluator: {evaluator_file}"
        )
    sys.path.insert(0, str(eval_dir))
    import evaluator
    return evaluator, evaluator_file


def atomic_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def output_is_complete(out_npy: Path, expected_n: int = 50000) -> bool:
    if not out_npy.exists():
        return False
    try:
        x = np.load(out_npy, mmap_mode="r")
        return x.shape == (expected_n, 2048) and x.dtype == np.float32
    except Exception:
        return False


def extract_pool3(evaluator_module, evaluator, src_npz: Path, out_npy: Path, batch_size: int):
    """
    Stream arr_0 from src_npz and run ONLY pool_3.
    """
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy = Path(str(out_npy) + ".tmp.npy")
    if tmp_npy.exists():
        tmp_npy.unlink()

    with evaluator_module.open_npz_array(str(src_npz), "arr_0") as reader:
        n = int(reader.remaining())
        if n != 50000:
            raise ValueError(f"{src_npz}: expected 50000 images, got {n}")

        out = np.lib.format.open_memmap(
            tmp_npy,
            mode="w+",
            dtype=np.float32,
            shape=(n, 2048),
        )

        offset = 0
        batches = reader.read_batches(batch_size)
        for batch in tqdm(
            batches,
            total=len(batches),
            desc=src_npz.name,
            dynamic_ncols=True,
        ):
            batch = batch.astype(np.float32, copy=False)
            pred = evaluator.sess.run(
                evaluator.pool_features,
                feed_dict={evaluator.image_input: batch},
            )
            pred = pred.reshape(pred.shape[0], -1).astype(np.float32, copy=False)
            if pred.shape[1] != 2048:
                raise RuntimeError(f"Unexpected pool_3 shape: {pred.shape}")

            b = pred.shape[0]
            out[offset:offset + b] = pred
            offset += b

        out.flush()
        del out

    if offset != 50000:
        raise RuntimeError(f"{src_npz}: wrote {offset} features, expected 50000")

    os.replace(tmp_npy, out_npy)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--guided-diffusion-dir", required=True)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--inception-pb",
        default=None,
        help="Optional local classify_image_graph_def.pb. If set, no model download is needed.",
    )
    p.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reference is assigned as job 0; with multiple workers only worker 0 handles it.",
    )
    p.add_argument(
        "--skip-done",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--only",
        default=None,
        help="Optional exact display name, tag, or 'REFERENCE'.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    npz_dir = Path(args.npz_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_dir.is_dir():
        raise FileNotFoundError(npz_dir)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >=1")
    if not (0 <= args.worker_index < args.num_workers):
        raise ValueError("worker-index out of range")

    jobs = []

    # Reference job gets virtual index 0.
    if args.include_reference:
        src = npz_dir / REFERENCE_FILENAME
        dst = out_dir / "reference_imagenet_val_50k_pool3.npy"
        manifest = out_dir / "reference_imagenet_val_50k_pool3.json"
        jobs.append((0, "REFERENCE", "reference", src, dst, manifest))

    # Model jobs get virtual indices 1..25.
    for i, (model, tag) in enumerate(MODELS, start=1):
        src = npz_dir / f"ifid-iis-{tag}{SAMPLE_SUFFIX}"
        dst = out_dir / f"{tag}_pool3.npy"
        manifest = out_dir / f"{tag}_pool3.json"
        jobs.append((i, model, tag, src, dst, manifest))

    selected = []
    for virtual_idx, model, tag, src, dst, manifest in jobs:
        if args.only is not None:
            if args.only not in {model, tag, "REFERENCE" if model == "REFERENCE" else ""}:
                continue
        elif virtual_idx % args.num_workers != args.worker_index:
            continue

        if args.skip_done and output_is_complete(dst):
            print(f"[SKIP DONE] {model}: {dst}", flush=True)
            continue

        if not src.exists():
            raise FileNotFoundError(f"Missing source NPZ for {model}: {src}")
        selected.append((virtual_idx, model, tag, src, dst, manifest))

    if not selected:
        print("[INFO] no pending jobs for this worker")
        return

    tf.disable_v2_behavior()
    gd_eval, evaluator_file = import_evaluator(args.guided_diffusion_dir)

    if args.inception_pb is not None:
        pb = Path(args.inception_pb).resolve()
        if not pb.exists():
            raise FileNotFoundError(pb)
        gd_eval.INCEPTION_V3_PATH = str(pb)
        print(f"[INCEPTION PB] {pb}", flush=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        evaluator = gd_eval.Evaluator(
            sess,
            batch_size=args.batch_size,
            softmax_batch_size=512,
        )

        # Warm up the exact same pool_3 graph.
        evaluator.warmup()

        for ji, (_, model, tag, src, dst, manifest) in enumerate(selected, 1):
            print("\n" + "=" * 100)
            print(f"[{ji}/{len(selected)}] {model}")
            print(f"SRC: {src}")
            print(f"DST: {dst}")
            print("=" * 100, flush=True)

            extract_pool3(
                gd_eval,
                evaluator,
                src_npz=src,
                out_npy=dst,
                batch_size=args.batch_size,
            )

            x = np.load(dst, mmap_mode="r")
            info = {
                "schema_version": 1,
                "model": model,
                "tag": tag,
                "source_npz": str(src),
                "source_size_bytes": int(src.stat().st_size),
                "source_mtime_ns": int(src.stat().st_mtime_ns),
                "pool3_npy": str(dst),
                "pool3_shape": list(x.shape),
                "pool3_dtype": str(x.dtype),
                "guided_evaluator_file": str(evaluator_file.resolve()),
                "feature": "OpenAI guided-diffusion Inception-v3 pool_3",
            }
            atomic_write_json(manifest, info)
            print(f"[OK] {dst} shape={x.shape} dtype={x.dtype}", flush=True)


if __name__ == "__main__":
    main()
