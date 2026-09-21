#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf


def import_evaluator(guided_diffusion_dir: str):
    eval_dir = Path(guided_diffusion_dir).resolve() / "evaluations"
    evaluator_file = eval_dir / "evaluator.py"

    if not evaluator_file.exists():
        raise FileNotFoundError(
            f"Cannot find guided-diffusion evaluator: {evaluator_file}"
        )

    sys.path.insert(0, str(eval_dir))
    import evaluator

    return evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guided-diffusion-dir", required=True)
    parser.add_argument("--ref-npz", required=True)
    parser.add_argument("--out-npy", required=True)
    parser.add_argument("--tf-batch-size", type=int, default=64)
    parser.add_argument("--softmax-batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ref_npz = Path(args.ref_npz)
    out_npy = Path(args.out_npy)
    out_json = Path(str(out_npy) + ".json")

    if not ref_npz.exists():
        raise FileNotFoundError(ref_npz)

    if (
        out_npy.exists()
        and out_npy.stat().st_size > 100_000_000
        and not args.overwrite
    ):
        print(f"[SKIP] Reference cache already exists: {out_npy}")
        return

    out_npy.parent.mkdir(parents=True, exist_ok=True)

    tf.disable_v2_behavior()
    guided_evaluator = import_evaluator(args.guided_diffusion_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        evaluator = guided_evaluator.Evaluator(
            session,
            batch_size=args.tf_batch_size,
            softmax_batch_size=args.softmax_batch_size,
        )

        print("[1/2] Warming up evaluator...")
        evaluator.warmup()

        print("[2/2] Computing reference activations...")
        ref_acts = evaluator.read_activations(str(ref_npz))
        ref_pool = np.asarray(ref_acts[0])

    tmp_npy = Path(str(out_npy) + ".tmp.npy")

    if tmp_npy.exists():
        tmp_npy.unlink()

    np.save(tmp_npy, ref_pool)
    os.replace(tmp_npy, out_npy)

    metadata = {
        "ref_npz": str(ref_npz),
        "out_npy": str(out_npy),
        "shape": list(ref_pool.shape),
        "dtype": str(ref_pool.dtype),
    }

    out_json.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))
    print(f"[OK] Saved reference pool: {out_npy}")


if __name__ == "__main__":
    main()
