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
    parser.add_argument("--reference-pool", required=True)
    parser.add_argument("--sample-npz", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tf-batch-size", type=int, default=64)
    parser.add_argument("--softmax-batch-size", type=int, default=512)
    args = parser.parse_args()

    reference_pool_path = Path(args.reference_pool)
    sample_npz = Path(args.sample_npz)
    out_json = Path(args.out_json)

    if not reference_pool_path.exists():
        raise FileNotFoundError(reference_pool_path)

    if not sample_npz.exists():
        raise FileNotFoundError(sample_npz)

    tf.disable_v2_behavior()
    guided_evaluator = import_evaluator(args.guided_diffusion_dir)

    reference_pool = np.load(reference_pool_path, mmap_mode="r")

    print(
        f"[REFERENCE] shape={reference_pool.shape}, "
        f"dtype={reference_pool.dtype}"
    )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        evaluator = guided_evaluator.Evaluator(
            session,
            batch_size=args.tf_batch_size,
            softmax_batch_size=args.softmax_batch_size,
        )

        print("[1/3] Warming up evaluator...")
        evaluator.warmup()

        print("[2/3] Computing sample activations...")
        sample_acts = evaluator.read_activations(str(sample_npz))
        sample_pool = sample_acts[0]

        print(
            f"[SAMPLE] shape={sample_pool.shape}, "
            f"dtype={sample_pool.dtype}"
        )

        print("[3/3] Computing iIS / iPrec / iRec...")
        iis = evaluator.compute_inception_score(sample_pool)

        iprec, irec = evaluator.compute_prec_recall(
            reference_pool,
            sample_pool,
        )

    results = {
        "model": args.model,
        "tag": args.tag,
        "sample_npz": str(sample_npz),
        "reference_pool": str(reference_pool_path),

        "iIS": float(iis),
        "iPrec": float(iprec),
        "iRec": float(irec),

        # 同时保留标准字段名，兼容之前的结果文件。
        "inception_score": float(iis),
        "precision": float(iprec),
        "recall": float(irec),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)

    tmp_json = Path(str(out_json) + ".tmp")
    tmp_json.write_text(json.dumps(results, indent=2))
    os.replace(tmp_json, out_json)

    print(json.dumps(results, indent=2))
    print(f"[OK] Saved: {out_json}")


if __name__ == "__main__":
    main()
