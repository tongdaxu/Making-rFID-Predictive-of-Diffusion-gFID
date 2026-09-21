#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import tensorflow.compat.v1 as tf


def import_guided_evaluator(guided_diffusion_dir: str):
    gd = Path(guided_diffusion_dir).resolve()
    eval_dir = gd / "evaluations"
    if not (eval_dir / "evaluator.py").exists():
        raise FileNotFoundError(f"Cannot find {eval_dir / 'evaluator.py'}")
    sys.path.insert(0, str(eval_dir))
    import evaluator as gd_eval
    return gd_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guided-diffusion-dir", type=str, required=True)
    parser.add_argument("--sample-npz", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--tf-batch-size", type=int, default=64)
    parser.add_argument("--softmax-batch-size", type=int, default=512)
    args = parser.parse_args()

    gd_eval = import_guided_evaluator(args.guided_diffusion_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        evaluator = gd_eval.Evaluator(
            sess,
            batch_size=args.tf_batch_size,
            softmax_batch_size=args.softmax_batch_size,
        )

        print("warming up TensorFlow...")
        evaluator.warmup()

        print("computing sample activations...")
        sample_acts = evaluator.read_activations(args.sample_npz)

        print("computing inception score...")
        inception_score = evaluator.compute_inception_score(sample_acts[0])

    results = {
        "sample_npz": args.sample_npz,
        "inception_score": float(inception_score),
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
