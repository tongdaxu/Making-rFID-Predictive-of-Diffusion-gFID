#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute frozen iFID-R v1 directly from saved pool_3 features.

No image NPZ is needed here.

For each VAE:
    iFID_GD = FID(reference_pool3, sample_pool3)
    iPrec, iRec = Kynkaanniemi improved precision/recall through the
                  OpenAI guided-diffusion ManifoldEstimator (default k=3)
    iFID_R_v1 = iFID_GD / (1 - iRec)

This intentionally uses the same pool_3 feature arrays for both FID and recall.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf


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


def import_evaluator(guided_diffusion_dir: str):
    eval_dir = Path(guided_diffusion_dir).resolve() / "evaluations"
    evaluator_file = eval_dir / "evaluator.py"
    if not evaluator_file.exists():
        raise FileNotFoundError(evaluator_file)
    sys.path.insert(0, str(eval_dir))
    import evaluator
    return evaluator


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--guided-diffusion-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--only", default=None)
    p.add_argument(
        "--skip-done",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p.parse_args()


def main():
    args = parse_args()
    feature_dir = Path(args.feature_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gd_eval = import_evaluator(args.guided_diffusion_dir)

    ref_path = feature_dir / "reference_imagenet_val_50k_pool3.npy"
    ref = np.load(ref_path, mmap_mode="r")
    if ref.shape != (50000, 2048) or ref.dtype != np.float32:
        raise ValueError(f"Bad reference feature file: {ref_path}, {ref.shape}, {ref.dtype}")

    # FID reference statistics can be computed once.
    ref_np = np.asarray(ref)
    ref_stats = gd_eval.FIDStatistics(
        np.mean(ref_np, axis=0),
        np.cov(ref_np, rowvar=False),
    )

    tf.disable_v2_behavior()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    rows = []

    with tf.Session(config=config) as sess:
        # This does NOT instantiate Evaluator, so it does not need the Inception PB.
        manifold = gd_eval.ManifoldEstimator(sess)

        # Cache reference radii once; this is expensive and shared by all models.
        print("[REFERENCE] computing k=3 manifold radii once...", flush=True)
        ref_radii = manifold.manifold_radii(ref_np)
        print("[REFERENCE] radii done", flush=True)

        for i, (model, tag) in enumerate(MODELS, 1):
            if args.only is not None and args.only not in {model, tag}:
                continue

            sample_path = feature_dir / f"{tag}_pool3.npy"
            out_json = out_dir / f"{tag}.json"

            if args.skip_done and out_json.exists():
                try:
                    old = json.loads(out_json.read_text())
                    if "iFID_R_v1" in old.get("metrics", {}):
                        print(f"[SKIP DONE] {model}")
                        rows.append(old)
                        continue
                except Exception:
                    pass

            sample = np.load(sample_path, mmap_mode="r")
            if sample.shape != (50000, 2048) or sample.dtype != np.float32:
                raise ValueError(f"Bad sample feature file: {sample_path}, {sample.shape}, {sample.dtype}")
            sample_np = np.asarray(sample)

            print(f"\n[{i}/25] {model}", flush=True)

            sample_stats = gd_eval.FIDStatistics(
                np.mean(sample_np, axis=0),
                np.cov(sample_np, rowvar=False),
            )
            ifid_gd = float(sample_stats.frechet_distance(ref_stats))

            print("  sample manifold radii...", flush=True)
            sample_radii = manifold.manifold_radii(sample_np)

            print("  precision / recall...", flush=True)
            pr = manifold.evaluate_pr(
                ref_np, ref_radii,
                sample_np, sample_radii,
            )
            iprec = float(pr[0][0])
            irec = float(pr[1][0])

            denom = 1.0 - irec
            if denom <= 0:
                raise ValueError(f"{model}: invalid iRec={irec}")
            ifid_r = ifid_gd / denom

            result = {
                "schema_version": 1,
                "metric_name": "iFID-R_v1",
                "formula": "iFID_GD / (1 - iRec)",
                "model": model,
                "tag": tag,
                "protocol": {
                    "reference_count": 50000,
                    "sample_count": 50000,
                    "feature": "OpenAI guided-diffusion Inception-v3 pool_3",
                    "feature_dim": 2048,
                    "precision_recall_nhood_size": 3,
                    "matched_feature_sets": True,
                    "interpolation": "original iFID SLERP alpha=0.5 top10 200000x50000",
                },
                "inputs": {
                    "reference_pool3": str(ref_path),
                    "sample_pool3": str(sample_path),
                },
                "metrics": {
                    "iFID_GD": ifid_gd,
                    "iPrec": iprec,
                    "iRec": irec,
                    "one_minus_iRec": denom,
                    "iFID_R_v1": ifid_r,
                },
            }
            atomic_json(out_json, result)
            rows.append(result)

            print(json.dumps(result["metrics"], indent=2), flush=True)
            del sample, sample_np, sample_stats, sample_radii

    # Re-read all completed JSONs in canonical order into one CSV.
    csv_rows = []
    for model, tag in MODELS:
        p = out_dir / f"{tag}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        m = r["metrics"]
        csv_rows.append({
            "VAE Name": model,
            "tag": tag,
            "iFID_GD": m["iFID_GD"],
            "iPrec": m["iPrec"],
            "iRec": m["iRec"],
            "1-iRec": m["one_minus_iRec"],
            "iFID_R_v1": m["iFID_R_v1"],
        })

    csv_path = out_dir / "ifid_r_v1_scores.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["VAE Name", "tag", "iFID_GD", "iPrec", "iRec", "1-iRec", "iFID_R_v1"],
        )
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n[CSV] {csv_path}")
    print(f"[N] {len(csv_rows)}")


if __name__ == "__main__":
    main()
