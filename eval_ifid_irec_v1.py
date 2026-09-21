#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict evaluator for recall-adjusted interpolated FID (iFID-R v1).

Definition
----------
Given ONE fixed set of interpolated images produced by the original iFID
protocol, extract OpenAI guided-diffusion Inception-v3 pool_3 features once.

Let:
    Phi_r = reference ImageNet features
    Phi_i = interpolated-sample features

Compute, in the SAME feature space and from the SAME Phi_i:
    iFID_GD = FID(Phi_r, Phi_i)
    iRec    = improved recall(Phi_r, Phi_i), k=3
    iPrec   = improved precision(Phi_r, Phi_i), k=3
    iIS     = Inception Score(Phi_i)

Frozen candidate:
    iFID_R_v1 = iFID_GD / (1 - iRec)

For the formal v1 experiment, use exactly 50,000 reference features and
50,000 interpolated samples. The script enforces this by default.

The precision/recall implementation is OpenAI guided-diffusion's evaluator.py,
whose ManifoldEstimator is adapted from Kynkaanniemi et al. (NeurIPS 2019),
"Improved Precision and Recall Metric for Assessing Generative Models".

Important:
- This is intentionally the OpenAI guided-diffusion / Inception-pool_3 variant.
- The original Kynkaanniemi et al. paper used VGG-16 fc2 as its main feature
  space, while also reporting that Inception-v3 behaves similarly.
- Do NOT mix an old iFID number from another FID implementation into the
  numerator of iFID-R v1. The numerator here is recomputed from the exact same
  OpenAI Inception activations used for iRec.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf


FORMULA_NAME = "iFID-R_v1"
FORMULA = "iFID_GD / (1 - iRec)"
EXPECTED_N = 50_000
EXPECTED_DIM = 2048


def import_evaluator(guided_diffusion_dir: str):
    eval_dir = Path(guided_diffusion_dir).resolve() / "evaluations"
    evaluator_file = eval_dir / "evaluator.py"
    if not evaluator_file.exists():
        raise FileNotFoundError(
            f"Cannot find guided-diffusion evaluator: {evaluator_file}"
        )

    sys.path.insert(0, str(eval_dir))
    import evaluator  # noqa: E402

    return evaluator, evaluator_file


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_reference_pool(path: Path) -> np.ndarray:
    """
    Accept either:
      - .npy containing [N, 2048] pool_3 activations
      - .npz with one unambiguous 2-D [N,2048] array, or key 'pool_3'
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r")
        return arr

    if path.suffix.lower() == ".npz":
        obj = np.load(path, mmap_mode="r")
        keys = list(obj.keys())

        if "pool_3" in keys:
            return obj["pool_3"]

        candidates = [
            k for k in keys
            if getattr(obj[k], "ndim", None) == 2
            and obj[k].shape[1] == EXPECTED_DIM
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Could not identify a unique [N,{EXPECTED_DIM}] reference "
                f"activation array in {path}. keys={keys}, candidates={candidates}"
            )
        return obj[candidates[0]]

    raise ValueError(
        f"--reference-pool must be .npy or .npz activations, got: {path}"
    )


def validate_pool(name: str, x: np.ndarray):
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D [N,D], got shape={x.shape}")
    if x.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"{name} feature dim must be {EXPECTED_DIM}, got shape={x.shape}"
        )
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"{name} must be floating point, got dtype={x.dtype}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Strict matched iFID/iRec evaluator for iFID-R v1."
    )
    p.add_argument("--guided-diffusion-dir", required=True)
    p.add_argument(
        "--reference-pool",
        required=True,
        help="Cached OpenAI evaluator pool_3 activations for the real reference set.",
    )
    p.add_argument(
        "--sample-npz",
        required=True,
        help="NPZ containing arr_0 = interpolated images, NHWC uint8 [N,H,W,3].",
    )
    p.add_argument("--out-json", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)

    p.add_argument("--tf-batch-size", type=int, default=64)
    p.add_argument("--softmax-batch-size", type=int, default=512)

    p.add_argument(
        "--expected-n",
        type=int,
        default=EXPECTED_N,
        help="Formal v1 uses 50,000 real features and 50,000 interpolated images.",
    )
    p.add_argument(
        "--allow-nonstandard-count",
        action="store_true",
        help="For smoke/debug only. Formal v1 should NOT use this.",
    )
    p.add_argument(
        "--legacy-ifid",
        type=float,
        default=None,
        help="Optional old iFID value for audit only; never used in iFID-R formula.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    reference_pool_path = Path(args.reference_pool).resolve()
    sample_npz = Path(args.sample_npz).resolve()
    out_json = Path(args.out_json).resolve()

    if not reference_pool_path.exists():
        raise FileNotFoundError(reference_pool_path)
    if not sample_npz.exists():
        raise FileNotFoundError(sample_npz)

    # Validate sample NPZ format before starting TensorFlow.
    sample_obj = np.load(sample_npz, mmap_mode="r")
    if "arr_0" not in sample_obj.files:
        raise KeyError(f"{sample_npz} has no arr_0; keys={sample_obj.files}")
    sample_images = sample_obj["arr_0"]
    if sample_images.ndim != 4 or sample_images.shape[-1] != 3:
        raise ValueError(
            f"sample arr_0 must be NHWC RGB [N,H,W,3], got {sample_images.shape}"
        )
    n_sample_images = int(sample_images.shape[0])
    sample_image_dtype = str(sample_images.dtype)
    del sample_obj, sample_images

    reference_pool = load_reference_pool(reference_pool_path)
    validate_pool("reference_pool", reference_pool)
    n_ref = int(reference_pool.shape[0])

    if not args.allow_nonstandard_count:
        if n_ref != args.expected_n:
            raise ValueError(
                f"Formal {FORMULA_NAME} requires reference_pool N={args.expected_n}, "
                f"but got N={n_ref}. If this is an old 10k OpenAI reference pool, "
                "do NOT use it for the formal matched experiment."
            )
        if n_sample_images != args.expected_n:
            raise ValueError(
                f"Formal {FORMULA_NAME} requires sample N={args.expected_n}, "
                f"but sample NPZ has N={n_sample_images}."
            )

    print("=" * 80)
    print(f"metric              : {FORMULA_NAME}")
    print(f"formula             : {FORMULA}")
    print(f"model/tag           : {args.model} / {args.tag}")
    print(f"reference_pool      : {reference_pool_path}")
    print(f"reference shape     : {reference_pool.shape}")
    print(f"sample_npz          : {sample_npz}")
    print(f"sample image count  : {n_sample_images}")
    print(f"sample image dtype  : {sample_image_dtype}")
    print("=" * 80)

    tf.disable_v2_behavior()
    guided_evaluator, evaluator_file = import_evaluator(
        args.guided_diffusion_dir
    )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        evaluator = guided_evaluator.Evaluator(
            session,
            batch_size=args.tf_batch_size,
            softmax_batch_size=args.softmax_batch_size,
        )

        print("[1/5] Warming up evaluator...", flush=True)
        evaluator.warmup()

        print("[2/5] Computing interpolated-sample activations...", flush=True)
        sample_acts = evaluator.read_activations(str(sample_npz))
        sample_pool = sample_acts[0]
        validate_pool("sample_pool", sample_pool)

        if sample_pool.shape[0] != n_sample_images:
            raise RuntimeError(
                f"Activation count mismatch: images={n_sample_images}, "
                f"sample_pool={sample_pool.shape[0]}"
            )

        print(
            f"[SAMPLE POOL] shape={sample_pool.shape}, dtype={sample_pool.dtype}",
            flush=True,
        )

        print("[3/5] Computing matched iFID in the SAME pool_3 space...", flush=True)
        # Critical methodological point:
        # BOTH FID and iRec use the same reference_pool and sample_pool.
        ref_stats = evaluator.compute_statistics(
            np.asarray(reference_pool)
        )
        sample_stats = evaluator.compute_statistics(sample_pool)
        ifid_gd = sample_stats.frechet_distance(ref_stats)

        print("[4/5] Computing iIS...", flush=True)
        iis = evaluator.compute_inception_score(sample_pool)

        print("[5/5] Computing iPrec / iRec (k-NN manifold)...", flush=True)
        iprec, irec = evaluator.compute_prec_recall(
            np.asarray(reference_pool),
            sample_pool,
        )

        nhood_sizes = tuple(
            int(x) for x in evaluator.manifold_estimator.nhood_sizes
        )

    denom = 1.0 - float(irec)
    if denom <= 0.0:
        raise ValueError(
            f"iRec={irec} gives non-positive denominator 1-iRec={denom}; "
            f"{FORMULA_NAME} is undefined."
        )
    ifid_r_v1 = float(ifid_gd) / denom

    print("[HASH] reference pool...", flush=True)
    reference_sha256 = sha256_file(reference_pool_path)
    print("[HASH] sample NPZ...", flush=True)
    sample_sha256 = sha256_file(sample_npz)

    results = {
        "schema_version": 1,
        "metric_name": FORMULA_NAME,
        "formula": FORMULA,
        "model": args.model,
        "tag": args.tag,

        "protocol": {
            "feature_extractor": "OpenAI guided-diffusion Inception-v3 pool_3",
            "feature_dim": EXPECTED_DIM,
            "precision_recall_method": (
                "Kynkaanniemi et al. NeurIPS 2019 manifold precision/recall, "
                "via OpenAI guided-diffusion evaluator.py"
            ),
            "nhood_sizes": list(nhood_sizes),
            "formal_expected_n": int(args.expected_n),
            "nonstandard_count_allowed": bool(args.allow_nonstandard_count),
            "matched_feature_sets": True,
            "note": (
                "iFID_GD, iPrec, and iRec are computed from the same "
                "reference_pool and the same interpolated sample_pool."
            ),
        },

        "inputs": {
            "sample_npz": str(sample_npz),
            "sample_npz_sha256": sample_sha256,
            "sample_count": int(sample_pool.shape[0]),
            "sample_image_dtype": sample_image_dtype,
            "reference_pool": str(reference_pool_path),
            "reference_pool_sha256": reference_sha256,
            "reference_count": int(n_ref),
            "guided_evaluator_file": str(evaluator_file.resolve()),
        },

        "metrics": {
            "iFID_GD": float(ifid_gd),
            "iIS": float(iis),
            "iPrec": float(iprec),
            "iRec": float(irec),
            "one_minus_iRec": float(denom),
            "iFID_R_v1": float(ifid_r_v1),
        },

        # Flat aliases for convenient CSV aggregation / backward compatibility.
        "ifid_gd": float(ifid_gd),
        "iIS": float(iis),
        "iPrec": float(iprec),
        "iRec": float(irec),
        "iFID_R_v1": float(ifid_r_v1),
        "inception_score": float(iis),
        "precision": float(iprec),
        "recall": float(irec),
    }

    if args.legacy_ifid is not None:
        results["audit_only"] = {
            "legacy_iFID": float(args.legacy_ifid),
            "legacy_minus_matched_GD_iFID": (
                float(args.legacy_ifid) - float(ifid_gd)
            ),
            "warning": (
                "legacy_iFID is audit-only and is NOT used in iFID_R_v1."
            ),
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = Path(str(out_json) + ".tmp")
    tmp_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp_json, out_json)

    print("\nRESULT")
    print(json.dumps(results["metrics"], indent=2))
    print(f"[OK] Saved: {out_json}")


if __name__ == "__main__":
    main()
