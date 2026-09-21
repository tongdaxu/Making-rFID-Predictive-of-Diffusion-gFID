#!/usr/bin/env python3
"""Summarize hidden-iFID JSON result files into one ordered CSV.

Expected layout:
  <results-root>/<tag>/hidden_ifid_metrics.json

The script recursively discovers all files named hidden_ifid_metrics.json,
keeps the project's fixed model order, preserves all known settings/metrics,
and appends any extra JSON fields as additional columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple


MODEL_ORDER: List[Tuple[str, str]] = [
    ("ifid-sdvae", "SD-VAE"),
    ("ifid-fluxvae", "FLUX-VAE"),
    ("ifid-qwenimgvae", "QwenImg-VAE"),
    ("ifid-sd3vae", "SD3-VAE"),
    ("ifid-softvq", "SOFT-VQ"),
    ("ifid-maetok", "MAE-TOK"),
    ("ifid-detok", "DE-TOK"),
    ("ifid-dmvae", "DM-VAE"),
    ("ifid-repae-sdvae", "REPAE-SDVAE"),
    ("ifid-vtpb", "VTPB"),
    ("ifid-vtpl", "VTPL"),
    ("ifid-vtps", "VTPS"),
    ("ifid-eqvae", "EQ-VAE"),
    ("ifid-invae", "IN-VAE"),
    ("ifid-vavae", "VA-VAE"),
    ("ifid-vavae64", "VA-VAE-64"),
    ("ifid-rae", "RAE"),
    ("ifid-flux2", "FLUX2-VAE"),
    ("ifid-svg", "SVG"),
    ("ifid-uae", "UAE"),
    ("ifid-scalerae-siglip2", "ScaleRAE-SigLIP2"),
    ("ifid-scalerae-webssl", "ScaleRAE-WEBSSL"),
    ("ifid-svg-t2i-p256", "SVG-T2I-P-256"),
    ("ifid-dcae", "DCAE"),
    ("ifid-pae-dino", "PAE-DINO"),
]

TAG_TO_MODEL = dict(MODEL_ORDER)
ORDER_INDEX = {tag: i for i, (tag, _model) in enumerate(MODEL_ORDER)}

SETTING_KEYS = [
    "exp_path",
    "train_steps",
    "ckpt_key",
    "depth",
    "t_input_before_shift",
    "path_type",
    "alpha",
    "nn_metric_full_hidden",
    "coarse_topk",
    "coarse_backend",
    "n_reference",
    "n_query",
    "tokens",
    "hidden_dim",
    "seed",
]

METRIC_KEYS = [
    "main_hidden_fid_raw",
    "main_hidden_fid_token_ln",
    "sanity_alpha0_fid_raw",
    "sanity_alpha0_fid_token_ln",
    "sanity_alpha1_fid_raw",
    "sanity_alpha1_fid_token_ln",
    "random_neighbor_alpha05_fid_raw",
    "random_neighbor_alpha05_fid_token_ln",
    "nn_same_class_rate",
    "query_hidden_mean_full_l2",
    "query_hidden_mean_pooled_l2",
    "query_pooled_cov_trace_raw",
    "query_pooled_cov_trace_token_ln",
    "split_half_fid_raw",
    "split_half_fid_token_ln",
]

DERIVED_KEYS = [
    "main_over_random_raw",
    "main_minus_split_half_raw",
    "main_over_random_token_ln",
    "main_minus_split_half_token_ln",
    "alpha0_abs_raw",
    "alpha0_abs_token_ln",
]

BASE_KEYS = ["Model", "Tag", "Result File"]


def scalar_or_json(value: Any) -> Any:
    """Keep scalar CSV values simple; serialize nested structures deterministically."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_ratio(a: Any, b: Any) -> float | str:
    x = finite_number(a)
    y = finite_number(b)
    if x is None or y is None or y == 0.0:
        return ""
    return x / y


def safe_difference(a: Any, b: Any) -> float | str:
    x = finite_number(a)
    y = finite_number(b)
    if x is None or y is None:
        return ""
    return x - y


def safe_abs(a: Any) -> float | str:
    x = finite_number(a)
    return "" if x is None else abs(x)


def infer_tag(path: Path, root: Path) -> str:
    # Standard layout uses the parent folder as the tag.
    if path.parent != root:
        return path.parent.name
    return path.stem


def flatten_result(path: Path, root: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, Mapping):
        raise ValueError("top-level JSON value is not an object")

    settings = payload.get("settings", {})
    metrics = payload.get("metrics", {})
    if not isinstance(settings, Mapping):
        raise ValueError("'settings' is not an object")
    if not isinstance(metrics, Mapping):
        raise ValueError("'metrics' is not an object")

    tag = infer_tag(path, root)
    row: Dict[str, Any] = {
        "Model": TAG_TO_MODEL.get(tag, tag),
        "Tag": tag,
        "Result File": str(path),
    }

    for key, value in settings.items():
        row[key] = scalar_or_json(value)
    for key, value in metrics.items():
        row[key] = scalar_or_json(value)

    row["main_over_random_raw"] = safe_ratio(
        metrics.get("main_hidden_fid_raw"),
        metrics.get("random_neighbor_alpha05_fid_raw"),
    )
    row["main_minus_split_half_raw"] = safe_difference(
        metrics.get("main_hidden_fid_raw"),
        metrics.get("split_half_fid_raw"),
    )
    row["main_over_random_token_ln"] = safe_ratio(
        metrics.get("main_hidden_fid_token_ln"),
        metrics.get("random_neighbor_alpha05_fid_token_ln"),
    )
    row["main_minus_split_half_token_ln"] = safe_difference(
        metrics.get("main_hidden_fid_token_ln"),
        metrics.get("split_half_fid_token_ln"),
    )
    row["alpha0_abs_raw"] = safe_abs(metrics.get("sanity_alpha0_fid_raw"))
    row["alpha0_abs_token_ln"] = safe_abs(metrics.get("sanity_alpha0_fid_token_ln"))
    return row


def ordered_fieldnames(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    seen = set(BASE_KEYS + SETTING_KEYS + METRIC_KEYS + DERIVED_KEYS)
    extras = sorted({key for row in rows for key in row.keys() if key not in seen})
    return BASE_KEYS + SETTING_KEYS + METRIC_KEYS + DERIVED_KEYS + extras


def row_sort_key(row: Mapping[str, Any]) -> Tuple[int, str]:
    tag = str(row.get("Tag", ""))
    return (ORDER_INDEX.get(tag, len(ORDER_INDEX)), tag)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/video_ssd/kongzishang/xutongda/git/IFID/"
            "exps_interpolate/hidden_ifid_results_100k50k"
        ),
        help="Directory containing per-model result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <root>/hidden_ifid_summary.csv",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "hidden_ifid_summary.csv"
    )

    if not root.is_dir():
        print(f"ERROR: results root does not exist: {root}", file=sys.stderr)
        return 2

    paths = sorted(root.rglob("hidden_ifid_metrics.json"))
    if not paths:
        print(f"ERROR: no hidden_ifid_metrics.json found under: {root}", file=sys.stderr)
        return 3

    rows: List[Dict[str, Any]] = []
    errors: List[Tuple[Path, str]] = []
    for path in paths:
        try:
            rows.append(flatten_result(path, root))
        except Exception as exc:  # keep summarizing other valid models
            errors.append((path, f"{type(exc).__name__}: {exc}"))

    if not rows:
        print("ERROR: every discovered result file failed to parse", file=sys.stderr)
        for path, message in errors:
            print(f"  {path}: {message}", file=sys.stderr)
        return 4

    rows.sort(key=row_sort_key)
    fields = ordered_fieldnames(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found result files : {len(paths)}")
    print(f"Written rows       : {len(rows)}")
    print(f"Output CSV         : {output}")
    if errors:
        print(f"Parse failures     : {len(errors)}", file=sys.stderr)
        for path, message in errors:
            print(f"  {path}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
