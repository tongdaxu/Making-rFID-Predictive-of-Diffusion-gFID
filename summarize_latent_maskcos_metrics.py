#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize IFID latent mask-cos metric outputs.

Reads <root>/<vae_name>/metrics.json written by eval_latent_maskcos_metrics.py
and writes:
  <root>/maskcos_summary.csv
  <root>/maskcos_summary.md

The main table contains the 8 requested metrics:
  align_0.1, entropy_0.1, align_0.4, entropy_0.4,
  align_0.6, entropy_0.6, align_0.9, entropy_0.9.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def fmt(v: Any, ndigits: int = 8) -> str:
    if v is None:
        return ""
    try:
        x = float(v)
    except Exception:
        return str(v)
    if math.isnan(x) or math.isinf(x):
        return ""
    return f"{x:.{ndigits}f}"


def load_metrics_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(errors="ignore"))
    except Exception as e:
        print(f"[WARN] failed to read {p}: {e}")
        return None


def model_name_from_metrics_path(metrics_path: Path, data: Dict[str, Any]) -> str:
    # Prefer output directory name; it is stable and human-readable.
    return metrics_path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root directory containing per-VAE metrics.json files.")
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--out-md", type=str, default=None)
    parser.add_argument("--t-values", nargs="+", type=str, default=["0.1", "0.4", "0.6", "0.9"])
    parser.add_argument("--digits", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    out_csv = Path(args.out_csv) if args.out_csv else root / "maskcos_summary.csv"
    out_md = Path(args.out_md) if args.out_md else root / "maskcos_summary.md"

    metrics_files = sorted(root.glob("*/metrics.json"))
    rows: List[Dict[str, Any]] = []

    for mp in metrics_files:
        data = load_metrics_json(mp)
        if data is None:
            continue
        m = data.get("metrics", {}) or {}
        model = model_name_from_metrics_path(mp, data)

        row: Dict[str, Any] = {
            "model": model,
            "vae_config": data.get("vae_config", ""),
            "num_images_used": data.get("num_images_used", ""),
            "batch_size": data.get("batch_size", ""),
            "resolution": data.get("resolution", ""),
            "mask_ratio": data.get("mask_ratio", ""),
            "normalize_latent": data.get("normalize_latent", ""),
            "latent_shape": json.dumps(data.get("latent_shape", None)),
            "spatial_shape": json.dumps(data.get("spatial_shape", None)),
            "metrics_json": str(mp),
        }
        for t in args.t_values:
            # keys in eval script are f"{float(t):g}"; convert to that canonical key.
            try:
                tk = f"{float(t):g}"
            except Exception:
                tk = t
            row[f"align_{tk}"] = m.get(f"align_{tk}")
            row[f"entropy_{tk}"] = m.get(f"entropy_{tk}")
            row[f"align_{tk}_std"] = m.get(f"align_{tk}_std")
            row[f"entropy_{tk}_std"] = m.get(f"entropy_{tk}_std")
        rows.append(row)

    rows.sort(key=lambda r: str(r["model"]))

    metric_fields: List[str] = []
    for t in args.t_values:
        try:
            tk = f"{float(t):g}"
        except Exception:
            tk = t
        metric_fields.extend([f"align_{tk}", f"entropy_{tk}"])

    meta_fields = [
        "model",
        "num_images_used",
        "batch_size",
        "resolution",
        "mask_ratio",
        "normalize_latent",
        "latent_shape",
        "spatial_shape",
    ]
    extra_fields = ["vae_config", "metrics_json"]
    fields = meta_fields + metric_fields + extra_fields

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for k in metric_fields:
                out[k] = fmt(out.get(k), args.digits)
            writer.writerow({k: out.get(k, "") for k in fields})

    with out_md.open("w") as f:
        md_fields = ["model"] + metric_fields
        f.write("| " + " | ".join(md_fields) + " |\n")
        f.write("| " + " | ".join(["---"] + ["---:"] * (len(md_fields) - 1)) + " |\n")
        for row in rows:
            vals = [str(row["model"])]
            vals.extend(fmt(row.get(k), 6) for k in metric_fields)
            f.write("| " + " | ".join(vals) + " |\n")

    missing = []
    for row in rows:
        for k in metric_fields:
            if row.get(k) is None:
                missing.append((row["model"], k))

    print("=" * 100)
    print(f"Root: {root}")
    print(f"Found metrics.json files: {len(metrics_files)}")
    print(f"Rows written: {len(rows)}")
    print(f"CSV: {out_csv}")
    print(f"Markdown: {out_md}")
    print(f"Missing metric cells: {len(missing)}")
    for model, key in missing[:50]:
        print(f"[MISSING] {model} {key}")
    if len(missing) > 50:
        print(f"... and {len(missing) - 50} more")
    print("=" * 100)

    print("\nPreview:")
    preview_fields = ["model"] + metric_fields
    widths = {"model": 38}
    for k in metric_fields:
        widths[k] = max(12, len(k))
    print(" ".join(f"{k:{widths[k]}}" for k in preview_fields))
    print("-" * (sum(widths[k] for k in preview_fields) + len(preview_fields) - 1))
    for row in rows[:80]:
        parts = [f"{str(row['model']):{widths['model']}}"]
        for k in metric_fields:
            parts.append(f"{fmt(row.get(k), 4):>{widths[k]}}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
