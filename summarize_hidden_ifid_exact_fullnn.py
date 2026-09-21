#!/usr/bin/env python3
"""Summarize exact full-hidden NN hidden-iFID JSON results into an ordered CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


# Fixed output order requested by the user.
# model, accepted result-directory tags, accepted SiT experiment directories
ORDER = [
    ("SD-VAE", ["ifid-sdvae"], ["sit-xl-sdvae-400k"]),
    ("SD3-VAE", ["ifid-sd3vae"], ["sit-xl-sd3-shift-400k"]),
    ("FLUX-VAE", ["ifid-fluxvae"], ["sit-xl-flux-shift-400k"]),
    ("DC-AE", ["ifid-dcae"], ["sit-xl-dcae-400k"]),
    ("SOFT-VQ", ["ifid-softvq"], ["sit-XL-softvq-400k", "sit-xl-softvq-400k"]),
    ("VA-VAE", ["ifid-vavae"], ["sit-xl-vavae-400k"]),
    ("VA-VAE-64", ["ifid-vavae64"], ["sit-xl-vavae64-shift-400k"]),
    ("EQ-VAE", ["ifid-eqvae"], ["sit-XL-eqvae-400k", "sit-xl-eqvae-400k"]),
    ("MAE-TOK", ["ifid-maetok"], ["sit-XL-maetok-400k", "sit-xl-maetok-400k"]),
    ("IN-VAE", ["ifid-invae"], ["sit-xl-invae-400k"]),
    ("REPAE-SDVAE", ["ifid-repae-sdvae", "ifid-repae"], ["sit-xl-repae-400k"]),
    ("DE-TOK", ["ifid-detok"], ["sit-xl-detok-400k"]),
    ("QwenImg-VAE", ["ifid-qwenimgvae", "ifid-qw"], ["sit-xl-qw-shift-400k"]),
    ("RAE", ["ifid-rae"], ["sit-xl-rae-shift-400k"]),
    ("SVG", ["ifid-svg"], ["sit-xl-svg-400k"]),
    ("FLUX2-VAE", ["ifid-flux2", "ifid-flux2vae"], ["sit-xl-flux2vae-400k"]),
    (
        "SVG T2I",
        ["ifid-svg-t2i-p256", "ifid-svgt2i-p256"],
        ["sit-xl-svgt2iP-400k"],
    ),
    ("DM-VAE", ["ifid-dmvae"], ["sit-xl-dmvae-400k"]),
    ("UAE", ["ifid-uae"], ["sit-xl-uae-400k"]),
    ("VTP-S", ["ifid-vtps"], ["sit-xl-vtps-400k"]),
    ("VTP-B", ["ifid-vtpb"], ["sit-xl-vtpb-400k"]),
    ("VTP-L", ["ifid-vtpl"], ["sit-xl-vtpl-400k"]),
    (
        "RAE T2I SigLIP-2",
        ["ifid-scalerae-siglip2", "ifid-raet2i-siglip2", "ifid-raet2i_siglip2"],
        ["sit-xl-raet2i_siglip2-400k"],
    ),
    (
        "RAE T2I WebSSL",
        ["ifid-scalerae-webssl", "ifid-raet2i-webssl", "ifid-raet2i_webssl"],
        ["sit-xl-raet2i_webssl-400k"],
    ),
    (
        "PAE DINOv2",
        ["ifid-pae-dino", "ifid-pae-dinov2"],
        ["sit-xl-pae-dinov2-400k"],
    ),
]

SETTING_FIELDS = [
    "exp_path",
    "train_steps",
    "ckpt_key",
    "depth",
    "t_input_before_shift",
    "path_type",
    "alpha",
    "nn_metric_full_hidden",
    "retrieval_protocol",
    "coarse_representation",
    "coarse_topk",
    "exact_reference_chunk",
    "n_reference",
    "n_query",
    "tokens",
    "hidden_dim",
    "seed",
    "preload_reference_gpu",
    "query_batch_size",
]

METRIC_FIELDS = [
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

FIELDS = ["model", "result_tag", "result_file"] + SETTING_FIELDS + METRIC_FIELDS


def scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("hidden_ifid_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            settings = payload.get("settings", {})
            metrics = payload.get("metrics", {})
            if not isinstance(settings, dict) or not isinstance(metrics, dict):
                raise ValueError("settings and metrics must both be JSON objects")

            exp_path = str(settings.get("exp_path", ""))
            records.append(
                {
                    "path": path,
                    "tag": path.parent.name,
                    "exp_basename": Path(exp_path).name if exp_path else "",
                    "settings": settings,
                    "metrics": metrics,
                    "mtime": path.stat().st_mtime,
                }
            )
        except Exception as exc:
            print(f"[PARSE ERROR] {path}: {type(exc).__name__}: {exc}")
    return records


def choose_record(
    records: list[dict[str, Any]],
    accepted_tags: list[str],
    accepted_expdirs: list[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    matches = [
        record
        for record in records
        if record["tag"] in accepted_tags
        or record["exp_basename"] in accepted_expdirs
    ]
    if not matches:
        return None, []

    # If duplicates exist, prefer the largest run and then the newest JSON.
    matches.sort(
        key=lambda record: (
            int(record["settings"].get("n_reference", 0) or 0),
            int(record["settings"].get("n_query", 0) or 0),
            float(record["mtime"]),
        ),
        reverse=True,
    )
    return matches[0], matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/video_ssd/kongzishang/xutongda/git/IFID/"
            "exps_interpolate/logs_hidden_ifid_fullsketch_l2_50k50k"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/video_ssd/kongzishang/xutongda/git/IFID/"
            "exps_interpolate/logs_hidden_ifid_fullsketch_l2_50k50k.csv"
        ),
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Result directory does not exist: {root}")

    records = load_records(root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    duplicates: list[tuple[str, list[Path]]] = []
    incomplete: list[tuple[str, Path, list[str]]] = []

    for model, tags, expdirs in ORDER:
        record, matches = choose_record(records, tags, expdirs)
        if record is None:
            missing.append(f"{model}: tags={tags}, exp_dirs={expdirs}")
            continue

        if len(matches) > 1:
            duplicates.append((model, [m["path"] for m in matches]))

        settings = record["settings"]
        metrics = record["metrics"]
        missing_metrics = [field for field in METRIC_FIELDS if field not in metrics]
        if missing_metrics:
            incomplete.append((model, record["path"], missing_metrics))

        row: dict[str, Any] = {
            "model": model,
            "result_tag": record["tag"],
            "result_file": str(record["path"]),
        }
        for field in SETTING_FIELDS:
            row[field] = scalar(settings.get(field, ""))
        for field in METRIC_FIELDS:
            row[field] = scalar(metrics.get(field, ""))
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print(f"Result root: {root}")
    print(f"JSON found:  {len(records)}")
    print(f"CSV output:  {output}")
    print(f"Rows saved:  {len(rows)} / {len(ORDER)}")
    print("=" * 100)

    if missing:
        print("\n[MISSING RESULTS]")
        for item in missing:
            print(item)

    if duplicates:
        print("\n[DUPLICATE MATCHES: selected largest/newest]")
        for model, paths in duplicates:
            print(model)
            for path in paths:
                print(f"  {path}")

    if incomplete:
        print("\n[INCOMPLETE RESULTS]")
        for model, path, fields in incomplete:
            print(f"{model}: {path}")
            print(f"  missing metrics: {', '.join(fields)}")

    print("\n[CSV PREVIEW]")
    with output.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            print(line.rstrip())
            if index >= 5:
                print("...")
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
