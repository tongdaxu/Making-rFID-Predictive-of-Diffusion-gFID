#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path


# 固定的 25 模型顺序：
# YAML 文件名、展示名称、实验 tag
MODELS = [
    ("SDVAE.yaml", "SDVAE", "sdvae"),
    ("SD3VAE.yaml", "SD3VAE", "sd3vae"),
    ("FLUXVAE.yaml", "FLUXVAE", "fluxvae"),
    ("DCAE.yaml", "DCAE", "dcae"),
    ("SOFTVQ.yaml", "SOFTVQ", "softvq"),
    ("VAVAE.yaml", "VAVAE", "vavae"),
    ("VAVAE64.yaml", "VAVAE64", "vavae64"),
    ("EQVAE.yaml", "EQVAE", "eqvae"),
    ("MAETOK.yaml", "MAETOK", "maetok"),
    ("INVAE.yaml", "INVAE", "invae"),
    ("REPAEVAE.yaml", "REPAEVAE", "repae-sdvae"),
    ("DETOK.yaml", "DETOK", "detok"),
    ("QWVAE.yaml", "QWVAE", "qwenimgvae"),
    ("RAE.yaml", "RAE", "rae"),
    ("SVG.yaml", "SVG", "svg"),
    ("FLUX2VAE.yaml", "FLUX2VAE", "flux2"),
    (
        "SVGT2I_P_STAGE1_256.yaml",
        "SVGT2I_P_STAGE1_256",
        "svg-t2i-p256",
    ),
    ("DMVAE.yaml", "DMVAE", "dmvae"),
    ("UAE.yaml", "UAE", "uae"),
    ("VTPS.yaml", "VTPS", "vtps"),
    ("VTPB.yaml", "VTPB", "vtpb"),
    ("VTPL.yaml", "VTPL", "vtpl"),
    (
        "ScaleRAE_SIGLIP2_NONORM.yaml",
        "ScaleRAE-SigLIP2",
        "scalerae-siglip2",
    ),
    (
        "ScaleRAE_WEBSSL.yaml",
        "ScaleRAE-WEBSSL",
        "scalerae-webssl",
    ),
    (
        "PAE_DINOv2L_d32.yaml",
        "PAE_DINOv2L_d32",
        "pae-dino",
    ),
]


def read_metrics(json_path: Path):
    if not json_path.exists():
        return None, "MISSING_JSON"

    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        return None, f"INVALID_JSON: {exc}"

    # 同时兼容新旧字段名。
    iis = data.get("iIS", data.get("inception_score"))
    iprec = data.get("iPrec", data.get("precision"))
    irec = data.get("iRec", data.get("recall"))

    metrics = {
        "iIS": iis,
        "iPrec": iprec,
        "iRec": irec,
    }

    missing_keys = [
        key
        for key, value in metrics.items()
        if value is None
    ]

    if missing_keys:
        return metrics, "MISSING_" + "_".join(missing_keys)

    return metrics, "OK"


def format_number(value):
    if value is None:
        return ""

    try:
        return f"{float(value):.8f}"
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result-dir",
        default=(
            "/share/xutongda/REPA-E/exps_interpolate/"
            "results_iis_iprec_irec_from_intp_dist"
        ),
    )
    parser.add_argument(
        "--output-name",
        default="interpolate_iis_iprec_irec_25models_ordered.csv",
    )
    parser.add_argument(
        "--full-output-name",
        default="interpolate_iis_iprec_irec_25models_ordered_full.csv",
    )

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_csv = result_dir / args.output_name
    full_output_csv = result_dir / args.full_output_name

    result_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    full_rows = []
    problems = []

    for order, (yaml_name, model, tag) in enumerate(MODELS, start=1):
        exp_name = (
            f"ifid-iis-{tag}-"
            f"slerp-a05-top10-200000x50000"
        )

        json_path = result_dir / f"{exp_name}.json"

        metrics, status = read_metrics(json_path)

        if metrics is None:
            iis = None
            iprec = None
            irec = None
        else:
            iis = metrics["iIS"]
            iprec = metrics["iPrec"]
            irec = metrics["iRec"]

        rows.append(
            {
                "Model": model,
                "iIS": format_number(iis),
                "iPrec": format_number(iprec),
                "iRec": format_number(irec),
            }
        )

        full_rows.append(
            {
                "Order": order,
                "YAML": yaml_name,
                "Model": model,
                "Tag": tag,
                "Experiment": exp_name,
                "iIS": format_number(iis),
                "iPrec": format_number(iprec),
                "iRec": format_number(irec),
                "Status": status,
                "JSON": str(json_path),
            }
        )

        if status != "OK":
            problems.append(
                {
                    "model": model,
                    "tag": tag,
                    "status": status,
                    "path": json_path,
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Model",
                "iIS",
                "iPrec",
                "iRec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with full_output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Order",
                "YAML",
                "Model",
                "Tag",
                "Experiment",
                "iIS",
                "iPrec",
                "iRec",
                "Status",
                "JSON",
            ],
        )
        writer.writeheader()
        writer.writerows(full_rows)

    complete_count = len(MODELS) - len(problems)

    print("=" * 120)
    print(f"Result directory: {result_dir}")
    print(f"Ordered CSV:      {output_csv}")
    print(f"Full CSV:         {full_output_csv}")
    print(f"Complete:         {complete_count}/{len(MODELS)}")
    print(f"Incomplete:       {len(problems)}")
    print("=" * 120)

    print("\nORDERED RESULTS")
    print(
        f"{'No.':<4} "
        f"{'Model':<28} "
        f"{'iIS':>14} "
        f"{'iPrec':>14} "
        f"{'iRec':>14} "
        f"{'Status':<20}"
    )
    print("-" * 100)

    for row in full_rows:
        print(
            f"{row['Order']:<4} "
            f"{row['Model']:<28} "
            f"{row['iIS']:>14} "
            f"{row['iPrec']:>14} "
            f"{row['iRec']:>14} "
            f"{row['Status']:<20}"
        )

    if problems:
        print("\nMISSING / INCOMPLETE RESULTS")
        print("-" * 100)

        for problem in problems:
            print(
                f"{problem['model']:<28} "
                f"{problem['status']:<30} "
                f"{problem['path']}"
            )

        raise SystemExit(1)

    print("\n[OK] All 25 models contain iIS, iPrec and iRec.")


if __name__ == "__main__":
    main()
