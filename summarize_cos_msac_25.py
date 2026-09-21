#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

MODELS = [
    ("SD-VAE", "SDVAE"),
    ("SD3-VAE", "SD3VAE"),
    ("FLUX-VAE", "FLUXVAE"),
    ("DC-AE", "DCAE"),
    ("SOFT-VQ", "SOFTVQ"),
    ("VA-VAE", "VAVAE"),
    ("VA-VAE-64", "VAVAE64"),
    ("EQ-VAE", "EQVAE"),
    ("MAE-TOK", "MAETOK"),
    ("IN-VAE", "INVAE"),
    ("REPAE-SDVAE", "REPAEVAE"),
    ("DE-TOK", "DETOK"),
    ("QwenImg-VAE", "QWVAE"),
    ("RAE", "RAE"),
    ("SVG", "SVG"),
    ("FLUX2-VAE", "FLUX2VAE"),
    ("SVG T2I", "SVGT2I_P_STAGE1_256"),
    ("DM-VAE", "DMVAE"),
    ("UAE", "UAE"),
    ("VTP-S", "VTPS"),
    ("VTP-B", "VTPB"),
    ("VTP-L", "VTPL"),
    ("RAE T2I SigLIP-2", "ScaleRAE_SIGLIP2_NONORM"),
    ("RAE T2I WebSSL", "ScaleRAE_WEBSSL"),
    ("PAE DINOv2", "PAE_DINOv2L_d32"),
]

FIELDS = [
    "semantic_midpoint_cos",
    "semantic_midpoint_cos_dist",
    "semantic_midpoint_cos_meanpatch",
    "semantic_midpoint_cos_dist_meanpatch",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--out",
        default="cos_msac_25_scores.csv",
    )
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    missing = []

    for display, folder in MODELS:
        path = root / folder / "metrics.json"
        row = {"VAE Name": display}

        if not path.exists():
            missing.append(folder)
            for k in FIELDS:
                row[k] = "NA"
            row["vae_nn_cos_mean"] = "NA"
        else:
            data = json.loads(path.read_text())
            metrics = data["metrics"]
            for k in FIELDS:
                row[k] = metrics.get(k, "NA")
            row["vae_nn_cos_mean"] = data[
                "nearest_neighbor"
            ].get("mean_selected_cosine", "NA")

        rows.append(row)

    fieldnames = (
        ["VAE Name"]
        + FIELDS
        + ["vae_nn_cos_mean"]
    )

    out = Path(args.out)
    with out.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\t".join(fieldnames))
    for row in rows:
        print("\t".join(str(row[x]) for x in fieldnames))

    print("\nSaved:", out.resolve())
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
