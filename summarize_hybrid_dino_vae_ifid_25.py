#!/usr/bin/env python3
"""
Summarize the six hybrid/global iFID settings in the fixed 25-VAE order.
"""

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

COLS = [
    ("K1", "hybrid_ifid_dinoK1_vae_cos_top1_slerp_a0.5"),
    ("K10", "hybrid_ifid_dinoK10_vae_cos_top1_slerp_a0.5"),
    ("K50", "hybrid_ifid_dinoK50_vae_cos_top1_slerp_a0.5"),
    ("K200", "hybrid_ifid_dinoK200_vae_cos_top1_slerp_a0.5"),
    ("K1000", "hybrid_ifid_dinoK1000_vae_cos_top1_slerp_a0.5"),
    ("VAE_global_hard_top1_50k", "global_hard_ifid_vae_cos_top1_50k_slerp_a0.5"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--out",
        default="hybrid_dino_vae_ifid_6cols.csv",
    )
    ap.add_argument(
        "--diagnostics-out",
        default="hybrid_dino_vae_ifid_hit_rates.csv",
    )
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    diag_rows = []
    missing = []

    for display, folder in MODELS:
        path = root / folder / "metrics.json"
        row = {"model": display}
        drow = {"model": display}

        if not path.exists():
            missing.append(folder)
            for label, _key in COLS:
                row[label] = "NA"
            for k in (1, 10, 50, 200, 1000):
                drow[f"globalNN_in_DINO_top{k}"] = "NA"
        else:
            data = json.loads(path.read_text())
            metrics = data.get("metrics", {})
            diag = data.get("diagnostics", {})

            for label, key in COLS:
                row[label] = metrics.get(key, "NA")

            for k in (1, 10, 50, 200, 1000):
                drow[f"globalNN_in_DINO_top{k}"] = diag.get(
                    f"global_hard_nn_in_dino_top{k}",
                    "NA",
                )

        rows.append(row)
        diag_rows.append(drow)

    fields = ["model"] + [x[0] for x in COLS]
    with Path(args.out).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    dfields = ["model"] + [
        f"globalNN_in_DINO_top{k}"
        for k in (1, 10, 50, 200, 1000)
    ]
    with Path(args.diagnostics_out).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=dfields)
        w.writeheader()
        w.writerows(diag_rows)

    print("\t".join(fields))
    for row in rows:
        print(
            "\t".join(str(row[x]) for x in fields)
        )

    print(f"\nSaved: {Path(args.out).resolve()}")
    print(
        "Hit-rate diagnostics:",
        Path(args.diagnostics_out).resolve(),
    )

    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
