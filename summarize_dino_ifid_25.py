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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="dino_ifid_scores.csv")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    missing = []

    for display, folder in MODELS:
        path = root / folder / "metrics.json"
        score = "NA"
        dino_pair_cos = "NA"
        dino_pair_rel_l2 = "NA"

        if not path.exists():
            missing.append(folder)
        else:
            d = json.loads(path.read_text())
            metrics = d.get("metrics", {})
            if len(metrics) == 1:
                score = next(iter(metrics.values()))
            else:
                # Robust fallback if extra metrics are added later.
                for k, v in metrics.items():
                    if k.startswith("dino_ifid_"):
                        score = v
                        break

            diag = d.get("diagnostics", {})
            dino_pair_cos = diag.get(
                "mean_vae_latent_cosine_of_dino_pairs", "NA"
            )
            dino_pair_rel_l2 = diag.get(
                "mean_vae_latent_relative_l2_of_dino_pairs", "NA"
            )

        rows.append(
            {
                "model": display,
                "dino_ifid": score,
                "vae_latent_cos_of_dino_pairs": dino_pair_cos,
                "vae_latent_rel_l2_of_dino_pairs": dino_pair_rel_l2,
            }
        )

    fields = list(rows[0].keys())

    with Path(args.out).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(row[x]) for x in fields))

    print(f"\nSaved: {Path(args.out).resolve()}")
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
