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
    ap.add_argument("--out", default="patch_ifid_p1357_scores.csv")
    ap.add_argument("--patch-sizes", type=int, nargs="+", default=[1, 3, 5, 7])
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    ps = args.patch_sizes

    fields = ["model"] + [f"patch_ifid_p{p}" for p in ps]
    rows = []
    missing = []

    for display, folder in MODELS:
        path = root / folder / "metrics.json"
        row = {"model": display}

        if not path.exists():
            for p in ps:
                row[f"patch_ifid_p{p}"] = "NA"
            missing.append(folder)
            rows.append(row)
            continue

        d = json.loads(path.read_text())
        metrics = d.get("metrics", {})
        for p in ps:
            row[f"patch_ifid_p{p}"] = metrics.get(f"patch_ifid_p{p}", "NA")
        rows.append(row)

    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(out.resolve())
    print()
    print("\t".join(fields))
    for r in rows:
        vals = [str(r[x]) for x in fields]
        print("\t".join(vals))

    if missing:
        print("\nMissing:", ", ".join(missing))


if __name__ == "__main__":
    main()
