#!/usr/bin/env python3
"""Collect 25-VAE Local-Score rFID metrics in fixed benchmark order."""

import argparse
import csv
import json
from pathlib import Path

MODELS = [
    "SDVAE", "SD3VAE", "FLUXVAE", "DCAE", "SOFTVQ", "VAVAE", "VAVAE64",
    "EQVAE", "MAETOK", "INVAE", "REPAEVAE", "DETOK", "QWVAE", "RAE",
    "SVG", "FLUX2VAE", "SVGT2I_P_STAGE1_256", "DMVAE", "UAE", "VTPS",
    "VTPB", "VTPL", "ScaleRAE_SIGLIP2_NONORM", "ScaleRAE_WEBSSL",
    "PAE_DINOv2L_d32",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out_csv) if args.out_csv else root / "local_score_rfid_25.csv"

    rows = []
    for model in MODELS:
        p = root / model / "metrics.json"
        if not p.exists():
            # Runner uses exact config stem, so this is the expected location.
            row = {"model": model, "status": "missing"}
            rows.append(row)
            if args.strict:
                raise FileNotFoundError(p)
            continue

        d = json.loads(p.read_text())
        m = d.get("metrics", {})
        rows.append(
            {
                "model": model,
                "status": "ok",
                "rfid": m.get("rfid", ""),
                "ls_rfid_t0.2_p1": m.get("ls_rfid_t0.2_p1", ""),
                "ls_rfid_t0.4_p3": m.get("ls_rfid_t0.4_p3", ""),
                "candidate_topk": d.get("candidate_topk", ""),
                "ref_size": d.get("ref_size", ""),
                "val_size": d.get("val_size", ""),
                "tshift": d.get("tshift", ""),
                "native_latent_shape": json.dumps(d.get("native_latent_shape")),
                "token_grid": json.dumps(d.get("token_grid")),
                "metrics_json": str(p),
            }
        )

    fields = [
        "model", "status", "rfid",
        "ls_rfid_t0.2_p1", "ls_rfid_t0.4_p3",
        "candidate_topk", "ref_size", "val_size", "tshift",
        "native_latent_shape", "token_grid", "metrics_json",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(out)
    for r in rows:
        print(
            f"{r['model']:28s} "
            f"t.2={str(r.get('ls_rfid_t0.2_p1','')):>12s} "
            f"t.4={str(r.get('ls_rfid_t0.4_p3','')):>12s} "
            f"{r['status']}"
        )


if __name__ == "__main__":
    main()
