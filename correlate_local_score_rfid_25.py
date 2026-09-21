#!/usr/bin/env python3
"""
Correlate 25-VAE LS-rFID with an existing gFID CSV.

Examples
--------
Auto-use all numeric gFID columns:
  python correlate_local_score_rfid_25.py \
    --ls-csv local_score_rfid_25.csv \
    --gfid-csv gfid_25.csv \
    --model-col model

Choose exact columns:
  python correlate_local_score_rfid_25.py \
    --ls-csv local_score_rfid_25.csv \
    --gfid-csv gfid_25.csv \
    --model-col VAE \
    --gfid-cols "SiT-B w/o CFG" "SiT-B w CFG" "SiT-XL w/o CFG" "SiT-XL w CFG"

If the gFID CSV has no model-name column but is already in the exact same
25-model order, use --assume-row-order.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


ORDER = [
    "SDVAE", "SD3VAE", "FLUXVAE", "DCAE", "SOFTVQ", "VAVAE", "VAVAE64",
    "EQVAE", "MAETOK", "INVAE", "REPAEVAE", "DETOK", "QWVAE", "RAE",
    "SVG", "FLUX2VAE", "SVGT2I_P_STAGE1_256", "DMVAE", "UAE", "VTPS",
    "VTPB", "VTPL", "ScaleRAE_SIGLIP2_NONORM", "ScaleRAE_WEBSSL",
    "PAE_DINOv2L_d32",
]


def canon(s):
    s = str(s).lower()
    s = s.replace("qwenimgvae", "qwvae")
    s = s.replace("qwenimagevae", "qwvae")
    s = s.replace("qwenvae", "qwvae")
    s = s.replace("repaesdvae", "repaevae")
    s = s.replace("repae-sdvae", "repaevae")
    s = s.replace("flux2-vae", "flux2vae")
    s = s.replace("sd-vae", "sdvae")
    s = s.replace("sd3-vae", "sd3vae")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def corr_triplet(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return len(x), np.nan, np.nan, np.nan
    return (
        len(x),
        pearsonr(x, y).statistic,
        spearmanr(x, y).statistic,
        kendalltau(x, y).statistic,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-csv", required=True)
    ap.add_argument("--gfid-csv", required=True)
    ap.add_argument("--model-col", default="model")
    ap.add_argument("--gfid-cols", nargs="*", default=None)
    ap.add_argument("--assume-row-order", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ls = pd.read_csv(args.ls_csv)
    gf = pd.read_csv(args.gfid_csv)

    ls = ls[ls["status"].fillna("") == "ok"].copy()

    if args.assume_row_order:
        if len(gf) != len(ORDER):
            raise ValueError(
                f"--assume-row-order requires exactly {len(ORDER)} gFID rows; got {len(gf)}"
            )
        gf = gf.copy()
        gf["_model"] = ORDER
    else:
        if args.model_col not in gf.columns:
            raise KeyError(
                f"model column {args.model_col!r} not found; columns={list(gf.columns)}"
            )
        gf = gf.copy()
        gf["_model"] = gf[args.model_col]

    ls["_key"] = ls["model"].map(canon)
    gf["_key"] = gf["_model"].map(canon)

    merged = ls.merge(gf, on="_key", how="inner", suffixes=("_ls", "_gfid"))
    if len(merged) == 0:
        raise RuntimeError("No model names matched between LS and gFID CSVs.")

    metric_cols = ["rfid", "ls_rfid_t0.2_p1", "ls_rfid_t0.4_p3"]

    if args.gfid_cols:
        missing = [c for c in args.gfid_cols if c not in merged.columns]
        if missing:
            raise KeyError(f"Missing requested gFID columns: {missing}")
        gcols = args.gfid_cols
    else:
        blocked = set(metric_cols + [
            "candidate_topk", "ref_size", "val_size", "tshift",
        ])
        gcols = []
        for c in gf.columns:
            if c in (args.model_col, "_model", "_key") or c in blocked:
                continue
            vals = pd.to_numeric(merged[c], errors="coerce")
            if vals.notna().sum() >= 3:
                gcols.append(c)

    rows = []
    for metric in metric_cols:
        x = pd.to_numeric(merged[metric], errors="coerce").to_numpy(float)
        for gc in gcols:
            y = pd.to_numeric(merged[gc], errors="coerce").to_numpy(float)
            n, pcc, srcc, krcc = corr_triplet(x, y)
            rows.append(
                {
                    "metric": metric,
                    "gfid_column": gc,
                    "n": n,
                    "PCC": pcc,
                    "SRCC": srcc,
                    "KRCC": krcc,
                }
            )

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ls_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "ls_rfid_gfid_merged.csv"
    corr_path = out_dir / "ls_rfid_gfid_correlations.csv"

    merged.to_csv(merged_path, index=False)
    pd.DataFrame(rows).to_csv(corr_path, index=False)

    print(f"matched models: {len(merged)}")
    print(f"merged: {merged_path}")
    print(f"correlations: {corr_path}")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
