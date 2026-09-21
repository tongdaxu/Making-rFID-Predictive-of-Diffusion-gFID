#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

MODELS = [
    "SDVAE", "SD3VAE", "FLUXVAE", "DCAE", "SOFTVQ", "VAVAE", "VAVAE64",
    "EQVAE", "MAETOK", "INVAE", "REPAEVAE", "DETOK", "QWVAE", "RAE",
    "SVG", "FLUX2VAE", "SVGT2I_P_STAGE1_256", "DMVAE", "UAE", "VTPS",
    "VTPB", "VTPL", "ScaleRAE_SIGLIP2_NONORM", "ScaleRAE_WEBSSL",
    "PAE_DINOv2L_d32",
]
METRICS = ["rfid"] + [
    f"ls_rfid_t{t}_p{p}"
    for t in ("0.2", "0.4")
    for p in (1, 3, 5, 7)
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out_csv) if a.out_csv else root / "local_score_rfid_25_p_sweep.csv"
    rows = []
    for model in MODELS:
        p = root / model / "metrics.json"
        row = {"model": model, "status": "missing"}
        for m in METRICS:
            row[m] = ""
        if p.exists():
            d = json.loads(p.read_text())
            mm = d.get("metrics", {})
            row["status"] = "ok"
            for m in METRICS:
                row[m] = mm.get(m, "")
            row["candidate_topk"] = d.get("candidate_topk", "")
            row["ref_size"] = d.get("ref_size", "")
            row["val_size"] = d.get("val_size", "")
            row["tshift"] = d.get("tshift", "")
            row["metrics_json"] = str(p)
        elif a.strict:
            raise FileNotFoundError(p)
        rows.append(row)

    fields = ["model", "status"] + METRICS + [
        "candidate_topk", "ref_size", "val_size", "tshift", "metrics_json"
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(out)

if __name__ == "__main__":
    main()
