#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import csv

ROOT = Path("/share/xutongda/REPA-E/exps_interpolate")
RESULT_DIR = ROOT / "results_iis_from_intp_dist"
OUT_CSV = RESULT_DIR / "iis_from_intp_dist_ordered.csv"
OUT_FULL_CSV = RESULT_DIR / "iis_from_intp_dist_ordered_full.csv"

MODELS = [
    ("SDVAE.yaml", "SDVAE", "sdvae", "regular"),
    ("SD3VAE.yaml", "SD3VAE", "sd3vae", "regular"),
    ("FLUXVAE.yaml", "FLUXVAE", "fluxvae", "regular"),
    ("DCAE.yaml", "DCAE", "dcae", "regular"),
    ("SOFTVQ.yaml", "SOFTVQ", "softvq", "regular"),
    ("VAVAE.yaml", "VAVAE", "vavae", "regular"),
    ("VAVAE64.yaml", "VAVAE64", "vavae64", "regular"),
    ("EQVAE.yaml", "EQVAE", "eqvae", "regular"),
    ("MAETOK.yaml", "MAETOK", "maetok", "regular"),
    ("INVAE.yaml", "INVAE", "invae", "regular"),
    ("REPAEVAE.yaml", "REPAEVAE", "repae-sdvae", "regular"),
    ("DETOK.yaml", "DETOK", "detok", "regular"),
    ("QWVAE.yaml", "QWVAE", "qwenimgvae", "regular"),
    ("RAE.yaml", "RAE", "rae", "special"),
    ("SVG.yaml", "SVG", "svg", "special"),
    ("FLUX2VAE.yaml", "FLUX2VAE", "flux2", "regular"),
    ("SVGT2I_P_STAGE1_256.yaml", "SVGT2I_P_STAGE1_256", "svg-t2i-p256", "special"),
    ("DMVAE.yaml", "DMVAE", "dmvae", "regular"),
    ("VTPS.yaml", "VTPS", "vtps", "regular"),
    ("VTPB.yaml", "VTPB", "vtpb", "regular"),
    ("VTPL.yaml", "VTPL", "vtpl", "regular"),
    ("PAE_DINOv2L_d32.yaml", "PAE_DINOv2L_d32", "pae-dino", "regular"),
]

EXP_TMPL = "ifid-iis-{tag}-slerp-a05-top10-200000x50000"

rows = []
full_rows = []
missing = []

for yaml_name, model, tag, group in MODELS:
    exp_name = EXP_TMPL.format(tag=tag)
    path = RESULT_DIR / f"{exp_name}.json"

    if not path.exists():
        missing.append((model, tag, path))
        rows.append({
            "Model": model,
            "iIS": "",
        })
        full_rows.append({
            "YAML": yaml_name,
            "Model": model,
            "Tag": tag,
            "Group": group,
            "iIS": "",
            "sample_npz": "",
            "json": str(path),
            "status": "MISSING",
        })
        continue

    data = json.loads(path.read_text())

    rows.append({
        "Model": model,
        "iIS": data.get("inception_score", ""),
    })

    full_rows.append({
        "YAML": yaml_name,
        "Model": model,
        "Tag": tag,
        "Group": group,
        "iIS": data.get("inception_score", ""),
        "sample_npz": data.get("sample_npz", ""),
        "json": str(path),
        "status": "OK",
    })

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Model", "iIS"])
    writer.writeheader()
    writer.writerows(rows)

with OUT_FULL_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["YAML", "Model", "Tag", "Group", "iIS", "sample_npz", "json", "status"],
    )
    writer.writeheader()
    writer.writerows(full_rows)

print("=" * 100)
print(f"RESULT_DIR: {RESULT_DIR}")
print(f"CSV:        {OUT_CSV}")
print(f"FULL CSV:   {OUT_FULL_CSV}")
print(f"DONE:       {len(MODELS) - len(missing)}/{len(MODELS)}")
print("=" * 100)

if missing:
    print("\n[MISSING]")
    for model, tag, path in missing:
        print(f"{model:24s} {tag:18s} {path}")

print("\n[PREVIEW]")
print("Model,iIS")
for r in rows:
    print(f"{r['Model']},{r['iIS']}")
