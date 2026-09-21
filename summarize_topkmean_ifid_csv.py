#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import re

LOG_DIR = Path("/share/xutongda/REPA-E/exps_interpolate/logs_ifid_topkmean")
OUT_CSV = LOG_DIR / "topkmean_ifid_metrics_ordered.csv"
OUT_FULL_CSV = LOG_DIR / "topkmean_ifid_metrics_ordered_full.csv"

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

EXP_TMPL = "ifid-iis-topkmean-{tag}-slerp-a05-top10-200000x50000.out"

METRIC_PATTERNS = {
    "iFID": re.compile(r"^ifid\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", re.M),
    "rFID": re.compile(r"^rfid\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", re.M),
    "PSNR": re.compile(r"^psnr\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", re.M),
    "SSIM": re.compile(r"^ssim\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", re.M),
    "LPIPS": re.compile(r"^lpips\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", re.M),
}

def last_match(pattern, text):
    matches = pattern.findall(text)
    return matches[-1] if matches else ""

rows_simple = []
rows_full = []
missing = []

for yaml_name, display_name, tag, group in MODELS:
    log_path = LOG_DIR / EXP_TMPL.format(tag=tag)

    metrics = {
        "iFID": "",
        "PSNR": "",
        "LPIPS": "",
        "SSIM": "",
        "rFID": "",
    }

    status = "OK"

    if not log_path.exists():
        status = "MISSING_LOG"
        missing.append((display_name, tag, status, str(log_path)))
    else:
        text = log_path.read_text(errors="ignore")

        for key, pat in METRIC_PATTERNS.items():
            metrics[key] = last_match(pat, text)

        if not all(metrics.values()):
            status = "MISSING_METRIC"
            missing.append((display_name, tag, status, str(log_path)))

    rows_simple.append({
        "Model": display_name,
        "iFID": metrics["iFID"],
        "PSNR": metrics["PSNR"],
        "LPIPS": metrics["LPIPS"],
        "SSIM": metrics["SSIM"],
        "rFID": metrics["rFID"],
    })

    rows_full.append({
        "YAML": yaml_name,
        "Model": display_name,
        "Tag": tag,
        "Group": group,
        "iFID": metrics["iFID"],
        "PSNR": metrics["PSNR"],
        "LPIPS": metrics["LPIPS"],
        "SSIM": metrics["SSIM"],
        "rFID": metrics["rFID"],
        "Status": status,
        "Log": str(log_path),
    })

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Model", "iFID", "PSNR", "LPIPS", "SSIM", "rFID"],
    )
    writer.writeheader()
    writer.writerows(rows_simple)

with OUT_FULL_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["YAML", "Model", "Tag", "Group", "iFID", "PSNR", "LPIPS", "SSIM", "rFID", "Status", "Log"],
    )
    writer.writeheader()
    writer.writerows(rows_full)

print("=" * 100)
print(f"LOG_DIR:      {LOG_DIR}")
print(f"CSV:          {OUT_CSV}")
print(f"FULL CSV:     {OUT_FULL_CSV}")
print(f"DONE:         {len(MODELS) - len(missing)}/{len(MODELS)}")
print("=" * 100)

if missing:
    print("\n[MISSING / INCOMPLETE]")
    for name, tag, status, path in missing:
        print(f"{name:24s} {tag:18s} {status:16s} {path}")

print("\n[PREVIEW]")
print("Model,iFID,PSNR,LPIPS,SSIM,rFID")
for r in rows_simple:
    print(f"{r['Model']},{r['iFID']},{r['PSNR']},{r['LPIPS']},{r['SSIM']},{r['rFID']}")
