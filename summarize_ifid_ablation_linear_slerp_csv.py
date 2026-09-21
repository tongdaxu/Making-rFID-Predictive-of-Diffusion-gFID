#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import re
import os

LOG_DIR = Path("/video_ssd/kongzishang/xutongda/git/IFID/nohup_logs_imagenet_ifid_interp_200k50k")

# 默认包含 slerp 0.5；如果只想要 slerp 0.1~0.4：
#   INCLUDE_SLERP_05=0 python summarize_ifid_ablation_linear_slerp_csv.py
INCLUDE_SLERP_05 = os.environ.get("INCLUDE_SLERP_05", "1") != "0"

OUT_CSV = LOG_DIR / (
    "imagenet_ifid_ablation_linear01-05_slerp01-05_ordered.csv"
    if INCLUDE_SLERP_05
    else "imagenet_ifid_ablation_linear01-05_slerp01-04_ordered.csv"
)
OUT_FULL_CSV = LOG_DIR / (
    "imagenet_ifid_ablation_linear01-05_slerp01-05_ordered_full.csv"
    if INCLUDE_SLERP_05
    else "imagenet_ifid_ablation_linear01-05_slerp01-04_ordered_full.csv"
)

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

LINEAR_ALPHAS = ["0.1", "0.2", "0.3", "0.4", "0.5"]
SLERP_ALPHAS = ["0.1", "0.2", "0.3", "0.4"] + (["0.5"] if INCLUDE_SLERP_05 else [])

METRIC_RE = re.compile(r"^ifid\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", re.M)

def alpha_tag(alpha: str) -> str:
    return "a" + alpha.replace(".", "")

def read_last_ifid(log_path: Path):
    if not log_path.exists():
        return "", "MISSING_LOG"

    text = log_path.read_text(errors="ignore")
    matches = METRIC_RE.findall(text)

    if not matches:
        return "", "MISSING_IFID"

    return matches[-1], "OK"

def log_path_for(tag: str, intp: str, alpha: str):
    return LOG_DIR / f"imagenet-ifid-200k50k-{tag}-{intp}-{alpha_tag(alpha)}.out"

columns = (
    ["Model"]
    + [f"linear_{a}" for a in LINEAR_ALPHAS]
    + [f"slerp_{a}" for a in SLERP_ALPHAS]
)

full_columns = (
    ["YAML", "Model", "Tag", "Group"]
    + [f"linear_{a}" for a in LINEAR_ALPHAS]
    + [f"slerp_{a}" for a in SLERP_ALPHAS]
    + ["Status", "MissingOrIncomplete"]
)

rows = []
full_rows = []
problems = []

for yaml_name, display_name, tag, group in MODELS:
    row = {"Model": display_name}
    full = {
        "YAML": yaml_name,
        "Model": display_name,
        "Tag": tag,
        "Group": group,
    }

    missing_items = []

    for alpha in LINEAR_ALPHAS:
        col = f"linear_{alpha}"
        path = log_path_for(tag, "linear", alpha)
        val, status = read_last_ifid(path)
        row[col] = val
        full[col] = val
        if status != "OK":
            missing_items.append(f"{col}:{status}:{path}")

    for alpha in SLERP_ALPHAS:
        col = f"slerp_{alpha}"
        path = log_path_for(tag, "slerp", alpha)
        val, status = read_last_ifid(path)
        row[col] = val
        full[col] = val
        if status != "OK":
            missing_items.append(f"{col}:{status}:{path}")

    full["Status"] = "OK" if not missing_items else "INCOMPLETE"
    full["MissingOrIncomplete"] = " | ".join(missing_items)

    if missing_items:
        problems.append((display_name, tag, missing_items))

    rows.append(row)
    full_rows.append(full)

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

with OUT_FULL_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=full_columns)
    writer.writeheader()
    writer.writerows(full_rows)

print("=" * 100)
print(f"LOG_DIR:       {LOG_DIR}")
print(f"CSV:           {OUT_CSV}")
print(f"FULL CSV:      {OUT_FULL_CSV}")
print(f"SLERP 0.5:     {'included' if INCLUDE_SLERP_05 else 'excluded'}")
print(f"DONE:          {len(MODELS) - len(problems)}/{len(MODELS)}")
print("=" * 100)

if problems:
    print("\n[MISSING / INCOMPLETE]")
    for name, tag, items in problems:
        print(f"\n{name} ({tag})")
        for item in items:
            print(f"  {item}")

print("\n[PREVIEW]")
print(",".join(columns))
for r in rows:
    print(",".join(str(r.get(c, "")) for c in columns))
