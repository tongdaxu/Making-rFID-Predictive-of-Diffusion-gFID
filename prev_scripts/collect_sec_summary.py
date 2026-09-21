#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import csv
from pathlib import Path

root = Path("/video_ssd/kongzishang/xutongda/git/IFID/latent_metrics_viv_cds_srss/sec")
out_csv = root / "sec_summary.csv"

rows = []
for model_dir in sorted(root.iterdir()):
    if not model_dir.is_dir() or model_dir.name == "logs":
        continue
    p = model_dir / "metrics.json"
    if not p.exists():
        continue
    try:
        m = json.loads(p.read_text())
    except Exception as e:
        rows.append({"model": model_dir.name, "error": str(e)})
        continue

    rows.append({
        "model": model_dir.name,
        "SEC@0.25": m.get("SEC@0.25", "NA"),
        "SEC@0.5": m.get("SEC@0.5", "NA"),
        "dist_type": m.get("dist_type", "NA"),
        "max_dist": m.get("max_dist", "NA"),
        "num_images": m.get("num_images", "NA"),
        "latent_shape": "x".join(map(str, m.get("latent_shape", []))) if m.get("latent_shape") else "NA",
        "spatial_shape": "x".join(map(str, m.get("spatial_shape", []))) if m.get("spatial_shape") else "NA",
        "vae_config": m.get("vae_config", "NA"),
        "error": "",
    })

fieldnames = [
    "model", "SEC@0.25", "SEC@0.5", "dist_type", "max_dist",
    "num_images", "latent_shape", "spatial_shape", "vae_config", "error"
]
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {out_csv}")
print(f"{'model':30s}\t{'SEC@0.25':>12s}\t{'SEC@0.5':>12s}")
print("-" * 60)
for r in rows:
    def fmt(x):
        return f"{x:.6f}" if isinstance(x, float) else str(x)
    print(f"{r['model']:30s}\t{fmt(r.get('SEC@0.25','NA')):>12s}\t{fmt(r.get('SEC@0.5','NA')):>12s}")
