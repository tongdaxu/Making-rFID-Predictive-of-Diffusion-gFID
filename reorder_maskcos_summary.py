from pathlib import Path
import csv

ROOT = Path("/video_ssd/kongzishang/xutongda/git/IFID/latent_metrics_maskcos")
IN_CSV = ROOT / "maskcos_summary.csv"
OUT_CSV = ROOT / "maskcos_summary_ordered.csv"

ORDER_CONFIGS = [
    "SDVAE.yaml",
    "SD3VAE.yaml",
    "FLUXVAE.yaml",
    "DCAE.yaml",
    "SOFTVQ.yaml",
    "VAVAE.yaml",
    "VAVAE64.yaml",
    "EQVAE.yaml",
    "MAETOK.yaml",
    "INVAE.yaml",
    "REPAEVAE.yaml",
    "DETOK.yaml",
    "QWVAE.yaml",
    "RAE.yaml",
    "SVG.yaml",
    "FLUX2VAE.yaml",
    "SVGT2I_P_STAGE1_256.yaml",
    "DMVAE.yaml",
    "UAE.yaml",
    "VTPS.yaml",
    "VTPB.yaml",
    "VTPL.yaml",
    "ScaleRAE_SIGLIP2_NONORM.yaml",
    "ScaleRAE_WEBSSL.yaml",
    "PAE_DINOv2L_d32.yaml",
]

ORDER = [x.replace(".yaml", "") for x in ORDER_CONFIGS]

ALIASES = {
    # 如果 summary 里用了其他名字，可以在这里补映射
    "SD3": "SD3VAE",
    "FLUX": "FLUXVAE",
    "REPAE": "REPAEVAE",
    "SVGT2IP": "SVGT2I_P_STAGE1_256",
    "SVGT2iP": "SVGT2I_P_STAGE1_256",
    "PAE_DINOv2L": "PAE_DINOv2L_d32",
}

if not IN_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

with IN_CSV.open(newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

if not fieldnames:
    raise RuntimeError(f"Empty or invalid CSV: {IN_CSV}")

# 找 model 列
model_col = None
for c in fieldnames:
    if c.lower() in {"model", "name", "vae", "config"}:
        model_col = c
        break

if model_col is None:
    raise RuntimeError(f"Cannot find model column. Columns: {fieldnames}")

def normalize_name(x: str) -> str:
    x = str(x).strip()
    x = x.replace(".yaml", "")
    x = x.replace("\r", "")
    return ALIASES.get(x, x)

by_model = {}
dups = []

for r in rows:
    m = normalize_name(r.get(model_col, ""))
    if not m:
        continue
    if m in by_model:
        dups.append(m)
    by_model[m] = r

ordered_rows = []
missing = []

for m in ORDER:
    if m in by_model:
        ordered_rows.append(by_model[m])
    else:
        # 缺失的模型也保留一行，方便你看出来还没跑完
        blank = {k: "" for k in fieldnames}
        blank[model_col] = m
        ordered_rows.append(blank)
        missing.append(m)

extra = sorted(set(by_model.keys()) - set(ORDER))

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(ordered_rows)

print("=" * 100)
print(f"Input CSV:  {IN_CSV}")
print(f"Output CSV: {OUT_CSV}")
print(f"Input rows: {len(rows)}")
print(f"Ordered rows: {len(ordered_rows)}")
print(f"Missing in input: {len(missing)}")
for m in missing:
    print(f"[MISSING] {m}")
print(f"Extra rows not in requested order: {len(extra)}")
for m in extra:
    print(f"[EXTRA] {m}")
if dups:
    print(f"Duplicate model rows: {len(dups)}")
    for m in sorted(set(dups)):
        print(f"[DUP] {m}")
print("=" * 100)
