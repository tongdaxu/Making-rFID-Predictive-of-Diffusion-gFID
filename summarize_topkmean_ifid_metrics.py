from pathlib import Path
import re
import csv

LOG_DIR = Path("/share/xutongda/REPA-E/exps_interpolate/logs_ifid_topkmean")

OUT_TSV = LOG_DIR / "topkmean_metrics_ordered.tsv"
OUT_MD = LOG_DIR / "topkmean_metrics_ordered.md"
OUT_MISSING = LOG_DIR / "topkmean_metrics_missing.txt"

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

METRIC_RE = re.compile(
    r"^(ifid|rfid|psnr|ssim|msssim|lpips)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.IGNORECASE,
)

def parse_metrics(path: Path):
    metrics = {}
    text = path.read_text(errors="ignore").splitlines()
    for line in text:
        m = METRIC_RE.match(line.strip())
        if m:
            key = m.group(1).lower()
            val = m.group(2)
            metrics[key] = val
    return metrics

rows = []
missing = []

for yaml_name, display, tag, group in MODELS:
    log_path = LOG_DIR / EXP_TMPL.format(tag=tag)

    if not log_path.exists():
        # 兜底：万一文件名有轻微变化，按 tag 找一下
        cands = sorted(LOG_DIR.glob(f"*topkmean-{tag}-slerp-a05-top10-200000x50000.out"))
        if cands:
            log_path = cands[0]
        else:
            missing.append(f"{display}\t{tag}\tNO_LOG\t{log_path}")
            rows.append({
                "Model": display,
                "Config": yaml_name,
                "Tag": tag,
                "iFID": "",
                "PSNR": "",
                "LPIPS": "",
                "SSIM": "",
                "rFID": "",
                "Log": str(log_path),
            })
            continue

    metrics = parse_metrics(log_path)

    required = ["ifid", "psnr", "lpips", "ssim", "rfid"]
    lack = [k for k in required if k not in metrics]
    if lack:
        missing.append(f"{display}\t{tag}\tMISSING_METRIC:{','.join(lack)}\t{log_path}")

    rows.append({
        "Model": display,
        "Config": yaml_name,
        "Tag": tag,
        "iFID": metrics.get("ifid", ""),
        "PSNR": metrics.get("psnr", ""),
        "LPIPS": metrics.get("lpips", ""),
        "SSIM": metrics.get("ssim", ""),
        "rFID": metrics.get("rfid", ""),
        "Log": str(log_path),
    })

# TSV
with OUT_TSV.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Model", "Config", "Tag", "iFID", "PSNR", "LPIPS", "SSIM", "rFID", "Log"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)

# Markdown
with OUT_MD.open("w") as f:
    f.write("| Model | Config | Tag | iFID | PSNR | LPIPS | SSIM | rFID |\n")
    f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(
            f"| {r['Model']} | {r['Config']} | {r['Tag']} | "
            f"{r['iFID']} | {r['PSNR']} | {r['LPIPS']} | {r['SSIM']} | {r['rFID']} |\n"
        )

OUT_MISSING.write_text("\n".join(missing) + ("\n" if missing else ""))

print("=" * 100)
print(f"LOG_DIR:     {LOG_DIR}")
print(f"TSV:         {OUT_TSV}")
print(f"Markdown:    {OUT_MD}")
print(f"Missing:     {OUT_MISSING}")
print(f"Done rows:   {len(rows)}")
print(f"Missing cnt: {len(missing)}")
print("=" * 100)

print("\n[TSV Preview]")
print("Model\tConfig\tTag\tiFID\tPSNR\tLPIPS\tSSIM\trFID")
for r in rows:
    print(f"{r['Model']}\t{r['Config']}\t{r['Tag']}\t{r['iFID']}\t{r['PSNR']}\t{r['LPIPS']}\t{r['SSIM']}\t{r['rFID']}")

if missing:
    print("\n[MISSING / INCOMPLETE]")
    for x in missing:
        print(x)
else:
    print("\n[OK] all logs parsed successfully.")
