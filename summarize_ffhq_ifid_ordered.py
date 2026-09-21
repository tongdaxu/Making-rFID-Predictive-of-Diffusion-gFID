from pathlib import Path
import csv
import re

LOG_DIR = Path("/video_ssd/kongzishang/xutongda/git/IFID/nohup_logs_ffhq")
OUT_CSV = LOG_DIR / "ffhq_ifid_ordered_summary.csv"

# 按用户指定顺序；缺失的三类 VAE 不放入这里。
ORDER = [
    ("SDVAE.yaml", "ffhq-ifid-sdvae.out"),
    ("SD3VAE.yaml", "ffhq-ifid-sd3vae.out"),
    ("FLUXVAE.yaml", "ffhq-ifid-fluxvae.out"),
    ("DCAE.yaml", "ffhq-ifid-dcae.out"),
    ("SOFTVQ.yaml", "ffhq-ifid-softvq.out"),
    ("VAVAE.yaml", "ffhq-ifid-vavae.out"),
    ("VAVAE64.yaml", "ffhq-ifid-vavae64.out"),
    ("EQVAE.yaml", "ffhq-ifid-eqvae.out"),
    ("MAETOK.yaml", "ffhq-ifid-maetok.out"),
    ("INVAE.yaml", "ffhq-ifid-invae.out"),
    ("REPAEVAE.yaml", "ffhq-ifid-repae-sdvae.out"),
    ("DETOK.yaml", "ffhq-ifid-detok.out"),
    ("QWVAE.yaml", "ffhq-ifid-qwenimgvae.out"),
    ("RAE.yaml", "ffhq-ifid-rae.out"),
    ("SVG.yaml", "ffhq-ifid-svg.out"),
    ("FLUX2VAE.yaml", "ffhq-ifid-flux2.out"),
    ("SVGT2I_P_STAGE1_256.yaml", "ffhq-ifid-svg-t2i-p256.out"),
    ("DMVAE.yaml", "ffhq-ifid-dmvae.out"),
    ("VTPS.yaml", "ffhq-ifid-vtps.out"),
    ("VTPB.yaml", "ffhq-ifid-vtpb.out"),
    ("VTPL.yaml", "ffhq-ifid-vtpl.out"),
    ("PAE_DINOv2L_d32.yaml", "ffhq-ifid-pae-dino.out"),
]

# 输出列顺序
FIELDS = ["model", "iFID", "PSNR", "LPIPS", "SSIM", "rFID"]

# log 内部 metric 名到输出列名
METRIC_MAP = {
    "ifid": "iFID",
    "psnr": "PSNR",
    "lpips": "LPIPS",
    "ssim": "SSIM",
    "rfid": "rFID",
}

def parse_log(path: Path):
    text = path.read_text(errors="ignore")

    result = {}
    for raw_name, out_name in METRIC_MAP.items():
        # 取最后一次出现的值，避免日志里有中间重复输出
        matches = re.findall(
            rf"^{raw_name}=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            text,
            flags=re.MULTILINE,
        )
        result[out_name] = matches[-1] if matches else ""

    return result

rows = []
missing_logs = []
incomplete = []

for model, log_name in ORDER:
    log_path = LOG_DIR / log_name

    if not log_path.exists():
        missing_logs.append((model, log_path))
        continue

    metrics = parse_log(log_path)

    if not all(metrics.get(c, "") for c in ["iFID", "PSNR", "LPIPS", "SSIM", "rFID"]):
        incomplete.append((model, log_path, metrics))
        continue

    rows.append({
        "model": model,
        "iFID": metrics["iFID"],
        "PSNR": metrics["PSNR"],
        "LPIPS": metrics["LPIPS"],
        "SSIM": metrics["SSIM"],
        "rFID": metrics["rFID"],
    })

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print("=" * 100)
print(f"Log dir:    {LOG_DIR}")
print(f"Output CSV: {OUT_CSV}")
print(f"Rows saved: {len(rows)}")
print("=" * 100)

if missing_logs:
    print("\n[MISSING LOGS]")
    for model, path in missing_logs:
        print(f"{model}: {path}")

if incomplete:
    print("\n[INCOMPLETE LOGS]")
    for model, path, metrics in incomplete:
        print(f"{model}: {path}")
        print(metrics)

print("\n[PREVIEW]")
with OUT_CSV.open() as f:
    print(f.read())
