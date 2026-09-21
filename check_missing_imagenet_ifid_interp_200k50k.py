from pathlib import Path
import csv

LOG_DIR = Path("/video_ssd/kongzishang/xutongda/git/IFID/nohup_logs_imagenet_ifid_interp_200k50k")

OUT_CSV = LOG_DIR / "missing_imagenet_ifid_interp_200k50k.csv"
OUT_TXT = LOG_DIR / "missing_imagenet_ifid_interp_200k50k.txt"

MODELS = [
    ("SDVAE.yaml", "sdvae"),
    ("SD3VAE.yaml", "sd3vae"),
    ("FLUXVAE.yaml", "fluxvae"),
    ("DCAE.yaml", "dcae"),
    ("SOFTVQ.yaml", "softvq"),
    ("VAVAE.yaml", "vavae"),
    ("VAVAE64.yaml", "vavae64"),
    ("EQVAE.yaml", "eqvae"),
    ("MAETOK.yaml", "maetok"),
    ("INVAE.yaml", "invae"),
    ("REPAEVAE.yaml", "repae-sdvae"),
    ("DETOK.yaml", "detok"),
    ("QWVAE.yaml", "qwenimgvae"),
    ("RAE.yaml", "rae"),
    ("SVG.yaml", "svg"),
    ("FLUX2VAE.yaml", "flux2"),
    ("SVGT2I_P_STAGE1_256.yaml", "svg-t2i-p256"),
    ("DMVAE.yaml", "dmvae"),
    ("VTPS.yaml", "vtps"),
    ("VTPB.yaml", "vtpb"),
    ("VTPL.yaml", "vtpl"),
    ("PAE_DINOv2L_d32.yaml", "pae-dino"),
]

EXPERIMENTS = [
    ("linear", "0.1"),
    ("linear", "0.2"),
    ("linear", "0.3"),
    ("linear", "0.4"),
    ("linear", "0.25"),
    ("linear", "0.5"),
    ("slerp", "0.1"),
    ("slerp", "0.2"),
    ("slerp", "0.3"),
    ("slerp", "0.4"),
    ("slerp", "0.25"),
]

def alpha_tag(a: str) -> str:
    return "a" + a.replace(".", "")

rows = []

for model, tag in MODELS:
    for intp, alpha in EXPERIMENTS:
        exp_name = f"imagenet-ifid-200k50k-{tag}-{intp}-{alpha_tag(alpha)}"
        out_path = LOG_DIR / f"{exp_name}.out"
        done_path = LOG_DIR / f"{exp_name}.out.done"

        if not out_path.exists():
            status = "NO_OUT"
        elif not done_path.exists():
            status = "OUT_WITHOUT_DONE"
        else:
            status = "DONE"

        if status != "DONE":
            rows.append({
                "model": model,
                "tag": tag,
                "intp": intp,
                "alpha": alpha,
                "exp": exp_name,
                "status": status,
                "out": str(out_path),
                "done": str(done_path),
            })

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "tag", "intp", "alpha", "exp", "status", "out", "done"],
    )
    writer.writeheader()
    writer.writerows(rows)

with OUT_TXT.open("w") as f:
    for r in rows:
        f.write(f"{r['model']}\t{r['intp']}\t{r['alpha']}\t{r['status']}\t{r['exp']}\n")

total_expected = len(MODELS) * len(EXPERIMENTS)
done_count = total_expected - len(rows)

print("=" * 100)
print(f"LOG_DIR:        {LOG_DIR}")
print(f"Expected total: {total_expected}")
print(f"Done:           {done_count}")
print(f"Missing/failed: {len(rows)}")
print(f"CSV:            {OUT_CSV}")
print(f"TXT:            {OUT_TXT}")
print("=" * 100)

if rows:
    print("\n[MISSING / NOT DONE]")
    for r in rows:
        print(f"{r['model']} | {r['intp']} | alpha={r['alpha']} | {r['status']} | {r['exp']}")
else:
    print("\n[OK] all 110 experiments have .out.done")
