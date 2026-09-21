#!/usr/bin/env python3
"""
Collect PSNR / LPIPS columns from proxy_psnr_lpips B + XL results.

Designed for spreadsheet layout:
  B rFID | B PSNR | B LPIPS | XL rFID | XL PSNR | XL LPIPS

Outputs one txt file per spreadsheet column, with one value per line
in the fixed VAE row order below, so each file can be pasted directly
into one spreadsheet column.

Default setting order matches the current table:
  step xpred 0.8 ODE
  step xpred 0.8 SDE
  step xpred 0.4 ODE
  step xpred 0.4 SDE
  step xpred 0.2 ODE
  step xpred 0.2 SDE

Edit SETTINGS below if needed, e.g. add:
  ("direct", 0.1), ("direct", 0.4), ("direct", 0.8)

Missing results are emitted as blank lines, preserving row alignment.
"""

import argparse
import json
from pathlib import Path

SETTINGS = [
    ("ode", 0.8),
    ("sde", 0.8),
    ("ode", 0.4),
    ("sde", 0.4),
    ("ode", 0.2),
    ("sde", 0.2),
]

MODELS = [
    ("SD-VAE",              "sit-b-sdvae-400k",            "sit-xl-sdvae-400k"),
    ("SD3-VAE",             "sit-b-sd3-400k",              "sit-xl-sd3-shift-400k"),
    ("FLUX-VAE",            "sit-b-flux-shift-400k",       "sit-xl-flux-shift-400k"),
    ("DC-AE",               "sit-b-dcae-400k",             "sit-xl-dcae-400k"),
    ("SOFT-VQ",             "sit-b-softvq-400k",           "sit-XL-softvq-400k"),
    ("VA-VAE",              "sit-b-vavae-400k",            "sit-xl-vavae-400k"),
    ("VA-VAE-64",           "sit-b-vavae64-400k",          "sit-xl-vavae64-shift-400k"),
    ("EQ-VAE",              "sit-b-eqvae-400k",            "sit-XL-eqvae-400k"),
    ("MAE-TOK",             "sit-b-maetok-400k",           "sit-XL-maetok-400k"),
    ("IN-VAE",              "sit-b-invae-400k",            "sit-xl-invae-400k"),
    ("REPAE-SDVAE",         "sit-b-repae-400k",            "sit-xl-repae-400k"),
    ("DE-TOK",              "sit-b-detok-400k",            "sit-xl-detok-400k"),
    ("QwenImg-VAE",         "sit-b-qw-shift-400k",         "sit-xl-qw-shift-400k"),
    ("RAE",                 "sit-b-rae-shift-400k",        "sit-xl-rae-shift-400k"),
    ("SVG",                 "sit-b-svg-400k",              "sit-xl-svg-400k"),
    ("FLUX2-VAE",           "sit-b-flux2vae-400k",         "sit-xl-flux2vae-400k"),
    ("SVG T2I",             "sit-b-svgt2iP-400k",          "sit-xl-svgt2iP-400k"),
    ("DM-VAE",              "sit-b-dmvae-400k",            "sit-xl-dmvae-400k"),
    ("UAE",                 "sit-b-uae-400k",              "sit-xl-uae-400k"),
    ("VTP-S",               "sit-b-vtps-400k",             "sit-xl-vtps-400k"),
    ("VTP-B",               "sit-b-vtpb-400k",             "sit-xl-vtpb-400k"),
    ("VTP-L",               "sit-b-vtpl-400k",             "sit-xl-vtpl-400k"),
    ("RAE T2I SigLIP-2",    "sit-b-raet2i_siglip2-400k",   "sit-xl-raet2i_siglip2-400k"),
    ("RAE T2I WebSSL",      "sit-b-raet2i_webssl-400k",    "sit-xl-raet2i_webssl-400k"),
    ("PAE DINOv2",          "sit-b-pae-dinov2-400k",       "sit-xl-pae-dinov2-400k"),
]


def metric_key(kind, t):
    return f"{kind}_t{float(t):g}"


def setting_label(kind, t):
    if kind == "direct":
        return f"one_step_{float(t):g}"
    return f"step_xpred_{float(t):g}_{kind.upper()}"


def find_metrics_json(root, exp_name, needed_keys):
    root = Path(root)
    candidates = list(root.glob(f"{exp_name}_step*/metrics.json"))
    good = []
    for p in candidates:
        try:
            d = json.loads(p.read_text())
            metrics = d.get("metrics", {})
            if needed_keys.issubset(metrics.keys()):
                good.append(p)
        except Exception:
            pass
    if not good:
        return None
    return max(good, key=lambda p: p.stat().st_mtime)


def load_metrics(root, exp_name, needed_keys):
    p = find_metrics_json(root, exp_name, needed_keys)
    if p is None:
        return {}, None
    return json.loads(p.read_text()).get("metrics", {}), p


def fmt(v):
    return "" if v is None else f"{float(v):.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--b-root",
        default="/share/xutongda/REPA-E/exps_interpolate/proxy_psnr_lpips_b_all",
    )
    ap.add_argument(
        "--xl-root",
        default="/share/xutongda/REPA-E/exps_interpolate/proxy_psnr_lpips_xl_all",
    )
    ap.add_argument(
        "--out-dir",
        default="./proxy_psnr_lpips_columns",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Also print every generated column to terminal.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    needed_keys = {metric_key(k, t) for k, t in SETTINGS}

    rows = []
    manifest = ["vae\tb_metrics_json\txl_metrics_json\tb_status\txl_status"]

    for vae, b_exp, xl_exp in MODELS:
        b_metrics, b_path = load_metrics(args.b_root, b_exp, needed_keys)
        xl_metrics, xl_path = load_metrics(args.xl_root, xl_exp, needed_keys)

        rows.append({"vae": vae, "b": b_metrics, "xl": xl_metrics})
        manifest.append(
            "\t".join([
                vae,
                str(b_path or ""),
                str(xl_path or ""),
                "FOUND" if b_path else "MISSING",
                "FOUND" if xl_path else "MISSING",
            ])
        )

    (out_dir / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    (out_dir / "00_VAE_names.txt").write_text(
        "\n".join(r["vae"] for r in rows) + "\n"
    )

    created = []
    col = 1

    # Exact spreadsheet order around the existing rFID columns:
    # B rFID | [B PSNR] | [B LPIPS] | XL rFID | [XL PSNR] | [XL LPIPS]
    for kind, t in SETTINGS:
        key = metric_key(kind, t)
        label = setting_label(kind, t)

        for backbone, metric in [
            ("B", "psnr"),
            ("B", "lpips"),
            ("XL", "psnr"),
            ("XL", "lpips"),
        ]:
            side = backbone.lower()
            vals = []
            for row in rows:
                v = row[side].get(key, {}).get(metric)
                vals.append(fmt(v))

            name = f"{col:02d}_{label}_{backbone}_{metric.upper()}.txt"
            (out_dir / name).write_text("\n".join(vals) + "\n")
            created.append(name)
            col += 1

    guide = [
        "Each file = one spreadsheet column, 25 rows.",
        "Blank line = missing/not finished result.",
        "",
        "Spreadsheet group layout:",
        "B rFID | B PSNR | B LPIPS | XL rFID | XL PSNR | XL LPIPS",
        "",
        "Generated paste order:",
    ]
    guide += [f"{i+1:02d}. {name}" for i, name in enumerate(created)]
    (out_dir / "README_PASTE_ORDER.txt").write_text("\n".join(guide) + "\n")

    print(f"Output directory: {out_dir.resolve()}")
    print(f"Rows per column: {len(rows)}")
    print("")
    for name in created:
        print(name)

    missing_b = [r["vae"] for r in rows if not r["b"]]
    missing_xl = [r["vae"] for r in rows if not r["xl"]]
    print("")
    print("Missing B :", ", ".join(missing_b) if missing_b else "none")
    print("Missing XL:", ", ".join(missing_xl) if missing_xl else "none")

    if args.show:
        for name in created:
            print("\n" + "=" * 70)
            print(name)
            print("=" * 70)
            print((out_dir / name).read_text(), end="")


if __name__ == "__main__":
    main()
