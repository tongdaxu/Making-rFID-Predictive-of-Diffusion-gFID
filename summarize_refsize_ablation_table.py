#!/usr/bin/env python3

import csv
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


ROOT = Path("/share/xutongda/REPA-E/exps_interpolate")

MODELS = [
    ("SD-VAE", "sdvae", 46.74, 25.91),
    ("SD3-VAE", "sd3vae", 51.39, 26.38),
    ("FLUX-VAE", "fluxvae", 63.32, 34.06),
    ("DC-AE", "dcae", 42.01, 26.68),
    ("SOFT-VQ", "softvq", 29.11, 15.88),
    ("VA-VAE", "vavae", 17.63, 8.57),
    ("VA-VAE-64", "vavae64", 32.51, 15.09),
    ("EQ-VAE", "eqvae", 37.81, 20.81),
    ("MAE-TOK", "maetok", 13.73, 6.27),
    ("IN-VAE", "invae", 49.17, 25.65),
    ("REPAE", "repae-sdvae", 26.06, 12.95),
    ("DE-TOK", "detok", 20.27, 11.97),
    ("QW-VAE", "qwenimgvae", 48.34, 23.62),

    # No SiT/B result for RAE.
    ("RAE", "rae", None, 4.25),

    ("SVG", "svg", 18.63, 7.65),
    ("FLUX2-VAE", "flux2", 22.75, 10.18),
    ("SVG-T2I", "svg-t2i-p256", 10.69, 5.63),
    ("DM-VAE", "dmvae", 8.69, 4.65),
    ("VTP-S", "vtps", 16.72, 9.55),
    ("VTP-B", "vtpb", 16.24, 7.64),
    ("VTP-L", "vtpl", 15.69, 5.79),
    ("PAE", "pae-dino", 14.15, 5.65),
]


def parse_ifid(ref_size):
    log_dir = ROOT / f"logs_ifid_refsize_{ref_size}"

    result = {}
    paths = {}

    for model, tag, _, _ in MODELS:
        log = log_dir / (
            f"ifid-ref{ref_size}-{tag}-"
            f"slerp-a05-top10.out"
        )

        paths[tag] = log
        result[tag] = None

        if not log.exists():
            continue

        text = log.read_text(errors="ignore")

        matches = re.findall(
            r"^ifid=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)",
            text,
            flags=re.MULTILINE,
        )

        if matches:
            result[tag] = float(matches[-1])

    return result, paths


def calc_corr(x, y):
    if len(x) < 3:
        return None

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    return (
        float(pearsonr(x, y).statistic),
        float(spearmanr(x, y).statistic),
        float(kendalltau(x, y).statistic),
    )


def correlation(ref_size):
    ifids, paths = parse_ifid(ref_size)

    missing = []
    completed = []

    b_ifid = []
    b_gfid = []
    b_models = []

    xl_ifid = []
    xl_gfid = []
    xl_models = []

    for model, tag, gfid_b, gfid_xl in MODELS:
        value = ifids[tag]

        if value is None:
            missing.append(model)
            continue

        completed.append(model)

        if gfid_b is not None:
            b_ifid.append(value)
            b_gfid.append(gfid_b)
            b_models.append(model)

        xl_ifid.append(value)
        xl_gfid.append(gfid_xl)
        xl_models.append(model)

    result = {
        "ref_size": ref_size,
        "ifids": ifids,
        "paths": paths,
        "completed": completed,
        "missing": missing,
        "n_complete": len(completed),
        "complete": len(completed) == len(MODELS),

        # Expected final sample sizes:
        # SiT/B = 21 because RAE is unavailable
        # SiT/XL = 22
        "b_n": len(b_ifid),
        "b_expected": 21,
        "xl_n": len(xl_ifid),
        "xl_expected": 22,

        "b_corr": calc_corr(b_ifid, b_gfid),
        "xl_corr": calc_corr(xl_ifid, xl_gfid),

        "b_models": b_models,
        "xl_models": xl_models,
    }

    return result


def read_runtime():
    path = (
        ROOT
        / "runtime_sdvae_refsize_4a800"
        / "sdvae_refsize_runtime.csv"
    )

    result = {}

    if not path.exists():
        return result

    try:
        with path.open() as f:
            for row in csv.DictReader(f):
                try:
                    ref = int(row["RefImages"])
                    result[ref] = {
                        "seconds": float(row["Seconds"]),
                        "minutes": float(row["Minutes"]),
                        "hours": float(row["Hours"]),
                    }
                except Exception:
                    continue
    except Exception:
        pass

    return result


def fmt_corr(value):
    if value is None:
        return ""

    return f"{value:.2f}"


def fmt_triplet(values):
    if values is None:
        return ("", "", "")

    return tuple(fmt_corr(x) for x in values)


def fmt_time(runtime, n):
    if n not in runtime:
        return ""

    return f"{runtime[n]['minutes']:.1f} min"


def print_status(result):
    ref = result["ref_size"]

    print()
    print("=" * 100)
    print(f"REFERENCE SIZE = {ref}")
    print("=" * 100)

    print(
        f"iFID complete: "
        f"{result['n_complete']}/22"
    )

    if result["missing"]:
        print(
            "Missing: "
            + ", ".join(result["missing"])
        )
    else:
        print("Missing: none")

    print()

    if result["b_corr"] is not None:
        flag = (
            "FINAL"
            if result["b_n"] == result["b_expected"]
            else "PARTIAL"
        )

        print(
            f"SiT/B w/o CFG [{flag}] "
            f"n={result['b_n']}/{result['b_expected']}: "
            f"PCC={result['b_corr'][0]:.6f}, "
            f"SRCC={result['b_corr'][1]:.6f}, "
            f"KRCC={result['b_corr'][2]:.6f}"
        )
    else:
        print(
            f"SiT/B w/o CFG: insufficient data "
            f"n={result['b_n']}/{result['b_expected']}"
        )

    if result["xl_corr"] is not None:
        flag = (
            "FINAL"
            if result["xl_n"] == result["xl_expected"]
            else "PARTIAL"
        )

        print(
            f"SiT/XL w/o CFG [{flag}] "
            f"n={result['xl_n']}/{result['xl_expected']}: "
            f"PCC={result['xl_corr'][0]:.6f}, "
            f"SRCC={result['xl_corr'][1]:.6f}, "
            f"KRCC={result['xl_corr'][2]:.6f}"
        )
    else:
        print(
            f"SiT/XL w/o CFG: insufficient data "
            f"n={result['xl_n']}/{result['xl_expected']}"
        )

    print()
    print("Individual iFID values:")
    print("-" * 100)

    for model, tag, _, _ in MODELS:
        value = result["ifids"][tag]

        if value is None:
            print(
                f"{model:<24} "
                f"{'--':>12}  MISSING"
            )
        else:
            print(
                f"{model:<24} "
                f"{value:>12.4f}  OK"
            )


def overleaf_row(
    setting,
    interp,
    alpha,
    nimg,
    b_corr,
    xl_corr,
    runtime,
):
    b = fmt_triplet(b_corr)
    xl = fmt_triplet(xl_corr)

    return (
        f"{setting} & {interp} & {alpha} & {nimg} & "
        f"{b[0]} & {b[1]} & {b[2]} & "
        f"{xl[0]} & {xl[1]} & {xl[2]} & "
        f"{runtime} \\\\"
    )


def main():
    runtime = read_runtime()

    r50 = correlation(50000)
    r1m = correlation(1000000)

    # ------------------------------------------------------------
    # Full status
    # ------------------------------------------------------------

    print_status(r50)
    print_status(r1m)

    # ------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------

    print()
    print("=" * 100)
    print("SD-VAE RUNTIME ON 4x A800")
    print("=" * 100)

    for n in [50000, 200000, 1000000]:
        if n in runtime:
            r = runtime[n]
            print(
                f"{n:7d}: "
                f"{r['seconds']:.2f}s = "
                f"{r['minutes']:.3f}min = "
                f"{r['hours']:.4f}h"
            )
        else:
            print(f"{n:7d}: MISSING")

    t50 = fmt_time(runtime, 50000)
    t200 = fmt_time(runtime, 200000)
    t1m = fmt_time(runtime, 1000000)

    # ------------------------------------------------------------
    # Existing Overleaf rows
    # ------------------------------------------------------------

    print()
    print("=" * 100)
    print("OVERLEAF")
    print("=" * 100)

    existing_rows = [
        ("(a)", "Linear", "0.1", "200k",
         (0.14, 0.07, 0.03),
         (0.23, 0.19, 0.13)),

        ("(b)", "Linear", "0.3", "200k",
         (0.33, 0.34, 0.24),
         (0.60, 0.65, 0.51)),

        ("(c)", "Linear", "0.5", "200k",
         (0.69, 0.74, 0.55),
         (0.75, 0.79, 0.61)),

        ("(d)", "Slerp", "0.1", "200k",
         (-0.14, -0.11, -0.12),
         (0.14, 0.15, 0.11)),

        ("(e)", "Slerp", "0.3", "200k",
         (0.25, 0.35, 0.27),
         (0.61, 0.65, 0.53)),

        ("(f)", "Slerp", "0.5", "200k",
         (0.85, 0.82, 0.62),
         (0.88, 0.84, 0.65)),
    ]

    for setting, interp, alpha, nimg, b, xl in existing_rows:
        print(
            overleaf_row(
                setting,
                interp,
                alpha,
                nimg,
                b,
                xl,
                t200,
            )
        )

    # ------------------------------------------------------------
    # 50k row
    # ------------------------------------------------------------

    if r50["complete"]:
        print(
            overleaf_row(
                "(g)",
                "Slerp",
                "0.5",
                "50k",
                r50["b_corr"],
                r50["xl_corr"],
                t50,
            )
        )
    else:
        print(
            "% (g) INCOMPLETE: "
            f"{r50['n_complete']}/22 iFID finished; "
            "do not use partial correlations in the paper."
        )

        # Still print diagnostic partial values as a LaTeX comment.
        if r50["b_corr"] is not None and r50["xl_corr"] is not None:
            print(
                "% partial 50k: "
                + overleaf_row(
                    "(g)",
                    "Slerp",
                    "0.5",
                    "50k",
                    r50["b_corr"],
                    r50["xl_corr"],
                    t50,
                )
            )

    # ------------------------------------------------------------
    # 1M row
    # ------------------------------------------------------------

    if r1m["complete"]:
        print(
            overleaf_row(
                "(h)",
                "Slerp",
                "0.5",
                "1000k",
                r1m["b_corr"],
                r1m["xl_corr"],
                t1m,
            )
        )
    else:
        print(
            "% (h) INCOMPLETE: "
            f"{r1m['n_complete']}/22 iFID finished; "
            "do not use partial correlations in the paper."
        )

        if r1m["b_corr"] is not None and r1m["xl_corr"] is not None:
            print(
                "% partial 1000k: "
                + overleaf_row(
                    "(h)",
                    "Slerp",
                    "0.5",
                    "1000k",
                    r1m["b_corr"],
                    r1m["xl_corr"],
                    t1m,
                )
            )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"50k : {r50['n_complete']}/22 "
        f"{'COMPLETE' if r50['complete'] else 'PARTIAL'}"
    )

    print(
        f"1M  : {r1m['n_complete']}/22 "
        f"{'COMPLETE' if r1m['complete'] else 'PARTIAL'}"
    )

    # IMPORTANT:
    # Never exit with error just because experiments are incomplete.
    print()
    print("[OK] Summary finished. Missing experiments were tolerated.")


if __name__ == "__main__":
    main()
