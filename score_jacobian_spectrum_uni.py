import os
import re
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_FILES = ["metrics.json", "eig_nontrivial.npy", "summary.csv"]


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def has_required_files(folder: str) -> bool:
    return all(os.path.exists(os.path.join(folder, f)) for f in REQUIRED_FILES)


def find_result_dirs(parent_dir: str, recursive: bool = True) -> List[str]:
    result_dirs = []
    if recursive:
        for root, _, _ in os.walk(parent_dir):
            if has_required_files(root):
                result_dirs.append(root)
    else:
        for name in sorted(os.listdir(parent_dir)):
            path = os.path.join(parent_dir, name)
            if os.path.isdir(path) and has_required_files(path):
                result_dirs.append(path)
    return sorted(result_dirs)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def infer_model_name(out_dir: str, metrics: Dict) -> str:
    """
    Prefer VAE config name. Fall back to folder name.
    """
    vae_config = metrics.get("vae_config", "")
    base = os.path.splitext(os.path.basename(vae_config))[0]

    if base:
        name = base.upper()
    else:
        folder = os.path.basename(out_dir).lower()
        if "vavae64" in folder:
            name = "VAVAE64"
        elif "vavae" in folder:
            name = "VAVAE"
        elif "sdvae" in folder:
            name = "SDVAE"
        elif "flux" in folder:
            name = "FLUXVAE"
        elif "dcae" in folder:
            name = "DCAE"
        elif "sd3" in folder:
            name = "SD3VAE"
        else:
            name = os.path.basename(out_dir)

    # Normalize common names.
    name = name.replace("FLUX", "FLUXVAE") if name == "FLUX" else name
    return name


def is_bn_result(out_dir: str, metrics: Dict) -> bool:
    """
    Only keep bn/channel-normalized results.

    Works for either:
      metric_norm == bn_channel
    or folder names like:
      score_jacobian_sdvae_bn_t05_exact_global_k512
    """
    metric_norm = str(metrics.get("metric_norm", "")).lower()
    folder = os.path.basename(out_dir).lower()

    return (
        metric_norm in ["bn_channel", "bn", "channel_bn"]
        or "_bn_" in folder
        or folder.endswith("_bn")
    )


def short_title(model_name: str, metrics: Dict) -> str:
    t = metrics.get("t", "unknown")
    topk = metrics.get("topk", "unknown")
    return f"{model_name} | t={t}, topk={topk}, bn"


def load_result(out_dir: str, parent_dir: str) -> Dict:
    metrics = load_json(os.path.join(out_dir, "metrics.json"))
    summary = pd.read_csv(os.path.join(out_dir, "summary.csv"))

    if "lambda_max_nontriv" not in summary.columns:
        raise KeyError(f"{out_dir}/summary.csv missing lambda_max_nontriv")

    eig = np.load(os.path.join(out_dir, "eig_nontrivial.npy"), mmap_mode="r")
    flat = eig.reshape(-1)

    model_name = infer_model_name(out_dir, metrics)
    rel_name = os.path.relpath(out_dir, parent_dir)

    return {
        "out_dir": out_dir,
        "rel_name": rel_name,
        "model_name": model_name,
        "title": short_title(model_name, metrics),
        "metrics": metrics,
        "summary": summary,
        "flat": flat,
        "lambda_max": summary["lambda_max_nontriv"].to_numpy(),
        "base_eig": float(metrics["base_eig"]),
    }


def sample_array(arr, max_samples: int = 2_000_000) -> np.ndarray:
    """
    Efficient deterministic stride sampling from mmap/large arrays.
    """
    n = arr.shape[0]
    if n <= max_samples:
        return np.asarray(arr)

    step = max(1, n // max_samples)
    sampled = np.asarray(arr[::step])
    if sampled.shape[0] > max_samples:
        sampled = sampled[:max_samples]
    return sampled


def hist_stream(arr, bins: int, range_: Tuple[float, float], chunk_size: int = 5_000_000):
    """
    Histogram for huge mmap arrays without loading the full array into RAM.
    """
    total_counts = np.zeros(bins, dtype=np.float64)
    n = arr.shape[0]

    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        vals = np.asarray(arr[s:e])
        counts, edges = np.histogram(vals, bins=bins, range=range_)
        total_counts += counts

    return total_counts, edges


def to_density(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return counts
    bin_width = edges[1] - edges[0]
    return counts / total / bin_width


def centers_from_edges(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def round_up_nice(x: float) -> float:
    """
    Round axis max to a readable scale.
    """
    if x <= 0:
        return 1.0
    exp = np.floor(np.log10(x))
    base = 10 ** exp
    m = x / base
    if m <= 1:
        return 1 * base
    if m <= 2:
        return 2 * base
    if m <= 5:
        return 5 * base
    return 10 * base


def make_grid(n: int):
    if n <= 2:
        return 1, n
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    return int(np.ceil(n / 4)), 4


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_bulk_grid(results: List[Dict], save_path: str, args):
    """
    Same x/y axes for all models.
    x-axis: lambda(J), zoomed to bulk range.
    """
    n = len(results)
    rows, cols = make_grid(n)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5.0 * cols, 3.6 * rows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    hist_data = []
    global_ymax = 0.0

    for r in results:
        counts, edges = hist_stream(
            r["flat"],
            bins=args.bulk_bins,
            range_=(args.bulk_min, args.bulk_max),
            chunk_size=args.hist_chunk_size,
        )
        y = to_density(counts, edges) if args.density else counts
        global_ymax = max(global_ymax, float(np.max(y)))
        hist_data.append((r, y, edges))

    for ax, (r, y, edges) in zip(axes, hist_data):
        x = centers_from_edges(edges)
        ax.step(x, y, where="mid")
        ax.axvline(r["base_eig"], linestyle="--", label=f"base={r['base_eig']:.2f}")
        ax.axvline(0.0, linestyle=":", label="zero")
        ax.set_title(r["title"])
        ax.set_xlabel(r"$\lambda(J)$")
        ax.set_ylabel("density" if args.density else "count")
        ax.set_xlim(args.bulk_min, args.bulk_max)
        ax.set_ylim(0, global_ymax * 1.05 if global_ymax > 0 else 1)
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Bulk eigenvalue histogram of empirical score Jacobian J\n"
        r"x-axis = eigenvalue $\lambda(J)$, same axis for all BN-normalized models",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def plot_eig_tail_logy_grid(results: List[Dict], save_path: str, args):
    """
    Same x/y axes for all models.
    x-axis: lambda(J), wider range, log-y to show sparse positive tail.
    """
    n = len(results)
    rows, cols = make_grid(n)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5.0 * cols, 3.6 * rows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    hist_data = []
    global_ymax = 0.0
    global_ymin_positive = np.inf

    for r in results:
        counts, edges = hist_stream(
            r["flat"],
            bins=args.tail_bins,
            range_=(args.tail_min, args.tail_max),
            chunk_size=args.hist_chunk_size,
        )
        y = to_density(counts, edges) if args.density else counts

        positive_y = y[y > 0]
        if len(positive_y) > 0:
            global_ymax = max(global_ymax, float(np.max(positive_y)))
            global_ymin_positive = min(global_ymin_positive, float(np.min(positive_y)))

        hist_data.append((r, y, edges))

    if not np.isfinite(global_ymin_positive):
        global_ymin_positive = 1e-12

    for ax, (r, y, edges) in zip(axes, hist_data):
        x = centers_from_edges(edges)
        ax.step(x, y, where="mid")
        ax.set_yscale("log")
        ax.axvline(r["base_eig"], linestyle="--", label=f"base={r['base_eig']:.2f}")
        ax.axvline(0.0, linestyle=":", label="zero")
        ax.set_title(r["title"])
        ax.set_xlabel(r"$\lambda(J)$")
        ax.set_ylabel("density, log scale" if args.density else "count, log scale")
        ax.set_xlim(args.tail_min, args.tail_max)
        ax.set_ylim(global_ymin_positive * 0.5, global_ymax * 2 if global_ymax > 0 else 1)
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Eigenvalue tail histogram of empirical score Jacobian J\n"
        r"x-axis = eigenvalue $\lambda(J)$, y-axis = log count/density, same axes for all BN models",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def plot_lambda_max_logtail_grid(results: List[Dict], save_path: str, args):
    """
    Same x/y axes for all models.
    x-axis: log10(lambda_max(J)), positive samples only.
    """
    n = len(results)
    rows, cols = make_grid(n)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5.0 * cols, 3.6 * rows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    log_arrays = []
    for r in results:
        lmax = r["lambda_max"]
        pos = lmax[lmax > 0]
        log_pos = np.log10(pos + 1e-8)
        log_arrays.append(log_pos)

    nonempty = [x for x in log_arrays if len(x) > 0]
    if len(nonempty) == 0:
        print("[Skip] No positive lambda_max found.")
        return

    if args.lambda_log_min is None:
        x_min = min(float(np.percentile(x, 1)) for x in nonempty)
        x_min = np.floor(x_min)
    else:
        x_min = args.lambda_log_min

    if args.lambda_log_max is None:
        x_max = max(float(np.percentile(x, args.lambda_log_axis_percentile)) for x in nonempty)
        x_max = np.ceil(x_max)
    else:
        x_max = args.lambda_log_max

    hist_data = []
    global_ymax = 0.0

    for r, log_pos in zip(results, log_arrays):
        counts, edges = np.histogram(
            log_pos,
            bins=args.lambda_tail_bins,
            range=(x_min, x_max),
        )
        y = counts.astype(np.float64)
        if args.density and y.sum() > 0:
            y = to_density(y, edges)
        global_ymax = max(global_ymax, float(np.max(y)) if len(y) else 0.0)
        hist_data.append((r, y, edges, len(log_pos)))

    for ax, (r, y, edges, n_pos) in zip(axes, hist_data):
        x = centers_from_edges(edges)
        ax.step(x, y, where="mid")
        ax.set_title(f"{r['title']}\npos samples={n_pos}")
        ax.set_xlabel(r"$\log_{10}(\lambda_{\max}(J))$")
        ax.set_ylabel("density" if args.density else "number of samples")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, global_ymax * 1.05 if global_ymax > 0 else 1)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        r"Positive $\lambda_{\max}(J)$ log-tail"
        "\n"
        r"x-axis = $\log_{10}(\lambda_{\max}(J))$, same axes for all BN models",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def plot_lambda_max_survival(results: List[Dict], save_path: str, args):
    """
    Overlay survival curves:
      y = P(lambda_max > threshold)

    This is often clearer than histograms for comparing sharpness tails.
    """
    all_pos = []
    for r in results:
        pos = r["lambda_max"][r["lambda_max"] > 0]
        if len(pos) > 0:
            all_pos.append(pos)

    if len(all_pos) == 0:
        print("[Skip] No positive lambda_max found.")
        return

    concat_pos = np.concatenate(all_pos)

    if args.survival_xmax is None:
        xmax = np.percentile(concat_pos, args.survival_axis_percentile)
        xmax = round_up_nice(float(xmax))
    else:
        xmax = args.survival_xmax

    xmin = max(args.survival_xmin, 1e-8)
    thresholds = np.logspace(np.log10(xmin), np.log10(xmax), args.survival_points)

    plt.figure(figsize=(8, 5.5))

    for r in results:
        lmax = r["lambda_max"]
        surv = np.array([(lmax > th).mean() for th in thresholds])
        plt.plot(thresholds, surv, label=r["model_name"])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"threshold on $\lambda_{\max}(J)$")
    plt.ylabel(r"fraction of samples with $\lambda_{\max}(J)$ > threshold")
    plt.title(
        r"Survival curve of per-sample $\lambda_{\max}(J)$"
        "\nHigher curve = heavier sharpness tail"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=args.dpi)
    plt.close()
    print("Saved:", save_path)


def save_metrics_table(results: List[Dict], save_path: str):
    rows = []

    for r in results:
        metrics = r["metrics"]
        lmax = r["lambda_max"]
        flat_sample = sample_array(r["flat"], 2_000_000)

        row = {
            "model": r["model_name"],
            "out_dir": r["out_dir"],
            "vae_config": metrics.get("vae_config", ""),
            "t": metrics.get("t", None),
            "topk": metrics.get("topk", None),
            "metric_norm": metrics.get("metric_norm", ""),
            "train_size": metrics.get("train_size", None),
            "val_size": metrics.get("val_size", None),
            "dim": metrics.get("dim", None),
            "base_eig": r["base_eig"],

            "eig_p50_sampled": float(np.percentile(flat_sample, 50)),
            "eig_p95_sampled": float(np.percentile(flat_sample, 95)),
            "eig_p99_sampled": float(np.percentile(flat_sample, 99)),
            "eig_p999_sampled": float(np.percentile(flat_sample, 99.9)),
            "positive_eigenvalue_ratio_sampled": float(np.mean(flat_sample > 0)),

            "lambda_max_p50": float(np.percentile(lmax, 50)),
            "lambda_max_p90": float(np.percentile(lmax, 90)),
            "lambda_max_p95": float(np.percentile(lmax, 95)),
            "lambda_max_p99": float(np.percentile(lmax, 99)),
            "lambda_max_p995": float(np.percentile(lmax, 99.5)),
            "lambda_max_p999": float(np.percentile(lmax, 99.9)),
            "lambda_max_max": float(np.max(lmax)),
            "positive_sample_ratio": float(np.mean(lmax > 0)),
        }

        # Include exact-global diagnostics if present.
        summary = r["summary"]
        for col in [
            "topk_mass",
            "tail_mass",
            "ess_exact",
            "trace_cov_exact",
            "trace_cov_topk",
            "trace_cov_tail_ratio",
        ]:
            if col in summary.columns:
                arr = summary[col].to_numpy()
                row[f"{col}_mean"] = float(np.mean(arr))
                row[f"{col}_p50"] = float(np.percentile(arr, 50))
                row[f"{col}_p95"] = float(np.percentile(arr, 95))
                row[f"{col}_p99"] = float(np.percentile(arr, 99))

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "out_dir"])
    df.to_csv(save_path, index=False)
    print("Saved:", save_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    parent_dir = args.parent_dir
    if not os.path.isdir(parent_dir):
        raise NotADirectoryError(parent_dir)

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(parent_dir, "_bn_unified_figures")
    os.makedirs(save_dir, exist_ok=True)

    result_dirs = find_result_dirs(parent_dir, recursive=not args.no_recursive)

    results = []
    for out_dir in result_dirs:
        try:
            metrics = load_json(os.path.join(out_dir, "metrics.json"))
            if not is_bn_result(out_dir, metrics):
                continue

            r = load_result(out_dir, parent_dir)
            results.append(r)

        except Exception as e:
            print(f"[Skip] {out_dir}: {repr(e)}")

    if len(results) == 0:
        raise RuntimeError("No BN result dirs found.")

    # Stable and readable order.
    preferred_order = {
        "SDVAE": 0,
        "FLUXVAE": 1,
        "VAVAE": 2,
        "VAVAE64": 3,
        "DCAE": 4,
        "SD3VAE": 5,
    }
    results = sorted(results, key=lambda r: (preferred_order.get(r["model_name"], 999), r["model_name"], r["out_dir"]))

    print("BN results:")
    for r in results:
        print("  -", r["model_name"], "|", r["out_dir"])

    # Auto-set eigenvalue tail xmax if not given.
    if args.tail_max is None:
        sampled_positive = []
        for r in results:
            sample = sample_array(r["flat"], args.axis_sample_size)
            pos = sample[sample > 0]
            if len(pos) > 0:
                sampled_positive.append(pos)

        if len(sampled_positive) > 0:
            all_pos = np.concatenate(sampled_positive)
            tail_max = np.percentile(all_pos, args.eig_tail_axis_percentile)
            args.tail_max = round_up_nice(float(tail_max))
            args.tail_max = max(args.tail_max, 200.0)
        else:
            args.tail_max = 200.0

    print(f"Using unified eig tail x-range: [{args.tail_min}, {args.tail_max}]")

    # Save metrics table first.
    save_metrics_table(
        results,
        os.path.join(save_dir, "bn_unified_metrics.csv"),
    )

    # Main teacher figures.
    plot_bulk_grid(
        results,
        os.path.join(save_dir, "bn_compare_bulk_grid.png"),
        args,
    )

    plot_eig_tail_logy_grid(
        results,
        os.path.join(save_dir, "bn_compare_eig_tail_logy_grid.png"),
        args,
    )

    plot_lambda_max_logtail_grid(
        results,
        os.path.join(save_dir, "bn_compare_lambda_max_logtail_grid.png"),
        args,
    )

    plot_lambda_max_survival(
        results,
        os.path.join(save_dir, "bn_compare_lambda_max_survival.png"),
        args,
    )

    print("\nDone.")
    print("Saved all unified BN figures to:", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--parent-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-recursive", action="store_true")

    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--density", action="store_true", default=True)

    # Bulk histogram shared axis.
    parser.add_argument("--bulk-min", type=float, default=-5.0)
    parser.add_argument("--bulk-max", type=float, default=5.0)
    parser.add_argument("--bulk-bins", type=int, default=300)

    # Tail eigenvalue histogram shared axis.
    parser.add_argument("--tail-min", type=float, default=-5.0)
    parser.add_argument("--tail-max", type=float, default=None)
    parser.add_argument("--tail-bins", type=int, default=400)
    parser.add_argument("--eig-tail-axis-percentile", type=float, default=99.5)

    # Positive lambda_max log-tail shared axis.
    parser.add_argument("--lambda-tail-bins", type=int, default=100)
    parser.add_argument("--lambda-log-min", type=float, default=None)
    parser.add_argument("--lambda-log-max", type=float, default=None)
    parser.add_argument("--lambda-log-axis-percentile", type=float, default=99.5)

    # Survival curve.
    parser.add_argument("--survival-xmin", type=float, default=1.0)
    parser.add_argument("--survival-xmax", type=float, default=None)
    parser.add_argument("--survival-axis-percentile", type=float, default=99.9)
    parser.add_argument("--survival-points", type=int, default=300)

    # Efficiency.
    parser.add_argument("--hist-chunk-size", type=int, default=5_000_000)
    parser.add_argument("--axis-sample-size", type=int, default=2_000_000)

    args = parser.parse_args()
    main(args)

'''
python score_jacobian_spectrum_uni.py \
  --parent-dir ./score_j
'''