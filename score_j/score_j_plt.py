import os
import json
import argparse
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_FILES = ["metrics.json", "eig_nontrivial.npy", "summary.csv"]


def safe_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def has_required_files(folder: str) -> bool:
    return all(os.path.exists(os.path.join(folder, f)) for f in REQUIRED_FILES)


def find_result_dirs(parent_dir: str, recursive: bool = True):
    result_dirs = []

    if recursive:
        for root, dirs, files in os.walk(parent_dir):
            if has_required_files(root):
                result_dirs.append(root)
    else:
        for name in sorted(os.listdir(parent_dir)):
            path = os.path.join(parent_dir, name)
            if os.path.isdir(path) and has_required_files(path):
                result_dirs.append(path)

    return sorted(result_dirs)


def load_metrics(out_dir):
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "r") as f:
        return json.load(f)


def get_exp_name(out_dir, parent_dir):
    rel = os.path.relpath(out_dir, parent_dir)
    return safe_name(rel)


def get_title_prefix(metrics, out_dir):
    vae_config = metrics.get("vae_config", "")
    vae_name = os.path.splitext(os.path.basename(vae_config))[0]
    if not vae_name:
        vae_name = os.path.basename(out_dir)

    t = metrics.get("t", "unknown")
    topk = metrics.get("topk", "unknown")
    metric_norm = metrics.get("metric_norm", "unknown")
    weight_mode = metrics.get("weight_mode", None)

    if weight_mode is not None:
        return f"{vae_name}, t={t}, topk={topk}, norm={metric_norm}, {weight_mode}"
    return f"{vae_name}, t={t}, topk={topk}, norm={metric_norm}"


def save_fig_and_optionally_collect(fig_path, collect_dir, exp_name):
    if collect_dir is None:
        return

    os.makedirs(collect_dir, exist_ok=True)
    base = os.path.basename(fig_path)
    dst = os.path.join(collect_dir, f"{exp_name}__{base}")
    shutil.copy2(fig_path, dst)


def process_one_dir(out_dir, parent_dir, args):
    print("\n" + "=" * 80)
    print(f"[Process] {out_dir}")

    metrics = load_metrics(out_dir)

    eig_path = os.path.join(out_dir, "eig_nontrivial.npy")
    summary_path = os.path.join(out_dir, "summary.csv")

    base_eig = float(metrics["base_eig"])
    title_prefix = get_title_prefix(metrics, out_dir)
    exp_name = get_exp_name(out_dir, parent_dir)

    fig_dir = os.path.join(out_dir, args.figure_subdir)
    os.makedirs(fig_dir, exist_ok=True)

    print("Loading eig_nontrivial.npy...")
    all_eigs = np.load(eig_path, mmap_mode="r")
    flat = all_eigs.reshape(-1)

    print("Loading summary.csv...")
    summary = pd.read_csv(summary_path)
    if "lambda_max_nontriv" not in summary.columns:
        raise KeyError(f"{summary_path} does not contain column lambda_max_nontriv")

    lambda_max = summary["lambda_max_nontriv"].to_numpy()

    # ------------------------------------------------------------
    # Scalar summaries
    # ------------------------------------------------------------
    extra_metrics = {
        "exp_name": exp_name,
        "out_dir": out_dir,
        "vae_config": metrics.get("vae_config", ""),
        "t": metrics.get("t", None),
        "t_eff": metrics.get("t_eff", None),
        "topk": metrics.get("topk", None),
        "metric_norm": metrics.get("metric_norm", ""),
        "weight_mode": metrics.get("weight_mode", ""),
        "train_size": metrics.get("train_size", None),
        "val_size": metrics.get("val_size", None),
        "latent_shape": str(metrics.get("latent_shape", "")),
        "dim": metrics.get("dim", None),
        "base_eig": base_eig,

        "eig_p50": float(np.percentile(flat, 50)),
        "eig_p95": float(np.percentile(flat, 95)),
        "eig_p99": float(np.percentile(flat, 99)),
        "eig_p999": float(np.percentile(flat, 99.9)),
        "eig_p9999": float(np.percentile(flat, 99.99)),
        "eig_min": float(np.min(flat)),
        "eig_max": float(np.max(flat)),
        "positive_eigenvalue_ratio": float(np.mean(flat > 0)),

        "lambda_max_p50": float(np.percentile(lambda_max, 50)),
        "lambda_max_p90": float(np.percentile(lambda_max, 90)),
        "lambda_max_p95": float(np.percentile(lambda_max, 95)),
        "lambda_max_p99": float(np.percentile(lambda_max, 99)),
        "lambda_max_p995": float(np.percentile(lambda_max, 99.5)),
        "lambda_max_p999": float(np.percentile(lambda_max, 99.9)),
        "lambda_max_max": float(lambda_max.max()),
        "positive_sample_ratio": float(np.mean(lambda_max > 0)),
    }

    # If exact_global version has these fields, also summarize them.
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
            extra_metrics[f"{col}_mean"] = float(np.mean(arr))
            extra_metrics[f"{col}_p50"] = float(np.percentile(arr, 50))
            extra_metrics[f"{col}_p95"] = float(np.percentile(arr, 95))
            extra_metrics[f"{col}_p99"] = float(np.percentile(arr, 99))

    print("========== Key metrics ==========")
    print("positive_eigenvalue_ratio:", extra_metrics["positive_eigenvalue_ratio"])
    print("positive_sample_ratio:", extra_metrics["positive_sample_ratio"])
    print("lambda_max_p99:", extra_metrics["lambda_max_p99"])
    print("lambda_max_p999:", extra_metrics["lambda_max_p999"])

    # ------------------------------------------------------------
    # Figure 1: local bulk histogram
    # ------------------------------------------------------------
    fig_path = os.path.join(fig_dir, "hist_bulk_local_minus5_5.png")

    if args.overwrite or not os.path.exists(fig_path):
        plt.figure(figsize=(8, 5))
        plt.hist(flat, bins=args.bulk_bins, range=(args.bulk_min, args.bulk_max))
        plt.axvline(base_eig, linestyle="--", label=f"base eig = {base_eig:.2f}")
        plt.axvline(0.0, linestyle=":", label="zero")
        plt.title(f"Bulk eigenvalue histogram of J\n{title_prefix}")
        plt.xlabel("eigenvalue")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=args.dpi)
        plt.close()
        print("Saved:", fig_path)

    save_fig_and_optionally_collect(fig_path, args.collect_dir, exp_name)

    # ------------------------------------------------------------
    # Figure 2: log-y tail histogram
    # ------------------------------------------------------------
    fig_path = os.path.join(fig_dir, "hist_logy_tail_minus5_200.png")

    if args.overwrite or not os.path.exists(fig_path):
        plt.figure(figsize=(8, 5))
        plt.hist(flat, bins=args.tail_bins, range=(args.tail_min, args.tail_max))
        plt.yscale("log")
        plt.axvline(base_eig, linestyle="--", label=f"base eig = {base_eig:.2f}")
        plt.axvline(0.0, linestyle=":", label="zero")
        plt.title(f"Eigenvalue histogram with log-scale count\n{title_prefix}")
        plt.xlabel("eigenvalue")
        plt.ylabel("count, log scale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=args.dpi)
        plt.close()
        print("Saved:", fig_path)

    save_fig_and_optionally_collect(fig_path, args.collect_dir, exp_name)

    # ------------------------------------------------------------
    # Figure 3: positive lambda_max log10 tail
    # ------------------------------------------------------------
    positive_lmax = lambda_max[lambda_max > 0]
    fig_path = os.path.join(fig_dir, "hist_lambda_max_logtail.png")

    if len(positive_lmax) > 0:
        if args.overwrite or not os.path.exists(fig_path):
            plt.figure(figsize=(8, 5))
            plt.hist(np.log10(positive_lmax + 1e-8), bins=args.lambda_tail_bins)
            plt.title(f"Positive lambda_max log10 tail\n{title_prefix}")
            plt.xlabel("log10(lambda_max)")
            plt.ylabel("number of samples")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=args.dpi)
            plt.close()
            print("Saved:", fig_path)

        save_fig_and_optionally_collect(fig_path, args.collect_dir, exp_name)
    else:
        print("[Skip] No positive lambda_max samples.")

    # ------------------------------------------------------------
    # Optional: per-sample lambda_max histogram, not for teacher by default.
    # ------------------------------------------------------------
    if args.save_lambda_max_hist:
        right = np.percentile(lambda_max, args.lambda_max_plot_percentile)
        right = max(right, 1.0)

        fig_path = os.path.join(fig_dir, "hist_lambda_max_per_sample.png")

        if args.overwrite or not os.path.exists(fig_path):
            plt.figure(figsize=(8, 5))
            plt.hist(lambda_max, bins=300, range=(min(lambda_max.min(), base_eig), right))
            plt.axvline(0.0, linestyle=":", label="zero")
            plt.title(f"Per-sample lambda_max distribution\n{title_prefix}")
            plt.xlabel("lambda_max_nontriv")
            plt.ylabel("number of samples")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=args.dpi)
            plt.close()
            print("Saved:", fig_path)

        save_fig_and_optionally_collect(fig_path, args.collect_dir, exp_name)

    # ------------------------------------------------------------
    # Save per-folder extra metrics
    # ------------------------------------------------------------
    save_path = os.path.join(out_dir, "extra_spectrum_metrics.json")
    with open(save_path, "w") as f:
        json.dump(extra_metrics, f, indent=2)

    print("Saved:", save_path)

    return extra_metrics


def main(args):
    parent_dir = args.parent_dir
    if not os.path.isdir(parent_dir):
        raise NotADirectoryError(parent_dir)

    if args.collect_dir is None:
        args.collect_dir = os.path.join(parent_dir, "_teacher_figures")
    elif args.collect_dir.lower() in ["none", "null", "false"]:
        args.collect_dir = None

    if args.collect_dir is not None:
        os.makedirs(args.collect_dir, exist_ok=True)

    result_dirs = find_result_dirs(parent_dir, recursive=not args.no_recursive)

    print(f"Found {len(result_dirs)} result dirs under: {parent_dir}")
    for d in result_dirs:
        print("  -", d)

    all_rows = []

    for out_dir in result_dirs:
        try:
            row = process_one_dir(out_dir, parent_dir, args)
            all_rows.append(row)
        except Exception as e:
            print(f"[Error] Failed to process {out_dir}: {repr(e)}")

    if len(all_rows) > 0:
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(parent_dir, "all_extra_spectrum_metrics.csv")
        df.to_csv(csv_path, index=False)
        print("\nSaved summary CSV:", csv_path)

        json_path = os.path.join(parent_dir, "all_extra_spectrum_metrics.json")
        with open(json_path, "w") as f:
            json.dump(all_rows, f, indent=2)
        print("Saved summary JSON:", json_path)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--parent-dir", type=str, required=True)
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan immediate child folders, not recursively.",
    )

    parser.add_argument(
        "--figure-subdir",
        type=str,
        default="teacher_figures",
        help="Subfolder inside each result dir to save figures.",
    )
    parser.add_argument(
        "--collect-dir",
        type=str,
        default=None,
        help="Folder to collect copied figures. Default: parent_dir/_teacher_figures. Use 'none' to disable.",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)

    parser.add_argument("--bulk-min", type=float, default=-5.0)
    parser.add_argument("--bulk-max", type=float, default=5.0)
    parser.add_argument("--bulk-bins", type=int, default=300)

    parser.add_argument("--tail-min", type=float, default=-5.0)
    parser.add_argument("--tail-max", type=float, default=200.0)
    parser.add_argument("--tail-bins", type=int, default=300)

    parser.add_argument("--lambda-tail-bins", type=int, default=100)

    parser.add_argument(
        "--save-lambda-max-hist",
        action="store_true",
        help="Also save per-sample lambda_max histogram.",
    )
    parser.add_argument("--lambda-max-plot-percentile", type=float, default=99.5)

    args = parser.parse_args()
    main(args)

'''
python score_j_plt.py \
  --parent-dir ./score_j \
  --overwrite
'''