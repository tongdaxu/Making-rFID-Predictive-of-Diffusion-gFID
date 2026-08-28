import pandas as pd
import math

import numpy as np
from scipy.stats import rankdata


def _paired_finite(data, target_name, metric_name):
    """Extract paired finite observations."""
    target = data[target_name].to_numpy(dtype=float)
    metric = data[metric_name].to_numpy(dtype=float)

    mask = np.isfinite(target) & np.isfinite(metric)
    return target[mask], metric[mask]


def pearson_correlation(x, y):
    """Pearson correlation coefficient."""
    if len(x) < 2:
        return np.nan

    x = x - np.mean(x)
    y = y - np.mean(y)

    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denominator == 0:
        return np.nan

    return np.dot(x, y) / denominator


def spearman_correlation(x, y):
    """
    Spearman rank correlation.

    Average ranks are used for ties.
    """
    if len(x) < 2:
        return np.nan

    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")

    return pearson_correlation(x_rank, y_rank)


def kendall_tau_b(x, y):
    """
    Kendall's tau-b.

    Tau-b accounts for ties in either variable.
    """
    n = len(x)
    if n < 2:
        return np.nan

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]

        concordant += np.sum((dx * dy) > 0)
        discordant += np.sum((dx * dy) < 0)

        # Tied only in x or only in y.
        ties_x += np.sum((dx == 0) & (dy != 0))
        ties_y += np.sum((dy == 0) & (dx != 0))

    denominator = np.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )

    if denominator == 0:
        return np.nan

    return (concordant - discordant) / denominator


def _top_k_indices(values, k, higher_is_better):
    """Indices of the best k values."""
    k = min(k, len(values))

    if higher_is_better:
        return np.argsort(-values, kind="stable")[:k]

    return np.argsort(values, kind="stable")[:k]


def top_k_overlap(
    target,
    metric,
    k,
    target_higher_is_better=False,
    metric_higher_is_better=False,
):
    """
    Fraction of ground-truth top-k models recovered by the metric.

    Returns a number in [0, 1].
    """
    if len(target) == 0 or k <= 0:
        return np.nan

    k = min(k, len(target))

    target_top_k = _top_k_indices(
        target, k, target_higher_is_better
    )
    metric_top_k = _top_k_indices(
        metric, k, metric_higher_is_better
    )

    overlap = np.intersect1d(target_top_k, metric_top_k).size
    return overlap / k


def selection_regret(
    target,
    metric,
    target_higher_is_better=False,
    metric_higher_is_better=False,
):
    """
    Ground-truth regret caused by selecting the proxy metric's best model.

    For a lower-is-better target:
        regret = target[selected_by_metric] - min(target)

    For a higher-is-better target:
        regret = max(target) - target[selected_by_metric]

    A perfect selection has regret 0.
    """
    if len(target) == 0:
        return np.nan

    if metric_higher_is_better:
        selected_index = np.argmax(metric)
    else:
        selected_index = np.argmin(metric)

    selected_target = target[selected_index]

    if target_higher_is_better:
        optimal_target = np.max(target)
        return optimal_target - selected_target

    optimal_target = np.min(target)
    return selected_target - optimal_target


def evaluate_metric(
    data,
    target_name,
    metric_name,
    k=5,
    target_higher_is_better=False,
    metric_higher_is_better=False,
):
    """
    Evaluate one proxy metric against a ground-truth target.
    """
    target, metric = _paired_finite(
        data, target_name, metric_name
    )

    return {
        "metric": metric_name,
        "num_samples": len(target),
        "pearson": pearson_correlation(target, metric),
        "spearman": spearman_correlation(target, metric),
        "kendall_tau": kendall_tau_b(target, metric),
        f"top_{k}_overlap": top_k_overlap(
            target,
            metric,
            k=k,
            target_higher_is_better=target_higher_is_better,
            metric_higher_is_better=metric_higher_is_better,
        ),
        "selection_regret": selection_regret(
            target,
            metric,
            target_higher_is_better=target_higher_is_better,
            metric_higher_is_better=metric_higher_is_better,
        ),
    }

data = pd.DataFrame({
    'vae_name': ['SD-VAE', 'SD3-VAE', 'FLUX-VAE', 'DC-AE ', 'SOFT-VQ', 'VA-VAE', 'VA-VAE-64', 'EQ-VAE', 'MAE-TOK', 'IN-VAE', 'REPAE-SDVAE', 'DE-TOK', 'QwenImg-VAE', 'RAE', 'SVG', 'FLUX2-VAE', 'SVG T2I', 'DM-VAE', 'VTP-S', 'VTP-B', 'VTP-L', ' PAE DINOv2', math.nan, math.nan, math.nan, math.nan],
    'sit_b_w_o_cfg_gfid': [46.74, 51.39, 63.32, 42.01, 29.11, 17.63, 32.51, 37.81, 13.73, 49.17, 26.06, 20.27, 48.34, math.nan, 18.63, 22.75, 10.69, 8.69, 16.72, 16.24, 15.69, 14.15, math.nan, math.nan, math.nan, math.nan],
    'sit_b_w_o_cfg_is': [30.43, 27.61, 22.22, 32.79, 47.19, 64.56, 39.13, 37.24, 92.7, 28.79, 52.01, 62.09, 29.1, math.nan, 78.27, 57.02, 102.98, 106.01, 69.04, 76.37, 84.72, 86.02, math.nan, math.nan, math.nan, math.nan],
    'sit_b_w_o_cfg_prec': [0.47, 0.43, 0.37, 0.49, 0.58, 0.66, 0.56, 0.54, 0.67, 0.46, 0.61, 0.66, 0.44, math.nan, 0.68, 0.62, 0.78, 0.76, 0.7, 0.7, 0.69, 0.69, math.nan, math.nan, math.nan, math.nan],
    'sit_b_w_o_cfg_rec': [0.63, 0.6, 0.57, 0.62, 0.59, 0.57, 0.59, 0.62, 0.58, 0.61, 0.61, 0.6, 0.59, math.nan, 0.51, 0.62, 0.52, 0.57, 0.58, 0.59, 0.57, 0.61, math.nan, math.nan, math.nan, math.nan],
    'sit_b_w_cfg_gfid': [9.91, 11.89, 14.38, 9.03, 7.48, 6.01, 7.76, 9.32, 5.69, 10.87, 6.46, 6.95, 10.97, math.nan, 8.66, 5.92, 6.44, 4.89, 6.07, 5.56, 5.5, 4.58, math.nan, math.nan, math.nan, math.nan],
    'sit_b_w_cfg_is': [162.9, 151.9, 134.72, 189.42, 217.76, 208.61, 208.07, 177.31, 237.12, 160.47, 176.37, 197.12, 148.86, math.nan, 233.68, 185.76, 155.28, 174.05, 213.12, 220.22, 232.28, 250.76, math.nan, math.nan, math.nan, math.nan],
    'sit_b_w_cfg_prec': [0.84, 0.82, 0.79, 0.85, 0.83, 0.84, 0.84, 0.88, 0.82, 0.84, 0.85, 0.88, 0.82, math.nan, 0.81, 0.84, 0.84, 0.84, 0.87, 0.85, 0.85, 0.86, math.nan, math.nan, math.nan, math.nan],
    'sit_b_w_cfg_rec': [0.32, 0.3, 0.26, 0.33, 0.36, 0.42, 0.32, 0.32, 0.45, 0.3, 0.43, 0.4, 0.3, math.nan, 0.33, 0.45, 0.46, 0.5, 0.43, 0.44, 0.43, 0.46, math.nan, math.nan, math.nan, math.nan],
    'sit_xl_w_o_cfg_gfid': [25.91, 26.38, 34.06, 26.68, 15.88, 8.57, 15.09, 20.81, 6.27, 25.65, 12.95, 11.97, 23.62, 4.25, 7.65, 10.18, 5.63, 4.65, 9.55, 7.64, 5.79, 5.65, math.nan, math.nan, math.nan, math.nan],
    'sit_xl_w_o_cfg_is': [53.9, 50.39, 41.57, 50.91, 76.09, 100.79, 73.06, 63.14, 132.73, 52.42, 85.11, 89.24, 55.86, 192.71, 127.75, 96.68, 143.94, 20.0, 98.19, 119.41, 140.61, 135.82, math.nan, math.nan, math.nan, math.nan],
    'sit_xl_w_o_cfg_prec': [0.6, 0.59, 0.54, 0.58, 0.67, 0.73, 0.68, 0.65, 0.75, 0.6, 0.7, 0.73, 0.61, 0.85, 0.76, 0.72, 0.82, 0.8, 0.75, 0.76, 0.78, 0.77, math.nan, math.nan, math.nan, math.nan],
    'sit_xl_w_o_cfg_rec': [0.62, 0.62, 0.63, 0.64, 0.6, 0.57, 0.58, 0.6, 0.6, 0.62, 0.6, 0.59, 0.61, 0.51, 0.56, 0.61, 0.54, 0.56, 0.59, 0.59, 0.59, 0.6, math.nan, math.nan, math.nan, math.nan],
    'sit_xl_w_cfg_gfid': [6.33, 7.16, 9.51, 6.08, 5.14, 4.2, 4.95, 6.24, 3.74, 6.56, 4.17, 4.83, 7.25, 3.5, 4.79, 4.1, 4.37, 3.39, 4.58, 4.02, 3.48, 2.85, math.nan, math.nan, math.nan, math.nan],
    'sit_xl_w_cfg_is': [178.64, 189.95, 160.82, 179.91, 240.19, 238.73, 183.03, 201.39, 199.88, 167.17, 211.95, 207.96, 180.18, 256.41, 247.56, 229.93, 216.93, 220.37, 223.93, 245.77, 211.17, 214.77, math.nan, math.nan, math.nan, math.nan],
    'sit_xl_w_cfg_prec': [0.84, 0.87, 0.84, 0.83, 0.86, 0.86, 0.83, 0.88, 0.81, 0.84, 0.86, 0.88, 0.85, 0.88, 0.85, 0.87, 0.87, 0.87, 0.87, 0.86, 0.83, 0.83, math.nan, math.nan, math.nan, math.nan],
    'sit_xl_w_cfg_rec': [0.43, 0.36, 0.38, 0.45, 0.44, 0.45, 0.47, 0.39, 0.54, 0.44, 0.47, 0.45, 0.41, 0.45, 0.46, 0.49, 0.48, 0.51, 0.47, 0.48, 0.53, 0.56, math.nan, math.nan, math.nan, math.nan],
    '-reconstruction_psnr': [-26.91, -31.28, -32.87, -24.55, -23.92, -27.71, -30.72, -26.16, -24.57, -28.92, -26.1, -25.24, -32.19, -19.21, -23.35, -34.33, -18.9, -22.46, -24.17, -26.32, -26.85, -27.28, math.nan, math.nan, math.nan, math.nan],
    'reconstruction_lpips': [0.06, 0.02, 0.01, 0.08, 0.09, 0.04, 0.02, 0.06, 0.09, 0.03, 0.06, 0.06, 0.04, 0.15, 0.1, 0.01, 0.16, 0.11, 0.1, 0.07, 0.05, 0.05, math.nan, math.nan, math.nan, math.nan],
    '-reconstruction_ssim': [-0.76, -0.89, -0.92, 0.65, -0.66, -0.8, -0.88, -0.74, -0.68, -0.83, -0.74, -0.71, -0.9, -0.5, -0.65, -0.94, -0.49, -0.63, -0.67, -0.75, -0.77, -0.79, math.nan, math.nan, math.nan, math.nan],
    'reconstruction_rfid': [0.7, 0.19, 0.18, 0.84, 0.59, 0.28, 0.14, 0.55, 0.58, 0.26, 0.53, 0.54, 1.51, 0.62, 0.68, 0.15, 1.14, 0.68, 1.17, 0.69, 0.35, 0.26, math.nan, math.nan, math.nan, math.nan],
    'equvariance_eq_loss': [0.67, 0.63, 0.69, 0.89, 1.16, 1.28, 0.91, 0.32, 0.27, 0.78, 0.91, 0.67, 0.6, 0.41, 0.78, 0.78, 0.74, 0.66, 0.78, 0.78, 0.8, 0.93, math.nan, math.nan, math.nan, math.nan],
    'equvariance_se_loss': [0.28, 0.18, 0.17, 0.35, 0.25, 0.29, 0.2, 0.22, 0.26, 0.23, 0.32, 0.19, 0.15, 0.39, 0.32, 0.17, 0.44, 0.29, 0.37, 0.3, 0.26, 0.28, math.nan, math.nan, math.nan, math.nan],
    'rank_of_data_gmm_loss': [10.67, 8.79, 18.06, 11.5, -25.73, -16.2, -6.21, 12.52, -11.46, 9.02, 34.72, 15.11, 12.69, -36.55, -8.92, 70.53, -12.57, 19.34, -5.12, -0.31, 7.7, -16.19, math.nan, math.nan, math.nan, math.nan],
    'rank_of_data_viv': [2.8, 1.02, 1.1, 1.36, math.nan, 0.57, 0.6, 2.13, math.nan, 1.22, 2.9, math.nan, 0.9, 0.78, 0.5, 1.3, 0.5, math.nan, 1.15, 1.15, 1.11, 1.13, math.nan, math.nan, math.nan, math.nan],
    'spatial_smoothness_lds': [0.103, 0.083, 0.122, 0.116, math.nan, 0.295, 0.27, 0.27, math.nan, 0.105, 0.107, math.nan, 0.122, 0.112, 0.097, 0.098, 0.097, math.nan, 0.071, 0.062, 0.06, 0.021, math.nan, math.nan, math.nan, math.nan],
    'spatial_smoothness_lds_1': [0.3, 0.234, 0.368, 0.255, math.nan, 0.676, 0.593, 0.62, math.nan, 0.322, 0.558, math.nan, 0.369, 0.29, 0.21, 0.341, 0.209, math.nan, 0.238, 0.196, 0.185, 0.044, math.nan, math.nan, math.nan, math.nan],
    'spatial_smoothness_cds': [0.029, 0.043, 0.035, 0.027, 0.09, 0.094, 0.08, 0.063, 0.002, 0.038, 0.063, 0.011, 0.049, 0.045, 0.036, 0.043, 0.036, 0.051, 0.042, 0.037, 0.032, 0.033, math.nan, math.nan, math.nan, math.nan],
    'spatial_smoothness_sec@05': [0.209, 0.088, 0.134, 0.245, 0.107, 0.047, 0.027, 0.01, 0.436, 0.2, 0.103, 0.244, 0.053, 0.112, 0.043, 0.24, 0.041, 0.184, 0.189, 0.2, 0.211, 0.185, math.nan, math.nan, math.nan, math.nan],
    'semantic_alignment_vf_loss': [0.64, 0.36, 0.41, 0.35, 0.18, 0.08, 0.09, 0.59, 0.33, 0.4, 0.65, 0.46, 0.36, 0.01, 0.06, 0.34, 0.06, 0.22, 0.21, 0.22, 0.22, 0.2, math.nan, math.nan, math.nan, math.nan],
    'semantic_alignment_acc_1_avg': [0.76, 0.58, 0.91, 2.23, 0.97, 17.25, 12.36, 1.14, 3.92, 3.02, 1.15, 4.96, 0.83, 73.58, 45.83, 6.62, 45.99, 25.38, 2.44, 2.95, 6.9, 13.69, math.nan, math.nan, math.nan, math.nan],
    'semantic_alignment_acc_5_avg': [2.62, 2.34, 3.37, 7.53, 3.31, 38.54, 31.35, 4.24, 13.69, 9.68, 4.1, 14.35, 3.1, 92.84, 75.85, 17.91, 75.91, 44.78, 7.77, 9.09, 19.31, 32.35, math.nan, math.nan, math.nan, math.nan],
    'acc_1_max': [0.34, 0.42, 1.24, 2.31, 0.14, 14.33, 14.58, 0.56, 4.51, 2.78, 0.4, 3.19, math.nan, 75.45, 49.73, 5.01, 49.94, 15.7, 0.22, 0.3, 0.92, 9.16, math.nan, math.nan, math.nan, math.nan],
    'acc_5_max': [1.65, 1.626, 4.252, 7.724, 0.68, 34.14, 35.12, 2.33, 14.67, 8.23, 1.62, 10.17, math.nan, 93.63, 78.51, 14.22, 78.84, 32.66, 1.748, 1.602, 2.744, 24.712, math.nan, math.nan, math.nan, math.nan],
    'lds_bn': [0.3, 0.433, 0.365, 0.232, math.nan, 0.699, 0.618, 0.627, math.nan, 0.313, 0.559, math.nan, 0.49, 0.357, 0.281, 0.342, 0.279, math.nan, 0.333, 0.3, 0.254, 0.285, math.nan, math.nan, math.nan, math.nan],
    'ifid_bn': [68.05, 46.1, 70.52, 45.12, 32.66, 25.01, 23.04, 54.18, 18.26, 47.35, 47.02, 21.05, 36.22, 6.05, 7.17, 19.13, 5.12, 12.91, 20.7, 22.92, 25.85, 22.07, math.nan, math.nan, math.nan, math.nan],
    'iis': [23.41, 40.38, 21.46, 28.33, 43.0, 55.45, 51.53, 27.02, 67.39, 31.87, 30.53, 46.18, 42.35, 142.81, 117.78, 55.32, 129.45, 76.99747467, 50.82, 52.41, 54.02, 55.03, math.nan, math.nan, math.nan, math.nan],
    'iprec': [0.25, 0.42, 0.23, 0.34, 0.47, 0.54, 0.53, 0.32, 0.6, 0.33, 0.41, 0.49, 0.43, 0.75, 0.73, 0.54, 0.79, 0.65392, 0.51, 0.49, 0.49, 0.53, math.nan, math.nan, math.nan, math.nan],
    'irec': [0.67, 0.77, 0.67, 0.69, 0.57, 0.63, 0.67, 0.69, 0.59, 0.71, 0.62, 0.72, 0.8, 0.69, 0.68, 0.7, 0.68, 0.69658, 0.64, 0.62, 0.57, 0.57, math.nan, math.nan, math.nan, math.nan],
    'yymm': [21.12, 24.03, 24.08, 24.1, 24.12, 25.01, 25.01, 25.02, 25.02, 25.04, 25.04, 25.07, 25.08, 25.1, 25.1, 25.11, 25.12, 25.12, 25.12, 25.12, 25.12, 26.05, math.nan, math.nan, math.nan, math.nan],
    'param_m': [83.65, 83.82, 83.82, 312.25, 608.92, 69.83, 70.29, 83.65, 173.89, 69.83, 83.65, 171.71, 126.89, 501.93, 82.96, 84.04, 71.73, 355.2, 167.17, 295.63, 731.57, 721.52, math.nan, math.nan, math.nan, math.nan],
    'dim': ['4x32x32', '16x32x32', '16x32x32', '32x8x8', '64x32', '32x16x16', '64x16x16', '4x32x32', '128x32', '32x16x16', '4x32x32', '128x32', '16x32x32', '768x16x16', '392x16x16', '128x16x16', '384x16x16', '256x32', '64x16x16', '64x16x16', '64x16x16', '32x16x16', math.nan, math.nan, math.nan, math.nan],
    'token': ['16x16', '16x16', '16x16', '8x8', 64.0, '16x16', '16x16', '16x16', 128.0, '16x16', '16x16', 128.0, '16x16', '16x16', '16x16', '16x16', '16x16', 256.0, '16x16', '16x16', '16x16', '16x16', math.nan, math.nan, math.nan, math.nan],
    'source': ['https://arxiv.org/abs/2112.10752', 'https://arxiv.org/abs/2403.03206', 'https://bfl.ai/blog/24-08-01-bfl', 'https://arxiv.org/abs/2410.10733', 'https://arxiv.org/abs/2412.10958', 'https://arxiv.org/abs/2501.01423', 'https://arxiv.org/abs/2501.01423', 'https://arxiv.org/abs/2502.09509', 'https://arxiv.org/abs/2502.03444', 'https://arxiv.org/abs/2504.10483', 'https://arxiv.org/abs/2504.10483', 'https://arxiv.org/html/2507.15856v1', 'https://arxiv.org/abs/2508.02324', 'https://arxiv.org/abs/2510.11690', 'https://arxiv.org/abs/2510.15301', 'https://bfl.ai/blog/flux-2', 'https://arxiv.org/abs/2512.11749', 'https://arxiv.org/abs/2512.07778', 'https://arxiv.org/abs/2512.13687', 'https://arxiv.org/abs/2512.13687', 'https://arxiv.org/abs/2512.13687', 'https://arxiv.org/abs/2605.07915', math.nan, math.nan, math.nan, math.nan],
})

datanew = pd.DataFrame({
    'vae_name': ['SD-VAE', 'SD3-VAE', 'FLUX-VAE', 'DC-AE', 'SOFT-VQ', 'VA-VAE', 'VA-VAE-64', 'EQ-VAE', 'MAE-TOK', 'IN-VAE', 'REPAE-SDVAE', 'DE-TOK', 'QwenImg-VAE', 'RAE', 'SVG', 'FLUX2-VAE', 'SVG T2I', 'DM-VAE', 'VTP-S', 'VTP-B', 'VTP-L', 'PAE DINOv2'],
    'sit_b_w_o_cfg_gfid': [46.74, 51.39, 63.32, 42.01, 29.11, 17.63, 32.51, 37.81, 13.73, 49.17, 26.06, 20.27, 48.34, math.nan, 18.63, 22.75, 10.69, 8.69, 16.72, 16.24, 15.69, 14.15],
    'sit_b_w_o_cfg_is': [30.43, 27.61, 22.22, 32.79, 47.19, 64.56, 39.13, 37.24, 92.7, 28.79, 52.01, 62.09, 29.1, math.nan, 78.27, 57.02, 102.98, 106.01, 69.04, 76.37, 84.72, 86.02],
    'sit_b_w_o_cfg_prec': [0.47, 0.43, 0.37, 0.49, 0.58, 0.66, 0.56, 0.54, 0.67, 0.46, 0.61, 0.66, 0.44, math.nan, 0.68, 0.62, 0.78, 0.76, 0.7, 0.7, 0.69, 0.69],
    'sit_b_w_o_cfg_rec': [0.63, 0.6, 0.57, 0.62, 0.59, 0.57, 0.59, 0.62, 0.58, 0.61, 0.61, 0.6, 0.59, math.nan, 0.51, 0.62, 0.52, 0.57, 0.58, 0.59, 0.57, 0.61],
    'sit_b_w_cfg_gfid': [9.91, 11.89, 14.38, 9.03, 7.48, 6.01, 7.76, 9.32, 5.69, 10.87, 6.46, 6.95, 10.97, math.nan, 8.66, 5.92, 6.44, 4.89, 6.07, 5.56, 5.5, 4.58],
    'sit_b_w_cfg_is': [162.9, 151.9, 134.72, 189.42, 217.76, 208.61, 208.07, 177.31, 237.12, 160.47, 176.37, 197.12, 148.86, math.nan, 233.68, 185.76, 155.28, 174.05, 213.12, 220.22, 232.28, 250.76],
    'sit_b_w_cfg_prec': [0.84, 0.82, 0.79, 0.85, 0.83, 0.84, 0.84, 0.88, 0.82, 0.84, 0.85, 0.88, 0.82, math.nan, 0.81, 0.84, 0.84, 0.84, 0.87, 0.85, 0.85, 0.86],
    'sit_b_w_cfg_rec': [0.32, 0.3, 0.26, 0.33, 0.36, 0.42, 0.32, 0.32, 0.45, 0.3, 0.43, 0.4, 0.3, math.nan, 0.33, 0.45, 0.46, 0.5, 0.43, 0.44, 0.43, 0.46],
    'sit_xl_w_o_cfg_gfid': [25.91, 26.38, 34.06, 26.68, 15.88, 8.57, 15.09, 20.81, 6.27, 25.65, 12.95, 11.97, 23.62, 4.25, 7.65, 10.18, 5.63, 4.65, 9.55, 7.64, 5.79, 5.65],
    'sit_xl_w_o_cfg_is': [53.9, 50.39, 41.57, 50.91, 76.09, 100.79, 73.06, 63.14, 132.73, 52.42, 85.11, 89.24, 55.86, 192.71, 127.75, 96.68, 143.94, 20.0, 98.19, 119.41, 140.61, 135.82],
    'sit_xl_w_o_cfg_prec': [0.6, 0.59, 0.54, 0.58, 0.67, 0.73, 0.68, 0.65, 0.75, 0.6, 0.7, 0.73, 0.61, 0.85, 0.76, 0.72, 0.82, 0.8, 0.75, 0.76, 0.78, 0.77],
    'sit_xl_w_o_cfg_rec': [0.62, 0.62, 0.63, 0.64, 0.6, 0.57, 0.58, 0.6, 0.6, 0.62, 0.6, 0.59, 0.61, 0.51, 0.56, 0.61, 0.54, 0.56, 0.59, 0.59, 0.59, 0.6],
    'sit_xl_w_cfg_gfid': [6.33, 7.16, 9.51, 6.08, 5.14, 4.2, 4.95, 6.24, 3.74, 6.56, 4.17, 4.83, 7.25, 3.5, 4.79, 4.1, 4.37, 3.39, 4.58, 4.02, 3.48, 2.85],
    'sit_xl_w_cfg_is': [178.64, 189.95, 160.82, 179.91, 240.19, 238.73, 183.03, 201.39, 199.88, 167.17, 211.95, 207.96, 180.18, 256.41, 247.56, 229.93, 216.93, 220.37, 223.93, 245.77, 211.17, 214.77],
    'sit_xl_w_cfg_prec': [0.84, 0.87, 0.84, 0.83, 0.86, 0.86, 0.83, 0.88, 0.81, 0.84, 0.86, 0.88, 0.85, 0.88, 0.85, 0.87, 0.87, 0.87, 0.87, 0.86, 0.83, 0.83],
    'sit_xl_w_cfg_rec': [0.43, 0.36, 0.38, 0.45, 0.44, 0.45, 0.47, 0.39, 0.54, 0.44, 0.47, 0.45, 0.41, 0.45, 0.46, 0.49, 0.48, 0.51, 0.47, 0.48, 0.53, 0.56],
    'reconstruction_psnr': [-26.91, -31.28, -32.87, -24.55, -23.92, -27.71, -30.72, -26.16, -24.57, -28.92, -26.1, -25.24, -32.19, -19.21, -23.35, -34.33, -18.9, -22.46, -24.17, -26.32, -26.85, -27.28],
    'reconstruction_lpips': [0.06, 0.02, 0.01, 0.08, 0.09, 0.04, 0.02, 0.06, 0.09, 0.03, 0.06, 0.06, 0.04, 0.15, 0.1, 0.01, 0.16, 0.11, 0.1, 0.07, 0.05, 0.05],
    'reconstruction_ssim': [-0.76, -0.89, -0.92, 0.65, -0.66, -0.8, -0.88, -0.74, -0.68, -0.83, -0.74, -0.71, -0.9, -0.5, -0.65, -0.94, -0.49, -0.63, -0.67, -0.75, -0.77, -0.79],
    'reconstruction_rfid': [0.7, 0.19, 0.18, 0.84, 0.59, 0.28, 0.14, 0.55, 0.58, 0.26, 0.53, 0.54, 1.51, 0.62, 0.68, 0.15, 1.14, 0.68, 1.17, 0.69, 0.35, 0.26],
    'equvariance_eq_loss': [0.67, 0.63, 0.69, 0.89, 1.16, 1.28, 0.91, 0.32, 0.27, 0.78, 0.91, 0.67, 0.6, 0.41, 0.78, 0.78, 0.74, 0.66, 0.78, 0.78, 0.8, 0.93],
    'equvariance_se_loss': [0.28, 0.18, 0.17, 0.35, 0.25, 0.29, 0.2, 0.22, 0.26, 0.23, 0.32, 0.19, 0.15, 0.39, 0.32, 0.17, 0.44, 0.29, 0.37, 0.3, 0.26, 0.28],
    'rank_of_data_gmm_loss': [10.67, 8.79, 18.06, 11.5, -25.73, -16.2, -6.21, 12.52, -11.46, 9.02, 34.72, 15.11, 12.69, -36.55, -8.92, 70.53, -12.57, 19.34, -5.12, -0.31, 7.7, -16.19],
    'rank_of_data_viv': [2.8, 1.02, 1.1, 1.36, math.nan, 0.57, 0.6, 2.13, math.nan, 1.22, 2.9, math.nan, 0.9, 0.78, 0.5, 1.3, 0.5, math.nan, 1.15, 1.15, 1.11, 1.13],
    'spatial_smoothness_lds': [0.103, 0.083, 0.122, 0.116, math.nan, 0.295, 0.27, 0.27, math.nan, 0.105, 0.107, math.nan, 0.122, 0.112, 0.097, 0.098, 0.097, math.nan, 0.071, 0.062, 0.06, 0.021],
    'spatial_smoothness_lds_1': [0.3, 0.234, 0.368, 0.255, math.nan, 0.676, 0.593, 0.62, math.nan, 0.322, 0.558, math.nan, 0.369, 0.29, 0.21, 0.341, 0.209, math.nan, 0.238, 0.196, 0.185, 0.044],
    'spatial_smoothness_cds': [0.029, 0.043, 0.035, 0.027, 0.09, 0.094, 0.08, 0.063, 0.002, 0.038, 0.063, 0.011, 0.049, 0.045, 0.036, 0.043, 0.036, 0.051, 0.042, 0.037, 0.032, 0.033],
    'spatial_smoothness_sec@05': [0.209, 0.088, 0.134, 0.245, 0.107, 0.047, 0.027, 0.01, 0.436, 0.2, 0.103, 0.244, 0.053, 0.112, 0.043, 0.24, 0.041, 0.184, 0.189, 0.2, 0.211, 0.185],
    'semantic_alignment_vf_loss': [0.64, 0.36, 0.41, 0.35, 0.18, 0.08, 0.09, 0.59, 0.33, 0.4, 0.65, 0.46, 0.36, 0.01, 0.06, 0.34, 0.06, 0.22, 0.21, 0.22, 0.22, 0.2],
    'semantic_alignment_acc_1_avg': [0.76, 0.58, 0.91, 2.23, 0.97, 17.25, 12.36, 1.14, 3.92, 3.02, 1.15, 4.96, 0.83, 73.58, 45.83, 6.62, 45.99, 25.38, 2.44, 2.95, 6.9, 13.69],
    'semantic_alignment_acc_5_avg': [2.62, 2.34, 3.37, 7.53, 3.31, 38.54, 31.35, 4.24, 13.69, 9.68, 4.1, 14.35, 3.1, 92.84, 75.85, 17.91, 75.91, 44.78, 7.77, 9.09, 19.31, 32.35],
    'acc_1_max': [0.34, 0.42, 1.24, 2.31, 0.14, 14.33, 14.58, 0.56, 4.51, 2.78, 0.4, 3.19, math.nan, 75.45, 49.73, 5.01, 49.94, 15.7, 0.22, 0.3, 0.92, 9.16],
    'acc_5_max': [1.65, 1.626, 4.252, 7.724, 0.68, 34.14, 35.12, 2.33, 14.67, 8.23, 1.62, 10.17, math.nan, 93.63, 78.51, 14.22, 78.84, 32.66, 1.748, 1.602, 2.744, 24.712],
    'lds_bn': [0.3, 0.433, 0.365, 0.232, math.nan, 0.699, 0.618, 0.627, math.nan, 0.313, 0.559, math.nan, 0.49, 0.357, 0.281, 0.342, 0.279, math.nan, 0.333, 0.3, 0.254, 0.285],
    'ifid_bn': [68.05, 46.1, 70.52, 45.12, 32.66, 25.01, 23.04, 54.18, 18.26, 47.35, 47.02, 21.05, 36.22, 6.05, 7.17, 19.13, 5.12, 12.91, 20.7, 22.92, 25.85, 22.07],
    'iis': [23.41, 40.38, 21.46, 28.33, 43.0, 55.45, 51.53, 27.02, 67.39, 31.87, 30.53, 46.18, 42.35, 142.81, 117.78, 55.32, 129.45, 76.99747467, 50.82, 52.41, 54.02, 55.03],
    'iprec': [0.25, 0.42, 0.23, 0.34, 0.47, 0.54, 0.53, 0.32, 0.6, 0.33, 0.41, 0.49, 0.43, 0.75, 0.73, 0.54, 0.79, 0.65392, 0.51, 0.49, 0.49, 0.53],
    'irec': [0.67, 0.77, 0.67, 0.69, 0.57, 0.63, 0.67, 0.69, 0.59, 0.71, 0.62, 0.72, 0.8, 0.69, 0.68, 0.7, 0.68, 0.69658, 0.64, 0.62, 0.57, 0.57],
    'yymm': [21.12, 24.03, 24.08, 24.1, 24.12, 25.01, 25.01, 25.02, 25.02, 25.04, 25.04, 25.07, 25.08, 25.1, 25.1, 25.11, 25.12, 25.12, 25.12, 25.12, 25.12, 26.05],
    'param_m': [83.65, 83.82, 83.82, 312.25, 608.92, 69.83, 70.29, 83.65, 173.89, 69.83, 83.65, 171.71, 126.89, 501.93, 82.96, 84.04, 71.73, 355.2, 167.17, 295.63, 731.57, 721.52],
    'dim': ['4x32x32', '16x32x32', '16x32x32', '32x8x8', '64x32', '32x16x16', '64x16x16', '4x32x32', '128x32', '32x16x16', '4x32x32', '128x32', '16x32x32', '768x16x16', '392x16x16', '128x16x16', '384x16x16', '256x32', '64x16x16', '64x16x16', '64x16x16', '32x16x16'],
    'token': ['16x16', '16x16', '16x16', '8x8', 64.0, '16x16', '16x16', '16x16', 128.0, '16x16', '16x16', 128.0, '16x16', '16x16', '16x16', '16x16', '16x16', 256.0, '16x16', '16x16', '16x16', '16x16'],
    'source': ['https://arxiv.org/abs/2112.10752', 'https://arxiv.org/abs/2403.03206', 'https://bfl.ai/blog/24-08-01-bfl', 'https://arxiv.org/abs/2410.10733', 'https://arxiv.org/abs/2412.10958', 'https://arxiv.org/abs/2501.01423', 'https://arxiv.org/abs/2501.01423', 'https://arxiv.org/abs/2502.09509', 'https://arxiv.org/abs/2502.03444', 'https://arxiv.org/abs/2504.10483', 'https://arxiv.org/abs/2504.10483', 'https://arxiv.org/html/2507.15856v1', 'https://arxiv.org/abs/2508.02324', 'https://arxiv.org/abs/2510.11690', 'https://arxiv.org/abs/2510.15301', 'https://bfl.ai/blog/flux-2', 'https://arxiv.org/abs/2512.11749', 'https://arxiv.org/abs/2512.07778', 'https://arxiv.org/abs/2512.13687', 'https://arxiv.org/abs/2512.13687', 'https://arxiv.org/abs/2512.13687', 'https://arxiv.org/abs/2605.07915'],
    'ifid_linear_05': [84.3, 40.84, 91.29, 82.32, 26.43, 31.8, 32.79, 67.53, 19.16, 66.21, 92.1, 48.73, 35.17, 8.14, 8.42, 42.55, 5.98, 20.89, 26.17, 32.79, 33.54, 26.3],
    'ifid_linear_03': [14.69, 6.57, 26.6, 17.64, 6.71, 7.74, 5.57, 12.71, 5.9, 10.61, 19.13, 3.2, 3.47, 1.28, 2.32, 2.2, 1.68, 1.9479, 4.01, 5.36, 6.39, 1.74],
    'ifid_linear_01': [1.56, 0.52, 0.8, 1.66, 0.76, 0.67, 0.47, 0.83, 0.97, 0.65, 1.57, 0.61, 0.77, 0.64, 0.68, 0.31, 1.08, 0.7478, 1.27, 0.81, 0.51, 0.29],
    'ifid_slerp_03': [13.17, 6.69, 24.35, 11.63, 10.44, 6.29, 4.91, 11.28, 5.82, 8.62, 10.86, 2.59, 3.78, 1.21, 2.39, 2.75, 1.73, 1.92, 5.41, 8.82, 9.84, 1.57],
    'ifid_slerp_01': [1.25, 0.53, 1.07, 1.21, 0.91, 0.52, 0.37, 0.76, 0.95, 0.47, 0.85, 0.55, 0.91, 0.64, 0.69, 0.22, 1.11, 0.73, 1.28, 0.97, 0.72, 0.28],
})


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric(key1, name1, key2, name2, legend=False):
    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    x = pd.to_numeric(
        data[key1],
        errors="coerce",
    ).to_numpy()

    y = pd.to_numeric(
        data[key2],
        errors="coerce",
    ).to_numpy()

    names = data["vae_name"].astype(str).to_numpy()

    # Ignore NaN / Inf values
    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]
    names = names[mask]


    # ------------------------------------------------------------
    # 22-color palette
    # ------------------------------------------------------------
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#393b79",  # dark blue
        "#637939",  # dark olive
        "#8c6d31",  # ochre
        "#843c39",  # dark red
        "#7b4173",  # dark purple
        "#3182bd",  # medium blue
        "#31a354",  # medium green
        "#e6550d",  # burnt orange
        "#756bb1",  # violet
        "#636363",  # dark gray
        "#d95f0e",  # orange-red
        "#1b9e77",  # teal
    ]


    fig, ax = plt.subplots(figsize=(3, 5))

    # Scatter points
    for xi, yi, name, color in zip(x, y, names, colors):
        ax.scatter(
            xi,
            yi,
            s=70,
            color=color,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label=name,
            zorder=3,
        )


    # ------------------------------------------------------------
    # Linear regression
    # ------------------------------------------------------------
    slope, intercept = np.polyfit(x, y, 1)

    x_line = np.linspace(
        x.min(),
        x.max(),
        200,
    )

    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Linear regression",
        zorder=2,
    )


    # ------------------------------------------------------------
    # Pearson correlation (optional)
    # ------------------------------------------------------------
    r = np.corrcoef(x, y)[0, 1]

    print(f"Linear regression: y = {slope:.3f} x + {intercept:.3f}")
    print(f"Pearson correlation: {r:.3f}")


    # ------------------------------------------------------------
    # Figure formatting
    # ------------------------------------------------------------
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)

    ax.grid(
        alpha=1.0,
    )
    if legend:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
            frameon=False,
        )

    plt.tight_layout()
    plt.savefig(
        "{}_vs_{}.pdf".format(key1, key2),
        bbox_inches="tight",
    )
    plt.show()

