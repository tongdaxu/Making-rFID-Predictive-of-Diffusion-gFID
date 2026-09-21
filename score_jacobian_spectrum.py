import os
import json
import csv
import math
import argparse
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config


# ------------------------------------------------------------
# Basic utils
# ------------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def unwrap_latent(z):
    """
    Robustly unwrap VAE encode outputs.

    Important:
    torch.Tensor also has a .mode() method, so we must return Tensor directly
    before checking posterior-like .mode().
    """
    # Case 1: direct tensor
    if isinstance(z, torch.Tensor):
        return z

    # Case 2: tuple/list output, e.g. (posterior, h) or (z, ...)
    if isinstance(z, (tuple, list)):
        z = z[0]
        if isinstance(z, torch.Tensor):
            return z

    # Case 3: dict output
    if isinstance(z, dict):
        for key in ["z", "latent", "latents", "sample"]:
            if key in z:
                z = z[key]
                break
        if isinstance(z, torch.Tensor):
            return z

    # Case 4: posterior-like object, e.g. DiagonalGaussianDistribution
    # Do NOT apply this to torch.Tensor.
    if hasattr(z, "mode") and callable(z.mode):
        z = z.mode()
    elif hasattr(z, "sample") and callable(z.sample):
        z = z.sample()

    if isinstance(z, torch.Tensor):
        return z

    raise TypeError(f"Unsupported latent output type after unwrap: {type(z)}")

def encode_vae(vae, img, y=None):
    with torch.no_grad():
        try:
            z = vae.encode(img, y)
        except TypeError:
            z = vae.encode(img)
    return unwrap_latent(z)


def latent_channel_sum_sumsq_count(z: torch.Tensor):
    """
    Per-channel statistics for latent BN-style normalization.

    Supports:
      [B, C, H, W]
      [B, N, C]
      [B, C]
    """
    z = z.float()

    if z.dim() == 4:
        reduce_dims = (0, 2, 3)
        count = z.shape[0] * z.shape[2] * z.shape[3]
    elif z.dim() == 3:
        # token latent: [B, N, C]
        reduce_dims = (0, 1)
        count = z.shape[0] * z.shape[1]
    elif z.dim() == 2:
        reduce_dims = (0,)
        count = z.shape[0]
    else:
        raise ValueError(f"Unsupported latent shape: {tuple(z.shape)}")

    s = z.sum(dim=reduce_dims)
    ss = (z ** 2).sum(dim=reduce_dims)
    return s, ss, count


def normalize_latent_channelwise(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    if z.dim() == 4:
        return (z - mean[None, :, None, None]) / std[None, :, None, None]
    elif z.dim() == 3:
        return (z - mean[None, None, :]) / std[None, None, :]
    elif z.dim() == 2:
        return (z - mean[None, :]) / std[None, :]
    else:
        raise ValueError(f"Unsupported latent shape: {tuple(z.shape)}")


def flatten_latent(z: torch.Tensor):
    return z.reshape(z.shape[0], -1).contiguous()


def interpolant_scalar(t: float, path_type: str):
    """
    Same convention as your SiT.interpolant().
    """
    if path_type == "linear":
        alpha = 1.0 - t
        sigma = t
        d_alpha = -1.0
        d_sigma = 1.0
    elif path_type == "cosine":
        alpha = math.cos(t * math.pi / 2)
        sigma = math.sin(t * math.pi / 2)
        d_alpha = -math.pi / 2 * math.sin(t * math.pi / 2)
        d_sigma = math.pi / 2 * math.cos(t * math.pi / 2)
    else:
        raise NotImplementedError(path_type)

    return alpha, sigma, d_alpha, d_sigma


def maybe_shift_time(t: float, tshift: float):
    return tshift * t / (1.0 + (tshift - 1.0) * t)


# ------------------------------------------------------------
# Encode training latents
# ------------------------------------------------------------
@torch.no_grad()
def encode_training_latents(
    vae,
    loader,
    device,
    metric_norm: str = "bn_channel",
    store_dtype: torch.dtype = torch.float16,
):
    zs_cpu = []
    ys_cpu = []

    stat_sum = None
    stat_sumsq = None
    stat_count = 0
    latent_shape = None

    for img, y, name in tqdm(loader, desc="Encoding train latents"):
        img = img.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)

        z = encode_vae(vae, img, y_dev)

        if latent_shape is None:
            latent_shape = tuple(z.shape[1:])

        s, ss, cnt = latent_channel_sum_sumsq_count(z)
        if stat_sum is None:
            stat_sum = torch.zeros_like(s, dtype=torch.float64, device=device)
            stat_sumsq = torch.zeros_like(ss, dtype=torch.float64, device=device)

        stat_sum += s.to(device=device, dtype=torch.float64)
        stat_sumsq += ss.to(device=device, dtype=torch.float64)
        stat_count += cnt

        zs_cpu.append(z.detach().cpu().to(store_dtype))
        ys_cpu.append(y.detach().cpu())

    zs = torch.cat(zs_cpu, dim=0)
    ys = torch.cat(ys_cpu, dim=0)

    stat_count_t = torch.tensor(float(stat_count), device=device, dtype=torch.float64)
    latent_mean = stat_sum / stat_count_t
    latent_var = stat_sumsq / stat_count_t - latent_mean ** 2
    latent_std = torch.sqrt(torch.clamp(latent_var, min=1e-6))

    latent_mean = latent_mean.float().cpu()
    latent_std = latent_std.float().cpu()

    if metric_norm == "bn_channel":
        zs_metric = normalize_latent_channelwise(
            zs.float(),
            latent_mean,
            latent_std,
        ).to(store_dtype)
    elif metric_norm == "none":
        zs_metric = zs
    else:
        raise ValueError(metric_norm)

    train_flat = flatten_latent(zs_metric).contiguous()
    train_sq = (train_flat.float() ** 2).sum(dim=1).contiguous()

    return {
        "train_flat": train_flat,
        "train_sq": train_sq,
        "ys": ys,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "latent_shape": latent_shape,
        "num_train": train_flat.shape[0],
        "dim": train_flat.shape[1],
    }


# ------------------------------------------------------------
# Top-k / exact-global empirical weights
# ------------------------------------------------------------
@torch.no_grad()
def topk_train_neighbors(
    train_flat_cpu: torch.Tensor,
    train_sq_cpu: torch.Tensor,
    zt_flat: torch.Tensor,
    alpha: float,
    topk: int,
    train_chunk_size: int,
    device,
):
    """
    Original fast approximation: find top-k training z_i by distance

        || z_t - alpha * z_i ||^2

    This is kept for --weight-mode topk_local.
    """
    B, D = zt_flat.shape
    M = train_flat_cpu.shape[0]

    best_dist = torch.full((B, topk), float("inf"), device=device, dtype=torch.float32)
    best_idx = torch.full((B, topk), -1, device=device, dtype=torch.long)

    zt_sq = (zt_flat ** 2).sum(dim=1)  # [B]

    for start in range(0, M, train_chunk_size):
        end = min(start + train_chunk_size, M)

        train_chunk = train_flat_cpu[start:end].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        train_sq = train_sq_cpu[start:end].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        cross = zt_flat @ train_chunk.t()  # [B, m]
        dist = (
            zt_sq[:, None]
            + (alpha ** 2) * train_sq[None, :]
            - 2.0 * alpha * cross
        )

        k_this = min(topk, end - start)
        vals, idx = torch.topk(dist, k=k_this, dim=1, largest=False)
        idx = idx + start

        cat_dist = torch.cat([best_dist, vals], dim=1)
        cat_idx = torch.cat([best_idx, idx], dim=1)

        best_dist, pos = torch.topk(cat_dist, k=topk, dim=1, largest=False)
        best_idx = cat_idx.gather(1, pos)

        del train_chunk, train_sq, cross, dist, vals, idx, cat_dist, cat_idx
        torch.cuda.empty_cache()

    return best_dist, best_idx


@torch.no_grad()
def topk_and_exact_global_moments(
    train_flat_cpu: torch.Tensor,
    train_sq_cpu: torch.Tensor,
    zt_flat: torch.Tensor,
    alpha: float,
    sigma: float,
    topk: int,
    train_chunk_size: int,
    device,
):
    """
    Closer to the PPT formula.

    For each validation z_t, this scans ALL M training latents and computes
    the exact empirical softmax normalizer and first/second moments:

        w_i = exp(-||z_t - alpha z_i||^2 / 2 sigma^2) / Z
        mu = sum_{i=1}^M w_i z_i
        trace(C_w) = sum_i w_i ||z_i - mu||^2

    It also keeps top-k candidates only for the spectral low-rank part.
    Therefore:
      - weights are globally normalized over all M, not top-k renormalized;
      - mu is the exact all-M empirical posterior mean;
      - topk_mass / tail_mass tells you how much of the PPT sum is retained
        in the spectral approximation.

    This is not a full D x D eigendecomposition of the all-M covariance, but it
    is much closer to the PPT formula than --weight-mode topk_local.
    """
    B, D = zt_flat.shape
    M = train_flat_cpu.shape[0]

    best_dist = torch.full((B, topk), float("inf"), device=device, dtype=torch.float32)
    best_idx = torch.full((B, topk), -1, device=device, dtype=torch.long)

    # Streaming log-sum-exp state.
    logZ = torch.full((B,), -float("inf"), device=device, dtype=torch.float32)
    logZ2 = torch.full((B,), -float("inf"), device=device, dtype=torch.float32)  # log sum exp(2 logits)

    # These are maintained normalized by the current running logZ.
    mu = torch.zeros((B, D), device=device, dtype=torch.float32)
    m2 = torch.zeros((B,), device=device, dtype=torch.float32)  # E_w ||z||^2

    zt_sq = (zt_flat ** 2).sum(dim=1)  # [B]

    for start in range(0, M, train_chunk_size):
        end = min(start + train_chunk_size, M)

        train_chunk = train_flat_cpu[start:end].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        train_sq = train_sq_cpu[start:end].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        cross = zt_flat @ train_chunk.t()  # [B, m]
        dist = (
            zt_sq[:, None]
            + (alpha ** 2) * train_sq[None, :]
            - 2.0 * alpha * cross
        )
        logits = -dist / (2.0 * sigma ** 2)  # [B, m]

        # Update top-k candidates.
        k_this = min(topk, end - start)
        vals, idx = torch.topk(dist, k=k_this, dim=1, largest=False)
        idx = idx + start
        cat_dist = torch.cat([best_dist, vals], dim=1)
        cat_idx = torch.cat([best_idx, idx], dim=1)
        best_dist, pos = torch.topk(cat_dist, k=topk, dim=1, largest=False)
        best_idx = cat_idx.gather(1, pos)

        # Stable update of exact all-M posterior moments.
        chunk_logZ = torch.logsumexp(logits, dim=1)  # [B]
        new_logZ = torch.logaddexp(logZ, chunk_logZ)

        old_scale = torch.exp(logZ - new_logZ)  # 0 for initial -inf
        chunk_weights_scaled = torch.exp(logits - new_logZ[:, None])  # sums to chunk_mass under new_logZ

        mu = mu * old_scale[:, None] + chunk_weights_scaled @ train_chunk
        m2 = m2 * old_scale + (chunk_weights_scaled * train_sq[None, :]).sum(dim=1)
        logZ = new_logZ

        # Exact ESS needs sum_i w_i^2 = exp(logsumexp(2 logits) - 2 logZ).
        chunk_logZ2 = torch.logsumexp(2.0 * logits, dim=1)
        logZ2 = torch.logaddexp(logZ2, chunk_logZ2)

        del train_chunk, train_sq, cross, dist, logits, vals, idx, cat_dist, cat_idx
        del chunk_logZ, new_logZ, old_scale, chunk_weights_scaled, chunk_logZ2
        torch.cuda.empty_cache()

    trace_cov_exact = torch.clamp(m2 - (mu ** 2).sum(dim=1), min=0.0)
    sum_w2_exact = torch.exp(logZ2 - 2.0 * logZ)
    ess_exact = 1.0 / torch.clamp(sum_w2_exact, min=1e-30)

    return best_dist, best_idx, logZ.detach(), mu.detach(), trace_cov_exact.detach(), ess_exact.detach()


# ------------------------------------------------------------
# Jacobian spectrum from low-rank Gram trick
# ------------------------------------------------------------
@torch.no_grad()
def compute_jacobian_spectrum_batch(
    train_flat_cpu: torch.Tensor,
    best_idx: torch.Tensor,
    best_dist: torch.Tensor,
    alpha: float,
    sigma: float,
    dim: int,
    device,
    eig_batch_size: int = 8,
    weight_mode: str = "topk_local",
    logZ: torch.Tensor = None,
    mu_exact: torch.Tensor = None,
    trace_cov_exact: torch.Tensor = None,
    ess_exact: torch.Tensor = None,
):
    """
    J = -1/sigma^2 I + alpha^2/sigma^4 C_w

    Supported modes:

    1. weight_mode='topk_local'
       Original approximation. Renormalize weights inside top-k and use top-k mu.

    2. weight_mode='exact_global'
       Closer to PPT. Use globally normalized weights over all M for the top-k
       candidates, and use exact all-M mu. The spectrum is still computed from
       the top-k retained weighted components, but topk_mass/tail_mass and
       trace_cov_tail_ratio quantify how close this is to the all-M sum.
    """
    B, K = best_idx.shape

    base_eig = -1.0 / (sigma ** 2)
    coeff = (alpha ** 2) / (sigma ** 4)

    all_nontriv = []
    summaries = []

    for s in range(0, B, eig_batch_size):
        e = min(s + eig_batch_size, B)
        bsz = e - s

        idx_cpu = best_idx[s:e].detach().cpu()
        dist_gpu = best_dist[s:e].to(device=device, dtype=torch.float32)

        cand = train_flat_cpu[idx_cpu.reshape(-1)].reshape(bsz, K, dim)
        cand = cand.to(device=device, dtype=torch.float32, non_blocking=True)

        logits = -dist_gpu / (2.0 * sigma ** 2)

        if weight_mode == "topk_local":
            weights = torch.softmax(logits, dim=1)  # sums to 1 inside top-k
            mu = (weights[:, :, None] * cand).sum(dim=1)
            topk_mass = torch.ones((bsz,), device=device, dtype=torch.float32)
            tail_mass = torch.zeros((bsz,), device=device, dtype=torch.float32)
            ess = 1.0 / torch.clamp((weights ** 2).sum(dim=1), min=1e-30)
            trace_exact = None
            ess_all = None
        elif weight_mode == "exact_global":
            if logZ is None or mu_exact is None:
                raise ValueError("exact_global mode requires logZ and mu_exact.")
            logZ_b = logZ[s:e].to(device=device, dtype=torch.float32)
            weights = torch.exp(logits - logZ_b[:, None])  # globally normalized, sum <= 1 over top-k
            mu = mu_exact[s:e].to(device=device, dtype=torch.float32)
            topk_mass = weights.sum(dim=1)
            tail_mass = torch.clamp(1.0 - topk_mass, min=0.0)
            ess = (topk_mass ** 2) / torch.clamp((weights ** 2).sum(dim=1), min=1e-30)
            trace_exact = trace_cov_exact[s:e].to(device=device, dtype=torch.float32) if trace_cov_exact is not None else None
            ess_all = ess_exact[s:e].to(device=device, dtype=torch.float32) if ess_exact is not None else None
        else:
            raise ValueError(f"Unsupported weight_mode: {weight_mode}")

        centered = cand - mu[:, None, :]
        X = centered * torch.sqrt(torch.clamp(weights, min=1e-30))[:, :, None]
        gram = X @ X.transpose(1, 2)  # [b, K, K]

        # Numerical PSD hygiene. The covariance must be PSD, so eig_c >= 0.
        gram = 0.5 * (gram + gram.transpose(1, 2))
        eig_c = torch.linalg.eigvalsh(gram.double()).float()
        eig_c = torch.clamp(eig_c, min=0.0)
        eig_j = base_eig + coeff * eig_c

        eig_j_cpu = eig_j.detach().cpu().float().numpy()
        eig_c_cpu = eig_c.detach().cpu().float().numpy()
        all_nontriv.append(eig_j_cpu)

        topk_mass_cpu = topk_mass.detach().cpu().float().numpy()
        tail_mass_cpu = tail_mass.detach().cpu().float().numpy()
        ess_cpu = ess.detach().cpu().float().numpy()
        if trace_exact is not None:
            trace_exact_cpu = trace_exact.detach().cpu().float().numpy()
        else:
            trace_exact_cpu = [None] * bsz
        if ess_all is not None:
            ess_all_cpu = ess_all.detach().cpu().float().numpy()
        else:
            ess_all_cpu = [None] * bsz

        for row_j, row_c, mass, tail, ess_k, tr_exact, ess_e in zip(
            eig_j_cpu,
            eig_c_cpu,
            topk_mass_cpu,
            tail_mass_cpu,
            ess_cpu,
            trace_exact_cpu,
            ess_all_cpu,
        ):
            row_sorted = np.sort(row_j)
            pos_count = int((row_j > 0).sum())
            trace_topk = float(row_c.sum())

            if tr_exact is None:
                trace_tail = None
                trace_tail_ratio = None
                tr_exact_out = None
                ess_exact_out = None
            else:
                tr_exact_out = float(tr_exact)
                trace_tail = float(max(tr_exact_out - trace_topk, 0.0))
                trace_tail_ratio = float(trace_tail / max(tr_exact_out, 1e-30))
                ess_exact_out = float(ess_e)

            summaries.append({
                "lambda_min_nontriv": float(row_sorted[0]),
                "lambda_max_nontriv": float(row_sorted[-1]),
                "lambda_mean_nontriv": float(row_j.mean()),
                "lambda_std_nontriv": float(row_j.std()),
                "positive_nontriv_count": pos_count,
                "positive_full_ratio": float(pos_count / dim),
                "top10_mean": float(row_sorted[-10:].mean()) if len(row_sorted) >= 10 else float(row_sorted.mean()),
                "top50_mean": float(row_sorted[-50:].mean()) if len(row_sorted) >= 50 else float(row_sorted.mean()),
                "topk_mass": float(mass),
                "tail_mass": float(tail),
                "ess_topk": float(ess_k),
                "ess_exact": ess_exact_out,
                "trace_cov_topk": trace_topk,
                "trace_cov_exact": tr_exact_out,
                "trace_cov_tail": trace_tail,
                "trace_cov_tail_ratio": trace_tail_ratio,
            })

        del cand, dist_gpu, logits, weights, mu, centered, X, gram, eig_c, eig_j
        torch.cuda.empty_cache()

    all_nontriv = np.concatenate(all_nontriv, axis=0)
    return all_nontriv, summaries, base_eig, coeff


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if args.device is not None else "cuda")

    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_set = ImageNetValDataset(args.dataset_ref, transform, args.train_size)
    val_set = ImageNetValDataset(args.dataset, transform, args.val_size)

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # --------------------------------------------------------
    # Encode train latents
    # --------------------------------------------------------
    cache_path = os.path.join(args.out_dir, "train_latents_cache.pt")

    if args.use_cache and os.path.exists(cache_path):
        print(f"[Load cache] {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
    else:
        cache = encode_training_latents(
            vae=vae,
            loader=train_loader,
            device=device,
            metric_norm=args.metric_norm,
            store_dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.store_dtype],
        )
        if args.save_cache:
            print(f"[Save cache] {cache_path}")
            torch.save(cache, cache_path)

    train_flat = cache["train_flat"].contiguous()
    train_sq = cache["train_sq"].contiguous()
    latent_mean = cache["latent_mean"]
    latent_std = cache["latent_std"]
    latent_shape = cache["latent_shape"]
    dim = int(cache["dim"])

    print("========== Train latent cache ==========")
    print("latent_shape:", latent_shape)
    print("train_flat:", tuple(train_flat.shape), train_flat.dtype)
    print("dim:", dim)
    print("metric_norm:", args.metric_norm)
    print("weight_mode:", args.weight_mode)

    # --------------------------------------------------------
    # Time / interpolant
    # --------------------------------------------------------
    t_raw = float(args.t)
    t_eff = maybe_shift_time(t_raw, args.tshift)
    alpha, sigma, d_alpha, d_sigma = interpolant_scalar(t_eff, args.path_type)

    print("========== Interpolant ==========")
    print("path_type:", args.path_type)
    print("t_raw:", t_raw)
    print("tshift:", args.tshift)
    print("t_eff:", t_eff)
    print("alpha:", alpha)
    print("sigma:", sigma)
    print("base_eig:", -1.0 / (sigma ** 2))
    print("coeff:", (alpha ** 2) / (sigma ** 4))

    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    # --------------------------------------------------------
    # Validation loop
    # --------------------------------------------------------
    all_eigs = []
    all_summaries = []
    total_seen = 0

    for img, y, name in tqdm(val_loader, desc="Val z_t -> J spectrum"):
        img = img.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)

        z_val = encode_vae(vae, img, y_dev)

        if args.metric_norm == "bn_channel":
            z_val = normalize_latent_channelwise(
                z_val.float(),
                latent_mean.to(device),
                latent_std.to(device),
            )
        elif args.metric_norm == "none":
            z_val = z_val.float()
        else:
            raise ValueError(args.metric_norm)

        z_val_flat = flatten_latent(z_val).to(device=device, dtype=torch.float32)

        eps = torch.randn_like(z_val_flat)
        zt_flat = alpha * z_val_flat + sigma * eps

        if args.weight_mode == "exact_global":
            (
                best_dist,
                best_idx,
                logZ,
                mu_exact,
                trace_cov_exact,
                ess_exact,
            ) = topk_and_exact_global_moments(
                train_flat_cpu=train_flat,
                train_sq_cpu=train_sq,
                zt_flat=zt_flat,
                alpha=alpha,
                sigma=sigma,
                topk=args.topk,
                train_chunk_size=args.train_chunk_size,
                device=device,
            )
        elif args.weight_mode == "topk_local":
            best_dist, best_idx = topk_train_neighbors(
                train_flat_cpu=train_flat,
                train_sq_cpu=train_sq,
                zt_flat=zt_flat,
                alpha=alpha,
                topk=args.topk,
                train_chunk_size=args.train_chunk_size,
                device=device,
            )
            logZ = None
            mu_exact = None
            trace_cov_exact = None
            ess_exact = None
        else:
            raise ValueError(args.weight_mode)

        eig_nontriv, summaries, base_eig, coeff = compute_jacobian_spectrum_batch(
            train_flat_cpu=train_flat,
            best_idx=best_idx,
            best_dist=best_dist,
            alpha=alpha,
            sigma=sigma,
            dim=dim,
            device=device,
            eig_batch_size=args.eig_batch_size,
            weight_mode=args.weight_mode,
            logZ=logZ,
            mu_exact=mu_exact,
            trace_cov_exact=trace_cov_exact,
            ess_exact=ess_exact,
        )

        all_eigs.append(eig_nontriv)

        for local_i, s in enumerate(summaries):
            s["global_index"] = total_seen + local_i
            s["t"] = t_raw
            s["t_eff"] = t_eff
            s["alpha"] = alpha
            s["sigma"] = sigma
            s["base_eig"] = base_eig
            s["coeff"] = coeff
            s["dim"] = dim
            s["topk"] = args.topk
            s["weight_mode"] = args.weight_mode
            all_summaries.append(s)

        total_seen += z_val_flat.shape[0]

        del img, y_dev, z_val, z_val_flat, eps, zt_flat, best_dist, best_idx
        if args.weight_mode == "exact_global":
            del logZ, mu_exact, trace_cov_exact, ess_exact
        torch.cuda.empty_cache()

    all_eigs = np.concatenate(all_eigs, axis=0)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    eig_path = os.path.join(args.out_dir, "eig_nontrivial.npy")
    np.save(eig_path, all_eigs)

    csv_path = os.path.join(args.out_dir, "summary.csv")
    fieldnames = list(all_summaries[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_summaries:
            writer.writerow(row)

    def mean_key(key):
        vals = []
        for r in all_summaries:
            v = r.get(key, None)
            if v is None:
                continue
            try:
                if math.isnan(v):
                    continue
            except TypeError:
                pass
            vals.append(float(v))
        return float(np.mean(vals)) if vals else None

    lambda_max_arr = np.array([r["lambda_max_nontriv"] for r in all_summaries], dtype=np.float64)

    metrics = {
        "vae_config": args.vae_config,
        "dataset_ref": args.dataset_ref,
        "dataset": args.dataset,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "topk": args.topk,
        "weight_mode": args.weight_mode,
        "t": t_raw,
        "t_eff": t_eff,
        "path_type": args.path_type,
        "metric_norm": args.metric_norm,
        "latent_shape": latent_shape,
        "dim": dim,
        "base_eig": float(-1.0 / (sigma ** 2)),
        "coeff": float((alpha ** 2) / (sigma ** 4)),
        "nontriv_lambda_mean": float(all_eigs.mean()),
        "nontriv_lambda_std": float(all_eigs.std()),
        "nontriv_lambda_min": float(all_eigs.min()),
        "nontriv_lambda_max": float(all_eigs.max()),
        "positive_nontriv_ratio": float((all_eigs > 0).mean()),
        "positive_sample_ratio": float((lambda_max_arr > 0).mean()),
        "positive_full_ratio_mean": mean_key("positive_full_ratio"),
        "lambda_max_mean": mean_key("lambda_max_nontriv"),
        "lambda_max_p50": float(np.quantile(lambda_max_arr, 0.50)),
        "lambda_max_p90": float(np.quantile(lambda_max_arr, 0.90)),
        "lambda_max_p95": float(np.quantile(lambda_max_arr, 0.95)),
        "lambda_max_p99": float(np.quantile(lambda_max_arr, 0.99)),
        "lambda_max_p999": float(np.quantile(lambda_max_arr, 0.999)),
        "top10_mean": mean_key("top10_mean"),
        "top50_mean": mean_key("top50_mean"),
        "topk_mass_mean": mean_key("topk_mass"),
        "tail_mass_mean": mean_key("tail_mass"),
        "tail_mass_p95": float(np.quantile([r["tail_mass"] for r in all_summaries], 0.95)) if all_summaries[0].get("tail_mass") is not None else None,
        "tail_mass_p99": float(np.quantile([r["tail_mass"] for r in all_summaries], 0.99)) if all_summaries[0].get("tail_mass") is not None else None,
        "ess_topk_mean": mean_key("ess_topk"),
        "ess_exact_mean": mean_key("ess_exact"),
        "trace_cov_topk_mean": mean_key("trace_cov_topk"),
        "trace_cov_exact_mean": mean_key("trace_cov_exact"),
        "trace_cov_tail_ratio_mean": mean_key("trace_cov_tail_ratio"),
    }

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("========== Saved ==========")
    print("eig_nontrivial:", eig_path)
    print("summary:", csv_path)
    print("metrics:", metrics_path)
    print(json.dumps(metrics, indent=2))

    # Optional plot.
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(all_eigs.reshape(-1), bins=args.hist_bins)
        plt.axvline(metrics["base_eig"], linestyle="--")
        plt.title(f"Non-trivial eigenvalues of J, t={t_raw}")
        plt.xlabel("eigenvalue")
        plt.ylabel("count")
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, "hist_nontrivial.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print("hist:", fig_path)
    except Exception as e:
        print("[Warning] failed to plot histogram:", repr(e))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vae-config", type=str, required=True)
    parser.add_argument("--dataset-ref", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--out-dir", type=str, default="./score_jacobian_sdvae_t05")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=200000)
    parser.add_argument("--val-size", type=int, default=50000)

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--t", type=float, default=0.5)
    parser.add_argument("--tshift", type=float, default=1.0)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])

    parser.add_argument("--metric-norm", type=str, default="bn_channel", choices=["bn_channel", "none"])

    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--weight-mode", type=str, default="exact_global", choices=["exact_global", "topk_local"],
                        help="exact_global computes all-M logZ/mu/trace and uses globally normalized top-k spectrum; topk_local is the old fast approximation.")
    parser.add_argument("--train-chunk-size", type=int, default=4096)
    parser.add_argument("--eig-batch-size", type=int, default=8)

    parser.add_argument("--store-dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save-cache", action="store_true")
    parser.add_argument("--use-cache", action="store_true")

    parser.add_argument("--hist-bins", type=int, default=200)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)