import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from ifid.fid.fid import InceptionV3, calculate_frechet_distance
from ifid.fid.psnr import get_psnr
from ifid.fid.ssim import get_ssim_and_msssim
from ifid.fid.lpips import get_lpips
from ifid.dataset import ImageNetValDataset
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist
from omegaconf import OmegaConf
from ifid.vae.utils import instantiate_from_config
import shutil


def get_metric_dtype(name):
    if name == "fp32":
        return torch.float32
    elif name == "bf16":
        return torch.bfloat16
    elif name == "fp16":
        return torch.float16
    else:
        raise ValueError(f"Unsupported metric dtype: {name}")


def ddp_all_reduce_(x, op=dist.ReduceOp.SUM):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=op)
    return x


def latent_channel_sum_sumsq_count(z):
    """
    Returns per-channel sum, sumsq, count.

    Supports:
      z: [B, C, H, W]
      z: [B, N, C]
      z: [B, C]
    """
    z = z.float()

    if z.dim() == 4:
        # [B, C, H, W]
        reduce_dims = (0, 2, 3)
        count = z.shape[0] * z.shape[2] * z.shape[3]
    elif z.dim() == 3:
        # assume [B, N, C]
        reduce_dims = (0, 1)
        count = z.shape[0] * z.shape[1]
    elif z.dim() == 2:
        # [B, C]
        reduce_dims = (0,)
        count = z.shape[0]
    else:
        raise ValueError(f"Unsupported latent shape for stats: {z.shape}")

    s = z.sum(dim=reduce_dims)
    ss = (z ** 2).sum(dim=reduce_dims)

    return s, ss, count


def get_cache_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    elif name == "fp16":
        return torch.float16
    elif name == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported cache dtype: {name}")


def normalize_latent_channelwise(z, mean, std):
    """
    Supports:
      [B, C, H, W]
      [B, K, C, H, W]
      [B, N, C]
      [B, K, N, C]
      [B, C]
    """
    c = mean.numel()

    if z.dim() == 5:
        # [B, K, C, H, W]
        if z.shape[2] != c:
            raise ValueError(f"Expected [B,K,C,H,W] with C={c}, got {z.shape}")
        return (z - mean[None, None, :, None, None]) / std[None, None, :, None, None]

    elif z.dim() == 4:
        if z.shape[1] == c:
            # [B, C, H, W]
            return (z - mean[None, :, None, None]) / std[None, :, None, None]
        elif z.shape[-1] == c:
            # [B, K, N, C]
            return (z - mean[None, None, None, :]) / std[None, None, None, :]
        else:
            raise ValueError(f"Unsupported 4D latent shape={z.shape}, C={c}")

    elif z.dim() == 3:
        if z.shape[-1] == c:
            # [B, N, C]
            return (z - mean[None, None, :]) / std[None, None, :]
        elif z.shape[1] == c:
            # [B, C, N]
            return (z - mean[None, :, None]) / std[None, :, None]
        else:
            raise ValueError(f"Unsupported 3D latent shape={z.shape}, C={c}")

    elif z.dim() == 2:
        if z.shape[1] != c:
            raise ValueError(f"Expected [B,C] with C={c}, got {z.shape}")
        return (z - mean[None, :]) / std[None, :]

    else:
        raise ValueError(f"Unsupported latent shape: {z.shape}")


def flush_latent_cache(
    cache_files,
    buf_z,
    buf_y,
    rank_cache_dir,
    rank,
    group_id,
    local_start,
):
    if len(buf_z) == 0:
        return local_start, group_id

    z_cat = torch.cat(buf_z, dim=0).contiguous()
    y_cat = torch.cat(buf_y, dim=0).contiguous()

    local_end = local_start + z_cat.shape[0]

    path = os.path.join(rank_cache_dir, f"rank{rank:03d}_chunk{group_id:05d}.pt")

    torch.save(
        {
            "z": z_cat,
            "y": y_cat,
            "start": local_start,
            "end": local_end,
        },
        path,
    )

    cache_files.append(
        {
            "path": path,
            "start": local_start,
            "end": local_end,
            "num": z_cat.shape[0],
        }
    )

    buf_z.clear()
    buf_y.clear()

    return local_end, group_id + 1


def compute_local_topk_metric_from_cache(
    cache_files,
    z,
    y,
    latent_mean,
    latent_std,
    K,
    intp,
    metric_norm,
    pclass,
    device,
    chunk_size,
    metric_dtype,
):
    """
    Stream training latents from disk cache.
    Does not keep all zs in CPU/GPU memory.
    """
    y = y.to(device)
    z = z.to(device=device, dtype=metric_dtype, non_blocking=True)

    mean_d = latent_mean.to(device)
    std_d = latent_std.to(device)

    if metric_norm == "bn_channel":
        z_metric = normalize_latent_channelwise(
            z.float(),
            mean_d,
            std_d,
        ).to(metric_dtype)
    else:
        z_metric = z

    z_metric_flat = z_metric.reshape(z_metric.shape[0], -1)

    best_cost = None
    best_idx = None

    for meta in cache_files:
        pack = torch.load(meta["path"], map_location="cpu")
        z_all = pack["z"]
        y_all = pack["y"]
        base_start = int(pack["start"])

        for s in range(0, z_all.shape[0], chunk_size):
            e = min(s + chunk_size, z_all.shape[0])

            zs_chunk = z_all[s:e].to(
                device=device,
                dtype=metric_dtype,
                non_blocking=True,
            )

            if metric_norm == "bn_channel":
                zs_chunk_metric = normalize_latent_channelwise(
                    zs_chunk.float(),
                    mean_d,
                    std_d,
                ).to(metric_dtype)
            else:
                zs_chunk_metric = zs_chunk

            zs_chunk_flat = zs_chunk_metric.reshape(zs_chunk_metric.shape[0], -1)

            if intp == "linear":
                cost = (
                    (zs_chunk_flat ** 2).sum(1)[:, None]
                    + (z_metric_flat ** 2).sum(1)[None]
                    - 2 * zs_chunk_flat @ z_metric_flat.T
                )
            else:
                zs_norm = F.normalize(zs_chunk_flat, dim=1)
                z_norm = F.normalize(z_metric_flat, dim=1)
                cost = -(zs_norm @ z_norm.T)

            cost = cost.float()

            if pclass:
                ys_chunk = y_all[s:e].to(device)
                cost_y = 1e9 * torch.abs(ys_chunk[:, None] - y[None, :])
                cost = cost + cost_y

            k_now = min(K, cost.shape[0])
            chunk_cost, chunk_idx = torch.topk(cost, k=k_now, dim=0, largest=False)
            chunk_idx = chunk_idx + base_start + s

            if best_cost is None:
                best_cost = chunk_cost
                best_idx = chunk_idx
            else:
                merged_cost = torch.cat([best_cost, chunk_cost], dim=0)
                merged_idx = torch.cat([best_idx, chunk_idx], dim=0)

                best_cost, pos = torch.topk(
                    merged_cost,
                    k=K,
                    dim=0,
                    largest=False,
                )
                best_idx = torch.gather(merged_idx, dim=0, index=pos)

            del zs_chunk, zs_chunk_metric, zs_chunk_flat
            del cost, chunk_cost, chunk_idx

        del pack, z_all, y_all

    return best_cost, best_idx


def fetch_cached_z_by_indices(cache_files, indices):
    """
    indices: 1D CPU LongTensor, local indices into this rank's cached training latents.
    returns: CPU tensor [len(indices), ...]
    """
    indices = indices.cpu().long()
    out = None
    filled = torch.zeros(indices.shape[0], dtype=torch.bool)

    for meta in cache_files:
        start = int(meta["start"])
        end = int(meta["end"])

        mask = (indices >= start) & (indices < end)
        if not mask.any():
            continue

        pack = torch.load(meta["path"], map_location="cpu")
        z_all = pack["z"]

        local_idx = indices[mask] - start
        selected = z_all[local_idx]

        if out is None:
            out = torch.empty(
                indices.shape[0],
                *selected.shape[1:],
                dtype=selected.dtype,
                device="cpu",
            )

        out[mask] = selected
        filled[mask] = True

        del pack, z_all, selected

    if out is None or not filled.all():
        missing = indices[~filled]
        raise RuntimeError(f"Some cached indices were not found: {missing[:10]}")

    return out


def compute_local_topk_metric(
    zs,
    ys,
    z,
    y,
    latent_mean,
    latent_std,
    K,
    intp,
    metric_norm="none",
    pclass=False,
    device="cuda",
    chunk_size=256,
    metric_dtype=torch.float32,
):
    """
    Compute local topK nearest neighbors without materializing full zs_metric.

    zs:
      local training latent shard, shape [N, ...], can be CPU or CUDA.
    ys:
      local training labels, shape [N].
    z:
      current val latent batch, shape [B, ...], can be CPU or CUDA.
    y:
      current val labels, shape [B].

    Returns:
      local_cost: [K, B]
      local_idx:  [K, B], indices into local zs.
    """
    y = y.to(device)

    z = z.to(device=device, dtype=metric_dtype, non_blocking=True)

    if metric_norm == "bn_channel":
        z_metric = normalize_latent_channelwise(
            z.float(),
            latent_mean.to(device),
            latent_std.to(device),
        ).to(metric_dtype)
    else:
        z_metric = z

    z_metric_flat = z_metric.reshape(z_metric.shape[0], -1)  # [B, D]

    best_cost = None
    best_idx = None

    N = zs.shape[0]

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        zs_chunk = zs[start:end].to(
            device=device,
            dtype=metric_dtype,
            non_blocking=True,
        )

        if metric_norm == "bn_channel":
            zs_chunk_metric = normalize_latent_channelwise(
                zs_chunk.float(),
                latent_mean.to(device),
                latent_std.to(device),
            ).to(metric_dtype)
        else:
            zs_chunk_metric = zs_chunk

        zs_chunk_flat = zs_chunk_metric.reshape(zs_chunk_metric.shape[0], -1)

        if intp == "linear":
            cost = (
                (zs_chunk_flat ** 2).sum(1)[:, None]
                + (z_metric_flat ** 2).sum(1)[None]
                - 2 * zs_chunk_flat @ z_metric_flat.T
            )
        else:
            zs_norm = F.normalize(zs_chunk_flat, dim=1)
            z_norm = F.normalize(z_metric_flat, dim=1)
            cost = -(zs_norm @ z_norm.T)

        cost = cost.float()  # [chunk, B]

        if pclass:
            ys_chunk = ys[start:end].to(device)
            cost_y = 1e9 * torch.abs(ys_chunk[:, None] - y[None, :])
            cost = cost + cost_y

        k_now = min(K, cost.shape[0])
        chunk_cost, chunk_idx = torch.topk(
            cost,
            k=k_now,
            dim=0,
            largest=False,
        )
        chunk_idx = chunk_idx + start

        if best_cost is None:
            best_cost = chunk_cost
            best_idx = chunk_idx
        else:
            merged_cost = torch.cat([best_cost, chunk_cost], dim=0)
            merged_idx = torch.cat([best_idx, chunk_idx], dim=0)

            best_cost, pos = torch.topk(
                merged_cost,
                k=min(K, merged_cost.shape[0]),
                dim=0,
                largest=False,
            )
            best_idx = torch.gather(merged_idx, dim=0, index=pos)

        del zs_chunk, zs_chunk_metric, zs_chunk_flat, cost, chunk_cost, chunk_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_cost, best_idx




def discover_existing_cache(rank_cache_dir, rank, cache_group_size, bs):
    """
    Reuse an existing cache produced by this script without rewriting it.

    For the normal 1M/4-GPU setting each cache file except possibly the last
    contains cache_group_size * bs rows. We validate the first/last files and
    build the same contiguous local-index metadata used by the original path.
    """
    paths = sorted(
        os.path.join(rank_cache_dir, x)
        for x in os.listdir(rank_cache_dir)
        if x.startswith(f"rank{rank:03d}_chunk") and x.endswith(".pt")
    )
    if not paths:
        raise RuntimeError(f"No cache chunks found in {rank_cache_dir}")

    first = torch.load(paths[0], map_location="cpu")
    latent_shape = tuple(first["z"].shape[1:])
    first_start = int(first["start"])
    first_end = int(first["end"])
    if first_start != 0:
        raise RuntimeError(f"Unexpected first cache start={first_start}")
    nominal_num = first_end - first_start
    expected_nominal = cache_group_size * bs
    if nominal_num != expected_nominal and len(paths) > 1:
        raise RuntimeError(
            f"Unexpected cache chunk size={nominal_num}, expected={expected_nominal}"
        )
    del first

    # Every non-final chunk is created by the same flush threshold.
    cache_files = []
    local_start = 0
    for i, path in enumerate(paths):
        if i == len(paths) - 1:
            last = torch.load(path, map_location="cpu")
            start = int(last["start"])
            end = int(last["end"])
            num = int(last["z"].shape[0])
            if start != local_start or end - start != num:
                raise RuntimeError(
                    f"Non-contiguous final cache chunk: start={start}, "
                    f"expected={local_start}, end={end}, num={num}"
                )
            del last
        else:
            start = local_start
            num = nominal_num
            end = start + num

        cache_files.append(
            {"path": path, "start": start, "end": end, "num": num}
        )
        local_start = end

    return cache_files, latent_shape, local_start


def compute_local_topk_metric_from_cache_blocked(
    cache_files,
    query_records,
    latent_mean,
    latent_std,
    K,
    intp,
    metric_norm,
    pclass,
    device,
    chunk_size,
    metric_dtype,
):
    """
    Exact-math blocked version of compute_local_topk_metric_from_cache.

    Key property: each reference chunk is loaded and moved to GPU once, then
    the *original validation batches* are processed one by one against that
    chunk. We do NOT concatenate validation batches into a wider GEMM. Thus
    each distance matmul keeps the same [ref_chunk,D] x [D,bs] shape and the
    same reference-chunk merge order as the original implementation.

    This changes loop nesting only:
      old: for query_batch -> for every reference chunk
      new: for query_block -> for every reference chunk -> for query_batch

    Retrieval definition, FP32 arithmetic, K, metric, and top-k merge order are
    unchanged.
    """
    mean_d = latent_mean.to(device)
    std_d = latent_std.to(device)

    states = []
    for rec in query_records:
        z = rec["z"].to(device=device, dtype=metric_dtype, non_blocking=True)
        y = rec["y"].to(device)

        if metric_norm == "bn_channel":
            z_metric = normalize_latent_channelwise(
                z.float(), mean_d, std_d
            ).to(metric_dtype)
        else:
            z_metric = z

        z_metric_flat = z_metric.reshape(z_metric.shape[0], -1)
        if intp == "linear":
            z_norm = None
            z_sq = (z_metric_flat ** 2).sum(1)
        else:
            z_norm = F.normalize(z_metric_flat, dim=1)
            z_sq = None

        states.append(
            {
                "y": y,
                "z_metric_flat": z_metric_flat,
                "z_norm": z_norm,
                "z_sq": z_sq,
                "best_cost": None,
                "best_idx": None,
            }
        )

    for meta in cache_files:
        pack = torch.load(meta["path"], map_location="cpu")
        z_all = pack["z"]
        y_all = pack["y"]
        base_start = int(pack["start"])

        for s in range(0, z_all.shape[0], chunk_size):
            e = min(s + chunk_size, z_all.shape[0])

            zs_chunk = z_all[s:e].to(
                device=device,
                dtype=metric_dtype,
                non_blocking=True,
            )

            if metric_norm == "bn_channel":
                zs_chunk_metric = normalize_latent_channelwise(
                    zs_chunk.float(), mean_d, std_d
                ).to(metric_dtype)
            else:
                zs_chunk_metric = zs_chunk

            zs_chunk_flat = zs_chunk_metric.reshape(zs_chunk_metric.shape[0], -1)

            if intp == "linear":
                zs_sq = (zs_chunk_flat ** 2).sum(1)
                zs_norm = None
            else:
                zs_sq = None
                # Same F.normalize operation as original, computed once because
                # it is independent of the query batch.
                zs_norm = F.normalize(zs_chunk_flat, dim=1)

            ys_chunk = y_all[s:e].to(device) if pclass else None

            # IMPORTANT: process the original bs-sized query batches separately.
            for st in states:
                if intp == "linear":
                    cost = (
                        zs_sq[:, None]
                        + st["z_sq"][None]
                        - 2 * zs_chunk_flat @ st["z_metric_flat"].T
                    )
                else:
                    cost = -(zs_norm @ st["z_norm"].T)

                cost = cost.float()

                if pclass:
                    cost_y = 1e9 * torch.abs(
                        ys_chunk[:, None] - st["y"][None, :]
                    )
                    cost = cost + cost_y

                k_now = min(K, cost.shape[0])
                chunk_cost, chunk_idx = torch.topk(
                    cost, k=k_now, dim=0, largest=False
                )
                chunk_idx = chunk_idx + base_start + s

                if st["best_cost"] is None:
                    st["best_cost"] = chunk_cost
                    st["best_idx"] = chunk_idx
                else:
                    merged_cost = torch.cat(
                        [st["best_cost"], chunk_cost], dim=0
                    )
                    merged_idx = torch.cat(
                        [st["best_idx"], chunk_idx], dim=0
                    )
                    st["best_cost"], pos = torch.topk(
                        merged_cost,
                        k=K,
                        dim=0,
                        largest=False,
                    )
                    st["best_idx"] = torch.gather(
                        merged_idx, dim=0, index=pos
                    )

                del cost, chunk_cost, chunk_idx

            del zs_chunk, zs_chunk_metric, zs_chunk_flat, zs_sq, zs_norm
            if ys_chunk is not None:
                del ys_chunk

        del pack, z_all, y_all

    results = [(st["best_cost"], st["best_idx"]) for st in states]
    return results


def fetch_cached_candidates_for_block(
    cache_files,
    requests,
    rank,
    latent_shape,
    cache_dtype,
):
    """
    Fetch raw top-K candidates for a whole query block with one pass over the
    local cache files, instead of re-opening tens of large .pt files for every
    validation batch.

    `requests` contains the exact global-topK routing produced by the original
    code for each original validation batch. The returned tensors have exactly
    the same [B,K,*latent_shape] layout as the original per-batch fetch path.
    """
    outputs = []
    req_indices = []
    req_batch_ids = []
    req_flat_pos = []

    for bid, req in enumerate(requests):
        B = req["batch_size"]
        K = req["K"]
        out = torch.zeros(
            B,
            K,
            *latent_shape,
            dtype=cache_dtype,
            device="cpu",
        )
        outputs.append(out)

        mask = req["gpu_id_cpu"] == rank  # [B,K]
        if not mask.any():
            continue

        nz = mask.nonzero(as_tuple=False)  # row-major [n,2] => (batch,k)
        batch_indices = nz[:, 0]
        k_positions = nz[:, 1]

        local_k_values = req["local_k_cpu"][mask]
        idx = req["all_idx_cpu"][rank][local_k_values, batch_indices]
        flat_pos = batch_indices * K + k_positions

        req_indices.append(idx.long())
        req_batch_ids.append(torch.full_like(idx.long(), bid))
        req_flat_pos.append(flat_pos.long())

    if not req_indices:
        return outputs

    req_indices = torch.cat(req_indices)
    req_batch_ids = torch.cat(req_batch_ids)
    req_flat_pos = torch.cat(req_flat_pos)

    # Sorting makes each cache file touch a contiguous portion of requests and
    # lets us load every required large .pt file at most once per query block.
    order = torch.argsort(req_indices)
    req_indices = req_indices[order]
    req_batch_ids = req_batch_ids[order]
    req_flat_pos = req_flat_pos[order]

    for meta in cache_files:
        start = int(meta["start"])
        end = int(meta["end"])

        lo = int(torch.searchsorted(req_indices, torch.tensor(start), right=False))
        hi = int(torch.searchsorted(req_indices, torch.tensor(end), right=False))
        if lo >= hi:
            continue

        pack = torch.load(meta["path"], map_location="cpu")
        z_all = pack["z"]

        local_idx = req_indices[lo:hi] - start
        selected = z_all[local_idx]
        bids = req_batch_ids[lo:hi]
        flats = req_flat_pos[lo:hi]

        for bid in torch.unique(bids).tolist():
            m = bids == bid
            flat_out = outputs[bid].view(
                outputs[bid].shape[0] * outputs[bid].shape[1],
                *latent_shape,
            )
            flat_out[flats[m]] = selected[m]

        del pack, z_all, selected, bids, flats

    return outputs


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent-cache-dir", type=str, default="./latent_cache_fix")
    parser.add_argument("--cache-group-size", type=int, default=20)
    parser.add_argument(
        "--query-block-batches",
        type=int,
        default=20,
        help=(
            "Number of original validation batches sharing one reference-cache scan. "
            "Does not change the per-batch distance GEMM shape. For bs=25, 20 means "
            "500 queries per block."
        ),
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse an already-built latent cache under --latent-cache-dir.",
    )
    parser.add_argument(
        "--verify-first-batch",
        action="store_true",
        help=(
            "On the first query block, re-run the original one-batch disk scan "
            "and assert identical top-K indices plus matching candidate tensors. "
            "Costs one extra reference scan, but is useful for paper-number validation."
        ),
    )
    parser.add_argument(
        "--cache-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )

    parser.add_argument(
        "--metric-norm",
        type=str,
        default="none",
        choices=["none", "bn_channel"],
    )
    parser.add_argument(
        "--metric-chunk-size",
        type=int,
        default=256,
        help="Chunk size for training latent metric topK search.",
    )
    parser.add_argument(
        "--metric-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Computation dtype for chunked metric search. fp32 is most faithful.",
    )

    parser.add_argument("--vae-config", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=25)
    parser.add_argument(
        "--intp",
        type=str,
        default="slerp",
        choices=["linear", "slerp", "mask"],
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--exp-name", type=str, default="ifid")
    parser.add_argument("--small", type=int, default=200000)
    parser.add_argument("--small_val", type=int, default=-1)
    parser.add_argument("--sample-dir", type=str, default="./samples")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--dataset-ref", type=str, default="")
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-save", action="store_true")
    parser.add_argument("-pclass", action="store_true")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-max", type=int, default=100)

    return parser.parse_args()


# -------------------------
# SLERP
# -------------------------
def slerp(z1, z2, t, eps=1e-7):
    """
    z1, z2: same shape.
    t: float in [0, 1].
    """
    B = z1.shape[0]

    z1f = z1.reshape(B, -1)
    z2f = z2.reshape(B, -1)

    z1n = z1f / (z1f.norm(dim=1, keepdim=True) + eps)
    z2n = z2f / (z2f.norm(dim=1, keepdim=True) + eps)

    dot = (z1n * z2n).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w2 = torch.sin(t * theta) / (sin_theta + eps)

    zt = w1.unsqueeze(1) * z1f + w2.unsqueeze(1) * z2f

    small_t = sin_theta < 1e-6
    if small_t.any():
        zt[small_t] = (1 - t) * z1f[small_t] + t * z2f[small_t]

    return zt.reshape_as(z1)


# -------------------------
# Main
# -------------------------
def main(args):
    sample_folder_dir = os.path.join(args.sample_dir, args.exp_name)
    sub_folder_dir = os.path.join(args.sample_dir, args.exp_name + "_extra")
    src_dir = os.path.join(sub_folder_dir, "src_dist")
    nn_dir = os.path.join(sub_folder_dir, "nn_dist")
    recon_dir = os.path.join(sub_folder_dir, "recon_dist")
    intp_dir = os.path.join(sub_folder_dir, "intp_dist")

    os.makedirs(sample_folder_dir, exist_ok=True)
    os.makedirs(sub_folder_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(nn_dir, exist_ok=True)
    os.makedirs(intp_dir, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device
    set_seed(args.seed + accelerator.process_index)

    metric_dtype = get_metric_dtype(args.metric_dtype)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    train_set = ImageNetValDataset(args.dataset_ref, transform, args.small)
    val_set = ImageNetValDataset(args.dataset, transform, args.small_val)

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=8,
    )

    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for _, param in vae.named_parameters():
        param.requires_grad_(False)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    train_loader, vae, inception = accelerator.prepare(train_loader, vae, inception)

    # -------------------------
    # Encode training latents to DISK CACHE, or reuse an existing cache.
    # -------------------------
    cache_dtype = get_cache_dtype(args.cache_dtype)

    rank = accelerator.process_index
    world_size = accelerator.num_processes

    cache_root = os.path.join(
        args.latent_cache_dir,
        f"{args.exp_name}_small{args.small}_seed{args.seed}",
    )
    rank_cache_dir = os.path.join(cache_root, f"rank{rank:03d}")

    if args.reuse_cache:
        if not os.path.isdir(rank_cache_dir):
            raise RuntimeError(
                f"--reuse-cache requested but cache directory does not exist: "
                f"{rank_cache_dir}"
            )
        cache_files, latent_shape, local_start = discover_existing_cache(
            rank_cache_dir=rank_cache_dir,
            rank=rank,
            cache_group_size=args.cache_group_size,
            bs=args.bs,
        )
        if args.small % world_size == 0:
            expected_local = args.small // world_size
            if local_start != expected_local:
                raise RuntimeError(
                    f"Incomplete cache on rank {rank}: found {local_start} rows, "
                    f"expected {expected_local}. Refusing --reuse-cache."
                )

        if args.metric_norm != "none":
            raise RuntimeError(
                "--reuse-cache currently guarantees exact behavior for "
                "--metric-norm none, which is the paper iFID setting."
            )
        # These are unused when metric_norm == none.
        latent_mean = torch.zeros(1, device=device, dtype=torch.float32)
        latent_std = torch.ones(1, device=device, dtype=torch.float32)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("[REUSE CACHE]", cache_root)
            print("cache files per rank:", len(cache_files))
            print("local latent shape:", latent_shape)
            print("local cached num:", local_start)
    else:
        if os.path.exists(rank_cache_dir):
            shutil.rmtree(rank_cache_dir)
        os.makedirs(rank_cache_dir, exist_ok=True)

        cache_files = []
        buf_z, buf_y = [], []
        local_start = 0
        group_id = 0
        latent_shape = None

        stat_sum = None
        stat_sumsq = None
        stat_count = 0

        for img, y, name in tqdm(
            train_loader,
            disable=not accelerator.is_main_process,
            desc="encode reference cache",
        ):
            img = img.to(device)
            y = y.to(device)

            with torch.no_grad():
                try:
                    z = vae.encode(img, y)
                except TypeError:
                    z = vae.encode(img)

            if latent_shape is None:
                latent_shape = tuple(z.shape[1:])

            s_stat, ss, cnt = latent_channel_sum_sumsq_count(z)

            if stat_sum is None:
                stat_sum = torch.zeros_like(
                    s_stat, device=device, dtype=torch.float64
                )
                stat_sumsq = torch.zeros_like(
                    ss, device=device, dtype=torch.float64
                )

            stat_sum += s_stat.to(device=device, dtype=torch.float64)
            stat_sumsq += ss.to(device=device, dtype=torch.float64)
            stat_count += cnt

            buf_z.append(z.detach().cpu().to(cache_dtype))
            buf_y.append(y.detach().cpu())

            if len(buf_z) >= args.cache_group_size:
                local_start, group_id = flush_latent_cache(
                    cache_files=cache_files,
                    buf_z=buf_z,
                    buf_y=buf_y,
                    rank_cache_dir=rank_cache_dir,
                    rank=rank,
                    group_id=group_id,
                    local_start=local_start,
                )

        local_start, group_id = flush_latent_cache(
            cache_files=cache_files,
            buf_z=buf_z,
            buf_y=buf_y,
            rank_cache_dir=rank_cache_dir,
            rank=rank,
            group_id=group_id,
            local_start=local_start,
        )

        accelerator.wait_for_everyone()

        stat_count_tensor = torch.tensor(
            [stat_count], device=device, dtype=torch.float64
        )
        dist.all_reduce(stat_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(stat_sumsq, op=dist.ReduceOp.SUM)
        dist.all_reduce(stat_count_tensor, op=dist.ReduceOp.SUM)

        latent_mean = stat_sum / stat_count_tensor
        latent_var = stat_sumsq / stat_count_tensor - latent_mean ** 2
        latent_std = torch.sqrt(torch.clamp(latent_var, min=1e-6))
        latent_mean = latent_mean.float()
        latent_std = latent_std.float()

        if accelerator.is_main_process:
            print(
                f"metric_norm={args.metric_norm}, intp={args.intp}, "
                f"alpha={args.alpha}, top={args.top}, "
                f"cache_dtype={args.cache_dtype}, "
                f"cache_group_size={args.cache_group_size}"
            )
            print("cache_root:", cache_root)
            print("local latent shape:", latent_shape)
            print("local cached num:", local_start)
            if args.metric_norm == "bn_channel":
                print(
                    "latent_mean:", latent_mean.mean().item(),
                    "latent_std_mean:", latent_std.mean().item(),
                    "latent_std_min:", latent_std.min().item(),
                    "latent_std_max:", latent_std.max().item(),
                )

    # -------------------------
    # Validation loop -- BLOCKED REFERENCE SCAN
    # -------------------------
    pred_arr_in = []
    pred_arr_rec = []
    pred_arr_znn = []
    arr_psnr = []
    arr_ssim = []
    arr_msssim = []
    arr_lpips = []
    ents = []

    global_seen = 0
    saved_count = 0

    val_iter = iter(val_loader)
    total_val_batches = len(val_loader)
    done_val_batches = 0

    pbar = tqdm(
        total=total_val_batches,
        disable=not accelerator.is_main_process,
        desc="blocked val",
    )
    first_query_block = True

    while done_val_batches < total_val_batches:
        n_this_block = min(
            args.query_block_batches,
            total_val_batches - done_val_batches,
        )

        # Encode the next block using the ORIGINAL batch size/order.
        # For RAE/SVG/SVG-T2I the encoders are deterministic in eval mode.
        query_records = []
        for _ in range(n_this_block):
            img, y, name = next(val_iter)
            img = img.to(device)
            y = y.to(device)

            with torch.no_grad():
                try:
                    z = vae.encode(img, y)
                except TypeError:
                    z = vae.encode(img)

            if args.cpu:
                z = z.cpu().to(torch.bfloat16)

            query_records.append(
                {"img": img, "y": y, "name": name, "z": z}
            )

        K = args.top

        # One sequential reference-cache pass services the whole block while
        # preserving each original bs-sized distance computation separately.
        local_results = compute_local_topk_metric_from_cache_blocked(
            cache_files=cache_files,
            query_records=query_records,
            latent_mean=latent_mean,
            latent_std=latent_std,
            K=K,
            intp=args.intp,
            metric_norm=args.metric_norm,
            pclass=args.pclass,
            device=device,
            chunk_size=args.metric_chunk_size,
            metric_dtype=metric_dtype,
        )

        if args.verify_first_batch and first_query_block:
            verify_cost, verify_idx = compute_local_topk_metric_from_cache(
                cache_files=cache_files,
                z=query_records[0]["z"],
                y=query_records[0]["y"],
                latent_mean=latent_mean,
                latent_std=latent_std,
                K=K,
                intp=args.intp,
                metric_norm=args.metric_norm,
                pclass=args.pclass,
                device=device,
                chunk_size=args.metric_chunk_size,
                metric_dtype=metric_dtype,
            )
            blocked_cost, blocked_idx = local_results[0]
            if not torch.equal(verify_idx, blocked_idx):
                mismatch = int((verify_idx != blocked_idx).sum().item())
                raise RuntimeError(
                    f"Blocked retrieval verification FAILED: {mismatch} top-K "
                    f"indices differ on rank {rank}."
                )
            max_cost_diff = float(
                (verify_cost - blocked_cost).abs().max().item()
            )
            if max_cost_diff > 1e-6:
                raise RuntimeError(
                    f"Blocked retrieval verification FAILED: max cost diff="
                    f"{max_cost_diff:.3e} on rank {rank}."
                )
            if accelerator.is_main_process:
                print(
                    "[VERIFY PASS] blocked vs original local top-K: "
                    f"indices identical, max_cost_diff={max_cost_diff:.3e}"
                )
            del verify_cost, verify_idx

        # Reproduce the original per-batch global top-K routing, in order.
        requests = []
        for rec, (local_cost, local_idx) in zip(query_records, local_results):
            z = rec["z"]

            all_cost = accelerator.gather(local_cost)
            all_cost = all_cost.reshape(world_size, K, -1)

            all_idx = accelerator.gather(local_idx)
            all_idx = all_idx.reshape(world_size, K, -1)

            all_cost_flat = all_cost.permute(2, 0, 1).reshape(
                z.shape[0], -1
            )
            _, global_pos = torch.topk(
                all_cost_flat,
                k=K,
                dim=1,
                largest=False,
            )

            gpu_id = global_pos // K
            local_k = global_pos % K

            requests.append(
                {
                    "K": K,
                    "batch_size": z.shape[0],
                    "all_idx_cpu": all_idx.cpu(),
                    "local_k_cpu": local_k.cpu(),
                    "gpu_id_cpu": gpu_id.cpu(),
                }
            )

            del local_cost, local_idx, all_cost, all_idx, all_cost_flat
            del global_pos, gpu_id, local_k

        # Candidate fetch is blocked too: every needed .pt cache file is opened
        # at most once for this query block.
        candidate_cpu_list = fetch_cached_candidates_for_block(
            cache_files=cache_files,
            requests=requests,
            rank=rank,
            latent_shape=latent_shape,
            cache_dtype=cache_dtype,
        )

        if args.verify_first_batch and first_query_block:
            req = requests[0]
            B = req["batch_size"]
            old_candidate_cpu = torch.zeros(
                B, K, *latent_shape, dtype=cache_dtype, device="cpu"
            )
            mask = req["gpu_id_cpu"] == rank
            if mask.any():
                batch_indices = mask.nonzero(as_tuple=True)[0]
                idx = req["all_idx_cpu"][rank][
                    req["local_k_cpu"][mask], batch_indices
                ]
                selected_z = fetch_cached_z_by_indices(cache_files, idx)
                old_candidate_cpu[mask] = selected_z.to(cache_dtype)
                del selected_z
            if not torch.equal(old_candidate_cpu, candidate_cpu_list[0]):
                diff = float(
                    (old_candidate_cpu - candidate_cpu_list[0])
                    .abs().max().item()
                )
                raise RuntimeError(
                    f"Blocked candidate fetch verification FAILED on rank {rank}, "
                    f"max diff={diff:.3e}."
                )
            if accelerator.is_main_process:
                print("[VERIFY PASS] blocked candidate fetch is tensor-identical")
            del old_candidate_cpu

        # Finish each ORIGINAL validation batch in the original order. This
        # preserves Categorical sampling order and decode/FID accumulation order.
        for rec, znn_candidate_cpu in zip(query_records, candidate_cpu_list):
            img = rec["img"]
            y = rec["y"]
            name = rec["name"]
            z = rec["z"].to(device)

            znn_candidate = znn_candidate_cpu.to(device)
            dist.all_reduce(znn_candidate, op=dist.ReduceOp.SUM)

            znn_candidate_flat = znn_candidate.reshape(z.shape[0], K, -1)

            if args.metric_norm == "bn_channel":
                znn_candidate_for_weight = normalize_latent_channelwise(
                    znn_candidate.float(),
                    latent_mean.to(device),
                    latent_std.to(device),
                )
                z_for_weight = normalize_latent_channelwise(
                    z.float(),
                    latent_mean.to(device),
                    latent_std.to(device),
                )
            else:
                znn_candidate_for_weight = znn_candidate
                z_for_weight = z

            znn_candidate_metric_flat = znn_candidate_for_weight.reshape(
                z.shape[0], K, -1
            )
            z_metric_flat_for_weight = z_for_weight.reshape(z.shape[0], -1)

            local_cost_for_weight = torch.mean(
                (
                    z_metric_flat_for_weight[:, None]
                    - znn_candidate_metric_flat
                ) ** 2,
                dim=-1,
            ).to(torch.float32)

            weights = torch.softmax(-local_cost_for_weight, dim=1)
            cat_dist = torch.distributions.Categorical(probs=weights)
            ent = cat_dist.entropy()
            ents.append(torch.mean(ent).item())

            znn_idx = cat_dist.sample()
            weights = F.one_hot(
                znn_idx, num_classes=K
            ).to(znn_candidate_flat.dtype)

            znn = torch.sum(
                weights[:, :, None] * znn_candidate_flat,
                dim=1,
            ).reshape(*z.shape)

            if args.intp == "linear":
                z_intp = (1 - args.alpha) * z + args.alpha * znn
            elif args.intp == "slerp":
                z_intp = slerp(z, znn, args.alpha)
            elif args.intp == "mask":
                if len(z.shape) == 4:
                    mask = torch.zeros(
                        [z.shape[0], 1, z.shape[2], z.shape[3]],
                        device=device,
                        dtype=z.dtype,
                    )
                    mask[:, :, : z.shape[2] // 2, :] += 1
                else:
                    mask = torch.zeros(
                        [z.shape[0], z.shape[1], 1],
                        device=device,
                        dtype=z.dtype,
                    )
                    mask[:, : z.shape[1] // 2, :] += 1
                z_intp = (1 - mask) * z + mask * znn
            else:
                raise ValueError(f"Unsupported intp: {args.intp}")

            with torch.no_grad():
                img_in = img
                raw_vae = accelerator.unwrap_model(vae)

                try:
                    img_recon = raw_vae.decode(z, y)
                    img_intp = raw_vae.decode(z_intp, y)
                    img_nn = raw_vae.decode(znn, y)
                except TypeError:
                    img_recon = raw_vae.decode(z)
                    img_intp = raw_vae.decode(z_intp)
                    img_nn = raw_vae.decode(znn)

                img_in = (img_in + 1) / 2
                img_recon = (img_recon + 1) / 2
                img_intp = (img_intp + 1) / 2
                img_nn = (img_nn + 1) / 2

                if args.save and accelerator.is_main_process:
                    img_nn_np = (
                        torch.clamp(255.0 * img_nn, 0, 255)
                        .permute(0, 2, 3, 1)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )
                    img_recon_np = (
                        torch.clamp(255.0 * img_recon, 0, 255)
                        .permute(0, 2, 3, 1)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )
                    img_in_np = (
                        torch.clamp(255.0 * img_in, 0, 255)
                        .permute(0, 2, 3, 1)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )
                    img_intp_np = (
                        torch.clamp(255.0 * img_intp, 0, 255)
                        .permute(0, 2, 3, 1)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )

                    for j in range(len(name)):
                        should_save = (
                            (args.save_max < 0 or saved_count < args.save_max)
                            and (global_seen % args.save_every == 0)
                        )
                        if should_save:
                            Image.fromarray(img_in_np[j]).save(
                                f"{src_dir}/{name[j]}"
                            )
                            Image.fromarray(img_recon_np[j]).save(
                                f"{recon_dir}/{name[j]}"
                            )
                            Image.fromarray(img_nn_np[j]).save(
                                f"{nn_dir}/{name[j]}"
                            )
                            Image.fromarray(img_intp_np[j]).save(
                                f"{intp_dir}/{name[j]}"
                            )
                            saved_count += 1
                        global_seen += 1

                pred_psnr = get_psnr(img_in, img_recon, zero_mean=True)
                pred_ssim, pred_msssim = get_ssim_and_msssim(
                    img_in, img_recon, zero_mean=True
                )
                pred_lpips = get_lpips(
                    img_in,
                    img_recon,
                    zero_mean=True,
                    network_type="alex",
                )

                arr_psnr.append(torch.mean(pred_psnr).item())
                arr_ssim.append(torch.mean(pred_ssim).item())
                arr_msssim.append(torch.mean(pred_msssim).item())
                arr_lpips.append(torch.mean(pred_lpips).item())

                pred_in = inception(img_in)[0]
                pred_rec = inception(img_recon)[0]
                pred_intp = inception(img_intp)[0]

                if pred_in.size(2) != 1:
                    pred_in = adaptive_avg_pool2d(pred_in, (1, 1))
                if pred_rec.size(2) != 1:
                    pred_rec = adaptive_avg_pool2d(pred_rec, (1, 1))
                if pred_intp.size(2) != 1:
                    pred_intp = adaptive_avg_pool2d(pred_intp, (1, 1))

                pred_arr_in.append(
                    pred_in.squeeze(-1).squeeze(-1).cpu().numpy()
                )
                pred_arr_rec.append(
                    pred_rec.squeeze(-1).squeeze(-1).cpu().numpy()
                )
                pred_arr_znn.append(
                    pred_intp.squeeze(-1).squeeze(-1).cpu().numpy()
                )

            del znn_candidate, znn_candidate_flat
            del znn_candidate_for_weight, z_for_weight
            del znn_candidate_metric_flat, z_metric_flat_for_weight
            del local_cost_for_weight, weights, znn, z_intp
            del img_recon, img_intp, img_nn, pred_psnr, pred_ssim
            del pred_msssim, pred_lpips, pred_in, pred_rec, pred_intp

            done_val_batches += 1
            pbar.update(1)

        del query_records, local_results, requests, candidate_cpu_list
        # No per-validation-batch empty_cache(): it synchronizes the allocator
        # and was a major source of bubbles. Explicit tensor deletion above is
        # sufficient; clear once per query block only.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        first_query_block = False

    pbar.close()

    pred_arr_in = np.concatenate(pred_arr_in)
    pred_arr_rec = np.concatenate(pred_arr_rec)
    pred_arr_znn = np.concatenate(pred_arr_znn)

    if accelerator.is_main_process:
        mu_in = pred_arr_in.mean(0)
        sigma_in = np.cov(pred_arr_in, rowvar=False)

        mu_rec = pred_arr_rec.mean(0)
        sigma_rec = np.cov(pred_arr_rec, rowvar=False)

        mu_znn = pred_arr_znn.mean(0)
        sigma_znn = np.cov(pred_arr_znn, rowvar=False)

        ifid = calculate_frechet_distance(mu_in, sigma_in, mu_znn, sigma_znn)
        rfid = calculate_frechet_distance(mu_in, sigma_in, mu_rec, sigma_rec)

        print("ifid={0:.4f}".format(ifid))
        print("rfid={0:.4f}".format(rfid))
        print("psnr={0:.4f}".format(np.mean(arr_psnr)))
        print("ssim={0:.4f}".format(np.mean(arr_ssim)))
        print("msssim={0:.4f}".format(np.mean(arr_msssim)))
        print("lpips={0:.4f}".format(np.mean(arr_lpips)))
        if len(ents) > 0:
            print("topk_entropy={0:.4f}".format(np.mean(ents)))


if __name__ == "__main__":
    args = parse_args()
    print("evaluating", args.exp_name)
    with torch.no_grad():
        main(args)
