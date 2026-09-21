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



def discover_existing_cache_resharded(
    cache_root,
    physical_rank,
    physical_world_size,
    cache_world_size,
    cache_group_size,
    bs,
):
    """
    Zero-copy reshard an existing cache built with fewer ranks.

    Example: cache_world_size=4, physical_world_size=8.
      physical 0/1 -> old logical rank 0, first/second 125k rows
      physical 2/3 -> old logical rank 1, first/second 125k rows
      ...

    No .pt tensor is rewritten or copied. We only select a contiguous subset
    of whole cache files and remap their *metadata* local indices to start at 0.
    For exact paper runs we require split boundaries to coincide with cache-file
    boundaries; with bs=25 and cache_group_size=20, each file has 500 rows and
    250k -> 125k is exactly aligned.
    """
    if physical_world_size % cache_world_size != 0:
        raise RuntimeError(
            f"physical world size {physical_world_size} must be divisible by "
            f"cache world size {cache_world_size}"
        )

    shard_factor = physical_world_size // cache_world_size
    logical_rank = physical_rank // shard_factor
    shard_id = physical_rank % shard_factor

    source_dir = os.path.join(cache_root, f"rank{logical_rank:03d}")
    if not os.path.isdir(source_dir):
        raise RuntimeError(
            f"Legacy cache directory missing for logical rank {logical_rank}: "
            f"{source_dir}"
        )

    full_files, latent_shape, full_count = discover_existing_cache(
        rank_cache_dir=source_dir,
        rank=logical_rank,
        cache_group_size=cache_group_size,
        bs=bs,
    )

    if full_count % shard_factor != 0:
        raise RuntimeError(
            f"Legacy rank {logical_rank} has {full_count} rows, not divisible "
            f"by shard_factor={shard_factor}"
        )

    shard_count = full_count // shard_factor
    shard_begin = shard_id * shard_count
    shard_end = shard_begin + shard_count

    selected = []
    for meta in full_files:
        start = int(meta["start"])
        end = int(meta["end"])
        overlaps = end > shard_begin and start < shard_end
        if not overlaps:
            continue
        # We intentionally refuse partial-file slicing. It is possible, but
        # whole-file boundaries keep the implementation and exactness audit
        # simple and avoid ever rewriting/copying the 700+ GiB cache.
        if start < shard_begin or end > shard_end:
            raise RuntimeError(
                "Legacy cache split boundary cuts through a cache file: "
                f"file=[{start},{end}), shard=[{shard_begin},{shard_end}). "
                "Choose a compatible physical/cache world-size pair."
            )
        selected.append(
            {
                "path": meta["path"],
                "start": start - shard_begin,
                "end": end - shard_begin,
                "num": int(meta["num"]),
            }
        )

    if not selected:
        raise RuntimeError(
            f"No files selected for physical rank {physical_rank} from "
            f"logical rank {logical_rank}"
        )

    selected_count = sum(int(m["num"]) for m in selected)
    if selected_count != shard_count:
        raise RuntimeError(
            f"Resharded physical rank {physical_rank}: selected {selected_count} "
            f"rows, expected {shard_count}"
        )

    # Check remapped metadata is contiguous 0..shard_count.
    cursor = 0
    for meta in selected:
        if int(meta["start"]) != cursor:
            raise RuntimeError(
                f"Non-contiguous remapped cache on physical rank {physical_rank}: "
                f"got start={meta['start']}, expected={cursor}"
            )
        cursor = int(meta["end"])
    if cursor != shard_count:
        raise RuntimeError(
            f"Remapped cache ends at {cursor}, expected {shard_count}"
        )

    return {
        "cache_files": selected,
        "latent_shape": latent_shape,
        "local_count": shard_count,
        "logical_rank": logical_rank,
        "logical_offset": shard_begin,
        "logical_full_count": full_count,
        "full_cache_files": full_files,
        "shard_factor": shard_factor,
        "shard_id": shard_id,
        "source_dir": source_dir,
    }


def merge_physical_topk_to_logical(
    accelerator,
    local_cost,
    local_idx,
    K,
    physical_world_size,
    logical_world_size,
    shard_factor,
    logical_offset,
):
    """
    Gather physical-shard top-K results and reconstruct the old logical-rank
    top-K layout before the global top-K is computed.

    local_idx is first converted from physical-shard-local indexing back to the
    original logical-cache index via logical_offset. The final tensors are
    [logical_world_size, K, B], in exactly the same logical rank order as the
    original cache world size (e.g. old ranks 0,1,2,3).
    """
    local_idx_original = local_idx + int(logical_offset)

    all_phys_cost = accelerator.gather(local_cost).reshape(
        physical_world_size, K, -1
    )
    all_phys_idx = accelerator.gather(local_idx_original).reshape(
        physical_world_size, K, -1
    )

    logical_costs = []
    logical_indices = []

    for logical_rank in range(logical_world_size):
        p0 = logical_rank * shard_factor
        best_cost = all_phys_cost[p0]
        best_idx = all_phys_idx[p0]

        for sid in range(1, shard_factor):
            p = p0 + sid
            merged_cost = torch.cat([best_cost, all_phys_cost[p]], dim=0)
            merged_idx = torch.cat([best_idx, all_phys_idx[p]], dim=0)
            best_cost, pos = torch.topk(
                merged_cost,
                k=K,
                dim=0,
                largest=False,
            )
            best_idx = torch.gather(merged_idx, dim=0, index=pos)

        logical_costs.append(best_cost)
        logical_indices.append(best_idx)

    return (
        torch.stack(logical_costs, dim=0),
        torch.stack(logical_indices, dim=0),
    )


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
        # IMPORTANT for zero-copy cache resharding: metadata may remap the
        # local index range while the tensor file itself keeps its original
        # 4-GPU cache start/end fields. Normal caches have identical values.
        base_start = int(meta["start"])

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
    logical_rank,
    logical_offset,
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

        # Global routing is expressed in OLD logical-cache rank IDs, not
        # physical 8-GPU worker IDs. Each physical worker owns a contiguous
        # slice of one logical rank and fills only candidates inside that slice.
        logical_mask = req["owner_cpu"] == logical_rank  # [B,K]
        if not logical_mask.any():
            continue

        nz = logical_mask.nonzero(as_tuple=False)  # row-major (batch,k)
        batch_indices = nz[:, 0]
        k_positions = nz[:, 1]
        local_k_values = req["local_k_cpu"][logical_mask]
        original_idx = req["all_idx_cpu"][logical_rank][
            local_k_values, batch_indices
        ].long()

        shard_count = int(cache_files[-1]["end"])
        shard_begin = int(logical_offset)
        shard_end = shard_begin + shard_count
        owned = (original_idx >= shard_begin) & (original_idx < shard_end)
        if not owned.any():
            continue

        idx = original_idx[owned] - shard_begin
        flat_pos = (batch_indices[owned] * K + k_positions[owned]).long()

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
        "--candidate-block-batches",
        type=int,
        default=20,
        help=(
            "Number of original validation batches whose raw top-K candidate "
            "latents are materialized at once. This controls host-memory usage "
            "only and does not change NN search, top-K order, sampling order, "
            "or metric arithmetic."
        ),
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse an already-built latent cache under --latent-cache-dir.",
    )
    parser.add_argument(
        "--reuse-cache-world-size",
        type=int,
        default=0,
        help=(
            "World size that originally built an existing cache. Example: "
            "set 4 when reusing a 4-GPU cache from an 8-GPU run. Zero means "
            "the cache world size equals the current physical world size."
        ),
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

    # Logical cache topology. For a normal run logical==physical. For zero-copy
    # 4->8 reuse, two physical workers emulate each original logical cache rank.
    cache_world_size = (
        int(args.reuse_cache_world_size)
        if args.reuse_cache and int(args.reuse_cache_world_size) > 0
        else world_size
    )
    if world_size % cache_world_size != 0:
        raise RuntimeError(
            f"Current world_size={world_size} is not divisible by "
            f"cache_world_size={cache_world_size}"
        )
    shard_factor = world_size // cache_world_size
    logical_rank = rank // shard_factor
    shard_id = rank % shard_factor
    logical_offset = 0
    full_cache_files_for_verify = None

    if args.reuse_cache:
        if cache_world_size == world_size:
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
            full_cache_files_for_verify = cache_files
        else:
            info = discover_existing_cache_resharded(
                cache_root=cache_root,
                physical_rank=rank,
                physical_world_size=world_size,
                cache_world_size=cache_world_size,
                cache_group_size=args.cache_group_size,
                bs=args.bs,
            )
            cache_files = info["cache_files"]
            latent_shape = info["latent_shape"]
            local_start = info["local_count"]
            logical_rank = info["logical_rank"]
            logical_offset = info["logical_offset"]
            shard_factor = info["shard_factor"]
            shard_id = info["shard_id"]
            full_cache_files_for_verify = info["full_cache_files"]

            expected_local = args.small // world_size
            if local_start != expected_local:
                raise RuntimeError(
                    f"Resharded cache on physical rank {rank}: found "
                    f"{local_start} rows, expected {expected_local}."
                )

        if args.metric_norm != "none":
            raise RuntimeError(
                "--reuse-cache currently guarantees exact behavior for "
                "--metric-norm none, which is the paper iFID setting."
            )
        latent_mean = torch.zeros(1, device=device, dtype=torch.float32)
        latent_std = torch.ones(1, device=device, dtype=torch.float32)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("[REUSE CACHE]", cache_root)
            print(
                f"[CACHE TOPOLOGY] physical_world={world_size}, "
                f"logical_cache_world={cache_world_size}, "
                f"shard_factor={shard_factor}"
            )
        print(
            f"[CACHE SHARD] physical_rank={rank} -> logical_rank={logical_rank}, "
            f"shard_id={shard_id}, logical_offset={logical_offset}, "
            f"files={len(cache_files)}, rows={local_start}",
            flush=True,
        )
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

    # A newly built cache always uses the current physical world size, so its
    # logical topology is identical to the physical topology.
    if not args.reuse_cache:
        cache_world_size = world_size
        shard_factor = 1
        logical_rank = rank
        shard_id = 0
        logical_offset = 0
        full_cache_files_for_verify = cache_files

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

    if accelerator.is_main_process:
        print(
            f"[BLOCK CONFIG] query_block_batches={args.query_block_batches}, "
            f"candidate_block_batches={args.candidate_block_batches}, bs={args.bs}",
            flush=True,
        )

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

        # Reconstruct the ORIGINAL cache-rank topology before global top-K.
        # In the 4->8 case, two physical workers are merged back into one old
        # logical rank, preserving old logical rank order 0,1,2,3.
        # Reproduce the original per-batch global top-K routing, in order.
        requests = []
        first_logical_for_verify = None
        for batch_i, (rec, (local_cost, local_idx)) in enumerate(
            zip(query_records, local_results)
        ):
            logical_cost, logical_idx = merge_physical_topk_to_logical(
                accelerator=accelerator,
                local_cost=local_cost,
                local_idx=local_idx,
                K=K,
                physical_world_size=world_size,
                logical_world_size=cache_world_size,
                shard_factor=shard_factor,
                logical_offset=logical_offset,
            )

            if batch_i == 0:
                first_logical_for_verify = (logical_cost, logical_idx)

            # Global top-K over OLD logical cache ranks, in their original order.
            all_cost_flat = logical_cost.permute(2, 0, 1).reshape(
                rec["z"].shape[0], -1
            )
            _, global_pos = torch.topk(
                all_cost_flat,
                k=K,
                dim=1,
                largest=False,
            )

            owner = global_pos // K          # old logical cache rank ID
            local_k = global_pos % K

            requests.append(
                {
                    "K": K,
                    "batch_size": rec["z"].shape[0],
                    "all_idx_cpu": logical_idx.cpu(),
                    "local_k_cpu": local_k.cpu(),
                    "owner_cpu": owner.cpu(),
                }
            )

            if batch_i != 0:
                del logical_cost, logical_idx
            del local_cost, local_idx, all_cost_flat
            del global_pos, owner, local_k

        # Real-data exactness audit. In a 4->8 reuse run, one physical worker
        # per old logical rank re-runs the ORIGINAL full-cache scan for the first
        # query batch and compares against the pair-merged logical top-K. All
        # ranks participate in a fail all-reduce so a mismatch stops cleanly.
        if args.verify_first_batch and first_query_block:
            logical_cost0, logical_idx0 = first_logical_for_verify
            local_fail = torch.zeros(1, device=device, dtype=torch.int32)
            local_max_diff = 0.0
            local_mismatch = 0

            if shard_factor == 1:
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
                target_cost = logical_cost0[rank]
                target_idx = logical_idx0[rank]
                local_mismatch = int((verify_idx != target_idx).sum().item())
                local_max_diff = float((verify_cost - target_cost).abs().max().item())
                if local_mismatch or local_max_diff > 1e-6:
                    local_fail.fill_(1)
                del verify_cost, verify_idx
            elif shard_id == 0:
                verify_cost, verify_idx = compute_local_topk_metric_from_cache(
                    cache_files=full_cache_files_for_verify,
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
                target_cost = logical_cost0[logical_rank]
                target_idx = logical_idx0[logical_rank]
                local_mismatch = int((verify_idx != target_idx).sum().item())
                local_max_diff = float((verify_cost - target_cost).abs().max().item())
                if local_mismatch or local_max_diff > 1e-6:
                    local_fail.fill_(1)
                print(
                    f"[VERIFY LOGICAL RANK {logical_rank}] "
                    f"mismatch={local_mismatch}, max_cost_diff={local_max_diff:.3e}",
                    flush=True,
                )
                del verify_cost, verify_idx

            dist.all_reduce(local_fail, op=dist.ReduceOp.MAX)
            if int(local_fail.item()) != 0:
                raise RuntimeError(
                    "4->8 logical-rank verification FAILED. Refusing to continue."
                )
            if accelerator.is_main_process:
                print(
                    "[VERIFY PASS] physical shards reconstruct the original "
                    "logical-cache top-K exactly"
                )

        if first_logical_for_verify is not None:
            del first_logical_for_verify

        # The expensive NN search above may use a large query block (e.g. 100)
        # so one reference-cache scan services many query batches.  However, a
        # raw RAE top-K candidate tensor is huge (~187.5 MiB per bs=25 batch at
        # K=10, FP32).  Materializing candidates for all 100 batches on every
        # rank would consume ~18.3 GiB/rank (~146 GiB across 8 ranks) in host
        # memory and can trigger a cgroup/SIGKILL.
        #
        # Therefore candidate fetch + decode are split into smaller sub-blocks.
        # This is an I/O/memory scheduling change only: requests are consumed in
        # exactly the original batch order, and each batch's candidate tensor,
        # all_reduce, stochastic top-K sampling, slerp, decode and FID path are
        # unchanged.
        candidate_block_batches = max(1, int(args.candidate_block_batches))
        for sub_start in range(0, n_this_block, candidate_block_batches):
            sub_end = min(sub_start + candidate_block_batches, n_this_block)
            sub_requests = requests[sub_start:sub_end]
            sub_records = query_records[sub_start:sub_end]

            candidate_cpu_list = fetch_cached_candidates_for_block(
                cache_files=cache_files,
                requests=sub_requests,
                logical_rank=logical_rank,
                logical_offset=logical_offset,
                latent_shape=latent_shape,
                cache_dtype=cache_dtype,
            )

            if args.verify_first_batch and first_query_block and sub_start == 0:
                req = sub_requests[0]
                B = req["batch_size"]

                # Expected candidate tensor under the ORIGINAL cache topology.
                expected_partial = torch.zeros(
                    B, K, *latent_shape, dtype=cache_dtype, device="cpu"
                )
                if shard_factor == 1 or shard_id == 0:
                    old_owner = rank if shard_factor == 1 else logical_rank
                    full_files = (
                        cache_files if shard_factor == 1
                        else full_cache_files_for_verify
                    )
                    mask = req["owner_cpu"] == old_owner
                    if mask.any():
                        batch_indices = mask.nonzero(as_tuple=True)[0]
                        idx = req["all_idx_cpu"][old_owner][
                            req["local_k_cpu"][mask], batch_indices
                        ]
                        selected_z = fetch_cached_z_by_indices(full_files, idx)
                        expected_partial[mask] = selected_z.to(cache_dtype)
                        del selected_z

                expected_gpu = expected_partial.to(device)
                dist.all_reduce(expected_gpu, op=dist.ReduceOp.SUM)

                actual_gpu = candidate_cpu_list[0].to(device)
                dist.all_reduce(actual_gpu, op=dist.ReduceOp.SUM)

                candidate_equal = torch.equal(expected_gpu, actual_gpu)
                local_fail = torch.tensor(
                    [0 if candidate_equal else 1], device=device, dtype=torch.int32
                )
                dist.all_reduce(local_fail, op=dist.ReduceOp.MAX)
                if int(local_fail.item()) != 0:
                    max_diff = float((expected_gpu - actual_gpu).abs().max().item())
                    raise RuntimeError(
                        f"Resharded candidate fetch verification FAILED; "
                        f"max diff={max_diff:.3e}."
                    )
                if accelerator.is_main_process:
                    print(
                        "[VERIFY PASS] 4->8 candidate reconstruction is tensor-identical "
                        "to original logical-cache fetch"
                    )
                del expected_partial, expected_gpu, actual_gpu, local_fail

            # Finish each ORIGINAL validation batch in the original order. This
            # preserves Categorical sampling order and decode/FID accumulation order.
            for rec, znn_candidate_cpu in zip(sub_records, candidate_cpu_list):
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


            del candidate_cpu_list, sub_requests, sub_records

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
