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


def latent_channel_sum_sumsq_count(z):
    """
    Returns per-channel sum, sumsq, count.
    Supports:
      z: [B, C, H, W]
      z: [B, N, C]  # optional fallback
      z: [B, C]     # optional fallback
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
        raise ValueError(f"Unsupported latent shape: {z.shape}")

    s = z.sum(dim=reduce_dims)
    ss = (z ** 2).sum(dim=reduce_dims)

    return s, ss, count


def normalize_latent_channelwise(z, mean, std):
    """
    Channel-wise normalization for distance computation only.

    Supports common latent layouts:
      [B, C, H, W]
      [B, K, C, H, W]
      [B, N, C]
      [B, K, N, C]
      [B, C]
      [B, K, H, W]  # for index/grid-like latents, normalize last dim if it matches mean
    """
    C = mean.numel()

    # [B, C]
    if z.dim() == 2:
        if z.shape[1] != C:
            raise ValueError(f"Cannot match channel dim: z={z.shape}, mean={mean.shape}")
        return (z - mean[None, :]) / std[None, :]

    # [B, N, C] or [B, C, N]
    elif z.dim() == 3:
        # Prefer last dim, consistent with your original [B, N, C] assumption.
        if z.shape[-1] == C:
            return (z - mean[None, None, :]) / std[None, None, :]
        elif z.shape[1] == C:
            return (z - mean[None, :, None]) / std[None, :, None]
        else:
            raise ValueError(f"Cannot match channel dim: z={z.shape}, mean={mean.shape}")

    # [B, C, H, W] or [B, K, N, C] / [B, K, H, W]
    elif z.dim() == 4:
        # Standard image latent layout: [B, C, H, W]
        if z.shape[1] == C:
            return (z - mean[None, :, None, None]) / std[None, :, None, None]

        # Candidate layout for 3D latent: [B, K, N, C]
        # Or SoftVQ grid-like case: [B, K, H, W], where original stat treated W as C.
        elif z.shape[-1] == C:
            return (z - mean[None, None, None, :]) / std[None, None, None, :]

        else:
            raise ValueError(f"Cannot match channel dim: z={z.shape}, mean={mean.shape}")

    # [B, K, C, H, W] or [B, K, H, W, C]
    elif z.dim() == 5:
        # Candidate layout for 4D BCHW latent: [B, K, C, H, W]
        if z.shape[2] == C:
            return (z - mean[None, None, :, None, None]) / std[None, None, :, None, None]

        # Candidate layout for channel-last latent: [B, K, H, W, C]
        elif z.shape[-1] == C:
            return (z - mean[None, None, None, None, :]) / std[None, None, None, None, :]

        else:
            raise ValueError(f"Cannot match channel dim: z={z.shape}, mean={mean.shape}")

    else:
        raise ValueError(f"Unsupported latent shape: {z.shape}")        

def cosine_similarity_chunked(
    t1_cpu,  # (B1, C) on CPU
    t2_cpu,  # (B2, C) on CPU or GPU
    chunk_size=20000,
    device="cuda",
    dtype=torch.float16,  # fp16/bf16 recommended
):
    """
    Returns cosine similarity matrix of shape (B1, B2)
    Computed in GPU chunks along B1.
    """

    assert t1_cpu.device.type == "cpu"

    # Move and normalize t2 once
    t2 = t2_cpu.to(device=device, dtype=dtype, non_blocking=True)
    t2 = F.normalize(t2, dim=1)

    B1 = t1_cpu.shape[0]
    B2 = t2.shape[0]

    # Allocate output on CPU (avoid GPU OOM)
    sim_out = torch.empty(B1, B2, dtype=dtype, device="cpu")

    for start in range(0, B1, chunk_size):
        end = min(start + chunk_size, B1)

        # Move chunk to GPU
        t1_chunk = t1_cpu[start:end].to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

        # Normalize
        t1_chunk = F.normalize(t1_chunk, dim=1)

        # Compute cosine similarity via matmul
        sim_chunk = t1_chunk @ t2.T  # (chunk, B2)

        # Move back to CPU
        sim_out[start:end] = sim_chunk.cpu()

        del t1_chunk, sim_chunk
        torch.cuda.empty_cache()

    return sim_out


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric-norm",
        type=str,
        default="none",
        choices=["none", "bn_channel"],
    )
    parser.add_argument("--vae-config", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=25)
    parser.add_argument(
        "--intp", type=str, default="slerp", choices=["linear", "slerp", "mask"]
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
    z1, z2: (B, C, H, W)
    t: float in [0, 1]
    """
    B = z1.shape[0]

    # Flatten per sample
    z1f = z1.reshape(B, -1)
    z2f = z2.reshape(B, -1)

    # Normalize
    z1n = z1f / (z1f.norm(dim=1, keepdim=True) + eps)
    z2n = z2f / (z2f.norm(dim=1, keepdim=True) + eps)

    # Angle between vectors
    dot = (z1n * z2n).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # SLERP weights
    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w2 = torch.sin(t * theta) / (sin_theta + eps)

    # Interpolate and reshape
    zt = w1.unsqueeze(1) * z1f + w2.unsqueeze(1) * z2f

    small_t = sin_theta < 1e-6
    if small_t.any():
        zt[small_t] = (1 - t) * z1f[small_t] + t * z2f[small_t]

    return zt.reshape_as(z1)


# -------------------------
# Main
# -------------------------
def main(args):

    sample_folder_dir = os.path.join(f"{args.sample_dir}", args.exp_name)
    sub_folder_dir = os.path.join(f"{args.sample_dir}", args.exp_name + "_extra")
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

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=8)

    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for name, param in vae.named_parameters():
        param.requires_grad_(False)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    train_loader, vae, inception = accelerator.prepare(train_loader, vae, inception)

    # -------------------------
    # Encode training latents (SHARDED)
    # -------------------------
    zs, ys = [], []

    stat_sum = None
    stat_sumsq = None
    stat_count = 0

    for img, y, name in tqdm(train_loader, disable=not accelerator.is_main_process):
        img = img.to(device)
        y = y.to(device)

        with torch.no_grad():
            try:
                z = vae.encode(img, y)
            except TypeError:
                z = vae.encode(img)

        # ---- collect BN stats before moving to CPU ----
        s, ss, cnt = latent_channel_sum_sumsq_count(z)

        if stat_sum is None:
            stat_sum = torch.zeros_like(s, device=device, dtype=torch.float64)
            stat_sumsq = torch.zeros_like(ss, device=device, dtype=torch.float64)

        stat_sum += s.to(device=device, dtype=torch.float64)
        stat_sumsq += ss.to(device=device, dtype=torch.float64)
        stat_count += cnt

        if args.cpu:
            z = z.cpu().to(torch.bfloat16)

        zs.append(z)
        ys.append(y)

    # ---- DDP all_reduce global training statistics ----
    stat_count_tensor = torch.tensor([stat_count], device=device, dtype=torch.float64)

    dist.all_reduce(stat_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(stat_sumsq, op=dist.ReduceOp.SUM)
    dist.all_reduce(stat_count_tensor, op=dist.ReduceOp.SUM)

    latent_mean = stat_sum / stat_count_tensor
    latent_var = stat_sumsq / stat_count_tensor - latent_mean ** 2
    latent_std = torch.sqrt(torch.clamp(latent_var, min=1e-6))

    latent_mean = latent_mean.float()
    latent_std = latent_std.float()

    zs = torch.cat(zs, dim=0)  # raw local shard
    ys = torch.cat(ys, dim=0)

    if accelerator.is_main_process:
        print(f"metric_norm={args.metric_norm}, intp={args.intp}, alpha={args.alpha}, top={args.top}")
        if args.metric_norm == "bn_channel":
            print(
                "latent_mean:",
                latent_mean.mean().item(),
                "latent_std_mean:",
                latent_std.mean().item(),
                "latent_std_min:",
                latent_std.min().item(),
                "latent_std_max:",
                latent_std.max().item(),
            )

    # ---- metric version only for NN distance ----
    if args.cpu:
        latent_mean_metric = latent_mean.cpu()
        latent_std_metric = latent_std.cpu()
    else:
        latent_mean_metric = latent_mean.to(device)
        latent_std_metric = latent_std.to(device)

    if args.metric_norm == "bn_channel":
        zs_metric = normalize_latent_channelwise(zs.float(), latent_mean_metric, latent_std_metric)
    else:
        zs_metric = zs

    if args.cpu:
        zs_metric = zs_metric.to(torch.bfloat16)

    zs_metric_flat = zs_metric.reshape(zs_metric.shape[0], -1)

    # -------------------------
    # Validation loop
    # -------------------------
    pred_arr_in = []
    pred_arr_rec = []
    pred_arr_znn = []
    arr_psnr = []
    arr_ssim = []
    arr_msssim = []
    arr_lpips = []
    ents = []

    # val loader is not shared across process
    global_seen = 0
    saved_count = 0
    for img, y, name in tqdm(val_loader, disable=not accelerator.is_main_process):
        img = img.to(device)
        y = y.to(device)
        with torch.no_grad():
            try:
                z = vae.encode(img, y)
            except TypeError:
                z = vae.encode(img)

        if accelerator.is_main_process and global_seen == 0:
            print("DEBUG z shape:", z.shape)
            print("DEBUG zs shape:", zs.shape)
            print("DEBUG latent_mean shape:", latent_mean.shape)
            print("DEBUG metric_norm:", args.metric_norm)

        if args.cpu:
            z = z.cpu().to(torch.bfloat16)

        z_flat = z.reshape(z.shape[0], -1)

        # ---- metric z for distance only ----
        if args.metric_norm == "bn_channel":
            z_metric = normalize_latent_channelwise(z.float(), latent_mean_metric, latent_std_metric)
        else:
            z_metric = z

        if args.cpu:
            z_metric = z_metric.cpu().to(torch.bfloat16)

        z_metric_flat = z_metric.reshape(z_metric.shape[0], -1)

        if args.intp == "linear":
            cost = (
                (zs_metric_flat ** 2).sum(1)[:, None]
                + (z_metric_flat ** 2).sum(1)[None]
                - 2 * zs_metric_flat @ z_metric_flat.T
            )
        else:
            if args.cpu:
                cost = -cosine_similarity_chunked(
                    zs_metric_flat,
                    z_metric_flat,
                    device=device,
                    dtype=torch.bfloat16,
                )
            else:
                dot = zs_metric_flat @ z_metric_flat.T
                cos = dot / (
                    zs_metric_flat.norm(dim=1)[:, None]
                    * z_metric_flat.norm(dim=1)[None]
                    + 1e-8
                )
                cost = -cos

        cost = cost.to(device)

        if args.pclass:
            cost_y = 1e9 * (torch.abs(ys[:,None] - y[None,:]))
            cost = cost + cost_y

        z = z.to(device)
        z_flat = z_flat.to(device)

        K = args.top
        world_size = accelerator.num_processes
        rank = accelerator.process_index

        local_cost, local_idx = torch.topk(
            cost, k=K, dim=0, largest=False
        )  # both are [K, B]

        # gather costs
        all_cost = accelerator.gather(local_cost)
        all_cost = all_cost.reshape(world_size, K, -1)  # [G, K, B]

        # gather indices
        all_idx = accelerator.gather(local_idx)
        all_idx = all_idx.reshape(world_size, K, -1)  # [G, K, B]

        # flatten GPU and K dims
        all_cost_flat = all_cost.permute(2, 0, 1).reshape(z.shape[0], -1)
        # shape: [B, G*K]

        global_cost, global_pos = torch.topk(
            all_cost_flat, k=K, dim=1, largest=False
        )  # [B, K]

        gpu_id = global_pos // K  # [B, K]
        local_k = global_pos % K  # [B, K]

        znn_candidate = torch.zeros(z.shape[0], K, *zs.shape[1:], device=device)

        if args.cpu:
            znn_candidate = znn_candidate.cpu().to(torch.bfloat16)
            all_idx = all_idx.cpu()
            local_k = local_k.cpu()
            gpu_id = gpu_id.cpu()

        for i in range(world_size):
            mask = gpu_id == i
            if mask.any():
                if rank == i:
                    idx = all_idx[i][local_k[mask], mask.nonzero(as_tuple=True)[0]]
                    znn_candidate[mask] = zs[idx]

        znn_candidate = znn_candidate.to(device)

        dist.all_reduce(znn_candidate, op=dist.ReduceOp.SUM)
        if accelerator.is_main_process and global_seen == 0:
            print("DEBUG znn_candidate shape:", znn_candidate.shape)

        # raw candidate latent, used for final znn selection / decode
        znn_candidate_flat = znn_candidate.reshape(z.shape[0], K, -1)

        # metric latent, used only for topK sampling weight
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

        znn_candidate_metric_flat = znn_candidate_for_weight.reshape(z.shape[0], K, -1)
        z_metric_flat_for_weight = z_for_weight.reshape(z.shape[0], -1)

        local_cost = torch.mean(
            (z_metric_flat_for_weight[:, None] - znn_candidate_metric_flat) ** 2,
            dim=-1,
        )

        local_cost = local_cost.to(torch.float32)
        weights = torch.softmax(-local_cost, dim=1)

        cat_dist = torch.distributions.Categorical(probs=weights)

        ent = cat_dist.entropy()
        ents.append(torch.mean(ent).item())

        znn_idx = cat_dist.sample()

        weights = F.one_hot(znn_idx, num_classes=K).to(znn_candidate_flat.dtype)

        znn = torch.sum(
            weights[:, :, None] * znn_candidate_flat,
            dim=1,
        ).reshape(*z.shape)

        # ---- interpolate ----
        if args.intp == "linear":
            z_intp = (1 - args.alpha) * z + args.alpha * znn
        elif args.intp == "slerp":
            z_intp = slerp(z, znn, args.alpha)
        elif args.intp == "mask":
            if len(z.shape) == 4:
                mask = torch.zeros([z.shape[0], 1, z.shape[2], z.shape[3]]).to(device)
                mask[:, :, : z.shape[2] // 2, :] += 1
            else:
                mask = torch.zeros([z.shape[0], z.shape[1], 1]).to(device)
                mask[:, : z.shape[1] // 2, :] += 1
            z_intp = (1 - mask) * z + mask * znn
        else:
            assert 0
        # ---- decode + inception ----
        with torch.no_grad():
            img_in = img
            # img_recon = accelerator.unwrap_model(vae).decode(z)
            # img_intp = accelerator.unwrap_model(vae).decode(z_intp)
            # img_nn = accelerator.unwrap_model(vae).decode(znn)
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

                if accelerator.is_main_process and global_seen == 0:
                    print("z shape:", z.shape)
                    print("znn_candidate shape:", znn_candidate.shape)
                    print("latent_mean shape:", latent_mean.shape)
                # for j, sample in enumerate(img_in_np):
                #     Image.fromarray(sample).save(f"{src_dir}/{name[j]}")
                # for j, sample in enumerate(img_recon_np):
                #     Image.fromarray(sample).save(f"{recon_dir}/{name[j]}")
                # for j, sample in enumerate(img_nn_np):
                #     Image.fromarray(sample).save(f"{nn_dir}/{name[j]}")
                # for j, sample in enumerate(img_intp_np):
                #     Image.fromarray(sample).save(f"{intp_dir}/{name[j]}")
                for j in range(len(name)):
                    should_save = (
                        (args.save_max < 0 or saved_count < args.save_max)
                        and (global_seen % args.save_every == 0)
                    )

                    if should_save:
                        Image.fromarray(img_in_np[j]).save(f"{src_dir}/{name[j]}")
                        Image.fromarray(img_recon_np[j]).save(f"{recon_dir}/{name[j]}")
                        Image.fromarray(img_nn_np[j]).save(f"{nn_dir}/{name[j]}")
                        Image.fromarray(img_intp_np[j]).save(f"{intp_dir}/{name[j]}")
                        saved_count += 1

                    global_seen += 1

            pred_psnr = get_psnr(img_in, img_recon, zero_mean=True)
            pred_ssim, pred_msssim = get_ssim_and_msssim(
                img_in, img_recon, zero_mean=True
            )
            pred_lpips = get_lpips(
                img_in, img_recon, zero_mean=True, network_type="alex"
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
            pred_arr_in.append(pred_in.squeeze(-1).squeeze(-1).cpu().numpy())
            pred_arr_rec.append(pred_rec.squeeze(-1).squeeze(-1).cpu().numpy())
            pred_arr_znn.append(pred_intp.squeeze(-1).squeeze(-1).cpu().numpy())

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


if __name__ == "__main__":
    args = parse_args()
    print("evaluating", args.exp_name)
    with torch.no_grad():
        main(args)
