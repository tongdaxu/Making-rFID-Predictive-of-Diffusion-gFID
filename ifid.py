import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from ifid.fid.fid import InceptionV3, calculate_frechet_distance
from ifid.dataset import ImageNetValDataset
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist
import random
import sys
from omegaconf import OmegaConf
from ifid.vae.utils import instantiate_from_config

def cosine_similarity_chunked(
    t1_cpu,          # (B1, C) on CPU
    t2_cpu,          # (B2, C) on CPU or GPU
    chunk_size=20000,
    device="cuda",
    dtype=torch.float16,   # fp16/bf16 recommended
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
        sim_chunk = t1_chunk @ t2.T   # (chunk, B2)

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
    parser.add_argument("--vae-config", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--intp", type=str, default="slerp", choices=["linear", "slerp", "mask"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--exp-name", type=str, default="ifid")
    parser.add_argument("--small", type=int, default=-1)
    parser.add_argument("--sample-dir", type=str, default="./samples")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--dataset-ref", type=str, default="")
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-save", action="store_true")

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

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_set = ImageNetValDataset(args.dataset, transform, args.small)
    val_set   = ImageNetValDataset(args.dataset_ref, transform)

    train_loader = DataLoader(train_set, batch_size=25, shuffle=False, num_workers=8)
    val_loader   = DataLoader(val_set, batch_size=25, shuffle=False, num_workers=8)

    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for name, param in vae.named_parameters():
        param.requires_grad_(False)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    train_loader, vae, inception = accelerator.prepare(
        train_loader, vae, inception
    )

    # -------------------------
    # Encode training latents (SHARDED)
    # -------------------------
    zs, ys = [], []
    for img, y, name in tqdm(train_loader, disable=not accelerator.is_local_main_process):
        img = img.to(device)
        y = y.to(device)
        with torch.no_grad():
            z = vae.encode(img)
        if args.cpu:
            z = z.cpu().to(torch.bfloat16)
        zs.append(z)
        ys.append(y)

    zs = torch.cat(zs, dim=0)           # LOCAL shard only
    ys = torch.cat(ys, dim=0)
    zs_flat = zs.reshape(zs.shape[0], -1)

    # -------------------------
    # Validation loop
    # -------------------------
    pred_arr_in = []
    pred_arr_znn = []
    ents = []

    for img, y, name in tqdm(val_loader, disable=not accelerator.is_local_main_process):
        img = img.to(device)
        y = y.to(device)
        with torch.no_grad():
            z = vae.encode(img)

        if args.cpu:
            z = z.cpu().to(torch.bfloat16)

        z_flat = z.reshape(z.shape[0], -1)

        # ---- local distance ----
        if args.intp == "linear":
            cost = (
                (zs_flat ** 2).sum(1)[:, None]
                + (z_flat ** 2).sum(1)[None]
                - 2 * zs_flat @ z_flat.T
            )
        else:
            if args.cpu:
                cost = -cosine_similarity_chunked(
                    zs_flat,
                    z_flat,
                    device=device,
                    dtype=torch.bfloat16,
                )
            else:
                dot = zs_flat @ z_flat.T
                cos = dot / (
                    zs_flat.norm(dim=1)[:, None] *
                    z_flat.norm(dim=1)[None] + 1e-8
                )
                cost = -cos

        cost = cost.to(device)

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
        all_idx = all_idx.reshape(world_size, K, -1)    # [G, K, B]

        # flatten GPU and K dims
        all_cost_flat = all_cost.permute(2, 0, 1).reshape(z.shape[0], -1)
        # shape: [B, G*K]

        global_cost, global_pos = torch.topk(
            all_cost_flat, k=K, dim=1, largest=False
        )  # [B, K]

        gpu_id = global_pos // K          # [B, K]
        local_k = global_pos % K          # [B, K]

        znn_candidate = torch.zeros(
            z.shape[0], K, *zs.shape[1:], device=device
        )

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
        znn_candidate_flat = znn_candidate.reshape(z.shape[0], K, -1)
        local_cost = torch.mean((z_flat[:,None] - znn_candidate_flat)**2, dim=-1)
        local_cost = local_cost.to(torch.float32)
        weights = torch.softmax(-local_cost, dim=1)  # [B, K]

        cat_dist = torch.distributions.Categorical(
            probs=weights,
        )

        ent = cat_dist.entropy()
        ents.append(torch.mean(ent).item())

        znn_idx = cat_dist.sample()
        weights = F.one_hot(znn_idx, num_classes=K)
        if args.cpu:
            weights = weights.to(torch.bfloat16) 
        znn = torch.sum(weights[:,:,None] * znn_candidate_flat,dim=1).reshape(*z.shape)

        # ---- interpolate ----
        if args.intp == "linear":
            z_intp = (1 - args.alpha) * z + args.alpha * znn
        elif args.intp == "slerp":
            z_intp = slerp(z, znn, args.alpha)
        elif args.intp == "mask":
            if len(z.shape)==4:
                mask = torch.zeros([z.shape[0], 1, z.shape[2], z.shape[3]]).to(device)
                mask[:,:,:z.shape[2]//2,:] += 1
            else:
                mask = torch.zeros([z.shape[0], z.shape[1], 1]).to(device)
                mask[:,:z.shape[1]//2,:] += 1
            z_intp = (1 - mask) * z + mask * znn
        else:
            assert(0)
        # ---- decode + inception ----
        with torch.no_grad():
            img_in = img
            img_recon = accelerator.unwrap_model(vae).decode(z)
            img_intp = accelerator.unwrap_model(vae).decode(z_intp)
            img_nn = accelerator.unwrap_model(vae).decode(znn)

            img_in = (img_in + 1) / 2
            img_recon = (img_recon + 1) / 2
            img_intp = (img_intp + 1) / 2
            img_nn = (img_nn + 1) / 2

            if args.save:

                img_nn_np = torch.clamp(
                    255. * img_nn, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                img_recon_np = torch.clamp(
                    255. * img_recon, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                img_in_np = torch.clamp(
                    255. * img_in, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                img_intp_np = torch.clamp(
                    255. * img_intp, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                for j, sample in enumerate(img_in_np):
                    Image.fromarray(sample).save(f"{src_dir}/{name[j]}")
                for j, sample in enumerate(img_recon_np):
                    Image.fromarray(sample).save(f"{recon_dir}/{name[j]}")
                for j, sample in enumerate(img_nn_np):
                    Image.fromarray(sample).save(f"{nn_dir}/{name[j]}")
                for j, sample in enumerate(img_intp_np):
                    Image.fromarray(sample).save(f"{intp_dir}/{name[j]}")

            pred_in = inception(img_in)[0]
            pred_intp = inception(img_intp)[0]
            if pred_in.size(2) != 1:
                pred_in = adaptive_avg_pool2d(pred_in, (1, 1))
            if pred_intp.size(2) != 1:
                pred_intp = adaptive_avg_pool2d(pred_intp, (1, 1))
            pred_arr_in.append(pred_in.squeeze(-1).squeeze(-1).cpu().numpy())
            pred_arr_znn.append(pred_intp.squeeze(-1).squeeze(-1).cpu().numpy())
    
    pred_arr_in = np.concatenate(pred_arr_in)
    pred_arr_znn = np.concatenate(pred_arr_znn)

    pred_arr_in = accelerator.gather(
        torch.from_numpy(pred_arr_in).to(device)
    ).cpu().numpy()
    pred_arr_znn = accelerator.gather(
        torch.from_numpy(pred_arr_znn).to(device)
    ).cpu().numpy()

    if accelerator.is_main_process:
        mu_in = pred_arr_in.mean(0)
        sigma_in = np.cov(pred_arr_in, rowvar=False)

        mu_znn = pred_arr_znn.mean(0)
        sigma_znn = np.cov(pred_arr_znn, rowvar=False)

        fid = calculate_frechet_distance(mu_in, sigma_in, mu_znn, sigma_znn)

        print('ifid=',fid)


if __name__ == "__main__":
    args = parse_args()
    print("evaluating", args.exp_name)
    with torch.no_grad():
        main(args)
