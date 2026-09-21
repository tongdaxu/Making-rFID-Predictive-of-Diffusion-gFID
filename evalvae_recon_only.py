import os
import argparse
from PIL import Image

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from ifid.dataset import ImageNetValDataset
from ifid.vae.utils import instantiate_from_config
from ifid.fid.psnr import get_psnr
from ifid.fid.ssim import get_ssim_and_msssim
from ifid.fid.lpips import get_lpips


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vae-config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=25)
    parser.add_argument("--exp-name", type=str, default="vae-recon")
    parser.add_argument("--sample-dir", type=str, default="./samples")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--small-val", type=int, default=-1)

    parser.add_argument("-save", action="store_true")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-max", type=int, default=100)
    parser.add_argument("--deterministic-kl", action="store_true")

    return parser.parse_args()


def save_images(img_01, names, out_dir, saved_count, global_seen, save_every, save_max):
    """
    img_01: [B, 3, H, W], range [0,1]
    """
    os.makedirs(out_dir, exist_ok=True)

    img_np = (
        torch.clamp(img_01 * 255.0, 0, 255)
        .permute(0, 2, 3, 1)
        .to("cpu", dtype=torch.uint8)
        .numpy()
    )

    for j in range(len(names)):
        should_save = (
            (save_max < 0 or saved_count < save_max)
            and (global_seen % save_every == 0)
        )

        if should_save:
            Image.fromarray(img_np[j]).save(os.path.join(out_dir, names[j]))
            saved_count += 1

        global_seen += 1

    return saved_count, global_seen


def encode_vae(raw_vae, img, y=None, deterministic_kl=False):
    """
    默认走 wrapper 自己的 encode。
    如果 deterministic_kl=True 且是 KLVAE wrapper，则尝试 posterior.mode()。
    """
    if deterministic_kl and hasattr(raw_vae, "vae"):
        try:
            posterior = raw_vae.vae.encode(img, sample=False)
            return posterior.mode()
        except Exception:
            pass

    try:
        return raw_vae.encode(img, y)
    except TypeError:
        return raw_vae.encode(img)


def decode_vae(raw_vae, z, y=None):
    try:
        return raw_vae.decode(z, y)
    except TypeError:
        return raw_vae.decode(z)


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    set_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        print(f"evaluating recon only: {args.exp_name}")
        print(f"vae config: {args.vae_config}")
        print(f"dataset: {args.dataset}")
        print(f"deterministic_kl: {args.deterministic_kl}")

    out_root = os.path.join(args.sample_dir, args.exp_name + "_recon_only")
    src_dir = os.path.join(out_root, "src")
    recon_dir = os.path.join(out_root, "recon")

    if accelerator.is_main_process and args.save:
        os.makedirs(src_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ]
    )

    val_set = ImageNetValDataset(args.dataset, transform, args.small_val)
    val_loader = DataLoader(
        val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for _, param in vae.named_parameters():
        param.requires_grad_(False)

    val_loader, vae = accelerator.prepare(val_loader, vae)

    sum_psnr = torch.tensor(0.0, device=device)
    sum_ssim = torch.tensor(0.0, device=device)
    sum_msssim = torch.tensor(0.0, device=device)
    sum_lpips = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)

    saved_count = 0
    global_seen = 0

    for img, y, name in tqdm(val_loader, disable=not accelerator.is_local_main_process):
        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        raw_vae = accelerator.unwrap_model(vae)

        with torch.no_grad():
            z = encode_vae(
                raw_vae,
                img,
                y=y,
                deterministic_kl=args.deterministic_kl,
            )
            img_recon = decode_vae(raw_vae, z, y=y)

            # 指标在 [-1,1] 上算，zero_mean=True 是正确的
            img_metric = img
            recon_metric = img_recon.clamp(-1.0, 1.0)

            pred_psnr = get_psnr(img_metric, recon_metric, zero_mean=True)
            pred_ssim, pred_msssim = get_ssim_and_msssim(
                img_metric,
                recon_metric,
                zero_mean=True,
            )
            pred_lpips = get_lpips(
                img_metric,
                recon_metric,
                zero_mean=True,
                network_type="alex",
            )

            bs = img.shape[0]
            sum_psnr += pred_psnr.sum()
            sum_ssim += pred_ssim.sum()
            sum_msssim += pred_msssim.sum()
            sum_lpips += pred_lpips.sum()
            total_count += bs

            if args.save and accelerator.is_main_process:
                img_01 = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
                recon_01 = ((img_recon + 1.0) / 2.0).clamp(0.0, 1.0)

                saved_count, global_seen = save_images(
                    img_01,
                    name,
                    src_dir,
                    saved_count,
                    global_seen,
                    args.save_every,
                    args.save_max,
                )
                # recon 使用同一个 global_seen 会错位，所以单独保存更简单
                # 这里直接按当前 batch 名字保存，不参与 save_every 逻辑
                recon_np = (
                    torch.clamp(recon_01 * 255.0, 0, 255)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()
                )
                for j in range(len(name)):
                    src_path = os.path.join(src_dir, name[j])
                    if os.path.exists(src_path):
                        Image.fromarray(recon_np[j]).save(os.path.join(recon_dir, name[j]))

    metrics = torch.stack([sum_psnr, sum_ssim, sum_msssim, sum_lpips, total_count])
    metrics = accelerator.reduce(metrics, reduction="sum")

    if accelerator.is_main_process:
        total = metrics[4].item()
        mean_psnr = (metrics[0] / metrics[4]).item()
        mean_ssim = (metrics[1] / metrics[4]).item()
        mean_msssim = (metrics[2] / metrics[4]).item()
        mean_lpips = (metrics[3] / metrics[4]).item()

        print("============================================================")
        print(f"exp={args.exp_name}")
        print(f"num_images={int(total)}")
        print(f"psnr={mean_psnr:.4f}")
        print(f"ssim={mean_ssim:.4f}")
        print(f"msssim={mean_msssim:.4f}")
        print(f"lpips={mean_lpips:.4f}")
        print("============================================================")


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        main(args)