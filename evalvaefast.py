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


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-config", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=25)
    parser.add_argument("--exp-name", type=str, default="ifid")
    parser.add_argument("--sample-dir", type=str, default="./samples")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("-save", action="store_true")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-max", type=int, default=100)

    return parser.parse_args()


# -------------------------
# Main
# -------------------------
def main(args):

    sample_folder_dir = os.path.join(f"{args.sample_dir}", args.exp_name)
    sub_folder_dir = os.path.join(f"{args.sample_dir}", args.exp_name + "_extra")
    src_dir = os.path.join(sub_folder_dir, "src_dist")
    recon_dir = os.path.join(sub_folder_dir, "recon_dist")

    os.makedirs(sample_folder_dir, exist_ok=True)
    os.makedirs(sub_folder_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

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

    val_set = ImageNetValDataset(args.dataset, transform, small=-1)

    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=8)

    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for name, param in vae.named_parameters():
        param.requires_grad_(False)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    vae, inception = accelerator.prepare(vae, inception)

    # -------------------------
    # Validation loop
    # -------------------------
    pred_arr_in = []
    pred_arr_rec = []
    arr_psnr = []
    arr_ssim = []
    arr_msssim = []
    arr_lpips = []

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

        z = z.to(device)

        # ---- decode + inception ----
        with torch.no_grad():
            img_in = img
            raw_vae = accelerator.unwrap_model(vae)
            img_recon = raw_vae.decode(z)

            img_in = (img_in + 1) / 2
            img_recon = (img_recon + 1) / 2

            if args.save and accelerator.is_main_process:
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
                for j, sample in enumerate(img_in_np):
                    Image.fromarray(sample).save(f"{src_dir}/{name[j]}")
                for j, sample in enumerate(img_recon_np):
                    Image.fromarray(sample).save(f"{recon_dir}/{name[j]}")
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
                        saved_count += 1

                    global_seen += 1

            pred_psnr = get_psnr(img_in, img_recon, zero_mean=False)
            # pred_ssim, pred_msssim = get_ssim_and_msssim(
            #     img_in, img_recon, zero_mean=True
            # )

            pred_lpips = get_lpips(
                img_in, img_recon, zero_mean=False, network_type="alex"
            )
            arr_psnr.append(torch.mean(pred_psnr).item())
            #arr_ssim.append(torch.mean(pred_ssim).item())
            # arr_msssim.append(torch.mean(pred_msssim).item())
            arr_lpips.append(torch.mean(pred_lpips).item())

            pred_in = inception(img_in)[0]
            pred_rec = inception(img_recon)[0]
            if pred_in.size(2) != 1:
                pred_in = adaptive_avg_pool2d(pred_in, (1, 1))
            if pred_rec.size(2) != 1:
                pred_rec = adaptive_avg_pool2d(pred_rec, (1, 1))
            pred_arr_in.append(pred_in.squeeze(-1).squeeze(-1).cpu().numpy())
            pred_arr_rec.append(pred_rec.squeeze(-1).squeeze(-1).cpu().numpy())

    pred_arr_in = np.concatenate(pred_arr_in)
    pred_arr_rec = np.concatenate(pred_arr_rec)

    if accelerator.is_main_process:
        mu_in = pred_arr_in.mean(0)
        sigma_in = np.cov(pred_arr_in, rowvar=False)

        mu_rec = pred_arr_rec.mean(0)
        sigma_rec = np.cov(pred_arr_rec, rowvar=False)

        rfid = calculate_frechet_distance(mu_in, sigma_in, mu_rec, sigma_rec)

        print("rfid={0:.4f}".format(rfid))
        print("psnr={0:.4f}".format(np.mean(arr_psnr)))
        # print("ssim={0:.4f}".format(np.mean(arr_ssim)))
        # print("msssim={0:.4f}".format(np.mean(arr_msssim)))
        print("lpips={0:.4f}".format(np.mean(arr_lpips)))


if __name__ == "__main__":
    args = parse_args()
    print("evaluating", args.exp_name)
    with torch.no_grad():
        main(args)
