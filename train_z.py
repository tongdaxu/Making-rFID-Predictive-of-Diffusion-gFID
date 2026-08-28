import argparse
import copy
import logging
import os
import json
import math
from pathlib import Path
from collections import OrderedDict
from omegaconf import OmegaConf
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta, datetime
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb
from ifid.fid.psnr import get_psnr
from ifid.dataset import CustomINH5Dataset
from ifid.sit.sit import SiT_models
from ifid.sit.samplers import euler_sampler
from ifid.vae.utils import instantiate_from_config
from PIL import Image 
from accelerate.utils import DistributedDataParallelKwargs
import lpips

try:
    from turihub import Hub
except:
    Hub = None

logger = get_logger(__name__)

def preprocess_imgs_vae(imgs):
    # imgs: (B, C, H, W) -> (B, C, H, W), [0, 255] uint8 -> [-1, 1] float32
    return imgs.float() / 127.5 - 1.0

#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    # Create model:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for name, param in vae.named_parameters():
        param.requires_grad_(False)

    train_dataset = CustomINH5Dataset(args.data_dir)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    vae.eval()

    for epoch in range(args.epochs):

        for raw_image, y in train_dataloader:
            raw_image = raw_image.to(device)
            labels = y.to(device)

            # 1). Forward pass: VAE
            processed_image = preprocess_imgs_vae(raw_image)

            vae.eval()
            with torch.no_grad():
                z = vae.encode(processed_image)

            z = z.requires_grad_(True)

            optimizer = torch.optim.AdamW(
                [z],
                lr=2e-2,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

            for _ in tqdm(range(5000)):

                xhat = vae.decode(z)

                loss = torch.mean((processed_image - xhat) ** 2)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                psnr = torch.mean(get_psnr(processed_image, xhat, zero_mean=True))

                print(psnr.item())
            assert(0)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")
    # dataset params
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam-weight-decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam-epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max-grad-norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max-grad-norm-vae", default=1.0, type=float, help="Max gradient norm."
    )

    # seed params
    parser.add_argument("--seed", type=int, default=0)

    # cpu params
    parser.add_argument("--num-workers", type=int, default=4)

    # vae params
    parser.add_argument("--vae-config", type=str, default="")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)