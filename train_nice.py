import argparse
import copy
import logging
import os
import json
import math
from pathlib import Path
from collections import OrderedDict
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb

from ifid.dataset import CustomINH5Dataset, SimpleDataset
from ifid.sit.sit import SiT_models
from ifid.sit.samplers import euler_sampler
from ifid.vae.utils import instantiate_from_config
from ifid.fid.psnr import get_psnr
from datetime import timedelta, datetime
from PIL import Image 
import lpips

from accelerate.utils import DistributedDataParallelKwargs

Hub = None

logger = get_logger(__name__)


def count_trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def preprocess_imgs_vae(imgs):
    # imgs: (B, C, H, W) -> (B, C, H, W), [0, 255] uint8 -> [-1, 1] float32
    return imgs.float() / 127.5 - 1.0

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = (
        x.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return x


def sample_posterior(moments, latents_scale=1.0, latents_bias=0.0):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale  # normalize
    return z

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    assert args.gradient_accumulation_steps == 1
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # set up the logger and checkpoint dirs
    if accelerator.is_main_process:
        os.makedirs(
            args.output_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, "w") as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    torch.backends.cudnn.benchmark = True

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)

    # Create model:
    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)

    print("trainable parameters:")
    for name, param in vae.named_parameters():
        if "flow" in name:
            param.requires_grad_(True)
            print(name)
        else:
            param.requires_grad_(False)

    fake_in = torch.zeros([1, 3, args.resolution, args.resolution]).to(device)
    fake_z = vae.encode(fake_in)[0]

    if len(fake_z.shape) == 3:
        # 2d latent
        latent_size = fake_z.shape[-1]
        in_channels = fake_z.shape[0]
        xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)
    elif len(fake_z.shape) == 2:
        # 1d latent
        latent_size = fake_z.shape[0]
        in_channels = fake_z.shape[-1]
        xT = torch.randn((n, latent_size, in_channels), device=device)
    else:
        assert 0

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Define the optimizers for SiT, VAE, and VAE loss function separately
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=1e-4,
    )

    # optimizer = torch.optim.SGD(
    #     vae.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=1e-4,
    # )

    # Setup data
    train_dataset = SimpleDataset(args.data_dir, 256)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    vae.train()

    # resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + ".pt"
        ckpt_path = f"{args.cont_dir}/checkpoints/{ckpt_name}"

        # If the checkpoint exists, we load the checkpoint and resume the training
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(ckpt["opt"])
        global_step = ckpt["steps"]

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    # Model compilation for better performance

    vae, optimizer, train_dataloader = accelerator.prepare(
        vae, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="diffusion vae arena",
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}"}},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(args.epochs):

        for raw_image in train_dataloader:
            raw_image = raw_image.to(device)

            with accelerator.autocast():
                # 1). Forward pass: VAE
                processed_image = raw_image
                z, info = vae(processed_image)

                flow_loss = torch.mean(torch.sum(z**2, dim=1))

                accelerator.backward(flow_loss)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    psnr_float = torch.mean(get_psnr(processed_image, info["xhat"], zero_mean=True, integer=False))
                    psnr_int = torch.mean(get_psnr(processed_image, info["xhat"], zero_mean=True, integer=True))

            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "flow_loss": accelerator.gather(flow_loss).mean().detach().item(),
                    "base_loss": accelerator.gather(info["base_loss"]).mean().detach().item(),
                    "psnr_float": accelerator.gather(psnr_float).mean().detach().item(),
                    "psnr_int": accelerator.gather(psnr_int).mean().detach().item(),
                    "epoch": epoch,
                }

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:

                    # model might be compiled, we extract the original model
                    checkpoint = {
                        "state_dict": accelerator.unwrap_model(vae).state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging params
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)
    parser.add_argument("--wandb-history-path", type=str, default=None)

    # SiT model params
    parser.add_argument(
        "--model",
        type=str,
        default="SiT-XL/2",
        choices=SiT_models.keys(),
        help="The model to train.",
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to compile the model for faster training",
    )

    # dataset params
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision params
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
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
        default=0.95,
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