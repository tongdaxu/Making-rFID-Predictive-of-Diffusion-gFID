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
from ifid.sit.sit import SiTT2I
from ifid.sit.samplers import euler_sampler, euler_maruyama_sampler
from ifid.vae.utils import instantiate_from_config
from PIL import Image 
from accelerate.utils import DistributedDataParallelKwargs
import lpips
from ifid.dataset import _prepare_blip3o_loader, _prepare_t2i_hf_loader
from ifid.sit.text_encoder import setup_text_encoder, encode_text
import random

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


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Also perform EMA on BN buffers
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float64,
        ):
            # Apply EMA only to float buffers
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            # Direct copy for non-float buffers
            ema_buffers[name].copy_(buffer)


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


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


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
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, timeout_kwargs],
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
    sample_batch_size = 8
    if args.ploss:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    # Create model:
    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for name, param in vae.named_parameters():
        param.requires_grad_(False)

    fake_in = torch.zeros([1, 3, args.resolution, args.resolution]).to(device)
    fake_z = vae.encode(fake_in)[0]

    if accelerator.is_main_process:
        logger.info(f"VAE fake_z shape: {tuple(fake_z.shape)}")
        logger.info(f"VAE fake_z numel: {fake_z.numel()}")
        logger.info(f"VAE fake_z mean/std: {fake_z.mean().item():.4f}/{fake_z.std().item():.4f}")

    if len(fake_z.shape) == 3:
        # 2d latent
        latent_size = fake_z.shape[-1]
        in_channels = fake_z.shape[0]
        xT = torch.randn((sample_batch_size, in_channels, latent_size, latent_size), device=device)
    elif len(fake_z.shape) == 2:
        # 1d latent
        latent_size = fake_z.shape[0]
        in_channels = fake_z.shape[-1]
        xT = torch.randn((sample_batch_size, latent_size, in_channels), device=device)
    else:
        assert 0

    tshift = math.sqrt(float(fake_z.numel()) / 4096.0)

    model = SiTT2I(
        input_size=latent_size,
        patch_size=1 if latent_size == 16 else 32,
        in_channels=in_channels,
        tshift=tshift,
    )

    # make a copy of the model for EMA
    model = model.to(device)
    ema = copy.deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    # Apply SyncBN if more than 1 GPU is used
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Define the optimizers for SiT, VAE, and VAE loss function separately
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    local_batch_size = int(args.batch_size // accelerator.num_processes)
    dataset_config = {
        "data_dir": "./data/blip3o-256",
        "splits": ["journeydb", "long-caption", "short-caption"]
    }
    val_dataset_configs = [
        {
            "dataset_name": "geneval",
            "config": {
                "data_dir": "./data"
            },
        },
        {
            "dataset_name": "dpgbench",
            "config": {
                "data_dir": "./data"
            },
        },
        {
            "dataset_name": "genaibench",
            "config": {
                "data_dir": "./data"
            },
        },
    ]
    train_dataloader = _prepare_blip3o_loader(
        dataset_config,
        image_size=256, 
        batch_size=local_batch_size,
        num_workers=args.num_workers,
        world_size=accelerator.num_processes,
        transform=None,
    )
    val_dataloaders = [
        _prepare_t2i_hf_loader(**val_dataset_config, 
            image_size=256, 
            batch_size=sample_batch_size,
            num_workers=args.num_workers,
            world_size=accelerator.num_processes,
            rank=accelerator.process_index,
            transform=None,
            shuffle=False,
        ) for val_dataset_config in val_dataset_configs
    ]

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # Start with eval mode for all models
    model.eval()
    ema.eval()
    vae.eval()

    # resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + ".pt"
        ckpt_path = f"{args.cont_dir}/checkpoints/{ckpt_name}"

        # If the checkpoint exists, we load the checkpoint and resume the training
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        (optimizer.load_state_dict(ckpt["opt"]),)
        global_step = ckpt["steps"]

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    # Model compilation for better performance

    model, vae, optimizer = accelerator.prepare(
        model, vae, optimizer
    )

    text_encoder_config = {
        "type": "text",
        "cfg_dropout_prob": 0.1,
        "text_encoder": {
            "model_name": "Qwen/Qwen3-0.6B",
            "max_length": 256,
        }
    }
    with accelerator.main_process_first():
        text_encoder = setup_text_encoder(text_encoder_config, device)

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
        model.train()
        vae.eval()

        for raw_image, y in train_dataloader:
            raw_image = raw_image.to(device)            

            model.train()
            requires_grad(model, True)

            with accelerator.autocast():
                # 1). Forward pass: VAE
                processed_image = raw_image

                # dropout
                y = ["" if random.random() < args.cfg_prob else s for s in y]
                with torch.no_grad():
                    text_embed, text_mask = encode_text(text_encoder, y)
                    z = vae.encode(processed_image)
                    # xhat = vae.decode(z)
                # 2). Backward pass: VAE, compute the VAE loss, backpropagate, and update the VAE; Then, compute the riminator loss and update the discriminator
                #    loss_kwargs used for SiT forward function, create here and can be reused for both VAE and SiT
                loss_kwargs = dict(
                    path_type=args.path_type,
                    prediction=args.prediction,
                    weighting=args.weighting,
                )
                # Record the time_input and noises for the VAE alignment, so that we avoid sampling again
                time_input = None
                noises = None

                # 3). Forward pass: SiT
                # **Avoid diffusion loss to backpropagate to the VAE, so we detach the `z`**
                loss_kwargs["weighting"] = args.weighting
                sit_outputs = model(
                    x=z,
                    y=text_embed,
                    time_input=time_input,
                    noises=noises,
                    y_mask=text_mask,
                    loss_kwargs=loss_kwargs,
                )

                sit_loss = sit_outputs["denoising_loss"].mean()

                if args.ploss:
                    pred_x = accelerator.unwrap_model(vae).decode(sit_outputs["pred_x"])
                    ploss = loss_fn_vgg(pred_x, processed_image).mean()
                    sit_loss += ploss
                else:
                    ploss = None

                accelerator.backward(sit_loss)

                if accelerator.sync_gradients:
                    grad_norm_sit = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # psnr = torch.mean(get_psnr(processed_image, xhat, zero_mean=True, integer=True))

                # 5). Update SiT EMA
                if accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(model)
                    update_ema(
                        ema,
                        unwrapped_model._orig_mod if args.compile else unwrapped_model,
                    )

            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "sit_loss": accelerator.gather(sit_loss).mean().detach().item(),
                    "denoising_loss": accelerator.gather(sit_outputs["denoising_loss"])
                    .mean()
                    .detach()
                    .item(),
                    "grad_norm_sit": accelerator.gather(grad_norm_sit)
                    .mean()
                    .detach()
                    .item(),
                    "epoch": epoch,
                    # "psnr": accelerator.gather(psnr).mean().detach().item(),
                }

                if ploss is not None:
                    logs["ploss"] = accelerator.gather(ploss).mean().detach().item()

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # `model` and `vae` are wrapped by the `accelerator` object, so we need to unwrap them
                    unwrapped_model = accelerator.unwrap_model(model)

                    # model might be compiled, we extract the original model
                    original_model = (
                        unwrapped_model._orig_mod if args.compile else unwrapped_model
                    )
                    checkpoint = {
                        "model": original_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if global_step == 1 or (
                global_step % args.sampling_steps == 0 and global_step > 0
            ):
                # NOTE: Inference should use eval mode
                model.eval()
                vae.eval()
                ys = []
                for valy in val_dataloaders[0]:
                    ys += valy[1]
                ys = ys[:sample_batch_size]
                ys_null = ["" for _ in range(len(ys))]
                with torch.no_grad(), accelerator.autocast():
                    ys, ys_mask = encode_text(text_encoder, ys)
                    ys_null, ys_null_mask = encode_text(text_encoder, ys_null)
                    unwrapped_model = accelerator.unwrap_model(model)
                    samples = euler_maruyama_sampler(
                        unwrapped_model,
                        xT,
                        ys,
                        num_steps=50,
                        cfg_scale=6.0,
                        guidance_low=0.0,
                        guidance_high=1.0,
                        path_type=args.path_type,
                        heun=False,
                        y_mask=ys_mask,
                        y_null=ys_null,
                        y_null_mask=ys_null_mask
                    ).to(ys.dtype)
                    samples = unwrapped_model.denormalize_latents(samples)
                    samples = accelerator.unwrap_model(vae).decode(
                        samples
                    )
                    samples = (samples + 1) / 2.0
                out_samples = accelerator.gather(samples.to(torch.float32))

                accelerator.log({"samples": wandb.Image(array2grid(out_samples))})

                logging.info("Generating EMA samples done.")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()

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

    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--qk-norm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fused-attn", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to compile the model for faster training",
    )

    # dataset params
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision params
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--ploss", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument(
        "--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
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

    # loss params
    parser.add_argument(
        "--path-type", type=str, default="linear", choices=["linear", "cosine"]
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="v",
        choices=["v"],
        help="currently we only support v-prediction",
    )
    parser.add_argument(
        "--prediction-internal",
        type=str,
        default="v",
        choices=["v", "x"],
        help="internal prediction type, could be x or v, only used for loss computation and does not affect the model architecture",
    )
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument(
        "--weighting",
        default="uniform",
        type=str,
        choices=["uniform", "lognormal"],
        help="Loss weihgting, uniform or lognormal",
    )

    # vae params
    parser.add_argument("--vae-config", type=str, default="")
    parser.add_argument("--token-drop", type=int, default=-1)
    parser.add_argument("--token-drop-mode", type=str, default="drop")
    parser.add_argument("--token-drop-param", type=float, default=1.0)
    parser.add_argument("--token-drop-dim", type=int, default=1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)