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
from ifid.vae.utils import instantiate_from_config, load_encoders
from ifid.loss.losses import ReconstructionLoss_Simple
from PIL import Image 
from accelerate.utils import DistributedDataParallelKwargs
import lpips
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from dictdot import dictdot
import vision_aided_loss
from ifid.sit.samplers import euler_sampler, euler_maruyama_sampler, edict_sampler, edict_inverter

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

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    else:
        raise NotImplementedError

    return x

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
    sample_batch_size = 64 // accelerator.num_processes
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)

    with open(os.path.join(args.load_exp_path, "args.json"), "r") as f:
        config = dictdot(json.load(f))

    # Create model:
    vae_config = OmegaConf.load(config.vae_config)
    vae = instantiate_from_config(vae_config).to(device)

    for name, param in vae.named_parameters():
        if "decoder" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    loss_cfg = OmegaConf.load(args.loss_cfg_path)

    with accelerator.local_main_process_first():
        vae_loss_fn = ReconstructionLoss_Simple(
            loss_cfg
        ).to(device)

    z_dims = None

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
        xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)
    elif len(fake_z.shape) == 2:
        # 1d latent
        latent_size = fake_z.shape[0]
        in_channels = fake_z.shape[-1]
        xT = torch.randn((n, latent_size, in_channels), device=device)
    else:
        assert 0

    with accelerator.local_main_process_first():
        disc = vision_aided_loss.Discriminator(cv_type='dino', loss_type='multilevel_sigmoid_s', device=device, num_classes=config.num_classes).to(device)
    disc.cv_ensemble.requires_grad_(False) # Freeze feature extractor

    tshift = math.sqrt(float(fake_z.numel()) / 4096.0)
    block_kwargs = {"fused_attn": config.fused_attn, "qk_norm": config.qk_norm}
    model = SiT_models[config.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=config.num_classes,
        class_dropout_prob=config.cfg_prob,
        bn_momentum=config.bn_momentum,
        tshift=tshift,
        prediction_internal=config.prediction_internal,
        tk_drop=config.token_drop,
        tk_drop_mode=config.token_drop_mode,
        tk_drop_param=config.token_drop_param,
        tk_drop_dim=config.token_drop_dim,
        z_dims=z_dims,
        **block_kwargs,
    ).to(device)

    # make a copy of the model for EMA
    train_step_str = str(args.load_train_steps).zfill(7)
    state_dict = torch.load(
        os.path.join(args.load_exp_path, "checkpoints", train_step_str + ".pt"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict["ema"], strict=False)
    model.eval()
    requires_grad(model, False)

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer_vae = torch.optim.AdamW(
        vae.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_disc = torch.optim.AdamW(
        disc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = CustomINH5Dataset(args.data_dir)
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

    # Start with eval mode for all models
    model.eval()
    vae.eval()

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    # Model compilation for better performance

    vae, disc, optimizer_vae, optimizer_disc, train_dataloader = accelerator.prepare(
        vae, disc, optimizer_vae, optimizer_disc, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="diffusion vae arena",
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}"}},
        )

    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(args.epochs):
        model.eval()
        vae.train()

        for raw_image, y in train_dataloader:
            raw_image = raw_image.to(device)
            labels = y.to(device)

            with accelerator.autocast():
                # 1). Forward pass: VAE
                processed_image = preprocess_imgs_vae(raw_image)

                z, info = vae(processed_image)
                recon_image, posterior = info["xhat"], info["posterior"]

                with torch.no_grad():
                    init_noise = torch.randn_like(z)
                    sampling_kwargs = dict(
                        model=model,
                        latents=init_noise,
                        y=labels,
                        num_steps=50,
                        heun=False,
                        cfg_scale=1.0,
                        guidance_low=0.0,
                        guidance_high=1.0,
                        path_type=config.path_type,
                    )
                    samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)

                # with torch.autograd.set_detect_anomaly(True):
                decoded_image = vae(x=None, z=samples)[1]["xhat"]

                loss_gen = disc(decoded_image, c=labels, for_G=True)
                vae_loss, vae_loss_dict = vae_loss_fn(processed_image, recon_image, posterior, global_step, "generator")
                # vae_loss = vae_loss.mean() + loss_gen.mean() * args.vae_gen_coeff
                vae_loss = loss_gen.mean()

                if global_step >= args.disc_warmup:
                    # vae update 
                    vae_loss.backward()
                    optimizer_vae.step()

                optimizer_vae.zero_grad(set_to_none=True)

                psnr = torch.mean(get_psnr(processed_image, recon_image.detach(), zero_mean=True, integer=True))

                # disc update
                loss_disc_r = disc(processed_image, c=labels, for_real=True).mean()
                # loss_disc = disc(processed_image, c=labels, for_real=True) + disc(decoded_image.clone().detach(), c=labels, for_real=False)
                loss_disc_r.backward()
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)

                loss_disc_f = disc(decoded_image.detach(), c=labels, for_real=False).mean()
                loss_disc_f.backward()
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)

                loss_disc = loss_disc_r + loss_disc_f

            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "epoch": epoch,
                    "psnr": accelerator.gather(psnr).mean().detach().item(),
                    "vae_loss": accelerator.gather(vae_loss).mean().detach().item(),
                    "reconstruction_loss": accelerator.gather(vae_loss_dict["reconstruction_loss"].mean()).mean().detach().item(),
                    "perceptual_loss": accelerator.gather(vae_loss_dict["perceptual_loss"].mean()).mean().detach().item(),
                    "kl_loss": accelerator.gather(vae_loss_dict["kl_loss"].mean()).mean().detach().item(),
                    "loss_disc": accelerator.gather(loss_disc).mean().detach().item(),
                    "loss_gen": accelerator.gather(loss_gen).mean().detach().item(),
                }

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # `model` and `vae` are wrapped by the `accelerator` object, so we need to unwrap them
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_vae = accelerator.unwrap_model(vae)

                    # model might be compiled, we extract the original model
                    original_model = (
                        unwrapped_model._orig_mod if args.compile else unwrapped_model
                    )
                    checkpoint = {
                        "model": original_model.state_dict(),
                        "vae": unwrapped_vae.state_dict(),
                        "opt_disc": optimizer_disc.state_dict(),
                        "opt_vae": optimizer_vae.state_dict(),
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
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    samples = euler_sampler(
                        unwrapped_model,
                        xT,
                        ys,
                        num_steps=50,
                        cfg_scale=1.0,
                        guidance_low=0.0,
                        guidance_high=1.0,
                        path_type=config.path_type,
                        heun=False,
                    ).to(torch.float32)
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
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)
    parser.add_argument("--wandb-history-path", type=str, default=None)
    parser.add_argument("--disc-warmup", type=int, default=-1)

    parser.add_argument("--load-exp-path", type=str, required=True)
    parser.add_argument("--load-train-steps", type=str, required=True)

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

    # alignment
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-l')
    parser.add_argument("--vae-gen-coeff", type=float, default=1.0)

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
    parser.add_argument("--num-workers", type=int, default=16)

    # loss params
    parser.add_argument(
        "--path-type", type=str, default="linear", choices=["linear", "cosine"]
    )
    parser.add_argument("--loss-cfg-path", type=str, default="./configs/l1_lpips_kl.yaml")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)