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
from PIL import Image 
from accelerate.utils import DistributedDataParallelKwargs
import lpips
import torchvision.transforms.v2 as v2
from ifid.sit.sit import mean_flat
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import torch.nn.functional as F

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

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    else:
        raise NotImplementedError

    return x


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

_crop = v2.RandomResizedCrop(
    (256, 256),
    scale=(0.85, 1.0),
    interpolation=v2.InterpolationMode.BICUBIC,
    antialias=True,
)

_jitter = v2.RandomApply(
    [
        v2.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        )
    ],
    p=0.8,
)

def augment_crop(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (B, C, H, W), float in [-1, 1]

    Returns:
        Augmented tensor in [-1, 1].
    """
    assert x.ndim == 4

    # [-1, 1] -> [0, 1]
    x = (x + 1.0) * 0.5

    out = []
    for img in x:

        # Always crop + resize
        img = _crop(img)

        # Color jitter with p=0.8
        # img = _jitter(img)

        out.append(img)

    x = torch.stack(out)

    # [0, 1] -> [-1, 1]
    return x.mul(2).sub(1).clamp(-1, 1)

def augment_mask(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (B, C, H, W), float in [-1, 1]

    Returns:
        Augmented tensor in [-1, 1].
    """
    assert x.ndim == 4

    # [-1, 1] -> [0, 1]
    x = (x + 1.0) * 0.5

    out = []
    for img in x:

        # Always crop + resize
        img = _crop(img)

        # Color jitter with p=0.8
        # img = _jitter(img)

        out.append(img)

    x = torch.stack(out)

    # [0, 1] -> [-1, 1]
    return x.mul(2).sub(1).clamp(-1, 1)

def random_spatial_mask(x, mask_ratio=0.75):
    # Generate random mask per sample
    B, C, H, W = x.shape
    device = x.device
    mask = torch.rand(B, H, W, device=device) < mask_ratio # Apply mask to all channels
    x_masked = x.masked_fill(mask.unsqueeze(1), 0.0)
    return x_masked
#################################################################################
#                                  Training Loop                                #
#################################################################################


def cal_align_cos(fs, fs_tilde):
    fs_tilde = torch.nn.functional.normalize(fs_tilde, dim=-1)
    fs = torch.nn.functional.normalize(fs, dim=-1)
    align_loss = mean_flat((fs * fs_tilde).sum(dim=-1))
    return align_loss

def cal_entropy_cos(fs):
    fs = torch.nn.functional.normalize(fs, dim=-1)
    sim = torch.bmm(
        fs.permute(1, 0, 2),
        fs.permute(1, 2, 0),
    ).permute(1, 2, 0)
    idx = torch.arange(sim.size(0), device=sim.device)
    sim[idx, idx, :] = 0
    entropy_loss = mean_flat(sim)
    return entropy_loss

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
    if args.ploss:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    # Create model:
    vae_config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for name, param in vae.named_parameters():
        param.requires_grad_(False)

    if args.enc_type != None:
        with accelerator.local_main_process_first():
            encoders, encoder_types, architectures = load_encoders(
                args.enc_type, device, args.resolution
            )
    else:
        raise NotImplementedError()

    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]

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

    tshift = math.sqrt(float(fake_z.numel()) / 4096.0)

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob,
        bn_momentum=args.bn_momentum,
        tshift=tshift,
        prediction_internal=args.prediction_internal,
        tk_drop=args.token_drop,
        tk_drop_mode=args.token_drop_mode,
        tk_drop_param=args.token_drop_param,
        tk_drop_dim=args.token_drop_dim,
        z_dims=z_dims,
        **block_kwargs,
    )

    # make a copy of the model for EMA
    model = model.to(device)
    model_copy = copy.deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training

    ema = copy.deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    requires_grad(model_copy, False)

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

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    update_ema(model_copy, model, decay=0)

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

    model, vae, optimizer, train_dataloader = accelerator.prepare(
        model, vae, optimizer, train_dataloader
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
        model.train()

        for raw_image, y in train_dataloader:
            raw_image = raw_image.to(device)
            labels = y.to(device)

            model.train()
            requires_grad(model, True)

            with accelerator.autocast():
                # 1). Forward pass: VAE
                processed_image = preprocess_imgs_vae(raw_image)

                vae.eval()
                with torch.no_grad():
                    processed_image_plus = augment_crop(processed_image)
                    z, z_plus = torch.chunk(vae.encode(torch.cat([processed_image, processed_image_plus], dim=0)), chunks=2, dim=0)
                    xhat = vae.decode(z)

                    zs = []
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        zf = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: zf = zf[:, 1:] 
                        if 'dinov2' in encoder_type: zf = zf['x_norm_patchtokens']
                        zs.append(zf)

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
                    y=labels,
                    zs=zs,
                    loss_kwargs=loss_kwargs,
                    time_input=time_input,
                    noises=noises,
                    return_feat=False,
                )

                sit_loss = sit_outputs["denoising_loss"].mean()

                if args.ploss:
                    pred_x = accelerator.unwrap_model(vae).decode(sit_outputs["pred_x"])
                    ploss = loss_fn_vgg(pred_x, processed_image).mean()
                    sit_loss += ploss
                else:
                    ploss = None

                if args.dloss > 0.0:
                    # pred_x = accelerator.unwrap_model(vae).decode(sit_outputs["pred_x"])

                    time_in = sit_outputs["time_input"]
                    time_next = time_in * 0.5
                    a_next = ((time_next * (1 - time_in)) / (time_in * (1 - time_next) + 1e-3)).to(z.device)
                    z_next = a_next * z + (1 - a_next) * sit_outputs["pred_x"]
                    next_outputs = model_copy(
                        x=z_next,
                        y=labels,
                        loss_kwargs=loss_kwargs,
                        time_input=time_next,
                        noises=noises,
                        return_feat=False,
                    )
                    pred_x = accelerator.unwrap_model(vae).decode(next_outputs["pred_x"])

                    x_t = ((pred_x + 1.0) / 2.0) * 255.0
                    z_dis = encoder.forward_features(preprocess_raw_image(x_t, encoder_type))['x_norm_patchtokens']
                    z_ref = zs[0]
                    z_dis = F.normalize(z_dis, dim=-1) 
                    z_ref = F.normalize(z_ref, dim=-1) 
                    dloss = mean_flat(-(z_dis * z_ref).sum(dim=-1))
                    dloss = torch.mean(dloss)
                    sit_loss += dloss * args.dloss
                else:
                    dloss = None

                if args.spdloss > 0.0:
                    '''
                    time_output = sit_outputs["time_input"].reshape(z.shape[0])
                    time_mask = (time_output < 0.5).to(device=z.device, dtype=z.dtype)
                    pred_x = accelerator.unwrap_model(vae).decode(sit_outputs["pred_x"])
                    spdloss = torch.mean((pred_x - processed_image)**2, dim=tuple(range(1, z.ndim)))
                    spdloss = torch.sum(spdloss * time_mask) / (torch.sum(time_mask) + 1e-3)
                    '''
                    time_in = sit_outputs["time_input"]
                    time_next = time_in * 0.5
                    a_next = ((time_next * (1 - time_in)) / (time_in * (1 - time_next) + 1e-3)).to(z.device)
                    z_next = a_next * z + (1 - a_next) * sit_outputs["pred_x"]
                    next_outputs = model_copy(
                        x=z_next,
                        y=labels,
                        loss_kwargs=loss_kwargs,
                        time_input=time_next,
                        noises=noises,
                        return_feat=False,
                    )
                    pred_x = accelerator.unwrap_model(vae).decode(next_outputs["pred_x"])
                    spdloss = torch.mean((pred_x - processed_image)**2)
                    sit_loss += spdloss * args.spdloss
                else:
                    spdloss = None

                proj_loss = sit_outputs["proj_loss"].mean()

                diffusion_loss = sit_loss + proj_loss * args.proj_coeff
                accelerator.backward(diffusion_loss)

                if accelerator.sync_gradients:
                    grad_norm_sit = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                psnr = torch.mean(get_psnr(processed_image, xhat, zero_mean=True, integer=True))

                # 5). Update SiT EMA
                if accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(model)
                    update_ema(
                        ema,
                        unwrapped_model._orig_mod if args.compile else unwrapped_model,
                    )
                    update_ema(
                        model_copy,
                        unwrapped_model._orig_mod if args.compile else unwrapped_model,
                        decay=0.0,
                    )

            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "sit_loss": accelerator.gather(sit_loss).mean().detach().item(),
                    "proj_loss": accelerator.gather(proj_loss).mean().detach().item(),
                    "denoising_loss": accelerator.gather(sit_outputs["denoising_loss"]).mean().detach().item(),
                    "grad_norm_sit": accelerator.gather(grad_norm_sit).mean().detach().item(),
                    "epoch": epoch,
                    "psnr": accelerator.gather(psnr).mean().detach().item(),
                }

                if ploss is not None:
                    logs["ploss"] = accelerator.gather(ploss).mean().detach().item()

                if spdloss is not None:
                    logs["spdloss"] = accelerator.gather(spdloss).mean().detach().item()

                if dloss is not None:
                    logs["dloss"] = accelerator.gather(dloss).mean().detach().item()

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
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    samples = euler_sampler(
                        unwrapped_model,
                        xT,
                        ys,
                        num_steps=50,
                        cfg_scale=2.0,
                        guidance_low=0.0,
                        guidance_high=1.0,
                        path_type=args.path_type,
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
    parser.add_argument("--data-dir", type=str, default="data")
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
        "--spdloss", type=float, default=0.0,
    )
    parser.add_argument(
        "--dloss", type=float, default=0.0,
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

    parser.add_argument("--enc-type", type=str, default='dinov2-vit-l')
    parser.add_argument("--proj-coeff", type=float, default=0.5)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)