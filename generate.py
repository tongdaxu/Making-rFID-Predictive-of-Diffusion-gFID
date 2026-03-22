# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

Ref:
    https://github.com/sihyun-yu/REPA/blob/main/generate.py
"""

import argparse
import gc
import json
import math
import os

from dictdot import dictdot
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tqdm import tqdm
import shutil

from ifid.sit.sit import SiT_models
from ifid.sit.samplers import euler_sampler, euler_maruyama_sampler, edict_sampler, edict_inverter
from train import denormalize_latents
from datetime import timedelta
from omegaconf import OmegaConf
from ifid.vae.utils import instantiate_from_config
from ifid.fid.psnr import get_psnr
from eval_intp import save_tensor_image
import torch.nn.functional as F

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = (
        args.tf32
    )  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), (
        "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    )
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = (args.global_seed + rank) * dist.get_world_size() // 2
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.pproc_batch_size
    global_batch_size = n * dist.get_world_size()

    if args.exp_path is None or args.train_steps is None:
        if rank == 0:
            print(
                "The `exp_path` or `train_steps` is not provided, setting `exp_path` and `train_steps` to default values."
            )
        args.exp_path = "pretrained/sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k"
        args.train_steps = 400_000

    with open(os.path.join(args.exp_path, "args.json"), "r") as f:
        config = dictdot(json.load(f))

    # Create model:

    vae_config = OmegaConf.load(config.vae_config)
    vae = instantiate_from_config(vae_config).to(device)
    vae.eval()
    for name, param in vae.named_parameters():
        param.requires_grad_(False)

    fake_in = torch.zeros([1, 3, config.resolution, config.resolution]).to(device)
    fake_z = vae.encode(fake_in)[0]

    if len(fake_z.shape) == 3:
        # 2d latent
        latent_size = fake_z.shape[-1]
        in_channels = fake_z.shape[0]
        torch.randn((n, in_channels, latent_size, latent_size), device=device)
        vae_1d = False
    elif len(fake_z.shape) == 2:
        # 1d latent
        latent_size = fake_z.shape[0]
        in_channels = fake_z.shape[-1]
        torch.randn((n, latent_size, in_channels), device=device)
        vae_1d = True
    else:
        assert 0

    tshift = math.sqrt(float(fake_z.numel()) / 4096.0)
    print("using time shift {0:.4f}".format(tshift))

    block_kwargs = {"fused_attn": config.fused_attn, "qk_norm": config.qk_norm}
    model = SiT_models[config.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=config.num_classes,
        class_dropout_prob=config.cfg_prob,
        bn_momentum=config.bn_momentum,
        tshift=tshift,
        **block_kwargs,
    ).to(device)

    exp_name = os.path.basename(args.exp_path)
    train_step_str = str(args.train_steps).zfill(7)
    state_dict = torch.load(
        os.path.join(args.exp_path, "checkpoints", train_step_str + ".pt"),
        map_location=f"cuda:{device}",
    )
    model.load_state_dict(state_dict["ema"], strict=False)
    model.eval()  # Important! To disable label dropout during sampling

    if vae_1d:
        latents_scale = (
            state_dict["ema"]["bn.running_var"]
            .rsqrt()
            .view(1, 1, in_channels)
            .to(device)
        )
        latents_bias = (
            state_dict["ema"]["bn.running_mean"].view(1, 1, in_channels).to(device)
        )
    else:
        latents_scale = (
            state_dict["ema"]["bn.running_var"]
            .rsqrt()
            .view(1, in_channels, 1, 1)
            .to(device)
        )
        latents_bias = (
            state_dict["ema"]["bn.running_mean"].view(1, in_channels, 1, 1).to(device)
        )

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    assert args.cfg_scale >= 1.0, "cfg_scale should be >= 1.0"

    sample_folder_dir = f"{args.sample_dir}/{exp_name}_{train_step_str}_cfg{args.cfg_scale}-{args.guidance_low}-{args.guidance_high}"
    os.makedirs(sample_folder_dir, exist_ok=True)

    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(
        math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size
    )
    if rank == 0:
        print(sample_folder_dir)
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert total_samples % dist.get_world_size() == 0, (
        "total_samples must be divisible by world_size"
    )
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, (
        "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    )
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        if vae_1d:
            z = torch.randn(n, latent_size, model.in_channels, device=device)
        else:
            z = torch.randn(
                n, model.in_channels, latent_size, latent_size, device=device
            )
        y = torch.randint(0, config.num_classes, (n,), device=device)

        assert not args.heun or args.mode == "ode", (
            "Heun's method is only available for ODE sampling."
        )

        # Sample images:
        sampling_kwargs = dict(
            model=model,
            latents=z,
            y=y,
            num_steps=args.num_steps,
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
        )

        with torch.no_grad():
            if args.mode == "sde":
                samples, samples_alt = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32), None
            elif args.mode == "ode":
                samples, samples_alt = euler_sampler(**sampling_kwargs).to(torch.float32), None
            elif args.mode == "edict":
                # convert pixel sit into a VAE
                # read image [-1, 1]
                # image 3x256x256 -> PIXELVAE.encode() -> z
                # z -> normalize -> znrom
                # znorm -> edict_inverter[0] -> latent

                # latent -> edict_sampler[0] -> znorm
                # znorm -> denormalize -> z
                # z -> PIXELVAE.decode() -> image 3x256x256

                samples, samples_2, samples_hist= edict_sampler(**sampling_kwargs)
                samples, samples_2 = samples.to(torch.float32), samples_2.to(torch.float32)
                sampling_kwargs["latents"] = samples
                # sampling_kwargs["latents_2"] = samples_2
                inv_samples, inv_samples_2, inverse_hist = edict_inverter(**sampling_kwargs)
                mse_z = torch.mean((inv_samples - z)**2)
                print(mse_z, mse_z / torch.mean(z**2))

                samples_hist = samples_hist[::-1]
                for i in range(len(samples_hist)):
                    print(i, torch.mean((samples_hist[i] - inverse_hist[i])**2).item())

                save_tensor_image(F.pixel_shuffle(torch.clamp(inv_samples, -1, 1), 16), "z1.png")
                save_tensor_image(F.pixel_shuffle(torch.clamp(inv_samples_2, -1, 1), 16), "z2.png")

                sampling_kwargs["latents"] = inv_samples.to(torch.float32)
                samples_alt, samples_alt_2, _ = edict_sampler(**sampling_kwargs)
                samples_alt, samples_alt_2 = samples_alt.to(torch.float32), samples_alt_2.to(torch.float32)
            else:
                raise NotImplementedError()

            samples = vae.decode(
                denormalize_latents(samples, latents_scale, latents_bias)
            )
            samples = (samples + 1) / 2.0

            if samples_alt is not None:
                samples_alt = vae.decode(
                    denormalize_latents(samples_alt, latents_scale, latents_bias)
                )
                samples_alt = (samples_alt + 1) / 2.0
                pred_psnr = get_psnr(samples, samples_alt, zero_mean=True)
                print("edict psnr: {0:.2f}".format(torch.mean(pred_psnr).item()))

            samples = (
                torch.clamp(255.0 * samples, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

                assert(0)
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        # print("create npz done.")
        from ifid.fid.fid import calculate_fid_given_paths

        fid_num = args.num_fid_samples
        print("Calculating FID with {} number of samples".format(fid_num))
        fid_reference_file = args.fid_reference_file
        fid = calculate_fid_given_paths(
            [fid_reference_file, sample_folder_dir],
            batch_size=50,
            dims=2048,
            device="cuda",
            num_workers=8,
            sp_len=fid_num,
        )
        print("fid=", fid)
        shutil.rmtree(sample_folder_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed params
    parser.add_argument("--global-seed", type=int, default=0)

    # precision params
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.",
    )

    # logging/saving params
    parser.add_argument("--sample-dir", type=str, default="samples")

    # ckpt params
    parser.add_argument(
        "--exp-path",
        type=str,
        default=None,
        help="Path to the specific experiment directory.",
    )
    parser.add_argument(
        "--train-steps",
        type=str,
        default=None,
        help="The checkpoint of the model to sample from.",
    )

    # number of samples
    parser.add_argument("--pproc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument(
        "--path-type", type=str, default="linear", choices=["linear", "cosine"]
    )
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--heun",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Heun's method for ODE sampling.",
    )
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--fid-reference-file", type=str, default="")

    args = parser.parse_args()
    main(args)
