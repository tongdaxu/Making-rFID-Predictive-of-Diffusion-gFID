# ifid/vae/flux2vae.py

import os
import math
from typing import Optional

import torch
from torch import nn
from einops import rearrange
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL


class Flux2VAE(nn.Module):
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        repo_id: str = "Comfy-Org/flux2-dev",
        filename: str = "split_files/vae/flux2-vae.safetensors",
        resolution: int = 256,
        strict: bool = True,
    ):
        super().__init__()
        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            latent_channels=32,
            norm_num_groups=32,
            sample_size=resolution,
            scaling_factor=1.0,
            shift_factor=None,
            force_upcast=False,
            use_quant_conv=True,
            use_post_quant_conv=True,
            mid_block_add_attention=True,
        )

        self.ps = [2, 2]
        self.bn_eps = 1e-4
        self.bn_momentum = 0.1

        # packed channels = 32 * 2 * 2 = 128
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * 32,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

        if ckpt_path is None:
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
        elif not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt_path does not exist: {ckpt_path}")

        print(f"[Flux2VAE] loading ckpt: {ckpt_path}")
        state_dict = load_file(ckpt_path, device="cpu")

        # Split VAE weights and FLUX2 latent BN stats.
        bn_sd = {}
        vae_sd = {}

        for k, v in state_dict.items():
            if k.startswith("bn."):
                bn_sd[k[len("bn."):]] = v
            else:
                vae_sd[k] = v

        print(f"[Flux2VAE] total ckpt keys: {len(state_dict)}")
        print(f"[Flux2VAE] vae keys: {len(vae_sd)}")
        print(f"[Flux2VAE] bn keys: {len(bn_sd)}")

        vae_msg = self.vae.load_state_dict(vae_sd, strict=strict)
        bn_msg = self.bn.load_state_dict(bn_sd, strict=False)

        print("[Flux2VAE] VAE missing:", len(vae_msg.missing_keys))
        print("[Flux2VAE] VAE unexpected:", len(vae_msg.unexpected_keys))
        if len(vae_msg.missing_keys) > 0:
            print("[Flux2VAE] first VAE missing:", vae_msg.missing_keys[:20])
        if len(vae_msg.unexpected_keys) > 0:
            print("[Flux2VAE] first VAE unexpected:", vae_msg.unexpected_keys[:20])

        print("[Flux2VAE] BN missing:", bn_msg.missing_keys)
        print("[Flux2VAE] BN unexpected:", bn_msg.unexpected_keys)

        self.vae.eval()
        self.bn.eval()

        for p in self.parameters():
            p.requires_grad_(False)

    def normalize(self, z):
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    @torch.no_grad()
    def encode(self, x, *args, **kwargs):
        # x: B,3,H,W, range [-1,1]
        posterior = self.vae.encode(x).latent_dist

        # FLUX2 official logic uses mean, not posterior sampling.
        z = posterior.mean

        # B,32,32,32 -> B,128,16,16 for 256 input
        z = rearrange(
            z,
            "b c (i pi) (j pj) -> b (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )

        z = self.normalize(z)
        return z

    @torch.no_grad()
    def decode(self, z, *args, **kwargs):
        # B,128,16,16 -> B,32,32,32
        z = self.inv_normalize(z)

        z = rearrange(
            z,
            "b (c pi pj) i j -> b c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )

        dec = self.vae.decode(z).sample
        return dec