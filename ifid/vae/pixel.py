from ifid.vae.autoencoder import AutoencoderKL
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from train import normalize_latents, denormalize_latents
from ifid.sit.samplers import euler_sampler, euler_maruyama_sampler, edict_sampler, edict_inverter


class PIXELVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return F.pixel_unshuffle(x, 16)

    def decode(self, z, *args, **kwargs):
        return F.pixel_shuffle(z, 16)

'''
convert pixel sit into a VAE:

1. read image [-1, 1]

2. image 3x256x256 -> PIXELVAE.encode() -> z

3. z -> normalize -> znrom
function: normalize_latents

4. znorm -> edict_inverter[0] -> latent
function: edict_inverter, 取第0个返回

5. latent -> edict_sampler[0] -> znorm
function: edict_sampler, 取第0个返回

6. znorm -> denormalize -> z
function: denormalize_latents

7. z -> PIXELVAE.decode() -> image 3x256x256
'''


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/PIXELVAE.yaml")
    vae = instantiate_from_config(configs)
    z = vae.encode(torch.randn([1, 3, 256, 256]))
    xhat = vae.decode(z)
    print(z.shape, xhat.shape)
