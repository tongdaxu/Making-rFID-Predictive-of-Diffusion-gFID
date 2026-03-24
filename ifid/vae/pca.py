import os
import torch
import torch.nn.functional as F
from torch import nn
from contextlib import nullcontext
from timm.models import create_model

from dataclasses import dataclass

from einops import rearrange
from torch import Tensor

from huggingface_hub import hf_hub_download

import torch
import torch_dct as dct

class DCTVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, x, *args, **kwargs):
        y = F.pixel_unshuffle(x, 16)
        z = dct.dct_2d(y)
        return z

    def decode(self, z, *args, **kwargs):
        y = dct.idct_2d(z)
        x = F.pixel_shuffle(y, 16)
        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/DCT.yaml")
    vae = instantiate_from_config(configs)
    z = vae.encode(torch.randn([1, 3, 256, 256]))
    xhat = vae.decode(z)
    print(z.shape, xhat.shape)
