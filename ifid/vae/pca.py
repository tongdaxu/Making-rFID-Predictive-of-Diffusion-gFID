import os
import torch
import torch.nn.functional as F
from torch import nn
from contextlib import nullcontext
from timm.models import create_model

from dataclasses import dataclass
from PIL import Image
import numpy as np

from einops import rearrange
from torch import Tensor

from huggingface_hub import hf_hub_download
import torchvision
import torch
import torch_dct as dct
import torch
import matplotlib.pyplot as plt
from skimage import data
import pytorch_wavelet as wavelet
from haar_pytorch import HaarForward, HaarInverse

def read_image_tensor(path):
    """
    Read image and return tensor of shape (1,3,256,256) in range [-1,1]
    """
    img = Image.open(path).convert("RGB").resize((256, 256))
    x = torch.from_numpy(np.array(img)).float() / 255.0  # [0,1]
    x = x.permute(2, 0, 1)                               # (3,H,W)
    x = x * 2 - 1                                         # [-1,1]
    x = x.unsqueeze(0)                                    # (1,3,256,256)
    return x


def save_tensor_image(x, path):
    """
    Save tensor of shape (1,3,256,256) with range [-1,1] to PNG
    """
    x = x.detach().cpu()

    if x.dim() == 4:
        x = x[0]

    x = (x + 1) / 2                       # [-1,1] -> [0,1]
    x = x.clamp(0, 1)

    x = (x * 255).byte()
    x = x.permute(1, 2, 0).numpy()        # (H,W,3)

    img = Image.fromarray(x)
    img.save(path)

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


class HAARVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fw = HaarForward()
        self.bw = HaarInverse()

    def encode(self, x, *args, **kwargs):
        
        a128 = self.fw(x)
        a128, b128 = a128[:,:3], a128[:,3:]

        a64 = self.fw(a128)
        a64, b64 = a64[:,:3], a64[:,3:]

        a32 = self.fw(a64)
        a32, b32 = a32[:,:3], a32[:,3:]

        a16 = self.fw(a32)

        b32 = F.pixel_unshuffle(b32, 2)
        b64 = F.pixel_unshuffle(b64, 4)
        b128 = F.pixel_unshuffle(b128, 8)

        # 12, 9 * 4, 9 * 16, 9 * 64
        z16 = torch.cat([a16, b32, b64, b128], dim=1)

        return z16

    def decode(self, z, *args, **kwargs):
        a16, b32, b64, b128 = torch.split(z, [12, 36, 144, 576], dim=1)

        b32 = F.pixel_shuffle(b32, 2)
        b64 = F.pixel_shuffle(b64, 4)
        b128 = F.pixel_shuffle(b128, 8)
        
        a32 = self.bw(a16)
        a32 = torch.cat([a32, b32], dim=1)
        
        a64 = self.bw(a32)
        a64 = torch.cat([a64, b64], dim=1)

        a128 = self.bw(a64)
        a128 = torch.cat([a128, b128], dim=1)

        x256_hat = self.bw(a128)

        return x256_hat

if __name__ == "__main__":


    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/HAAR.yaml")
    sdvae = instantiate_from_config(configs)
    x = torch.randn([1, 3, 256, 256])
    z = sdvae.encode(x)
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape, torch.mean(torch.abs(x - xhat)))
