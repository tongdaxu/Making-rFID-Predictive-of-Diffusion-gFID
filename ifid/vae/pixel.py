from ifid.vae.autoencoder import AutoencoderKL
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ifid.vae.flux2vae import OrthogonalConv1x1
import random
import INN
from INN.CouplingModels.NICEModel.conv import Conv2dNICE
from ifid.vae.dino import ChannelShuffle

class PIXELVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return F.pixel_unshuffle(x, 16)

    def decode(self, z, *args, **kwargs):
        return F.pixel_shuffle(z, 16)

class PIXELD4(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, x, *args, **kwargs):
        xd4 = F.interpolate(x, scale_factor=0.25, mode="bicubic")
        zd4 = F.pixel_unshuffle(xd4, 4)
        xd4hat = F.interpolate(xd4, scale_factor=4, mode="bicubic")
        xres = x - xd4hat
        zres = F.pixel_unshuffle(xres, 16)
        return torch.cat([zd4, zres], dim=1)

    def decode(self, z, *args, **kwargs):
        zd4, zres = z[:, :48], z[:, 48:]
        xd4 = F.pixel_shuffle(zd4, 4)
        xres = F.pixel_shuffle(zres, 16)
        xd4hat = F.interpolate(xd4, scale_factor=4, mode="bicubic")
        x = xres + xd4hat
        return x

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat, }

class PIXELFLOW(nn.Module):
    def __init__(self, flow_ckpt_path=None):
        super().__init__()
        self.flow = INN.Sequential(
            Conv2dNICE(768, 3),
            ChannelShuffle(),
            Conv2dNICE(768, 3),
            ChannelShuffle(),
            Conv2dNICE(768, 3),
            ChannelShuffle(),
            Conv2dNICE(768, 3),
        )
        self.flow.computing_p(True)
        if flow_ckpt_path is not None:
            flow_ckpt = torch.load(flow_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
            miss_keys, unexp_keys = self.load_state_dict(flow_ckpt, strict=False)
            print("miss_keys: ", miss_keys)
            print("unexp_keys: ", unexp_keys)

    def encode(self, x, *args, **kwargs):
        z = F.pixel_unshuffle(x, 16)
        z = self.flow(z)[0]
        return z

    def decode(self, z, *args, **kwargs):
        z = self.flow.inverse(z)
        z = F.pixel_shuffle(z, 16)
        return z

    def forward(self, x):
        z = self.encode(x)
        with torch.no_grad():
            z_base = F.pixel_unshuffle(x, 16)
            xhat = self.decode(z)
        return z, {"xhat": xhat, "base_loss": torch.mean(torch.sum(z_base**2, dim=1))}

class PIXELVAEPCA(nn.Module):
    def __init__(self, *args, flow_ckpt_path=None, debug=-1, **kwargs):
        super().__init__()
        self.embed_dim = 768
        self.vae = PIXELVAE()
        self.flow = OrthogonalConv1x1(self.embed_dim)
        # self.mask = [(i+1)*(self.embed_dim // 4) for i in range(4)]
        self.mask = [32, 64, 192, 768]
        if flow_ckpt_path is not None:
            flow_ckpt = torch.load(flow_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
            miss_keys, unexp_keys = self.load_state_dict(flow_ckpt, strict=False)
            print("miss_keys: ", miss_keys)
            print("unexp_keys: ", unexp_keys)
        self.debug = debug

    def encode(self, x, *args, **kwargs):
        return F.pixel_unshuffle(x, 16)

    def decode(self, z, *args, **kwargs):
        return F.pixel_shuffle(z, 16)

    def encode(self, x, *args, **kwargs):
        z = self.vae.encode(x)
        z = self.flow(z)
        if self.debug > 0:
            z[:,self.debug:] = 0
        return z
    
    def decode(self, z, *args, **kwargs):
        if self.debug > 0:
            z[:,self.debug:] = 0
        z = self.flow(z, reverse=True)
        xhat = self.vae.decode(z)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        keep_d = random.choice(self.mask)
        z_partial = torch.cat([z[:, :keep_d], torch.zeros_like(z)[:, keep_d:]], dim=1)
        xhat_partial = self.decode(z_partial)
        with torch.no_grad():
            xhat_test = self.decode(z)
            xhat = self.vae.decode(self.vae.encode(x))
        return z, {"xhat": xhat, "xhat_partial": xhat_partial, "xhat_test": xhat_test}

    def test(self, x, keep_id):
        z = self.vae.encode(x)
        keep_d = self.mask[keep_id]

        z_original = torch.cat([z[:, :keep_d], torch.zeros_like(z)[:, keep_d:]], dim=1)
        z_flow = self.flow(
                    torch.cat(
                        [self.flow(z)[:, :keep_d], torch.zeros_like(z)[:, keep_d:]],
                    dim=1),
                 reverse=True)

        x_original = self.vae.decode(z_original)
        x_flow = self.vae.decode(z_flow)

        return x_original, x_flow

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

    configs = OmegaConf.load("./configs/PIXD4.yaml")
    vae = instantiate_from_config(configs)
    x = torch.randn([1, 3, 256, 256])
    z = vae.encode(x)
    xhat = vae.decode(z)
    print(z.shape, xhat.shape, torch.mean((x - xhat)**2))
