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
import torch.nn.functional as F
from omegaconf import OmegaConf
from ifid.vae.utils import instantiate_from_config
from ifid.vae.dino import Conv2dNICEFast2, ChannelShuffle
import INN
import torch_dct as dct


class Flux2VAE(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
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

        if not os.path.exists(ckpt_path):
            repo_id_1, repo_id_2, fname = ckpt_path.split("/", 2)
            repo_id = f"{repo_id_1}/{repo_id_2}"
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
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
    
class Flux2VAERes(nn.Module):
    def __init__(self, ckpt_path: str, debug=False, debug_mode="zero", debug_rae_yaml="", resolution: int = 256, strict: bool = True):
        super().__init__()
        self.vae = Flux2VAE(ckpt_path, resolution, strict)
        self.downrate = 16
        self.embed_dim = 128
        self.debug=debug
        self.debug_mode=debug_mode
        if self.debug_mode == "rae":
            rae_configs = OmegaConf.load(debug_rae_yaml)
            self.rae = instantiate_from_config(rae_configs)
        else:
            self.rae = None

    def encode(self, x, *args, **kwargs):
        z = self.vae.encode(x)
        xhat = self.vae.decode(z)
        res = x - xhat
        res = F.pixel_unshuffle(res, self.downrate)
        if self.debug:
            if self.debug_mode == "zero":
                res = torch.zeros_like(res)  # 0 padding debug
            elif self.debug_mode == "repeat":
                factor = res.shape[1] // self.embed_dim
                res = torch.cat([z] * factor, dim=1)  # repeat debug
            elif self.debug_mode == "rae":
                res = self.rae.encode(x)  # noise debug
        return torch.cat([z, res], dim=1)
    
    def decode(self, z, *args, **kwargs):
        z, res = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res = F.pixel_shuffle(res, self.downrate)
        xhat = self.vae.decode(z)
        if self.debug:
            res = torch.zeros_like(res)  # debug purpose only
        x = xhat + res
        return x

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}
    
class Flux2Lossy(nn.Module):
    def __init__(self, ckpt_path: str, scale=4, resolution: int = 256, strict: bool = True):
        super().__init__()
        self.vae = Flux2VAE(ckpt_path, resolution, strict)
        self.embed_dim = 128
        self.scale = scale

    def encode(self, x, *args, **kwargs):
        z = self.vae.encode(x)
        zlow = F.interpolate(z, scale_factor=1/self.scale, mode='bilinear', antialias=True)
        zlow = F.interpolate(zlow, scale_factor=self.scale)
        zres = z - zlow
        return torch.cat([zlow, zres], dim=1)
    
    def decode(self, z, *args, **kwargs):
        zlow, zres = z[:, :self.embed_dim], z[:, self.embed_dim:]
        recz = zlow + zres
        xhat = self.vae.decode(recz)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}

class Flux2VAEResflow(nn.Module):
    def __init__(self, ckpt_path: str, flow_ckpt_path=None, round=False, size="small", resolution: int = 256, strict: bool = True):
        super().__init__()
        self.vae = Flux2VAE(ckpt_path, resolution, strict)
        self.downrate = 16
        self.embed_dim = 128
        self.size = size
        self.round = round
        if self.size == "small":
            self.flow = INN.Sequential(
                INN.Conv2d(768, 3),
            )
        elif self.size == "medium":
            self.flow = INN.Sequential(
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
            )
        elif self.size == "large":
            self.flow = INN.Sequential(
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
                ChannelShuffle(),
                INN.Conv2d(768, 3),
            )
        else:
            raise ValueError
        
        # self alignment
        self.flow_align = nn.Conv2d(
            self.embed_dim, 768, kernel_size=1, bias=False
        )

        self.flow.computing_p(False)

        if flow_ckpt_path is not None:
            flow_ckpt = torch.load(flow_ckpt_path, map_location="cpu", weights_only=False)["flow"]
            miss_keys, unexp_keys = self.flow.load_state_dict(flow_ckpt, strict=False)
            print("miss_keys: ", miss_keys)
            print("unexp_keys: ", unexp_keys)

    def get_last_layer(self):
        if self.size == "medium":
            return self.flow[6].m.f[4].weight
        elif self.size == "small":
            return self.flow[0].m.f[4].weight
        elif self.size == "large":
            return self.flow[30].m.f[4].weight
        else:
            raise ValueError

    def encode(self, x, *args, **kwargs):
        with torch.no_grad():
            z = self.vae.encode(x)
            xhat = self.vae.decode(z)
            if self.round:
                xhat = torch.round(xhat * 127.5) / 127.5
            res = x - xhat
            res = F.pixel_unshuffle(res, self.downrate)
        if self.training:
            noise = torch.empty_like(res).uniform_(-0.5, 0.5) / 127.5
            res = res + noise
            res_z = self.flow(res * 127.5)
        else:
            res_z = self.flow(res * 127.5)
        return torch.cat([z, res_z], dim=1)
    
    def decode(self, z, *args, **kwargs):
        z, res_z = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res = self.flow.inverse(res_z) / 127.5
        res = F.pixel_shuffle(res, self.downrate)
        xhat = self.vae.decode(z)
        if self.round:
            xhat = torch.round(xhat * 127.5) / 127.5
        x = xhat + res
        return x

    def forward(self, x):
        z = self.encode(x)
        flux_z, res_z = z[:, :self.embed_dim], z[:, self.embed_dim:]
        with torch.no_grad():
            xhat = self.decode(z)
            res = xhat - x
            noise = torch.empty_like(res).uniform_(-0.5, 0.5) / 127.5
            res = res + noise
            res_norm = torch.mean((res * 127.5)**2)
            if self.training:
                flux_z = self.flow_align(flux_z)
            else:
                flux_z = None
        return z, {"xhat": xhat, "flux_z": flux_z,
                   "res_z": res_z, "res_norm": res_norm, "last_layer": self.get_last_layer()}

def window_16x16(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    assert h % 16 == 0 and w % 16 == 0

    patches = F.unfold(x, kernel_size=16, stride=16)
    patches = patches.view(n, c, 16, 16, h // 16, w // 16)
    patches = patches.permute(0, 1, 4, 5, 2, 3)

    return patches

class Flux2VAEDCT(nn.Module):
    def __init__(self, ckpt_path: str, resolution: int = 256, strict: bool = True):
        super().__init__()
        self.vae = Flux2VAE(ckpt_path, resolution, strict)
        self.downrate = 16
        self.embed_dim = 128

    def encode(self, x, *args, **kwargs):
        z = self.vae.encode(x)
        xhat = self.vae.decode(z)
        res = x - xhat
        res_dct = window_16x16(res)
        res_dct = dct.dct_2d(res_dct)
        res_dct = res_dct.reshape(res_dct.shape[0], res_dct.shape[1], res_dct.shape[2], res_dct.shape[3], -1)
        res = F.pixel_unshuffle(res, self.downrate)
        return torch.cat([z, res], dim=1)
    
    def decode(self, z, *args, **kwargs):
        z, res = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res = F.pixel_shuffle(res, self.downrate)
        xhat = self.vae.decode(z)
        x = xhat + res
        return x

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}

import random 

class OrthogonalConv1x1(nn.Module):
    def __init__(self, channels=128, init_identity=True, dim=2):
        super().__init__()
        self.channels = channels
        if init_identity:
            A = torch.zeros(channels, channels)
        else:
            A = 0.01 * torch.randn(channels, channels)
        self.dim = dim
        # Raw parameter; we will skew-symmetrize it in forward
        self.A_raw = nn.Parameter(A)

    def get_weight(self, reverse=False):
        """
        Return W where W^T W = I.
        """
        C = self.channels
        A = self.A_raw - self.A_raw.T  # skew-symmetric
        I = torch.eye(C, device=A.device, dtype=A.dtype)
        # W = (I - A)(I + A)^(-1)
        # use solve instead of explicit inverse:
        # W^T = solve((I + A)^T, (I - A)^T)
        W = torch.linalg.solve(I + A, I - A)
        if reverse:
            W = W.T
        return W

    def forward(self, x, reverse=False):
        """
        x: [B, C, H, W]
        """
        c = x.shape[1]
        assert c == self.channels
        W = self.get_weight(reverse=reverse)
        if self.dim == 2:
            W = W.view(c, c, 1, 1)
            y = F.conv2d(x, W)
        elif self.dim == 1:
            W = W.view(c, c, 1)
            y = F.conv1d(x, W)
        else:
            raise ValueError
        return y

class FLUX2VAEPCA(nn.Module):
    def __init__(self, ckpt_path: str, resolution: int = 256, strict: bool = True, flow_ckpt_path = None, debug=-1):
        super().__init__()
        self.vae = Flux2VAE(ckpt_path, resolution, strict)
        self.embed_dim = 128
        self.flow = OrthogonalConv1x1(self.embed_dim)
        self.mask = [(i+1)*32 for i in range(4)]
        if flow_ckpt_path is not None:
            flow_ckpt = torch.load(flow_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
            miss_keys, unexp_keys = self.load_state_dict(flow_ckpt, strict=False)
            print("miss_keys: ", miss_keys)
            print("unexp_keys: ", unexp_keys)
        self.debug = debug


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


class FLUX2VAEPCASpatial(nn.Module):
    def __init__(self, ckpt_path: str, resolution: int = 256, strict: bool = True, flow_ckpt_path = None, debug=-1):
        super().__init__()
        self.vae = Flux2VAE(ckpt_path, resolution, strict)
        self.spatial_dim = 256
        self.flow = OrthogonalConv1x1(self.spatial_dim, dim=1)
        self.mask = [(i+1)*64 for i in range(4)]
        if flow_ckpt_path is not None:
            flow_ckpt = torch.load(flow_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
            miss_keys, unexp_keys = self.load_state_dict(flow_ckpt, strict=False)
            print("miss_keys: ", miss_keys)
            print("unexp_keys: ", unexp_keys)
        self.debug = debug

    def encode(self, x, *args, **kwargs):
        # b, c, h, w -> b, l, c
        z = self.vae.encode(x)
        b, c, h, w = z.shape
        z = z.reshape(b, c, h*w).permute(0, 2, 1)
        z = self.flow(z)
        if self.debug > 0:
            z[:,self.debug:] = 0
        return z
    
    def decode(self, z, *args, **kwargs):
        if self.debug > 0:
            z[:,self.debug:] = 0
        z = self.flow(z, reverse=True)
        b, l, c = z.shape
        z = z.permute(0, 2, 1).reshape(b, c, int(math.sqrt(l)), int(math.sqrt(l)))
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
        b, c, h, w = z.shape
        z = z.reshape(b, c, h*w).permute(0, 2, 1)
        keep_d = self.mask[keep_id]
        z_original = torch.cat([z[:, :keep_d], torch.zeros_like(z)[:, keep_d:]], dim=1)
        z_flow = self.flow(
                    torch.cat(
                        [self.flow(z)[:, :keep_d], torch.zeros_like(z)[:, keep_d:]],
                    dim=1),
                 reverse=True)

        z_original = z_original.permute(0, 2, 1).reshape(b, c, h, w)
        z_flow = z_flow.permute(0, 2, 1).reshape(b, c, h, w)

        x_original = self.vae.decode(z_original)
        x_flow = self.vae.decode(z_flow)
        
        return x_original, x_flow

import torch
from torchvision import transforms
from PIL import Image

def load_jpeg_to_tensor(image_path):
    """
    Load a JPEG image into a (1, 3, 256, 256) tensor with values in [-1, 1].
    
    Args:
        image_path (str): Path to the JPEG image
        
    Returns:
        torch.Tensor: Shape (1, 3, 256, 256) with values in [-1, 1]
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Maps to [-1, 1]
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    
    return tensor

if __name__ == "__main__":

    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("/mnt/task_runtime/configs/FLUX2PCASpatial.yaml")
    vae = instantiate_from_config(configs)
    x = load_jpeg_to_tensor("/root/data/trove/imagenet-1.0.0/data/raw/validation/n01440764/ILSVRC2012_val_00000293.JPEG")
    z = vae.encode(x)
    xhat = vae.decode(z)
    print(z.shape, xhat.shape, torch.mean((x - xhat) ** 2), torch.mean((torch.round(x*127.5) - torch.round(xhat*127.5)) ** 2))
