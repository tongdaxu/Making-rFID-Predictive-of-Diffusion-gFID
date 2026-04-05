import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

import INN
from INN.CouplingModels.NICEModel.conv import Conv2dNICE
from ifid.vae.autoencoder import Decoder

class FlowVAEBKB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.flow = INN.Sequential(
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            INN.PixelShuffle2d(2), # 8
            Conv2dNICE(12, 3),
            Conv2dNICE(12, 3),
            INN.PixelShuffle2d(2), # 4
            Conv2dNICE(48, 3),
            Conv2dNICE(48, 3),
            INN.PixelShuffle2d(2), # 2
            Conv2dNICE(192, 3),
            Conv2dNICE(192, 3),
            INN.PixelShuffle2d(2), # 1
            Conv2dNICE(768, 3, w=1),
            Conv2dNICE(768, 3, w=1),
        )
        self.flow.computing_p(True)
        self.flow_lam = 1

def unnormalize(tensor, mean, std):
    """
    tensor: (C, H, W) or (B, C, H, W)
    mean, std: list or tuple of length C
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)

    if tensor.dim() == 4:  # batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean

class DINOFlowVAE(nn.Module):
    def __init__(self, ckpt_path, *args, **kwargs):
        super().__init__()
        self.backbone = FlowVAEBKB()
        self.decoder = Decoder(out_ch=3, ch_mult=[1, 1, 2, 4, 4], num_res_blocks=2, z_channels=768, mid_attn=False)
        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu")
        if "teacher" in vae_ckpt.keys():
            vae_ckpt = vae_ckpt["teacher"]
        elif "vae" in vae_ckpt.keys():
            vae_ckpt = vae_ckpt["vae"]
        else:
            assert(0)
        miss_keys, unexp_keys = self.load_state_dict(vae_ckpt, strict=False)
        print("missing: ", miss_keys)
        print("unexp_keys: ", unexp_keys)

    def encode(self, x, *args, **kwargs):
        # use dino
        x = (x + 1.0) / 2.0
        x = torchvision.transforms.functional.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        z = self.backbone.flow(x)[0]
        return z

    def decode(self, z, inv=False, *args, **kwargs):
        x = self.decoder(z)
        with torch.no_grad():
            xinv = self.backbone.flow.inverse(z)
            xinv = unnormalize(xinv, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            xinv = xinv * 2.0 - 1.0
        if inv:
            return xinv
        else:
            return x

    def forward(self, x, *args, **kwargs):
        with torch.no_grad():
            z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/DINOFLOWVAE.yaml")
    vae = instantiate_from_config(configs)
    z = vae.encode(torch.randn([1, 3, 256, 256]))
    xhat = vae.decode(z)
    print(z.shape, xhat.shape)

