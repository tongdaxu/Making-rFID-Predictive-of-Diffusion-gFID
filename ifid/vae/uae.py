from ifid.vae.unified_ae.uae import UAE
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import os

class UAEVAE(nn.Module):
    def __init__(self, ckpt_path, uae_config):
        super().__init__()
        if not os.path.isfile(ckpt_path):
            a, b, filename = ckpt_path.split("/", 2)
            repo_id = f"{a}/{b}"
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.uae = UAE(**uae_config)

    def encode(self, x):
        return self.uae.encode(x)

    def decode(self, z):
        return self.uae.decode(z)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("/video_ssd/kongzishang/xutongda/git/IFID/configs/UAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
