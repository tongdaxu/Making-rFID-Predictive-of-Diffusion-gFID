import torch
import torch.nn as nn
import os
import json

from huggingface_hub import snapshot_download

from ifid.vae.vtp_module.vtp_hf import VTPConfig, VTPModel
from safetensors.torch import load

class VTPVAE(nn.Module):
    def __init__(self, ckpt_folder):
        super().__init__()
        if not os.path.exists(ckpt_folder):
            ckpt_folder = snapshot_download(
                repo_id=ckpt_folder,
            )
        json_path = os.path.join(ckpt_folder, "config.json")
        ckpt_path = os.path.join(ckpt_folder, "model.safetensors")

        with open(json_path) as f:
            vtp_cfg_kwargs = json.load(f)

        vtp_cfg = VTPConfig(**vtp_cfg_kwargs)
        self.vtp = VTPModel(vtp_cfg)

        with open(ckpt_path, "rb") as f:
            ckpt_data = f.read()

        unexp, miss = self.vtp.load_state_dict(load(ckpt_data), strict=False)
        print("unexpected keys", unexp)
        print("missing keys", miss)

    def encode(self, x):
        return self.vtp.get_reconstruction_latents(x)

    def decode(self, z):
        return self.vtp.get_latents_decoded_images(z)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return x, {"xhat": xhat}

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("/video_ssd/kongzishang/xutongda/git/IFID/configs/VTPB.yaml")
    vae = instantiate_from_config(configs)
    z = vae.encode(torch.randn([1, 3, 256, 256]))
    xhat = vae.decode(z)
    print(z.shape, xhat.shape)
