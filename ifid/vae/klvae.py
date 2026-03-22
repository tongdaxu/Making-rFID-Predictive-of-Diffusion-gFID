from ifid.vae.autoencoder import AutoencoderKL
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import os


class KLVAE(nn.Module):
    def __init__(self, embed_dim, ch_mult, ckpt_path, *args, **kwargs):
        super().__init__()

        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu")
        self.vae = AutoencoderKL(
            embed_dim=embed_dim,
            ch_mult=ch_mult,
            use_variational=True,
        )
        self.vae.load_state_dict(vae_ckpt, strict=False)

    def encode(self, x, *args, **kwargs):
        return self.vae.encode(x, sample=True)

    def decode(self, z, *args, **kwargs):
        return self.vae.decode(z).sample


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/SDVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/REPAEVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/INVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/VAVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/VAVAE64.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
