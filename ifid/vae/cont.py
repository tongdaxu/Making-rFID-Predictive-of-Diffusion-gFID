import warnings

warnings.filterwarnings("ignore", message=".*timm_vit.*")

from ifid.vae.continous_tokenizer.tokenizer import SoftVQModel, AEModel
import torch
import torch.nn as nn


class SOFTVQ(nn.Module):
    def __init__(self, hf_repo="SoftVQVAE/softvq-l-64", **kwargs):
        super().__init__()
        self.model = SoftVQModel.from_pretrained(hf_repo)

    def encode(self, x):
        z, _, _ = self.model.encode(x)
        return z

    def decode(self, z):
        return self.model.decode(z)


class MAETOK(nn.Module):
    def __init__(self, hf_repo="MAETok/maetok-b-128", **kwargs):
        super().__init__()
        self.model = AEModel.from_pretrained(hf_repo)

    def encode(self, x):
        z, _, _ = self.model.encode(x)
        return z

    def decode(self, z):
        return self.model.decode(z)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/SOFTVQ.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/MAETOK.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
