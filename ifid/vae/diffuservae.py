import torch
import torch.nn as nn


class DIFFUSERVAE(nn.Module):
    def __init__(self, hf_repo):
        super().__init__()
        from diffusers import AutoencoderKL

        self.model = AutoencoderKL.from_pretrained(hf_repo, subfolder="vae")

    def encode(
        self,
        x,
        return_reg_log=False,
        unregularized=False,
    ):
        qzx = self.model.encode(x, return_dict=False)[0]
        z = qzx.sample()
        return z

    def decode(self, z):
        xhat = self.model.decode(z, return_dict=False)[0]
        return xhat


class QWVAE(nn.Module):
    def __init__(self, hf_repo="Qwen/Qwen-Image", **kwargs):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage

        self.model = AutoencoderKLQwenImage.from_pretrained(hf_repo, subfolder="vae")

    def encode(
        self,
        x,
        return_reg_log=False,
        unregularized=False,
    ):
        qzx = self.model.encode(x[:, :, None], return_dict=False)
        z = qzx[0].sample()[:, :, 0]
        return z

    def decode(self, z):
        xhat = self.model.decode(z[:, :, None], return_dict=False)[0][:, :, 0]
        return xhat


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/QWVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/SD3VAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/FLUXVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
