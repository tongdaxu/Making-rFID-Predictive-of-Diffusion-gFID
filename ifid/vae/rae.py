import torch
import torch.nn as nn
import torch.nn.functional as F

from ifid.vae.rae_module.rae import instantiate_rae_from_config, parse_rae_configs


class RAE(nn.Module):
    def __init__(self, rae_config, *args, **kwargs):
        super().__init__()
        rae_config, *_ = parse_rae_configs(rae_config)
        self.rae = instantiate_rae_from_config(rae_config)

    def encode(
        self,
        x,
        *args, **kwargs
    ):
        # x -> [0, 1]
        x = (x + 1.0) / 2.0
        z = self.rae.encode(x)
        return z

    def decode(self, z, *args, **kwargs):
        xhat = self.rae.decode(z)
        xhat = xhat * 2.0 - 1.0
        return xhat
    
    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}


class RAERes(nn.Module):
    def __init__(self, rae_config, *args, **kwargs):
        super().__init__()
        rae_config, *_ = parse_rae_configs(rae_config)
        self.rae = instantiate_rae_from_config(rae_config)
        self.downrate = 16
        self.embed_dim = 768

    def encode(
        self,
        x,
        *args, **kwargs
    ):
        # x -> [0, 1]
        x_in = (x + 1.0) / 2.0
        z = self.rae.encode(x_in)
        xhat = self.rae.decode(z)
        xhat = xhat * 2.0 - 1.0

        res = x - xhat
        res = F.pixel_unshuffle(res, self.downrate)
        return torch.cat([z, res], dim=1)

    def decode(self, z, *args, **kwargs):
        z, res = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res = F.pixel_shuffle(res, self.downrate)

        xhat = self.rae.decode(z)
        xhat = xhat * 2.0 - 1.0

        x = xhat + res
        return x

class RAEResDebug(nn.Module):
    def __init__(self, rae_config, *args, **kwargs):
        super().__init__()
        rae_config, *_ = parse_rae_configs(rae_config)
        self.rae = instantiate_rae_from_config(rae_config)
        self.downrate = 16
        self.embed_dim = 768

    def encode(
        self,
        x,
        *args, **kwargs
    ):
        # x -> [0, 1]
        x_in = (x + 1.0) / 2.0
        z = self.rae.encode(x_in)
        xhat = self.rae.decode(z)
        xhat = xhat * 2.0 - 1.0

        res = x - xhat
        res = F.pixel_unshuffle(res, self.downrate)
        res = torch.zeros_like(res)  # debug purpose only
        return torch.cat([z, res], dim=1)

    def decode(self, z, *args, **kwargs):
        z, res = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res = F.pixel_shuffle(res, self.downrate)

        xhat = self.rae.decode(z)
        xhat = xhat * 2.0 - 1.0

        x = xhat
        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("/mnt/task_runtime/configs/RAERes.yaml")
    x = torch.randn([1, 3, 256, 256])
    vae = instantiate_from_config(configs)
    z = vae.encode(x)
    xhat = vae.decode(z)
    print(z.shape, xhat.shape, torch.mean((x - xhat) ** 2))
