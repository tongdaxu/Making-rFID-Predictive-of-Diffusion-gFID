from ifid.vae.autoencoder import AutoencoderKL
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class KLVAE(nn.Module):
    def __init__(self, embed_dim, ch_mult, ckpt_path, attn_resolutions=(16,), mid_attn=True, use_pre_post=False, *args, **kwargs):
        super().__init__()

        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in vae_ckpt:
            vae_ckpt = vae_ckpt["state_dict"]
        if "vae" in vae_ckpt:
            vae_ckpt = vae_ckpt["vae"]
            new_vae_ckpt = {}
            for name in vae_ckpt.keys():
                if "vae." in name:
                    new_name = name.removeprefix("vae.")
                    new_vae_ckpt[new_name] = vae_ckpt[name]
            vae_ckpt = new_vae_ckpt
        self.vae = AutoencoderKL(
            embed_dim=embed_dim,
            ch_mult=ch_mult,
            use_variational=True,
            mid_attn=mid_attn,
            use_pre_post=use_pre_post,
            attn_resolutions=attn_resolutions,
        )
        miss_keys, unexp_keys = self.vae.load_state_dict(vae_ckpt, strict=False)
        print(f"Missing keys: {miss_keys}")
        print(f"Unexpected keys: {unexp_keys}")

    def encode(self, x, *args, **kwargs):
        return self.vae.encode(x, sample=True)

    def decode(self, z, *args, **kwargs):
        return self.vae.decode(z).sample

    def forward(self, x: torch.Tensor, z = None) -> torch.Tensor:
        if z is None:
            posterior = self.vae.encode(x, sample=False)
            z = posterior.sample()
        else:
            posterior = None
        x_rec = self.decode(z)
        return z, {"xhat": x_rec, "posterior": posterior}

class KLVAERes(nn.Module):
    def __init__(self, embed_dim, ch_mult, ckpt_path, *args, **kwargs):
        super().__init__()

        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in vae_ckpt:
            vae_ckpt = vae_ckpt["state_dict"]
        self.vae = AutoencoderKL(
            embed_dim=embed_dim,
            ch_mult=ch_mult,
            use_variational=True,
        )
        self.embed_dim = embed_dim
        self.downrate = 2 ** (len(ch_mult) - 1)
        miss_keys, unexp_keys = self.vae.load_state_dict(vae_ckpt, strict=False)
        print(f"Missing keys: {miss_keys}")
        print(f"Unexpected keys: {unexp_keys}")

    def encode(self, x, *args, **kwargs):
        z = self.vae.encode(x, sample=True)
        xhat = self.vae.decode(z).sample
        res = x - xhat
        res = F.pixel_unshuffle(res, self.downrate)
        return torch.cat([z, res], dim=1)

    def decode(self, z, *args, **kwargs):
        z, res = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res = F.pixel_shuffle(res, self.downrate)
        xhat = self.vae.decode(z).sample
        x = xhat + res
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, {"xhat": x_rec}


class KLVAEResDebug(nn.Module):
    def __init__(self, embed_dim, ch_mult, ckpt_path, *args, **kwargs):
        super().__init__()

        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in vae_ckpt:
            vae_ckpt = vae_ckpt["state_dict"]
        self.vae = AutoencoderKL(
            embed_dim=embed_dim,
            ch_mult=ch_mult,
            use_variational=True,
        )
        self.embed_dim = embed_dim
        self.downrate = 2 ** (len(ch_mult) - 1)
        miss_keys, unexp_keys = self.vae.load_state_dict(vae_ckpt, strict=False)
        print(f"Missing keys: {miss_keys}")
        print(f"Unexpected keys: {unexp_keys}")

    def encode(self, x, *args, **kwargs):
        z = self.vae.encode(x, sample=True)
        xhat = self.vae.decode(z).sample
        res = x - xhat
        res = F.pixel_unshuffle(res, self.downrate)
        res = torch.zeros_like(res)  # debug purpose only
        return torch.cat([z, res], dim=1)

    def decode(self, z, *args, **kwargs):
        z, res = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res = F.pixel_shuffle(res, self.downrate)
        xhat = self.vae.decode(z).sample
        x = xhat
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, {"xhat": x_rec}


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config
    x = torch.randn([1, 3, 256, 256])
    configs = OmegaConf.load("../../configs/VAVAERes.yaml")
    vae = instantiate_from_config(configs)
    z = vae.encode(x)
    xhat = vae.decode(z)
    print(z.shape, xhat.shape, torch.mean((x - xhat) ** 2))
