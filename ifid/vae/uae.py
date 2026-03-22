import torch
import torch.nn as nn
from ifid.vae.utils import instantiate_from_config
from omegaconf import OmegaConf
import os
from huggingface_hub import hf_hub_download


class UAE(nn.Module):
    def __init__(self, uae_config, ckpt_path, *args, **kwargs):
        super().__init__()
        self.uae = instantiate_from_config(uae_config)
        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.uae.load_state_dict(state_dict["model"])

    def encode(
        self,
        x,
        *args, **kwargs
    ):

        freq_ratio_tensor = torch.full(
            (x.size(0),),
            1.0,
            device=x.device,
            dtype=x.dtype,
        )

        # x -> [0, 1]
        x = (x + 1.0) / 2.0
        z = self.uae.encode(
            x,
            apply_frequency=True,
            freq_ratio=freq_ratio_tensor,
        )
        return z

    def decode(self, z, *args, **kwargs):
        xhat = self.uae.decode(z)
        xhat = xhat.clamp(0.0, 1.0)
        xhat = xhat * 2.0 - 1.0
        return xhat


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/UAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
