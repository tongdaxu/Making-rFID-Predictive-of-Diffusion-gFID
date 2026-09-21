import os
import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class EQVAE(nn.Module):
    """
    Wrapper for official EQ-VAE LDM-format checkpoint.

    Official EQ-VAE behavior:
        posterior = model.encode(x)
        z = posterior.sample()
        x_rec = model.decode(z)

    This wrapper adapts it to IFID evalvae.py:
        z = vae.encode(x)
        x_rec = vae.decode(z)
    """

    def __init__(
        self,
        config_path,
        ckpt_path,
        eqvae_ldm_path=None,
        sample=True,
        strict=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.sample = sample

        # 官方 repo 里 ldm 包一般在:
        #   /path/to/eqvae/train_eqvae/ldm
        #
        # 所以这里需要把 /path/to/eqvae/train_eqvae 加到 PYTHONPATH。
        if eqvae_ldm_path is not None and eqvae_ldm_path != "":
            eqvae_ldm_path = os.path.abspath(eqvae_ldm_path)
            if eqvae_ldm_path not in sys.path:
                sys.path.insert(0, eqvae_ldm_path)

        from ldm.models.autoencoder import AutoencoderKL

        cfg = OmegaConf.load(config_path)

        # 官方 config 结构是:
        # model:
        #   target: ldm.models.autoencoder.AutoencoderKL
        #   params:
        #     embed_dim: 4
        #     ddconfig: ...
        #     lossconfig: ...
        model_params = OmegaConf.to_container(
            cfg.model.params,
            resolve=True,
        )

        self.vae = AutoencoderKL(**model_params)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        missing, unexpected = self.vae.load_state_dict(ckpt, strict=strict)

        print(f"[EQVAE] loaded official LDM checkpoint: {ckpt_path}")
        print(f"[EQVAE] missing keys: {missing}")
        print(f"[EQVAE] unexpected keys: {unexpected}")

    def encode(self, x, *args, **kwargs):
        posterior = self.vae.encode(x)

        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()

        return z.to(torch.float32)

    def decode(self, z, *args, **kwargs):
        param = next(self.vae.parameters())
        z = z.to(device=param.device, dtype=param.dtype)

        x = self.vae.decode(z)

        if not torch.is_tensor(x):
            raise RuntimeError(f"Unexpected EQ-VAE decode output type: {type(x)}")

        return x