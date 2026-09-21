import torch
import torch.nn as nn
from ifid.vae.autoencoder import AutoencoderKL
from huggingface_hub import hf_hub_download
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

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Lightning / LDM checkpoint
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # remove possible wrappers
        clean_state_dict = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in [
                "first_stage_model.",
                "module.",
                "model.",
                "vae.",
            ]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            clean_state_dict[nk] = v

        self.vae = AutoencoderKL(
            embed_dim=embed_dim,
            ch_mult=ch_mult,
            use_variational=True,
        )

        missing, unexpected = self.vae.load_state_dict(clean_state_dict, strict=False)

        print(f"[KLVAE] loaded from: {ckpt_path}")
        print(f"[KLVAE] missing keys: {len(missing)}")
        print(f"[KLVAE] unexpected keys: {len(unexpected)}")

        for k in missing[:30]:
            print("[KLVAE][missing]", k)
        for k in unexpected[:30]:
            print("[KLVAE][unexpected]", k)

        bad_missing = [
            k for k in missing
            if k.startswith("encoder.")
            or k.startswith("decoder.")
            or k.startswith("quant_conv.")
            or k.startswith("post_quant_conv.")
        ]

        if len(bad_missing) > 0:
            raise RuntimeError(
                "KLVAE checkpoint/model mismatch. "
                f"Bad missing keys: {bad_missing[:30]}"
            )

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
