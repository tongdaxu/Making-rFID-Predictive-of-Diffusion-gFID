import torch
import torch.nn as nn

from ifid.vae.continous_tokenizer.tokenizer import FlowModel, ModelArgs
from ifid.vae.continous_tokenizer.tokenizer import SoftVQModel, AEModel

import warnings
warnings.filterwarnings("ignore", message=".*timm_vit.*")

class ContinuousTokenizerVAE(nn.Module):
    def __init__(self, ckpt_file, *args, **kwargs):
        super().__init__()

        self.supports_y = False
        self.num_classes = 0

        modelargs = ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4],
            decoder_ch_mult=[1, 1, 2, 2, 4],
            num_latent_tokens=256,
            codebook_embed_dim=32,
            entropy_loss_ratio=0.0,
            enc_type="flow",
            encoder_model="vit_base_patch14_dinov2.lvd142m",
            dec_type="vit",
            decoder_model="vit_tinytiny_patch14_dinov2_movq2",
            use_ape=False,
            use_rope=True,
            rope_mixed=True,
            rope_theta=10.0,
            enc_token_drop=0.4,
            enc_token_drop_max=0.6,
            aux_loss_mask=True,
            aux_hog_dec=False,
            aux_dino_dec=False,
            aux_clip_dec=False,
            repa=True,
            repa_model="vit_large_patch14_dinov2.lvd142m",
            repa_patch_size=16,
            repa_proj_dim=1024,
            repa_loss_weight=0.1,
            repa_align="repeat",
        )

        self.model = FlowModel(config=modelargs)

        flow_ckpt = torch.load(ckpt_file, map_location="cpu")["model"]
        missing, unexpected = self.model.load_state_dict(flow_ckpt, strict=False)
        print("missing", missing)
        print("unexp", unexpected)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode(self, x, *args, **kwargs):
        return self.model.encode(x)

    def decode(self, z, *args, **kwargs):
        return self.model.decode(z)


class SOFTVQ(nn.Module):
    def __init__(self, hf_repo="SoftVQVAE/softvq-l-64", *args, **kwargs):
        super().__init__()
        self.model = SoftVQModel.from_pretrained(hf_repo)

    def encode(self, x, *args, **kwargs):
        z, _, _ = self.model.encode(x)
        return z

    def decode(self, z, *args, **kwargs):
        return self.model.decode(z)


class MAETOK(nn.Module):
    def __init__(self, hf_repo="MAETok/maetok-b-128", *args, **kwargs):
        super().__init__()
        self.model = AEModel.from_pretrained(hf_repo)

    def encode(self, x, *args, **kwargs):
        z, _, _ = self.model.encode(x)
        return z

    def decode(self, z, *args, **kwargs):
        return self.model.decode(z)


class FLOWEXPORT(nn.Module):
    def __init__(self, hf_repo="MAETok/maetok-b-128", *args, **kwargs):
        super().__init__()
        self.model = AEModel.from_pretrained(hf_repo)

    def encode(self, x, *args, **kwargs):
        z, _, _ = self.model.encode(x)
        return z

    def decode(self, z, *args, **kwargs):
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
