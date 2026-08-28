import torch
import torch.nn as nn
import os
import json
import numpy as np

from huggingface_hub import snapshot_download
from safetensors.torch import load
from ifid.vae.autoencoder import Encoder, Decoder, DiagonalGaussianDistribution
from ifid.vae.vtp_module.vtp_hf import VTPConfig, VTPModel
import torch.nn.functional as F

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

class VTPResVAE(nn.Module):
    def __init__(self, vtp_ckpt, use_mse_decoder=False, ckpt_path=None):
        super().__init__()
        self.foundation_model = VTPVAE(vtp_ckpt)
        self.vf_feature_dim = 64
        self.encoder = Encoder(
            z_channels=self.vf_feature_dim,
            double_z=True,
            ch_mult=[1,2,4,8,8],
            num_res_blocks=2,
            attn_resolutions=[],
            mid_attn=False,
        )
        self.decoder = Decoder(
            z_channels=self.vf_feature_dim,
            ch_mult=[1,2,4,8,8],
            num_res_blocks=2,
            attn_resolutions=[],
            mid_attn=False,
        )
        self.use_mse_decoder = use_mse_decoder
        if self.use_mse_decoder:
            self.decoder_vtp = Decoder(
                z_channels=self.vf_feature_dim,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[1, 2, 4, 8, 8],
                num_res_blocks=2,
                attn_resolutions=[],
                mid_attn=False,
            )
        if ckpt_path is not None:
            miss_keys, unexp_keys = self.load_state_dict(torch.load(ckpt_path, map_location='cpu')["state_dict"], strict=False)
            print("missing: ", miss_keys)
            print("unexp_keys: ", unexp_keys)

    def encode(self, x):
        aux_z = self.foundation_model.encode(x)
        if self.use_mse_decoder:
            aux_x = self.decoder_vtp(aux_z)
        else:
            aux_x = self.foundation_model.decode(aux_z)
        res_x = x - aux_x
        res_z = DiagonalGaussianDistribution(self.encoder(res_x)).sample()
        return torch.cat([aux_z, res_z], dim=1)

    def decode(self, z):
        aux_z, res_z = z[:, :self.vf_feature_dim], z[:, self.vf_feature_dim:]
        res_x_hat = self.decoder(res_z)
        if self.use_mse_decoder:
            aux_x = self.decoder_vtp(aux_z)
        else:
            aux_x = self.foundation_model.decode(aux_z)
        return aux_x + res_x_hat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return x, {"xhat": xhat}

class VTPRes(nn.Module):
    def __init__(self, vtp_ckpt):
        super().__init__()
        self.foundation_model = VTPVAE(vtp_ckpt)
        self.downrate = 16
        self.embed_dim = 64

    def encode(self, x):
        aux_z = self.foundation_model.encode(x)
        if self.use_mse_decoder:
            aux_x = self.decoder_vtp(aux_z)
        else:
            aux_x = self.foundation_model.decode(aux_z)
        res_x = x - aux_x
        res_z = F.pixel_unshuffle(res_x, self.downrate)
        return torch.cat([aux_z, res_z], dim=1)

    def decode(self, z):
        aux_z, res_z = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res_x = F.pixel_shuffle(res_z, self.downrate)
        aux_x = self.foundation_model.decode(aux_z)
        x = aux_x + res_x
        return x

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return x, {"xhat": xhat}


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config
    from ifid.fid.psnr import get_psnr
    from PIL import Image

    def read_image_tensor(path):
        """
        Read image and return tensor of shape (1,3,256,256) in range [-1,1]
        """
        img = Image.open(path).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.array(img)).float() / 255.0  # [0,1]
        x = x.permute(2, 0, 1)                               # (3,H,W)
        x = x * 2 - 1                                         # [-1,1]
        x = x.unsqueeze(0)                                    # (1,3,256,256)
        return x

    configs = OmegaConf.load("/video_ssd/kongzishang/xutongda/git/IFID/configs/VTPB.yaml")
    vae = instantiate_from_config(configs).cuda()
    x = read_image_tensor("/video_ssd/kongzishang/xutongda/git/IFID/samples/ifid-rae_extra/src_dist/ILSVRC2012_val_00000236.JPEG").cuda()
    z = vae.encode(x)
    xhat = vae.decode(z)
    psnr = get_psnr(x, xhat, zero_mean=True)
    print(z.shape, psnr)
