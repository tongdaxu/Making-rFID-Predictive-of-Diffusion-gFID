import timm
import torch
import torch.nn as nn
from ifid.vae.autoencoder import Decoder

def renorm_dinov3(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    x_01 = (x + 1.0) / 2.0
    x_norm = (x_01 - mean) / std
    return x_norm

def get_dinov3_encoder(feat_indice):
    model = timm.create_model("hf-hub:timm/vit_large_patch16_dinov3.lvd_1689m", pretrained=True, dynamic_img_size=True, features_only=True, 
                              out_indices=feat_indice)
    model.requires_grad_(False)
    return model

class DINOEncoder(nn.Module):
    def __init__(self, feat_indice, z_channels, down_factor):
        super().__init__()
        self.model = get_dinov3_encoder(feat_indice)
        self.feature_dim = 1024 * len(feat_indice) 
        self.conv_out = nn.Conv2d(self.feature_dim, z_channels, kernel_size=1, bias=False)
        self.down_factor=down_factor

    def forward_dinov3(self, x):
        b, c, h, w = x.shape
        x_in = renorm_dinov3(x)
        outputs = self.model(x_in)
        return torch.cat(outputs,dim=1)

    def forward(self, x):
        with torch.no_grad():
            x = self.forward_dinov3(x)
        return self.conv_out(x)
    
class DINOVAE(nn.Module):
    def __init__(self, feat_indice, z_channels, down_factor, ckpt_path=None):
        super().__init__()
        self.encoder = DINOEncoder(feat_indice, z_channels, down_factor)
        self.decoder = Decoder(z_channels=z_channels, resolution=256, out_ch=3, num_res_blocks=2, attn_resolutions=(), ch_mult=[1,2,4,8,8], mid_attn=False)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
            miss_keys, unexp_keys = self.load_state_dict(ckpt, strict=False)
            print("missing: ", miss_keys)
            print("unexp_keys: ", unexp_keys)

    def encode(self, x, *args, **kwargs):
        return self.encoder(x)

    def decode(self, z, *args, **kwargs):
        return self.decoder(z)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config
    configs = OmegaConf.load("/mnt/task_runtime/configs/DINOv3PIT_-1.yaml")
    vae = instantiate_from_config(configs)
    x = torch.randn(2, 3, 256, 256)
    out = vae.encode(x)
    xhat = vae.decode(out)
    print(out.shape)
