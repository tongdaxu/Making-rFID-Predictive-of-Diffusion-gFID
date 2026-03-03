import os
import torch
from torch import nn
from contextlib import nullcontext
from timm.models import create_model

from dataclasses import dataclass

from einops import rearrange
from torch import Tensor

from huggingface_hub import hf_hub_download


def init_weights(
    model: nn.Module, conv_std_or_gain: float = 0.02, other_std: float = 0.02
):
    """
    :param model: the model to be inited
    :param conv_std_or_gain: how to init every conv layer `m`
        > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
        < 0: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
    :param other_std: how to init every linear layer or embedding layer
        use nn.init.trunc_normal_(m.weight.data, std=other_std)
    """
    skip = abs(conv_std_or_gain) > 10
    if skip:
        return
    print(
        f"[init_weights] {type(model).__name__} with {'std' if conv_std_or_gain > 0 else 'gain'}={abs(conv_std_or_gain):g}"
    )
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=other_std)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=other_std)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(
            m,
            (
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            ),
        ):
            nn.init.trunc_normal_(
                m.weight.data, std=conv_std_or_gain
            ) if conv_std_or_gain > 0 else nn.init.xavier_normal_(
                m.weight.data, gain=-conv_std_or_gain
            )  # todo: StyleSwin: (..., gain=.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(
            m,
            (
                nn.LayerNorm,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.SyncBatchNorm,
                nn.GroupNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
            ),
        ):
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, z_channels * 2, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.block_in = block_in
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor, grad_ckpt=False) -> Tensor:
        # get dtype for proper tracing
        # upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        if z.ndim == 3:
            z = rearrange(z, "b (h w) c -> b c h w", h=16, w=16).contiguous()

        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

    def post_init(self, z_channels):
        self.conv_in = nn.Sequential(
            Upsample(z_channels),
            nn.Conv2d(z_channels, self.block_in, kernel_size=3, stride=1, padding=1),
        )

    def get_last_layer(
        self,
    ):
        return self.conv_out.weight


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


class Denormalize(nn.Module):
    def __init__(
        self,
        mean,
        std,
    ):
        super(Denormalize, self).__init__()
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return x * self.std + self.mean


class DINOEncoder(nn.Module):
    def __init__(
        self,
        model_size="base",
        patch_size=16,
        image_size=256,
    ):
        super().__init__()
        self.dim = {
            "base": 768,
            "large": 1024,
        }[model_size]
        self.de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if model_size == "base":
            self.model = create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=True,
                patch_size=patch_size,
                img_size=image_size,
            )
        elif model_size == "large":
            self.model = create_model(
                "vit_large_patch14_dinov2.lvd142m",
                pretrained=True,
                patch_size=patch_size,
                img_size=image_size,
            )

    def forward(self, x):
        return self.model.forward_features(self.scale(self.de_scale(x)))[
            :, self.model.num_prefix_tokens :
        ]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)

    def get_last_layer(self):
        return self.mlp[-1].weight


class DMVAE(nn.Module):
    def __init__(
        self,
        z_channels: int = 16,
        image_size: int = 256,
        model_size: str = "base",
        patch_size: int = 16,
        conv_std_or_gain: float = 0.02,
    ):
        super().__init__()
        self.encoder = DINOEncoder(model_size, patch_size=patch_size)
        self.decoder = Decoder(
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            in_channels=3,
            resolution=256,
            z_channels=16,
        )
        self.decoder.post_init(z_channels=z_channels)
        self.bottle_neck = MLP(in_dim=self.encoder.dim, out_dim=z_channels)

        init_weights(self.decoder.conv_in, conv_std_or_gain)
        init_weights(self.bottle_neck, conv_std_or_gain)
        init_weights(self.decoder, conv_std_or_gain)

    def forward(self, x, freeze_encoder=False, return_latent=False):
        ctx = torch.no_grad() if freeze_encoder else nullcontext()
        with ctx:
            latent_tokens = self.encoder(x)
        latent_tokens = self.bottle_neck(latent_tokens)
        x_rec = self.decoder(latent_tokens)
        if return_latent:
            return x_rec.float(), latent_tokens
        return x_rec.float()

    @torch.inference_mode()
    def encode(self, x):
        latent_tokens = self.encoder(x)
        latent_tokens = self.bottle_neck(latent_tokens)
        return latent_tokens

    @torch.inference_mode()
    def decode(self, latent_tokens):
        return self.decoder(latent_tokens)

    def load_pretrained(self, state_dict_path, ema=False):
        if not os.path.exists(state_dict_path):
            print(
                f"[WARNING] VAE state_dict_path {state_dict_path} not found, skip loading"
            )
            return
        try:
            ckpt = torch.load(state_dict_path, map_location="cpu")
        except:
            ckpt = torch.load(state_dict_path, map_location="cpu", weights_only=False)
        if ema and "vae_ema" in ckpt:
            self.load_state_dict(ckpt["vae_ema"], strict=True)
        else:
            self.load_state_dict(ckpt["vae_wo_ddp"], strict=True)


class DMVAEEXPORT(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()

        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )

        self.model = DMVAE(z_channels=32, model_size="large")
        self.model.load_pretrained(ckpt_path)

    def encode(
        self,
        x,
        return_reg_log=False,
        unregularized=False,
    ):
        z = self.model.encode(x)
        return z

    def decode(self, z):
        xhat = self.model.decode(z)
        return xhat


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/DMVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
