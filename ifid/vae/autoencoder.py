"""
Ref:
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
"""

from dictdot import dictdot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def _forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

    def forward(self, x, temb):
        return checkpoint(self._forward, x, temb, use_reentrant=False)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        mid_attn=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid_attn = mid_attn
        if self.mid_attn:
            self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        if self.mid_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        mid_attn=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print(
        #     "Working with z of shape {} = {} dimensions.".format(
        #         self.z_shape, np.prod(self.z_shape)
        #     )
        # )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid_attn = mid_attn
        if self.mid_attn:
            self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        if self.mid_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(
            self.mean.shape, device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        embed_dim,
        ch_mult,
        num_res_blocks=2,
        use_variational=True,
        use_pre_post=True,
        mid_attn=True,
    ):
        super().__init__()
        self.encoder = Encoder(
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=embed_dim,
            mid_attn=mid_attn,
        )
        self.decoder = Decoder(
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=embed_dim,
            mid_attn=mid_attn,
        )
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        if use_pre_post:
            self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        else:
            self.quant_conv = torch.nn.Identity()
            self.post_quant_conv = torch.nn.Identity()

    def encode(self, x, sample=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)
        if sample:
            z = posterior.sample()
            return z
        else:
            return posterior

    def decode(self, z):
        # NOTE: We wrap the output in a dict to be consistent with the output
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dictdot(dict(sample=dec))

    def forward(self, x, return_recon=True, return_idem=False):
        posterior = self.encode(x)
        z = posterior.sample()

        recon = None
        if return_recon:
            recon = self.decode(z).sample
        return posterior, z, recon


class AutoencoderPIT(nn.Module):
    def __init__(self, pit_config, use_flow=False):
        super().__init__()
        from pit.util import instantiate_from_config

        config = OmegaConf.load(pit_config)
        self.model = instantiate_from_config(config.model)

    def encode(self, x, sample=False):
        with torch.no_grad():
            z, _ = self.model.encode(x, return_reg_log=True)
        return z

    def decode(self, z):
        # NOTE: We wrap the output in a dict to be consistent with the output
        with torch.no_grad():
            dec = self.model.decode(z)
        return dictdot(dict(sample=dec))

    def forward(self, x, return_recon=True):
        z = self.encode(x)
        posterior = None

        recon = None
        if return_recon:
            recon = self.decode(z).sample
        return posterior, z, recon

    def get_grad_norm(self, loss):
        norm = 0
        for name, param in self.flow.named_parameters():
            if "9." not in name:
                continue
            if "weight" not in name:
                continue
            if "m1" not in name:
                continue
            if "f.4." not in name:
                continue
            grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
            norm += torch.norm(grad)

        return norm


class AutoencoderCont(nn.Module):
    def __init__(self, cont_config, use_ae=False, use_flow=False):
        super().__init__()
        from ifid.vae.continous_tokenizer.tokenizer import SoftVQModel, AEModel

        if use_ae:
            self.model = AEModel.from_pretrained(cont_config)
        else:
            self.model = SoftVQModel.from_pretrained(cont_config)
        self.use_flow = False

    def encode(self, x, sample=False):
        z, _, _ = self.model.encode(x)
        return z

    def decode(self, z):
        rec = self.model.decode(z)
        return dictdot(dict(sample=rec))

    def forward(self, x, return_recon=True):
        z = self.encode(x)
        recon = self.decode(z).sample
        return None, z, recon


# Predefined VAE architectures
def VAE_F8D4(**kwargs):
    # [B, 4, 32, 32]
    return AutoencoderKL(
        embed_dim=4, ch_mult=[1, 2, 4, 4], use_variational=True, **kwargs
    )


def VAE_F8D4Flow(**kwargs):
    # [B, 4, 32, 32]
    return AutoencoderKL(
        embed_dim=4, ch_mult=[1, 2, 4, 4], use_variational=True, use_flow=True, **kwargs
    )


def VAE_F8D16(**kwargs):
    # [B, 16, 32, 32]
    return AutoencoderKL(
        embed_dim=16,
        ch_mult=[1, 4, 8, 8],
        num_res_blocks=3,
        use_variational=True,
        use_pre_post=False,
        mid_attn=False,
        **kwargs,
    )


def VAE_F8D16Flow(**kwargs):
    # [B, 16, 32, 32]
    return AutoencoderKL(
        embed_dim=16,
        ch_mult=[1, 4, 8, 8],
        num_res_blocks=3,
        use_variational=True,
        use_flow=True,
        use_pre_post=False,
        mid_attn=False,
        **kwargs,
    )


def VAE_F16D32(**kwargs):
    # [B, 32, 16, 16] (used in VA-VAE and our model)
    return AutoencoderKL(
        embed_dim=32, ch_mult=[1, 1, 2, 2, 4], use_variational=True, **kwargs
    )


def VAE_F16D64(**kwargs):
    # [B, 32, 16, 16] (used in VA-VAE and our model)
    return AutoencoderKL(
        embed_dim=64, ch_mult=[1, 1, 2, 2, 4], use_variational=True, **kwargs
    )


def VAE_F16D32Flow(**kwargs):
    # [B, 32, 16, 16] (used in VA-VAE and our model)
    return AutoencoderKL(
        embed_dim=32,
        ch_mult=[1, 1, 2, 2, 4],
        use_variational=True,
        use_flow=True,
        **kwargs,
    )


def VAE_FLUX(**kwargs):
    return AutoencoderPIT(
        "/video_ssd/kongzishang/xutongda/git/REPA-E/configs/FLUX.yaml"
    )


def VAE_FLUXFlow(**kwargs):
    return AutoencoderPIT(
        "/workspace/cogview_dev/xutd/xu/pytorch-image-tokenizer/configs/FLUX.yaml",
        use_flow=True,
    )


def VAE_SD3(**kwargs):
    return AutoencoderPIT("/video_ssd/kongzishang/xutongda/git/REPA-E/configs/SD3.yaml")


def VAE_QW(**kwargs):
    return AutoencoderPIT("/video_ssd/kongzishang/xutongda/git/REPA-E/configs/QW.yaml")


def VAE_DETOK(**kwargs):
    return AutoencoderPIT(
        "/video_ssd/kongzishang/xutongda/git/REPA-E/configs/DETOK.yaml"
    )


def VAE_SD3Flow(**kwargs):
    return AutoencoderPIT(
        "/workspace/cogview_dev/xutd/xu/pytorch-image-tokenizer/configs/SD3.yaml",
        use_flow=True,
    )


def VAE_WAN(**kwargs):
    return AutoencoderPIT(
        "/workspace/cogview_dev/xutd/xu/pytorch-image-tokenizer/configs/Wan2.2-I2V-A14B-Diffusers.yaml"
    )


def VAE_WANFlow(**kwargs):
    return AutoencoderPIT(
        "/workspace/cogview_dev/xutd/xu/pytorch-image-tokenizer/configs/Wan2.2-I2V-A14B-Diffusers.yaml",
        use_flow=True,
    )


def VAE_DMVAE(**kwargs):
    return AutoencoderPIT(
        "/video_ssd/kongzishang/xutongda/git/REPA-E/configs/DMVAE.yaml"
    )


def VAE_RAE(**kwargs):
    return AutoencoderPIT("/video_ssd/kongzishang/xutongda/git/REPA-E/configs/RAE.yaml")


def VAE_SoftVQ(**kwargs):
    return AutoencoderCont("/share/xutongda/hfhome/hub/softvq-l-64")


def VAE_MAETOK(**kwargs):
    return AutoencoderCont("/share/xutongda/hfhome/hub/maetok-b-128", use_ae=True)


vae_models = {
    "f8d4": VAE_F8D4,  # [B, 4, 32, 32]
    "f8d4flow": VAE_F8D4Flow,  # [B, 4, 32, 32]
    "f8d16": VAE_F8D16,  # [B, 4, 32, 32]
    "f8d16flow": VAE_F8D16Flow,  # [B, 4, 32, 32]
    "f16d32": VAE_F16D32,  # [B, 32, 16, 16]
    "f16d64": VAE_F16D64,  # [B, 32, 16, 16]
    "f16d32flow": VAE_F16D32Flow,  # [B, 32, 16, 16]
    "flux": VAE_FLUX,  # [B, 32, 16, 16]
    "fluxflow": VAE_FLUXFlow,  # [B, 32, 16, 16]
    "sd3": VAE_SD3,  # [B, 32, 16, 16]
    "qw": VAE_QW,  # [B, 32, 16, 16]
    "sd3flow": VAE_SD3Flow,  # [B, 32, 16, 16]
    "wan": VAE_WAN,  # [B, 32, 16, 16]
    "wanflow": VAE_WANFlow,  # [B, 32, 16, 16]
    "softvq": VAE_SoftVQ,
    "maetok": VAE_MAETOK,
    "rae": VAE_RAE,  # [B, 768, 16, 16]
    "dmvae": VAE_DMVAE,  # [B, 256, 32]
    "detok": VAE_DETOK,  # [B, 256, 16]
}

if __name__ == "__main__":
    model = VAE_F8D4()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
