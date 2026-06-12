import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from ifid.unet.openaimodel import UNetModel

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        # this is for use in crossattn
        c = batch[:, None]
        c = self.embedding(c)
        return c

class UNet(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        path_type="edm",
        input_size=32,
        patch_size=1,
        in_channels=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        bn_momentum=0.1,
        tshift=1.0,
        **block_kwargs,  # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        self.bn = torch.nn.BatchNorm2d(
            in_channels,
            eps=1e-4,
            momentum=bn_momentum,
            affine=False,
            track_running_stats=True,
        )

        self.tshift = tshift
        self.bn.reset_running_stats()
        self.pre_unet = nn.PixelUnshuffle(self.patch_size)

        self.class_embedder = ClassEmbedder(
            512, n_classes=self.num_classes+1,
        )
        self.unet_model = UNetModel(
            image_size=(input_size//self.patch_size),
            in_channels=in_channels*(self.patch_size**2),
            out_channels=in_channels*(self.patch_size**2),
            model_channels=256,
            attention_resolutions=(4,2,1),
            num_res_blocks=2,
            channel_mult=(1,2,4),
            num_head_channels=32,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=512,
            # num_classes=self.num_classes,
        )
        self.post_unet = nn.PixelShuffle(self.patch_size)

    def shift_time(self, t):
        shifted_t = self.tshift * t / (1 + (self.tshift - 1) * t)
        return shifted_t

    def init_bn(self, latents_scale, latents_bias):
        # latents_scale = 1 / sqrt(variance); latents_bias = mean
        self.bn.running_mean = latents_bias
        self.bn.running_var = (1.0 / latents_scale).pow(2)

    def extract_latents_stats(self):
        # rsqrt is the reciprocal of the square root
        latent_stats = dict(
            latents_scale=self.bn.running_var.rsqrt(),
            latents_bias=self.bn.running_mean,
        )
        return latent_stats

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def interpolant(self, t, path_type=None):
        if path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0], inputs[1])
            return inputs

        return custom_forward

    def forward(
        self,
        x,
        y,
        loss_kwargs,
        time_input=None,
        noises=None,
    ):
        """
        Forward pass of SiT, integrating the loss function computation
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images UNNORMALIZED)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        loss_kwargs: dictionary of loss function arguments, should contain: `weighting`, `path_type`, `prediction`,
        time_input: optionally provide a tensor of timesteps to use for the forward pass, otherwise sample from a distribution
        noises: optionally provide a tensor of noises to use for the forward pass, otherwise sample from a distribution
        """
        # Normalize the input x with batch norm running stats
        normalized_x = self.bn(x)

        # sample timesteps if not provided
        if time_input is None:
            if loss_kwargs["weighting"] == "uniform":
                time_input = torch.rand((normalized_x.shape[0], 1, 1, 1))
            elif loss_kwargs["weighting"] == "lognormal":
                # sample timestep according to log-normal distribution of sigmas following EDM
                rnd_normal = torch.rand((normalized_x.shape[0], 1, 1, 1))
                sigma = rnd_normal.exp()
                if loss_kwargs["path_type"] == "linear":
                    time_input = sigma / (1 + sigma)
                elif loss_kwargs["path_type"] == "cosine":
                    time_input = 2 / np.pi * torch.atan(sigma)
            else:
                raise NotImplementedError(
                    f"Weighting scheme {loss_kwargs['weighting']} not implemented."
                )
        time_input = time_input.to(device=normalized_x.device, dtype=normalized_x.dtype)

        time_input = self.shift_time(time_input)

        # sample noises if not provided
        if noises is None:
            noises = torch.randn_like(normalized_x)
        else:
            noises = noises.to(device=normalized_x.device, dtype=normalized_x.dtype)

        # compute interpolant
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(
            time_input, path_type=loss_kwargs["path_type"]
        )

        model_input = alpha_t * normalized_x + sigma_t * noises

        if loss_kwargs["prediction"] == "v":
            model_target = d_alpha_t * normalized_x + d_sigma_t * noises
        else:
            raise NotImplementedError()  # TODO: add x or eps prediction

        # label dropout
        drop_mask = torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob
        y[drop_mask] = self.num_classes

        # unet forward
        model_output = self.post_unet(self.unet_model(x=self.pre_unet(model_input), timesteps=time_input[:,0,0,0], context=self.class_embedder(y)))

        # loss computation
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        ret_dict = {
            "model_output": model_output,
            "denoising_loss": denoising_loss,
            "time_input": time_input,
            "noises": noises,
        }

        return ret_dict

    @torch.no_grad()
    def inference(self, x, t, y):
        model_output = self.post_unet(self.unet_model(x=self.pre_unet(x), timesteps=t, context=self.class_embedder(y)))
        return model_output

    @torch.no_grad()
    def forward_feats(self, x, t, y, depth):
        raise NotImplementedError

def UNet_1(**kwargs):
    return UNet(
        patch_size=1,
        **kwargs,
    )

def UNet_2(**kwargs):
    return UNet(
        patch_size=2,
        **kwargs,
    )
