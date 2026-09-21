import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModel, Dinov2Model, SiglipModel
from ifid.vae.scale_rae_decoder import MultimodalDecoder


class ScaleRAE(nn.Module):
    """
    IFID wrapper for Scale-RAE / RAE-style reconstruction.

    Input:
      x: [-1, 1], shape [B, 3, H, W]

    encode:
      image -> visual patch tokens
      output shape: [B, T, D]

    decode:
      patch tokens -> reconstructed image
      output shape: [B, 3, H, W], range [-1, 1]
    """

    def __init__(
        self,
        encoder_name: str,
        decoder_repo: str,
        encoder_type: str = "siglip2",
        encoder_input_size: int = 224,
        num_patches: int = 256,
        output_size: int = 256,
        select_layer: int = -1,
        feature_layer_norm: bool = False,
        normalization_stat_path: str = None,
        normalize_latents: bool = False,
        trust_remote_code: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.decoder_repo = decoder_repo
        self.encoder_type = encoder_type.lower()
        self.encoder_input_size = encoder_input_size
        self.num_patches = num_patches
        self.output_size = output_size
        self.select_layer = select_layer
        self.feature_layer_norm = feature_layer_norm
        self.normalize_latents = normalize_latents
        self.trust_remote_code = trust_remote_code

        # ------------------------------------------------------------
        # Encoder processor / normalization
        # ------------------------------------------------------------
        self.image_processor = AutoImageProcessor.from_pretrained(
            encoder_name,
            trust_remote_code=trust_remote_code,
        )

        image_mean = getattr(self.image_processor, "image_mean", [0.5, 0.5, 0.5])
        image_std = getattr(self.image_processor, "image_std", [0.5, 0.5, 0.5])

        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # ------------------------------------------------------------
        # Encoder model
        # ------------------------------------------------------------
        if self.encoder_type in ["siglip", "siglip2"]:
            # Match official RAE SigLIP2wNorm.
            self.encoder = SiglipModel.from_pretrained(encoder_name)
            self.vision_model = self.encoder.vision_model

            if hasattr(self.vision_model, "post_layernorm"):
                self.vision_model.post_layernorm.elementwise_affine = False
                self.vision_model.post_layernorm.weight = None
                self.vision_model.post_layernorm.bias = None

        elif self.encoder_type in ["webssl", "dino", "dinov2"]:
            # Match Scale-RAE DinoVisionTower:
            # use Dinov2Model, NOT Dinov2WithRegistersModel.
            # Do NOT remove layernorm affine here.
            self.encoder = Dinov2Model.from_pretrained(encoder_name)
            self.vision_model = self.encoder

        else:
            self.encoder = AutoModel.from_pretrained(
                encoder_name,
                trust_remote_code=trust_remote_code,
            )
            self.vision_model = (
                self.encoder.vision_model
                if hasattr(self.encoder, "vision_model")
                else self.encoder
            )
        # ------------------------------------------------------------
        # Decoder config / checkpoint
        # ------------------------------------------------------------
        if os.path.isdir(decoder_repo):
            decoder_config_path = os.path.join(decoder_repo, "config.json")
            decoder_ckpt_path = os.path.join(decoder_repo, "model.pt")
        else:
            decoder_config_path = hf_hub_download(
                repo_id=decoder_repo,
                filename="config.json",
            )
            decoder_ckpt_path = hf_hub_download(
                repo_id=decoder_repo,
                filename="model.pt",
            )

        self.decoder = MultimodalDecoder(
            pretrained_encoder_path=encoder_name,
            general_decoder_config=decoder_config_path,
            num_patches=num_patches,
            drop_cls_token=True,
            decoder_path=decoder_ckpt_path,
        )

        # ------------------------------------------------------------
        # Optional latent stats
        # ------------------------------------------------------------
        if normalization_stat_path is not None:
            stat = torch.load(normalization_stat_path, map_location="cpu")

            if "mean" not in stat:
                raise KeyError(
                    f"stat file must contain key 'mean', got keys: {stat.keys()}"
                )

            mean = stat["mean"].float()

            if "std" in stat:
                std = stat["std"].float()
            elif "var" in stat:
                std = torch.sqrt(stat["var"].float().clamp_min(1e-12))
            else:
                raise KeyError(
                    f"stat file must contain key 'std' or 'var', got keys: {stat.keys()}"
                )

            print("[ScaleRAE] loaded latent stats:", normalization_stat_path)
            print("[ScaleRAE] mean shape:", tuple(mean.shape))
            print("[ScaleRAE] std shape:", tuple(std.shape))
            print("[ScaleRAE] std min/max:", std.min().item(), std.max().item())
        else:
            mean = torch.empty(0)
            std = torch.empty(0)

        self.register_buffer("latents_mean", mean, persistent=False)
        self.register_buffer("latents_std", std, persistent=False)

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        self._last_hw = None

    # ------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------
    def _preprocess_for_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [-1,1]
        return: encoder-normalized image tensor
        """
        self._last_hw = tuple(x.shape[-2:])

        x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)

        if x.shape[-2:] != (self.encoder_input_size, self.encoder_input_size):
            x = F.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )

        mean = self.image_mean.to(device=x.device, dtype=x.dtype)
        std = self.image_std.to(device=x.device, dtype=x.dtype)

        x = (x - mean) / std
        return x

    def _to_patch_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        Convert encoder output to [B, num_patches, D].
        """
        if features.ndim != 3:
            raise RuntimeError(f"Expected [B,T,D], got {features.shape}")

        t = features.shape[1]

        if t == self.num_patches:
            return features

        if t == self.num_patches + 1:
            return features[:, 1:]

        target_h = int(math.sqrt(self.num_patches))
        if target_h * target_h != self.num_patches:
            raise RuntimeError(f"num_patches must be square, got {self.num_patches}")

        # no CLS case
        h = int(math.sqrt(t))
        if h * h == t:
            b, _, d = features.shape
            features = features.view(b, h, h, d).permute(0, 3, 1, 2).contiguous()
            features = F.interpolate(
                features.float(),
                size=(target_h, target_h),
                mode="bilinear",
                align_corners=False,
            ).to(features.dtype)
            features = features.permute(0, 2, 3, 1).contiguous()
            return features.view(b, self.num_patches, d)

        # CLS case
        t2 = t - 1
        h2 = int(math.sqrt(t2))
        if h2 * h2 == t2:
            features = features[:, 1:]
            b, _, d = features.shape
            features = features.view(b, h2, h2, d).permute(0, 3, 1, 2).contiguous()
            features = F.interpolate(
                features.float(),
                size=(target_h, target_h),
                mode="bilinear",
                align_corners=False,
            ).to(features.dtype)
            features = features.permute(0, 2, 3, 1).contiguous()
            return features.view(b, self.num_patches, d)

        raise RuntimeError(
            f"Cannot convert token length {t} to num_patches={self.num_patches}."
        )

    # ------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self._preprocess_for_encoder(x)

        if self.encoder_type in ["webssl", "dino", "dinov2"]:
            enc_param = next(self.vision_model.parameters())
            x = x.to(device=enc_param.device, dtype=enc_param.dtype)

            # Match Scale-RAE DinoVisionTower._forward:
            # image_forward_outs = self.vision_tower.forward(images)
            outputs = self.vision_model(
                x,
                return_dict=True,
            )

            features = outputs.last_hidden_state

            # Match feature_select(..., select_feature='patch'):
            # sequence_output[:, 1:]
            # Only drop CLS token. Do NOT drop 5 tokens.
            if features.shape[1] == self.num_patches + 1:
                features = features[:, 1:]

        else:
            outputs = self.vision_model(
                x,
                output_hidden_states=True,
                interpolate_pos_encoding=True,
            )

            # Match official RAE SigLIP2wNorm:
            # use last_hidden_state, not hidden_states[-1].
            features = outputs.last_hidden_state

        features = self._to_patch_tokens(features)

        if self.feature_layer_norm:
            features = F.layer_norm(
                features,
                (features.shape[-1],),
                weight=None,
                bias=None,
                eps=1e-6,
            )

        if self.normalize_latents and self.latents_mean.numel() > 0:
            mean = self.latents_mean.to(device=features.device, dtype=features.dtype)
            std = self.latents_std.to(device=features.device, dtype=features.dtype)

            features = (features - mean) / (std + 1e-6)

            # Debug print; delete after sanity check if too noisy.
            print(
                "[ScaleRAE] applied latent norm:",
                "z_mean=", features.mean().item(),
                "z_std=", features.std().item(),
            )

        return features

    # ------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------
    @torch.no_grad()
    def decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Scale-RAE decode.

        If z is normalized by encode(), first invert normalization.

        Important:
          The copied Scale-RAE GeneralDecoder expects a CLS token and
          drop_cls_token=True. Calling general_decoder(z) or
          general_decoder(z, drop_cls_token=False) will raise:
            NotImplementedError("drop_cls_token is not implemented")
        """
        if z.ndim == 4:
            z = z.flatten(2).transpose(1, 2).contiguous()

        if z.ndim != 3:
            raise RuntimeError(f"Expected z as [B,T,D], got {z.shape}")

        # Invert latent normalization before decoder.
        if self.normalize_latents and self.latents_mean.numel() > 0:
            mean = self.latents_mean.to(device=z.device, dtype=z.dtype)
            std = self.latents_std.to(device=z.device, dtype=z.dtype)
            z = z * (std + 1e-6) + mean

        general_decoder = self.decoder.decoder

        decoder_dtype = next(general_decoder.parameters()).dtype
        z = z.to(dtype=decoder_dtype)

        # Scale-RAE GeneralDecoder expects [B, 1 + num_patches, D].
        empty_cls = torch.zeros(
            (z.shape[0], 1, z.shape[-1]),
            device=z.device,
            dtype=z.dtype,
        )
        z_with_cls = torch.cat([empty_cls, z], dim=1)

        # Important: use drop_cls_token=True.
        out = general_decoder(z_with_cls, drop_cls_token=True).logits
        x_rec = general_decoder.unpatchify(out)

        mean = self.image_mean.to(device=x_rec.device, dtype=x_rec.dtype)
        std = self.image_std.to(device=x_rec.device, dtype=x_rec.dtype)

        x01 = x_rec * std + mean
        x01 = x01.float().clamp(0.0, 1.0)

        target_hw = self._last_hw if self._last_hw is not None else (
            self.output_size,
            self.output_size,
        )

        if tuple(x01.shape[-2:]) != tuple(target_hw):
            x01 = F.interpolate(
                x01,
                size=target_hw,
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)

        return x01 * 2.0 - 1.0

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    cfg = OmegaConf.load("../../configs/ScaleRAE_SIGLIP2_NONORM.yaml")
    vae = instantiate_from_config(cfg).cuda().eval()

    x = torch.rand(1, 3, 256, 256, device="cuda") * 2.0 - 1.0
    with torch.no_grad():
        z = vae.encode(x)
        y = vae.decode(z)

    print("z:", z.shape, z.mean().item(), z.std().item())
    print("y:", y.shape, y.min().item(), y.max().item(), y.mean().item(), y.std().item())