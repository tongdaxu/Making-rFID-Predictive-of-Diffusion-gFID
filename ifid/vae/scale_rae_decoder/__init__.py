import torch
import torch.nn as nn
from typing import Optional

from transformers import AutoConfig, AutoImageProcessor

from .decoder import GeneralDecoder


def _get_encoder_hidden_size(pretrained_encoder_path: str) -> int:
    cfg = AutoConfig.from_pretrained(pretrained_encoder_path)

    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)

    if hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"):
        return int(cfg.vision_config.hidden_size)

    if hasattr(cfg, "vision_config") and isinstance(cfg.vision_config, dict):
        if "hidden_size" in cfg.vision_config:
            return int(cfg.vision_config["hidden_size"])

    raise RuntimeError(
        f"Cannot infer hidden_size from encoder config: {pretrained_encoder_path}"
    )


class MultimodalDecoder(nn.Module):
    """
    Minimal local wrapper for Scale-RAE / RAE GeneralDecoder.

    It loads:
      - GeneralDecoder config.json
      - decoder model.pt

    Forward returns image in [0,1].
    """

    def __init__(
        self,
        pretrained_encoder_path: str,
        general_decoder_config: str,
        num_patches: int,
        drop_cls_token: bool = False,
        decoder_path: Optional[str] = None,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(general_decoder_config)

        # The copied decoder.py expects this field to be "eager" or "sdpa".
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"

        # Make sure decoder hidden size matches encoder output dim.
        config.hidden_size = _get_encoder_hidden_size(pretrained_encoder_path)

        self.drop_cls_token = drop_cls_token
        self.num_patches = num_patches

        processor = AutoImageProcessor.from_pretrained(pretrained_encoder_path)
        image_mean = getattr(processor, "image_mean", [0.5, 0.5, 0.5])
        image_std = getattr(processor, "image_std", [0.5, 0.5, 0.5])

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

        self.decoder = GeneralDecoder(config, num_patches=num_patches)

        if decoder_path is not None:
            state_dict = torch.load(decoder_path, map_location="cpu")

            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            missing, unexpected = self.decoder.load_state_dict(
                state_dict,
                strict=False,
            )

            print("[ScaleRAE Decoder] loaded from:", decoder_path)
            print("[ScaleRAE Decoder] missing keys:", len(missing))
            print("[ScaleRAE Decoder] unexpected keys:", len(unexpected))

            for k in missing[:30]:
                print("[ScaleRAE Decoder][missing]", k)
            for k in unexpected[:30]:
                print("[ScaleRAE Decoder][unexpected]", k)

            bad_missing = [
                k for k in missing
                if "decoder_pred" in k
                or "decoder_embed" in k
                or "decoder_layers" in k
                or "decoder_norm" in k
            ]

            if len(bad_missing) > 0:
                raise RuntimeError(
                    "ScaleRAE decoder checkpoint/model mismatch. "
                    f"Bad missing keys: {bad_missing[:30]}"
                )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        image_features:
          [B, T, D] or [B, T+1, D]

        returns:
          image in [0,1]
        """
        if image_features.ndim != 3:
            raise RuntimeError(
                f"Expected image_features [B,T,D], got {image_features.shape}"
            )

        if self.drop_cls_token and image_features.shape[1] == self.num_patches + 1:
            image_features = image_features[:, 1:]

        out = self.decoder(image_features, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(out)

        mean = self.image_mean.to(device=x_rec.device, dtype=x_rec.dtype)
        std = self.image_std.to(device=x_rec.device, dtype=x_rec.dtype)

        x_rec = x_rec * std + mean
        x_rec = x_rec.clamp(0.0, 1.0)

        return x_rec