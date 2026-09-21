import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer

from ifid.vae.autoencoder import Decoder


def create_small_vit_s(
    output_dim=8,
    patch_size=16,
    img_size=256,
    hidden_dim=384,
    vit_type="vit-s",
):
    """
    Lightweight residual ViT used by SVG / SVG-T2I Autoencoder-R.

    Output:
      [B, output_dim, N]
    """
    if vit_type != "vit-s":
        raise NotImplementedError(
            f"Only vit-s is implemented in this lightweight IFID wrapper, got {vit_type}"
        )

    vit_config = {
        "image_size": img_size,
        "patch_size": patch_size,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": hidden_dim,
        "mlp_dim": 1536,
        "num_classes": output_dim,
        "dropout": 0.1,
        "attention_dropout": 0.1,
    }

    model = VisionTransformer(**vit_config)

    model.heads = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )

    def forward_custom(x):
        x = model._process_input(x)

        batch_size = x.shape[0]
        cls_tokens = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = model.encoder(x)

        # remove CLS, keep patch tokens
        x = x[:, 1:, :]  # [B, N, hidden_dim]

        x = model.heads(x)  # [B, N, output_dim]

        return x.transpose(1, 2).contiguous()  # [B, output_dim, N]

    model.forward = forward_custom
    return model


def match_distribution(h, h_vit, eps=1e-6):
    """
    Match residual ViT feature distribution to DINO feature distribution.

    h:
      [B, D1, N]
    h_vit:
      [B, D2, N]
    """
    mean_h = h.mean(dim=(0, 2), keepdim=True)
    std_h = h.std(dim=(0, 2), keepdim=True)

    mean_h_scalar = mean_h.mean().detach()
    std_h_scalar = std_h.mean().detach()

    mean_vit = h_vit.mean(dim=(0, 2), keepdim=True)
    std_vit = h_vit.std(dim=(0, 2), keepdim=True)

    mean_vit_scalar = mean_vit.mean().detach()
    std_vit_scalar = std_vit.mean().detach()

    h_vit_normed = (h_vit - mean_vit_scalar) / (std_vit_scalar + eps)
    h_vit_aligned = h_vit_normed * std_h_scalar + mean_h_scalar

    return h_vit_aligned


class SVGT2I(nn.Module):
    """
    IFID wrapper for SVG-T2I Autoencoder-P / Autoencoder-R.

    IFID convention:
      input x:  [-1, 1]
      output y: [-1, 1]

    SVG-T2I official convention:
      encode() expects DINO-normalized input.
      decode() reconstructs image in [-1, 1].

    Expected latent shape:
      P-stage1-256:
        [B, 384, 16, 16]

      R-stage1-256:
        [B, 392, 16, 16]
    """

    def __init__(
        self,
        ddconfig,
        dinoconfig,
        embed_dim=32,
        extra_vit_config=None,
        ckpt_path=None,
        image_size=256,
        output_clamp=True,
        strict=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.image_size = image_size
        self.output_clamp = output_clamp
        self.strict = strict
        self.embed_dim = embed_dim

        self.decoder = Decoder(**ddconfig)

        self.encoder = torch.hub.load(
            repo_or_dir=dinoconfig["dinov3_location"],
            model=dinoconfig["model_name"],
            source="local",
            weights=dinoconfig["weights"],
        ).eval()

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.use_extra_vit = extra_vit_config is not None
        self.use_outnorm = False

        if self.use_extra_vit:
            self.extra_vit = create_small_vit_s(
                output_dim=extra_vit_config.get("output_dim", 8),
                patch_size=extra_vit_config.get("patch_size", 16),
                img_size=extra_vit_config.get("img_size", image_size),
                hidden_dim=extra_vit_config.get("hidden_dim", 384),
                vit_type=extra_vit_config.get("vit_type", "vit-s"),
            )

            self.mask_ratio = extra_vit_config.get("mask_ratio", 0.0)
            self.use_outnorm = extra_vit_config.get("use_outnorm", False)
            self.frozen_vit = extra_vit_config.get("frozen", False)

            if self.frozen_vit:
                self.extra_vit.eval()
                for p in self.extra_vit.parameters():
                    p.requires_grad_(False)

            if self.mask_ratio > 0:
                self.mask_token = nn.Parameter(
                    torch.zeros(1, extra_vit_config.get("output_dim", 8), 1)
                )
                nn.init.normal_(self.mask_token, std=0.02)

            # Official code creates this, although it is not used in the simple path.
            self.norm_vit = nn.LayerNorm(
                extra_vit_config.get("output_dim", 8) + embed_dim
            )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self._last_hw = None

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, strict=strict)

        self.eval()

    def init_from_ckpt(self, path, strict=False, ignore_keys=None):
        ignore_keys = ignore_keys or []

        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                print(f"[SVGT2I] deleting key from checkpoint: {k}")
                del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=strict)

        print("[SVGT2I] restored from:", path)
        print("[SVGT2I] missing keys:", len(missing))
        print("[SVGT2I] unexpected keys:", len(unexpected))

        for k in missing[:30]:
            print("[SVGT2I][missing]", k)
        for k in unexpected[:30]:
            print("[SVGT2I][unexpected]", k)

    def _to_dino_input(self, x):
        """
        x:
          [-1, 1]

        return:
          ImageNet-normalized image for DINOv3
        """
        self._last_hw = tuple(x.shape[-2:])

        x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)

        if tuple(x01.shape[-2:]) != (self.image_size, self.image_size):
            x01 = F.interpolate(
                x01,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)

        mean = self.imagenet_mean.to(device=x01.device, dtype=x01.dtype)
        std = self.imagenet_std.to(device=x01.device, dtype=x01.dtype)

        return (x01 - mean) / std

    @torch.no_grad()
    def encode(self, x, *args, **kwargs):
        """
        P-stage1:
          [B, 384, 16, 16]

        R-stage1:
          [B, 392, 16, 16]
        """
        x_dino = self._to_dino_input(x)

        h = self.encoder.forward_features(x_dino)["x_norm_patchtokens"]
        # DINOv3 output should be [B, N, 384]
        # Convert to [B, 384, N]
        h = h.permute(0, 2, 1).contiguous()

        if self.use_extra_vit:
            h_vit = self.extra_vit(x_dino)  # [B, 8, N]

            if self.use_outnorm:
                h_vit = match_distribution(h, h_vit)

            h = torch.cat([h, h_vit], dim=1)  # [B, 392, N]

        patch_h = int(x_dino.shape[2] // 16)
        patch_w = int(x_dino.shape[3] // 16)

        h = h.view(h.shape[0], -1, patch_h, patch_w).contiguous()
        return h

    @torch.no_grad()
    def decode(self, z, *args, **kwargs):
        y = self.decoder(z)

        if self.output_clamp:
            y = y.float().clamp(-1.0, 1.0)

        target_hw = self._last_hw if self._last_hw is not None else (
            self.image_size,
            self.image_size,
        )

        if tuple(y.shape[-2:]) != tuple(target_hw):
            y = F.interpolate(
                y,
                size=target_hw,
                mode="bicubic",
                align_corners=False,
            ).clamp(-1.0, 1.0)

        return y

    @torch.no_grad()
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    cfg = OmegaConf.load("../../configs/SVGT2I_P_STAGE1_256.yaml")
    vae = instantiate_from_config(cfg).cuda().eval()

    x = torch.randn(1, 3, 256, 256, device="cuda")
    z = vae.encode(x)
    y = vae.decode(z)
    print("z:", z.shape, z.mean().item(), z.std().item())
    print("y:", y.shape, y.min().item(), y.max().item(), y.mean().item(), y.std().item())