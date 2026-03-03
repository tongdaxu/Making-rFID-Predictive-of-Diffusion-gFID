import torch
from ifid.vae.autoencoder import Decoder
from torchvision.models.vision_transformer import VisionTransformer
import torch.nn as nn


def create_small_vit_s(output_dim=8, patch_size=16, img_size=256):
    """
    Create a lightweight ViT-S model.

    Args:
        output_dim: Output feature dimension.
        patch_size: Patch size for input images.
        img_size: Input image size.
    """
    # Compute number of patches (e.g., 256/16 = 16 → 16x16 = 256 patches)
    (img_size // patch_size) ** 2

    # Small ViT-S configuration
    vit_config = {
        "image_size": img_size,
        "patch_size": patch_size,
        "num_layers": 6,  # fewer layers for a lightweight model
        "num_heads": 8,  # fewer attention heads
        "hidden_dim": 384,  # smaller hidden dimension
        "mlp_dim": 1536,  # typically 4x hidden_dim
        "num_classes": output_dim,
        "dropout": 0.1,
        "attention_dropout": 0.1,
    }

    model = VisionTransformer(**vit_config)

    # Replace classification head with a linear projection to output_dim
    # The output shape will be (B, 8, 256)
    model.heads = nn.Sequential(
        nn.Linear(vit_config["hidden_dim"], vit_config["hidden_dim"]),
        nn.GELU(),
        nn.Linear(vit_config["hidden_dim"], output_dim),
    )

    # Print parameter counts
    sum(p.numel() for p in model.parameters())
    sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Define custom forward to return patch-level features
    def forward_custom(x):
        # Extract features via ViT
        x = model._process_input(x)
        x.shape[1]

        # Add class token
        batch_size = x.shape[0]
        cls_tokens = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Pass through Transformer encoder
        x = model.encoder(x)

        # Remove class token, keep patch tokens only
        x = x[:, 1:, :]  # shape: (B, 256, 384)

        # Apply head projection
        x = model.heads(x)  # shape: (B, 256, 8)

        # Transpose to (B, 8, 256)
        return x.transpose(1, 2)

    model.forward = forward_custom
    return model


def match_distribution(h, h_vit, eps=1e-6):
    """
    Match h_vit distribution to h distribution.

    Args:
        h: [B, D1, N]   (DINO features)
        h_vit: [B, D2, N] (ViT features)
    """
    # Compute global mean and std for DINO features
    mean_h = h.mean(dim=(0, 2), keepdim=True)
    std_h = h.std(dim=(0, 2), keepdim=True)

    mean_h_scalar = mean_h.mean().detach()
    std_h_scalar = std_h.mean().detach()

    # Compute mean and std for ViT features
    mean_vit = h_vit.mean(dim=(0, 2), keepdim=True)
    std_vit = h_vit.std(dim=(0, 2), keepdim=True)

    mean_vit_scalar = mean_vit.mean().detach()
    std_vit_scalar = std_vit.mean().detach()

    # Normalize and re-scale
    h_vit_normed = (h_vit - mean_vit_scalar) / (std_vit_scalar + eps)
    h_vit_aligned = h_vit_normed * std_h_scalar + mean_h_scalar

    return h_vit_aligned


class SVGEXPORT(nn.Module):
    def __init__(
        self,
        ddconfig,
        dinoconfig,
        embed_dim,
        extra_vit_config=None,
        ckpt_path=None,
    ):
        super().__init__()
        self.decoder = Decoder(**ddconfig)
        self.encoder = torch.hub.load(
            repo_or_dir=dinoconfig["dinov3_location"],
            model=dinoconfig["model_name"],
            source="local",
            weights=dinoconfig["weights"],
        ).eval()

        self.use_outnorm = False

        if extra_vit_config is not None:
            self.use_extra_vit = True
            self.extra_vit = create_small_vit_s(
                output_dim=extra_vit_config["output_dim"]
            )
            self.use_outnorm = extra_vit_config.get("use_outnorm", False)
            self.norm_vit = nn.LayerNorm(extra_vit_config["output_dim"] + embed_dim)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        """Load checkpoint with optional key filtering."""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                print(f"Deleting key {k} from state_dict.")
                del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        x = (x + 1) / 2

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        h = self.encoder.forward_features(x)["x_norm_patchtokens"]  # [B, D, N]
        h = h.permute(0, 2, 1)  # Adjust to [B, N, D] or [B, D, N]

        if self.use_extra_vit:
            h_vit = self.extra_vit(x)

            if self.use_outnorm:
                h_vit = match_distribution(h, h_vit)

            h = torch.cat([h, h_vit], dim=1)

        h = h.view(
            h.shape[0], -1, int(x.shape[2] // 16), int(x.shape[3] // 16)
        ).contiguous()
        return h

    def decode(self, z):
        return self.decoder(z)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/SVG.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
