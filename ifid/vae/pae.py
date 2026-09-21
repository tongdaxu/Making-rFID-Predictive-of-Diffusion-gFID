"""
IFID wrapper for PAE (Prior-Aligned AutoEncoder).

Put this file at:
    ifid/vae/pae.py

Then use a config like:
    configs/PAE_DINOv2L_d32.yaml

External convention for this IFID project:
    encode(x): x is [-1, 1], returns latent [B, C, H, W]
    decode(z): returns image in [-1, 1]

Official PAE convention:
    input image is [0, 1]
    PAE.encode returns [B, C, H, W]
    PAE.decode returns [B, 3, H, W] in [0, 1]
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


class PAEWrapper(nn.Module):
    def __init__(
        self,
        pae_repo_dir: str = "./third_party/PAE",
        auto_clone: bool = True,
        hf_repo_id: str = "yuezhengrong/PAE-collections",
        pae_config_relpath: str = "pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml",
        ckpt_filename: str = "AE-models/dinov2-large.pt",
        model_dtype: str = "bf16",
        device_dtype_for_encoder: bool = True,
        output_size: int = 256,
        clamp_decode: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.pae_repo_dir = Path(pae_repo_dir).expanduser().resolve()
        self.auto_clone = bool(auto_clone)
        self.hf_repo_id = hf_repo_id
        self.pae_config_relpath = pae_config_relpath
        self.ckpt_filename = ckpt_filename
        self.output_size = int(output_size)
        self.clamp_decode = bool(clamp_decode)
        self.verbose = bool(verbose)
        self.device_dtype_for_encoder = bool(device_dtype_for_encoder)

        self.model_dtype = self._parse_dtype(model_dtype)

        self._ensure_official_repo()
        self._add_official_repo_to_path()

        from tokenizer.pae import PAE as OfficialPAE  # imported after sys.path is set

        cfg_path = self.pae_repo_dir / self.pae_config_relpath
        if not cfg_path.exists():
            raise FileNotFoundError(f"PAE config not found: {cfg_path}")

        train_cfg = OmegaConf.load(str(cfg_path))
        if "stage_1" not in train_cfg:
            raise KeyError(f"Expected key 'stage_1' in {cfg_path}")

        stage_1 = train_cfg.stage_1
        params = OmegaConf.to_container(stage_1.get("params", {}), resolve=True)
        params = self._resolve_relative_paths(params)

        if self.verbose:
            print("[PAEWrapper] official repo:", self.pae_repo_dir)
            print("[PAEWrapper] official config:", cfg_path)
            print("[PAEWrapper] HF repo:", self.hf_repo_id)
            print("[PAEWrapper] ckpt filename:", self.ckpt_filename)
            print("[PAEWrapper] model dtype:", self.model_dtype)

        self.pae = OfficialPAE(**params)
        self._load_checkpoint()

        if self.model_dtype is not None:
            self.pae = self.pae.to(dtype=self.model_dtype)

        self.pae.eval()
        for p in self.pae.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _parse_dtype(name: Optional[str]):
        if name is None:
            return None
        name = str(name).lower()
        if name in ["none", "fp32", "float32"]:
            return torch.float32
        if name in ["bf16", "bfloat16"]:
            return torch.bfloat16
        if name in ["fp16", "float16", "half"]:
            return torch.float16
        raise ValueError(f"Unsupported model_dtype: {name}")

    def _ensure_official_repo(self):
        if (self.pae_repo_dir / "pae_with_generator" / "tokenizer" / "pae.py").exists():
            return
        if not self.auto_clone:
            raise FileNotFoundError(
                f"Official PAE repo not found at {self.pae_repo_dir}. "
                "Set auto_clone=True or clone https://github.com/ZhengrongYue/PAE there."
            )
        self.pae_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"[PAEWrapper] cloning official repo into {self.pae_repo_dir}")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ZhengrongYue/PAE.git",
                str(self.pae_repo_dir),
            ],
            check=True,
        )

    def _add_official_repo_to_path(self):
        root = str((self.pae_repo_dir / "pae_with_generator").resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    def _resolve_relative_paths(self, params: dict) -> dict:
        """Official YAML uses paths such as configs/decoder/ViTL relative to pae_with_generator."""
        out = dict(params)
        base = self.pae_repo_dir / "pae_with_generator"
        for key in [
            "decoder_config_path",
            "pretrained_decoder_path",
            "pretrained_delta_encoder_path",
            "pretrained_compressor_path",
            "pretrained_pae_path",
        ]:
            val = out.get(key, None)
            if isinstance(val, str) and val and not os.path.isabs(val):
                if val.startswith("configs/") or val.startswith("tokenizer/") or val.startswith("pae_with_generator/"):
                    if val.startswith("pae_with_generator/"):
                        out[key] = str((self.pae_repo_dir / val).resolve())
                    else:
                        out[key] = str((base / val).resolve())
        return out

    def _load_checkpoint(self):
        ckpt_path = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=self.ckpt_filename,
        )
        if self.verbose:
            print("[PAEWrapper] loading checkpoint:", ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        cleaned = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("module.", "_orig_mod."):
                while new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            cleaned[new_key] = value

        msg = self.pae.load_state_dict(cleaned, strict=False)
        if self.verbose:
            print("[PAEWrapper] missing keys:", len(msg.missing_keys))
            if len(msg.missing_keys) > 0:
                print("[PAEWrapper] first missing keys:", msg.missing_keys[:20])
            print("[PAEWrapper] unexpected keys:", len(msg.unexpected_keys))
            if len(msg.unexpected_keys) > 0:
                print("[PAEWrapper] first unexpected keys:", msg.unexpected_keys[:20])

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        IFID convention:
        input x: [-1, 1]
        PAE convention:
        input x01: [0, 1]

        PAE uses FlashAttention, so CUDA forward must use fp16/bf16.
        """
        dev = next(self.pae.parameters()).device

        # external [-1,1] -> official [0,1]
        x01 = (x.float() + 1.0) / 2.0
        x01 = x01.clamp(0.0, 1.0).to(device=dev)

        if dev.type == "cuda":
            amp_dtype = self.model_dtype

            # FlashAttention does not support fp32.
            if amp_dtype not in (torch.float16, torch.bfloat16):
                amp_dtype = torch.bfloat16

            # Important: input must also be fp16/bf16.
            x01 = x01.to(dtype=amp_dtype)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z = self.pae.encode(x01)
        else:
            z = self.pae.encode(x01.float())

        if isinstance(z, (tuple, list)):
            z = z[0]

        if not torch.is_tensor(z):
            raise TypeError(f"Unexpected PAE.encode return type: {type(z)}")

        return z.float().contiguous()
        
    @torch.no_grad()
    def decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        IFID convention:
        input z: latent
        output image: [-1, 1]

        PAE decode returns [0, 1], so convert back to [-1, 1].
        """
        dev = next(self.pae.parameters()).device

        z = z.to(device=dev)

        if dev.type == "cuda":
            amp_dtype = self.model_dtype

            # FlashAttention does not support fp32.
            if amp_dtype not in (torch.float16, torch.bfloat16):
                amp_dtype = torch.bfloat16

            z = z.to(dtype=amp_dtype)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                x01 = self.pae.decode(z)
        else:
            x01 = self.pae.decode(z.float())

        if isinstance(x01, (tuple, list)):
            x01 = x01[0]

        if not torch.is_tensor(x01):
            raise TypeError(f"Unexpected PAE.decode return type: {type(x01)}")

        x01 = x01.float().clamp(0.0, 1.0)
        return x01 * 2.0 - 1.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# Backward-compatible alias for shorter YAML target if you prefer.
PAE = PAEWrapper


def _psnr_from_m11(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x01 = ((x.float() + 1.0) / 2.0).clamp(0, 1)
    y01 = ((y.float() + 1.0) / 2.0).clamp(0, 1)
    mse = torch.mean((x01 - y01) ** 2, dim=(1, 2, 3))
    return -10.0 * torch.log10(mse + eps)


def _load_image_as_m11(path: str, image_size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return tfm(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="", help="Optional path to one RGB image.")
    parser.add_argument("--out-dir", type=str, default="./debug_pae")
    parser.add_argument("--pae-repo-dir", type=str, default="./third_party/PAE")
    parser.add_argument("--hf-repo-id", type=str, default="yuezhengrong/PAE-collections")
    parser.add_argument("--pae-config-relpath", type=str, default="pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml")
    parser.add_argument("--ckpt-filename", type=str, default="AE-models/dinov2-large.pt")
    parser.add_argument("--model-dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = PAEWrapper(
        pae_repo_dir=args.pae_repo_dir,
        hf_repo_id=args.hf_repo_id,
        pae_config_relpath=args.pae_config_relpath,
        ckpt_filename=args.ckpt_filename,
        model_dtype=args.model_dtype,
        output_size=args.image_size,
        verbose=True,
    ).to(device).eval()

    if args.image:
        x = _load_image_as_m11(args.image, args.image_size).to(device)
    else:
        # deterministic synthetic image for smoke testing if no image is given
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, args.image_size),
            torch.linspace(0, 1, args.image_size),
            indexing="ij",
        )
        x01 = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0).unsqueeze(0)
        x = x01 * 2.0 - 1.0
        x = x.to(device)

    with torch.no_grad():
        z = vae.encode(x)
        y = vae.decode(z)
        psnr = _psnr_from_m11(x, y).mean().item()

    print("[PAEWrapper:test] x:", tuple(x.shape), x.min().item(), x.max().item())
    print("[PAEWrapper:test] z:", tuple(z.shape), z.dtype, z.float().mean().item(), z.float().std().item())
    print("[PAEWrapper:test] y:", tuple(y.shape), y.min().item(), y.max().item())
    print(f"[PAEWrapper:test] PSNR: {psnr:.3f} dB")

    save_image(((x.float() + 1) / 2).clamp(0, 1), os.path.join(args.out_dir, "input.png"))
    save_image(((y.float() + 1) / 2).clamp(0, 1), os.path.join(args.out_dir, "recon.png"))
    print("[PAEWrapper:test] saved:", os.path.join(args.out_dir, "input.png"))
    print("[PAEWrapper:test] saved:", os.path.join(args.out_dir, "recon.png"))


if __name__ == "__main__":
    main()
