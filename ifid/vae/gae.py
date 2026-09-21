"""
IFID wrapper for GAE: Geometric Autoencoder for Diffusion Models.

Put this file at:
    ifid/vae/gae.py

Expected IFID interface:
    encode(x[, y]) -> z, where x is in [-1, 1]
    decode(z[, y]) -> x_rec, returned in [-1, 1]

Official repo:
    https://github.com/sii-research/GAE
Official checkpoint HF repo:
    sii-research/gae-imagenet256-f16d32
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf


_GAE_GIT_URL = "https://github.com/sii-research/GAE.git"
_DEFAULT_HF_REPO = "sii-research/gae-imagenet256-f16d32"
_DEFAULT_CKPT = "d32/gae_d32_inference.ckpt"
_DEFAULT_CONFIG_RELPATH = "configs/ae/gae_d32.yaml"


def _ifid_root_default() -> Path:
    # ifid/vae/gae.py -> IFID root is parents[2]
    return Path(__file__).resolve().parents[2]


def _default_repo_dir() -> Path:
    return _ifid_root_default() / "third_party" / "GAE"


def _canonical_device(device: Optional[torch.device | str]) -> Optional[torch.device]:
    if device is None:
        return None
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def _dtype_from_string(s: str) -> torch.dtype:
    s = str(s).lower()
    if s in ("fp32", "float32", "32"):
        return torch.float32
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported model_dtype={s}")


def _clone_repo_if_needed(repo_dir: Path, git_url: str = _GAE_GIT_URL) -> None:
    repo_dir = Path(repo_dir)
    if (repo_dir / "tokenizer" / "gae.py").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GAEWrapper] cloning official repo to: {repo_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", git_url, str(repo_dir)],
        check=True,
    )


def _download_hf_file(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to auto-download GAE weights. "
            "Install it with: pip install huggingface_hub"
        ) from e

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )


class GAE(nn.Module):
    """
    Thin wrapper around official GAE FoundationEncoderModel.

    The official model returns a posterior object from encode(); this wrapper returns
    posterior.mode() by default, because IFID/downstream tokenizer evaluation should
    normally use deterministic latents.
    """

    def __init__(
        self,
        repo_dir: Optional[str] = None,
        auto_clone: bool = True,
        gae_config_relpath: str = _DEFAULT_CONFIG_RELPATH,
        hf_repo: str = _DEFAULT_HF_REPO,
        ckpt_filename: str = _DEFAULT_CKPT,
        ckpt_path: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        model_dtype: str = "fp32",
        encode_mode: str = "mode",  # "mode" or "sample"
        clamp_decode: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.repo_dir = Path(repo_dir) if repo_dir is not None else _default_repo_dir()
        self.auto_clone = bool(auto_clone)
        self.gae_config_relpath = gae_config_relpath
        self.hf_repo = hf_repo
        self.ckpt_filename = ckpt_filename
        self.ckpt_path = ckpt_path
        self.hf_cache_dir = hf_cache_dir
        self.model_dtype = _dtype_from_string(model_dtype)
        self.encode_mode = encode_mode
        self.clamp_decode = bool(clamp_decode)
        self.verbose = bool(verbose)
        self._model: Optional[nn.Module] = None

        if self.encode_mode not in ("mode", "sample"):
            raise ValueError("encode_mode must be 'mode' or 'sample'")

    def _ensure_model(self, device: Optional[torch.device] = None) -> nn.Module:
        device = _canonical_device(device)
        if self._model is not None:
            if device is not None:
                self._model.to(device)
            return self._model

        if self.auto_clone:
            _clone_repo_if_needed(self.repo_dir)
        elif not (self.repo_dir / "tokenizer" / "gae.py").exists():
            raise FileNotFoundError(
                f"GAE repo not found at {self.repo_dir}. "
                f"Clone {_GAE_GIT_URL} there or set auto_clone=True."
            )

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        if device is not None and device.type == "cuda":
            torch.cuda.set_device(device.index)

        from tokenizer.utils.util import instantiate_from_config as gae_instantiate

        config_path = self.repo_dir / self.gae_config_relpath
        if not config_path.exists():
            raise FileNotFoundError(f"Official GAE config not found: {config_path}")

        cfg = OmegaConf.load(str(config_path))
        if self.verbose:
            print(f"[GAEWrapper] official repo: {self.repo_dir}")
            print(f"[GAEWrapper] official config: {config_path}")

        model = gae_instantiate(cfg.model)

        ckpt = self.ckpt_path
        if ckpt is None:
            ckpt = _download_hf_file(
                repo_id=self.hf_repo,
                filename=self.ckpt_filename,
                cache_dir=self.hf_cache_dir,
            )
        if self.verbose:
            print(f"[GAEWrapper] loading checkpoint: {ckpt}")
        model.init_from_ckpt(ckpt)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        if self.model_dtype != torch.float32:
            model.to(dtype=self.model_dtype)
        if device is not None:
            model.to(device)

        self._model = model
        return model

    def to(self, *args, **kwargs):  # noqa: D401
        ret = super().to(*args, **kwargs)
        device = None
        if len(args) > 0:
            try:
                device = torch.device(args[0])
            except Exception:
                device = None
        if "device" in kwargs and kwargs["device"] is not None:
            device = torch.device(kwargs["device"])
        if device is not None:
            self._ensure_model(device)
        return ret

    def eval(self):  # noqa: D401
        super().eval()
        if self._model is not None:
            self._model.eval()
        return self

    def train(self, mode: bool = True):  # noqa: D401
        # Keep official GAE frozen/eval even if outer wrapper train() is called.
        super().train(False)
        if self._model is not None:
            self._model.eval()
        return self

    def _model_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        model = self._ensure_model()
        p = next(model.parameters())
        return p.device, p.dtype

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        x: [B,3,H,W] in [-1,1]. Returns z: [B,32,16,16] float32 by default.
        """
        model = self._ensure_model(x.device)
        p = next(model.parameters())
        x_in = x.to(device=p.device, dtype=p.dtype)
        posterior, _ = model.encode(x_in)
        if self.encode_mode == "sample":
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z.float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        z: [B,32,16,16]. Returns reconstruction in [-1,1] approximately.
        """
        model = self._ensure_model(z.device)
        p = next(model.parameters())
        z_in = z.to(device=p.device, dtype=p.dtype)
        x = model.decode(z_in).float()
        if self.clamp_decode:
            x = x.clamp(-1.0, 1.0)
        return x


def _pil_to_tensor_m11(path: Optional[str], device: torch.device) -> torch.Tensor:
    if path is None:
        # simple deterministic synthetic image, just for shape/dtype sanity.
        t = torch.linspace(-1, 1, 256, device=device)
        yy, xx = torch.meshgrid(t, t, indexing="ij")
        x = torch.stack([xx, yy, 0.5 * torch.sin(6.28 * xx)], dim=0).unsqueeze(0)
        return x.clamp(-1, 1)

    img = Image.open(path).convert("RGB")
    # match ADM-style resize/crop approximately
    img = img.resize((256, 256), Image.BICUBIC)
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    arr = arr.view(256, 256, 3).permute(2, 0, 1).float() / 127.5 - 1.0
    return arr.unsqueeze(0).to(device)


def _psnr_m11(x: torch.Tensor, y: torch.Tensor) -> float:
    x01 = (x.float().clamp(-1, 1) + 1) / 2
    y01 = (y.float().clamp(-1, 1) + 1) / 2
    mse = F.mse_loss(x01, y01).item()
    if mse <= 0:
        return 99.0
    return float(-10.0 * torch.log10(torch.tensor(mse)).item())


def _save_tensor(path: str, x: torch.Tensor) -> None:
    x01 = (x[0].detach().cpu().float().clamp(-1, 1) + 1) / 2
    arr = (x01.permute(1, 2, 0).numpy() * 255.0 + 0.5).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=str, default=str(_default_repo_dir()))
    parser.add_argument("--gae-config-relpath", type=str, default=_DEFAULT_CONFIG_RELPATH)
    parser.add_argument("--hf-repo", type=str, default=_DEFAULT_HF_REPO)
    parser.add_argument("--ckpt-filename", type=str, default=_DEFAULT_CKPT)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--model-dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--encode-mode", type=str, default="mode", choices=["mode", "sample"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="./debug_gae")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    vae = GAE(
        repo_dir=args.repo_dir,
        gae_config_relpath=args.gae_config_relpath,
        hf_repo=args.hf_repo,
        ckpt_filename=args.ckpt_filename,
        ckpt_path=args.ckpt_path,
        hf_cache_dir=args.hf_cache_dir,
        model_dtype=args.model_dtype,
        encode_mode=args.encode_mode,
    ).to(device).eval()

    x = _pil_to_tensor_m11(args.image, device)
    z = vae.encode(x)
    y = vae.decode(z)

    print(f"[GAEWrapper:test] x: {tuple(x.shape)} {x.dtype} range=({x.min().item():.4f},{x.max().item():.4f})")
    print(f"[GAEWrapper:test] z: {tuple(z.shape)} {z.dtype} mean/std={z.mean().item():.4f}/{z.std().item():.4f}")
    print(f"[GAEWrapper:test] y: {tuple(y.shape)} {y.dtype} range=({y.min().item():.4f},{y.max().item():.4f})")
    print(f"[GAEWrapper:test] PSNR: {_psnr_m11(x, y):.3f} dB")

    _save_tensor(os.path.join(args.out_dir, "input.png"), x)
    _save_tensor(os.path.join(args.out_dir, "recon.png"), y)
    print(f"[GAEWrapper:test] saved: {args.out_dir}/input.png")
    print(f"[GAEWrapper:test] saved: {args.out_dir}/recon.png")


if __name__ == "__main__":
    main()
