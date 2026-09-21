"""
IFID wrapper for RAEv2 stage-1 autoencoders.

Drop this file into: ifid/vae/raev2.py

The wrapper adapts RAEv2's official convention to the IFID codebase:
  - IFID/evalvae images are in [-1, 1]
  - RAEv2 official encode() expects [0, 1] or [0, 255] and internally rescales [0,1] -> [0,255]
  - RAEv2 official decode() returns [0, 1]
  - This wrapper returns decode output in [-1, 1]

It can also download the official RAEv2 repo and required HF checkpoint files.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image


RAEV2_GITHUB = "https://github.com/nanovisionx/RAEv2.git"
RAEV2_ZIP = "https://github.com/nanovisionx/RAEv2/archive/refs/heads/main.zip"
DEFAULT_HF_REPO = "nyu-visionx/RAEv2-models"

# Official DINOv3 loader filenames from RAEv2/src/encoders/models/dinov3_loader.py
DINOV3_SHA = {
    "s16": "08c60483",
    "s16plus": "4057cbaa",
    "b16": "73cec8be",
    "l16": "8aa4cbdd",
    "h16plus": "7c1da9a5",
    "7b16": "a955f4ea",
}


def _patch_torch_amp_for_dinov3():
    """
    Compatibility patch for older PyTorch versions.

    DINOv3 uses:
        from torch.amp import custom_fwd, custom_bwd

    and decorates functions like:
        @custom_fwd(device_type="cuda")
        @custom_bwd(device_type="cuda")

    Older PyTorch may only provide torch.cuda.amp.custom_fwd/custom_bwd,
    whose decorator signatures are different. This patch adapts both forms.
    """
    import types
    import torch

    if not hasattr(torch, "amp"):
        torch.amp = types.SimpleNamespace()

    if not hasattr(torch.amp, "custom_fwd"):
        from torch.cuda.amp import custom_fwd as cuda_custom_fwd

        def custom_fwd(*args, **kwargs):
            # New API may pass device_type="cuda"; old cuda API does not accept it.
            kwargs.pop("device_type", None)

            # Used as: @custom_fwd
            if len(args) == 1 and callable(args[0]):
                return cuda_custom_fwd(args[0], **kwargs)

            # Used as: @custom_fwd(device_type="cuda", cast_inputs=...)
            def decorator(fn):
                return cuda_custom_fwd(fn, **kwargs)

            return decorator

        torch.amp.custom_fwd = custom_fwd

    if not hasattr(torch.amp, "custom_bwd"):
        from torch.cuda.amp import custom_bwd as cuda_custom_bwd

        def custom_bwd(*args, **kwargs):
            # New API may pass device_type="cuda"; old cuda API does not accept it.
            kwargs.pop("device_type", None)

            # Used as: @custom_bwd
            if len(args) == 1 and callable(args[0]):
                return cuda_custom_bwd(args[0])

            # Used as: @custom_bwd(device_type="cuda")
            def decorator(fn):
                return cuda_custom_bwd(fn)

            return decorator

        torch.amp.custom_bwd = custom_bwd


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "RAEv2"


def _safe_symlink_or_copy(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _ensure_official_repo(repo_dir: Path) -> None:
    repo_dir = Path(repo_dir)
    marker = repo_dir / "src" / "stage1" / "rae.py"
    if marker.exists():
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists() and not marker.exists():
        # incomplete previous download
        shutil.rmtree(repo_dir)

    print(f"[RAEv2Wrapper] official repo not found, downloading to: {repo_dir}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", RAEV2_GITHUB, str(repo_dir)],
            check=True,
        )
    except Exception as git_err:
        print(f"[RAEv2Wrapper] git clone failed: {git_err}")
        print("[RAEv2Wrapper] fallback: downloading main.zip")
        zip_path = repo_dir.parent / "RAEv2-main.zip"
        urllib.request.urlretrieve(RAEV2_ZIP, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(repo_dir.parent)
        extracted = repo_dir.parent / "RAEv2-main"
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        extracted.rename(repo_dir)
        zip_path.unlink(missing_ok=True)

    if not marker.exists():
        raise FileNotFoundError(f"Failed to prepare official RAEv2 repo at {repo_dir}")


def _strip_pretrained_prefix(path: str) -> str:
    p = path.replace("\\", "/")
    if p.startswith("pretrained_models/"):
        p = p[len("pretrained_models/") :]
    return p


def _download_hf_file(repo_id: str, filename: str, local_path: Path) -> Path:
    """Download a file from HF to the exact local path expected by official RAEv2."""
    if local_path.exists() or local_path.is_symlink():
        return local_path
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    print(f"[RAEv2Wrapper] downloading HF file: {repo_id}/{filename}")
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    _safe_symlink_or_copy(cached, local_path)
    return local_path


def _maybe_download_path(repo_dir: Path, hf_repo_id: str, path_value: Optional[str]) -> Optional[str]:
    """Resolve a decoder/stats path and download if it points into pretrained_models."""
    if path_value is None:
        return None
    path_str = str(path_value)
    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    local_path = repo_dir / path_str
    if path_str.startswith("pretrained_models/"):
        hf_filename = _strip_pretrained_prefix(path_str)
        _download_hf_file(hf_repo_id, hf_filename, local_path)
    return str(local_path)


def _parse_dinov3_base(encoder_name: str) -> Optional[str]:
    # Examples:
    #   dinov3-vit-l16
    #   dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]
    if not encoder_name.startswith("dinov3"):
        return None
    parts = encoder_name.split("-")
    if len(parts) != 3:
        return None
    model_config = parts[2]
    base = model_config.split("[")[0]
    return base


def _ensure_encoder_weights(repo_dir: Path, hf_repo_id: str, encoder_name: str) -> None:
    base = _parse_dinov3_base(encoder_name)
    if base is None:
        return
    if base not in DINOV3_SHA:
        print(f"[RAEv2Wrapper] unknown DINOv3 base={base}; skip encoder auto-download")
        return
    model_name = f"dinov3_vit{base}"
    fname = f"{model_name}_pretrain_lvd1689m-{DINOV3_SHA[base]}.pth"
    local_path = repo_dir / "pretrained_models" / "encoders" / "dinov3" / fname
    hf_filename = f"encoders/dinov3/{fname}"
    _download_hf_file(hf_repo_id, hf_filename, local_path)


def _canonical_device(device):
    if device is None:
        return None

    device = torch.device(device)

    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    return device

class RAEv2(nn.Module):
    """RAEv2 wrapper compatible with IFID's instantiate_from_config()."""

    def __init__(
        self,
        official_repo_dir: Optional[str] = None,
        sampling_config_relpath: str = "configs/stage1/sampling/dinov3l-k7-general.yaml",
        hf_repo_id: str = DEFAULT_HF_REPO,
        model_dtype: str = "bf16",
        use_autocast: bool = True,
        # Direct overrides. Leave None to use official sampling YAML.
        encoder_name: Optional[str] = None,
        resolution: Optional[int] = None,
        decoder_config_path: Optional[str] = None,
        pretrained_decoder_path: Optional[str] = None,
        normalization_stat_path: Optional[str] = None,
        noise_tau: Optional[float] = None,
        decoder_patch_size: Optional[int] = None,
        eps: Optional[float] = None,
        auto_download: bool = True,
    ):
        super().__init__()
        self.repo_dir = Path(official_repo_dir) if official_repo_dir else _repo_root_default()
        self.sampling_config_relpath = sampling_config_relpath
        self.hf_repo_id = hf_repo_id
        self.use_autocast = bool(use_autocast)
        self.auto_download = bool(auto_download)
        self._rae = None

        if model_dtype == "bf16":
            self.model_dtype = torch.bfloat16
        elif model_dtype == "fp16":
            self.model_dtype = torch.float16
        elif model_dtype == "fp32":
            self.model_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported model_dtype: {model_dtype}")

        if self.auto_download:
            _ensure_official_repo(self.repo_dir)
        else:
            marker = self.repo_dir / "src" / "stage1" / "rae.py"
            if not marker.exists():
                raise FileNotFoundError(f"Official RAEv2 repo missing: {self.repo_dir}")

        self.params = self._load_params_from_config()

        # Apply explicit overrides.
        overrides = dict(
            encoder_name=encoder_name,
            resolution=resolution,
            decoder_config_path=decoder_config_path,
            pretrained_decoder_path=pretrained_decoder_path,
            normalization_stat_path=normalization_stat_path,
            noise_tau=noise_tau,
            decoder_patch_size=decoder_patch_size,
            eps=eps,
        )
        for k, v in overrides.items():
            if v is not None:
                self.params[k] = v

        # Resolve/download paths that official code expects.
        if self.auto_download:
            _ensure_encoder_weights(self.repo_dir, self.hf_repo_id, self.params["encoder_name"])
        self.params["decoder_config_path"] = str(
            self._resolve_repo_path(self.params.get("decoder_config_path", "configs/decoder/ViTXL"))
        )
        self.params["pretrained_decoder_path"] = _maybe_download_path(
            self.repo_dir, self.hf_repo_id, self.params.get("pretrained_decoder_path")
        )
        self.params["normalization_stat_path"] = _maybe_download_path(
            self.repo_dir, self.hf_repo_id, self.params.get("normalization_stat_path")
        )
        self.params.setdefault("noise_tau", 0.0)
        self.params.setdefault("resolution", 256)

        print("[RAEv2Wrapper] official repo:", self.repo_dir)
        print("[RAEv2Wrapper] sampling config:", self.sampling_config_relpath)
        print("[RAEv2Wrapper] HF repo:", self.hf_repo_id)
        print("[RAEv2Wrapper] encoder:", self.params.get("encoder_name"))
        print("[RAEv2Wrapper] decoder:", self.params.get("pretrained_decoder_path"))
        print("[RAEv2Wrapper] stats:", self.params.get("normalization_stat_path"))
        print("[RAEv2Wrapper] model dtype:", self.model_dtype)

    def _load_params_from_config(self) -> Dict[str, Any]:
        cfg_path = self._resolve_repo_path(self.sampling_config_relpath)
        if not cfg_path.exists():
            raise FileNotFoundError(f"RAEv2 sampling config not found: {cfg_path}")
        cfg = OmegaConf.to_object(OmegaConf.load(cfg_path))
        stage_1 = cfg.get("stage_1") or cfg.get("stage1")
        if stage_1 is None:
            raise ValueError(f"Config missing stage_1 section: {cfg_path}")
        params = dict(stage_1.get("params") or {})
        if "encoder_name" not in params:
            raise ValueError(f"Config missing stage_1.params.encoder_name: {cfg_path}")
        return params

    def _resolve_repo_path(self, p: str | Path) -> Path:
        p = Path(str(p))
        return p if p.is_absolute() else self.repo_dir / p

    def _ensure_rae(self, device: Optional[torch.device] = None) -> nn.Module:
        if self._rae is not None:
            if device is not None:
                self._rae.to(device)
            return self._rae

        src_dir = self.repo_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        device = _canonical_device(device)

        if device is not None and device.type == "cuda":
            torch.cuda.set_device(device.index)

        _patch_torch_amp_for_dinov3()

        from stage1 import RAE as OfficialRAE

        rae = OfficialRAE(**self.params)
        
        if device is not None:
            rae = rae.to(device)
        rae.eval()
        for p in rae.parameters():
            p.requires_grad_(False)
        self._rae = rae
        return self._rae

    def to(self, *args, **kwargs):  # keep lazy build compatible with evalvae.py's .to(device)
        ret = super().to(*args, **kwargs)
        device = None
        if len(args) > 0:
            if isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], str):
                device = torch.device(args[0])
            elif torch.is_tensor(args[0]):
                device = args[0].device
        if device is None and "device" in kwargs and kwargs["device"] is not None:
            device = torch.device(kwargs["device"])
        if device is not None:
            device = _canonical_device(device)
            self._ensure_rae(device)
        return ret

    def _amp_ctx(self, device: torch.device):
        if not self.use_autocast or device.type != "cuda" or self.model_dtype == torch.float32:
            return torch.autocast(device_type="cuda", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.model_dtype)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # IFID convention [-1,1] -> official RAEv2 convention [0,1].
        device = x.device
        rae = self._ensure_rae(device)
        x01 = ((x.float() + 1.0) / 2.0).clamp(0.0, 1.0).to(device)
        with self._amp_ctx(device):
            z = rae.encode(x01)
        if isinstance(z, (tuple, list)):
            z = z[0]
        return z.float().contiguous()

    @torch.no_grad()
    def decode(self, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        device = z.device
        rae = self._ensure_rae(device)
        z_in = z.to(device)
        with self._amp_ctx(device):
            x01 = rae.decode(z_in)
        if isinstance(x01, (tuple, list)):
            x01 = x01[0]
        x01 = x01.float().clamp(0.0, 1.0)
        return x01 * 2.0 - 1.0

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.decode(self.encode(x))


# Backward-compatible alias if you prefer target: ifid.vae.raev2.RAEv2Wrapper
RAEv2Wrapper = RAEv2


def _load_image(path: Optional[str], size: int) -> torch.Tensor:
    if path is None:
        # deterministic gradient test image, [1,3,H,W] in [-1,1]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij"
        )
        img = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0).unsqueeze(0)
        return img * 2.0 - 1.0
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.BICUBIC)
    arr = torch.from_numpy(__import__("numpy").array(img)).float() / 255.0
    return arr.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def _save_img(x: torch.Tensor, path: str) -> None:
    x01 = ((x.detach().cpu().float().squeeze(0) + 1.0) / 2.0).clamp(0, 1)
    arr = (x01.permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(arr).save(path)


def _psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    x01 = ((x.float() + 1.0) / 2.0).clamp(0, 1)
    y01 = ((y.float() + 1.0) / 2.0).clamp(0, 1)
    mse = F.mse_loss(x01, y01).item()
    return float("inf") if mse <= 0 else -10.0 * math.log10(mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo-dir", type=str, default=None)
    parser.add_argument("--sampling-config-relpath", type=str, default="configs/stage1/sampling/dinov3l-k7-general.yaml")
    parser.add_argument("--hf-repo-id", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--model-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="./debug_raev2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-autodownload", action="store_true")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    vae = RAEv2(
        official_repo_dir=args.official_repo_dir,
        sampling_config_relpath=args.sampling_config_relpath,
        hf_repo_id=args.hf_repo_id,
        model_dtype=args.model_dtype,
        auto_download=not args.no_autodownload,
    ).to(device)
    vae.eval()

    resolution = int(vae.params.get("resolution", 256))
    x = _load_image(args.image, resolution).to(device)
    z = vae.encode(x)
    y = vae.decode(z)

    os.makedirs(args.out_dir, exist_ok=True)
    _save_img(x, os.path.join(args.out_dir, "input.png"))
    _save_img(y, os.path.join(args.out_dir, "recon.png"))

    print("[RAEv2Wrapper:test] x:", tuple(x.shape), x.min().item(), x.max().item())
    print("[RAEv2Wrapper:test] z:", tuple(z.shape), z.dtype, z.mean().item(), z.std().item())
    print("[RAEv2Wrapper:test] y:", tuple(y.shape), y.min().item(), y.max().item())
    print(f"[RAEv2Wrapper:test] PSNR: {_psnr(x, y):.3f} dB")
    print("[RAEv2Wrapper:test] saved:", os.path.join(args.out_dir, "input.png"))
    print("[RAEv2Wrapper:test] saved:", os.path.join(args.out_dir, "recon.png"))


if __name__ == "__main__":
    main()
