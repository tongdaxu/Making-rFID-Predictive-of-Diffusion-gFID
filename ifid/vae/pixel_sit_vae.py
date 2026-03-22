import json
import math
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ifid.sit.sit import SiT_models
from ifid.sit.samplers import edict_sampler, edict_inverter
from train import normalize_latents, denormalize_latents


class PIXELVAE(nn.Module):
    def __init__(self, downsample_factor: int = 16):
        super().__init__()
        self.downsample_factor = downsample_factor

    def encode(self, x):
        return F.pixel_unshuffle(x, self.downsample_factor)

    def decode(self, z):
        return F.pixel_shuffle(z, self.downsample_factor)


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = str(dtype).lower()
    if dtype in ["fp32", "float32", "torch.float32"]:
        return torch.float32
    if dtype in ["fp16", "float16", "torch.float16", "half"]:
        return torch.float16
    if dtype in ["bf16", "bfloat16", "torch.bfloat16"]:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


class PixelSiTVAE(nn.Module):
    """
    Encode:
        image [-1, 1]
        -> PIXELVAE.encode()
        -> normalize_latents()
        -> edict_inverter(...)[0]
        -> latent

    Decode:
        latent
        -> edict_sampler(...)[0]
        -> denormalize_latents()
        -> PIXELVAE.decode()
        -> image [-1, 1]
    """

    # defaults
    DEFAULT_SIT_MODEL = "SiT-XL/1"
    DEFAULT_RESOLUTION = 256
    DEFAULT_NUM_CLASSES = 1000
    DEFAULT_CFG_PROB = 0.1
    DEFAULT_BN_MOMENTUM = 0.1
    DEFAULT_FUSED_ATTN = True
    DEFAULT_QK_NORM = False
    DEFAULT_PIXEL_SHUFFLE_FACTOR = 16
    DEFAULT_NUM_STEPS = 250
    DEFAULT_P = 0.9
    DEFAULT_STATE_KEY = "ema"
    DEFAULT_COMPUTE_DTYPE = "float32"
    DEFAULT_STRICT = True

    def __init__(
        self,
        exp_path: Optional[str] = None,
        train_steps: Optional[Union[int, str]] = None,
        ckpt_file: Optional[str] = None,
        args_json: Optional[str] = None,

        # optional overrides
        sit_model: Optional[str] = None,
        resolution: Optional[int] = None,
        num_classes: Optional[int] = None,
        cfg_prob: Optional[float] = None,
        bn_momentum: Optional[float] = None,
        fused_attn: Optional[bool] = None,
        qk_norm: Optional[bool] = None,

        # pixel / edict params
        pixel_shuffle_factor: int = DEFAULT_PIXEL_SHUFFLE_FACTOR,
        num_steps: int = DEFAULT_NUM_STEPS,
        p: float = DEFAULT_P,

        # loading params
        state_key: str = DEFAULT_STATE_KEY,
        compute_dtype: Union[str, torch.dtype] = DEFAULT_COMPUTE_DTYPE,
        strict: bool = DEFAULT_STRICT,
    ):
        super().__init__()

        self.supports_y = True
        self.num_steps = num_steps
        self.p = p
        self.compute_dtype = _resolve_dtype(compute_dtype)

        self.pixel_vae = PIXELVAE(downsample_factor=pixel_shuffle_factor)

        if exp_path is not None:
            if args_json is None:
                args_json = os.path.join(exp_path, "args.json")
            if ckpt_file is None:
                if train_steps is None:
                    raise ValueError("When exp_path is provided, train_steps must also be provided.")
                ckpt_file = os.path.join(
                    exp_path,
                    "checkpoints",
                    f"{int(train_steps):07d}.pt",
                )

        if ckpt_file is None:
            raise ValueError("ckpt_file is required, or provide exp_path + train_steps.")

        raw_ckpt = torch.load(ckpt_file, map_location="cpu")

        if isinstance(raw_ckpt, dict) and state_key in raw_ckpt:
            state_dict = raw_ckpt[state_key]
        else:
            state_dict = raw_ckpt

        ckpt_args = raw_ckpt.get("args", None)

        cfg = self._build_config(
            args_json=args_json,
            ckpt_args=ckpt_args,
            sit_model=sit_model,
            resolution=resolution,
            num_classes=num_classes,
            cfg_prob=cfg_prob,
            bn_momentum=bn_momentum,
            fused_attn=fused_attn,
            qk_norm=qk_norm,
        )

        self.sit_model_name = cfg["sit_model"]
        self.resolution = cfg["resolution"]
        self.num_classes = cfg["num_classes"]

        with torch.no_grad():
            fake_in = torch.zeros(1, 3, self.resolution, self.resolution)
            fake_z = self.pixel_vae.encode(fake_in)

        if fake_z.ndim != 4:
            raise ValueError(
                f"Expected 2D latent from PIXELVAE, but got shape {tuple(fake_z.shape)}"
            )

        latent_size = fake_z.shape[-1]
        in_channels = fake_z.shape[1]
        tshift = math.sqrt(float(fake_z.numel()) / 4096.0)

        self.model = SiT_models[self.sit_model_name](
            input_size=latent_size,
            in_channels=in_channels,
            num_classes=cfg["num_classes"],
            class_dropout_prob=cfg["cfg_prob"],
            bn_momentum=cfg["bn_momentum"],
            tshift=tshift,
            fused_attn=cfg["fused_attn"],
            qk_norm=cfg["qk_norm"],
        )

        raw_ckpt = torch.load(ckpt_file, map_location="cpu")
        if isinstance(raw_ckpt, dict) and state_key in raw_ckpt:
            state_dict = raw_ckpt[state_key]
        else:
            state_dict = raw_ckpt

        # remove training-only projector weights
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("projectors.")
        }

        load_result = self.model.load_state_dict(filtered_state_dict, strict=False)
        print("[PixelSiTVAE] missing keys:", load_result.missing_keys)
        print("[PixelSiTVAE] unexpected keys:", load_result.unexpected_keys)

        self.model.eval()

        for p_ in self.model.parameters():
            p_.requires_grad_(False)

        if "bn.running_var" not in state_dict or "bn.running_mean" not in state_dict:
            raise KeyError(
                "Checkpoint does not contain 'bn.running_var' and 'bn.running_mean'."
            )

        latents_scale = state_dict["bn.running_var"].rsqrt().view(1, in_channels, 1, 1)
        latents_bias = state_dict["bn.running_mean"].view(1, in_channels, 1, 1)

        self.register_buffer("latents_scale", latents_scale.float(), persistent=True)
        self.register_buffer("latents_bias", latents_bias.float(), persistent=True)

    def _load_args_json(self, args_json: Optional[str]) -> Dict[str, Any]:
        if args_json is None or not os.path.isfile(args_json):
            return {}
        with open(args_json, "r") as f:
            return json.load(f)

    def _build_config(
        self,
        args_json: Optional[str],
        ckpt_args,
        sit_model: Optional[str],
        resolution: Optional[int],
        num_classes: Optional[int],
        cfg_prob: Optional[float],
        bn_momentum: Optional[float],
        fused_attn: Optional[bool],
        qk_norm: Optional[bool],
    ) -> Dict[str, Any]:
        json_cfg = self._load_args_json(args_json)

        cfg = {
            "sit_model": self.DEFAULT_SIT_MODEL,
            "resolution": self.DEFAULT_RESOLUTION,
            "num_classes": self.DEFAULT_NUM_CLASSES,
            "cfg_prob": self.DEFAULT_CFG_PROB,
            "bn_momentum": self.DEFAULT_BN_MOMENTUM,
            "fused_attn": self.DEFAULT_FUSED_ATTN,
            "qk_norm": self.DEFAULT_QK_NORM,
        }

        # 1) args.json
        mapping = {
            "model": "sit_model",
            "resolution": "resolution",
            "num_classes": "num_classes",
            "cfg_prob": "cfg_prob",
            "bn_momentum": "bn_momentum",
            "fused_attn": "fused_attn",
            "qk_norm": "qk_norm",
        }
        for src_key, dst_key in mapping.items():
            if src_key in json_cfg:
                cfg[dst_key] = json_cfg[src_key]

        # 2) checkpoint["args"]
        if ckpt_args is not None:
            ckpt_args_dict = vars(ckpt_args) if not isinstance(ckpt_args, dict) else ckpt_args
            for src_key, dst_key in mapping.items():
                if src_key in ckpt_args_dict:
                    cfg[dst_key] = ckpt_args_dict[src_key]

        # 3) explicit yaml override
        explicit_cfg = {
            "sit_model": sit_model,
            "resolution": resolution,
            "num_classes": num_classes,
            "cfg_prob": cfg_prob,
            "bn_momentum": bn_momentum,
            "fused_attn": fused_attn,
            "qk_norm": qk_norm,
        }
        for k, v in explicit_cfg.items():
            if v is not None:
                cfg[k] = v

        return cfg
        
    def _prepare_y(self, y, batch_size, device):
        if y is None:
            raise ValueError(
                "PixelSiTVAE requires class labels y because the wrapped SiT/EDICT model is class-conditional."
            )
        if isinstance(y, int):
            y = torch.full((batch_size,), y, device=device, dtype=torch.long)
        elif torch.is_tensor(y):
            y = y.to(device=device, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported y type: {type(y)}")
        return y

    @torch.no_grad()
    def encode(self, x, y=None, *args, **kwargs):
        y = self._prepare_y(y, x.shape[0], x.device)

        z = self.pixel_vae.encode(x).float()

        z_norm = normalize_latents(
            z,
            self.latents_scale.to(device=z.device, dtype=z.dtype),
            self.latents_bias.to(device=z.device, dtype=z.dtype),
        )

        latent, _, _ = edict_inverter(
            model=self.model,
            latents=z_norm.to(dtype=self.compute_dtype),
            y=y,
            num_steps=self.num_steps,
            p=self.p,
        )
        return latent.float()

    @torch.no_grad()
    def decode(self, latent, y=None, *args, **kwargs):
        y = self._prepare_y(y, latent.shape[0], latent.device)

        z_norm, _, _ = edict_sampler(
            model=self.model,
            latents=latent.to(dtype=self.compute_dtype),
            y=y,
            num_steps=self.num_steps,
            p=self.p,
        )
        z_norm = z_norm.float()

        z = denormalize_latents(
            z_norm,
            self.latents_scale.to(device=z_norm.device, dtype=z_norm.dtype),
            self.latents_bias.to(device=z_norm.device, dtype=z_norm.dtype),
        )

        x = self.pixel_vae.decode(z.float())
        return x