# ifid/vae/unified_ae.py

from __future__ import annotations

from contextlib import nullcontext
from math import sqrt
from typing import Any, Dict, Optional, Protocol

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor

from .decoders import GeneralDecoder
from .encoders import ARCHS


class Stage1Protocal(Protocol):
    patch_size: int
    hidden_size: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...


class UnifiedUAE(nn.Module):
    def __init__(
        self,
        # ---- encoder configs ----
        encoder_cls: str = "Dinov2withNorm",
        encoder_config_path: str = "facebook/dinov2-base",
        encoder_input_size: int = 224,
        encoder_params: Optional[Dict[str, Any]] = None,
        encoder_trainable: bool = False,
        # ---- decoder configs ----
        decoder_config_path: str = "vit_mae-base",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization ----
        noise_tau: float = 0.8,
        add_train_noise: bool = False,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        # ---- official UAE frequency config ----
        frequency_config: Optional[Dict[str, Any]] = None,
        frequency_trainable: bool = True,  # kept for config compatibility, unused
    ):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------
        encoder_params = dict(encoder_params or {})
        encoder_cls_obj = ARCHS[encoder_cls]

        self.encoder: Stage1Protocal = encoder_cls_obj(**encoder_params)
        self.encoder_trainable = bool(encoder_trainable)

        if self.encoder_trainable:
            self.encoder.requires_grad_(True)
        else:
            self.encoder.requires_grad_(False)

        print(f"[UnifiedUAE] encoder_config_path: {encoder_config_path}")

        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)

        self.register_buffer("encoder_mean", encoder_mean, persistent=False)
        self.register_buffer("encoder_std", encoder_std, persistent=False)

        self.encoder_input_size = int(encoder_input_size)
        self.encoder_patch_size = int(self.encoder.patch_size)
        self.latent_dim = int(self.encoder.hidden_size)

        assert self.encoder_input_size % self.encoder_patch_size == 0, (
            f"encoder_input_size {self.encoder_input_size} must be divisible by "
            f"encoder_patch_size {self.encoder_patch_size}"
        )

        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        # -------------------------
        # Decoder
        # -------------------------
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))

        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        self.noise_tau = float(noise_tau)
        self.add_train_noise = bool(add_train_noise)
        self.reshape_to_2d = bool(reshape_to_2d)

        # -------------------------
        # Official UAE-style frequency config
        # -------------------------
        freq_cfg = dict(frequency_config or {})

        self.freq_transform = str(freq_cfg.get("freq_transform", "none")).lower()
        self.freq_order = str(freq_cfg.get("freq_order", "zigzag")).lower()
        self.use_frequency_module = self.freq_transform == "dct"

        if self.freq_transform not in {"none", "dct"}:
            raise ValueError(
                f"Unsupported freq_transform={self.freq_transform}. "
                "Supported values are: none, dct."
            )

        if self.freq_order not in {"zigzag", "radial"}:
            raise ValueError(
                f"Unsupported freq_order={self.freq_order}. "
                "Supported values are: zigzag, radial."
            )

        if self.use_frequency_module and not self.reshape_to_2d:
            raise ValueError("DCT frequency module requires reshape_to_2d=True.")

        split_cfg = dict(freq_cfg.get("band_split", {}))

        self.band_split_enabled = bool(split_cfg.get("enabled", False))
        self.band_split_mode = str(split_cfg.get("mode", "low")).lower()
        self.band_split_count = split_cfg.get(
            "count",
            split_cfg.get("keep_count", None),
        )
        self.band_split_ratio = split_cfg.get(
            "ratio",
            split_cfg.get("keep_ratio", None),
        )

        if self.band_split_mode not in {
            "none",
            "all",
            "low",
            "keep_low",
            "high",
            "keep_high",
            "drop_high",
            "drop_low",
        }:
            raise ValueError(
                f"Unsupported band_split.mode={self.band_split_mode}."
            )

        self._dct_cache: Dict[Any, Any] = {}
        self.register_buffer(
            "_freq_perm",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

        # Keep decoder neutral.
        if hasattr(self.decoder, "band_mask_fusion"):
            self.decoder.band_mask_fusion = "none"
        if hasattr(self.decoder, "set_band_mask_dim"):
            self.decoder.set_band_mask_dim(0)

        # -------------------------
        # Latent normalization stats
        # -------------------------
        self.latent_mean: Optional[torch.Tensor] = None
        self.latent_var: Optional[torch.Tensor] = None
        self.do_normalization = False
        self.eps = float(eps)

        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")

            self.latent_mean = stats.get("mean", None)
            self.latent_var = stats.get("var", None)

            # Some codebases use these names.
            if self.latent_mean is None:
                self.latent_mean = stats.get("latent_mean", None)
            if self.latent_var is None:
                self.latent_var = stats.get("latent_var", None)

            self.do_normalization = (
                self.latent_mean is not None and self.latent_var is not None
            )

            print(f"[UnifiedUAE] Loaded normalization stats from {normalization_stat_path}")
            print(f"[UnifiedUAE] do_normalization={self.do_normalization}")

        # -------------------------
        # Optional decoder-only checkpoint
        # Usually not needed if outer wrapper loads full ckpt.
        # Kept for backward compatibility.
        # -------------------------
        if pretrained_decoder_path is not None:
            print(f"[UnifiedUAE] Loading pretrained decoder from {pretrained_decoder_path}")

            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")

            if isinstance(state_dict, dict):
                if "model" in state_dict and isinstance(state_dict["model"], dict):
                    decoder_state = {
                        key.removeprefix("decoder."): value
                        for key, value in state_dict["model"].items()
                        if key.startswith("decoder.")
                    }
                    if decoder_state:
                        state_dict = decoder_state

                elif "state_dict" in state_dict and isinstance(
                    state_dict["state_dict"], dict
                ):
                    decoder_state = {
                        key.removeprefix("decoder."): value
                        for key, value in state_dict["state_dict"].items()
                        if key.startswith("decoder.")
                    }
                    if decoder_state:
                        state_dict = decoder_state

            decoder_current = self.decoder.state_dict()
            filtered_state: Dict[str, torch.Tensor] = {}

            for key, tensor in state_dict.items():
                if key in decoder_current and decoder_current[key].shape == tensor.shape:
                    filtered_state[key] = tensor

            keys = self.decoder.load_state_dict(filtered_state, strict=False)

            if len(keys.missing_keys) > 0:
                print(
                    "[UnifiedUAE] Missing keys when loading pretrained decoder:",
                    keys.missing_keys,
                )

    # ---------------------------------------------------------------------
    # DCT frequency module
    # ---------------------------------------------------------------------

    def _build_dct_matrix(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Orthonormal DCT-II matrix.
        """
        k = torch.arange(n, device=device, dtype=dtype).view(n, 1)
        i = torch.arange(n, device=device, dtype=dtype).view(1, n)

        mat = torch.cos((torch.pi / n) * (i + 0.5) * k)

        mat[0] *= 1.0 / sqrt(n)
        if n > 1:
            mat[1:] *= sqrt(2.0 / n)

        return mat

    def _get_dct_matrices(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            h,
            w,
            device.type,
            device.index if device.index is not None else -1,
            str(dtype),
        )

        cached = self._dct_cache.get(key, None)
        if cached is not None:
            return cached

        mat_h = self._build_dct_matrix(h, device, dtype)
        mat_w = self._build_dct_matrix(w, device, dtype)

        self._dct_cache[key] = (mat_h, mat_w)
        return mat_h, mat_w

    def _dct2(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        mat_h, mat_w = self._get_dct_matrices(h, w, x.device, x.dtype)

        x_flat = x.reshape(-1, h, w)
        x_flat = torch.matmul(mat_h, x_flat)
        x_flat = torch.matmul(x_flat, mat_w.t())

        return x_flat.reshape(b, c, h, w)

    def _idct2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of the orthonormal DCT-II above.
        """
        b, c, h, w = x.shape
        mat_h, mat_w = self._get_dct_matrices(h, w, x.device, x.dtype)

        x_flat = x.reshape(-1, h, w)
        x_flat = torch.matmul(mat_h.t(), x_flat)
        x_flat = torch.matmul(x_flat, mat_w)

        return x_flat.reshape(b, c, h, w)

    def _zigzag_perm(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Zigzag order from top-left low-frequency region.
        """
        coords = []

        for s in range(h + w - 1):
            if s % 2 == 0:
                x = min(s, w - 1)
                y = s - x

                while x >= 0 and y < h:
                    coords.append((y, x))
                    x -= 1
                    y += 1
            else:
                y = min(s, h - 1)
                x = s - y

                while y >= 0 and x < w:
                    coords.append((y, x))
                    y -= 1
                    x += 1

        linear = [y * w + x for y, x in coords]
        return torch.tensor(linear, device=device, dtype=torch.long)

    def _radial_perm(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )

        radius = torch.sqrt(yy.square() + xx.square())
        return torch.argsort(radius.flatten(), dim=0)

    def _get_frequency_perm(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self._freq_perm.numel() == h * w and self._freq_perm.device == device:
            return self._freq_perm

        if self.freq_order == "zigzag":
            perm = self._zigzag_perm(h, w, device)
        elif self.freq_order == "radial":
            perm = self._radial_perm(h, w, device)
        else:
            raise ValueError(f"Unsupported freq_order={self.freq_order}")

        self._freq_perm = perm
        return perm

    def _resolve_keep_count(self, token_count: int) -> int:
        if self.band_split_count is not None:
            keep = int(self.band_split_count)
        elif self.band_split_ratio is not None:
            keep = int(round(float(self.band_split_ratio) * token_count))
        else:
            keep = token_count

        return max(0, min(keep, token_count))

    def _build_band_mask(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
        apply_band_mask: Optional[bool] = None,
    ) -> Optional[torch.Tensor]:
        """
        Returns:
          mask: [1, 1, H, W]
          or None.
        """
        if not self.band_split_enabled:
            return None

        if apply_band_mask is False:
            return None

        mode = self.band_split_mode

        if mode in {"none", "all"}:
            return None

        token_count = h * w
        keep = self._resolve_keep_count(token_count)

        perm = self._get_frequency_perm(h, w, device)
        mask_flat = torch.zeros(token_count, device=device, dtype=dtype)

        if mode in {"low", "keep_low"}:
            # Keep lowest-frequency positions.
            mask_flat[perm[:keep]] = 1.0

        elif mode in {"high", "keep_high"}:
            # Keep highest-frequency positions.
            mask_flat[perm[token_count - keep:]] = 1.0

        elif mode == "drop_high":
            # Drop highest `keep`; keep the lower-frequency remainder.
            keep_low = token_count - keep
            mask_flat[perm[:keep_low]] = 1.0

        elif mode == "drop_low":
            # Drop lowest `keep`; keep the higher-frequency remainder.
            mask_flat[perm[keep:]] = 1.0

        else:
            raise ValueError(f"Unsupported band_split.mode={mode}")

        return mask_flat.view(1, 1, h, w)

    def _apply_dct_frequency_module(
        self,
        z_map: torch.Tensor,
        *,
        apply_band_mask: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        z_map: [B, C, H, W]

        UAE official-style frequency path:
          z_map -> DCT2 -> frequency mask -> IDCT2
        """
        if not self.use_frequency_module:
            return z_map

        original_dtype = z_map.dtype

        # DCT/IDCT in fp32 is safer.
        if z_map.dtype in (torch.float16, torch.bfloat16):
            z_work = z_map.float()
        else:
            z_work = z_map

        b, c, h, w = z_work.shape

        freq = self._dct2(z_work)

        mask = self._build_band_mask(
            h=h,
            w=w,
            device=z_work.device,
            dtype=freq.dtype,
            apply_band_mask=apply_band_mask,
        )

        if mask is not None:
            freq = freq * mask

        z_filtered = self._idct2(freq)

        if z_filtered.dtype != original_dtype:
            z_filtered = z_filtered.to(original_dtype)

        return z_filtered

    # ---------------------------------------------------------------------
    # Encode / Decode
    # ---------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        *,
        apply_frequency: Optional[bool] = None,
        apply_band_mask: Optional[bool] = None,
        dropout: Optional[torch.Tensor] = None,      # compatibility only
        freq_ratio: Optional[torch.Tensor] = None,   # compatibility only
        freeze_encoder: bool = False,
        return_pre_frequency: bool = False,
        return_frequency_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Official UAE convention:
          input x should be [0,1].

        The outer UAE wrapper handles:
          IFID [-1,1] -> official UAE [0,1]
        """
        use_frequency = (
            self.use_frequency_module
            if apply_frequency is None
            else bool(apply_frequency)
        )

        encoder_ctx = torch.no_grad() if freeze_encoder else nullcontext()

        with encoder_ctx:
            _, _, h, w = x.shape

            if h != self.encoder_input_size or w != self.encoder_input_size:
                x = nn.functional.interpolate(
                    x,
                    size=(self.encoder_input_size, self.encoder_input_size),
                    mode="bicubic",
                    align_corners=False,
                )

            x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)

            z = self.encoder(x)

            if self.reshape_to_2d:
                b, n, c = z.shape
                latent_hw = int(sqrt(n))

                if latent_hw * latent_hw != n:
                    raise ValueError(f"Expected square token count, got n={n}")

                z = z.transpose(1, 2).view(b, c, latent_hw, latent_hw)

        if freeze_encoder:
            z = z.detach()

        pre_frequency_latent = z

        if use_frequency:
            z = self._apply_dct_frequency_module(
                z,
                apply_band_mask=apply_band_mask,
            )

        if self.training and self.add_train_noise and self.noise_tau > 0:
            noise_sigma = self.noise_tau * torch.rand(
                (z.size(0),) + (1,) * (z.dim() - 1),
                device=z.device,
                dtype=z.dtype,
            )
            z = z + noise_sigma * torch.randn_like(z)

        if self.do_normalization:
            mean = self._format_latent_stat(self.latent_mean, z.device, z.dtype)
            var = self._format_latent_stat(self.latent_var, z.device, z.dtype)

            if mean is not None and var is not None:
                z = (z - mean) / torch.sqrt(var + z.new_tensor(self.eps))

        if return_pre_frequency and return_frequency_tokens:
            return z, pre_frequency_latent, None

        if return_pre_frequency:
            return pre_frequency_latent, z

        if return_frequency_tokens:
            return z, None

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            mean = self._format_latent_stat(self.latent_mean, z.device, z.dtype)
            var = self._format_latent_stat(self.latent_var, z.device, z.dtype)

            if mean is not None and var is not None:
                z = z * torch.sqrt(var + z.new_tensor(self.eps)) + mean

        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)

        try:
            output = self.decoder(
                z,
                band_mask=None,
                drop_cls_token=False,
            ).logits
        except TypeError:
            output = self.decoder(
                z,
                drop_cls_token=False,
            ).logits

        try:
            x_rec = self.decoder.unpatchify(output)
        except TypeError:
            x_rec = self.decoder.unpatchify(output, original_image_size=None)

        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(
            x_rec.device
        )

        return x_rec

    # ---------------------------------------------------------------------
    # Latent stat formatting
    # ---------------------------------------------------------------------

    def _format_latent_stat(
        self,
        stat: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if stat is None:
            return None

        tensor = stat.to(device=device, dtype=dtype)

        if not self.reshape_to_2d:
            if (
                tensor.ndim == 2
                and tensor.shape[0] == self.base_patches
                and tensor.shape[1] == self.latent_dim
            ):
                return tensor.unsqueeze(0)

            if tensor.ndim == 1 and tensor.numel() == self.latent_dim:
                return tensor.view(1, 1, self.latent_dim)

            return tensor

        h = w = int(sqrt(self.base_patches))

        if tensor.ndim == 1 and tensor.numel() == self.latent_dim:
            return tensor.view(1, self.latent_dim, 1, 1)

        if (
            tensor.ndim == 2
            and tensor.shape[0] == self.base_patches
            and tensor.shape[1] == self.latent_dim
        ):
            return tensor.view(h, w, self.latent_dim).permute(2, 0, 1).unsqueeze(0)

        if (
            tensor.ndim == 3
            and tensor.shape[0] == h
            and tensor.shape[1] == w
            and tensor.shape[2] == self.latent_dim
        ):
            return tensor.permute(2, 0, 1).unsqueeze(0)

        if tensor.ndim == 4:
            if tensor.shape[0] == 1:
                return tensor
            return tensor.view(1, *tensor.shape)

        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, apply_frequency=None)
        return self.decode(z)