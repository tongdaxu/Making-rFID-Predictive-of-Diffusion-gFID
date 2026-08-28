import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class MMDiTJointAttention(nn.Module):
    """
    Joint attention with modality-specific QKV and output projections.

    Image and text tokens attend over the concatenated image-text sequence,
    while retaining separate projection parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        fused_attn: bool = True,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        # Separate projections preserve modality-specific representations.
        self.x_qkv = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=qkv_bias,
        )
        self.text_qkv = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=qkv_bias,
        )

        self.x_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)

        if qk_norm:
            # Normalize each attention head independently.
            self.x_q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.x_k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.text_q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.text_k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        else:
            self.x_q_norm = nn.Identity()
            self.x_k_norm = nn.Identity()
            self.text_q_norm = nn.Identity()
            self.text_k_norm = nn.Identity()

    def _make_qkv(
        self,
        tokens: torch.Tensor,
        projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            tokens: (B, L, C)

        Output:
            q, k, v: (B, H, L, D)
        """
        batch_size, seq_len, _ = tokens.shape

        qkv = projection(tokens)
        qkv = qkv.reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        return qkv.unbind(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
                Image tokens, shape (B, Lx, C).

            text:
                Text tokens, shape (B, Lt, C).

            text_mask:
                Optional boolean mask of shape (B, Lt).
                True means a valid text token.
                False means padding.

        Returns:
            x_output:
                Joint-attention output for image tokens, shape (B, Lx, C).

            text_output:
                Joint-attention output for text tokens, shape (B, Lt, C).
        """
        batch_size, image_len, _ = x.shape
        text_len = text.shape[1]

        x_q, x_k, x_v = self._make_qkv(x, self.x_qkv)
        text_q, text_k, text_v = self._make_qkv(text, self.text_qkv)

        x_q = self.x_q_norm(x_q)
        x_k = self.x_k_norm(x_k)
        text_q = self.text_q_norm(text_q)
        text_k = self.text_k_norm(text_k)

        # Sequence order is [text, image].
        q = torch.cat([text_q, x_q], dim=2)
        k = torch.cat([text_k, x_k], dim=2)
        v = torch.cat([text_v, x_v], dim=2)

        attention_bias = None

        if text_mask is not None:
            if text_mask.shape != (batch_size, text_len):
                raise ValueError(
                    f"text_mask must have shape {(batch_size, text_len)}, "
                    f"but received {tuple(text_mask.shape)}"
                )

            text_mask = text_mask.to(
                device=x.device,
                dtype=torch.bool,
            )

            image_mask = torch.ones(
                batch_size,
                image_len,
                device=x.device,
                dtype=torch.bool,
            )

            valid_key_mask = torch.cat(
                [text_mask, image_mask],
                dim=1,
            )

            # Additive mask broadcasts over heads and query positions:
            # (B, 1, 1, Lt + Lx).
            attention_bias = torch.zeros(
                batch_size,
                1,
                1,
                text_len + image_len,
                device=x.device,
                dtype=q.dtype,
            )
            attention_bias.masked_fill_(
                ~valid_key_mask[:, None, None, :],
                torch.finfo(q.dtype).min,
            )

        if self.fused_attn:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_bias,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attention = torch.matmul(
                q * self.scale,
                k.transpose(-2, -1),
            )

            if attention_bias is not None:
                attention = attention + attention_bias

            attention = attention.softmax(dim=-1)
            output = torch.matmul(attention, v)

        # (B, H, L, D) -> (B, L, C)
        output = output.transpose(1, 2).reshape(
            batch_size,
            text_len + image_len,
            self.hidden_size,
        )

        text_output, x_output = output.split(
            [text_len, image_len],
            dim=1,
        )

        text_output = self.text_proj(text_output)
        x_output = self.x_proj(x_output)

        # Prevent padded text positions from accumulating residual updates.
        if text_mask is not None:
            text_output = text_output * text_mask.unsqueeze(-1).to(
                text_output.dtype
            )

        return x_output, text_output


class MMDiTBlock(nn.Module):
    """
    MMDiT block with:

      1. Separate image and text streams
      2. Joint image-text attention
      3. Separate image and text MLPs
      4. adaLN-Zero conditioning for both streams

    Expected shapes:
        x:    (B, Lx, C)
        text: (B, Lt, C)
        c:    (B, C)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        update_text: bool = True,
        **block_kwargs,
    ):
        super().__init__()

        self.update_text = update_text

        qk_norm = block_kwargs.get("qk_norm", False)
        fused_attn = block_kwargs.get("fused_attn", True)

        # Image-stream normalization.
        self.x_norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.x_norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )

        # Text-stream normalization.
        self.text_norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.text_norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.attn = MMDiTJointAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.x_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0,
        )

        self.text_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0,
        )

        # Each stream receives:
        #   shift_attn, scale_attn, gate_attn,
        #   shift_mlp,  scale_mlp,  gate_mlp
        self.x_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.text_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.initialize_adaLN_zero()

    def initialize_adaLN_zero(self) -> None:
        # Initially make all residual branches contribute zero.
        nn.init.zeros_(self.x_adaLN_modulation[-1].weight)
        nn.init.zeros_(self.x_adaLN_modulation[-1].bias)

        nn.init.zeros_(self.text_adaLN_modulation[-1].weight)
        nn.init.zeros_(self.text_adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        c: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
                Image tokens, shape (B, Lx, C).

            text:
                Text tokens, shape (B, Lt, C).

            c:
                Global conditioning, usually timestep embedding,
                shape (B, C).

            text_mask:
                Optional shape (B, Lt), where True means valid and
                False means padding.

        Returns:
            Updated image and text streams.
        """
        (
            x_shift_attn,
            x_scale_attn,
            x_gate_attn,
            x_shift_mlp,
            x_scale_mlp,
            x_gate_mlp,
        ) = self.x_adaLN_modulation(c).chunk(6, dim=-1)

        (
            text_shift_attn,
            text_scale_attn,
            text_gate_attn,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = self.text_adaLN_modulation(c).chunk(6, dim=-1)

        x_attn_input = modulate(
            self.x_norm1(x),
            x_shift_attn,
            x_scale_attn,
        )

        text_attn_input = modulate(
            self.text_norm1(text),
            text_shift_attn,
            text_scale_attn,
        )

        x_attn_output, text_attn_output = self.attn(
            x=x_attn_input,
            text=text_attn_input,
            text_mask=text_mask,
        )

        x = x + x_gate_attn.unsqueeze(1) * x_attn_output

        if self.update_text:
            text = (
                text
                + text_gate_attn.unsqueeze(1) * text_attn_output
            )

        x = x + x_gate_mlp.unsqueeze(1) * self.x_mlp(
            modulate(
                self.x_norm2(x),
                x_shift_mlp,
                x_scale_mlp,
            )
        )

        if self.update_text:
            text = text + text_gate_mlp.unsqueeze(1) * self.text_mlp(
                modulate(
                    self.text_norm2(text),
                    text_shift_mlp,
                    text_scale_mlp,
                )
            )

        if text_mask is not None:
            # Optional: keep padded token states exactly zero.
            text = text * text_mask.unsqueeze(-1).to(text.dtype)

        return x, text