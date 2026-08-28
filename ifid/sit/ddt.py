import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed



def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class RoPE(nn.Module):
    def __init__(self, dim, vis_len, cond_len=0, extra_tokens=0, theta=10000.,):
        super().__init__()
        # 2D RoPE for vision
        d, T = dim // 2, int(vis_len ** 0.5)
        vis_freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))  # [D//4]
        vis_base_angles = torch.outer(torch.arange(T).float(), vis_freqs)  # [T, D//4]
        vis_angles = torch.cat([
            vis_base_angles[:, None].expand(-1, T, -1),
            vis_base_angles[None, :].expand(T, -1, -1)
        ], dim=-1).reshape(vis_len, d)  # [T, T, D//2] -> [L', D//2]
        # no PE for extra (cls) or cond tokens
        extra_angles = torch.zeros(extra_tokens, dim // 2)
        cond_angles = torch.zeros(cond_len, dim // 2)
        angles = torch.cat([extra_angles, vis_angles, cond_angles], dim=0).repeat_interleave(2, dim=-1)  # [L, D]
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class NormAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads, self.dim, self.head_dim = num_heads, dim, dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope, attn_mask=None):
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rope(q), rope(k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        return self.proj(out)


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size, n_tokens=4, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size)))

    def forward(self, t, return_base_embed=False):
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        if return_base_embed:
            t_embed = t_embed.unsqueeze(1)
            return t_embed, self.learnable_tokens + t_embed
        else:
            return self.learnable_tokens + t_embed.unsqueeze(1)


class ConditionEmbedder(nn.Module):
    def __init__(self, hidden_size, num_classes=1000, context_dim=768, condition_type="label", n_tokens=8):
        super().__init__()
        self.condition_type = condition_type
        self.hidden_size = hidden_size

        if condition_type == "label":
            self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
            self.learnable_tokens = nn.Parameter(torch.normal(0, 1 / hidden_size**0.5, (n_tokens, hidden_size)))
        elif condition_type == "text":
            self.norm = RMSNorm(context_dim)
            self.proj = nn.Linear(context_dim, hidden_size)
        else:
            raise ValueError(f"Unknown condition_type: {condition_type}")

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.condition_type == "label":
            return self.learnable_tokens + self.embedding_table(y).unsqueeze(1)
        else:
            return self.proj(self.norm(y))



def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DDTEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * hidden_size * mlp_ratio))

    def forward(self, x, rope, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DDTDecoderBlock(DDTEncoderBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(hidden_size, num_heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6*hidden_size)
        )

    def forward(self, x, c, rope, attn_mask=None):
        modulation = self.adaln_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope, attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, cls_dim=None):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
        if cls_dim is not None:
            self.cls_linear = nn.Linear(hidden_size, cls_dim)

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        if hasattr(self, 'cls_linear'):
            cls_pred = self.cls_linear(x[:, 0, :])
            return self.linear(x[:, 1:, :]), cls_pred
        return self.linear(x)


class DiTwDDTHead(nn.Module):
    def __init__(
        self,
        input_size=16,
        in_channels=768,
        patch_size=[1, 1],
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        mlp_ratio=4.0,
        enable_repa=False,
        repa_layer_depth=8,
        z_dim=None,
        enable_reg=False,
        num_classes=1000,
        condition_type="label",
        context_dim=768,
        cond_arch=None,
        is_meanflow=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.enc_hidden_size, dec_hidden_size = hidden_size
        self.num_enc_blocks, self.num_dec_blocks = depth
        self.s_patch_size, self.x_patch_size = patch_size
        enc_num_heads, dec_num_heads = num_heads

        self.repa_layer_depth = repa_layer_depth
        self.enable_reg = enable_reg
        self.is_meanflow = is_meanflow

        self.s_embedder = PatchEmbed(input_size, self.s_patch_size, in_channels, self.enc_hidden_size)
        self.x_embedder = PatchEmbed(input_size, self.x_patch_size, in_channels, dec_hidden_size)
        self.s_projector = nn.Linear(self.enc_hidden_size, dec_hidden_size) if self.enc_hidden_size != dec_hidden_size else nn.Identity()

        # MeanFlow conditions on two times (t, t-r); add a second time embedder + tokens for absolute t
        self.num_cond_tokens = cond_arch.num_t_tokens * (2 if is_meanflow else 1) + cond_arch.num_c_tokens
        self.t_embedder = GaussianFourierEmbedding(self.enc_hidden_size, cond_arch.num_t_tokens)
        if is_meanflow:
            self.t_abs_embedder = GaussianFourierEmbedding(self.enc_hidden_size, cond_arch.num_t_tokens)
        self.ctx_embedder = ConditionEmbedder(
            self.enc_hidden_size, num_classes, context_dim, condition_type, cond_arch.num_c_tokens
        )

        self.blocks = []
        for _ in range(self.num_enc_blocks):
            self.blocks.append(DDTEncoderBlock(self.enc_hidden_size, enc_num_heads, mlp_ratio))
        for _ in range(self.num_dec_blocks):
            self.blocks.append(DDTDecoderBlock(dec_hidden_size, dec_num_heads, mlp_ratio))
        self.blocks = nn.ModuleList(self.blocks)

        self.final_layer = DDTFinalLayer(dec_hidden_size, self.x_patch_size, in_channels, cls_dim=z_dim if enable_reg else None)
        self.enc_rope = RoPE(self.enc_hidden_size // enc_num_heads, self.s_embedder.num_patches, self.num_cond_tokens, extra_tokens=int(enable_reg))
        self.dec_rope = RoPE(dec_hidden_size // dec_num_heads, self.x_embedder.num_patches, extra_tokens=int(enable_reg))
        if enable_repa:
            self.repa_projector = nn.Linear(self.enc_hidden_size, z_dim)
        if enable_reg:
            self.cls_in_proj_enc = nn.Linear(z_dim, self.enc_hidden_size)
            self.cls_in_norm_enc = RMSNorm(self.enc_hidden_size)
            self.cls_in_proj_dec = nn.Linear(z_dim, dec_hidden_size)
            self.cls_in_norm_dec = RMSNorm(dec_hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Patch embedders
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Condition embedders
        if hasattr(self.ctx_embedder, "mlp"):
            nn.init.normal_(self.ctx_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ctx_embedder.mlp[2].weight, std=0.02)
        if hasattr(self.ctx_embedder, "embedding_table"):
            nn.init.normal_(self.ctx_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            if hasattr(block, "adaln_modulation"):
                nn.init.constant_(block.adaln_modulation[-1].weight, 0)
                nn.init.constant_(block.adaln_modulation[-1].bias, 0)

        # Timestep embedding MLP
        for t_embedder in ("t_embedder", "t_abs_embedder"):
            if hasattr(self, t_embedder):
                nn.init.normal_(getattr(self, t_embedder).mlp[0].weight, std=0.02)
                nn.init.normal_(getattr(self, t_embedder).mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if hasattr(self.final_layer, "cls_linear"):
            nn.init.constant_(self.final_layer.cls_linear.weight, 0)
            nn.init.constant_(self.final_layer.cls_linear.bias, 0)

    def unpatchify(self, x, p):
        """[N, T, patch_size**2 * C] -> [N, C, H, W]"""
        h, c = int(x.shape[1] ** 0.5), self.in_channels
        x = x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h*p, h*p)
        return x

    def _build_sequence(self, x, t, condition_kwargs):
        """Returns sequence concatenated with all condition tokens, and the base timestep embedding (no learnable tokens)"""
        seq = []
        if self.enable_reg:
            cls_in = self.cls_in_norm_enc(self.cls_in_proj_enc(condition_kwargs["cls_t"]))
            seq.append(cls_in.unsqueeze(1))
        seq.append(self.s_embedder(x))
        t_emb_base, t_emb = self.t_embedder(t, return_base_embed=True)
        seq.append(t_emb)
        if self.is_meanflow:
            seq.append(self.t_abs_embedder(condition_kwargs["t_abs"]))
        seq.append(self.ctx_embedder(condition_kwargs["context"]))
        seq = torch.cat(seq, dim=1)
        return seq, t_emb_base

    def _build_attn_mask(self, seq, condition_kwargs):
        # Create multiplicative mask template
        attn_mask = torch.ones((seq.shape[0], seq.shape[1]), device=seq.device)
        cond_mask = condition_kwargs.get("attn_mask")
        if cond_mask is not None:
            attn_mask[:, -cond_mask.shape[1]:] = cond_mask
        # Convert to additive mask
        attn_mask = (1.0 - attn_mask[:, None, None, :]) * torch.finfo(seq.dtype).min
        return attn_mask

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        s, n = int(self.enable_reg), self.s_embedder.num_patches
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(seq[:, :s + n, :])
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :s + n, :]))

        x = self.x_embedder(x)
        if self.enable_reg:
            cls_in = self.cls_in_norm_dec(self.cls_in_proj_dec(condition_kwargs["cls_t"]))
            x = torch.cat([cls_in.unsqueeze(1), x], dim=1)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        if self.enable_reg:
            x, cls_pred = self.final_layer(x, seq)
        else:
            x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)
        if self.enable_reg:
            x = (x, cls_pred)

        if return_intermediate:
            return x, zt_intermediate
        return x


class DiTwDDTHeadIG(DiTwDDTHead):
    def __init__(self, base_model_depth=8, **kwargs):
        super().__init__(**kwargs)
        self.base_model_depth = base_model_depth

        self.base_final_layer = DDTFinalLayer(self.enc_hidden_size, self.s_patch_size, self.in_channels)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.base_final_layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.base_final_layer.linear.weight, 0)
        nn.init.constant_(self.base_final_layer.linear.bias, 0)

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        x_base = None
        seq, t_emb_base = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(seq, condition_kwargs)
        s, n = int(self.enable_reg), self.s_embedder.num_patches
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(seq[:, :s + n, :])
            if (i + 1) == self.base_model_depth:
                x_base = seq[:, s:s + n, :]
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :s + n, :]))

        x = self.x_embedder(x)
        if self.enable_reg:
            cls_in = self.cls_in_norm_dec(self.cls_in_proj_dec(condition_kwargs["cls_t"]))
            x = torch.cat([cls_in.unsqueeze(1), x], dim=1)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)

        if self.enable_reg:
            x, cls_pred = self.final_layer(x, seq)
        else:
            x = self.final_layer(x, seq)
        x = self.unpatchify(x, self.x_patch_size)

        x_base = F.silu(t_emb_base + x_base)
        x_base = self.base_final_layer(x_base, x_base)
        x_base = self.unpatchify(x_base, self.s_patch_size)

        out = (x, x_base, cls_pred) if self.enable_reg else (x, x_base)
        if return_intermediate:
            return out, zt_intermediate
        return out

class TokenResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        num_output_tokens: int,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()

        self.latents = nn.Parameter(
            torch.randn(1, num_output_tokens, dim) * 0.02
        )

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    batch_first=True,
                ),
                "norm_q": nn.LayerNorm(dim),
                "norm_kv": nn.LayerNorm(dim),
                "norm_ff": nn.LayerNorm(dim),
                "ff": nn.Sequential(
                    nn.Linear(dim, 4 * dim),
                    nn.GELU(),
                    nn.Linear(4 * dim, dim),
                ),
            })
            for _ in range(num_layers)
        ])

    def forward(
        self,
        text_tokens: torch.Tensor,       # (B, L, C)
        padding_mask: torch.Tensor | None = None,  # (B, L), True = padding
    ) -> torch.Tensor:
        batch_size = text_tokens.shape[0]

        x = self.latents.expand(batch_size, -1, -1)  # (B, M, C)

        for layer in self.layers:
            q = layer["norm_q"](x)
            kv = layer["norm_kv"](text_tokens)

            attended, _ = layer["cross_attn"](
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=padding_mask,
                need_weights=False,
            )

            x = x + attended
            x = x + layer["ff"](layer["norm_ff"](x))

        return x