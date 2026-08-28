from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import dataclasses
from copy import deepcopy

import torch
from torch.cuda.amp import autocast

class TextEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_length=256):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(model_name).eval()

        self.feature_dim = self.text_model.config.hidden_size

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.text_model.parameters()).device

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.text_model(
            **tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=False,
        )
        return {
            "tokens": outputs[1][-1],
            "attention_mask": tokens["attention_mask"],
        }
    
def apply_cfg_dropout(model_conds, model_conds_null, cfg_dropout_prob=0.1):
    mask = torch.rand(model_conds['context'].shape[0], device=model_conds['context'].device) < cfg_dropout_prob
    return {
        k: torch.where(mask.view(-1, *([1]*(v.ndim-1))), model_conds_null[k], v) if v is not None else None
        for k, v in model_conds.items()
    }, mask


def get_null_cond(text_encoder,conditioning_type, num_classes, batch_size, device):
    if conditioning_type == "text":
        _null_context, _null_attn_mask = encode_text(text_encoder, [""])
    else:
        _null_context, _null_attn_mask = torch.tensor([num_classes], device=device), None
    rtn = dict(context=_null_context, attn_mask=_null_attn_mask)
    rtn = {k: v.expand(batch_size, *v.shape[1:]) if v is not None else None for k, v in rtn.items()}
    return rtn


def setup_text_encoder(config, device):
    """Build text encoder if conditioning.type == 'text', else return None.
    Sets config.conditioning.text_feature_dim and context_dim from the encoder.
    """
    if config.get("type") != "text":
        return None
    text_encoder = TextEncoder(**config.get("text_encoder")).to(device)
    config["context_dim"] = text_encoder.feature_dim
    return text_encoder


def encode_text(text_encoder, y):
    """Encode text conditions. Returns (encoder_hidden_states, encoder_attention_mask)."""
    with torch.no_grad():
        enc_out = text_encoder(y)
        return enc_out["tokens"], enc_out["attention_mask"]


def get_fixed_viz_batch_conditions(viz_fixed, y, condition_type, text_encoder, device):
    """Get fixed conditions for the first batch for consistent visualization."""
    if viz_fixed['context'] is not None:
        return viz_fixed
    n = viz_fixed['zs'].shape[0]
    if condition_type == "label":
        viz_fixed['context'] = y[:n].clone().to(device)
    else:
        with torch.no_grad():
            enc_out = text_encoder(y[:n])
            viz_fixed['context'] = enc_out["tokens"]
            viz_fixed['attn_mask'] = enc_out["attention_mask"]
    return viz_fixed