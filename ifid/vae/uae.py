from ifid.vae.unified_ae.uae import UAE
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import os

def zigzag_indices(n=16, device=None):
    rows, cols = [], []

    for s in range(2 * n - 1):
        if s % 2 == 0:
            # up-right
            r_start = min(s, n - 1)
            r_end = max(-1, s - n)
            for r in range(r_start, r_end, -1):
                c = s - r
                rows.append(r)
                cols.append(c)
        else:
            # down-left
            c_start = min(s, n - 1)
            c_end = max(-1, s - n)
            for c in range(c_start, c_end, -1):
                r = s - c
                rows.append(r)
                cols.append(c)

    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(cols, dtype=torch.long, device=device),
    )

def forward_zigzag(z):
    """
    z: (B, C, 16, 16)
    returns: (B, C, 256)
    """
    assert z.ndim == 4
    assert z.shape[-2:] == (16, 16)

    rows, cols = zigzag_indices(16, device=z.device)
    return z[:, :, rows, cols]

def reverse_zigzag(v):
    """
    v: (B, C, 256)
    returns: (B, C, 16, 16)
    """
    assert v.ndim == 3
    assert v.shape[-1] == 256

    B, C, _ = v.shape
    out = torch.empty((B, C, 16, 16), dtype=v.dtype, device=v.device)

    rows, cols = zigzag_indices(16, device=v.device)
    out[:, :, rows, cols] = v

    return out

class UAEVAE(nn.Module):
    def __init__(self, ckpt_path, uae_config, mode="pixel"):
        super().__init__()
        if not os.path.isfile(ckpt_path):
            a, b, filename = ckpt_path.split("/", 2)
            repo_id = f"{a}/{b}"
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.uae = UAE(**uae_config)
        self.mode = mode
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)['model']
        missing_keys, unexpected_keys = self.uae.load_state_dict(ckpt)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    def encode(self, x):
        x = (x + 1.0) / 2.0
        z, z_freq = self.uae.encode(x, return_frequency_tokens=True)
        if self.mode == "pixel":
            return z # 2D
        elif self.mode == "frequency":
            z_freq_1d = forward_zigzag(z_freq)
            z_freq_1d = z_freq_1d.permute(0, 2, 1) # b, l, c
            return z_freq_1d
        else:
            raise ValueError

    def decode(self, z):
        if self.mode == "pixel":
            z = z
        elif self.mode == "frequency":
            z = z.permute(0, 2, 1)
            z = reverse_zigzag(z)
            z = self.uae._idct2(z)
        else:
            raise ValueError
        xhat = self.uae.decode(z)
        xhat = xhat * 2.0 - 1.0
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("./configs/UAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
