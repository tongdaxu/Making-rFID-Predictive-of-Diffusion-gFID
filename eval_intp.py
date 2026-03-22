from cont.modelling.tokenizer import FlowModel, ModelArgs
from PIL import Image
import torch
from torchvision import transforms

from PIL import Image
import torch
import numpy as np
from ifid.vae.utils import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn.functional as F

def slerp(z1, z2, t, eps=1e-7):
    """
    z1, z2: (B, C, H, W)
    t: float in [0, 1]
    """
    B = z1.shape[0]

    # Flatten per sample
    z1f = z1.reshape(B, -1)
    z2f = z2.reshape(B, -1)

    # Normalize
    z1n = z1f / (z1f.norm(dim=1, keepdim=True) + eps)
    z2n = z2f / (z2f.norm(dim=1, keepdim=True) + eps)

    # Angle between vectors
    dot = (z1n * z2n).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # SLERP weights
    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w2 = torch.sin(t * theta) / (sin_theta + eps)

    # Interpolate and reshape
    zt = w1.unsqueeze(1) * z1f + w2.unsqueeze(1) * z2f

    small_t = sin_theta < 1e-6
    if small_t.any():
        zt[small_t] = (1 - t) * z1f[small_t] + t * z2f[small_t]

    return zt.reshape_as(z1)

def read_image_tensor(path):
    """
    Read image and return tensor of shape (1,3,256,256) in range [-1,1]
    """
    img = Image.open(path).convert("RGB").resize((256, 256))
    x = torch.from_numpy(np.array(img)).float() / 255.0  # [0,1]
    x = x.permute(2, 0, 1)                               # (3,H,W)
    x = x * 2 - 1                                         # [-1,1]
    x = x.unsqueeze(0)                                    # (1,3,256,256)
    return x


def save_tensor_image(x, path):
    """
    Save tensor of shape (1,3,256,256) with range [-1,1] to PNG
    """
    x = x.detach().cpu()

    if x.dim() == 4:
        x = x[0]

    x = (x + 1) / 2                       # [-1,1] -> [0,1]
    x = x.clamp(0, 1)

    x = (x * 255).byte()
    x = x.permute(1, 2, 0).numpy()        # (H,W,3)

    img = Image.fromarray(x)
    img.save(path)

if __name__ == "__main__":
    img1_path = "/video_ssd/kongzishang/xutongda/git/IFID/samples/ifid-transformer-cont_tokenizer_extra_pclass/nn_dist/ILSVRC2012_val_00000236.JPEG"
    img2_path = "/video_ssd/kongzishang/xutongda/git/IFID/samples/ifid-transformer-cont_tokenizer_extra_pclass/src_dist/ILSVRC2012_val_00000236.JPEG" 

    vae_config = OmegaConf.load("./configs/PIXELVAE.yaml")
    vae = instantiate_from_config(vae_config)
    vae.eval()
    vae.cuda()

    with torch.no_grad():
        # inverse
        x1 = read_image_tensor(img1_path).cuda()
        x2 = read_image_tensor(img2_path).cuda()

        z1 = vae.encode(x1, y=1000)
        z2 = vae.encode(x2, y=1000)

        print(torch.std(z1), torch.std(z2))

        # zintp = (z1 + z2) / 2.0
        zintp = slerp(z1, z2, 0.5)
        xhat = vae.decode(zintp, y=1000)

        save_tensor_image(xhat, "demo.png")

        save_tensor_image(F.pixel_shuffle(torch.clamp(z1, -1, 1), 16), "z1.png")
        save_tensor_image(F.pixel_shuffle(torch.clamp(z2, -1, 1), 16), "z2.png")

