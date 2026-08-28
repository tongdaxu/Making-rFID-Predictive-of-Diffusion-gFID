import torch.nn as nn
import torch
from ifid.vae.autoencoder import Encoder, Decoder

class UnitSphereRegularizer(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def get_trainable_parameters(self):
        yield from ()
    
    def normalize(self, z):
        return z / (torch.norm(z, dim=1, keepdim=True) + self.eps)

    def forward(self, z: torch.Tensor):
        z = self.normalize(z)
        return z


class GAE(nn.Module):
    def __init__(self, ckpt_path, *args, **kwargs):
        super().__init__()
        self.regularizer = UnitSphereRegularizer()
        self.encoder = Encoder(
            ch=128,
            out_ch=3,
            in_channels=3,
            ch_mult=(1,2,4,8,8),
            num_res_blocks=2,
            attn_resolutions=(),
            mid_attn=False,
            z_channels=64,
            double_z=False,
        )
        self.decoder = Decoder(
            ch=128,
            out_ch=3,
            in_channels=3,
            ch_mult=(1,2,4,8,8),
            num_res_blocks=2,
            attn_resolutions=(),
            mid_attn=False,
            z_channels=64,
        )
        if ckpt_path is not None:
            miss_keys, unexp_keys = self.load_state_dict(torch.load(ckpt_path, map_location='cpu')["state_dict"], strict=False)
            print("missing: ", miss_keys)
            print("unexp_keys: ", unexp_keys)

    def forward(self, x, *args, **kwargs):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return z, {"xhat": xhat}

    def encode(self, x, *args, **kwargs):
        z = self.encoder(x)
        z = self.regularizer(z)
        return z

    def decode(self, z, *args, **kwargs):
        return self.decoder(z)
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config
    import torch
    from torchvision import transforms
    from PIL import Image

    def load_jpeg_to_tensor(image_path):
        """
        Load a JPEG image into a (1, 3, 256, 256) tensor with values in [-1, 1].
        
        Args:
            image_path (str): Path to the JPEG image
            
        Returns:
            torch.Tensor: Shape (1, 3, 256, 256) with values in [-1, 1]
        """
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Maps to [-1, 1]
        ])
        
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        
        return tensor

    configs = OmegaConf.load("/mnt/task_runtime/configs/GAE.yaml")
    vae = instantiate_from_config(configs)
    x = load_jpeg_to_tensor("/root/data/trove/imagenet-1.0.0/data/raw/validation/n01440764/ILSVRC2012_val_00000293.JPEG")
    z = vae.encode(x)
    xhat = vae.decode(z)
    print(z.shape, xhat.shape, torch.mean((x - xhat) ** 2))
