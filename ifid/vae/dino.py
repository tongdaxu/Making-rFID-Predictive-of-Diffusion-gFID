import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from huggingface_hub import hf_hub_download

import INN
from INN.CouplingModels.NICEModel.conv import Conv2dNICE
from ifid.vae.autoencoder import Decoder

import INN.INNAbstract as INNAbstract
from INN.CouplingModels.conv import CouplingConv
from INN.CouplingModels.utils import _default_2d_coupling_function

def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from keys if present."""
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict  # nothing to do
    return {k[len("module."):]: v for k, v in state_dict.items()}

class ConvNICEfast(CouplingConv):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(ConvNICEfast, self).__init__(num_feature=channels, mask=mask)
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = x2 + self.m1(x1)
        x1 = x1 + self.m2(x2)
        return torch.cat([x1, x2], dim=1)
    
    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        y1 = y1 - self.m2(y2)
        y2 = y2 - self.m1(y1)        
        return torch.cat([y1, y2], dim=1)    

    def logdet(self, **args):
        return 0


class Conv2dNICEFast(ConvNICEfast):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(Conv2dNICEFast, self).__init__(channels, kernel_size, w=w, activation_fn=activation_fn, m=m, mask=mask)
        self.kernel_size = kernel_size
        self.m1 = _default_2d_coupling_function(channels // 2, kernel_size, activation_fn, w=w)
        self.m2 = _default_2d_coupling_function(channels // 2, kernel_size, activation_fn, w=w)

    def forward(self, x, log_p0=0, log_det_J=0):
        y = super(Conv2dNICEFast, self).forward(x)
        if self.compute_p:
            return y, log_p0, log_det_J + self.logdet()
        else:
            return y
    
    def inverse(self, y, **args):
        x = super(Conv2dNICEFast, self).inverse(y)
        return x
    
    def __repr__(self):
        return f'Conv2dNICE(channels={self.num_feature}, kernel_size={self.kernel_size})'


class ConvNICEfast2(CouplingConv):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(ConvNICEfast2, self).__init__(num_feature=channels, mask=mask)
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = x2 + self.m(x1)
        x1 = x1 + self.m(x2)
        return torch.cat([x1, x2], dim=1)
    
    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        y1 = y1 - self.m(y2)
        y2 = y2 - self.m(y1)        
        return torch.cat([y1, y2], dim=1)    

    def logdet(self, **args):
        return 0

class Conv2dNICEFast2(ConvNICEfast2):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(Conv2dNICEFast2, self).__init__(channels, kernel_size, w=w, activation_fn=activation_fn, m=m, mask=mask)
        self.kernel_size = kernel_size
        self.m = _default_2d_coupling_function(channels // 2, kernel_size, activation_fn, w=w)

    def forward(self, x, log_p0=0, log_det_J=0):
        y = super(Conv2dNICEFast2, self).forward(x)
        if self.compute_p:
            return y, log_p0, log_det_J + self.logdet()
        else:
            return y
    
    def inverse(self, y, **args):
        x = super(Conv2dNICEFast2, self).inverse(y)
        return x
    
    def __repr__(self):
        return f'Conv2dNICE(channels={self.num_feature}, kernel_size={self.kernel_size})'


class ChannelShuffle(INNAbstract.INNModule):
    def __init__(self):
        INNAbstract.INNModule.__init__(self)
    
    def forward(self, x, log_p0=0, log_det_J=0):
        b, c, h, w = x.shape
        x1, x2 = x[:,::2], x[:,1::2]
        xs = torch.cat([x1, x2], dim=1)
        if self.compute_p:
            return xs, log_p0, log_det_J
        else:
            return xs

    def inverse(self, xs, **args):
        b, c, h, w = xs.shape
        x1, x2 = xs[:, :c//2], xs[:, c//2:]
        x = torch.zeros_like(xs)
        x[:,::2], x[:,1::2] = x1, x2
        return x

class FlowVAEBKB(nn.Module):
    def __init__(self, size="small", *args, **kwargs):
        super().__init__()
        self.size = size
        if self.size == "small":
            self.flow = INN.Sequential(
                Conv2dNICE(3, 3), # 16
                Conv2dNICE(3, 3), # 16
                INN.PixelShuffle2d(2), # 8
                Conv2dNICE(12, 3),
                Conv2dNICE(12, 3),
                INN.PixelShuffle2d(2), # 4
                Conv2dNICE(48, 3),
                Conv2dNICE(48, 3),
                INN.PixelShuffle2d(2), # 2
                Conv2dNICE(192, 3),
                Conv2dNICE(192, 3),
                INN.PixelShuffle2d(2), # 1
                Conv2dNICE(768, 3, w=1),
                Conv2dNICE(768, 3, w=1),
            )
        elif size == "medium":
            self.flow = INN.Sequential(
                Conv2dNICE(3, 3), # 16
                Conv2dNICE(3, 3), # 16
                INN.PixelShuffle2d(2), # 8
                Conv2dNICE(12, 3),
                Conv2dNICE(12, 3),
                INN.PixelShuffle2d(2), # 4
                Conv2dNICE(48, 3),
                Conv2dNICE(48, 3),
                INN.PixelShuffle2d(2), # 2
                Conv2dNICE(192, 3),
                Conv2dNICE(192, 3),
                INN.PixelShuffle2d(2), # 1
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
            )
        elif size == "large":
            self.flow = INN.Sequential(
                Conv2dNICE(3, 3), # 16
                Conv2dNICE(3, 3), # 16
                INN.PixelShuffle2d(2), # 8
                Conv2dNICE(12, 3),
                Conv2dNICE(12, 3),
                INN.PixelShuffle2d(2), # 4
                Conv2dNICE(48, 3),
                Conv2dNICE(48, 3),
                INN.PixelShuffle2d(2), # 2
                Conv2dNICE(192, 3),
                Conv2dNICE(192, 3),
                INN.PixelShuffle2d(2), # 1
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
                Conv2dNICE(768, 3, w=2),
            )
        else:
            raise NotImplementedError
        self.flow.computing_p(True)
        self.flow_lam = 1

class FlowVAEFastBKB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.flow = INN.Sequential(
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            INN.PixelShuffle2d(2), # 8
            Conv2dNICEFast(12, 3),
            Conv2dNICEFast(12, 3),
            INN.PixelShuffle2d(2), # 4
            Conv2dNICEFast(48, 3),
            Conv2dNICEFast(48, 3),
            INN.PixelShuffle2d(2), # 2
            Conv2dNICEFast(192, 3),
            Conv2dNICEFast(192, 3),
            INN.PixelShuffle2d(2), # 1
            Conv2dNICEFast(768, 3, w=1),
            Conv2dNICEFast(768, 3, w=1),
        )
        self.flow.computing_p(True)
        self.flow_lam = 1

class FlowVAEFast2BKB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.flow = INN.Sequential(
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            INN.PixelShuffle2d(2), # 8
            Conv2dNICEFast2(12, 3),
            Conv2dNICEFast2(12, 3),
            INN.PixelShuffle2d(2), # 4
            Conv2dNICEFast2(48, 3),
            Conv2dNICEFast2(48, 3),
            INN.PixelShuffle2d(2), # 2
            Conv2dNICEFast2(192, 3),
            Conv2dNICEFast2(192, 3),
            INN.PixelShuffle2d(2), # 1
            Conv2dNICEFast2(768, 3),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            Conv2dNICEFast2(768, 3),
        )
        self.flow.computing_p(True)
        self.flow_lam = 1

class FlowVAEFast2LargerBKB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.flow = INN.Sequential(
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            INN.PixelShuffle2d(2), # 8
            Conv2dNICEFast2(12, 3),
            Conv2dNICEFast2(12, 3),
            INN.PixelShuffle2d(2), # 4
            Conv2dNICEFast2(48, 3),
            Conv2dNICEFast2(48, 3),
            INN.PixelShuffle2d(2), # 2
            Conv2dNICEFast2(192, 3),
            Conv2dNICEFast2(192, 3),
            INN.PixelShuffle2d(2), # 1
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
        )
        self.flow.computing_p(True)
        self.flow_lam = 1

class FlowVAEFast2XLargerBKB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.flow = INN.Sequential(
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            INN.PixelShuffle2d(2), # 8
            Conv2dNICEFast2(12, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(12, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(12, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(12, 3),
            INN.PixelShuffle2d(2), # 4
            Conv2dNICEFast2(48, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(48, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(48, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(48, 3),
            INN.PixelShuffle2d(2), # 2
            Conv2dNICEFast2(192, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(192, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(192, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(192, 3),
            INN.PixelShuffle2d(2), # 1
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
            ChannelShuffle(),
            Conv2dNICEFast2(768, 3),
        )
        self.flow.computing_p(True)
        self.flow_lam = 1

def unnormalize(tensor, mean, std):
    """
    tensor: (C, H, W) or (B, C, H, W)
    mean, std: list or tuple of length C
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)

    if tensor.dim() == 4:  # batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean

class DINOFlowVAE(nn.Module):
    def __init__(self, ckpt_path, size="small", *args, **kwargs):
        super().__init__()
        self.backbone = FlowVAEBKB(size=size)
        self.decoder = Decoder(out_ch=3, ch_mult=[1, 1, 2, 4, 4], num_res_blocks=2, z_channels=768, mid_attn=False)
        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "teacher" in vae_ckpt.keys():
            vae_ckpt = vae_ckpt["teacher"]
        elif "vae" in vae_ckpt.keys():
            vae_ckpt = vae_ckpt["vae"]
        else:
            assert(0)
        vae_ckpt = strip_module_prefix(vae_ckpt)
        miss_keys, unexp_keys = self.load_state_dict(vae_ckpt, strict=False)
        print("missing: ", miss_keys)
        print("unexp_keys: ", unexp_keys)

    def encode(self, x, *args, **kwargs):
        # use dino
        x = (x + 1.0) / 2.0
        x = torchvision.transforms.functional.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        z = self.backbone.flow(x)[0]
        return z

    def decode(self, z, inv=False, *args, **kwargs):
        x = self.decoder(z)
        with torch.no_grad():
            xinv = self.backbone.flow.inverse(z)
            xinv = unnormalize(xinv, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            xinv = xinv * 2.0 - 1.0
        if inv:
            return xinv
        else:
            return x

    def forward(self, x, *args, **kwargs):
        with torch.no_grad():
            z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}


class DINOFlowVAERes(nn.Module):
    def __init__(self, ckpt_path, size="small", debug=False, *args, **kwargs):
        super().__init__()
        self.backbone = FlowVAEBKB(size=size)
        self.decoder = Decoder(out_ch=3, ch_mult=[1, 1, 2, 4, 4], num_res_blocks=2, z_channels=768, mid_attn=False)
        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "teacher" in vae_ckpt.keys():
            vae_ckpt = vae_ckpt["teacher"]
        elif "vae" in vae_ckpt.keys():
            vae_ckpt = vae_ckpt["vae"]
        else:
            assert(0)
        vae_ckpt = strip_module_prefix(vae_ckpt)
        miss_keys, unexp_keys = self.load_state_dict(vae_ckpt, strict=False)
        print("missing: ", miss_keys)
        print("unexp_keys: ", unexp_keys)

        self.downrate = 16
        self.embed_dim = 768
        self.debug = debug

    def encode(self, x, *args, **kwargs):
        # use dino
        x_ori = x
        x = (x + 1.0) / 2.0
        x = torchvision.transforms.functional.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        z = self.backbone.flow(x)[0]
        xhat = self.decoder(z)
        res = x_ori - xhat
        res = F.pixel_unshuffle(res, self.downrate)
        if self.debug:
            res = torch.zeros_like(res)  # debug purpose only
        return torch.cat([z, res], dim=1)

    def decode(self, z, *args, **kwargs):
        z, res = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res = F.pixel_shuffle(res, self.downrate)
        xhat = self.decoder(z)
        if self.debug:
            res = torch.zeros_like(res)  # debug purpose only
        x = xhat + res
        return x

    def forward(self, x, *args, **kwargs):
        with torch.no_grad():
            z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}

class PixelUnshuffle2d(INNAbstract.PixelShuffleModule):
    def __init__(self, r):
        super(PixelUnshuffle2d, self).__init__()
        self.r = r
        self.shuffle = nn.PixelShuffle(r)
        self.unshuffle = nn.PixelUnshuffle(r)
    
    def PixelShuffle(self, x):
        return self.unshuffle(x)
    
    def PixelUnshuffle(self, x):
        return self.shuffle(x)

class DINOFlowVAEFast(nn.Module):
    def __init__(self, ckpt_path, size="base", *args, **kwargs):
        super().__init__()
        if size == "xlarger":
            self.backbone = FlowVAEFast2XLargerBKB()
        elif size == "larger":
            self.backbone = FlowVAEFast2LargerBKB()
        elif size == "large":
            self.backbone = FlowVAEFast2BKB()
        elif size == "base":
            self.backbone = FlowVAEFastBKB()
        else:
            raise ValueError(f"Unknown size: {size}")
        self.decoder = Decoder(out_ch=3, ch_mult=[1, 1, 2, 4, 8], num_res_blocks=2, z_channels=768, mid_attn=False)

        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "teacher" in vae_ckpt.keys():
            vae_ckpt = vae_ckpt["teacher"]
        elif "vae" in vae_ckpt.keys():
            vae_ckpt = vae_ckpt["vae"]
        else:
            assert(0)
        vae_ckpt = strip_module_prefix(vae_ckpt)
        miss_keys, unexp_keys = self.load_state_dict(vae_ckpt, strict=False)
        print("missing: ", miss_keys)
        print("unexp_keys: ", unexp_keys)

    def encode(self, x, *args, **kwargs):
        # use dino
        z = self.backbone.flow(x)[0]
        return z

    def decode(self, z, inv=False, *args, **kwargs):
        if inv:
            return self.backbone.flow.inverse(z)
        else:
            return self.decoder(z)

    def forward(self, x, *args, **kwargs):
        with torch.no_grad():
            z = self.encode(x)
        xhat = self.decode(z)
        return z, {"xhat": xhat}

class DINOFlowVAEFastRes(nn.Module):
    def __init__(self, ckpt_path, size="base", *args, **kwargs):
        super().__init__()
        if size == "larger":
            self.backbone = FlowVAEFast2LargerBKB()
        elif size == "large":
            self.backbone = FlowVAEFast2BKB()
        elif size == "base":
            self.backbone = FlowVAEFastBKB()
        else:
            raise ValueError(f"Unknown size: {size}")
        self.decoder = Decoder(out_ch=3, ch_mult=[1, 1, 2, 4, 8], num_res_blocks=2, z_channels=768, mid_attn=False)

        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "teacher" in vae_ckpt.keys():
            # dino checkpoint
            vae_ckpt = vae_ckpt["teacher"]
        elif "vae" in vae_ckpt.keys():
            # train_decoder.py checkpoint
            vae_ckpt = vae_ckpt["vae"]
        else:
            assert(0)
        vae_ckpt = strip_module_prefix(vae_ckpt)
        miss_keys, unexp_keys = self.load_state_dict(vae_ckpt, strict=False)
        print("missing: ", miss_keys)
        print("unexp_keys: ", unexp_keys)

        self.downrate = 16
        self.embed_dim = 768

    def encode(self, x, *args, **kwargs):
        # use dino
        z = self.backbone.flow(x)[0]
        xhat = self.decoder(z)
        res = x - xhat
        res = F.pixel_unshuffle(res, self.downrate)
        return torch.cat([z, res], dim=1)

    def decode(self, z, *args, **kwargs):
        z, res = z[:, :self.embed_dim], z[:, self.embed_dim:]
        res = F.pixel_shuffle(res, self.downrate)
        xhat = self.decoder(z)
        x = xhat + res
        return x

    def forward(self, x, *args, **kwargs):
        with torch.no_grad():
            z = self.encode(x)
        return z, {"xhat": x}


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("/mnt/task_runtime/configs/DINOFLOWVAEFAST2LRes.yaml")
    vae = instantiate_from_config(configs)
    x = torch.randn([1, 3, 256, 256])
    z = vae.encode(x)
    xhat = vae.decode(z)
    print(z.shape, xhat.shape, torch.mean((x - xhat) ** 2))
