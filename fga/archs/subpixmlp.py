import math
import torch
from torch import nn as nn
import torch.nn.functional as F

from fga.archs.arch_util import MLP, conv_flops


class GetFourierFeatures(nn.Module):
    r"""Get Fourier Features.
    From: 
        Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
        
    Args:
        dim (int):    Number of input channels.
        depth (int):  Number of FC layer of the MLP module.
        scale (int):  Upscaling factor.
    """
    def __init__(self, dim, scale, depth):
        super(GetFourierFeatures, self).__init__()
        assert dim % 2 == 0, 'number of channels must be divisible by 2.'
        self.dim = dim
        self.scale = scale
        self.norm = nn.LayerNorm(dim)

        self.mlp = MLP(dim, dim, dim, num_layer=depth)

    def make_coord(self, shape, flatten=False):
        """ Make coordinates at grid centers.
        """
        v0, v1 = 0, 1
        coord_seqs = []
        for i, n in enumerate(shape):
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=0)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret


    def pixel_shuffle_coord(self, x):
        b, c, h, w = x.shape

        tar_h = h * self.scale
        tar_w = w * self.scale

        # Preparing coordinates
        upsampled_coord = self.make_coord((tar_h, tar_w)).to(x.device).unsqueeze(0)
        upsampled_coord -= F.interpolate(self.make_coord((h, w)).to(x.device).unsqueeze(0), scale_factor=self.scale, mode='nearest')
        upsampled_coord[:, 0, ...] *= h
        upsampled_coord[:, 1, ...] *= w
        scale = self.scale ** 2

        coord = F.pixel_unshuffle(upsampled_coord, self.scale)
        coord = coord.view(1, 2, scale, h, w).transpose(1, 2) # 1 s^2 2 h/s w/s

        return coord

    def pixel_shuffle_ff(self, x, coord):
        b, c, h, w = x.shape
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous() # b c h w

        scale = self.scale ** 2
        x = x.view(b, -1, scale, 2, h, w) # b c/(2*s^2) s^2 2 h w

        # b c/(2*s^2) s^2 2 h w * 1 1 s^2 2 h w
        x = torch.mul(x, coord.unsqueeze(1)) # b c/(2*s^2) s^2 2 h w
        x = torch.sum(x, dim=3) # b c/(2*s^2) s^2 h w
        x = x.view(b, -1, h, w) # b c/2 h w

        x = torch.cat((torch.cos(2.*torch.pi*x), torch.sin(2.*torch.pi*x)), dim=1) # b c h w

        return x

    def forward(self, x):
        shortcut = x
        coord = self.pixel_shuffle_coord(x)

        # feature-based fourier features
        x = self.pixel_shuffle_ff(x, coord)
        x = self.mlp(x * shortcut)
        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, scale={self.scale}')

    def flops(self, h, w):
        """
        Approx. FLOPs of Fourier-feature block.
        - MLP(1×1 Conv) FLOPs = depth × C × C × H × W
        - MLP(1×1 Conv) FLOPs = depth × C × C × H × W
        """
        flops = 0
        flops += h * w * self.dim # LayerNorm
        flops += conv_flops(h, w, self.dim, self.dim, k=1)  # 1x1 Conv
        flops += conv_flops(h, w, self.dim, self.dim, k=1)  # 1x1 Conv

        return flops

class SubPixelMLP(nn.Module):
    r"""Sub-pixel MLP: Fourier-feature enhancement, and sub-pixel shuffling (PixelShuffle).

    Args:
        dim (int):    Number of input channels.
        depth (int):  Number of FC layer of the MLP module.
        scale (int):  Upscaling factor.
    """
    def __init__(self,
                 dim=64,
                 depth=0,
                 scale=4):
        super(SubPixelMLP, self).__init__()
        self.dim = dim
        self.depth = depth
        self.scale = scale

        self.embeds = nn.ModuleList()
        self.ffs = nn.ModuleList()

        if (scale & (scale - 1)) == 0:  # scale = 2^n
            scale_step = 2
        elif scale == 3:
            scale_step = 3
        else:
            raise ValueError(f'scale {scale} is not supported. Now only supported scales: 2^n and 3')

        layers = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for i in range(int(math.log(scale, scale_step))):
                layers.append(nn.Conv2d(dim, dim * scale_step ** 2, 3, 1, 1))
                layers.append(GetFourierFeatures(dim=dim * scale_step ** 2, scale=scale_step, depth=depth))
                layers.append(nn.PixelShuffle(scale_step))
        elif scale == 3:
            layers.append(nn.Conv2d(dim, dim * scale_step ** 2, 3, 1, 1))
            layers.append(GetFourierFeatures(dim=dim * scale_step ** 2, scale=scale_step, depth=depth))
            layers.append(nn.PixelShuffle(scale_step))
        else:
            raise ValueError(f'scale {scale} is not supported. Now only supported scales: 2^n')

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, depth={self.depth}, scale={self.scale}')
    
    def flops(self, h, w):
        """
        FLOPs of all Conv + FF stages inside SubPixelMLP.
        """
        flops = 0
        cur_h, cur_w, cur_c = h, w, self.dim
        scale_step = 2 if (self.scale & (self.scale - 1)) == 0 else 3
        n_stage = int(math.log(self.scale, scale_step)) if scale_step == 2 else 1

        for n in range(n_stage):
            out_c = cur_c * (scale_step ** 2)

            # 3×3 Conv
            flops += conv_flops(cur_h, cur_w, cur_c, out_c, k=3)

            # Fourier-feature (FLOPs)
            ff_flops = self.layers[int((3*n)+1)].flops(cur_h, cur_w)
            flops += ff_flops

            # PixelShuffle
            cur_h *= scale_step
            cur_w *= scale_step
            cur_c = out_c // (scale_step ** 2)

        return flops