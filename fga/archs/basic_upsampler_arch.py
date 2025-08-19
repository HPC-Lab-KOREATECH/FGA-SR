from fga.archs.arch_util import conv_flops
from torch import nn as nn

from fga.archs.upsamplers import DeConv, SubPixelConv, InterpolationConv

from basicsr.utils.registry import ARCH_REGISTRY


# baseline
@ARCH_REGISTRY.register()
class BasicUpsampler(nn.Module):
    r""" upsampler

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        upscale (int): Upscale factor.
        img_range (float): Image range.
        rgb_norm (bool): If True, normalize the RGB image.
        type (str): Type of upsampler. Default: 'SubPixelConv'.
        interpolation (str): Type of interpolation. Default: 'nearest'.
    """
    def __init__(self,
                 # common args
                 back_embed_dim=None,
                 dim=64,
                 out_dim=3,
                 upscale=4,
                 # upsampler args
                 u_type='SubPixelConv',
                 interpolation='nearest'):
        super(BasicUpsampler, self).__init__()
        self.back_embed_dim = back_embed_dim
        self.dim = dim
        self.out_dim = out_dim
        self.upscale = upscale

        self.u_type = u_type
        self.interpolation = interpolation

        self.embed = nn.Sequential(nn.Conv2d(self.back_embed_dim, dim, 3, 1, 1),
                                    nn.LeakyReLU(inplace=True)) if back_embed_dim is not None else nn.Identity()

        if u_type == 'SubPixelConv':
            self.upsample = SubPixelConv(
                dim=dim,
                upscale=upscale
                )
        elif u_type == 'DeConv':
            self.upsample = DeConv(
                dim=dim,
                upscale=upscale
                )
        elif u_type == 'InterpolationConv':
            self.upsample = InterpolationConv(
                dim=dim,
                upscale=upscale,
                interpolation=interpolation
                )

        self.unembed = nn.Conv2d(dim, out_dim, 3, 1, 1) if out_dim is not None else nn.Identity()

    def forward(self, x):
        x = self.embed(x)
        x = self.upsample(x)
        x = self.unembed(x)

        return x

    def extra_repr(self) -> str:
        return (f'back_embed_dim={self.back_embed_dim}, dim={self.dim}, out_dim={self.out_dim}, upscale={self.upscale},\n'
                f'u_type={self.u_type}, interpolation={self.interpolation}, \n')
    
    def flops(self, h, w):
        """
        Total FLOPs of the FGA upsampler, including:
        embed-conv  → SubPixelMLP → CAB(attn+MLP) → unembed-conv
        """
        flops = 0

        # (a) 3×3 embed Conv
        flops += conv_flops(h, w, self.back_embed_dim, self.dim, k=3)

        # (b) SubPixelConv → HR resolution feature map
        subpixelmlp_flops = self.upsample.flops(h, w)
        flops += subpixelmlp_flops

        # (c) 3×3 unembed Conv (HR)
        hr_h, hr_w = h * self.upscale, w * self.upscale
        flops += conv_flops(hr_h, hr_w, self.dim, self.out_dim, k=3)

        return flops