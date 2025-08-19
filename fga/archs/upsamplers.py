import torch
import math
from torch.nn import functional as F
from torch import nn as nn


class InterpolationConv(nn.Module):
    def __init__(self,
                 dim=64,
                 upscale=4,
                 interpolation='nearest'):
        super(InterpolationConv, self).__init__()
        self.dim = dim
        self.upscale = upscale
        self.interpolation = interpolation

        layers = []
        if (upscale & (upscale - 1)) == 0:  # upscale = 2^n
            for i in range(int(math.log(upscale, 2))):
                if self.interpolation == 'nearest':
                    layers.append(nn.UpsamplingNearest2d(scale_factor=2))
                    layers.append(nn.Conv2d(dim, dim, 3, 1, 1))
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                elif self.interpolation == 'bilinear':
                    layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
                    layers.append(nn.Conv2d(dim, dim, 3, 1, 1))
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                elif self.interpolation == 'bicubic':
                    class Bicubic(nn.Module):
                        def __init__(self, upscale_factor=2):
                            super(Bicubic, self).__init__()
                            self.upscale_factor = upscale_factor
                        def forward(self, x):
                            return F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
                    layers.append(Bicubic(2))
                    layers.append(nn.Conv2d(dim, dim, 3, 1, 1))
                    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            raise ValueError(f'upscale {upscale} is not supported. Now only supported upscales: 2^n')

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, upscale={self.upscale}, interpolation={self.interpolation}')


class SubPixelConv(nn.Module):
    def __init__(self,
                 dim=64,
                 upscale=4,
                 multi_layer=False,
                 icnr_init=False):
        super(SubPixelConv, self).__init__()
        self.dim = dim
        self.upscale = upscale

        layers = []

        if (upscale & (upscale - 1)) == 0:  # upscale = 2^n
            for i in range(int(math.log(upscale, 2))):
                if multi_layer:
                    layers.append(nn.Conv2d(dim, dim * 2, 3, 1, 1))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Conv2d(dim * 2, dim * 4, 3, 1, 1))
                else:
                    layers.append(nn.Conv2d(dim, dim * 4, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
        elif upscale == 3:
            layers.append(nn.Conv2d(dim, dim * 9, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'upscale {upscale} is not supported. Now only supported upscales: 2^n')
        self.layers = nn.Sequential(*layers)

        if icnr_init:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            weight = self.ICNR(m.weight, initializer=nn.init.normal_,
                        upscale_factor=self.upscale, mean=0.0, std=0.02)
            m.weight.data.copy_(weight)

    def ICNR(self, tensor, initializer, upscale_factor=2, *args, **kwargs):
        "tensor: the 2-dimensional Tensor or more"
        upscale_factor_squared = upscale_factor * upscale_factor
        assert tensor.shape[0] % upscale_factor_squared == 0, \
            ("The size of the first dimension: "
            f"tensor.shape[0] = {tensor.shape[0]}"
            " is not divisible by square of upscale_factor: "
            f"upscale_factor = {upscale_factor}")
        sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                                *tensor.shape[1:])
        sub_kernel = initializer(sub_kernel, *args, **kwargs)

        return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

    def forward(self, x):
        return self.layers(x)

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, upscale={self.upscale}')

    def flops(self, H, W):
        flops = 0
        if (self.upscale & (self.upscale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(self.upscale, 2))):
                flops += H * W * self.dim * self.dim * 4 * 9
                H, W = H * 2, W * 2
        elif self.upscale == 3:
            flops += H * W * self.dim * self.dim * 9 * 9
        return flops
    

class DeConv(nn.Module):
    def __init__(self,
                 dim=64,
                 upscale=4,
                 icnr_init=False):
        super(DeConv, self).__init__()
        self.dim = dim
        self.upscale = upscale

        layers = []

        if (upscale & (upscale - 1)) == 0:  # upscale = 2^n
            for i in range(int(math.log(upscale, 2))):
                layers.append(nn.ConvTranspose2d(dim, dim, 3, 2, 1, 1))
                # layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            raise ValueError(f'upscale {upscale} is not supported. Now only supported upscales: 2^n')
        self.layers = nn.Sequential(*layers)

        if icnr_init:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            weight = self.ICNR(m.weight, initializer=nn.init.normal_,
                        upscale_factor=self.upscale, mean=0.0, std=0.02)
            m.weight.data.copy_(weight)

    def ICNR(self, tensor, initializer, upscale_factor=2, *args, **kwargs):
        "tensor: the 2-dimensional Tensor or more"
        upscale_factor_squared = upscale_factor * upscale_factor
        assert tensor.shape[0] % upscale_factor_squared == 0, \
            ("The size of the first dimension: "
            f"tensor.shape[0] = {tensor.shape[0]}"
            " is not divisible by square of upscale_factor: "
            f"upscale_factor = {upscale_factor}")
        sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                                *tensor.shape[1:])
        sub_kernel = initializer(sub_kernel, *args, **kwargs)

        return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

    def forward(self, x):
        return self.layers(x)

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, upscale={self.upscale}')