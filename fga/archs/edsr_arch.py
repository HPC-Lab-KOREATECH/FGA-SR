# Reference: https://github.com/sanghyun-son/EDSR-PyTorch/tree/master/src/model/edsr.py

import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

from fga.archs.arch_util import conv_flops, default_conv, MeanShift, ResBlock, Upsampler


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


# EDSR, patch size 48x48, batch size 16, learning rate 1e-4, rgb range 255, DIV2K dataset
@ARCH_REGISTRY.register()
class EDSR_(nn.Module):
    def __init__(self,
                 n_colors=3,
                 n_feats=256,
                 res_scale=0.1,
                 n_resblocks=32,
                 scale=[4],
                 rgb_range=255,
                 conv=default_conv):
        super(EDSR_, self).__init__()
        kernel_size = 3
        scale = scale[0]
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1)
        self.add_mean = MeanShift(rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, return_feat=False):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x
        
        x = self.tail(res)
        x = self.add_mean(x)
        if return_feat:
            return x, res
        else:
            return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

    def flops(self, h, w, b=1):
        flops = 0

        # head
        for m in self.head:
            in_c, out_c, k = m.in_channels, m.out_channels, m.kernel_size[0]
            flops += conv_flops(h, w, in_c, out_c, k)

        l = len(self.body) - 1

        # body
        for i in range(l):
            flops += self.body[i].flops(h, w)
        flops += conv_flops(h, w, out_c, out_c, k)

        return flops * b
