# Reference: https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py

import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

from fga.archs.arch_util import conv_flops, default_conv, MeanShift, Upsampler


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
    def flops(self, h, w):
        # GAP: ignore / 1×1 Conv X2
        c = self.conv_du[0].in_channels
        red = self.conv_du[0].out_channels
        flops = conv_flops(1, 1, c, red, 1)
        flops += conv_flops(1, 1, red, c, 1)
        return flops


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res
   
    def flops(self, h, w):
        k = self.body[0].kernel_size[0]         # 1st conv
        c = self.body[0].in_channels
        flops = 2 * conv_flops(h, w, c, c, k)   # conv x2
        flops += self.body[-1].flops(h, w)      # CALayer
        return flops


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
    def flops(self, h, w):
        flops = 0
        n_rb = len(self.body) - 1          # RCAB
        for i in range(n_rb):
            flops += self.body[i].flops(h, w)
        # last conv
        last = self.body[-1]
        flops += conv_flops(h, w,
                            last.in_channels,
                            last.out_channels,
                            last.kernel_size[0])
        return flops

## Residual Channel Attention Network (RCAN), patch size 48x48, batch size 16, learning rate 1e-4, rgb range 255, DIV2K dataset, using chop
@ARCH_REGISTRY.register()
class RCAN_(nn.Module):
    def __init__(self,
                 n_colors=3,
                 n_feats=64,
                 res_scale=1,
                 reduction=16,
                 n_resgroups=10,
                 n_resblocks=20,
                 scale=[4],
                 rgb_range=255,
                 conv=default_conv):
        super(RCAN_, self).__init__()
        kernel_size = 3
        reduction = reduction
        scale = scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
            
    def flops(self, h, w, b=1):
        flops = 0
        # head conv
        head_conv = self.head[0] # Conv2d
        flops += conv_flops(h, w,
                            head_conv.in_channels,
                            head_conv.out_channels,
                            head_conv.kernel_size[0])

        # body ─ Residual Groups
        for rg in self.body[:-1]:   # before last conv
            flops += rg.flops(h, w)

        # body last conv
        last_body = self.body[-1]
        flops += conv_flops(h, w,
                            last_body.in_channels,
                            last_body.out_channels,
                            last_body.kernel_size[0])
        return flops * b