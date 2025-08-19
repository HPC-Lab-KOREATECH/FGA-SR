# reference: https://github.com/wwlCape/HAN/blob/master/src/model/han.py

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

from fga.archs.arch_util import bmm_flops, conv_flops, default_conv, MeanShift, Upsampler


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


class DAM_Module(nn.Module):
    """ Deep attention module"""
    def __init__(self, in_dim):
        super(DAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


class GAM_Module(nn.Module):
    """ Global
    attention module"""
    def __init__(self, in_dim):
        super(GAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))

        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out

    def flops(self, n, c, h, w):
        # QK^T + Attention*V
        qk = bmm_flops(n, n, c * h * w)
        av = bmm_flops(n, c * h * w, n)
        return qk + av
    

class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))

        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x
    
    def flops(self, h, w, c):
        # 3D conv: 커널 (3,3,3) ⇒ 공간 위치당 2*3*3*3*c=54c
        return 2 * h * w * c * 3 * 3 * 3
    

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


# Holistic Attention Network (HAN), patch size 64x64, batch size 16, learning rate 1e-5, rgb range 255, DIV2K dataset (fine-tuned RCAN)
@ARCH_REGISTRY.register()
class HAN(nn.Module):
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
        super(HAN, self).__init__()

        n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
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
        self.ga = CSAM_Module(n_feats)
        self.da = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        #pdb.set_trace()
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            #print(name)
            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1),res1],1)
        #res = self.body(x)
        out1 = res
        #res3 = res.unsqueeze(1)
        #res = torch.cat([res1,res3],1)
        res = self.da(res1)
        out2 = self.last_conv(res)

        out1 = self.ga(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)

        res += x
        #res = self.ga(res)

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def flops(self, h, w, b=1):
        flops = 0

        # head conv
        head = self.head[0]
        flops += conv_flops(h, w,
                            head.in_channels,
                            head.out_channels,
                            head.kernel_size[0])

        # body (Residual Groups)
        for rg in self.body[:-1]:          # before last conv
            flops += rg.flops(h, w)

        # body last conv
        last_body = self.body[-1]
        flops += conv_flops(h, w,
                            last_body.in_channels,
                            last_body.out_channels,
                            last_body.kernel_size[0])

        # DA(Layer Attention) ─ N = n_resgroups
        n = len(self.body) - 1
        c = head.out_channels
        flops += self.da.flops(n, c, h, w)

        # GA(Channel-Spatial)  + last_conv + last
        flops += self.ga.flops(h, w, c)
        flops += conv_flops(h, w, c * 11, c, 3)   # last_conv
        flops += conv_flops(h, w, c * 2,  c, 3)   # last

        return flops * b 