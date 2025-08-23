import torch
from einops import rearrange
from torch import nn as nn

from fga.archs.subpixmlp import SubPixelMLP
from fga.archs.arch_util import MLP, trunc_normal_, conv_flops

from basicsr.utils.registry import ARCH_REGISTRY


def window_partition(x, window_size):
    """
    Args:
        x: (b, c, h, w)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size*window_size, c)
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, c)

    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): window size
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, c, h, w)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(b, -1, h, w)

    return x


# Overlapping Window-based Cross Resolution Attention (OWXRA)
class OWXRA(nn.Module):
    r"""Overlapping Window-based Cross Resolution Attention (OWXRA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        overlap_ratio (float): The ratio of overlap between adjacent windows.
        num_heads (int): Number of attention heads, but only support 1 head.
        upscale (int): Upscale factor.
        kv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """
    def __init__(self,
                 dim,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 upscale=2,
                 kv_bias=True,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.upscale = upscale
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.kv = nn.Linear(dim, dim * 2,  bias=kv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)

        self.window_size_up = self.upscale * self.window_size

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((self.window_size_up + self.overlap_win_size - 1) * (self.window_size_up + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        relative_position_index = self.calculate_rpi_owxra()
        self.register_buffer('relative_position_index', relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim,dim)

    def calculate_rpi_owxra(self):
        # calculate relative position index for OCA
        window_size_up = self.upscale * self.window_size # HR window size
        window_size_cur = self.window_size + int(self.overlap_ratio * self.window_size) # LR window size (overlap)

        coords_h = torch.arange(window_size_up)
        coords_w = torch.arange(window_size_up)
        coords_up = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_up_flatten = torch.flatten(coords_up, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_cur)
        coords_w = torch.arange(window_size_cur)
        coords_cur = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_cur_flatten = torch.flatten(coords_cur, 1)  # 2, wse*wse

        relative_coords = coords_cur_flatten[:, None, :] - coords_up_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_up - window_size_cur + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_up - window_size_cur + 1

        relative_coords[:, :, 0] *= window_size_up + window_size_cur - 1
        relative_position_index = relative_coords.sum(-1)

        return relative_position_index

    def forward(self, x, x2):
        b, h, w, c = x.shape

        kv = self.kv(x).permute(0, 3, 1, 2) # b, 2c, h, w

        q_windows = x2
        kv_windows = self.unfold(kv) # b, cowow, nw

        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nwb, owow, c
        k_windows, v_windows = kv_windows.unbind(0) # nwb, owow, c

        nwb, ww4, _ = q_windows.shape
        _, owow, _ = k_windows.shape

        d = self.dim // self.num_heads
        q = q_windows.reshape(nwb, ww4, self.num_heads, d).permute(0, 2, 1, 3) # nwb, nH, ww4, d
        k = k_windows.reshape(nwb, owow, self.num_heads, d).permute(0, 2, 1, 3) # nwb, nH, owow, d
        v = v_windows.reshape(nwb, owow, self.num_heads, d).permute(0, 2, 1, 3) # nwb, nH, owow, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size_up * self.window_size_up, self.overlap_win_size * self.overlap_win_size, -1)  # 4ww, owow, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, 4ww, owow
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x2 = (attn @ v).transpose(1, 2).reshape(nwb, ww4, self.dim)

        x2 = self.proj(x2)

        return x2

    def flops(self, h, w):
        """
        Rough FLOPs for attention inside an OWXRA block.
        Args:
            h, w (int): spatial size of the LR feature map entering CAL
            b (int): batch size
        """
        n_win = (h // self.window_size) * (w // self.window_size)         # number of windows
        d_head = self.dim // self.num_heads
        q_tokens   = (self.window_size_up) ** 2                                 # HR window tokens
        kv_tokens  = (self.overlap_win_size) ** 2                               # LR(overlap) tokens

        fl_per_attn = 0
        # K V projection
        fl_per_attn += self.num_heads * q_tokens * self.dim * self.dim * 2
        # Q*K
        fl_per_attn += self.num_heads * q_tokens * d_head * kv_tokens
        # attn*V
        fl_per_attn += self.num_heads * q_tokens * kv_tokens * d_head
        # output projection
        fl_per_attn += q_tokens * self.dim * self.dim

        return n_win * fl_per_attn
    

# Correlation Attention Layer (CAL)
class CAL(nn.Module):
    """Correlation Attention Layer (CAL)

    Args:
        dim (int): Number of input channels.
        upscale (int): Upscale factor.
        num_heads (int): Number of attention heads, but only support 1 head.
        window_size (int): The height and width of the window.
        drop_path (float): Dropout rate of attention weight.
        kv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        overlap_ratio (float, optional): The ratio of overlap between adjacent windows. Default: 0.5
    """
    def __init__(self,
                 dim,
                 upscale,
                 num_heads=1,
                 window_size=7,
                 kv_bias=True,
                 qk_scale=None,
                 overlap_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.upscale = upscale
        self.num_heads = num_heads
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio

        self.attn = OWXRA(
            dim,
            window_size=window_size,
            upscale=upscale,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            kv_bias=kv_bias,
            qk_scale=qk_scale)

        self.mlp = MLP(dim, dim, dim)

    def forward(self, x, x2):
        b, c, h, w = x2.shape

        x_windows = x.permute(0, 2, 3, 1)
        x2_windows = window_partition(x2, self.upscale * self.window_size)

        shortcut = x2_windows

        attn_windows = self.attn(x_windows, x2_windows)
        attn_windows = shortcut + attn_windows

        x = window_reverse(attn_windows, self.upscale * self.window_size, h, w)

        x = x + self.mlp(x)

        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, num_heads={self.num_heads}, \n'
                f'window_size={self.window_size}, overlap_ratio={self.overlap_ratio}, \n'
                f'LR_window_size={self.attn.overlap_win_size}, HR_window_size={self.attn.window_size_up}')

    def flops(self, h, w):
        hr_h, hr_w = h * self.upscale, w * self.upscale

        flops = 0
        # Attention FLOPs
        flops += self.attn.flops(h, w)

        # MLP FLOPs
        flops += 2 * self.dim * self.dim * hr_h * hr_w

        return flops


# Our main method
@ARCH_REGISTRY.register()
class FGA(nn.Module):
    r""" Fourier-Guided Attention Upsampler (FGA)

    Args:
        dim (int): Number of input channels.
        back_embed_dim (int, optional): Number of input channels for the embedding layer. If None, it will be set to `dim`. Default: None.
        out_dim (int): Number of output channels.
        upscale (int): Upscale factor.
        window_size (int): The height and width of the window.
        overlap_ratio (float): The ratio of overlap between adjacent windows.

    """
    def __init__(self,
                 # common args
                 dim=64,
                 back_embed_dim=None,
                 out_dim=None,
                 upscale=4,
                 dropout=0.,
                 # attention args
                 window_size=1,
                 overlap_ratio=4
                 ):
        super(FGA, self).__init__()

        self.dim = dim
        self.back_embed_dim = back_embed_dim if back_embed_dim is not None else dim
        self.out_dim = out_dim
        self.upscale = upscale
        self.dropout = dropout

        self.window_size = window_size
        self.overlap_ratio = overlap_ratio

        self.embed = nn.Sequential(nn.Conv2d(self.back_embed_dim, dim, 3, 1, 1),
                                    nn.LeakyReLU(inplace=True))

        self.coattn = CAL(
            dim=dim,
            upscale=upscale,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            )

        self.upsample = SubPixelMLP(
            dim=dim,
            scale=upscale,
            )
        
        self.dropout = nn.Dropout2d(p=dropout)
        self.unembed = nn.Conv2d(dim, out_dim, 3, 1, 1) if out_dim is not None else nn.Identity()

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(self.coattn(x, self.upsample(x)))
        x = self.unembed(x)
        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, back_embed_dim={self.back_embed_dim}, out_dim={self.out_dim}, upscale={self.upscale}, dropout={self.dropout}\n'
                f'window_size={self.window_size}, overlap_ratio={self.overlap_ratio}, \n')

    def flops(self, h, w, b=1):
        """
        Total FLOPs of the FGA upsampler, including:
        embed-conv  → SubPixelMLP → CAL(attn+MLP) → unembed-conv
        """
        flops = 0

        # (a) 3×3 embed Conv
        flops += conv_flops(h, w, self.back_embed_dim, self.dim, k=3)

        # (b) SubPixelMLP (Conv+FF) → HR resolution feature map
        subpixelmlp_flops = self.upsample.flops(h, w)
        flops += subpixelmlp_flops

        # (c) Corelation Attention Layer (CAL)
        coattn_flops = self.coattn.flops(h, w)
        flops += coattn_flops

        # (d) 3×3 unembed Conv (HR)
        hr_h, hr_w = h * self.upscale, w * self.upscale
        flops += conv_flops(hr_h, hr_w, self.dim, self.out_dim, k=3)

        return flops * b