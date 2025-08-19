from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from basicsr.utils.img_util import tensor2img, img2tensor, imfrombytes

# Reference: https://github.com/mhamilton723/FeatUp/blob/main/featup/util.py
def pca(image_feats_list, dim=3, fit_pca=None, max_samples=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = torch.nn.functional.interpolate(tensor, target_size, mode="bilinear")

        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[-2:]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        fit_pca = TorchPCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected

# -------------------------------------------------------------------------------------------
def get(filepath):
    filepath = str(filepath)
    with open(filepath, 'rb') as f:
        value_buf = f.read()
    return value_buf

def load_img(img_path, float32=True, pad=None, device='cuda'):
    img_bytes = get(img_path)
    img = imfrombytes(img_bytes, float32=float32)
    img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    if pad is not None:
        return pad_img(img, pad)
    return img

def pad_img(img, patch_size):
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = img.size()
    if h % patch_size != 0:
        mod_pad_h = patch_size - h % patch_size
    if w % patch_size != 0:
        mod_pad_w = patch_size - w % patch_size
    img = torch.nn.functional.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    return img, mod_pad_h, mod_pad_w

def imread_bytes(path,
                 flag="color",
                 to_float32=False):
    flag_map = dict(color=cv2.IMREAD_COLOR,
                    gray=cv2.IMREAD_GRAYSCALE,
                    unchanged=cv2.IMREAD_UNCHANGED)
    with open(path, "rb") as f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), flag_map[flag])
    if to_float32:
        img = img.astype(np.float32) / 255.0
    return img


def np2tensor(img: np.ndarray,
              bgr2rgb=True,
              dtype=torch.float32):
    if img.ndim == 2:
        img = img[..., None]
    if bgr2rgb and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).to(dtype)
    return tensor


def tensor2np(x,
              rgb2bgr=True,
              denorm=(0, 1)):
    x = x.detach().cpu().clamp(*denorm)
    x = (x - denorm[0]) / (denorm[1] - denorm[0])
    x = x.squeeze(0).numpy().transpose(1, 2, 0)
    if rgb2bgr and x.shape[2] == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return (x * 255.0).round().astype(np.uint8)

def pad_mod(x, modulo):
    b, c, h, w = x.shape
    pad_h = (modulo - h % modulo) % modulo
    pad_w = (modulo - w % modulo) % modulo
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w

def img_visualize(img, path):
    fig, ax = plt.subplots()

    x = img.shape[1] / fig.dpi
    y = img.shape[0] / fig.dpi
    fig.set_figwidth(x)
    fig.set_figheight(y)

    ax.imshow(img)

    ax.set_axis_off()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.imshow(img)

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def imgs_visualize_row(
    imgs, titles=None, title_fontsize=18, title_color='black',
    cmap_for_2d='viridis',
    dpi=100, savepath=None
):
    def _to_numpy(im):
        if torch.is_tensor(im):
            t = im.detach().cpu().float()
            if t.dim() == 4:   # B,C,H,W -> take first
                t = t[0]
            if t.dim() == 3:   # C,H,W
                return tensor2img(t, rgb2bgr=False, out_type=np.float32)
            if t.dim() == 2:   # H,W
                a = t.numpy()
                a = (a - a.min()) / (a.max() - a.min() + 1e-12)
                return a
            raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")
        elif isinstance(im, np.ndarray):
            return im
        else:
            raise TypeError(type(im))

    imgs_np = [_to_numpy(im) for im in imgs]
    n = len(imgs_np)

    Hs = [im.shape[0] for im in imgs_np]
    Ws = [im.shape[1] for im in imgs_np]
    Hmax = max(Hs)
    Wsum = sum(Ws)

    title_h_in = (title_fontsize / 72.0) * 1.2 if titles else 0.0
    fig_w_in = Wsum / dpi
    fig_h_in = Hmax / dpi + title_h_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    axes = []

    top_margin = title_h_in / fig_h_in
    y0 = 0.0
    ax_h = 1.0 - top_margin

    x = 0.0
    for j, im in enumerate(imgs_np):
        w_frac = Ws[j] / Wsum
        ax = fig.add_axes([x, y0, w_frac, ax_h])
        if im.ndim == 2:
            ax.imshow(im, cmap=cmap_for_2d, interpolation='nearest')
        else:
            ax.imshow(im, interpolation='nearest')
        ax.axis('off')
        axes.append(ax)
        x += w_frac

    if titles:
        if len(titles) < n:
            titles = list(titles) + [""] * (n - len(titles))
        else:
            titles = titles[:n]
        x = 0.0
        for j in range(n):
            w_frac = Ws[j] / Wsum
            cx = x + w_frac / 2
            fig.text(cx, 1.0, titles[j], ha='center', va='bottom',
                     fontsize=title_fontsize, color=title_color)
            x += w_frac

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"saved: {savepath}")

    plt.show()

def feature_print(features, path):
    img_visualize(tensor2img([features.unsqueeze(1)], rgb2bgr=False)[:, :, 0], path)

def fft_log_norm(x):
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None]
    # using PCA
    x = pca([x], 1)[0][0]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    fft = torch.log1p(torch.abs(fft))
    fft = fft / fft.max()
    return fft