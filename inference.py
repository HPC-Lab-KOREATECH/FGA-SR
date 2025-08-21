"""
FGA-only SR inference (x4)
"""

import gc
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
import torch.nn.functional as F

from fga.archs.edsr_arch import EDSR_ as EDSR
from fga.archs.rcan_arch import RCAN_ as RCAN
from fga.archs.han_arch import HAN
from fga.archs.nlsn_arch import NLSN
from fga.archs.swinir_arch import SwinIR_ as SwinIR

from fga.archs.fga_arch import FGA


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ===== Spec tables (from options/test's yml files) =====
BACKBONE_SPECS = {
    "EDSR": {
        "builder": "edsr",
        "args": dict(n_colors=3, n_feats=256, res_scale=0.1, n_resblocks=32, scale=[4], rgb_range=255),
        "img_range": 255.0,
        "fga_back_dim": 256,
    },
    "EDSR-LW": {
        "builder": "edsr",
        "args": dict(n_colors=3, n_feats=64, res_scale=1.0, n_resblocks=16, scale=[4], rgb_range=255),
        "img_range": 255.0,
        "fga_back_dim": 64,
    },
    "RCAN": {
        "builder": "rcan",
        "args": dict(n_colors=3, n_feats=64, n_resgroups=10, n_resblocks=20, reduction=16, res_scale=1.0, scale=[4], rgb_range=255),
        "img_range": 255.0,
        "fga_back_dim": 64,
    },
    "RCAN-LW": {
        "builder": "rcan",
        "args": dict(n_colors=3, n_feats=64, n_resgroups=5, n_resblocks=10, reduction=16, res_scale=1.0, scale=[4], rgb_range=255),
        "img_range": 255.0,
        "fga_back_dim": 64,
    },
    "HAN": {
        "builder": "han",
        "args": dict(n_colors=3, n_feats=64, n_resgroups=10, n_resblocks=20, reduction=16, res_scale=1.0, scale=[4], rgb_range=255),
        "img_range": 255.0,
        "fga_back_dim": 64,
    },
    "HAN-LW": {
        "builder": "han",
        "args": dict(n_colors=3, n_feats=64, n_resgroups=10, n_resblocks=10, reduction=16, res_scale=1.0, scale=[4], rgb_range=255),
        "img_range": 255.0,
        "fga_back_dim": 64,
    },
    "NLSN": {
        "builder": "nlsn",
        "args": dict(n_colors=3, n_feats=256, n_hashes=4, chunk_size=144, n_resblocks=32, res_scale=0.1, scale=[4], rgb_range=1),
        "img_range": 1.0,
        "fga_back_dim": 256,
    },
    "NLSN-LW": {
        "builder": "nlsn",
        "args": dict(n_colors=3, n_feats=64, n_hashes=4, chunk_size=72, n_resblocks=16, res_scale=1.0, scale=[4], rgb_range=1),
        "img_range": 1.0,
        "fga_back_dim": 64,
    },
    "SWINIR": {
        "builder": "swinir",
        "args": dict(
            upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
            depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
            mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv"
        ),
        "img_range": 1.0,
        "fga_back_dim": 64,
    },
    "SWINIR-LW": {
        "builder": "swinir",
        "args": dict(
            upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
            depths=[6,6,6,6], embed_dim=60, num_heads=[6,6,6,6],
            mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv"
        ),
        "img_range": 1.0,
        "fga_back_dim": 64,
    },
}

def read_image_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    return torch.from_numpy(img).unsqueeze(0)

def save_image_rgb(path, ten):
    arr = ten.detach().clamp_(0, 1).mul_(255.0).round().byte().cpu().squeeze(0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)

def get_window_size(model) -> int:
    """Return model's window size if present; otherwise 1."""
    for attr in ("window_size", "win_size"):
        if hasattr(model, attr):
            v = getattr(model, attr)
            if isinstance(v, int):
                return v
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return int(v[0])
    try:
        return int(model.layers[0].blocks[0].attn.window_size)
    except Exception:
        return 1

def pad_to_multiple(img: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    return img, pad_h, pad_w

def unpad_sr(sr: torch.Tensor, pad_h: int, pad_w: int, scale: int) -> torch.Tensor:
    if pad_h or pad_w:
        _, _, H, W = sr.shape
        sr = sr[:, :, : H - pad_h * scale, : W - pad_w * scale]
    return sr

def pick_state_dict(obj):
    candidates = ["params_ema", "params", "state_dict", "net_g_ema", "model_ema", "net_g", "model"]
    for k in candidates:
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            return obj[k]
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise KeyError(f"Could not find a valid state_dict. Tried: {candidates}")

# ===== FGA insertion =====
def replace_upsampler(backbone, upsampler):
    if hasattr(backbone, "tail"):
        try:
            backbone.tail[-2] = upsampler
            backbone.tail[-1] = torch.nn.Identity()
            return backbone
        except Exception:
            pass
    if hasattr(backbone, "upsample"):
        backbone.upsample = upsampler
        if hasattr(backbone, "conv_last"):
            backbone.conv_last = torch.nn.Identity()
        return backbone
    if hasattr(backbone, "upsampler"):
        backbone.upsampler = upsampler
        if hasattr(backbone, "conv_last"):
            backbone.conv_last = torch.nn.Identity()
        return backbone
    raise NotImplementedError("FGA insertion point not found for this backbone.")

def make_fga(backbone):
    spec = BACKBONE_SPECS[backbone]
    return FGA(
        back_embed_dim=spec["fga_back_dim"],
        dim=64,
        out_dim=3,
        upscale=4,
        window_size=1,
        overlap_ratio=4,
    )

# ===== builders  =====
def build_edsr(args):
    return EDSR(**args)

def build_rcan(args):
    return RCAN(**args)

def build_han(args):
    return HAN(**args)

def build_nlsn(args):
    return NLSN(**args)

def build_swinir(args):
    return SwinIR(**args)

BUILDERS = {
    "edsr": build_edsr,
    "rcan": build_rcan,
    "han": build_han,
    "nlsn": build_nlsn,
    "swinir": build_swinir,
}

def load_model(backbone, ckpt_path):
    b = backbone.upper()
    if b not in BACKBONE_SPECS:
        raise ValueError(f"Unsupported backbone: {backbone}")
    spec = BACKBONE_SPECS[b]

    net = BUILDERS[spec["builder"]](spec["args"])

    fga = make_fga(b)
    net = replace_upsampler(net, fga)

    obj = torch.load(str(ckpt_path), map_location="cpu")
    state = pick_state_dict(obj)
    net.load_state_dict(state, strict=True)

    net.eval()
    net.to(DEVICE, dtype=DTYPE)
    img_range = float(spec["img_range"])    # 255.0 or 1.0
    return net, img_range

# ===== inference (optional: tiling) =====
@torch.no_grad()
def forward_full_or_tiled(
    model,
    img,    # [1,3,H,W] in [0,1]
    scale,
    img_range,  # 255.0 or 1.0
    tile=None,
    tile_overlap=32,
):
    def _clear_cuda():
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    win = max(1, get_window_size(model))
    x = img.to(DEVICE, dtype=DTYPE) * img_range
    x, pad_h, pad_w = pad_to_multiple(x, win)
    b, _, h, w = x.size()

    def _run(inp):
        return model(inp)

    # full-image path
    if tile is None:
        try:
            sr = _run(x)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            print("[WARN] OOM on full image. Falling back to tiled inference...")
            _clear_cuda()
            tile = 256
        else:
            sr = unpad_sr(sr, pad_h, pad_w, scale)
            return sr / img_range

    # tiled path
    if tile is None:
        tile = 256
    if tile % win != 0:
        tile = max(win, (tile // win) * win)
    tile = min(tile, h, w)

    stride = tile - tile_overlap
    if stride <= 0:
        stride = tile

    h_idx = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    w_idx = list(range(0, max(w - tile, 0) + 1, stride)) or [0]

    last_h = max(h - tile, 0)
    last_w = max(w - tile, 0)
    if h_idx[-1] != last_h:
        h_idx.append(last_h)
    if w_idx[-1] != last_w:
        w_idx.append(last_w)

    E = None
    W = None

    for hi in h_idx:
        for wi in w_idx:
            in_patch = x[..., hi:hi + tile, wi:wi + tile]
            out_patch = _run(in_patch)  # [b, c_out, tile*scale, tile*scale]
            if E is None:
                _, c_out, _, _ = out_patch.shape
                E = torch.zeros(b, c_out, h * scale, w * scale, device=DEVICE, dtype=DTYPE)
                W = torch.zeros_like(E)
            hs, ws = hi * scale, wi * scale
            he, we = (hi + tile) * scale, (wi + tile) * scale
            E[..., hs:he, ws:we].add_(out_patch)
            W[..., hs:he, ws:we].add_(torch.ones_like(out_patch))

    sr = E / (W + 1e-8)
    sr = unpad_sr(sr, pad_h, pad_w, scale)
    return sr / img_range

# ===== driver =====
def run_dir(
    backbone,
    input_dir,
    output_dir,
    weight_path,
    tile,
    tile_overlap,
):
    model, img_range = load_model(backbone, weight_path)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        lr = read_image_rgb(path)
        sr = forward_full_or_tiled(
            model, lr, scale=4, img_range=img_range, tile=tile, tile_overlap=tile_overlap
        )
        save_image_rgb(output_dir / f"{path.stem}.png", sr)

    print(f"[DONE] Saved to: {output_dir}")

def parse_args():
    p = argparse.ArgumentParser(description="FGA-SR Inference (x4)")
    p.add_argument("--backbone", type=str, required=True,
                   choices=["EDSR", "EDSR-LW", "RCAN", "RCAN-LW", "HAN", "HAN-LW", "SwinIR", "SwinIR-LW", "NLSN", "NLSN-LW"],
                   help="Backbone type (FGA upsampler is automatically attached).")
    p.add_argument("--input_dir", type=str, required=True, help="Low-resolution image folder")
    p.add_argument("--output_dir", type=str, required=True, help="Output folder")
    p.add_argument("--weight_path", type=str, required=True, help="Checkpoint (.pth)")
    p.add_argument("--tile", type=int, default=None, help="Tile size for tiled inference (default: None=whole image)")
    p.add_argument("--tile_overlap", type=int, default=32, help="Overlap between tiles")
    return p.parse_args()

def main():
    args = parse_args()
    run_dir(
        backbone=args.backbone,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        weight_path=Path(args.weight_path),
        tile=args.tile,
        tile_overlap=args.tile_overlap,
    )

if __name__ == "__main__":
    main()
