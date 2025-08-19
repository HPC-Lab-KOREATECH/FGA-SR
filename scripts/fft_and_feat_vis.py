import gc
import torch
from tqdm.auto import tqdm
from pathlib import Path
from typing import Dict, List
from __future__ import annotations

from fga.archs.basic_upsampler_arch import BasicUpsampler
from fga.archs.edsr_arch import EDSR_
from fga.archs.fga_arch import FGA

from scripts.vis_utils import feature_print, fft_log_norm, imread_bytes, np2tensor, pca


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32
Path("results_fft").mkdir(exist_ok=True)

def replace_upsampler(backbone, upsampler):
    if hasattr(backbone, 'tail'):
        backbone.tail[-2] = upsampler
        backbone.tail[-1] = torch.nn.Identity()
    elif hasattr(backbone, 'upsample'):
        backbone.upsample = upsampler
        backbone.conv_last = torch.nn.Identity()
    return backbone

def load_sr_model(backbone, upsampler, ckpt, param_key=None):
    if upsampler is not None:
        backbone = replace_upsampler(backbone, upsampler)
    state = torch.load(ckpt, map_location="cpu")
    if param_key is not None:
        state = state[param_key]
    backbone.load_state_dict(state, strict=True)
    return backbone.to(DEVICE).eval()

@torch.no_grad()
def visualize_fft_features(model_name,
                           model,
                           hook_module,
                           lr_files,
                           gt_files,
                           out_dir,
                           to_float32=False):

    out_dir.mkdir(exist_ok=True, parents=True)

    feats: List[torch.Tensor] = []
    if hook_module is not None:
        def _hook(_, __, output): feats.append(output)
        handle = hook_module.register_forward_hook(_hook)

    for idx, (lr_p, gt_p) in enumerate(
            tqdm(zip(lr_files, gt_files),
                 total=len(lr_files),
                 desc=f"[{model_name}]")):
        name = lr_p.stem
        feats.clear()
        lr = np2tensor(imread_bytes(lr_p, to_float32=to_float32)).unsqueeze(0).to(DEVICE)
        gt = np2tensor(imread_bytes(gt_p, to_float32=to_float32)).unsqueeze(0).to(DEVICE)

        if hook_module is None:
            sr, up_feat = model(lr, return_feat=True)
        else:
            sr = model(lr)
            up_feat = feats[0]

        feature_print(fft_log_norm(lr), out_dir / f"{name}_fft_lr.png")
        feature_print(fft_log_norm(sr), out_dir / f"{name}_fft_sr.png")
        feature_print(fft_log_norm(gt), out_dir / f"{name}_fft_gt.png")

        if hook_module is None:
            feature_print(fft_log_norm(up_feat), out_dir / f"{name}_fft_feat.png")
            feature_print(pca([up_feat], 1)[0][0], out_dir / f"{name}_feat.png")
        else:
            feature_print(fft_log_norm(up_feat), out_dir / f"{name}_fft_up_feat.png")
            feature_print(pca([up_feat], 1)[0][0], out_dir / f"{name}_up_feat.png")
        del lr, gt, sr, up_feat; gc.collect(); torch.cuda.empty_cache()

    if hook_module is not None:
        handle.remove()


if __name__ == "__main__":
    lr_dir = Path("datasets/urban100/subset/LRbicx4_square") # square cropping for figures
    gt_dir = Path("datasets/urban100/subset/GTmod12_square")
    lr_files = sorted(lr_dir.glob("*"))
    gt_files = sorted(gt_dir.glob("*"))

    MODELS: Dict[str, dict] = {
        "EDSR-IC": dict(
            backbone=EDSR_(
                n_colors=3,
                n_feats=256,
                res_scale=0.1,
                n_resblocks=32,
                scale=[4],
                rgb_range=255,
            ),
            upsampler=BasicUpsampler(
                # common args
                back_embed_dim=256,
                dim=64,
                out_dim=3,
                upscale=4,
                # upsampler args
                u_type='InterpolationConv',
            ),
            ckpt="experiments/upsampler_test/EDSR-DCx4-scratch-training/models/net_g_latest.pth",
            key="params_ema",
        ),    
        "EDSR-DC": dict(
            backbone=EDSR_(
                n_colors=3,
                n_feats=256,
                res_scale=0.1,
                n_resblocks=32,
                scale=[4],
                rgb_range=255,
            ),
            upsampler=BasicUpsampler(
                # common args
                back_embed_dim=256,
                dim=64,
                out_dim=3,
                upscale=4,
                # upsampler args
                u_type='DeConv',
            ),
            ckpt="experiments/upsampler_test/EDSR-DCx4-scratch-training/models/net_g_latest.pth",
            key="params_ema",
        ),    
        "EDSR-SPC": dict(
            backbone=EDSR_(
                n_colors=3,
                n_feats=256,
                res_scale=0.1,
                n_resblocks=32,
                scale=[4],
                rgb_range=255,
            ),
            upsampler=BasicUpsampler(
                # common args
                back_embed_dim=256,
                dim=64,
                out_dim=3,
                upscale=4,
                # upsampler args
                u_type='SubPixelConv',
            ),
            ckpt="experiments/upsampler_test/EDSR-SPCx4-scratch-training/models/net_g_latest.pth",
            key="params_ema",
        ),    
        "EDSR-FGA": dict(
            backbone=EDSR_(
                n_colors=3,
                n_feats=256,
                res_scale=0.1,
                n_resblocks=32,
                scale=[4],
                rgb_range=255,
            ),
            upsampler=FGA(
                # common args
                back_embed_dim=256,
                dim=64,
                out_dim=3,
                upscale=4,
                # attention args
                window_size=1,
                overlap_ratio=4,
            ),
            ckpt="experiments/upsampler_test/EDSR-FGAx4-scratch-training/models/net_g_latest.pth",
            key="params_ema",
        ),    
    }

    for name, cfg in MODELS.items():
        net = load_sr_model(
            backbone=cfg["backbone"],
            ckpt=cfg["ckpt"],
            param_key=cfg.get("key"),
            upsampler=cfg.get("upsampler")
        )
        hook_mod = net.tail[-2].upsample
        visualize_fft_features(
            model_name=name,
            model=net,
            hook_module=hook_mod,
            lr_files=lr_files,
            gt_files=gt_files,
            out_dir=Path("results_fft_up") / name,
        )
        hook_mod = None
        visualize_fft_features(
            model_name=name,
            model=net,
            hook_module=hook_mod,
            lr_files=lr_files,
            gt_files=gt_files,
            out_dir=Path("results_fft") / name,
        )

