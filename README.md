[![arXiv](https://img.shields.io/badge/arXiv-2508.10616-b31b1b.svg)](https://arxiv.org/abs/2508.10616)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/yourname/FGA-SR/pulls)

# FGA-SR: Fourier-Guided Attention Upsampling for Image Super-Resolution

**Official PyTorch implementation** of:

> **Fourier-Guided Attention Upsampling for Image Super-Resolution**  
> *Daejune Choi, Youchan No, Jinhyung Lee, Duksu Kim*  
> [[Paper]](https://arxiv.org/abs/2508.10616)

---

## 🔎 Summary

We propose **Fourier-Guided Attention (FGA)**, a **drop-in upsampler** for SISR that adds only ~**0.3M** params yet consistently boosts PSNR (**+0.14 dB** on lightweight and **+0.12 dB** on original-scale backbones) and substantially improves **frequency-domain fidelity** (e.g., **+26–29%** FRC-AUC on Urban100 for several backbones).  
FGA integrates:
- **Fourier-Feature MLP** for position-aware frequency encoding  
- **Cross-Resolution Correlation Attention (XRA)** for feature alignment  
- **Frequency-Domain L1 Loss** for spectral consistency

> Plug-and-play replacement for PixelShuffle / transposed-conv / NN-conv upsamplers.

---

## 🖼️ Overview

<p align="center">
  <img src="assets/network_structure.png" width="820" alt="FGA overview diagram"/>
</p>

<p align="center">
  <img src="assets/fft_and_feat.png" width="820" alt="FFT spectrum and Feature map results of basic and our upsampler"/>
</p>

---

## 📌 Updates

* ✅ **2025-08-14**: arXiv v1 released.
* ✅ **2025-08-20**: Public code release (this repository).
* ⬜ Model zoo & pretrained weights for ×4.
* ⬜ Visual results.

---

## ✨ Features
- **Drop-in**: Replace common upsamplers with FGA
- **Backbone-agnostic**: Works with CNN/Transformer SR (EDSR/RCAN/HAN/NLSN/SwinIR-style)
- **Lightweight**: ~+0.3M parameters
- **Spectral-aware**: Less aliasing/ringing via frequency-domain supervision

---

## 📦 Installation

Install **PyTorch** first (choose the proper CUDA build from pytorch.org), then:

```bash
pip install -r requirements.txt
python setup.py develop
```

Or use the provided [dockerfile](dockerfile).

---

## 📁 Data

- For data preparation, refer to:
  - **Training and testing datasets**
    - [Dataset Preparation (BasicSR)](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)
  - **Additionally, we offer the DTD235 test set (focusing on texture dataset)**
    - [Guide/Download](#)  <!-- TODO: add link -->

---

## 🚀 Quick Start

### Inference
```bash
# single image


# folder

```

### Training
```bash
# 1 GPU
python fga/train.py -opt options/train/options/EDSR/train_EDSR-FGAx4.yml

# DDP (multi-GPU)
torchrun --nproc_per_node=4 fga/train.py -opt options/train/options/SwinIR/train_SwinIR-FGAx4.yml --launcher pytorch
```
---

## 🔩 Integrate FGA into Your Model
```python
import torch
import torch.nn as nn
from fga.archs.fga_arch import FGA

class MySRNet(nn.Module):
    def __init__(self, scale=4, feat_dim=192):
        super().__init__()
        self.body = MyBackbone(feat_dim)
        self.upsampler = FGA(
                 dim=64,
                 back_embed_dim=feat_dim,
                 out_dim=3,
                 upscale=scale,
        )

    def forward(self, x):
        f = self.body(x)
        f_up = self.upsampler(f)
        return f_up
```

---

## 🧪 Results (×4, Lightweight (µ) & Origin combined)

**Backbones**: EDSR, RCAN, HAN, NLSN, SwinIR.  
**Settings**: For each backbone, **top row = SPC** (baseline), **bottom row = FGA** (ours).  
**Metric**: PSNR/SSIM on **Y-channel** (Set5 / Set14 / BSD100 / Urban100 / Manga109 / DTD235).
| Backbone | Upsampler | Set5 | Set14 | BSD100 | Urban100 | Manga109 | DTD235 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **EDSRµ** | SPC | 32.12 / 0.8953 | 28.59 / 0.7822 | 27.58 / 0.7378 | 26.08 / 0.7860 | 30.41 / 0.9077 | 29.73 / 0.7599 |
|  | **FGA** | **32.30 / 0.8968** | **28.68 / 0.7838** | **27.64 / 0.7392** | **26.27 / 0.7896** | **30.74 / 0.9106** | **29.80 / 0.7601** |
| **EDSR (orig.)** | SPC | 32.46 / 0.8968 | 28.80 / 0.7876 | 27.71 / 0.7420 | 26.64 / 0.8033 | 31.02 / 0.9148 | 29.91 / 0.7664 |
|  | **FGA** | **32.50 / 0.8995** | **28.80 / 0.7877** | **27.74 / 0.7437** | **26.67 / 0.8030** | **31.05 / 0.9158** | 29.89 / 0.7655 |
| **RCANµ** | SPC | 32.34 / 0.8975 | 28.69 / 0.7851 | 27.65 / 0.7406 | 26.44 / 0.7972 | 30.78 / 0.9124 | 29.88 / 0.7649 |
|  | **FGA** | **32.43 / 0.8984** | **28.76 / 0.7858** | **27.70 / 0.7414** | **26.52 / 0.7974** | **30.94 / 0.9135** | 29.90 / 0.7641 |
| **RCAN (orig.)** | SPC | 32.63 / 0.9002 | 28.87 / 0.7889 | 27.77 / 0.7436 | 26.82 / 0.8087 | 31.22 / 0.9173 | 30.00 / 0.7681 |
|  | **FGA** | 32.63 / **0.9007** | **28.88 / 0.7891** | **27.79 / 0.7449** | **26.83 / 0.8067** | **31.23 / 0.9161** | **30.02 / 0.7678** |
| **HANµ** | SPC | 32.44 / 0.8988 | 28.76 / 0.7865 | 27.70 / 0.7419 | 26.55 / 0.8002 | 30.96 / 0.9142 | 29.92 / 0.7665 |
|  | **FGA** | **32.52 / 0.8997** | **28.85 / 0.7878** | **27.75 / 0.7431** | **26.70 / 0.8024** | **31.21 / 0.9160** | **29.97 / 0.7660** |
| **HAN (orig.)** | SPC | 32.64 / 0.9002 | 28.90 / 0.7890 | 27.80 / 0.7442 | 26.85 / 0.8094 | 31.42 / 0.9177 | 30.06 / 0.7703 |
|  | **FGA** | **32.68 / 0.9013** | **28.93 / 0.7897** | **27.81 / 0.7454** | **26.91 / 0.8087** | **31.43 / 0.9181** | 30.06 / 0.7689 |
| **NLSNµ** | SPC | 32.30 / 0.8969 | 28.69 / 0.7841 | 27.65 / 0.7401 | 26.37 / 0.7937 | 30.75 / 0.9112 | 29.84 / 0.7653 |
|  | **FGA** | **32.48 / 0.8990** | **28.81 / 0.7863** | **27.73 / 0.7425** | **26.64 / 0.7994** | **31.17 / 0.9155** | **29.96 / 0.7666** |
| **NLSN (orig.)** | SPC | 32.59 / 0.9000 | 28.87 / 0.7891 | 27.78 / 0.7444 | 26.96 / 0.8109 | 31.27 / 0.9184 | 30.08 / 0.7715 |
|  | **FGA** | **32.72 / 0.9021** | **28.98 / 0.7907** | **27.85 / 0.7471** | **27.09 / 0.8136** | **31.59 / 0.9207** | **30.11 / 0.7713** |
| **SwinIRµ** | SPC | 32.49 / 0.8994 | 28.86 / 0.7884 | 27.75 / 0.7441 | 26.66 / 0.8040 | 31.23 / 0.9172 | 29.96 / 0.7683 |
|  | **FGA** | **32.59 / 0.9007** | **28.97 / 0.7905** | **27.82 / 0.7455** | **26.86 / 0.8064** | **31.55 / 0.9195** | **30.06 / 0.7693** |
| **SwinIR (orig.)** | SPC | 32.92 / 0.9044 | 29.09 / 0.7950 | 27.92 / 0.7489 | 27.45 / 0.8254 | 32.03 / 0.9260 | 30.28 / 0.7789 |
|  | **FGA** | **32.96 / 0.9052** | **29.13 / 0.7957** | **27.96 / 0.7510** | **27.47 / 0.8249** | **32.14 / 0.9261** | **30.29 / 0.7774** |

### 📐 Frequency-Domain Metric: FRC-AUC (Top-25% rings)

**Fourier Ring Correlation (FRC)** measures how well an SR result matches the **ground-truth in the frequency domain**.  
Instead of judging only in pixel space (PSNR/SSIM), FRC computes correlation **per radial frequency ring** after a 2D FFT, revealing **high-frequency fidelity** (textures, thin edges) and issues like **aliasing/checkerboard**.

- **How it works (brief)**  
  1) Convert images to Y-channel (recommended) and apply 2D FFT.  
  2) Partition the spectrum into rings with equal numbers of pixels.  
  3) Compute a normalized correlation on each ring → **FRC curve** in \[0, 1\] (low→high frequency).

- **High-Frequency AUC (top 25%)**  
  To summarize high-frequency faithfulness in one number, we average the **top quartile** of the FRC curve (highest rings).  
  Higher is better → fewer artifacts and better texture preservation.

- **Why it matters**  
  Two models with similar PSNR can behave very differently in high frequencies.  
  FRC complements PSNR/SSIM by directly probing the spectrum where textures live.

> 📓 **Details & code**: see **[`frc_demo.ipynb`](scripts/frc_demo.ipynb)** — it provides formulas, step-by-step computation, plotting, and CSV export for FRC curves and High-Freq AUC.

<details>
<summary><b>FRC-AUC results</b></summary>
| Backbone (scale) | Upsampler | Set5 | Set14 | BSD100 | Urban100 | Manga109 | DTD235 |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| EDSRµ | SPC | 0.2648 | 0.2093 | 0.2126 | 0.2226 | 0.2998 | 0.1625 |
|  | **FGA** | **0.2941** | **0.2347** | **0.2294** | **0.2804** | **0.3313** | **0.1758** |
| EDSR (orig.) | SPC | 0.2857 | 0.2281 | 0.2279 | 0.2809 | 0.3221 | 0.1745 |
|  | **FGA** | **0.2920** | **0.2358** | **0.2362** | **0.3010** | **0.3341** | **0.1771** |
| RCANµ | SPC | 0.2740 | 0.2186 | 0.2199 | 0.2534 | 0.3216 | 0.1691 |
|  | **FGA** | **0.3020** | **0.2373** | **0.2339** | **0.2975** | **0.3323** | **0.1771** |
| RCAN (orig.) | SPC | 0.2924 | 0.2367 | 0.2365 | 0.2962 | 0.3387 | 0.1788 |
|  | **FGA** | **0.2947** | **0.2433** | **0.2389** | **0.3097** | **0.3424** | **0.1795** |
| HANµ | SPC | 0.2824 | 0.2289 | 0.2299 | 0.2799 | 0.3287 | 0.1735 |
|  | **FGA** | **0.3028** | **0.2406** | **0.2378** | **0.3119** | **0.3487** | **0.1796** |
| HAN (orig.) | SPC | 0.2211 | 0.1815 | 0.1983 | 0.1810 | 0.3224 | 0.1564 |
|  | **FGA** | **0.2953** | **0.2397** | **0.2373** | **0.3070** | **0.3450** | **0.1776** |
| NLSNµ | SPC | 0.2657 | 0.2155 | 0.2144 | 0.2330 | 0.3142 | 0.1670 |
|  | **FGA** | **0.2963** | **0.2439** | **0.2366** | **0.3017** | **0.3439** | **0.1799** |
| NLSN (orig.) | SPC | 0.2851 | 0.2354 | 0.2357 | 0.3019 | 0.3380 | 0.1769 |
|  | **FGA** | **0.2999** | **0.2440** | **0.2415** | **0.3178** | **0.3502** | **0.1820** |
| SwinIRµ | SPC | 0.2655 | 0.1902 | 0.2209 | 0.2446 | 0.3342 | 0.1641 |
|  | **FGA** | **0.2965** | **0.2286** | **0.2392** | **0.3080** | **0.3548** | **0.1740** |
| SwinIR (orig.) | SPC | 0.2955 | 0.2238 | 0.2470 | 0.3371 | 0.3616 | 0.1805 |
|  | **FGA** | **0.3028** | **0.2336** | **0.2492** | **0.3477** | **0.3715** | **0.1815** |
</details>

> **Params (M)**: µ+SPC → {EDSR 1.5, RCAN 4.3, HAN 8.6, NLSN 1.6, SwinIR 1.2} vs **µ+FGA** → (+0.3M each); **Original+SPC** → {38.7, 15.6, 16.1, 39.7, 11.9} vs **Original+FGA** → (+0.3M each).

---

## 🧰 Model Zoo (placeholders)

| Scale | Backbone | Config | Weights | Note |
|:--:|:--:|:--:|:--:|:--|
| ×4 | Lightweight | [Guide/Download](#) | [Guide/Download](#) | — |
| ×4 | Full-capacity | [Guide/Download](#) | [Guide/Download](#) | — |

---

## 📝 Citation

```bibtex
@article{Choi2025FGA,
  title   = {Fourier-Guided Attention Upsampling for Image Super-Resolution},
  author  = {Choi, Daejune and No, Youchan and Lee, Jinhyung and Kim, Duksu},
  journal = {arXiv preprint arXiv:2508.10616},
  year    = {2025}
}
```

---

## 📄 License
This project is released under the **MIT License**. See [LICENSE](LICENSE).
```