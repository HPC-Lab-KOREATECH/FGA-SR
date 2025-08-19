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
* ✅ **2025-08-20**: Private code release (this repository).
* ⬜ Model zoo & pretrained weights for ×4.
* ⬜ Update inference code.
* ⬜ Offer visual SR results.

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

## 📁 Dataset

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
**Metric**: PSNR on **Y-channel** (Set5 / Set14 / BSD100 / Urban100 / Manga109 / DTD235).
| Backbone | Upsampler | Set5 | Set14 | BSD100 | Urban100 | Manga109 | DTD235 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **EDSRµ** | SPC | 32.12 | 28.59 | 27.58 | 26.08 | 30.41 | 29.73 |
|  | **FGA** | **32.30** | **28.68** | **27.64** | **26.27** | **30.74** | **29.80** |
| **EDSR (orig.)** | SPC | 32.46 | 28.80 | 27.71 | 26.64 | 31.02 | 29.91 |
|  | **FGA** | **32.50** | **28.80** | **27.74** | **26.67** | **31.05** | 29.89 |
| **RCANµ** | SPC | 32.34 | 28.69 | 27.65 | 26.44 | 30.78 | 29.88 |
|  | **FGA** | **32.43** | **28.76** | **27.70** | **26.52** | **30.94** | 29.90 |
| **RCAN (orig.)** | SPC | 32.63 | 28.87 | 27.77 | 26.82 | 31.22 | 30.00 |
|  | **FGA** | 32.63 / **0.9007** | **28.88** | **27.79** | **26.83** | **31.23** | **30.02** |
| **HANµ** | SPC | 32.44 | 28.76 | 27.70 | 26.55 | 30.96 | 29.92 |
|  | **FGA** | **32.52** | **28.85** | **27.75** | **26.70** | **31.21** | **29.97** |
| **HAN (orig.)** | SPC | 32.64 | 28.90 | 27.80 | 26.85 | 31.42 | 30.06 |
|  | **FGA** | **32.68** | **28.93** | **27.81** | **26.91** | **31.43** | 30.06 |
| **NLSNµ** | SPC | 32.30 | 28.69 | 27.65 | 26.37 | 30.75 | 29.84 |
|  | **FGA** | **32.48** | **28.81** | **27.73** | **26.64** | **31.17** | **29.96** |
| **NLSN (orig.)** | SPC | 32.59 | 28.87 | 27.78 | 26.96 | 31.27 | 30.08 |
|  | **FGA** | **32.72** | **28.98** | **27.85** | **27.09** | **31.59** | **30.11** |
| **SwinIRµ** | SPC | 32.49 | 28.86 | 27.75 | 26.66 | 31.23 | 29.96 |
|  | **FGA** | **32.59** | **28.97** | **27.82** | **26.86** | **31.55** | **30.06** |
| **SwinIR (orig.)** | SPC | 32.92 | 29.09 | 27.92 | 27.45 | 32.03 | 30.28 |
|  | **FGA** | **32.96** | **29.13** | **27.96** | **27.47** | **32.14** | **30.29** |

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

**Params (M)**
| Setting        | EDSR | RCAN |  HAN | NLSN | SwinIR |
| -------------- | :--: | :--: | :--: | :--: | :----: |
| µ + SPC        |  1.5 |  4.3 |  8.6 |  1.6 |   1.2  |
| µ + FGA        |  1.8 |  4.6 |  8.9 |  1.9 |   1.5  |
| Original + SPC | 38.7 | 15.6 | 16.1 | 39.7 |  11.9  |
| Original + FGA | 39.0 | 15.9 | 16.4 | 40.0 |  12.2  |

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
This project is released under the **Apache-2.0 License**. See [LICENSE](LICENSE).
```