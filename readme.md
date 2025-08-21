## 📁 Showcase (Network × Dataset)

**Models** : EDSR-origin, EDSR-LW, RCAN-origin, RCAN-LW, HAN-origin, HAN-LW, NLSN-origin, NLSN-LW, SwinIR-origin, SwinIR-LW
**Datasets** : Set5, Set14, Urban100, Manga109, BSDS100, DTD235

### Basic templete

```
./
├─ GT/
│  ├─ {DATASET}/
│  └─ {DATASET}_HF_vis/
├─ {NETWORK-FGAx4}/
│  ├─ {DATASET}/
│  └─ {DATASET}_HF_vis/
└─ {NETWORK}x4/
   ├─ {DATASET}/
   └─ {DATASET}_HF_vis/
```

---

### Examples

#### Example 1) `EDSR-origin, BSDS100`

```
./
├─ GT/
│  ├─ BSDS100/
│  └─ BSDS100_HF_vis/
├─ EDSR-FGAx4/
│  ├─ BSDS100/
│  └─ BSDS100_HF_vis/
└─ EDSRx4/
   ├─ BSDS100/
   └─ BSDS100_HF_vis/
```

#### Example 2) `EDSR-LW, Manga109`

```
./
├─ GT/
│  ├─ Manga109/
│  └─ Manga109_HF_vis/
├─ EDSR-LW-FGAx4/
│  ├─ Manga109/
│  └─ Manga109_HF_vis/
└─ EDSR-LWx4/
   ├─ Manga109/
   └─ Manga109_HF_vis/
```

#### Example 3) `SwinIR-LW, DTD235`

```
./
├─ GT/
│  ├─ DTD235/
│  └─ DTD235_HF_vis/
├─ SwinIR-LW-FGAx4/
│  ├─ DTD235/
│  └─ DTD235_HF_vis/
└─ SwinIR-LWx4/
   ├─ DTD235/
   └─ DTD235_HF_vis/
```