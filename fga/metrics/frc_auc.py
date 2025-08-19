import numpy as np

from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_frc_auc(img, img2, num_rho=64, full=False, eps=1e-8):
    assert img.shape == img2.shape, "Shapes must match"

    # Convert to grayscale if RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = img.mean(axis=-1)
        img2 = img2.mean(axis=-1)

    h, w = img.shape
    fft1 = np.fft.rfft2(img)
    fft2 = np.fft.rfft2(img2)

    product = fft1 * np.conj(fft2)
    power1 = np.abs(fft1) ** 2
    power2 = np.abs(fft2) ** 2

    # radial distance map
    fy = np.fft.fftfreq(h, d=1.0) * h
    fx = np.fft.rfftfreq(w, d=1.0) * w
    fy2d, fx2d = np.meshgrid(fy, fx, indexing="ij")
    r_map = np.sqrt(fx2d ** 2 + fy2d ** 2)

    # equal‑count bin edges via sorted radii quantiles
    r_flat = np.sort(r_map.flatten())
    total = len(r_flat)
    idxs = (np.linspace(0, total, num_rho + 1)).round().astype(int).clip(max=total - 1)
    bin_edges = r_flat[idxs]

    frc = np.zeros(num_rho, dtype=np.float32)
    eps = 1e-8
    for i in range(num_rho):
        if i == num_rho - 1:
            mask = (r_map >= bin_edges[i]) & (r_map <= bin_edges[i + 1])
        else:
            mask = (r_map >= bin_edges[i]) & (r_map < bin_edges[i + 1])
        if mask.any():
            num = np.abs(np.sum(product[mask]))
            denom = np.sqrt(np.sum(power1[mask]) * np.sum(power2[mask]))
            frc[i] = 0.0 if denom < eps else num / denom

    if not full:
        frc = frc[(num_rho//4) * 3:]

    frc_auc = np.mean(frc)

    return frc_auc