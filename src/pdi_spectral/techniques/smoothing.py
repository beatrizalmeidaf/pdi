"""Suavização (grupo de controle/âncora): gaussiano, mediana e bilateral."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._core import gaussian_blur, to_float, to_uint8


def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Filtro gaussiano separável, raio = ceil(3*sigma)."""
    return to_uint8(gaussian_blur(to_float(img), sigma))


def median(img: np.ndarray, k: int) -> np.ndarray:
    """Mediana k x k por canal. Borda replicada (como o cv2.medianBlur)."""
    r = k // 2
    p = np.pad(img, ((r, r), (r, r), (0, 0)), mode="edge")
    win = sliding_window_view(p, (k, k), axis=(0, 1))  # (H, W, C, k, k)
    flat = win.reshape(*win.shape[:3], k * k)
    mid = (k * k) // 2
    return np.partition(flat, mid, axis=-1)[..., mid].astype(np.uint8)


def bilateral(img: np.ndarray, sigma_s: float, sigma_r: float, radius: int | None = None) -> np.ndarray:
    """Filtro bilateral (Tomasi & Manduchi, 1998).

    sigma_s em pixels; sigma_r em unidades de intensidade [0, 1].
    Janela circular de raio ceil(2*sigma_s). Distância de cor euclidiana em RGB.
    """
    x = to_float(img)
    r = radius if radius is not None else max(1, int(np.ceil(2.0 * sigma_s)))
    h, w = x.shape[:2]
    p = np.pad(x, ((r, r), (r, r), (0, 0)), mode="reflect")
    num = np.zeros_like(x)
    den = np.zeros((h, w, 1))
    inv_s = 1.0 / (2.0 * sigma_s ** 2)
    inv_r = 1.0 / (2.0 * sigma_r ** 2)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dy * dy + dx * dx
            if d2 > r * r:
                continue
            sh = p[r + dy : r + dy + h, r + dx : r + dx + w]
            diff2 = ((sh - x) ** 2).sum(axis=-1, keepdims=True)
            wgt = np.exp(-d2 * inv_s - diff2 * inv_r)
            num += wgt * sh
            den += wgt
    return to_uint8(num / den)
