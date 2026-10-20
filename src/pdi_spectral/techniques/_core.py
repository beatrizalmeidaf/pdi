"""Primitivas de PDI implementadas do zero (apenas NumPy).

Convenções usadas em todo o pacote:
- Imagens de entrada/saída das técnicas: ``uint8`` com forma (H, W, 3), RGB.
- Cálculos internos: ``float64`` em [0, 1].
- Borda padrão: reflexão sem repetir a borda (``np.pad(mode="reflect")``),
  equivalente ao ``BORDER_REFLECT_101`` do OpenCV.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Conversões
# ---------------------------------------------------------------------------
def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / 255.0


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """RGB float [0,1] -> YCbCr float [0,1] (BT.601 full range, como no JPEG)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 0.5, ycc[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1)


def luma(img: np.ndarray) -> np.ndarray:
    """Luminância Y em float [0,1]. Aceita (..., H, W, 3) uint8 ou float."""
    x = img.astype(np.float64)
    if img.dtype == np.uint8:
        x = x / 255.0
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


# ---------------------------------------------------------------------------
# Convolução / correlação
# ---------------------------------------------------------------------------
def _pad_hw(x: np.ndarray, ry: int, rx: int, mode: str) -> np.ndarray:
    width = [(ry, ry), (rx, rx)] + [(0, 0)] * (x.ndim - 2)
    return np.pad(x, width, mode=mode)


def correlate2d(x: np.ndarray, kernel: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Correlação 2D de (H, W) ou (H, W, C) com um kernel (kh, kw) ímpar."""
    kh, kw = kernel.shape
    p = _pad_hw(x, kh // 2, kw // 2, mode)
    win = sliding_window_view(p, (kh, kw), axis=(0, 1))  # (H, W, [C], kh, kw)
    return np.einsum("...ij,ij->...", win, kernel)


def correlate1d(x: np.ndarray, kernel: np.ndarray, axis: int, mode: str = "reflect") -> np.ndarray:
    r = len(kernel) // 2
    width = [(0, 0)] * x.ndim
    width[axis] = (r, r)
    p = np.pad(x, width, mode=mode)
    win = sliding_window_view(p, len(kernel), axis=axis)  # janela vai para o último eixo
    return win @ kernel


def separable(x: np.ndarray, ky: np.ndarray, kx: np.ndarray, mode: str = "reflect") -> np.ndarray:
    return correlate1d(correlate1d(x, ky, 0, mode), kx, 1, mode)


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: np.ndarray, sigma: float, radius: int | None = None, mode: str = "reflect") -> np.ndarray:
    k = gaussian_kernel1d(sigma, radius)
    return separable(x, k, k, mode)


def box_filter(x: np.ndarray, size: int, mode: str = "reflect") -> np.ndarray:
    k = np.full(size, 1.0 / size)
    return separable(x, k, k, mode)


# ---------------------------------------------------------------------------
# Métricas simples (também do zero)
# ---------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim_luma(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM (Wang et al., 2004) na luminância, janela gaussiana sigma=1.5."""
    x, y = luma(a), luma(b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    g = lambda z: gaussian_blur(z, 1.5, radius=min(5, (min(z.shape) - 1) // 2))
    mx, my = g(x), g(y)
    sxx = g(x * x) - mx * mx
    syy = g(y * y) - my * my
    sxy = g(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())
