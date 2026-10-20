"""Aguçamento (grupo primário — lacuna de escopo): unsharp masking e laplaciano."""
from __future__ import annotations

import numpy as np

from ._core import correlate2d, gaussian_blur, to_float, to_uint8

LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
LAPLACIAN_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)


def unsharp_mask(img: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    """y = x + amount * (x - G_sigma * x)."""
    x = to_float(img)
    return to_uint8(x + amount * (x - gaussian_blur(x, sigma)))


def laplacian_sharpen(img: np.ndarray, c: float, neighbors: int = 4) -> np.ndarray:
    """y = x - c * Laplaciano(x)."""
    x = to_float(img)
    k = LAPLACIAN_4 if neighbors == 4 else LAPLACIAN_8
    return to_uint8(x - c * correlate2d(x, k))
