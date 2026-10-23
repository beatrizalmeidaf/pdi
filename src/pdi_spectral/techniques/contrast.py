"""Equalização / contraste (grupo primário): equalização global e CLAHE.

Ambas operam no canal Y (YCbCr) para não distorcer a cor, como é prática comum.
"""
from __future__ import annotations

import numpy as np

from ._core import rgb_to_ycbcr, to_float, ycbcr_to_rgb, to_uint8


def _apply_on_luma(img: np.ndarray, fn) -> np.ndarray:
    ycc = rgb_to_ycbcr(to_float(img))
    y8 = to_uint8(ycc[..., 0])
    ycc[..., 0] = fn(y8).astype(np.float64) / 255.0
    return to_uint8(ycbcr_to_rgb(ycc))


def equalize_u8(y: np.ndarray) -> np.ndarray:
    """Equalização de histograma global de uma imagem uint8 2D (mesma regra do OpenCV)."""
    hist = np.bincount(y.ravel(), minlength=256)
    nz = np.flatnonzero(hist)
    if len(nz) <= 1:
        return y.copy()
    cdf = np.cumsum(hist)
    cdf_min = cdf[nz[0]]
    lut = np.clip(np.rint((cdf - cdf_min) * 255.0 / (y.size - cdf_min)), 0, 255).astype(np.uint8)
    return lut[y]


def clahe_u8(y: np.ndarray, clip_limit: float, grid: tuple[int, int]) -> np.ndarray:
    """CLAHE (Zuiderveld, 1994) seguindo o algoritmo do OpenCV.

    grid = (linhas de blocos, colunas de blocos).
    """
    h, w = y.shape
    gy, gx = grid
    ph, pw = (-h) % gy, (-w) % gx
    yp = np.pad(y, ((0, ph), (0, pw)), mode="reflect") if (ph or pw) else y
    th, tw = yp.shape[0] // gy, yp.shape[1] // gx
    area = th * tw
    clip = max(int(clip_limit * area / 256), 1) if clip_limit > 0 else 0

    tiles = yp.reshape(gy, th, gx, tw).transpose(0, 2, 1, 3).reshape(gy, gx, area)
    luts = np.empty((gy, gx, 256), dtype=np.float64)
    for i in range(gy):
        for j in range(gx):
            hist = np.bincount(tiles[i, j], minlength=256).astype(np.int64)
            if clip > 0:
                excess = int(np.maximum(hist - clip, 0).sum())
                hist = np.minimum(hist, clip)
                batch, residual = divmod(excess, 256)
                hist += batch
                if residual:
                    step = max(256 // residual, 1)
                    hist[np.arange(0, 256, step)[:residual]] += 1
            luts[i, j] = np.clip(np.rint(np.cumsum(hist) * (255.0 / area)), 0, 255)

    def coords(n, t, g):
        f = np.arange(n) / t - 0.5
        a = np.floor(f).astype(int)
        frac = f - a
        return np.clip(a, 0, g - 1), np.clip(a + 1, 0, g - 1), frac

    ty1, ty2, ya = coords(h, th, gy)
    tx1, tx2, xa = coords(w, tw, gx)
    ty1, ty2, ya = ty1[:, None], ty2[:, None], ya[:, None]
    tx1, tx2, xa = tx1[None, :], tx2[None, :], xa[None, :]
    res = (luts[ty1, tx1, y] * (1 - xa) + luts[ty1, tx2, y] * xa) * (1 - ya) + (
        luts[ty2, tx1, y] * (1 - xa) + luts[ty2, tx2, y] * xa
    ) * ya
    return np.clip(np.rint(res), 0, 255).astype(np.uint8)


def auto_grid(shape: tuple[int, ...]) -> tuple[int, int]:
    """Grade de blocos dependente da resolução: 32px -> 2x2, 224px -> 8x8."""
    g = int(np.clip(min(shape[:2]) // 16, 2, 8))
    return (g, g)


def hist_eq(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Equalização global com dose alpha: Y' = (1-alpha) Y + alpha EQ(Y)."""

    def f(y8):
        eq = equalize_u8(y8).astype(np.float64)
        return np.clip(np.rint((1 - alpha) * y8 + alpha * eq), 0, 255)

    return _apply_on_luma(img, f)


def clahe(img: np.ndarray, clip_limit: float, grid: int | str = "auto", alpha: float = 1.0) -> np.ndarray:
    """CLAHE no canal Y. alpha < 1 mistura com o original (dose contínua abaixo do clip mínimo).

    Atenção: o clip efetivo é max(int(clip_limit * area_do_bloco / 256), 1). Em blocos de
    16x16 px (CIFAR, grade 2x2) isso arredonda o clip para inteiro: todo clip < 2 dá o mesmo resultado.
    """
    g = auto_grid(img.shape) if grid == "auto" else (int(grid), int(grid))

    def f(y8):
        eq = clahe_u8(y8, clip_limit, g).astype(np.float64)
        return np.clip(np.rint((1 - alpha) * y8 + alpha * eq), 0, 255) if alpha != 1.0 else eq

    return _apply_on_luma(img, f)
