"""Compressão JPEG (controle) implementada do zero — apenas a parte com perdas.

A codificação de entropia (Huffman) é sem perdas e não altera a imagem
decodificada, por isso é omitida. Etapas reproduzidas do libjpeg/IJG:
  1. RGB -> YCbCr (BT.601 full range);
  2. subamostragem de croma 4:2:0 (média 2x2);
  3. DCT-II 8x8 ortonormal com deslocamento de nível (-128);
  4. quantização com as tabelas do Anexo K escaladas pela qualidade (regra IJG);
  5. dequantização, IDCT, *fancy upsampling* (triangular) do croma e volta a RGB.
"""
from __future__ import annotations

import numpy as np

from ._core import rgb_to_ycbcr, ycbcr_to_rgb, to_float, to_uint8

Q_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)
Q_CHROMA = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.float64,
)


def _dct_matrix(n: int = 8) -> np.ndarray:
    u = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    c = np.cos((2 * x + 1) * u * np.pi / (2 * n)) * np.sqrt(2.0 / n)
    c[0, :] = np.sqrt(1.0 / n)
    return c


_C = _dct_matrix()


def scaled_table(base: np.ndarray, quality: int) -> np.ndarray:
    q = int(np.clip(quality, 1, 100))
    s = 5000 / q if q < 50 else 200 - 2 * q
    t = np.floor((base * s + 50) / 100)
    return np.clip(t, 1, 255)


def _blocks_roundtrip(plane: np.ndarray, table: np.ndarray) -> np.ndarray:
    """plane em [0,255], dimensões múltiplas de 8."""
    h, w = plane.shape
    b = (plane - 128.0).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    coef = np.einsum("ux,abxy,vy->abuv", _C, b, _C)
    coef = np.rint(coef / table) * table
    rec = np.einsum("ux,abuv,vy->abxy", _C, coef, _C)
    return rec.transpose(0, 2, 1, 3).reshape(h, w) + 128.0


def _fancy_upsample(c: np.ndarray) -> np.ndarray:
    """Upsampling 2x triangular (h2v2 fancy upsampling do libjpeg)."""
    p = np.pad(c, 1, mode="edge")
    h, w = c.shape
    out = np.empty((2 * h, 2 * w))
    center = p[1:-1, 1:-1]
    for di, sy in ((0, -1), (1, 1)):
        for dj, sx in ((0, -1), (1, 1)):
            vert = p[1 + sy : 1 + sy + h, 1:-1]
            horz = p[1:-1, 1 + sx : 1 + sx + w]
            diag = p[1 + sy : 1 + sy + h, 1 + sx : 1 + sx + w]
            out[di::2, dj::2] = (9 * center + 3 * vert + 3 * horz + diag) / 16.0
    return out


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    h, w = img.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    x = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    ycc = rgb_to_ycbcr(to_float(x)) * 255.0
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]

    def down(c):
        return c.reshape(c.shape[0] // 2, 2, c.shape[1] // 2, 2).mean(axis=(1, 3))

    ql, qc = scaled_table(Q_LUMA, quality), scaled_table(Q_CHROMA, quality)
    y_r = _blocks_roundtrip(y, ql)
    cb_r = _fancy_upsample(_blocks_roundtrip(down(cb), qc))
    cr_r = _fancy_upsample(_blocks_roundtrip(down(cr), qc))
    rgb = ycbcr_to_rgb(np.stack([y_r, cb_r, cr_r], axis=-1) / 255.0)
    return to_uint8(rgb)[:h, :w]
