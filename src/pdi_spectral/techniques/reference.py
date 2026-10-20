"""Implementações de referência (OpenCV / PIL / SciPy).

Usadas para (1) validar as versões do zero (``make verify`` / testes) e
(2) como *backend* rápido em execuções grandes (``backend: reference``).
Mesma assinatura das versões do zero.
"""
from __future__ import annotations

import io

import numpy as np

from . import superres as _sr
from ._core import to_float, to_uint8
from .contrast import auto_grid


def _cv2():
    import cv2

    return cv2


def gaussian(img, sigma):
    cv2 = _cv2()
    r = max(1, int(np.ceil(3.0 * sigma)))
    return cv2.GaussianBlur(img, (2 * r + 1, 2 * r + 1), sigma, borderType=cv2.BORDER_REFLECT_101)


def median(img, k):
    return _cv2().medianBlur(np.ascontiguousarray(img), int(k))


def bilateral(img, sigma_s, sigma_r, radius=None):
    # Atenção: em imagens coloridas o OpenCV usa distância L1 entre cores
    # (a versão do zero usa L2). Em escala de cinza são idênticas.
    r = radius if radius is not None else max(1, int(np.ceil(2.0 * sigma_s)))
    return _cv2().bilateralFilter(np.ascontiguousarray(img), 2 * r + 1, sigma_r * 255.0, sigma_s)


def unsharp_mask(img, amount, sigma=1.0):
    cv2 = _cv2()
    x = to_float(img)  # desfoque em float para não quantizar duas vezes
    r = max(1, int(np.ceil(3.0 * sigma)))
    blur = cv2.GaussianBlur(x, (2 * r + 1, 2 * r + 1), sigma, borderType=cv2.BORDER_REFLECT_101)
    return to_uint8(x + amount * (x - blur))


def laplacian_sharpen(img, c, neighbors=4):
    cv2 = _cv2()
    x = to_float(img)
    if neighbors == 4:
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    else:
        k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
    lap = cv2.filter2D(x, cv2.CV_64F, k, borderType=cv2.BORDER_REFLECT_101)
    return to_uint8(x - c * lap)


def _on_luma(img, fn):
    cv2 = _cv2()
    ycc = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = fn(np.ascontiguousarray(ycc[..., 0]))
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def hist_eq(img, alpha=1.0):
    cv2 = _cv2()

    def f(y):
        eq = cv2.equalizeHist(y).astype(np.float64)
        return np.clip(np.rint((1 - alpha) * y + alpha * eq), 0, 255).astype(np.uint8)

    return _on_luma(img, f)


def clahe(img, clip_limit, grid="auto"):
    cv2 = _cv2()
    g = auto_grid(img.shape) if grid == "auto" else (int(grid), int(grid))
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(g[1], g[0]))
    return _on_luma(img, op.apply)


def wiener(img, k, noise=None):
    from scipy.signal import wiener as _w

    x = to_float(img)
    out = np.stack([_w(x[..., c], (k, k), noise) for c in range(x.shape[2])], axis=-1)
    return to_uint8(np.nan_to_num(out))


def jpeg(img, quality):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality), subsampling=2)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _pil_resize(img, h, w):
    from PIL import Image

    return np.array(Image.fromarray(img).resize((w, h), Image.BICUBIC))


def _pil_down(img, scale):
    h, w = img.shape[:2]
    return _pil_resize(img, max(1, round(h / scale)), max(1, round(w / scale)))


def sr_bicubic(img, scale):
    h, w = img.shape[:2]
    return _pil_resize(_pil_down(img, scale), h, w)


def sr_learned(img, scale, method="espcn", _model_dir="data/sr_models"):
    return _sr.sr_learned(img, scale, method, _model_dir, _downscale=_pil_down, _resize=_pil_resize)
