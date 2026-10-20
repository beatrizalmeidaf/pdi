"""Restauração (controle): filtro de Wiener adaptativo local (Lee, 1980).

Sem PSF conhecida (as imagens são limpas), o Wiener atua como *denoiser*:
    y = mu + max(var - nu, 0) / max(var, nu) * (x - mu)
com mu e var locais (janela k x k) e nu = média das variâncias locais (por canal).
É a mesma formulação do ``scipy.signal.wiener``.
"""
from __future__ import annotations

import numpy as np

from ._core import box_filter, to_float, to_uint8


def wiener(img: np.ndarray, k: int, noise: float | None = None) -> np.ndarray:
    x = to_float(img)
    mu = box_filter(x, k)
    var = np.maximum(box_filter(x * x, k) - mu * mu, 0.0)
    nu = var.mean(axis=(0, 1), keepdims=True) if noise is None else noise
    gain = np.maximum(var - nu, 0.0) / np.maximum(var, np.maximum(nu, 1e-12))
    return to_uint8(mu + gain * (x - mu))
