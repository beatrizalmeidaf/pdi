"""Filtros espectrais ideais ("oráculo") para calibrar a sensibilidade por banda.

Não são técnicas de aprimoramento: servem para medir, por modelo e dataset,
quanto custa remover (ganho 0) ou amplificar (ganho > 1) a energia de UMA banda
radial, isoladamente. O perfil resultante S(banda) é usado para testar se o
dano das técnicas clássicas é previsível a partir do seu perfil de ΔE
(experimento E3, "transferência do oráculo").

Frequência radial normalizada r = sqrt(fx^2 + fy^2) em ciclos/pixel (0 a ~0.707).
O termo DC (r = 0) é sempre preservado.
"""
from __future__ import annotations

import numpy as np

from ._core import to_float, to_uint8


def _radius(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fx ** 2 + fy ** 2)


def band_gain(img: np.ndarray, r_lo: float, r_hi: float, gain: float) -> np.ndarray:
    """Multiplica por ``gain`` os coeficientes com r_lo < r <= r_hi (por canal)."""
    x = to_float(img)
    h, w = x.shape[:2]
    r = _radius(h, w)
    mask = np.ones((h, w))
    mask[(r > r_lo) & (r <= r_hi)] = gain
    F = np.fft.fft2(x, axes=(0, 1))
    y = np.real(np.fft.ifft2(F * mask[..., None], axes=(0, 1)))
    return to_uint8(y)


def lowpass(img: np.ndarray, cutoff: float) -> np.ndarray:
    return band_gain(img, cutoff, 1.0, 0.0)


def highpass(img: np.ndarray, cutoff: float) -> np.ndarray:
    """Remove (0, cutoff], preservando DC e o restante."""
    return band_gain(img, 0.0, cutoff, 0.0)
