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


def _step(x: np.ndarray, width: float) -> np.ndarray:
    """Degrau suave (cosseno elevado) de 0 a 1 em [−width/2, width/2]; width = 0 dá o degrau ideal."""
    if width <= 0:
        return (x > 0).astype(float)
    return np.clip(0.5 * (1 + np.sin(np.pi * np.clip(x / width, -0.5, 0.5))), 0, 1)


def band_mask(h: int, w: int, r_lo: float, r_hi: float, gain: float, soft: float = 0.0) -> np.ndarray:
    """Máscara radial: ``gain`` em r_lo < r <= r_hi e 1 fora; ``soft`` = largura da transição (ciclos/pixel)."""
    r = _radius(h, w)
    if soft <= 0:
        mask = np.ones((h, w))
        mask[(r > r_lo) & (r <= r_hi)] = gain
        return mask
    inside = _step(r - r_lo, soft) * (1 - _step(r - r_hi, soft))
    mask = 1 + (gain - 1) * inside
    mask[r == 0] = 1.0  # DC preservado
    return mask


def band_gain(img: np.ndarray, r_lo: float, r_hi: float, gain: float, soft: float = 0.0) -> np.ndarray:
    """Multiplica por ``gain`` os coeficientes com r_lo < r <= r_hi (por canal).

    ``soft`` > 0 troca o corte abrupto por uma transição em cosseno de largura ``soft`` em cada borda (filtro "suave"),
    para separar o efeito do corte abrupto (ondulação de Gibbs) do efeito de mexer na banda.
    """
    x = to_float(img)
    h, w = x.shape[:2]
    mask = band_mask(h, w, r_lo, r_hi, gain, soft)
    F = np.fft.fft2(x, axes=(0, 1))
    y = np.real(np.fft.ifft2(F * mask[..., None], axes=(0, 1)))
    return to_uint8(y)


def lowpass(img: np.ndarray, cutoff: float, soft: float = 0.0) -> np.ndarray:
    return band_gain(img, cutoff, 1.0, 0.0, soft)


def highpass(img: np.ndarray, cutoff: float, soft: float = 0.0) -> np.ndarray:
    """Remove (0, cutoff], preservando DC e o restante."""
    return band_gain(img, 0.0, cutoff, 0.0, soft)
