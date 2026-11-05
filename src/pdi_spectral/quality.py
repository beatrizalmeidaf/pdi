"""Métricas de qualidade perceptual SEM referência, implementadas do zero.

Servem para medir se a técnica cumpre o próprio objetivo declarado (o "ganho
perceptual") em cada dose, e assim achar a janela operacional:
    [dose em que a imagem já melhorou,  dose em que o classificador começa a perder]

  sharpness    gradiente médio de Sobel (Tenengrad)          -> aguçamento quer subir
  noise        desvio do ruído estimado (Immerkær, 1996)     -> suavização/denoising quer descer
  contrast     contraste RMS (desvio-padrão da luminância)    -> equalização quer subir
  entropy      entropia do histograma de luminância (bits)    -> equalização quer subir
  colorfulness Hasler & Süsstrunk (2003)                      -> informativo

São proxies objetivos do objetivo de cada técnica, não um juízo humano. Para
qualidade perceptual "geral", ative ``quality.brisque`` (requer ``pip install piq``).
"""
from __future__ import annotations

import numpy as np

from .techniques._core import luma

METRICS = ["sharpness", "noise", "contrast", "entropy", "colorfulness"]


def _sharpness(y: np.ndarray) -> np.ndarray:
    p = np.pad(y, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    gx = (p[:, :-2, 2:] + 2 * p[:, 1:-1, 2:] + p[:, 2:, 2:]) - (p[:, :-2, :-2] + 2 * p[:, 1:-1, :-2] + p[:, 2:, :-2])
    gy = (p[:, 2:, :-2] + 2 * p[:, 2:, 1:-1] + p[:, 2:, 2:]) - (p[:, :-2, :-2] + 2 * p[:, :-2, 1:-1] + p[:, :-2, 2:])
    return np.sqrt(gx ** 2 + gy ** 2).mean(axis=(1, 2))


def _noise_immerkaer(y: np.ndarray) -> np.ndarray:
    """sigma = sqrt(pi/2) / (6 (W-2)(H-2)) * sum |I * N|,  N = [[1,-2,1],[-2,4,-2],[1,-2,1]]."""
    c = y[:, 1:-1, 1:-1]
    conv = (y[:, :-2, :-2] + y[:, :-2, 2:] + y[:, 2:, :-2] + y[:, 2:, 2:]
            - 2 * (y[:, :-2, 1:-1] + y[:, 2:, 1:-1] + y[:, 1:-1, :-2] + y[:, 1:-1, 2:]) + 4 * c)
    h, w = y.shape[1:]
    return np.sqrt(np.pi / 2) * np.abs(conv).sum(axis=(1, 2)) / (6 * (w - 2) * (h - 2))


def _entropy(y8: np.ndarray) -> np.ndarray:
    out = np.empty(len(y8))
    for i, im in enumerate(y8):
        p = np.bincount(im.ravel(), minlength=256) / im.size
        p = p[p > 0]
        out[i] = -(p * np.log2(p)).sum()
    return out


def _colorfulness(imgs: np.ndarray) -> np.ndarray:
    x = imgs.astype(np.float64)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    s = np.sqrt(rg.std(axis=(1, 2)) ** 2 + yb.std(axis=(1, 2)) ** 2)
    m = np.sqrt(rg.mean(axis=(1, 2)) ** 2 + yb.mean(axis=(1, 2)) ** 2)
    return s + 0.3 * m


def measure(imgs: np.ndarray) -> dict[str, np.ndarray]:
    """imgs (N, H, W, 3) uint8 -> métricas por imagem."""
    y = luma(imgs)
    y8 = np.clip(np.rint(y * 255), 0, 255).astype(np.uint8)
    return {
        "sharpness": _sharpness(y),
        "noise": _noise_immerkaer(y),
        "contrast": y.std(axis=(1, 2)),
        "entropy": _entropy(y8),
        "colorfulness": _colorfulness(imgs),
    }


BRISQUE_MIN_SIDE = 128


def _brisque(imgs: np.ndarray, min_side: int = BRISQUE_MIN_SIDE) -> np.ndarray | None:
    """BRISQUE (Mittal et al., 2012) via piq, em CPU. Menor = melhor.

    Imagens menores que ``min_side`` são ampliadas (bicúbica do zero) antes: no CIFAR
    (32 px) o BRISQUE nativo é inconsistente (blur sigma=2 não sai pior que sigma=1);
    ampliado para 128 px ele piora em 100 % das imagens com blur e 97 % com JPEG Q10.
    Limitação conhecida: ampliado, não detecta ruído aditivo (não é uma das técnicas).
    """
    try:
        import piq
        import torch
    except ImportError:
        return None
    up = int(np.ceil(min_side / min(imgs.shape[1:3])))
    if up > 1:
        from .techniques.superres import resize

        imgs = np.stack([resize(im, im.shape[0] * up, im.shape[1] * up) for im in imgs])
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        try:
            return piq.brisque(x, data_range=1.0, reduction="none").numpy()
        except AssertionError:  # imagem degenerada (ex.: quase uniforme): o piq aborta o lote inteiro
            out = np.full(len(x), np.nan)
            for i in range(len(x)):
                try:
                    out[i] = float(piq.brisque(x[i : i + 1], data_range=1.0, reduction="none")[0])
                except AssertionError:
                    pass
            return out


def compare(orig: dict[str, np.ndarray], proc_imgs: np.ndarray, brisque_orig=None, brisque: bool = False) -> dict:
    """Variação relativa (proc - orig) / orig de cada métrica, por imagem."""
    proc = measure(proc_imgs)
    out = {}
    for k in METRICS:
        out[f"q_{k}"] = proc[k]
        out[f"dq_{k}"] = (proc[k] - orig[k]) / (np.abs(orig[k]) + 1e-8)
    if brisque and brisque_orig is not None:
        b = _brisque(proc_imgs)
        if b is not None:
            out["q_brisque"] = b
            out["q_brisque_orig"] = brisque_orig  # para resumir como razão das médias (ver brisque_fix.py)
            out["dq_brisque"] = (b - brisque_orig) / (np.abs(brisque_orig) + 1e-8)  # BRISQUE: menor = melhor
    return out


# ---------------------------------------------------------------------------
# LPIPS (Zhang et al., 2018): distância perceptual COM referência, preditor de comparação
# ---------------------------------------------------------------------------
LPIPS_MIN_SIDE = 64
_LPIPS: dict = {}


def lpips_distance(orig: np.ndarray, proc: np.ndarray, device: str = "cpu", net: str = "alex") -> np.ndarray | None:
    """LPIPS por imagem entre original e processada (maior = mais diferente).

    Imagens menores que 64 px (CIFAR) são ampliadas por interpolação bilinear antes:
    com 32 px as camadas finais da AlexNet ficam com 1x1 pixel.
    """
    try:
        import lpips
        import torch
    except ImportError:
        return None
    key = (net, device)
    if key not in _LPIPS:
        _LPIPS[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    m = _LPIPS[key]

    def t(x):
        x = torch.from_numpy(np.ascontiguousarray(x)).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        if min(x.shape[2:]) < LPIPS_MIN_SIDE:
            s = LPIPS_MIN_SIDE / min(x.shape[2:])
            x = torch.nn.functional.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        return x

    with torch.no_grad():
        return m(t(orig), t(proc)).flatten().float().cpu().numpy()
