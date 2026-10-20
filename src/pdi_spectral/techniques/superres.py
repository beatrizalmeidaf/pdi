"""Super-resolução (grupo primário): bicúbica do zero vs. redes clássicas (ESPCN, FSRCNN).

Protocolo: a imagem limpa (resolução da rede) é reduzida por um fator s
(aquisição em baixa resolução) e reconstruída de volta ao tamanho original por
(a) interpolação bicúbica ou (b) uma rede de SR. A comparação de interesse é
(b) contra (a): ambas partem da mesma entrada degradada, então a diferença
isola o efeito do "aprimoramento" aprendido (que sintetiza alta frequência).

As redes usam os pesos pré-treinados públicos lidos pelo ``cv2.dnn_superres``
(opencv-contrib-python). É a única técnica cujo núcleo não é implementado do zero.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._core import to_float, to_uint8


def _keys(t: np.ndarray, a: float = -0.5) -> np.ndarray:
    t = np.abs(t)
    t2, t3 = t * t, t * t * t
    w1 = (a + 2) * t3 - (a + 3) * t2 + 1
    w2 = a * t3 - 5 * a * t2 + 8 * a * t - 4 * a
    return np.where(t <= 1, w1, np.where(t < 2, w2, 0.0))


@lru_cache(maxsize=64)
def resample_matrix(n_in: int, n_out: int, antialias: bool = True) -> np.ndarray:
    """Matriz (n_out, n_in) de reamostragem bicúbica (Keys, a=-0.5).

    Na redução, o kernel é alargado pelo fator de escala (anti-aliasing),
    como no PIL. Bordas por replicação.
    """
    scale = n_in / n_out
    s = max(scale, 1.0) if antialias else 1.0
    support = 2.0 * s
    m = np.zeros((n_out, n_in))
    for i in range(n_out):
        c = (i + 0.5) * scale - 0.5
        js = np.arange(int(np.floor(c - support)) + 1, int(np.floor(c + support)) + 1)
        wts = _keys((js - c) / s)
        np.add.at(m[i], np.clip(js, 0, n_in - 1), wts)
    return m / m.sum(axis=1, keepdims=True)


def resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    x = to_float(img)
    mh = resample_matrix(x.shape[0], h)
    mw = resample_matrix(x.shape[1], w)
    H, W, C = x.shape
    rows = (mh @ x.reshape(H, W * C)).reshape(h, W, C)       # reamostra linhas
    out = np.tensordot(rows, mw, axes=(1, 1)).transpose(0, 2, 1)  # reamostra colunas
    return to_uint8(out)


def downscale(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return resize(img, max(1, round(h / scale)), max(1, round(w / scale)))


def sr_bicubic(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return resize(downscale(img, scale), h, w)


_TLS = threading.local()  # a rede do OpenCV não é thread-safe: uma instância por thread


def _load_sr(method: str, scale: int, model_dir: str):
    cache = _TLS.__dict__.setdefault("sr", {})
    key = (method, scale, model_dir)
    if key not in cache:
        import cv2  # opencv-contrib-python

        path = Path(model_dir) / f"{method.upper()}_x{scale}.pb"
        if not path.exists():
            raise FileNotFoundError(f"Pesos de SR não encontrados: {path} (rode `make download`)")
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(path))
        sr.setModel(method.lower(), scale)
        cache[key] = sr
    return cache[key]


def sr_learned(img: np.ndarray, scale: int, method: str = "espcn", _model_dir: str = "data/sr_models",
               _downscale=None, _resize=None) -> np.ndarray:
    h, w = img.shape[:2]
    small = (_downscale or downscale)(img, scale)
    sr = _load_sr(method, int(scale), str(_model_dir))
    up = sr.upsample(np.ascontiguousarray(small[..., ::-1]))[..., ::-1]
    if up.shape[:2] != (h, w):  # ex.: 32/3 -> 11 -> 33
        up = (_resize or resize)(np.ascontiguousarray(up), h, w)
    return np.ascontiguousarray(up)


def sr_available(method: str, scale: int, model_dir: str) -> bool:
    try:
        _load_sr(method, int(scale), str(model_dir))
        return True
    except Exception:
        return False
