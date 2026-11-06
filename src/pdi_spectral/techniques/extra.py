"""Técnicas de validação externa: 6 técnicas que NÃO entram no ajuste do IRE nem na calibração da margem.

Escolhidas depois da rodada final para testar o IRE e a margem conformal em técnicas de fato novas, cobrindo famílias
diferentes das 14 originais:
  box          filtro da média k × k                      suavização linear
  perona_malik difusão anisotrópica (Perona & Malik, 1990)  suavização que preserva bordas
  gamma        correção gama                               tom
  stretch      alongamento de contraste por percentis       contraste
  webp         compressão WebP (PIL/libwebp)               outro codec
  resample     reduz e volta ao tamanho (bilinear)          resolução

Cada função recebe RGB uint8 (H, W, 3) e devolve o mesmo formato. Versões do zero aqui; as de referência
(OpenCV/PIL) ficam no fim do arquivo e são comparadas nos testes.
"""
from __future__ import annotations

import io

import numpy as np

from ._core import box_filter, luma, to_float, to_uint8


# ---------------------------------------------------------------------------
# do zero
# ---------------------------------------------------------------------------
def box(img: np.ndarray, k: int) -> np.ndarray:
    """Média k × k (k ímpar), borda refletida sem repetir a borda (como o BORDER_REFLECT_101)."""
    return to_uint8(box_filter(to_float(img), int(k)))


def perona_malik(img: np.ndarray, n_iter: int, kappa: float = 0.1, lam: float = 0.2) -> np.ndarray:
    """Difusão anisotrópica de Perona & Malik, condutância exponencial, 4 vizinhos, por canal.

    kappa em unidades de intensidade [0, 1]; lam <= 0,25 para estabilidade.
    """
    x = to_float(img)
    for _ in range(int(n_iter)):
        p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
        dn = p[:-2, 1:-1] - x
        ds = p[2:, 1:-1] - x
        de = p[1:-1, 2:] - x
        dw = p[1:-1, :-2] - x
        c = lambda d: np.exp(-(d / kappa) ** 2)  # noqa: E731
        x = x + lam * (c(dn) * dn + c(ds) * ds + c(de) * de + c(dw) * dw)
    return to_uint8(x)


def gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Correção gama por tabela: y = 255 · (x / 255)^γ, nos 3 canais."""
    lut = to_uint8((np.arange(256) / 255.0) ** float(gamma))
    return lut[img]


def _stretch_bounds(img: np.ndarray, p: float) -> tuple[float, float]:
    y = luma(img)
    lo, hi = np.percentile(y, [p, 100.0 - p])
    return float(lo), float(hi)


def stretch(img: np.ndarray, p: float) -> np.ndarray:
    """Alongamento linear de contraste: os percentis p e 100 − p da luminância viram 0 e 1 (p em %)."""
    lo, hi = _stretch_bounds(img, p)
    if hi - lo < 1e-6:
        return img.copy()
    return to_uint8((to_float(img) - lo) / (hi - lo))


def webp(img: np.ndarray, quality: int) -> np.ndarray:
    """WebP com perdas (libwebp via PIL). Não há versão do zero: codec externo, como os modelos de SR."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="WEBP", quality=int(quality), method=4)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _bilinear(x: np.ndarray, h: int, w: int) -> np.ndarray:
    """Reamostragem bilinear (centros de pixel alinhados, sem antialias), como o cv2.INTER_LINEAR."""
    H, W = x.shape[:2]
    ys = np.clip((np.arange(h) + 0.5) * H / h - 0.5, 0, H - 1)
    xs = np.clip((np.arange(w) + 0.5) * W / w - 0.5, 0, W - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = x[y0][:, x0] * (1 - wx) + x[y0][:, x1] * wx
    bot = x[y1][:, x0] * (1 - wx) + x[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def resample(img: np.ndarray, scale: float) -> np.ndarray:
    """Reduz para scale × o tamanho e volta ao original, ambos bilineares (perde detalhe fino como um zoom digital)."""
    H, W = img.shape[:2]
    h, w = max(1, round(H * scale)), max(1, round(W * scale))
    x = to_float(img)
    return to_uint8(_bilinear(_bilinear(x, h, w), H, W))


# ---------------------------------------------------------------------------
# referência (OpenCV), para validação e para o backend rápido
# ---------------------------------------------------------------------------
def ref_box(img, k):
    import cv2

    return cv2.blur(img, (int(k), int(k)), borderType=cv2.BORDER_REFLECT_101)


def ref_gamma(img, gamma):
    import cv2

    lut = np.clip(np.rint(255.0 * (np.arange(256) / 255.0) ** float(gamma)), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def ref_stretch(img, p):
    import cv2

    lo, hi = _stretch_bounds(img, p)
    if hi - lo < 1e-6:
        return img.copy()
    lut = np.clip(np.rint(255.0 * ((np.arange(256) / 255.0) - lo) / (hi - lo)), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def ref_resample(img, scale):
    import cv2

    H, W = img.shape[:2]
    h, w = max(1, round(H * scale)), max(1, round(W * scale))
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
