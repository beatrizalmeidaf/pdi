"""Registro das técnicas de pré-processamento.

Cada técnica recebe uma imagem RGB uint8 (H, W, 3) e parâmetros nomeados e
devolve uma imagem do mesmo formato. Há duas implementações por técnica:
``scratch`` (do zero, NumPy) e ``reference`` (OpenCV/PIL/SciPy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import compression, contrast, oracle, reference, restoration, sharpening, smoothing, superres


@dataclass(frozen=True)
class Technique:
    name: str
    family: str          # família da ementa
    group: str           # primary | control | oracle
    expected: str        # efeito espectral esperado: remove | add | redistribute | mixed
    scratch: Callable
    reference: Callable


def _sr_espcn(img, scale, _model_dir="data/sr_models"):
    return superres.sr_learned(img, scale, "espcn", _model_dir)


def _sr_fsrcnn(img, scale, _model_dir="data/sr_models"):
    return superres.sr_learned(img, scale, "fsrcnn", _model_dir)


def _ref_espcn(img, scale, _model_dir="data/sr_models"):
    return reference.sr_learned(img, scale, "espcn", _model_dir)


def _ref_fsrcnn(img, scale, _model_dir="data/sr_models"):
    return reference.sr_learned(img, scale, "fsrcnn", _model_dir)


def _jpeg444(img, quality):
    return compression.jpeg(img, quality, "444")


def _ref_jpeg444(img, quality):
    return reference.jpeg(img, quality, "444")


TECHNIQUES: dict[str, Technique] = {
    t.name: t
    for t in [
        # --- grupo primário (lacuna de escopo) ---
        Technique("hist_eq", "equalizacao", "primary", "redistribute", contrast.hist_eq, reference.hist_eq),
        Technique("clahe", "equalizacao", "primary", "redistribute", contrast.clahe, reference.clahe),
        Technique("unsharp", "agucamento", "primary", "add", sharpening.unsharp_mask, reference.unsharp_mask),
        Technique("laplacian", "agucamento", "primary", "add", sharpening.laplacian_sharpen, reference.laplacian_sharpen),
        Technique("sr_bicubic", "super-resolucao", "primary", "remove", superres.sr_bicubic, reference.sr_bicubic),
        Technique("sr_espcn", "super-resolucao", "primary", "mixed", _sr_espcn, _ref_espcn),
        Technique("sr_fsrcnn", "super-resolucao", "primary", "mixed", _sr_fsrcnn, _ref_fsrcnn),
        # --- grupo de controle / âncora (bem documentado na literatura) ---
        Technique("gaussian", "suavizacao", "control", "remove", smoothing.gaussian, reference.gaussian),
        Technique("median", "suavizacao", "control", "remove", smoothing.median, reference.median),
        Technique("bilateral", "suavizacao", "control", "remove", smoothing.bilateral, reference.bilateral),
        # backend "reference" do Wiener = versão do zero (já rápida); o SciPy estima o ruído de
        # outro jeito e fica só para validação (tests/test_techniques.py)
        Technique("wiener", "restauracao", "control", "remove", restoration.wiener, restoration.wiener),
        Technique("jpeg", "compressao", "control", "remove", compression.jpeg, reference.jpeg),
        # variantes de dose baixa (mesma função, outro eixo de dose)
        Technique("clahe_blend", "equalizacao", "primary", "redistribute", contrast.clahe, reference.clahe),
        Technique("jpeg444", "compressao", "control", "remove", _jpeg444, _ref_jpeg444),
        # --- oráculo espectral (calibração) ---
        Technique("oracle_band", "oraculo", "oracle", "mixed", oracle.band_gain, oracle.band_gain),
        Technique("oracle_lowpass", "oraculo", "oracle", "remove", oracle.lowpass, oracle.lowpass),
        Technique("oracle_highpass", "oraculo", "oracle", "remove", oracle.highpass, oracle.highpass),
    ]
}

SR_TECHNIQUES = {"sr_espcn": "espcn", "sr_fsrcnn": "fsrcnn"}


def apply(name: str, img: np.ndarray, params: dict, backend: str = "scratch") -> np.ndarray:
    t = TECHNIQUES[name]
    fn = t.scratch if backend == "scratch" else t.reference
    out = fn(img, **params)
    assert out.shape == img.shape and out.dtype == np.uint8, (name, out.shape, out.dtype)
    return out


__all__ = ["TECHNIQUES", "SR_TECHNIQUES", "Technique", "apply"]
