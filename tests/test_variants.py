"""Variantes de dose baixa (clahe_blend, jpeg444, hist_eq alpha) e SR aprendida."""
import numpy as np
import pytest

from pdi_spectral import techniques as T
from pdi_spectral.techniques._core import psnr
from pdi_spectral.verify import synthetic_images

IMGS = synthetic_images(3, 32, seed=2)
SR_DIR = "data/sr_models"


@pytest.mark.parametrize("name,params", [("clahe_blend", {"clip_limit": 2.0}), ("hist_eq", {})])
def test_alpha_zero_is_identity_and_one_is_full(name, params):
    full = "clahe" if name == "clahe_blend" else name
    for im in IMGS:
        # alpha = 0 só passa pela ida e volta YCbCr: diferença de arredondamento
        assert np.abs(T.apply(name, im, {**params, "alpha": 0.0}).astype(int) - im).max() <= 1
        assert np.array_equal(T.apply(name, im, {**params, "alpha": 1.0}), T.apply(full, im, params))


def test_clahe_blend_dose_is_monotonic():
    d = [np.abs(T.apply("clahe_blend", IMGS[0], {"clip_limit": 2.0, "alpha": a}).astype(int) - IMGS[0]).mean()
         for a in (0.1, 0.3, 0.6, 1.0)]
    assert all(a < b for a, b in zip(d, d[1:]))


def test_clahe_clip_below_two_is_identical_at_32px():
    # documentado em contrast.clahe: blocos 16x16 arredondam o clip
    a = T.apply("clahe", IMGS[0], {"clip_limit": 1.0})
    assert np.array_equal(a, T.apply("clahe", IMGS[0], {"clip_limit": 1.9}))


def test_jpeg444_beats_420_at_high_quality():
    for im in IMGS:
        p444 = psnr(T.apply("jpeg444", im, {"quality": 100}), im)
        p420 = psnr(T.apply("jpeg", im, {"quality": 100}), im)
        assert p444 > p420 + 3


def test_jpeg444_matches_pil():
    pytest.importorskip("PIL")
    a = np.stack([T.apply("jpeg444", im, {"quality": 50}, "scratch") for im in IMGS])
    b = np.stack([T.apply("jpeg444", im, {"quality": 50}, "reference") for im in IMGS])
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 3.0


@pytest.mark.parametrize("name", ["sr_espcn", "sr_fsrcnn"])
@pytest.mark.parametrize("scale", [2, 3, 4])
def test_learned_sr_shape_and_backends(name, scale):
    pytest.importorskip("cv2")
    from pdi_spectral.techniques.superres import sr_available

    if not sr_available(T.SR_TECHNIQUES[name], scale, SR_DIR):
        pytest.skip("pesos de SR ausentes (make download-sr)")
    p = {"scale": scale, "_model_dir": SR_DIR}
    for im in IMGS:
        a = T.apply(name, im, p, "scratch")
        b = T.apply(name, im, p, "reference")
        assert np.abs(a.astype(int) - b.astype(int)).mean() < 2.0
        # parte da mesma entrada reduzida que a bicúbica: não pode ficar muito pior
        assert psnr(a, im) > psnr(T.apply("sr_bicubic", im, {"scale": scale}), im) - 1.0
