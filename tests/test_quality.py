"""Métricas de qualidade sem referência (objetivo perceptual de cada técnica)."""
import numpy as np
import pytest

from pdi_spectral import quality as Q
from pdi_spectral import techniques as T
from pdi_spectral.verify import synthetic_images

IMGS = synthetic_images(4, 64, seed=3)


def _gray(v: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(np.rint(v), 0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def test_constant_image():
    m = Q.measure(_gray(np.full((1, 32, 32), 128.0)))
    for k in Q.METRICS:
        assert np.isclose(m[k][0], 0.0), k


def test_entropy_of_uniform_histogram_is_8_bits():
    # cada um dos 256 níveis aparece 4 vezes; no cinza Y = R = G = B, então Y8 reproduz os níveis
    v = np.repeat(np.arange(256), 4).reshape(1, 32, 32).astype(float)
    assert np.isclose(Q.measure(_gray(v))["entropy"][0], 8.0)


def test_contrast_is_luma_std():
    v = np.where(np.arange(32)[None, None, :] < 16, 64.0, 192.0).repeat(32, axis=1)
    assert np.isclose(Q.measure(_gray(v))["contrast"][0], 64 / 255, atol=1e-3)


def test_noise_estimate_recovers_sigma_on_flat_image():
    sigma = 8.0
    v = 128 + np.random.default_rng(0).normal(0, sigma, (4, 128, 128))
    est = Q.measure(_gray(v))["noise"] * 255
    assert np.allclose(est, sigma, rtol=0.1)


def test_colorfulness_zero_for_gray_positive_for_color():
    assert np.allclose(Q.measure(_gray(np.random.default_rng(0).uniform(0, 255, (2, 16, 16))))["colorfulness"], 0)
    red = np.zeros((1, 16, 16, 3), np.uint8)
    red[..., 0] = 200
    assert Q.measure(red)["colorfulness"][0] > 50


@pytest.mark.parametrize("name,params,metric,sign", [
    ("unsharp", {"amount": 1.5, "sigma": 1.0}, "sharpness", +1),
    ("gaussian", {"sigma": 1.5}, "sharpness", -1),
    ("gaussian", {"sigma": 1.5}, "noise", -1),
    ("hist_eq", {"alpha": 1.0}, "contrast", +1),
])
def test_technique_moves_its_objective(name, params, metric, sign):
    proc = np.stack([T.apply(name, im, params) for im in IMGS])
    d = Q.compare(Q.measure(IMGS), proc)[f"dq_{metric}"]
    assert (np.sign(d) == sign).all(), d


def test_compare_identity_is_zero_and_relative():
    orig = Q.measure(IMGS)
    out = Q.compare(orig, IMGS)
    for k in Q.METRICS:
        assert np.allclose(out[f"dq_{k}"], 0) and np.allclose(out[f"q_{k}"], orig[k])
    assert "q_brisque" not in out  # BRISQUE só quando pedido


def test_brisque_upscales_small_images_and_flags_blur():
    pytest.importorskip("piq")
    x = synthetic_images(4, 32, seed=5)
    blur = np.stack([T.apply("gaussian", im, {"sigma": 2.0}) for im in x])
    b0, b1 = Q._brisque(x), Q._brisque(blur)
    assert b0.shape == (4,) and np.isfinite(b0).all()
    assert (b1 > b0).all()  # menor = melhor; o blur forte deve piorar todas
    out = Q.compare(Q.measure(x), blur, brisque_orig=b0, brisque=True)
    assert (out["dq_brisque"] > 0).all()
