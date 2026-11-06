"""Técnicas de validação externa (techniques/extra.py): versão do zero × referência e propriedades básicas."""
import numpy as np
import pytest

from pdi_spectral import techniques as T


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:48, 0:48]
    base = np.stack([yy * 4, xx * 4, (xx + yy) * 2], -1) + rng.normal(0, 12, (48, 48, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name,params", [("box", {"k": 5}), ("gamma", {"gamma": 0.7}), ("stretch", {"p": 2.0}),
                                         ("resample", {"scale": 0.5})])
def test_do_zero_igual_a_referencia(img, name, params):
    pytest.importorskip("cv2")
    a = T.apply(name, img, params, "scratch").astype(float)
    b = T.apply(name, img, params, "reference").astype(float)
    assert np.abs(a - b).mean() <= 0.5


@pytest.mark.parametrize("name,params", [("box", {"k": 1}), ("gamma", {"gamma": 1.0}), ("perona_malik", {"n_iter": 0}),
                                         ("resample", {"scale": 1.0})])
def test_identidade(img, name, params):
    assert np.array_equal(T.apply(name, img, params, "scratch"), img)


def test_perona_malik_suaviza_e_preserva_borda():
    img = np.zeros((32, 32, 3), np.uint8)
    img[:, 16:] = 200                                     # borda forte
    rng = np.random.default_rng(1)
    noisy = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    out = T.apply("perona_malik", noisy, {"n_iter": 8}, "scratch").astype(float)
    assert out[:, :14].std() < noisy[:, :14].astype(float).std()        # ruído cai nas regiões planas
    assert out[:, 17:].mean() - out[:, :15].mean() > 180                  # a borda continua


def test_stretch_satura_os_percentis(img):
    out = T.apply("stretch", img, {"p": 5.0}, "scratch")
    assert out.min() == 0 and out.max() == 255


def test_webp_qualidade_menor_erra_mais(img):
    e = [np.abs(T.apply("webp", img, {"quality": q}, "scratch").astype(float) - img).mean() for q in (95, 30)]
    assert e[1] > e[0]
