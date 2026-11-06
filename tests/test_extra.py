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


def test_filtro_de_banda_suave():
    from pdi_spectral.techniques import oracle

    rng = np.random.default_rng(2)
    img = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
    # soft = 0 reproduz o filtro ideal
    assert np.array_equal(oracle.band_gain(img, 0.1, 0.2, 0.0), oracle.band_gain(img, 0.1, 0.2, 0.0, soft=0.0))
    m = oracle.band_mask(48, 48, 0.1, 0.2, 0.0, soft=0.05)
    r = oracle._radius(48, 48)
    assert m[r == 0].item() == 1.0                       # DC preservado
    assert np.all(m[r > 0.3] == 1.0) and np.all((m >= 0) & (m <= 1))
    assert np.isclose(m[np.isclose(r, 0.15, atol=0.01)].mean(), 0.0, atol=1e-6)   # centro do anel removido
    assert 0 < m[np.isclose(r, 0.1, atol=0.01)].mean() < 1                         # transição na borda


def test_coerencia_por_anel():
    from pdi_spectral.spectral import SpectralAnalyzer, SpectralConfig
    from pdi_spectral.techniques import oracle

    rng = np.random.default_rng(3)
    img = (rng.random((4, 32, 32, 3)) * 255).astype(np.uint8)
    an = SpectralAnalyzer(SpectralConfig.from_dict(None), (32, 32))
    Fx, _ = an.transform(img)
    assert np.allclose(an.coherence_fine(Fx, Fx), 1.0, atol=1e-3)          # imagem igual: coerência 1 em todo anel
    proc = np.stack([oracle.band_gain(im, 0.25, 0.3125, 0.0) for im in img])
    c = an.coherence_fine(Fx, an.transform(proc)[0]).mean(0)
    i = int(np.argmin(c))
    assert 0.25 <= an.fine_centers[i] <= 0.32 and c[i] < 0.5               # o anel removido perde a coerência
