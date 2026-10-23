"""As técnicas do zero devem reproduzir as bibliotecas de referência."""
import numpy as np
import pytest

from pdi_spectral import techniques as T
from pdi_spectral.techniques._core import psnr
from pdi_spectral.verify import synthetic_images

cv2 = pytest.importorskip("cv2")
IMGS = synthetic_images(3, 64, seed=1)


def _both(name, **p):
    a = np.stack([T.apply(name, im, p, "scratch") for im in IMGS])
    b = np.stack([T.apply(name, im, p, "reference") for im in IMGS])
    return a, b


@pytest.mark.parametrize("name,params,max_mad", [
    ("gaussian", {"sigma": 1.5}, 0.6),
    ("median", {"k": 5}, 0.0),
    ("unsharp", {"amount": 1.5, "sigma": 1.0}, 0.6),
    ("laplacian", {"c": 1.0}, 0.6),
    ("hist_eq", {"alpha": 1.0}, 1.0),
    ("clahe", {"clip_limit": 3.0}, 1.0),
    ("sr_bicubic", {"scale": 2}, 2.0),
    ("jpeg", {"quality": 50}, 3.0),
])
def test_matches_reference(name, params, max_mad):
    a, b = _both(name, **params)
    mad = np.abs(a.astype(int) - b.astype(int)).mean()
    assert mad <= max_mad, f"{name}: diferença média {mad:.3f}"


def test_bilateral_gray_matches_opencv():
    # Cinza replicado em 3 canais: a distância L2 do zero vale sqrt(3)|d|.
    # Compara com o OpenCV em 1 canal usando sigma_r / sqrt(3).
    g = IMGS[..., :1].repeat(3, axis=-1)
    s_s, s_r = 2.0, 0.1
    a = np.stack([T.apply("bilateral", im, {"sigma_s": s_s, "sigma_r": s_r}, "scratch") for im in g])[..., 0]
    b = np.stack([cv2.bilateralFilter(np.ascontiguousarray(im[..., 0]), 9, s_r * 255 / np.sqrt(3), s_s) for im in g])
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 1.0


def test_wiener_interior_matches_scipy():
    # O SciPy estima o ruído com borda de zeros; fixa o mesmo nu nos dois.
    from pdi_spectral.techniques import reference, restoration

    a = np.stack([restoration.wiener(im, 5, noise=0.002) for im in IMGS])
    b = np.stack([reference.wiener(im, 5, noise=0.002) for im in IMGS])
    r = 3
    assert np.abs(a[:, r:-r, r:-r].astype(int) - b[:, r:-r, r:-r].astype(int)).mean() < 1.0


def test_jpeg_quality_monotonic():
    q = [psnr(T.apply("jpeg", IMGS[0], {"quality": k}), IMGS[0]) for k in (90, 50, 10)]
    assert q[0] > q[1] > q[2]


def test_oracle_identity_and_shape():
    x = IMGS[0]
    y = T.apply("oracle_band", x, {"r_lo": 0.1, "r_hi": 0.2, "gain": 1.0})
    assert np.abs(y.astype(int) - x.astype(int)).max() <= 1
