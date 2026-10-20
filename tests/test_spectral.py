"""O quantificador espectral mede o que diz medir."""
import numpy as np

from pdi_spectral import techniques as T
from pdi_spectral.spectral import SpectralAnalyzer, SpectralConfig
from pdi_spectral.verify import synthetic_images

IMGS = synthetic_images(4, 64, seed=2)
AN = SpectralAnalyzer(SpectralConfig(), (64, 64))


def _cmp(name, **p):
    proc = np.stack([T.apply(name, im, p) for im in IMGS])
    return AN.compare(IMGS, proc)


def test_identity_is_zero():
    m = AN.compare(IMGS, IMGS.copy())
    for b in ("low", "mid", "high"):
        assert np.allclose(m[f"dErel_{b}"], 0)
        assert np.allclose(m[f"coh_{b}"], 1, atol=1e-6)


def test_bands_partition_total_energy():
    m = AN.compare(IMGS, IMGS)
    Fx, _ = AN.transform(IMGS)
    prof = AN.profile(Fx)
    assert np.allclose(prof.sum(1), m["E_tot_orig"])
    assert np.allclose(m["E_low_orig"] + m["E_mid_orig"] + m["E_high_orig"], m["E_tot_orig"])


def test_energy_is_pixel_variance_without_window():
    an = SpectralAnalyzer(SpectralConfig(window="none"), (64, 64))
    Fx, _ = an.transform(IMGS)
    from pdi_spectral.techniques._core import luma

    assert np.allclose(an.profile(Fx).sum(1), luma(IMGS).var(axis=(1, 2)))


def test_sinusoid_lands_in_right_band():
    n = 64
    xx = np.arange(n)[None, :].repeat(n, 0)
    for f, band in ((4 / 64, "low"), (12 / 64, "mid"), (24 / 64, "high")):
        img = (127 + 100 * np.sin(2 * np.pi * f * xx)).astype(np.uint8)
        img = np.stack([img] * 3, -1)[None]
        flat = np.full_like(img, 127)
        m = AN.compare(flat, img, with_quality=False)
        e = {b: m[f"E_{b}_proc"][0] for b in ("low", "mid", "high")}
        assert max(e, key=e.get) == band, (f, e)


def test_smoothing_removes_and_sharpening_adds_high_freq():
    assert _cmp("gaussian", sigma=1.5)["dErel_high"].mean() < 0
    assert _cmp("unsharp", amount=2.0, sigma=1.0)["dErel_high"].mean() > 0


def test_oracle_bandstop_removes_band():
    m = _cmp("oracle_band", r_lo=0.25, r_hi=0.7072, gain=0.0)
    assert (m["dB_high"] < -10).all()  # windowing deixa um pequeno vazamento
