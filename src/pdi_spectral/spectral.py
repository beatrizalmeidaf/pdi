"""Quantificador espectral: ΔE por banda radial (lacunas 2 e 4).

Para cada par (original x, processada y), na luminância:
  1. remove-se a média (o termo DC é medido à parte: ``d_dc``);
  2. aplica-se uma janela de Hann 2D (reduz o vazamento causado pela
     periodicidade implícita da DFT, que cria falsa energia de alta frequência
     nas bordas);
  3. FFT 2D; espectro de potência P = |F|^2 normalizado de forma que
     sum(P) ~ variância dos pixels (energia AC por pixel);
  4. a potência é somada em anéis de frequência radial normalizada
     r = sqrt(fx^2 + fy^2) (ciclos/pixel): ``n_bins`` anéis finos em (0, 0.5]
     + um anel de "cantos" (0.5, 0.707], e bandas agregadas (baixa/média/alta).

Medidas por banda b (todas por imagem):
  dErel_b  = (E_b(y) - E_b(x)) / E_tot(x)     variação de energia, aditiva entre bandas
  dB_b     = 10 log10(E_b(y) / E_b(x))         variação relativa na própria banda
  dshape_b = E_b(y)/E_tot(y) - E_b(x)/E_tot(x) mudança da *forma* do espectro (sem ganho global)
  D_b      = E_b(y - x) / E_tot(x)             energia da perturbação na banda (estilo Yin et al.)
  coh_b    = Re<F_x, F_y>_b / sqrt(E_b(x) E_b(y)) coerência (preservação de fase/estrutura)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .techniques._core import luma, psnr, ssim_luma

R_MAX = float(np.sqrt(0.5))  # ~0.7071, canto do espectro
EPS = 1e-12


@dataclass
class SpectralConfig:
    channel: str = "luma"            # luma | rgb (soma da energia dos 3 canais)
    window: str = "hann"             # hann | none
    n_bins: int = 16
    bands: dict = field(default_factory=lambda: {"low": [0.0, 0.1], "mid": [0.1, 0.25], "high": [0.25, R_MAX]})

    @classmethod
    def from_dict(cls, d: dict | None) -> "SpectralConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SpectralAnalyzer:
    def __init__(self, cfg: SpectralConfig, shape: tuple[int, int]):
        self.cfg = cfg
        h, w = shape
        self.h, self.w = h, w
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        r = np.sqrt(fx ** 2 + fy ** 2).ravel()
        self.r = r
        edges = np.linspace(0.0, 0.5, cfg.n_bins + 1)
        self.fine_edges = np.append(edges, R_MAX + 1e-9)
        self.fine_centers = 0.5 * (self.fine_edges[:-1] + self.fine_edges[1:])
        n_fine = cfg.n_bins + 1  # + anel dos cantos
        idx = np.searchsorted(self.fine_edges, r, side="left") - 1  # (e_i, e_{i+1}]
        valid = r > 0
        self.M_fine = np.zeros((h * w, n_fine))
        self.M_fine[np.flatnonzero(valid), idx[valid]] = 1.0
        self.band_names = list(cfg.bands)
        self.M_band = np.zeros((h * w, len(self.band_names)))
        for j, name in enumerate(self.band_names):
            lo, hi = cfg.bands[name]
            hi = R_MAX + 1e-9 if hi >= R_MAX - 1e-6 else hi
            self.M_band[(r > lo) & (r <= hi), j] = 1.0
        if cfg.window == "hann":
            win = np.outer(np.hanning(h), np.hanning(w))
        else:
            win = np.ones((h, w))
        self.win = win
        self.norm = (h * w) ** 2 * np.mean(win ** 2)

    # -- transformada -----------------------------------------------------
    def _planes(self, imgs: np.ndarray) -> np.ndarray:
        """(N,H,W,3) uint8 -> (N,C,H,W) float."""
        if self.cfg.channel == "luma":
            return luma(imgs)[:, None]
        return (imgs.astype(np.float64) / 255.0).transpose(0, 3, 1, 2)

    def transform(self, imgs: np.ndarray):
        x = self._planes(imgs)
        dc = x.mean(axis=(2, 3), keepdims=True)
        F = np.fft.fft2((x - dc) * self.win, axes=(2, 3))
        return F, dc[:, :, 0, 0].mean(axis=1)

    def _energy(self, P: np.ndarray, M: np.ndarray) -> np.ndarray:
        n = P.shape[0]
        return (P.reshape(n, P.shape[1], -1).sum(axis=1)) @ M / self.norm

    def profile(self, F: np.ndarray) -> np.ndarray:
        """Energia por anel fino, (N, n_bins+1)."""
        return self._energy(np.abs(F) ** 2, self.M_fine)

    # -- comparação -------------------------------------------------------
    def compare(self, x_imgs, y_imgs, Fx=None, dcx=None, with_quality: bool = True) -> dict:
        if Fx is None:
            Fx, dcx = self.transform(x_imgs)
        Fy, dcy = self.transform(y_imgs)
        Px, Py = np.abs(Fx) ** 2, np.abs(Fy) ** 2
        Pd = np.abs(Fy - Fx) ** 2
        Cxy = np.real(Fx * np.conj(Fy))
        out: dict[str, np.ndarray] = {}
        tot_x = self._energy(Px, self.M_fine).sum(1)
        tot_y = self._energy(Py, self.M_fine).sum(1)
        tot_d = self._energy(Pd, self.M_fine).sum(1)
        out["d_dc"] = dcy - dcx
        out["E_tot_orig"] = tot_x
        out["E_tot_proc"] = tot_y
        out["dB_tot"] = 10 * np.log10((tot_y + EPS) / (tot_x + EPS))
        out["D_tot"] = tot_d / (tot_x + EPS)

        for M, names in ((self.M_band, self.band_names),
                         (self.M_fine, [f"f{i:02d}" for i in range(self.M_fine.shape[1])])):
            ex, ey, ed, cxy = (self._energy(P, M) for P in (Px, Py, Pd, Cxy))
            for j, nm in enumerate(names):
                out[f"dErel_{nm}"] = (ey[:, j] - ex[:, j]) / (tot_x + EPS)
                out[f"dB_{nm}"] = 10 * np.log10((ey[:, j] + EPS) / (ex[:, j] + EPS))
                out[f"D_{nm}"] = ed[:, j] / (tot_x + EPS)
                if not nm.startswith("f"):
                    out[f"E_{nm}_orig"] = ex[:, j]
                    out[f"E_{nm}_proc"] = ey[:, j]
                    out[f"dshape_{nm}"] = ey[:, j] / (tot_y + EPS) - ex[:, j] / (tot_x + EPS)
                    out[f"coh_{nm}"] = cxy[:, j] / np.sqrt(ex[:, j] * ey[:, j] + EPS)
        if "high" in self.band_names:
            out["hf_frac_delta"] = out["D_high"] / (out["D_tot"] + EPS)
        if with_quality:
            out["psnr"] = np.array([min(psnr(a, b), 100.0) for a, b in zip(x_imgs, y_imgs)])
            out["ssim"] = np.array([ssim_luma(a, b) for a, b in zip(x_imgs, y_imgs)])
        return out
