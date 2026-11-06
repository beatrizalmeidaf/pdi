"""O IRE soma o dano anel por anel; é isso que impede os filtros ideais de calibrá-lo em 224 px?

    python scripts/ire_oraculo_aditividade.py

Os filtros mexem em UM anel por vez; as técnicas reais espalham a perturbação por muitos anéis. Com expoente γ < 1 aplicado
a cada termo (forma atual, Σ w·x^γ), espalhar a mesma energia aumenta o risco previsto. Ajustada em perturbações
concentradas e aplicada a espalhadas, a forma atual exageraria o dano. A forma alternativa aplica o expoente sobre a soma,
(Σ w·x)^γ, que não depende de como a energia se distribui entre os anéis.
Ajusta as duas formas só nos filtros ideais e prevê as 14 técnicas reais (R², viés), por base e modelo.
Saída: docs/resultados/pub/ire_oraculo_aditividade.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from ire_transfer import _load  # noqa: E402

from pdi_spectral.thresholds import _r2, fit_ire, ire_matrix, ire_prob  # noqa: E402

RUNS = ["pub_cifar10_s0", "pub_imagenette_s0", "pub_imagenet"]


def fit_sum_power(X: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> np.ndarray:
    """MLE de P = 1 − exp(−(Σ w_j x_j)^γ), w >= 0, γ em [0,05; 3]. Devolve [w (unidades de X), γ]."""
    scale = X.mean(0) + 1e-12
    Z = X / scale
    n, d = Z.shape

    def f(th):
        w, g = th[:d], th[d]
        u = Z @ w + 1e-9
        s = u ** g
        nll = (-(y * np.log(-np.expm1(-s) + 1e-300)) + (1 - y) * s).mean() + l2 * w @ w
        r = (1 - y) - y / np.expm1(np.maximum(s, 1e-12))      # d nll / d s
        gw = (Z.T @ (r * g * u ** (g - 1))) / n + 2 * l2 * w
        gg = np.mean(r * s * np.log(u))
        return nll, np.r_[gw, gg]

    th0 = np.r_[np.full(d, 0.05), 1.0]
    res = minimize(f, th0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * d + [(0.05, 3.0)])
    return np.r_[res.x[:d] / scale, res.x[d]]


def prob_sum_power(X, th):
    return -np.expm1(-np.clip(X @ th[:-1], 0, None) ** th[-1])


def main(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for run in RUNS:
        ds, feats, dm = _load(run)
        _, _, do = _load(run, "oracle")
        for m in sorted(dm.model.unique()):
            g = dm[dm.model == m].reset_index(drop=True)
            o = do[do.model == m].reset_index(drop=True)
            idx = rng.choice(len(o), min(len(o), 200_000), replace=False)
            Xo, yo = ire_matrix(o.iloc[idx], feats), o.flip.to_numpy().astype(float)[idx]
            Xm = ire_matrix(g, feats)
            for forma, p in (("Σ w·x^γ (atual)", ire_prob(Xm, fit_ire(Xo, yo, 1e-4, None))),
                             ("(Σ w·x)^γ", prob_sum_power(Xm, fit_sum_power(Xo, yo)))):
                c = g.assign(p=p).groupby("cond_id")[["flip", "p"]].mean()
                rows.append(dict(dataset=ds, model=m, forma=forma, r2=_r2(c.flip.to_numpy(), c.p.to_numpy()),
                                 vies=float((c.p - c.flip).mean()), spearman=float(c.corr("spearman").iloc[0, 1])))
            print(rows[-2], rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("docs/resultados/pub/ire_oraculo_aditividade.csv", index=False)
    print(d.groupby(["dataset", "forma"])[["r2", "vies", "spearman"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
