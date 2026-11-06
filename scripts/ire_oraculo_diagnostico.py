"""Por que o IRE calibrado só com filtros ideais não serve em 224 px? Três diagnósticos.

    python scripts/ire_oraculo_diagnostico.py [--runs pub_cifar10_s0 pub_imagenette_s0 pub_imagenet]

1. Cobertura: a faixa de cada componente (energia por banda, cor, perda de coerência) nas condições dos filtros ideais
   cobre a das técnicas reais? Testa a hipótese "os filtros não produzem a perda de coerência das técnicas reais".
2. Expoente: ajusta nos filtros ideais com o γ livre e com o γ das técnicas reais fixado. Testa "o problema é o regime
   (perturbações grandes × finas)".
3. Subconjuntos: ajusta só com parte dos filtros (sem o anel mais baixo, só remover, só dobrar, só passa-baixa/alta,
   só filtros com flip < 30 %) e prevê as técnicas reais.
Saída: docs/resultados/pub/ire_oraculo_diagnostico.csv (2 e 3) e ire_oraculo_cobertura.csv (1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ire_transfer import _load  # noqa: E402

from pdi_spectral.thresholds import _r2, fit_ire, ire_matrix, ire_prob  # noqa: E402

R = Path("results")
DOCS = Path("docs/resultados/pub")
SUBSETS = {
    "todos os filtros": lambda c: c.str.len() > 0,
    "sem o anel mais baixo": lambda c: ~c.str.contains(r"@0$") | c.str.startswith("oracle_highpass"),
    "só remover anel (ganho 0)": lambda c: c.str.startswith("oracle_band_g0"),
    "só dobrar anel (ganho 2)": lambda c: c.str.startswith("oracle_band_g2"),
    "só passa-baixa/alta": lambda c: c.str.startswith(("oracle_lowpass", "oracle_highpass")),
}


def _eval(g, Xm, Xo, yo, gamma):
    th = fit_ire(Xo, yo, 1e-4, gamma)
    c = g.assign(p=ire_prob(Xm, th)).groupby("cond_id")[["flip", "p"]].mean()
    return dict(r2=_r2(c.flip.to_numpy(), c.p.to_numpy()), vies=float((c.p - c.flip).mean()),
                spearman=float(c.corr("spearman").iloc[0, 1]))


def coverage(run: str) -> pd.DataFrame:
    ds, feats, dm = _load(run)
    _, _, do = _load(run, "oracle")
    cols = [c for c in dm.columns if c.startswith(("D_f", "Dc_f", "coh_"))]
    out = []
    for kind, d in (("oraculo", do), ("real", dm)):
        x = d[["cond_id"] + cols].copy()
        for c in [c for c in cols if c.startswith("coh_")]:
            x["incoh_" + c[4:]] = 1 - x.pop(c).clip(-1, 1)
        out.append(x.groupby("cond_id").mean().assign(kind=kind))
    o, m = out[0].drop(columns="kind"), out[1].drop(columns="kind")
    return pd.DataFrame({"dataset": ds, "feature": o.columns, "oraculo_max": o.max().values,
                         "real_max": m.max().values, "real_acima_do_oraculo": [(m[c] > o[c].max() * 1.0001).mean()
                                                                                for c in o.columns]})


def diagnose(run: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ds, feats, dm = _load(run)
    _, _, do = _load(run, "oracle")
    w = pd.read_csv(R / run / "analysis" / "ire_weights.csv")
    rows = []
    for m in sorted(dm.model.unique()):
        g = dm[dm.model == m].reset_index(drop=True)
        o = do[do.model == m].reset_index(drop=True)
        Xm = ire_matrix(g, feats)
        cf = o.groupby("cond_id").flip.mean()
        gam = float(w[(w.model == m) & (w.fit == "all") & (w.feature == "_gamma")].w.iloc[0])
        subsets = dict(SUBSETS)
        subsets["só filtros com flip < 30 %"] = lambda c: c.isin(cf[cf < 0.3].index)
        for nome, f in subsets.items():
            idx = np.flatnonzero(f(o.cond_id).to_numpy())
            idx = rng.choice(idx, min(len(idx), 200_000), replace=False)
            Xo, yo = ire_matrix(o.iloc[idx], feats), o.flip.to_numpy().astype(float)[idx]
            rows.append(dict(dataset=ds, run=run, model=m, teste=nome, gamma="livre", **_eval(g, Xm, Xo, yo, None)))
            if nome == "todos os filtros":
                rows.append(dict(dataset=ds, run=run, model=m, teste=nome, gamma="das técnicas reais",
                                 **_eval(g, Xm, Xo, yo, gam)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["pub_cifar10_s0", "pub_imagenette_s0", "pub_imagenet"])
    a = ap.parse_args()
    cov = pd.concat([coverage(r) for r in a.runs], ignore_index=True)
    cov.to_csv(DOCS / "ire_oraculo_cobertura.csv", index=False)
    d = pd.concat([diagnose(r) for r in a.runs], ignore_index=True)
    d.to_csv(DOCS / "ire_oraculo_diagnostico.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["dataset", "teste", "gamma"])[["r2", "vies", "spearman"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
