"""Teste reverso: o IRE ajustado nas técnicas reais prevê os filtros de banda?

    python scripts/ire_oraculo_reverso.py

Se os pesos das 14 técnicas reais (ire_weights.csv, fit=all) também previrem os filtros ideais, suaves e parciais, a
falha do oráculo está só na calibração. Se não previrem, a descrição espectral do IRE (anéis, cor e coerência) não
distingue filtros de banda de técnicas reais: mesmas medidas, dano diferente.
Saída: docs/resultados/pub/ire_oraculo_reverso.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ire_transfer import _load  # noqa: E402

from pdi_spectral.thresholds import _r2, ire_matrix, ire_prob  # noqa: E402

BASES = [("pub_cifar10_s0", "pub_cifar10_s0"), ("pub_imagenette_s0", "pub_imagenette_s0"), ("pub_imagenet", "pub_imagenet")]


def main():
    rows = []
    for run, pref in BASES:
        w = pd.read_csv(Path("results") / run / "analysis" / "ire_weights.csv")
        w = w[w.fit == "all"]
        for filtro, r in (("ideal", run), ("suave", f"{pref}_oraculo_suave"), ("parcial", f"{pref}_oraculo_parcial")):
            if not (Path("results") / r / "oracle").exists():
                continue
            ds, feats, o = _load(r, "oracle")
            for m, g in o.groupby("model"):
                th = w[w.model == m].set_index("feature").w.reindex(feats + ["_gamma"]).to_numpy()
                c = g.assign(p=ire_prob(ire_matrix(g, feats), th)).groupby("cond_id")[["flip", "p"]].mean()
                big = c[c.flip > 0.02]
                rows.append(dict(dataset=ds, model=m, filtro=filtro, r2=_r2(c.flip.to_numpy(), c.p.to_numpy()),
                                 spearman=float(c.corr("spearman").iloc[0, 1]), vies=float((c.p - c.flip).mean()),
                                 razao_prev_obs=float((big.p / big.flip).median())))
        print(f"[reverso] {run}: ok", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("docs/resultados/pub/ire_oraculo_reverso.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["dataset", "filtro"])[["r2", "spearman", "vies", "razao_prev_obs"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
