"""Filtros de banda com borda suave calibram o IRE em 224 px? Teste da hipótese do corte abrupto.

    python scripts/ire_oraculo_suave.py             # depois de scripts/run_oraculo_suave.sh
    python scripts/ire_oraculo_suave.py --parcial   # filtros com atenuação parcial (scripts/run_oraculo_parcial.sh)

O diagnóstico anterior (ire_oraculo_diagnostico.py) mostrou que os filtros ideais que REMOVEM frequência fazem o IRE
superestimar o dano das técnicas reais em 224 px. Hipótese: o corte abrupto (ondulação de Gibbs) prejudica a rede mais
do que a energia e a coerência globais indicam. Se for isso, os mesmos filtros com borda suave devem calibrar melhor.

Para cada base e modelo, ajusta o IRE só nos filtros (ideais ou suaves; todos, só remoção ou só amplificação) e prevê o
flip por condição das 14 técnicas reais (R², viés, Spearman). Também compara o dano de cada filtro ideal × suave.
Com --parcial, compara os filtros originais (zerar/dobrar um anel) com filtros que mantêm 25, 50 ou 75 % do anel, e com os
dois juntos: testa se falta "dose dentro do anel" no oráculo.
Saída: docs/resultados/pub/ire_oraculo_{suave,parcial}.csv e ire_oraculo_{suave,parcial}_dano.csv.
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

DOCS = Path("docs/resultados/pub")
BASES = [("pub_cifar10_s0", "pub_cifar10_s0_oraculo_suave"), ("pub_imagenette_s0", "pub_imagenette_s0_oraculo_suave"),
         ("pub_imagenet", "pub_imagenet_oraculo_suave")]
SUB = {"todos": lambda c: c.str.len() > 0,
       "só remoção": lambda c: c.str.startswith(("oracle_band_g0", "oracle_lowpass", "oracle_highpass")),
       "só amplificação": lambda c: c.str.startswith("oracle_band_g2")}


def run_base(main_run: str, soft_run: str, seed: int = 0, label: str = "suave") -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ds, feats, dm = _load(main_run)
    _, _, ideal = _load(main_run, "oracle")
    _, _, soft = _load(soft_run, "oracle")
    sets = [("ideal", ideal), (label, soft)]
    if label == "parcial":
        sets.append(("ideal + parcial", pd.concat([ideal, soft.assign(cond_id="p_" + soft.cond_id)], ignore_index=True)))
    rows = []
    for m in sorted(dm.model.unique()):
        g = dm[dm.model == m].reset_index(drop=True)
        Xm = ire_matrix(g, feats)
        for filtro, o in sets:
            o = o[o.model == m].reset_index(drop=True)
            for nome, f in SUB.items():
                idx = np.flatnonzero(f(o.cond_id.str.replace("p_", "", regex=False)).to_numpy())
                if len(idx) == 0:
                    continue
                idx = rng.choice(idx, min(len(idx), 200_000), replace=False)
                th = fit_ire(ire_matrix(o.iloc[idx], feats), o.flip.to_numpy().astype(float)[idx], 1e-4, None)
                c = g.assign(p=ire_prob(Xm, th)).groupby("cond_id")[["flip", "p"]].mean()
                rows.append(dict(dataset=ds, model=m, filtro=filtro, subconjunto=nome,
                                 r2=_r2(c.flip.to_numpy(), c.p.to_numpy()), vies=float((c.p - c.flip).mean()),
                                 spearman=float(c.corr("spearman").iloc[0, 1]), gamma=float(th[-1])))
    # dano de cada filtro: ideal × suave (flip médio e energia removida/adicionada média)
    dano = []
    for filtro, o in (("ideal", ideal), (label, soft)):
        e = o.assign(D=o[[c for c in o.columns if c.startswith("D_f")]].sum(axis=1))
        dano.append(e.groupby("cond_id").agg(flip=("flip", "mean"), D=("D", "mean")).assign(filtro=filtro, dataset=ds))
    print(f"[suave] {ds}: ok", flush=True)
    return pd.DataFrame(rows), pd.concat(dano).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parcial", action="store_true")
    a = ap.parse_args()
    label = "parcial" if a.parcial else "suave"
    bases = [(m, s.replace("oraculo_suave", f"oraculo_{label}")) for m, s in BASES]
    res = [run_base(m, s, label=label) for m, s in bases if (Path("results") / s / "oracle").exists()]
    d = pd.concat([r[0] for r in res], ignore_index=True)
    dano = pd.concat([r[1] for r in res], ignore_index=True)
    d.to_csv(DOCS / f"ire_oraculo_{label}.csv", index=False)
    dano.to_csv(DOCS / f"ire_oraculo_{label}_dano.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["dataset", "subconjunto", "filtro"])[["r2", "vies", "spearman", "gamma"]].mean().round(2)
          .to_string())
    w = dano.pivot_table(index=["dataset", "cond_id"], columns="filtro", values=["flip", "D"])
    w["flip_por_D_ideal"] = w[("flip", "ideal")] / w[("D", "ideal")]
    w[f"flip_por_D_{label}"] = w[("flip", label)] / w[("D", label)]
    print((w[["flip_por_D_ideal", f"flip_por_D_{label}"]].groupby("dataset").median()).round(2).to_string())


if __name__ == "__main__":
    main()
