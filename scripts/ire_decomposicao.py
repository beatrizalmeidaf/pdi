"""De onde vem o risco previsto pelo IRE? Decomposição por componente, por técnica e modelo.

    python scripts/ire_decomposicao.py [--run pub_imagenet] [--stride 8]

Para cada (imagem, condição), o IRE é uma soma de termos w_j · x_j^γ. A parte de cada grupo é a média dos seus termos
dividida pela média do IRE total, nas imagens da técnica:
  lum_baixa        energia da perturbação de luminância nos 3 primeiros anéis
  lum_media_alta   energia de luminância nos demais anéis
  cor              energia da perturbação nos canais de cor (Cb, Cr)
  coerencia        perda de coerência espectral (estrutura e fase) nas bandas baixa, média e alta
Usa uma amostra de row groups do spectral.parquet (--stride). Saída: docs/resultados/pub/ire_decomposicao_<run>.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pdi_spectral.thresholds import ire_matrix

GROUPS = ["lum_baixa", "lum_media_alta", "cor", "coerencia"]


def group_of(f: str) -> str:
    if f.startswith("D_f"):
        return "lum_baixa" if int(f[3:]) <= 2 else "lum_media_alta"
    return "cor" if f.startswith("Dc") else "coerencia"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pub_imagenet")
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()
    base = Path("results") / a.run
    w = pd.read_csv(base / "analysis" / "ire_weights.csv")
    w = w[w.fit == "all"]
    feats = [f for f in w.feature.unique() if f != "_gamma"]
    raw = sorted({("coh_" + f[6:]) if f.startswith("incoh_") else f for f in feats})
    main_dir = next(p for p in (base / "main").iterdir() if p.is_dir() and not p.name.startswith("_"))
    pf = pq.ParquetFile(main_dir / "spectral.parquet")
    s = pf.read_row_groups(list(range(0, pf.num_row_groups, a.stride)), columns=["cond_id"] + raw).to_pandas()
    s = s.merge(pd.read_csv(main_dir / "conditions.csv")[["cond_id", "technique"]])
    X = ire_matrix(s, feats)
    grp = np.array([group_of(f) for f in feats])
    rows = []
    for m, g in w.groupby("model"):
        wm = g.set_index("feature").w
        C = (X ** wm["_gamma"]) * wm[feats].to_numpy()
        for t in s.technique.unique():
            sel = (s.technique == t).to_numpy()
            tot = C[sel].sum(1).mean()
            if tot <= 0:
                continue
            rows.append(dict(run=a.run, model=m, technique=t,
                             **{k: float(C[sel][:, grp == k].sum(1).mean() / tot) for k in GROUPS}))
    d = pd.DataFrame(rows)
    out = Path("docs/resultados/pub") / f"ire_decomposicao_{a.run}.csv"
    d.to_csv(out, index=False)
    print(d.groupby("technique")[GROUPS].mean().round(2).to_string())
    print(f"[decomposição] {len(s)} linhas amostradas -> {out}")


if __name__ == "__main__":
    main()
