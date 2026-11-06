"""De onde vem o risco previsto pelo IRE? Decomposição por componente, por técnica e modelo.

    python scripts/ire_decomposicao.py [--run pub_imagenet] [--stride 8]
    python scripts/ire_decomposicao.py --figura      # figura com as 3 bases (lê os CSVs já gerados)

Para cada (imagem, condição), o IRE é uma soma de termos w_j · x_j^γ. A parte de cada grupo é a média dos seus termos
dividida pela média do IRE total, nas imagens da técnica:
  lum_baixa        energia da perturbação de luminância nos 3 primeiros anéis
  lum_media_alta   energia de luminância nos demais anéis
  cor              energia da perturbação nos canais de cor (Cb, Cr)
  coerencia        perda de coerência espectral (estrutura e fase) nas bandas baixa, média e alta
Usa uma amostra de row groups do spectral.parquet (--stride). Saída: docs/resultados/pub/ire_decomposicao_<run>.csv e,
com --figura, docs/figures/ire_decomposicao.png.
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


RUNS_FIG = [("pub_cifar10_s0", "CIFAR-10 · 32 px"), ("pub_imagenette_s0", "Imagenette · 224 px"),
            ("pub_imagenet", "ImageNet-val · 224 px")]


def figure():
    from pdi_spectral.report import C, TECH_LABEL, TECH_ORDER, _finish, _style, _title

    plt = _style()
    parts = [("coerencia", "perda de estrutura/fase (coerência)", C["s1"]),
             ("lum_media_alta", "energia de luminância (média/alta freq.)", C["s2"]),
             ("cor", "energia de cor", C["s3"])]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.6), sharey=True)
    for ax, (run, title) in zip(axes, RUNS_FIG):
        d = pd.read_csv(Path("docs/resultados/pub") / f"ire_decomposicao_{run}.csv")
        m = d.groupby("technique")[GROUPS].mean().reindex([t for t in TECH_ORDER if t in set(d.technique)])
        y = np.arange(len(m))
        left = np.zeros(len(m))
        for key, lab, col in parts:
            v = 100 * m[key].to_numpy()
            ax.barh(y, v, left=left, color=col, height=0.72, label=lab)
            for i, (l, w) in enumerate(zip(left, v)):
                if w >= 9:
                    ax.text(l + w / 2, i, f"{w:.0f}", ha="center", va="center", fontsize=7.5, color="white")
            left += v
        ax.set_yticks(y, [TECH_LABEL[t] for t in m.index])
        ax.set_ylim(len(m) - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.set_title(title, loc="left")
        ax.set_xlabel("% do risco previsto (média dos modelos)")
        ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8)
    _title(fig, "De onde vem o risco previsto pelo IRE",
           "Parte de cada componente no IRE médio da técnica. A perda de estrutura/fase domina em todas as técnicas e bases. "
           "No aguçamento, a energia de luminância em alta frequência pesa 31–41 % em CIFAR-10 e Imagenette (13–15 % no "
           "ImageNet-val, onde a cor pesa mais). No JPEG 4:2:0, a cor pesa mais quanto maior a resolução: 9 → 18 → 33 %. "
           "O peso da energia na banda mais baixa sai 0: a coerência da banda baixa já carrega essa informação.")
    _finish(fig)
    fig.subplots_adjust(wspace=0.08)
    out = Path("docs/figures/ire_decomposicao.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[decomposição] figura em {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pub_imagenet")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--figura", action="store_true")
    a = ap.parse_args()
    if a.figura:
        figure()
        return
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
