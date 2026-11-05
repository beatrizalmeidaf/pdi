"""Agrega as análises de várias sementes: média, desvio-padrão e n por linha.

    python scripts/aggregate_seeds.py --runs pub_cifar10_s pub_imagenette_s --seeds 0 1 2 \
        --single pub_imagenet --out results/pub_agregado

Cada run é ``results/<prefixo><semente>/analysis``. ``--single`` junta runs sem sementes
(ex.: o ImageNet completo) nas mesmas tabelas, com sd vazio.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = {
    # arquivo: (chaves, colunas numéricas)
    "conditions_main.csv": (["dataset", "model", "technique", "level", "cond_id", "dose"],
                            ["acc_orig", "acc", "delta_acc", "flip", "lpips", "feat_cos"]),
    "thresholds.csv": (["dataset", "model", "technique"],
                       ["bmd_acc0.01", "bmdl_acc0.01", "bmdu_acc0.01", "bmd_flip0.01", "best_gain", "perc_dose"]),
    "loto.csv": (["dataset", "model", "features", "target"], ["r2", "spearman", "mae"]),
    "predictor_comparison.csv": (["dataset", "model", "predictor", "target"], ["r2", "spearman"]),
    "ire_summary.csv": (["dataset", "model"], ["r2_p_loto", "spearman_p_loto", "r2_p_all", "r2_p_oracle"]),
    "adaptation_summary.csv": (["dataset", "technique", "level", "cond_id"],
                               ["acc_orig", "acc_frozen", "acc_bn", "acc_ft_pp", "recoverable_frac", "info_loss"]),
}


def _agg(frames: list[pd.DataFrame], keys: list[str], cols: list[str]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in keys if k in df]
    cols = [c for c in cols if c in df]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    g = df.groupby(keys, dropna=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(g.count().add_suffix("_n"))
    return out.reset_index()


def _fmt(m, s, scale=1.0, nd=2):
    if pd.isna(m):
        return "—"
    return f"{m*scale:.{nd}f}" + ("" if pd.isna(s) else f" ± {s*scale:.{nd}f}")


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(sem dados)_\n"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--single", nargs="*", default=[])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/pub_agregado")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Path(a.results)
    dirs = [(R / f"{p}{s}" / "analysis", s) for p in a.runs for s in a.seeds] + [(R / r / "analysis", "-") for r in a.single]
    missing = [str(d) for d, _ in dirs if not d.exists()]
    if missing:
        print(f"[aviso] análises ausentes: {missing}")
    agg = {}
    for fname, (keys, cols) in TABLES.items():
        frames = [pd.read_csv(d / fname).assign(seed=s) for d, s in dirs if (d / fname).exists()]
        if frames:
            agg[fname] = _agg(frames, keys, cols)
            agg[fname].to_csv(out / fname.replace(".csv", "_agregado.csv"), index=False)

    rep = ["# Resultados agregados entre sementes\n",
           f"Sementes {', '.join(a.seeds)} para {', '.join(a.runs)} (cada semente = outro subconjunto de imagens e outro "
           f"fine-tuning); {', '.join(a.single) or '—'} sem sementes (todas as imagens). Valores: média ± desvio-padrão "
           "entre sementes.\n"]
    if "conditions_main.csv" in agg:
        b = agg["conditions_main.csv"].groupby(["dataset", "model"])[["acc_orig_mean", "acc_orig_sd"]].first().reset_index()
        b["acurácia original (%)"] = [_fmt(m, s, 100, 1) for m, s in zip(b.acc_orig_mean, b.acc_orig_sd)]
        rep += ["## Acurácia nas imagens originais\n", _md(b[["dataset", "model", "acurácia original (%)"]])]
    if "predictor_comparison.csv" in agg:
        p = agg["predictor_comparison.csv"]
        p = p[p.target == "flip"]
        # média sobre modelos, desvio entre sementes medido modelo a modelo e depois promediado
        t = p.groupby(["dataset", "predictor"]).agg(m=("r2_mean", "mean"), s=("r2_sd", "mean")).reset_index()
        t["R²"] = [_fmt(m, s) for m, s in zip(t.m, t.s)]
        piv = t.pivot(index="predictor", columns="dataset", values="R²").reset_index()
        order = t.groupby("predictor").m.mean().sort_values(ascending=False).index
        piv = piv.set_index("predictor").loc[order].reset_index()
        rep += ["## IRE × preditores existentes — R² LOTO do flip por condição (média dos modelos)\n", _md(piv)]
    if "loto.csv" in agg:
        p = agg["loto.csv"]
        t = p.groupby(["dataset", "features"]).agg(m=("r2_mean", "mean"), s=("r2_sd", "mean")).reset_index()
        t["R²"] = [_fmt(m, s) for m, s in zip(t.m, t.s)]
        rep += ["## R² LOTO da queda de acurácia por conjunto de preditores\n",
                _md(t.pivot(index="features", columns="dataset", values="R²").reset_index())]
    if "thresholds.csv" in agg:
        t = agg["thresholds.csv"]
        c = "bmd_acc0.01"
        if f"{c}_mean" in t:
            t["BMD 1 p.p."] = [_fmt(m, s, 1, 3) + (f" (n={int(n)})" if n < len(a.seeds) else "")
                               for m, s, n in zip(t[f"{c}_mean"], t[f"{c}_sd"], t[f"{c}_n"])]
            piv = t.pivot_table(index=["dataset", "technique"], columns="model", values="BMD 1 p.p.", aggfunc="first")
            rep += ["## Limiar BMD (queda de 1 p.p.)\n",
                    "n < número de sementes: em alguma semente o limiar não cruzou dentro das doses testadas.\n",
                    _md(piv.reset_index())]
    if "adaptation_summary.csv" in agg:
        t = agg["adaptation_summary.csv"]
        cols = ["dataset", "cond_id"]
        for c in ("acc_frozen", "acc_bn", "acc_ft_pp", "info_loss"):
            if f"{c}_mean" in t:
                t[c] = [_fmt(m, s, 100, 1) for m, s in zip(t[f"{c}_mean"], t[f"{c}_sd"])]
                cols.append(c)
        rep += ["## E4 — adaptação (%, média dos modelos ± sd entre sementes)\n", _md(t[cols])]
    (out / "summary.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"[agregado] {out / 'summary.md'}")


if __name__ == "__main__":
    main()
