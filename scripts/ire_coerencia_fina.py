"""Coerência por anel: com ela, os filtros de banda passam a calibrar o IRE em 224 px?

    python scripts/ire_coerencia_fina.py [--runs ...]

O IRE atual mede a energia da perturbação em 17 anéis, mas a perda de coerência (estrutura/fase) só em 3 bandas largas.
Hipótese: a perda de estrutura num anel estreito se dilui na medida grossa, e por isso filtros e técnicas reais têm as
mesmas medidas e danos diferentes em 224 px. O quantificador já guarda, anel a anel, a decomposição exata
D_f = A_f + P_f (amplitude + estrutura), com P_f = 2·sqrt(Ex·Ey)·(1 − coerência do anel) / E_total: a perda de coerência
por anel, ponderada pela energia.

Descrições comparadas (thresholds.ire_features):
  D_fine+chroma+coh     atual: energia em 17 anéis + cor + coerência em 3 bandas
  AP_fine+chroma        amplitude e estrutura em 17 anéis + cor
  AP_fine+chroma+coh    as duas
Para cada uma: (a) R² LOTO nas 14 técnicas reais (não pode piorar); (b) calibrar só com filtros (ideais, suaves,
parciais, todos) e prever as técnicas reais; (c) a transferência inversa, das técnicas reais para os filtros.
Saída: docs/resultados/pub/ire_coerencia_fina.csv.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pdi_spectral.analysis import drop_images
from pdi_spectral.thresholds import _r2, fit_ire, ire_features, ire_matrix, ire_prob

R = Path("results")
KINDS = ["D_fine+chroma+coh", "AP_fine+chroma", "AP_fine+chroma+coh"]
# --anel: coerência pura por anel (scripts/espectro_coerencia_anel.py), no lugar da ponderada pela energia
KINDS_ANEL = ["atual (coerência em 3 bandas)", "coerência em 17 anéis", "3 bandas + 17 anéis"]
COHF = [f"incoh_f{i:02d}" for i in range(17)]
ANEL = False
BASES = {"pub_cifar10_s0": "pub_cifar10_s0", "pub_imagenette_s0": "pub_imagenette_s0", "pub_imagenet": "pub_imagenet"}
MAX_ROWS = 150_000


def _load(run: str, kind: str, feats_all: list[str], image_ids=None) -> pd.DataFrame:
    base = R / run / kind
    d = next(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_"))
    raw = sorted({("coh_" + f[6:]) if f.startswith("incoh_") else f for f in feats_all})
    filt = [("image_id", "in", list(image_ids))] if image_ids is not None else None
    raw = [c for c in raw if not c.startswith("coh_f")]
    spec = pd.read_parquet(d / "spectral.parquet", columns=["image_id", "cond_id"] + raw, filters=filt)
    if ANEL:
        cf = pd.read_parquet(d / "coerencia_anel.parquet", filters=filt)
        cf.columns = [c.replace("cohf_", "coh_") for c in cf.columns]
        spec = spec.merge(cf, on=["image_id", "cond_id"])
    pred = pd.read_parquet(d / "predictions.parquet", columns=["image_id", "model", "cond_id", "technique", "flip"],
                           filters=filt)
    pred = pred[pred.cond_id != "original"]
    return pred.merge(spec, on=["image_id", "cond_id"])


def _feature_union(run: str) -> dict[str, list[str]]:
    d = next(p for p in (R / run / "main").iterdir() if p.is_dir() and not p.name.startswith("_"))
    names = pq.ParquetFile(d / "spectral.parquet").schema.names
    cols = pd.DataFrame(columns=names)
    if ANEL:
        base = ire_features(cols, "D_fine+chroma+coh")
        no_coh = [f for f in base if not f.startswith("incoh_")]
        return {KINDS_ANEL[0]: base, KINDS_ANEL[1]: no_coh + COHF, KINDS_ANEL[2]: base + COHF}
    return {k: ire_features(cols, k) for k in KINDS}


def _cond_metrics(g, p):
    c = g.assign(p=p).groupby("cond_id")[["flip", "p"]].mean()
    return dict(r2=_r2(c.flip.to_numpy(), c.p.to_numpy()), vies=float((c.p - c.flip).mean()),
                spearman=float(c.corr("spearman").iloc[0, 1]))


def run_base(task: tuple[str, str], seed: int = 0) -> pd.DataFrame:
    run, only_model = task
    rng = np.random.default_rng(seed)
    fk = _feature_union(run)
    allf = sorted({f for v in fk.values() for f in v})
    # imagens: as da rodada de filtros suaves/parciais (no ImageNet, 10 mil das 50 mil)
    soft_run, part_run = run.replace("pub_imagenet", "pub_imagenet") + "_oraculo_suave", run + "_oraculo_parcial"
    ids = pd.read_parquet(next(p for p in (R / soft_run / "oracle").iterdir() if p.is_dir() and not p.name.startswith("_"))
                          / "spectral.parquet", columns=["image_id"]).image_id.unique()
    real = _load(run, "main", allf, ids)
    real = real[real.model == only_model]
    filtros = {k: v[v.model == only_model] for k, v in {"ideais": _load(run, "oracle", allf, ids),
                                                         "suaves": _load(soft_run, "oracle", allf, ids),
                                                         "parciais": _load(part_run, "oracle", allf, ids)}.items()}
    bad = R / run / "analysis" / "excluded_flat_images.csv"
    if bad.exists():
        b = pd.read_csv(bad).assign(dataset="x")
        real = drop_images(real.assign(dataset="x"), b).drop(columns="dataset")
        filtros = {k: drop_images(v.assign(dataset="x"), b).drop(columns="dataset") for k, v in filtros.items()}
    filtros["todos os filtros"] = pd.concat([v.assign(cond_id=k + "_" + v.cond_id) for k, v in filtros.items()],
                                            ignore_index=True)
    rows = []
    for m in [only_model]:
        g = real[real.model == m].reset_index(drop=True)
        for kind, feats in fk.items():
            Xg, yg = ire_matrix(g, feats), g.flip.to_numpy().astype(float)
            # (a) LOTO nas técnicas reais
            p = np.full(len(g), np.nan)
            for t in g.technique.unique():
                te = (g.technique == t).to_numpy()
                tr = rng.choice(np.flatnonzero(~te), min(int((~te).sum()), MAX_ROWS), replace=False)
                p[te] = ire_prob(Xg[te], fit_ire(Xg[tr], yg[tr], 1e-4, None))
            rows.append(dict(run=run, model=m, features=kind, teste="LOTO nas técnicas reais", **_cond_metrics(g, p)))
            # (b) calibrar só com filtros, prever as técnicas reais; (c) inverso
            idx = rng.choice(len(g), min(len(g), MAX_ROWS), replace=False)
            th_real = fit_ire(Xg[idx], yg[idx], 1e-4, None)
            for nome, o in filtros.items():
                o = o[o.model == m].reset_index(drop=True)
                Xo, yo = ire_matrix(o, feats), o.flip.to_numpy().astype(float)
                j = rng.choice(len(o), min(len(o), MAX_ROWS), replace=False)
                th = fit_ire(Xo[j], yo[j], 1e-4, None)
                rows.append(dict(run=run, model=m, features=kind, teste=f"calibrado em filtros {nome}",
                                 **_cond_metrics(g, ire_prob(Xg, th))))
                if nome != "todos os filtros":
                    rows.append(dict(run=run, model=m, features=kind, teste=f"técnicas reais → filtros {nome}",
                                     **_cond_metrics(o, ire_prob(Xo, th_real))))
        print(f"[coerência fina] {run} {m}: ok", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(BASES))
    ap.add_argument("--anel", action="store_true", help="coerência pura por anel (precisa de coerencia_anel.parquet)")
    a = ap.parse_args()
    global ANEL
    ANEL = a.anel
    tasks = []
    for run in a.runs:
        d0 = next(p for p in (R / run / "main").iterdir() if p.is_dir() and not p.name.startswith("_"))
        models = pd.read_parquet(d0 / "predictions.parquet", columns=["model"]).model.unique()
        tasks += [(run, m) for m in sorted(models)]
    with ProcessPoolExecutor(len(tasks)) as ex:
        d = pd.concat(list(ex.map(run_base, tasks)), ignore_index=True)
    d.to_csv(f"docs/resultados/pub/ire_coerencia_{'anel' if a.anel else 'fina'}.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.groupby(["run", "teste", "features"])[["r2", "vies", "spearman"]].mean().round(2).unstack("features")
          ["r2"].to_string())


if __name__ == "__main__":
    main()
