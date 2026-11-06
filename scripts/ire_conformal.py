"""Margem conformal para a dose segura de uma técnica nova (substitui a margem fixa de ~30 %).

    python scripts/ire_conformal.py

Entrada: ire_condition_predictions.csv de cada análise da rodada final: flip observado e flip previsto pelo IRE ajustado
SEM aquela técnica (LOTO), por condição e modelo.

Regra de dose segura com margem: aceitar uma dose só se o IRE prevê flip <= c·τ (orçamento −ln(1 − c·τ)); c = 1 é a regra
sem margem. A margem fica no orçamento de flip, não na dose: assim vale para qualquer técnica, inclusive o JPEG, cujo limiar
pode cair já na qualidade 100.

Escore por (modelo, técnica) (thresholds.margin_scores): τ / p̂, onde p̂ é o flip previsto na dose em que o flip observado
cruza τ. A regra com fator c deixa o par seguro se c · escore <= 1.

Fator conformal por técnica (thresholds.conformal_margin): um escore por técnica, o do pior modelo; c = 1 / quantil
conformal desses escores. Se as técnicas forem trocáveis, uma técnica nova fica segura em TODOS os modelos com probabilidade
>= 1 − α. Com 14 técnicas, o menor α possível é 1/15 ≈ 0,07.

Avaliação:
  fixa_<c>                 fator fixo (1 = sem margem), cobertura por par e por técnica
  conformal_<unidade>_<α>  fator recalculado deixando cada técnica de fora e testado nela (LOTO honesto)
  entre bases              fator calibrado numa base e aplicado a outra
Saída: results/pub_agregado/ire_conformal*.csv e ire_dose_segura.csv; cópias em docs/resultados/pub/;
fator final por base para o `make safe-dose --margin`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pdi_spectral.config import load_config
from pdi_spectral.thresholds import conformal_margin, margin_scores

R = Path("results")
OUT = R / "pub_agregado"
DOCS = Path("docs/resultados/pub")
RUNS = {"pub_cifar10_s0": "configs/pub_cifar10.yaml", "pub_cifar10_s1": "configs/pub_cifar10.yaml",
        "pub_cifar10_s2": "configs/pub_cifar10.yaml", "pub_imagenette_s0": "configs/pub_imagenette.yaml",
        "pub_imagenette_s1": "configs/pub_imagenette.yaml", "pub_imagenette_s2": "configs/pub_imagenette.yaml",
        "pub_imagenet": "configs/pub_imagenet.yaml"}
ALPHAS = (0.1, 0.2)
FIXED = (1.0, 0.7, 0.5, 0.3, 0.2)


def scores(run: str) -> pd.DataFrame:
    cfg = load_config(RUNS[run])
    cp = pd.read_csv(R / run / "analysis" / "ire_condition_predictions.csv")
    return margin_scores(cp, cfg).assign(run=run)


def dose_cost(run: str, c: float, tau: float = 0.01) -> pd.DataFrame:
    """Quanto da dose realmente segura a regra com fator c aproveita: (d_rec − id) / (d_obs − id), por curva.

    Só curvas em que existe dose segura testada (o limiar observado cai depois da 1ª dose).
    """
    from pdi_spectral.thresholds import crossing, intensity, technique_meta

    cfg = load_config(RUNS[run])
    cp = pd.read_csv(R / run / "analysis" / "ire_condition_predictions.csv")
    rows = []
    for (model, t), g in cp.groupby(["model", "technique"]):
        meta = technique_meta(cfg, t)
        x = intensity(g.dose.to_numpy(), meta)
        o = np.argsort(x)
        x, obs, pred = x[o], g.flip.to_numpy()[o], g.p_loto.to_numpy()[o]
        idv = 0.0
        if meta.get("identity") is not None:
            idv = float(intensity(meta["identity"], meta))
            x, obs, pred = np.r_[idv, x], np.r_[0.0, obs], np.r_[0.0, pred]
        d_obs, so = crossing(x, obs[None], tau, "above")
        d_rec, sr = crossing(x, pred[None], c * tau, "above")
        # só curvas com dose segura testada: 2+ doses e o limiar observado depois da 1ª dose (CLAHE e SR ×2 ficam de fora)
        if g.dose.nunique() < 2 or so[0] != 0 or sr[0] == 2 or d_obs[0] <= x[1 if meta.get("identity") is not None else 0]:
            continue
        rows.append(dict(model=model, technique=t, aproveita=(d_rec[0] - idv) / (d_obs[0] - idv)))
    return pd.DataFrame(rows)


def coverage(sc: pd.DataFrame, m: float) -> tuple[float, float]:
    ok = m * sc.score <= 1 + 1e-9
    return float(ok.mean()), float(ok.groupby(sc.technique).all().mean())


def evaluate(sc: pd.DataFrame) -> list[dict]:
    run, ds = sc.run.iloc[0], sc.dataset.iloc[0]
    rows = []
    for m in FIXED:
        cp, ct = coverage(sc, m)
        rows.append(dict(run=run, dataset=ds, metodo=f"fixa_{m:g}", margem=m, cobertura_par=cp, cobertura_tecnica=ct))
    for a in ALPHAS:
        for unit in ("technique", "pair"):
            ms, okp, okt = [], [], []
            for t in sc.technique.unique():
                cal, test = sc[sc.technique != t], sc[sc.technique == t]
                m = conformal_margin(cal, a, unit)
                if not np.isfinite(m):
                    continue
                ok = m * test.score <= 1 + 1e-9
                ms.append(m), okp.append(ok.mean()), okt.append(ok.all())
            rows.append(dict(run=run, dataset=ds, metodo=f"conformal_{unit}_{a:g}", alpha=a,
                             margem=float(np.mean(ms)) if ms else np.nan,
                             cobertura_par=float(np.mean(okp)) if okp else np.nan,
                             cobertura_tecnica=float(np.mean(okt)) if okt else np.nan, n_tecnicas=len(ms)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    allsc = {r: scores(r) for r in RUNS if (R / r / "analysis" / "ire_condition_predictions.csv").exists()}
    pd.concat(allsc.values()).to_csv(OUT / "ire_conformal_escores.csv", index=False)

    ev = pd.DataFrame([row for sc in allsc.values() for row in evaluate(sc)])
    # margem final (todas as técnicas) por base: para usar no safe-dose
    final = []
    for r, sc in allsc.items():
        for a in ALPHAS:
            c = conformal_margin(sc, a, "technique")
            cost = dose_cost(r, c) if np.isfinite(c) else pd.DataFrame(columns=["aproveita"])
            cost0 = dose_cost(r, 1.0)
            final.append(dict(run=r, dataset=sc.dataset.iloc[0], alpha=a, margem=c, n_tecnicas=sc.technique.nunique(),
                              orcamento_tau_0_01=float(-np.log1p(-c * 0.01)) if np.isfinite(c) else np.nan,
                              aproveita_mediana=float(cost.aproveita.median()) if len(cost) else np.nan,
                              aproveita_sem_margem=float(cost0.aproveita.median())))
    final = pd.DataFrame(final)
    # entre bases: margem de uma base aplicada a outra
    cross = []
    for src, dst in [("pub_imagenette_s0", "pub_imagenet"), ("pub_imagenet", "pub_imagenette_s0"),
                     ("pub_cifar10_s0", "pub_imagenet"), ("pub_imagenet", "pub_cifar10_s0")]:
        if src in allsc and dst in allsc:
            for a in ALPHAS:
                m = conformal_margin(allsc[src], a, "technique")
                cp, ct = coverage(allsc[dst], m)
                cross.append(dict(origem=src, destino=dst, alpha=a, margem=m, cobertura_par=cp, cobertura_tecnica=ct))
    cross = pd.DataFrame(cross)

    ev.to_csv(OUT / "ire_conformal.csv", index=False)
    final.to_csv(OUT / "ire_conformal_margens.csv", index=False)
    cross.to_csv(OUT / "ire_conformal_entre_bases.csv", index=False)
    cov = ev[ev.metodo.str.startswith("fixa_")].pivot_table(index=["run", "dataset"], columns="margem",
                                                             values="cobertura_par").reset_index()
    cov.to_csv(OUT / "ire_dose_segura.csv", index=False)
    for f in ("ire_conformal.csv", "ire_conformal_margens.csv", "ire_conformal_entre_bases.csv",
              "ire_conformal_escores.csv", "ire_dose_segura.csv"):
        (DOCS / f).write_bytes((OUT / f).read_bytes())

    pd.set_option("display.width", 200)
    s = ev.groupby(["dataset", "metodo"])[["margem", "cobertura_par", "cobertura_tecnica"]].mean().round(2)
    print(s.to_string())
    print(final.groupby(["dataset", "alpha"])[["margem", "orcamento_tau_0_01", "aproveita_mediana",
                                                "aproveita_sem_margem"]].mean().round(3).to_string())
    sc = pd.concat(allsc.values())
    print("curvas com flip previsto não monótono:", int((~sc.monotona).sum()), "de", len(sc))
    print(sc.groupby(["dataset", "technique"]).score.max().unstack(0).round(2).to_string())
    print(cross.round(2).to_string())


if __name__ == "__main__":
    main()
