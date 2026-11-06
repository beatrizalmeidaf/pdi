"""Validação externa do IRE e da margem conformal com 6 técnicas que não entraram em nada do ajuste.

    python scripts/ire_validacao_externa.py            # depois de scripts/run_validacao_externa.sh

Técnicas novas (techniques/extra.py): box, perona_malik, gamma, stretch, webp, resample. Mesmas imagens e modelos da
rodada final (results/<run>_extra).

1. Teste externo do IRE: os pesos ajustados nas 14 técnicas originais (ire_weights.csv, fit=all) preveem o flip de cada
   condição das técnicas novas. R², Spearman e viés por modelo.
2. Teste externo da margem: o fator c calibrado nas 14 (ire_conformal_margens.csv, α = 0,1 e 0,2) é aplicado às novas;
   cobertura por técnica (todos os modelos seguros) e por par.
3. Calibração com 20 técnicas: IRE refeito deixando cada uma das 20 de fora (LOTO) e margem conformal recalculada sem a
   técnica testada; com 20 técnicas o menor α possível cai de 1/15 para 1/21 ≈ 0,048.
Saída: results/pub_agregado/ire_validacao_externa*.csv e cópias em docs/resultados/pub/.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ire_transfer import _load
from pdi_spectral.config import load_config
from pdi_spectral.thresholds import _r2, conformal_margin, fit_ire, ire_matrix, ire_prob, margin_scores

R = Path("results")
OUT = R / "pub_agregado"
DOCS = Path("docs/resultados/pub")
RUNS = [f"pub_cifar10_s{s}" for s in range(3)] + [f"pub_imagenette_s{s}" for s in range(3)] + ["pub_imagenet"]
NEW = ["box", "perona_malik", "gamma", "stretch", "webp", "resample"]
CFG = load_config("configs/pub_extra_imagenet.yaml")  # tem a meta (identidade, eixo) das 20 técnicas
MAX_ROWS = 300_000
ALPHAS = (0.05, 0.1, 0.2)


def _extra_name(run: str) -> str:
    return run + "_extra" if run != "pub_imagenet" else "pub_imagenet_extra"


def _load_pair(run: str):
    ds, feats, dm = _load(run)
    _, feats_x, dx = _load(_extra_name(run))
    assert feats == feats_x
    bad = R / run / "analysis" / "excluded_flat_images.csv"  # a rodada extra não mede contraste: usa a lista original
    if bad.exists():
        b = pd.read_csv(bad)
        dx = dx[~dx.image_id.isin(b.image_id)]
    return ds, feats, dm, dx


def _cp(g: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    return (g.assign(p_loto=p).groupby(["dataset", "model", "technique", "cond_id", "dose"])[["flip", "p_loto"]]
            .mean().reset_index())


def run_one(run: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ds, feats, dm, dx = _load_pair(run)
    w = pd.read_csv(R / run / "analysis" / "ire_weights.csv")
    w = w[w.fit == "all"]
    ext, cps_ext, cps_loto = [], [], []
    for m in sorted(dx.model.unique()):
        gx = dx[dx.model == m].reset_index(drop=True)
        th = w[w.model == m].set_index("feature").w.reindex(feats + ["_gamma"]).to_numpy()
        cp = _cp(gx, ire_prob(ire_matrix(gx, feats), th))
        cps_ext.append(cp)
        for t, h in [("todas as novas", cp)] + list(cp.groupby("technique")):
            ext.append(dict(dataset=ds, run=run, model=m, technique=t, r2=_r2(h.flip.to_numpy(), h.p_loto.to_numpy()),
                            spearman=float(h[["flip", "p_loto"]].corr("spearman").iloc[0, 1]) if len(h) > 2 else np.nan,
                            razao_prev_obs=float((h.p_loto / h.flip.clip(lower=1e-4))[h.flip > 0.02].median())
                            if (h.flip > 0.02).any() else np.nan))
        # LOTO com as 20 técnicas
        g = pd.concat([dm[dm.model == m], gx], ignore_index=True)
        X, y = ire_matrix(g, feats), g.flip.to_numpy().astype(float)
        p = np.full(len(g), np.nan)
        for t in g.technique.unique():
            te = (g.technique == t).to_numpy()
            tr = np.flatnonzero(~te)
            tr = rng.choice(tr, min(len(tr), MAX_ROWS), replace=False)
            p[te] = ire_prob(X[te], fit_ire(X[tr], y[tr], 1e-4, None))
        cps_loto.append(_cp(g, p))
    print(f"[externa] {run}: ok", flush=True)
    return dict(ext=pd.DataFrame(ext), cp_ext=pd.concat(cps_ext), cp_loto=pd.concat(cps_loto))


def coverage(sc: pd.DataFrame, c: float) -> tuple[float, float]:
    ok = c * sc.score <= 1 + 1e-9
    return float(ok.mean()), float(ok.groupby(sc.technique).all().mean())


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    with ProcessPoolExecutor(len(RUNS)) as ex:
        res = dict(zip(RUNS, ex.map(run_one, RUNS)))

    # 1. teste externo do IRE
    ext = pd.concat([r["ext"] for r in res.values()], ignore_index=True)
    ext.to_csv(OUT / "ire_validacao_externa.csv", index=False)

    # 2. margem calibrada nas 14, aplicada às 6 novas
    margens = pd.read_csv(OUT / "ire_conformal_margens.csv")
    cov = []
    for run, r in res.items():
        sc = margin_scores(r["cp_ext"], CFG)
        for a in (0.1, 0.2):
            c = float(margens[(margens.run == run) & (margens.alpha == a)].margem.iloc[0])
            for cc, nome in [(1.0, "sem margem"), (c, f"margem das 14, α = {a:g}")]:
                cp_, ct_ = coverage(sc, cc)
                cov.append(dict(run=run, dataset=sc.dataset.iloc[0], metodo=nome, alpha=a, margem=cc,
                                cobertura_par=cp_, cobertura_tecnica=ct_))
    cov = pd.DataFrame(cov).drop_duplicates(["run", "metodo"])
    cov.to_csv(OUT / "ire_validacao_externa_margem.csv", index=False)

    # 3. 20 técnicas: margem conformal recalculada deixando cada técnica de fora
    loto = []
    for run, r in res.items():
        sc = margin_scores(r["cp_loto"], CFG)
        for a in ALPHAS:
            ms, okp, okt, novas = [], [], [], []
            for t in sc.technique.unique():
                c = conformal_margin(sc[sc.technique != t], a, "technique")
                if not np.isfinite(c):
                    continue
                ok = c * sc[sc.technique == t].score <= 1 + 1e-9
                ms.append(c), okp.append(ok.mean()), okt.append(ok.all())
                novas.append(t in NEW)
            okt, novas = np.array(okt), np.array(novas)
            loto.append(dict(run=run, dataset=sc.dataset.iloc[0], alpha=a, n_tecnicas=sc.technique.nunique(),
                             margem_media=float(np.mean(ms)) if ms else np.nan,
                             margem_todas=conformal_margin(sc, a, "technique"),
                             cobertura_tecnica=float(okt.mean()) if len(okt) else np.nan,
                             cobertura_tecnica_novas=float(okt[novas].mean()) if novas.any() else np.nan,
                             cobertura_par=float(np.mean(okp)) if okp else np.nan,
                             r2_loto_20=_r2(r["cp_loto"].flip.to_numpy(), r["cp_loto"].p_loto.to_numpy())))
    loto = pd.DataFrame(loto)
    loto.to_csv(OUT / "ire_validacao_externa_20tecnicas.csv", index=False)

    for f in ("ire_validacao_externa.csv", "ire_validacao_externa_margem.csv", "ire_validacao_externa_20tecnicas.csv"):
        (DOCS / f).write_bytes((OUT / f).read_bytes())
    pd.set_option("display.width", 220)
    t = ext[ext.technique == "todas as novas"].groupby("dataset")[["r2", "spearman"]].agg(["mean", "min", "max"]).round(2)
    print("1. IRE ajustado nas 14, previsto nas 6 novas\n", t.to_string())
    print(ext[ext.technique != "todas as novas"].groupby(["dataset", "technique"])[["r2", "razao_prev_obs"]]
          .mean().round(2).unstack(0).to_string())
    print("2. margem das 14 nas 6 novas\n", cov.groupby(["dataset", "metodo"])[["margem", "cobertura_tecnica",
                                                                             "cobertura_par"]].mean().round(2).to_string())
    print("3. 20 técnicas (LOTO)\n", loto.groupby(["dataset", "alpha"])[["margem_todas", "cobertura_tecnica",
                                                                          "cobertura_tecnica_novas", "cobertura_par",
                                                                          "r2_loto_20"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
