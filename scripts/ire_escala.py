"""Pesos de banda compartilhados + 1 fator de escala por modelo: basta para o IRE transferir?

    python scripts/ire_escala.py                 # todas as rodadas pub_* (em paralelo)

O teste de transferência (ire_transfer.py) mostrou que, em 224 px, pesos ajustados em outros modelos acertam a ORDEM do dano
mas não o NÍVEL. Aqui os pesos de banda (w, c, u, γ) vêm dos outros modelos e o modelo-alvo ajusta só um número k:

    P(flip) = 1 - exp(-k · IRE_outros(x))

Fontes para k (nenhuma usa a técnica que está sendo prevista):
  escala_oraculo    só os 27 filtros ideais (E1) do modelo-alvo, sem nenhuma técnica real
  escala_gaussiano  só o desfoque gaussiano do modelo-alvo; o R² é medido nas outras técnicas

Protocolos (flip por condição do modelo-alvo, R² como em ire_summary.csv):
  outros_modelos_loto                      sem escala (referência, igual ao ire_transfer.py)
  outros_modelos_loto+escala_oraculo       modelo novo e técnica nova; calibração só com filtros ideais
  outros_modelos_loto+escala_gaussiano     modelo novo e técnica nova; calibração com 1 técnica real
  outra_base+escala_oraculo                pesos do mesmo modelo na outra base de 224 px; k nos filtros ideais da base-alvo

Além de R², Spearman e viés médio, mede o lado da garantia: `subestima` = fração das condições com flip > 1 % em que a
previsão fica abaixo do observado, e `pior_sub` = maior subestimação (p.p. de flip).
A cobertura da dose segura para técnica nova fica em scripts/ire_conformal.py.
Saída: results/pub_agregado/ire_escala*.csv, docs/resultados/pub/ e docs/figures/ire_transfer.png.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ire_transfer import L2, OUT, R, RUNS, _load

MAX_ROWS = 150_000
SCALE_ROWS = 200_000


def fit_scale(s0: np.ndarray, y: np.ndarray) -> float:
    """MLE de k em P = 1 - exp(-k·s0), k > 0."""
    from scipy.optimize import minimize_scalar

    s0 = np.clip(s0, 0, None) + 1e-12

    def nll(logk):
        s = np.exp(logk) * s0
        return float((-(y * np.log(-np.expm1(-s) + 1e-300)) + (1 - y) * s).mean())

    r = minimize_scalar(nll, bounds=(-12, 12), method="bounded")
    return float(np.exp(r.x))


def _metrics(g: pd.DataFrame, p: np.ndarray, mask: np.ndarray | None = None) -> dict:
    from pdi_spectral.thresholds import _r2

    h = g.assign(p=p)
    if mask is not None:
        h = h[mask]
    c = h.groupby("cond_id")[["flip", "p"]].mean()
    big = c[c.flip > 0.01]
    return dict(r2=_r2(c.flip.to_numpy(), c.p.to_numpy()), spearman=float(c.corr("spearman").iloc[0, 1]),
                vies=float((c.p - c.flip).mean()), subestima=float((big.p < big.flip).mean()) if len(big) else np.nan,
                pior_sub=float((big.flip - big.p).max()) if len(big) else np.nan)


def run_one(run: str, seed: int = 0) -> pd.DataFrame:
    from pdi_spectral.thresholds import fit_ire, ire_matrix, ire_score

    rng = np.random.default_rng(seed)
    ds, feats, df = _load(run)
    _, feats_o, dfo = _load(run, "oracle")
    assert feats_o == feats
    models = sorted(df.model.unique())
    by = {m: df[df.model == m].reset_index(drop=True) for m in models}
    X = {m: ire_matrix(by[m], feats) for m in models}
    y = {m: by[m].flip.to_numpy().astype(float) for m in models}
    oro = {m: dfo[dfo.model == m] for m in models}
    Xo = {m: ire_matrix(oro[m], feats) for m in models}
    yo = {m: oro[m].flip.to_numpy().astype(float) for m in models}

    def fit(rows):
        per = max(1, MAX_ROWS // len(rows))
        Xs, ys = [], []
        for m, mask in rows:
            idx = np.flatnonzero(mask)
            idx = rng.choice(idx, min(len(idx), per), replace=False)
            Xs.append(X[m][idx]), ys.append(y[m][idx])
        return fit_ire(np.vstack(Xs), np.concatenate(ys), L2, None)

    def scale(Xc, yc, w):
        idx = rng.choice(len(yc), min(len(yc), SCALE_ROWS), replace=False)
        return fit_scale(ire_score(Xc[idx], w), yc[idx])

    rows = []
    for m in models:
        g, others = by[m], [o for o in models if o != m]
        gauss = (g.technique == "gaussian").to_numpy()
        p_none, p_or, p_ga = (np.full(len(g), np.nan) for _ in range(3))
        k_or, k_ga = [], []
        for t in g.technique.unique():
            te = (g.technique == t).to_numpy()
            w = fit([(o, (by[o].technique != t).to_numpy()) for o in others])
            s = ire_score(X[m][te], w)
            p_none[te] = -np.expm1(-s)
            k = scale(Xo[m], yo[m], w)
            k_or.append(k)
            p_or[te] = -np.expm1(-k * s)
            if t != "gaussian":
                k = scale(X[m][gauss], y[m][gauss], w)
                k_ga.append(k)
                p_ga[te] = -np.expm1(-k * s)
        base = dict(dataset=ds, run=run, model=m)
        rows.append(dict(base, protocolo="outros_modelos_loto", k=1.0, **_metrics(g, p_none)))
        rows.append(dict(base, protocolo="outros_modelos_loto+escala_oraculo", k=float(np.median(k_or)), **_metrics(g, p_or)))
        rows.append(dict(base, protocolo="outros_modelos_loto+escala_gaussiano", k=float(np.median(k_ga)),
                         **_metrics(g, p_ga, ~gauss)))
        # o próprio modelo (LOTO), para comparar o lado da garantia
        p_own = np.full(len(g), np.nan)
        for t in g.technique.unique():
            te = (g.technique == t).to_numpy()
            p_own[te] = -np.expm1(-ire_score(X[m][te], fit([(m, ~te)])))
        rows.append(dict(base, protocolo="proprio_loto", k=1.0, **_metrics(g, p_own)))
    print(f"[escala] {run}: ok", flush=True)
    return pd.DataFrame(rows)


def cross_dataset(seed: int = 0) -> pd.DataFrame:
    from pdi_spectral.thresholds import ire_matrix, ire_score

    rng = np.random.default_rng(seed)
    wdir = OUT / "ire_transfer_pesos"
    rows = []
    for src, dst in [("pub_imagenette_s0", "pub_imagenet"), ("pub_imagenet", "pub_imagenette_s0")]:
        if not (wdir / f"{src}.csv").exists():
            continue
        w = pd.read_csv(wdir / f"{src}.csv")
        ds, feats, df = _load(dst)
        _, _, dfo = _load(dst, "oracle")
        for m, g in df.groupby("model"):
            wm = w[w.model == m].w.to_numpy()
            o = dfo[dfo.model == m]
            idx = rng.choice(len(o), min(len(o), SCALE_ROWS), replace=False)
            k = fit_scale(ire_score(ire_matrix(o.iloc[idx], feats), wm), o.flip.to_numpy().astype(float)[idx])
            s = ire_score(ire_matrix(g, feats), wm)
            base = dict(dataset=ds, run=dst, model=m, origem=src)
            rows.append(dict(base, protocolo="outra_base", k=1.0, **_metrics(g.reset_index(drop=True), -np.expm1(-s))))
            rows.append(dict(base, protocolo="outra_base+escala_oraculo", k=k,
                             **_metrics(g.reset_index(drop=True), -np.expm1(-k * s))))
    print("[escala] entre bases: ok", flush=True)
    return pd.DataFrame(rows)


ORDER = ["proprio_loto", "outros_modelos_loto", "outros_modelos_loto+escala_oraculo",
         "outros_modelos_loto+escala_gaussiano", "outra_base", "outra_base+escala_oraculo"]
LABEL = {"proprio_loto": "próprio modelo\n(técnica nova)", "outros_modelos_loto": "outros modelos\n+ técnica nova",
         "outros_modelos_loto+escala_oraculo": "… + escala pelos\nfiltros ideais",
         "outros_modelos_loto+escala_gaussiano": "… + escala pelo\ngaussiano",
         "outra_base": "outra base\n(mesmo modelo)", "outra_base+escala_oraculo": "… + escala pelos\nfiltros ideais"}


def figure(res: pd.DataFrame, pair: pd.DataFrame | None):
    from pdi_spectral.report import C, _finish, _style, _title

    plt = _style()
    dsl = [("cifar10", "CIFAR-10", C["s1"]), ("imagenette", "Imagenette", C["s2"]), ("imagenet_val", "ImageNet-val", C["s3"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.4), gridspec_kw=dict(width_ratios=[1.75, 1]))
    x = np.arange(len(ORDER))
    for k, (ds, name, col) in enumerate(dsl):
        sub = [res[(res.dataset == ds) & (res.protocolo == p)].r2 for p in ORDER]
        v = [s.mean() if len(s) else np.nan for s in sub]
        xx = x + (k - 1) * 0.26
        a1.bar(xx, np.nan_to_num(np.clip(v, 0, None)), 0.24, color=col, label=name)
        a1.vlines(xx, [max(s.min(), 0) if len(s) else np.nan for s in sub],
                  [s.max() if len(s) else np.nan for s in sub], color=C["ink2"], lw=1)
    for xv in (0.5, 3.5):
        a1.axvline(xv, color=C["axis"], lw=0.8)
    a1.set_xticks(x, [LABEL[p] for p in ORDER], fontsize=8)
    a1.set_ylim(0, 1)
    a1.set_ylabel("R² do flip por condição")
    a1.set_title("Média dos modelos (traço: mín.–máx.)", loc="left")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
    # painel 2: lado da garantia — maior subestimação do flip, por modelo, em 224 px
    short = {"convnext_tiny": "ConvNeXt", "efficientnet_b0": "EffNet-B0", "resnet18": "R18", "resnet50": "R50",
             "swin_t": "Swin", "vgg16_bn": "VGG16", "vit_b_16": "ViT"}
    sel = res[(res.dataset == "imagenet_val") & res.protocolo.isin(
        ["proprio_loto", "outros_modelos_loto", "outros_modelos_loto+escala_gaussiano"])]
    piv = sel.pivot_table(index="model", columns="protocolo", values="pior_sub").reindex(columns=[
        "proprio_loto", "outros_modelos_loto", "outros_modelos_loto+escala_gaussiano"]) * 100
    yy = np.arange(len(piv))
    for j, (p, col, lab) in enumerate([("proprio_loto", C["ink2"], "próprio modelo"),
                                       ("outros_modelos_loto", C["s2"], "outros modelos"),
                                       ("outros_modelos_loto+escala_gaussiano", C["s3"], "outros + escala (gaussiano)")]):
        a2.barh(yy + (j - 1) * 0.26, piv[p], 0.24, color=col, label=lab)
    a2.set_yticks(yy, [short.get(m, m) for m in piv.index], fontsize=8.5)
    a2.invert_yaxis()
    a2.grid(axis="x"), a2.grid(axis="y", visible=False)
    a2.set_xlabel("maior subestimação do flip (p.p.)")
    a2.set_title("ImageNet-val: pior subestimação do flip", loc="left")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    _title(fig, "O IRE calibrado num modelo serve para outro? Em 224 px, com 1 técnica real para a escala",
           "Pesos de banda ajustados nos outros modelos (sem a técnica prevista); o modelo-alvo ajusta só um número k. "
           "Com o desfoque gaussiano, k corrige os modelos extremos (ViT, ResNet-18, VGG); com só os filtros ideais, k "
           "superestima o dano (R² < 0, fora da escala). Em 32 px não é preciso escala.")
    _finish(fig)
    fig.savefig("docs/figures/ire_transfer.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--jobs", type=int, default=7)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.jobs) as ex:
        outs = list(ex.map(run_one, a.runs))
    res = pd.concat(outs + [cross_dataset()], ignore_index=True)
    res.to_csv(OUT / "ire_escala.csv", index=False)
    s = res.groupby(["dataset", "protocolo"]).agg(
        r2=("r2", "mean"), r2_min=("r2", "min"), r2_max=("r2", "max"), spearman=("spearman", "mean"),
        vies=("vies", "mean"), subestima=("subestima", "mean"), pior_sub=("pior_sub", "max"), k=("k", "median"),
    ).reset_index()
    s["ordem"] = s.protocolo.map({p: i for i, p in enumerate(ORDER)})
    s = s.sort_values(["dataset", "ordem"]).drop(columns="ordem")
    s.to_csv(OUT / "ire_escala_resumo.csv", index=False)
    for f in ("ire_escala.csv", "ire_escala_resumo.csv"):
        (Path("docs/resultados/pub") / f).write_bytes((OUT / f).read_bytes())
    pd.set_option("display.width", 200)
    print(s.round(3).to_string())
    figure(res, None)


if __name__ == "__main__":
    main()
