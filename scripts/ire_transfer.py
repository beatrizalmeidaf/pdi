"""O IRE transfere entre modelos? Calibra num conjunto de modelos e prevê outro.

    python scripts/ire_transfer.py                       # todas as rodadas pub_* (em paralelo)
    python scripts/ire_transfer.py --runs pub_cifar10_s0 --max-rows 50000

Protocolos (alvo = flip por condição do modelo-alvo, R² como em ire_summary.csv):
  proprio_loto         pesos do próprio modelo, técnica prevista fora do ajuste (o número do artigo até aqui)
  outros_modelos       pesos ajustados nos OUTROS modelos da mesma base (todas as técnicas)
  outros_modelos_loto  pesos ajustados nos outros modelos E nas outras técnicas: modelo e técnica nunca vistos
  par                  pesos de UM outro modelo (matriz origem × destino)
  outra_base           pesos do mesmo modelo calibrados na outra base de 224 px (Imagenette <-> ImageNet-val)

Saída: results/pub_agregado/ire_transfer*.csv, docs/resultados/pub/ e docs/figures/ire_transfer.png.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
OUT = R / "pub_agregado"
RUNS = [f"pub_cifar10_s{s}" for s in range(3)] + [f"pub_imagenette_s{s}" for s in range(3)] + ["pub_imagenet"]
FEATS_KIND = "D_fine+chroma+coh"
L2 = 1e-4


def _load(run: str, kind: str = "main"):
    import pyarrow.parquet as pq

    from pdi_spectral.analysis import drop_images, flat_images
    from pdi_spectral.thresholds import ire_features

    base = R / run / kind
    ds = next(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("_"))
    d = base / ds
    names = pq.ParquetFile(d / "spectral.parquet").schema.names
    feats = ire_features(pd.DataFrame(columns=names), FEATS_KIND)
    raw = sorted({("coh_" + f[6:]) if f.startswith("incoh_") else f for f in feats})
    qcols = [c for c in ("q_contrast", "dq_contrast") if c in names]
    spec = pd.read_parquet(d / "spectral.parquet", columns=["image_id", "cond_id"] + qcols + raw)
    pred = pd.read_parquet(d / "predictions.parquet",
                           columns=["image_id", "model", "cond_id", "technique", "dose", "flip"])
    spec["dataset"], pred["dataset"] = ds, ds
    bad = flat_images(spec, 1 / 255) if qcols else pd.DataFrame(columns=["dataset", "image_id"])
    if kind != "main":  # o oráculo não guarda contraste: usa a lista do main
        f = R / run / "analysis" / "excluded_flat_images.csv"
        bad = pd.read_csv(f) if f.exists() else bad
    spec, pred = drop_images(spec, bad), drop_images(pred, bad)
    pred = pred[pred.cond_id != "original"]
    df = pred.merge(spec[["dataset", "image_id", "cond_id"] + raw], on=["dataset", "image_id", "cond_id"])
    return ds, feats, df


def _cond_r2(g: pd.DataFrame, p: np.ndarray):
    from pdi_spectral.thresholds import _r2

    c = g.assign(p=p).groupby("cond_id")[["flip", "p"]].mean()
    bias = float((c.p - c.flip).mean())
    return _r2(c.flip.to_numpy(), c.p.to_numpy()), float(c.corr("spearman").iloc[0, 1]), bias


def run_one(run: str, max_rows: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pdi_spectral.thresholds import fit_ire, ire_matrix, ire_prob

    rng = np.random.default_rng(seed)
    ds, feats, df = _load(run)
    models = sorted(df.model.unique())
    by = {m: df[df.model == m].reset_index(drop=True) for m in models}
    X = {m: ire_matrix(by[m], feats) for m in models}
    y = {m: by[m].flip.to_numpy().astype(float) for m in models}

    def fit(rows: list[tuple[str, np.ndarray]]):
        """Ajusta nos (modelo, máscara) dados, com o mesmo nº de linhas por modelo (nenhum domina)."""
        per = max(1, max_rows // len(rows))
        Xs, ys = [], []
        for m, mask in rows:
            idx = np.flatnonzero(mask)
            idx = rng.choice(idx, min(len(idx), per), replace=False)
            Xs.append(X[m][idx]), ys.append(y[m][idx])
        return fit_ire(np.vstack(Xs), np.concatenate(ys), L2, None)

    allm = {m: np.ones(len(by[m]), bool) for m in models}
    own = {m: fit([(m, allm[m])]) for m in models}
    res, pair = [], []
    for m in models:
        g, others = by[m], [o for o in models if o != m]
        # pesos do próprio modelo, LOTO
        p = np.full(len(g), np.nan)
        for t in g.technique.unique():
            te = (g.technique == t).to_numpy()
            p[te] = ire_prob(X[m][te], fit([(m, ~te)]))
        res.append(dict(protocolo="proprio_loto", p=p))
        # outros modelos, todas as técnicas
        res.append(dict(protocolo="outros_modelos", p=ire_prob(X[m], fit([(o, allm[o]) for o in others]))))
        # outros modelos e outras técnicas
        p = np.full(len(g), np.nan)
        for t in g.technique.unique():
            te = (g.technique == t).to_numpy()
            w = fit([(o, (by[o].technique != t).to_numpy()) for o in others])
            p[te] = ire_prob(X[m][te], w)
        res.append(dict(protocolo="outros_modelos_loto", p=p))
        for r in res[-3:]:
            r2, sp, bias = _cond_r2(g, r.pop("p"))
            r.update(dataset=ds, run=run, model=m, r2=r2, spearman=sp, vies=bias)
        # pares origem -> destino
        for o in models:
            r2, sp, bias = _cond_r2(g, ire_prob(X[m], own[o]))
            pair.append(dict(dataset=ds, run=run, origem=o, destino=m, r2=r2, spearman=sp, vies=bias))
    w = pd.concat([pd.DataFrame({"dataset": ds, "run": run, "model": m, "feature": feats + ["_gamma"], "w": own[m]})
                   for m in models])
    w.to_csv(OUT / "ire_transfer_pesos" / f"{run}.csv", index=False)
    print(f"[transfer] {run}: ok", flush=True)
    return pd.DataFrame(res), pd.DataFrame(pair)


def cross_dataset(seed: int = 0) -> pd.DataFrame:
    """Pesos de um modelo calibrados numa base de 224 px aplicados à outra (mesmo modelo)."""
    from pdi_spectral.thresholds import ire_matrix, ire_prob

    wdir = OUT / "ire_transfer_pesos"
    rows = []
    pairs = [("pub_imagenette_s0", "pub_imagenet"), ("pub_imagenet", "pub_imagenette_s0")]
    for src, dst in pairs:
        if not ((wdir / f"{src}.csv").exists() and (R / dst).exists()):
            continue
        w = pd.read_csv(wdir / f"{src}.csv")
        ds, feats, df = _load(dst)
        for m, g in df.groupby("model"):
            wm = w[w.model == m]
            if wm.empty:
                continue
            assert wm.feature.tolist()[:-1] == feats
            r2, sp, bias = _cond_r2(g, ire_prob(ire_matrix(g, feats), wm.w.to_numpy()))
            rows.append(dict(protocolo="outra_base", origem=src, dataset=ds, run=dst, model=m, r2=r2, spearman=sp,
                             vies=bias))
    return pd.DataFrame(rows)


def figure(res: pd.DataFrame, pair: pd.DataFrame):
    from pdi_spectral.report import C, _finish, _style, _title

    plt = _style()
    order = ["proprio_loto", "outros_modelos", "outros_modelos_loto", "outra_base"]
    lab = {"proprio_loto": "próprio modelo\n(técnica nova)", "outros_modelos": "outros modelos",
           "outros_modelos_loto": "outros modelos\n+ técnica nova", "outra_base": "outra base\n(mesmo modelo)"}
    dsl = [("cifar10", "CIFAR-10", C["s1"]), ("imagenette", "Imagenette", C["s2"]), ("imagenet_val", "ImageNet-val", C["s3"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    x = np.arange(len(order))
    for k, (ds, name, col) in enumerate(dsl):
        v = [res[(res.dataset == ds) & (res.protocolo == p)].r2.mean() for p in order]
        lo = [res[(res.dataset == ds) & (res.protocolo == p)].r2.min() for p in order]
        hi = [res[(res.dataset == ds) & (res.protocolo == p)].r2.max() for p in order]
        xx = x + (k - 1) * 0.26
        a1.bar(xx, np.nan_to_num(v), 0.24, color=col, label=name)
        a1.vlines(xx, lo, hi, color=C["ink2"], lw=1)
    a1.set_xticks(x, [lab[p] for p in order], fontsize=8.5)
    a1.set_ylim(0, 1)
    a1.set_ylabel("R² do flip por condição")
    a1.set_title("Média dos modelos (traço: mín.–máx.)", loc="left")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    p = pair[pair.run == "pub_imagenet"]
    if p.empty:
        p = pair[pair.run == pair.run.iloc[0]]
    M = p.pivot(index="origem", columns="destino", values="r2")
    short = {"convnext_tiny": "ConvNeXt", "efficientnet_b0": "EffNet-B0", "resnet18": "R18", "resnet50": "R50",
             "swin_t": "Swin", "vgg16_bn": "VGG16", "vit_b_16": "ViT"}
    im = a2.imshow(M.clip(lower=0).values, cmap="Greens", vmin=0, vmax=1)
    a2.set_xticks(range(len(M.columns)), [short.get(c, c) for c in M.columns], rotation=45, ha="right", fontsize=8)
    a2.set_yticks(range(len(M.index)), [short.get(c, c) for c in M.index], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            a2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else C["ink"])
    a2.grid(False)
    a2.set_xlabel("aplicado em")
    a2.set_ylabel("calibrado em")
    a2.set_title("ImageNet-val: de um modelo para outro", loc="left")
    _title(fig, "O IRE calibrado num modelo serve para outro?",
           "R² prevendo a mudança de decisão de um modelo com pesos ajustados em outros modelos, em outras técnicas "
           "ou na outra base de 224 px.")
    _finish(fig)
    fig.savefig("docs/figures/ire_transfer.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--max-rows", type=int, default=150_000)
    ap.add_argument("--jobs", type=int, default=7)
    a = ap.parse_args()
    (OUT / "ire_transfer_pesos").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    with ProcessPoolExecutor(a.jobs) as ex:
        outs = list(ex.map(run_one, a.runs, [a.max_rows] * len(a.runs)))
    res = pd.concat([o[0] for o in outs], ignore_index=True)
    pair = pd.concat([o[1] for o in outs], ignore_index=True)
    res = pd.concat([res, cross_dataset()], ignore_index=True)
    res.to_csv(OUT / "ire_transfer.csv", index=False)
    pair.to_csv(OUT / "ire_transfer_pares.csv", index=False)
    s = res.groupby(["dataset", "protocolo"]).agg(r2=("r2", "mean"), r2_min=("r2", "min"), r2_max=("r2", "max"),
                                                    spearman=("spearman", "mean"), vies=("vies", "mean")).reset_index()
    s.to_csv(OUT / "ire_transfer_resumo.csv", index=False)
    Path("docs/resultados/pub").mkdir(parents=True, exist_ok=True)
    for f in ("ire_transfer.csv", "ire_transfer_pares.csv", "ire_transfer_resumo.csv"):
        (Path("docs/resultados/pub") / f).write_bytes((OUT / f).read_bytes())
    print(s.round(3).to_string())
    figure(res, pair)


if __name__ == "__main__":
    main()
