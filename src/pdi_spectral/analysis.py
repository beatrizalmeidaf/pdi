"""E3 — Análise estatística: dose-resposta, previsibilidade espectral e adaptação.

Saídas em ``results/<run_name>/analysis/``:
  conditions_main.csv / conditions_oracle.csv  tabela por (dataset, modelo, condição)
  loto.csv                 R^2 leave-one-technique-out por conjunto de preditores (H2)
  loto_predictions.csv     predições LOTO (para gráfico previsto x observado)
  oracle_transfer.csv      ajuste no oráculo -> previsão nas técnicas clássicas (H3)
  band_sensitivity.csv     dano por banda (oráculo)
  image_level.csv          AUC da previsão de *decision flip* por imagem
  adaptation_summary.csv   domain shift x perda de informação (H4)
  summary.md               relatório-resumo
  figures/*.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["dataset", "model", "family", "cond_id", "technique", "level", "group", "dose"]


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_kind(run_dir: Path, kind: str):
    base = run_dir / kind
    if not base.exists():
        return None, None
    preds, specs = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if (d / "predictions.parquet").exists():
            preds.append(pd.read_parquet(d / "predictions.parquet"))
            specs.append(pd.read_parquet(d / "spectral.parquet"))
    if not preds:
        return None, None
    return pd.concat(preds, ignore_index=True), pd.concat(specs, ignore_index=True)


# ---------------------------------------------------------------------------
# Tabela por condição
# ---------------------------------------------------------------------------
def _mcnemar_p(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    from scipy.stats import binom

    n = b + c
    k = np.minimum(b, c)
    p = np.where(n > 0, np.minimum(1.0, 2 * binom.cdf(k, np.maximum(n, 1), 0.5)), 1.0)
    return p


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def condition_table(pred: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "model", "image_id", "correct", "correct_top5"] + (["correct_sub"] if "correct_sub" in pred else [])
    orig = pred[pred.cond_id == "original"][cols].rename(columns={c: c + "_orig" for c in cols[3:]})
    p = pred[pred.cond_id != "original"].merge(orig, on=["dataset", "model", "image_id"])
    p["_d"] = p["correct"] - p["correct_orig"]
    p["_lost"] = ((p.correct_orig == 1) & (p.correct == 0)).astype(int)
    p["_gain"] = ((p.correct_orig == 0) & (p.correct == 1)).astype(int)
    agg = dict(n=("correct", "size"), acc=("correct", "mean"), acc_orig=("correct_orig", "mean"),
               acc_top5=("correct_top5", "mean"), acc_top5_orig=("correct_top5_orig", "mean"),
               flip=("flip", "mean"), kl=("kl", "mean"), dp_true=("dp_true", "mean"), conf=("conf", "mean"),
               d_std=("_d", "std"), lost=("_lost", "sum"), gained=("_gain", "sum"))
    if "correct_sub" in p:
        agg.update(acc_sub=("correct_sub", "mean"), acc_sub_orig=("correct_sub_orig", "mean"),
                   flip_sub=("flip_sub", "mean"))
    ct = p.groupby(KEYS, dropna=False).agg(**agg).reset_index()
    ct["delta_acc"] = ct["acc"] - ct["acc_orig"]
    ct["delta_se"] = ct["d_std"] / np.sqrt(ct["n"])
    ct["ci95_lo"] = ct["delta_acc"] - 1.96 * ct["delta_se"]
    ct["ci95_hi"] = ct["delta_acc"] + 1.96 * ct["delta_se"]
    ct["mcnemar_p"] = _mcnemar_p(ct["lost"].to_numpy(), ct["gained"].to_numpy())
    ct["mcnemar_p_holm"] = ct.groupby(["dataset", "model"])["mcnemar_p"].transform(lambda s: _holm(s.to_numpy()))
    ct["dose_rank"] = ct.groupby(["dataset", "technique"])["level"].transform(lambda s: (s + 1) / (s.max() + 1))
    num = spec.select_dtypes("number").columns
    sm = spec.groupby(["dataset", "cond_id"])[list(num)].mean().reset_index()
    return ct.merge(sm, on=["dataset", "cond_id"], how="left")


# ---------------------------------------------------------------------------
# Previsibilidade
# ---------------------------------------------------------------------------
def _band_names(ct: pd.DataFrame) -> list[str]:
    return [c[len("coh_"):] for c in ct.columns if c.startswith("coh_")]


def _fine_names(ct: pd.DataFrame) -> list[str]:
    return sorted(c[len("dErel_"):] for c in ct.columns if c.startswith("dErel_f"))


def feature_matrix(df: pd.DataFrame, fs: str) -> pd.DataFrame:
    bands, fine = _band_names(df), _fine_names(df)

    def posneg(cols):
        out = {}
        for c in cols:
            out[c + "+"] = df[c].clip(lower=0)
            out[c + "-"] = df[c].clip(upper=0)
        return pd.DataFrame(out, index=df.index)

    if fs == "dose_rank":
        return df[["dose_rank"]]
    if fs == "psnr":
        return df[["psnr"]]
    if fs == "ssim":
        return df[["ssim"]]
    if fs == "spec_bands":
        return posneg([f"dErel_{b}" for b in bands])
    if fs == "spec_bands_dB":
        return df[[f"dB_{b}" for b in bands]]
    if fs == "delta_bands":
        return df[[f"D_{b}" for b in bands]]
    if fs == "spec_fine":
        return posneg([f"dErel_{f}" for f in fine])
    if fs == "spec_bands+coh":
        return pd.concat([posneg([f"dErel_{b}" for b in bands]), df[[f"coh_{b}" for b in bands]]], axis=1)
    if fs == "spec_fine+coh":
        return pd.concat([posneg([f"dErel_{f}" for f in fine]), df[[f"coh_{b}" for b in bands]]], axis=1)
    raise ValueError(fs)


def _ridge():
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def _scores(y, yhat) -> dict:
    from scipy.stats import spearmanr

    m = ~np.isnan(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) < 3:
        return dict(r2=np.nan, spearman=np.nan, mae=np.nan, n=len(y))
    sst = ((y - y.mean()) ** 2).sum()
    return dict(r2=1 - ((y - yhat) ** 2).sum() / sst if sst > 0 else np.nan,
                spearman=spearmanr(y, yhat).statistic, mae=np.abs(y - yhat).mean(), n=len(y))


def loto(ct: pd.DataFrame, feature_sets: list[str], target: str = "delta_acc"):
    res, preds = [], []
    for (ds, model), g in ct.groupby(["dataset", "model"]):
        g = g.dropna(subset=[target]).reset_index(drop=True)
        y = g[target].to_numpy()
        for fs in feature_sets:
            X = feature_matrix(g, fs).to_numpy()
            yhat = np.full(len(g), np.nan)
            for t in g.technique.unique():
                te = (g.technique == t).to_numpy()
                if (~te).sum() < 4:
                    continue
                yhat[te] = _ridge().fit(X[~te], y[~te]).predict(X[te])
            res.append(dict(dataset=ds, model=model, family=g.family.iloc[0], features=fs, target=target, **_scores(y, yhat)))
            preds.append(g[["dataset", "model", "cond_id", "technique", "level", target]].assign(features=fs, pred=yhat))
    return pd.DataFrame(res), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def oracle_transfer(ct_or: pd.DataFrame, ct_main: pd.DataFrame, feature_sets: list[str], target="delta_acc"):
    res = []
    for (ds, model), g in ct_main.groupby(["dataset", "model"]):
        o = ct_or[(ct_or.dataset == ds) & (ct_or.model == model)].dropna(subset=[target])
        if len(o) < 4:
            continue
        for fs in feature_sets:
            mdl = _ridge().fit(feature_matrix(o, fs).to_numpy(), o[target].to_numpy())
            yhat = mdl.predict(feature_matrix(g, fs).to_numpy())
            res.append(dict(dataset=ds, model=model, family=g.family.iloc[0], features=fs, **_scores(g[target].to_numpy(), yhat)))
    return pd.DataFrame(res)


def image_level(pred, spec, feature_sets, max_rows=200_000, seed=0):
    """AUC (validação agrupada por técnica) da previsão de *decision flip* por imagem."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = pred[pred.cond_id != "original"][["dataset", "model", "image_id", "cond_id", "technique", "flip"]]
    df = p.merge(spec, on=["dataset", "image_id", "cond_id"])
    res = []
    for (ds, model), g in df.groupby(["dataset", "model"]):
        if len(g) > max_rows:
            g = g.sample(max_rows, random_state=seed)
        g = g.reset_index(drop=True)
        y = g["flip"].to_numpy()
        for fs in feature_sets:
            X = feature_matrix(g, fs).to_numpy()
            score = np.full(len(g), np.nan)
            for t in g.technique.unique():
                te = (g.technique == t).to_numpy()
                if len(np.unique(y[~te])) < 2:
                    continue
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[~te], y[~te])
                score[te] = m.predict_proba(X[te])[:, 1]
            ok = ~np.isnan(score)
            auc = roc_auc_score(y[ok], score[ok]) if len(np.unique(y[ok])) == 2 else np.nan
            res.append(dict(dataset=ds, model=model, features=fs, auc=auc, n=int(ok.sum()), flip_rate=y.mean()))
    return pd.DataFrame(res)


def adaptation_summary(adapt: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["acc_orig", "acc_frozen", "acc_bn", "acc_probe_oo", "acc_probe_op", "acc_probe_pp",
                        "acc_ft_oo", "acc_ft_pp", "recoverable_frac", "info_loss"] if c in adapt]
    return adapt.groupby(["dataset", "technique", "level", "cond_id", "group"])[cols].mean().reset_index()


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------
def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}", max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "_(sem dados)_\n"
    df = df.head(max_rows)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_analysis(cfg: dict) -> Path:
    from . import plots

    run_dir = Path(cfg["paths"]["results"]) / cfg["run_name"]
    out = run_dir / "analysis"
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    acfg = cfg.get("analysis", {})
    fsets = acfg.get("feature_sets", ["dose_rank", "psnr", "ssim", "delta_bands", "spec_bands", "spec_bands+coh", "spec_fine"])
    img_fsets = acfg.get("image_feature_sets", ["psnr", "spec_bands", "spec_bands+coh"])
    report = [f"# Resumo da análise — `{cfg['run_name']}`\n"]

    pred_m, spec_m = load_kind(run_dir, "main")
    pred_o, spec_o = load_kind(run_dir, "oracle")
    ct_m = ct_o = None
    if pred_m is not None:
        ct_m = condition_table(pred_m, spec_m)
        ct_m.to_csv(out / "conditions_main.csv", index=False)
        base = pred_m[pred_m.cond_id == "original"].groupby(["dataset", "model"])["correct"].mean().reset_index()
        report += ["## Acurácia nas imagens originais\n", _md(base.rename(columns={"correct": "acc_top1"}))]
        cols = ["dataset", "model", "technique", "level", "dose", "acc", "delta_acc", "ci95_lo", "ci95_hi", "flip",
                "mcnemar_p_holm", "dErel_low", "dErel_mid", "dErel_high", "psnr"]
        cols = [c for c in cols if c in ct_m]
        report += ["## E2 — Dose-resposta (inferência congelada)\n",
                   "Δacc em proporção (ex.: -0.05 = -5 p.p.). `flip` = fração de predições que mudaram.\n",
                   _md(ct_m.sort_values(["dataset", "model", "technique", "level"])[cols], max_rows=400)]
        lo_res, lo_pred = loto(ct_m, fsets)
        lo_res.to_csv(out / "loto.csv", index=False)
        lo_pred.to_csv(out / "loto_predictions.csv", index=False)
        piv = lo_res.pivot_table(index=["dataset", "model"], columns="features", values="r2").reset_index()
        report += ["## E3 — Previsibilidade (H2): R² leave-one-technique-out\n",
                   "Cada técnica é prevista por um modelo ajustado apenas nas outras. "
                   "`dose_rank` = parâmetro nominal (baseline da literatura); `psnr`/`ssim` = fidelidade em pixel; "
                   "`spec_*` = ΔE medido por banda.\n", _md(piv)]
        if acfg.get("image_level", True):
            il = image_level(pred_m, spec_m, img_fsets, acfg.get("image_level_max_rows", 200_000), cfg["seed"])
            il.to_csv(out / "image_level.csv", index=False)
            report += ["## E3 — Nível de imagem: AUC para prever *decision flip*\n", _md(il)]
        plots.dose_response(ct_m, fig)
        plots.delta_vs_energy(ct_m, fig, "dErel_high")
        plots.delta_vs_energy(ct_m, fig, "D_high")
        plots.loto_bars(lo_res, fig)
        best = lo_res.groupby("features")["r2"].mean().idxmax()
        plots.pred_vs_obs(lo_pred[lo_pred.features == best], fig, best)

    if pred_o is not None:
        ct_o = condition_table(pred_o, spec_o)
        ct_o.to_csv(out / "conditions_oracle.csv", index=False)
        bs = ct_o[ct_o.technique == "oracle_band"][["dataset", "model", "family", "cond_id", "dose", "delta_acc", "flip"]]
        bs.to_csv(out / "band_sensitivity.csv", index=False)
        report += ["## E1 — Sensibilidade por banda (oráculo)\n",
                   "`dose` = centro do anel (ciclos/pixel); ganho 0 = banda removida, ganho 2 = banda amplificada.\n",
                   _md(bs.pivot_table(index=["dataset", "model"], columns="cond_id", values="delta_acc").reset_index())]
        plots.band_sensitivity(ct_o, run_dir / "main", fig)
        if ct_m is not None:
            tr = oracle_transfer(ct_o, ct_m, acfg.get("transfer_feature_sets", ["spec_bands", "spec_fine", "spec_bands+coh"]))
            tr.to_csv(out / "oracle_transfer.csv", index=False)
            report += ["## E3 — Transferência oráculo → técnicas clássicas (H3)\n",
                       "Modelo ajustado SÓ nos filtros ideais e aplicado às técnicas reais.\n", _md(tr)]

    ad_path = run_dir / "adaptation" / "adaptation.csv"
    if ad_path.exists():
        ad = pd.read_csv(ad_path)
        sm = adaptation_summary(ad)
        sm.to_csv(out / "adaptation_summary.csv", index=False)
        report += ["## E4 — Domain shift × perda de informação (H4)\n",
                   "`recoverable_frac` ≈ 1: a queda é recuperada por re-treino da sonda (domain shift); "
                   "≈ 0: informação destruída. `info_loss` = probe_oo − probe_pp.\n", _md(sm)]
        plots.adaptation_bars(ad, fig)

    (out / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[analyze] relatório em {out / 'summary.md'}")
    return out
