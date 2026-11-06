"""Limiares de dose e Índice de Risco Espectral (IRE).

1) LIMIARES (modelagem dose-resposta, emprestando o vocabulário da toxicologia)
   Para cada (dataset, modelo, técnica), com a intensidade crescente x:
     - BMD_tau  "benchmark dose": menor x em que Δacc cai tau (ex.: 1 p.p.);
                BMDL = limite inferior do IC (dose segura conservadora)
     - BMD-flip menor x em que tau das predições mudam (dispensa rótulos)
     - NOAEL / LOAEL  maior dose testada sem dano significativo / primeira com dano
                significativo (IC de Δacc inteiro abaixo de 0)
     - melhor dose: argmax Δacc; ganho significativo se o IC fica acima de 0
                (hormese: melhora em dose baixa, piora em dose alta)
     - dose perceptual: menor x em que a técnica atinge seu objetivo declarado
                (ex.: +10 % de nitidez, -10 % de ruído, +10 % de contraste)
     - JANELA OPERACIONAL = [dose perceptual, BMDL]: a imagem já melhorou e o
                classificador ainda não perdeu tau. Vazia = todo ganho visual custa acurácia.
   Os ICs vêm de bootstrap pareado sobre as imagens (as mesmas imagens em todas as doses).

2) ÍNDICE DE RISCO ESPECTRAL (IRE) — proposta deste trabalho
       IRE(x, T) = sum_b  w_b * D_b(x, T(x))          w_b >= 0
       P(flip)   = 1 - exp(-IRE)
   D_b = energia da perturbação T(x) - x na banda b, relativa à energia da imagem.
   Leitura: cada unidade de energia alterada na banda b é uma "chance" independente
   (taxa w_b) de virar a decisão — um modelo de risco/Poisson. Os pesos w_b são o
   perfil de vulnerabilidade espectral do modelo, calibrado uma vez.
   Garantia: Δacc >= -P(flip) sempre (só predições que mudam podem virar erro). Então
       orçamento espectral:  sum_b w_b D_b <= -ln(1 - tau)   =>   queda de acurácia <= tau
   e a dose segura de QUALQUER técnica sai só do espectro, sem rodar a rede.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import load_kind


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def technique_meta(cfg: dict, name: str) -> dict:
    tc = cfg.get("techniques", {}).get(name, {})
    return dict(identity=tc.get("identity"), reverse=tc.get("reverse"), objective=tc.get("objective"),
                unit=tc.get("dose", ""))


def intensity(dose, meta: dict):
    d = np.asarray(dose, dtype=float)
    return meta["reverse"] - d if meta.get("reverse") is not None else d


def crossing(x: np.ndarray, Y: np.ndarray, level: float, mode: str):
    """Primeiro cruzamento de ``level`` (interpolação linear) em cada linha de Y.

    x (K,) crescente; Y (B, K). mode='below': y <= level; 'above': y >= level.
    Devolve (dose (B,), status (B,)) com status 0=ok, 1=abaixo da 1a dose, 2=censurado (nunca cruza).
    """
    Y = np.atleast_2d(Y)
    hit = Y <= level if mode == "below" else Y >= level
    has = hit.any(1)
    k = np.where(has, hit.argmax(1), 0)
    dose = np.full(len(Y), np.nan)
    status = np.where(has, 0, 2)
    first = has & (k == 0)
    dose[first] = x[0]
    status[first] = 1
    m = has & (k > 0)
    if m.any():
        r = np.arange(len(Y))[m]
        k1 = k[m]
        y0, y1 = Y[r, k1 - 1], Y[r, k1]
        x0, x1 = x[k1 - 1], x[k1]
        t = np.where(np.abs(y1 - y0) > 1e-12, (level - y0) / (y1 - y0), 1.0)
        dose[m] = x0 + np.clip(t, 0, 1) * (x1 - x0)
    return dose, status


def status_label(dose: float, status: int, x_tested_min: float, anchored: bool) -> str:
    """ok | <x_min (já cruza na 1a dose testada) | >x_max (não cruza) |
    entre_id_e_x_min (cruza entre a dose identidade e a 1a testada: valor interpolado, não medido)."""
    if status == 1:
        return "<x_min"
    if status == 2:
        return ">x_max"
    if anchored and np.isfinite(dose) and dose < x_tested_min:
        return "entre_id_e_x_min"
    return "ok"


def _boot_weights(n: int, B: int, rng) -> np.ndarray:
    """Pesos de bootstrap (B, n) normalizados: média reamostrada = M @ W.T."""
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
    return W / n


def _summ(d, status, ci):
    ok = status != 2
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    if ok.sum() == 0:
        return np.nan, np.nan, 1.0
    return float(np.nanquantile(d[ok], lo_q)), float(np.nanquantile(d[ok], hi_q)), float((~ok).mean())


# ---------------------------------------------------------------------------
# 1) Limiares
# ---------------------------------------------------------------------------
def compute_thresholds(pred: pd.DataFrame, spec: pd.DataFrame, cfg: dict):
    tcfg = cfg.get("thresholds", {})
    acc_tols = tcfg.get("acc_tol", [0.01, 0.05])
    flip_tols = tcfg.get("flip_tol", [0.01, 0.05])
    B = int(tcfg.get("bootstrap", 1000))
    ci = float(tcfg.get("ci", 0.90))
    rng = np.random.default_rng(cfg.get("seed", 0))

    orig = pred[pred.cond_id == "original"][["dataset", "model", "image_id", "correct"]].rename(columns={"correct": "c0"})
    p = pred[pred.cond_id != "original"].merge(orig, on=["dataset", "model", "image_id"])
    p["d"] = p["correct"] - p["c0"]
    qcols = [c for c in spec.columns if c.startswith("dq_")]
    qmean = spec.groupby(["dataset", "cond_id"])[qcols].mean().reset_index() if qcols else None
    if qmean is not None and {"q_brisque", "q_brisque_orig"} <= set(spec.columns):
        # BRISQUE pode ser ~0 ou negativo: a média das razões explode; usa a razão das médias
        from .brisque_fix import ratio_of_means

        rb = ratio_of_means(spec)
        qmean = qmean.drop(columns=["dq_brisque"], errors="ignore").merge(rb[["dataset", "cond_id", "dq_brisque"]],
                                                                          on=["dataset", "cond_id"], how="left")

    rows, curves = [], []
    for (ds, model, tech), g in p.groupby(["dataset", "model", "technique"]):
        meta = technique_meta(cfg, tech)
        lv = g.groupby("cond_id")["dose"].first()
        x_lv = intensity(lv.to_numpy(), meta)
        order = np.argsort(x_lv)
        conds, x = lv.index.to_numpy()[order], x_lv[order]
        D = g.pivot_table(index="cond_id", columns="image_id", values="d").loc[conds].to_numpy()
        F = g.pivot_table(index="cond_id", columns="image_id", values="flip").loc[conds].to_numpy()
        n = D.shape[1]
        W = _boot_weights(n, B, rng)
        mD, bD = D.mean(1), D @ W.T            # (K,), (K, B)
        mF, bF = F.mean(1), F @ W.T
        anchored = meta["identity"] is not None
        if anchored:  # dose identidade = imagem original: Δ = 0, flip = 0
            x0 = float(intensity(meta["identity"], meta))
            x = np.r_[x0, x]
            mD, mF = np.r_[0.0, mD], np.r_[0.0, mF]
            bD = np.vstack([np.zeros(B), bD])
            bF = np.vstack([np.zeros(B), bF])
        lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
        D_lo, D_hi = np.quantile(bD, lo_q, axis=1), np.quantile(bD, hi_q, axis=1)
        F_lo, F_hi = np.quantile(bF, lo_q, axis=1), np.quantile(bF, hi_q, axis=1)

        row = dict(dataset=ds, model=model, family=g.family.iloc[0], technique=tech, group=g.group.iloc[0],
                   unit=meta["unit"] + (" (invertido)" if meta.get("reverse") is not None else ""),
                   anchored=anchored, x_min=float(x[1 if anchored else 0]), x_max=float(x[-1]), n_images=n)
        for tau in acc_tols:
            d0, s0 = crossing(x, mD[None], -tau, "below")
            db, sb = crossing(x, bD.T, -tau, "below")
            lo, hi, pc = _summ(db, sb, ci)
            row.update({f"bmd_acc{tau:g}": d0[0], f"bmdl_acc{tau:g}": lo, f"bmdu_acc{tau:g}": hi,
                        f"bmd_acc{tau:g}_status": status_label(d0[0], s0[0], row["x_min"], anchored), f"bmd_acc{tau:g}_pcens": pc})
        for tau in flip_tols:
            d0, s0 = crossing(x, mF[None], tau, "above")
            db, sb = crossing(x, bF.T, tau, "above")
            lo, hi, pc = _summ(db, sb, ci)
            row.update({f"bmd_flip{tau:g}": d0[0], f"bmdl_flip{tau:g}": lo, f"bmdu_flip{tau:g}": hi,
                        f"bmd_flip{tau:g}_status": status_label(d0[0], s0[0], row["x_min"], anchored)})
        # NOAEL / LOAEL (dano significativo = IC inteiro abaixo de 0)
        sig_harm = np.flatnonzero(D_hi < 0)
        row["loael"] = float(x[sig_harm[0]]) if len(sig_harm) else np.nan
        row["noael"] = float(x[sig_harm[0] - 1]) if len(sig_harm) and sig_harm[0] > 0 else (float(x[-1]) if not len(sig_harm) else np.nan)
        # melhor dose / hormese
        k = int(np.argmax(mD))
        row.update(best_dose=float(x[k]), best_gain=float(mD[k]), best_gain_lo=float(D_lo[k]), best_gain_hi=float(D_hi[k]),
                   gain_significant=bool(D_lo[k] > 0 and mD[k] > 0))
        row["hormesis"] = bool(row["gain_significant"] and np.isfinite(row.get(f"bmd_acc{acc_tols[0]:g}", np.nan))
                               and row[f"bmd_acc{acc_tols[0]:g}"] > row["best_dose"])
        # objetivo perceptual da técnica
        obj = meta.get("objective")
        q_curve = np.full(len(x), np.nan)
        if obj and qmean is not None and f"dq_{obj['metric']}" in qmean:
            qm = qmean[qmean.dataset == ds].set_index("cond_id")[f"dq_{obj['metric']}"].reindex(conds).to_numpy()
            sign = -1.0 if obj.get("direction", "up") == "down" else 1.0
            q_curve = np.r_[0.0, qm] if anchored else qm
            dq, sq = crossing(x, (sign * q_curve)[None], float(obj.get("min_gain", 0.1)), "above")
            row.update(objective=f"{obj['metric']} {'↓' if sign < 0 else '↑'} {100*obj.get('min_gain', 0.1):.0f}%",
                       perc_dose=dq[0], perc_status=status_label(dq[0], sq[0], row["x_min"], anchored))
            safe = row.get(f"bmdl_acc{acc_tols[0]:g}")
            if not np.isfinite(safe):  # nunca cruzou: seguro até a maior dose testada
                safe = x[-1] if row[f"bmd_acc{acc_tols[0]:g}_status"] == ">x_max" else np.nan
            row["window_lo"], row["window_hi"] = dq[0], safe
            row["window_exists"] = bool(np.isfinite(dq[0]) and np.isfinite(safe) and dq[0] <= safe)
        # objetivo perceptual GERAL: BRISQUE (menor = melhor), se medido
        if qmean is not None and "dq_brisque" in qmean:
            bq = qmean[qmean.dataset == ds].set_index("cond_id")["dq_brisque"].reindex(conds).to_numpy()
            bq = np.r_[0.0, bq] if anchored else bq
            delta = float(tcfg.get("brisque_min_gain", 0.05))
            d_b, s_b = crossing(x, (-bq)[None], delta, "above")   # melhora >= delta
            d_w, s_w = crossing(x, bq[None], delta, "above")      # piora >= delta
            kb = int(np.nanargmin(bq))
            safe = row.get(f"bmdl_acc{acc_tols[0]:g}")
            if not np.isfinite(safe):
                safe = x[-1] if row[f"bmd_acc{acc_tols[0]:g}_status"] == ">x_max" else np.nan
            row.update(brisque_best_dose=float(x[kb]), brisque_best_change=float(bq[kb]),
                       brisque_better_dose=d_b[0], brisque_better_status=status_label(d_b[0], s_b[0], row["x_min"], anchored),
                       brisque_worse_dose=d_w[0], brisque_worse_status=status_label(d_w[0], s_w[0], row["x_min"], anchored),
                       brisque_window_exists=bool(np.isfinite(d_b[0]) and np.isfinite(safe) and d_b[0] <= safe))
        rows.append(row)
        for i in range(len(x)):
            curves.append(dict(dataset=ds, model=model, technique=tech, x=x[i], d_acc=mD[i], d_acc_lo=D_lo[i],
                               d_acc_hi=D_hi[i], flip=mF[i], flip_lo=F_lo[i], flip_hi=F_hi[i], dq_objective=q_curve[i]))
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ---------------------------------------------------------------------------
# 2) Índice de Risco Espectral
# ---------------------------------------------------------------------------
def ire_features(df: pd.DataFrame, kind: str = "D_fine+chroma+coh") -> list[str]:
    """Nomes das variáveis do IRE.

    D_*     energia da perturbação de luminância por banda
    A_*/P_* a mesma energia separada em amplitude (contraste) e estrutura/fase (D = A + P)
    dc2     mudança de brilho médio
    Dc_*    energia da perturbação de cor (Cb, Cr) por banda
    incoh_* perda de coerência espectral por banda = 1 - coh_* (mudança de ESTRUTURA/fase)
    """
    luma_fine = sorted(c for c in df.columns if c.startswith("D_f"))
    luma_bands = [c for c in df.columns if c.startswith("D_") and not c.startswith("D_f") and c != "D_tot"]
    chroma_fine = sorted(c for c in df.columns if c.startswith("Dc_f")) + (["Dc_dc"] if "Dc_dc" in df else [])
    chroma_bands = [c for c in df.columns if c.startswith("Dc_") and not c.startswith("Dc_f") and c != "Dc_tot"]
    incoh = [f"incoh_{c[4:]}" for c in df.columns if c.startswith("coh_")]
    ap_fine = sorted(c for c in df.columns if c.startswith("A_f")) + sorted(c for c in df.columns if c.startswith("P_f"))
    ap_bands = [c for c in df.columns if (c.startswith("A_") or c.startswith("P_")) and c[2] != "f"]
    base, _, extras = kind.partition("+")
    table = {"D_fine": luma_fine, "D_bands": luma_bands, "AP_fine": ap_fine, "AP_bands": ap_bands}
    if base not in table:
        raise ValueError(kind)
    feats = list(table[base])
    for e in [x for x in extras.split("+") if x]:
        if e == "chroma":
            feats += chroma_fine if base == "D_fine" else chroma_bands
        elif e == "coh":
            feats += incoh
        elif e == "dc":
            feats += ["dc2"] if "dc2" in df else []
        else:
            raise ValueError(kind)
    if not feats or ("chroma" in extras and not (chroma_fine or chroma_bands)):
        raise ValueError(f"sem colunas para {kind} (rode `main` de novo para gerar as features de cor)")
    return feats


def ire_matrix(df, feats: list[str]) -> np.ndarray:
    """Monta X >= 0; incoh_b = 1 - coh_b (coerência limitada a [-1, 1])."""
    cols = []
    for f in feats:
        if f.startswith("incoh_"):
            cols.append(1.0 - np.clip(np.asarray(df["coh_" + f[6:]], dtype=float), -1, 1))
        else:
            cols.append(np.asarray(df[f], dtype=float))
    return np.clip(np.column_stack(cols), 0, None)


def fit_ire(X: np.ndarray, y: np.ndarray, l2: float = 1e-4, gamma: float | None = None) -> np.ndarray:
    """Máxima verossimilhança de P(y=1) = 1 - exp(-sum_j w_j x_j^gamma), w >= 0.

    gamma=None estima o expoente (0.05 a 3); um número fixa o expoente.
    Devolve theta = [w_1..w_d, gamma] com w já nas unidades originais de X.
    """
    from scipy.optimize import minimize

    scale = X.mean(0) + 1e-12
    Z = X / scale
    n, d = Z.shape
    logZ = np.log(np.where(Z > 0, Z, 1.0))
    free = gamma is None

    def f(theta):
        w = theta[:d]
        g = theta[d] if free else gamma
        Zg = Z ** g
        s = Zg @ w + 1e-9
        nll = (-(y * np.log(-np.expm1(-s))) + (1 - y) * s).sum() / n + l2 * w @ w
        r = (1 - y) - y / np.expm1(s)
        grad = [Zg.T @ r / n + 2 * l2 * w]
        if free:
            grad.append([r @ ((Zg * logZ) @ w) / n])
        return nll, np.concatenate(grad)

    th0 = np.r_[np.full(d, 0.05), [1.0] if free else []]
    bounds = [(0, None)] * d + ([(0.05, 3.0)] if free else [])
    res = minimize(f, th0, jac=True, method="L-BFGS-B", bounds=bounds)
    g = res.x[d] if free else gamma
    return np.r_[res.x[:d] * scale ** (-g), g]


def ire_score(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return (X ** theta[-1]) @ theta[:-1]


def ire_prob(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return -np.expm1(-ire_score(X, theta))


def _per_image(pred, spec, feats):
    p = pred[pred.cond_id != "original"][["dataset", "model", "family", "image_id", "cond_id", "technique", "dose", "flip"]]
    raw = sorted({("coh_" + f[6:]) if f.startswith("incoh_") else f for f in feats})
    return p.merge(spec[["dataset", "image_id", "cond_id"] + raw], on=["dataset", "image_id", "cond_id"])


def run_ire(pred, spec, cfg, thr: pd.DataFrame, pred_or=None, spec_or=None):
    icfg = cfg.get("ire", {})
    kind = icfg.get("features", "D_fine+chroma+coh")
    max_rows = int(icfg.get("max_rows", 300_000))
    l2 = float(icfg.get("l2", 1e-4))
    gamma = icfg.get("gamma")  # None = estimado
    gamma = None if gamma in (None, "auto") else float(gamma)
    flip_tol = cfg.get("thresholds", {}).get("flip_tol", [0.01])[0]
    rng = np.random.default_rng(cfg.get("seed", 0))
    feats = ire_features(spec, kind)
    df = _per_image(pred, spec, feats)
    df_or = _per_image(pred_or, spec_or, feats) if pred_or is not None else None

    weights, cond_pred, thr_pred, summary = [], [], [], []
    for (ds, model), g in df.groupby(["dataset", "model"]):
        g = g.reset_index(drop=True)
        X, y = ire_matrix(g, feats), g["flip"].to_numpy().astype(float)
        sub = rng.choice(len(g), min(len(g), max_rows), replace=False)
        w = fit_ire(X[sub], y[sub], l2, gamma)
        weights.append(pd.DataFrame({"dataset": ds, "model": model, "feature": feats + ["_gamma"], "w": w, "fit": "all"}))
        # LOTO: cada técnica prevista por pesos ajustados nas outras
        yhat = np.full(len(g), np.nan)
        for t in g.technique.unique():
            te = (g.technique == t).to_numpy()
            tr = np.flatnonzero(~te)
            tr = rng.choice(tr, min(len(tr), max_rows), replace=False)
            yhat[te] = ire_prob(X[te], fit_ire(X[tr], y[tr], l2, gamma))
        g["p_loto"] = yhat
        g["p_all"] = ire_prob(X, w)
        if df_or is not None:
            o = df_or[(df_or.dataset == ds) & (df_or.model == model)]
            if len(o):
                so = rng.choice(len(o), min(len(o), max_rows), replace=False)
                w_or = fit_ire(ire_matrix(o, feats)[so], o["flip"].to_numpy().astype(float)[so], l2, gamma)
                weights.append(pd.DataFrame({"dataset": ds, "model": model, "feature": feats + ["_gamma"], "w": w_or, "fit": "oracle"}))
                g["p_oracle"] = ire_prob(X, w_or)
        agg = {c: (c, "mean") for c in ["flip", "p_loto", "p_all"] + (["p_oracle"] if "p_oracle" in g else [])}
        cp = g.groupby(["cond_id", "technique", "dose"]).agg(**agg).reset_index().assign(dataset=ds, model=model)
        cond_pred.append(cp)
        srow = dict(dataset=ds, model=model, family=g.family.iloc[0])
        for col in [c for c in ("p_loto", "p_all", "p_oracle") if c in cp]:
            srow[f"r2_{col}"] = _r2(cp.flip.to_numpy(), cp[col].to_numpy())
            srow[f"spearman_{col}"] = cp[["flip", col]].corr("spearman").iloc[0, 1]
        # limiar previsto pela fórmula (LOTO) x observado
        for t, h in cp.groupby("technique"):
            meta = technique_meta(cfg, t)
            x = intensity(h.dose.to_numpy(), meta)
            o = np.argsort(x)
            x, pl = x[o], h.p_loto.to_numpy()[o]
            if meta["identity"] is not None:
                x, pl = np.r_[float(intensity(meta["identity"], meta)), x], np.r_[0.0, pl]
            d_pred, s_pred = crossing(x, pl[None], flip_tol, "above")
            obs = thr[(thr.dataset == ds) & (thr.model == model) & (thr.technique == t)]
            d_obs = obs[f"bmd_flip{flip_tol:g}"].iloc[0] if len(obs) else np.nan
            lo = obs[f"bmdl_flip{flip_tol:g}"].iloc[0] if len(obs) else np.nan
            hi = obs[f"bmdu_flip{flip_tol:g}"].iloc[0] if len(obs) else np.nan
            thr_pred.append(dict(dataset=ds, model=model, technique=t, tau_flip=flip_tol, dose_pred=d_pred[0],
                                 pred_status=["ok", "<x_min", ">x_max"][s_pred[0]], dose_obs=d_obs, obs_lo=lo, obs_hi=hi,
                                 within_ci=bool(np.isfinite(d_pred[0]) and np.isfinite(lo) and lo <= d_pred[0] <= hi)))
        summary.append(srow)
    return (pd.concat(weights, ignore_index=True), pd.concat(cond_pred, ignore_index=True),
            pd.DataFrame(thr_pred), pd.DataFrame(summary))


def _r2(y, yhat):
    m = np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / sst) if sst > 0 else np.nan


# ---------------------------------------------------------------------------
# Margem conformal para a dose segura de uma técnica NOVA
# ---------------------------------------------------------------------------
def margin_scores(cp: pd.DataFrame, cfg: dict, tau: float = 0.01) -> pd.DataFrame:
    """Escore de erro da dose segura, por (modelo, técnica), na escala do flip.

    Para cada curva (técnica × modelo), acha a dose em que o flip OBSERVADO cruza τ e lê o flip que o IRE (LOTO: técnica
    fora do ajuste) previa ali: p̂. score = τ / p̂. A regra "aceitar a dose se o IRE prevê flip <= c·τ" deixa a técnica
    segura nesse modelo se c·score <= 1 (com p̂ crescente na dose). score > 1 = a fórmula subestimou o dano.
    Curvas cujo flip observado nunca cruza τ recebem score 0 (qualquer dose testada é segura).
    ``cp`` é ire_condition_predictions.csv. Funciona também para o JPEG, cujo limiar pode cair na 1ª dose (qualidade 100).
    """
    rows = []
    for (ds, model, t), g in cp.groupby(["dataset", "model", "technique"]):
        meta = technique_meta(cfg, t)
        x = intensity(g.dose.to_numpy(), meta)
        o = np.argsort(x)
        x, obs, pred = x[o], g.flip.to_numpy()[o], g.p_loto.to_numpy()[o]
        if meta.get("identity") is not None:
            x = np.r_[float(intensity(meta["identity"], meta)), x]
            obs, pred = np.r_[0.0, obs], np.r_[0.0, pred]
        d_obs, st = crossing(x, obs[None], tau, "above")
        if st[0] == 2:
            sc, p_at = 0.0, np.nan
        else:
            p_at = float(np.interp(d_obs[0], x, pred))
            sc = tau / p_at if p_at > 0 else np.inf
        rows.append(dict(dataset=ds, model=model, technique=t, dose_obs=float(d_obs[0]), p_prev=p_at, score=sc,
                         monotona=bool(np.all(np.diff(pred) >= -1e-9))))
    return pd.DataFrame(rows)


def conformal_margin(scores: pd.DataFrame, alpha: float, unit: str = "technique") -> float:
    """Fator c tal que, para uma técnica nova trocável com as de ``scores``, P(todos os modelos seguros) >= 1 − α.

    Uso: aceitar uma dose só se o IRE prevê flip <= c·τ (orçamento −ln(1 − c·τ)).

    unit="technique": um escore por técnica (o pior modelo), quantil conformal sobre as técnicas (cobertura
    garantida por técnica, sob troca entre técnicas). unit="pair": quantil sobre os pares modelo × técnica
    (cobertura média por par; não garantida, porque os pares de uma técnica são correlacionados).
    Devolve nan se houver poucas unidades para o α pedido (precisa de n >= 1/α − 1).
    """
    v = (scores.groupby("technique").score.max() if unit == "technique" else scores.score).to_numpy()
    n = len(v)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        return float("nan")
    q = np.sort(v)[k - 1]
    return float(1 / q) if q > 0 else 1.0


def budget(tau: float) -> float:
    """Orçamento de IRE para que a queda de acurácia (e o flip) fique <= tau."""
    return float(-np.log1p(-tau))


# ---------------------------------------------------------------------------
# Dose segura sem rede: aplica a técnica numa grade e mede só o espectro
# ---------------------------------------------------------------------------
def safe_dose(cfg: dict, technique: str, model: str, dataset: str, grid: list[float], tau: float,
              n_images: int = 200, weights_csv: Path | None = None, margin: float | None = None) -> pd.DataFrame:
    """Doses de ``grid`` em que o IRE prevê flip <= margin·τ (sem rodar a rede).

    ``margin`` (fator conformal c, ver scripts/ire_conformal.py) protege técnicas que o IRE não viu: com c calibrado
    deixando técnicas de fora, uma técnica nova fica segura em todos os modelos com probabilidade >= 1 − α.
    Padrão: ``ire.safe_margin`` da config (1 = sem margem).
    """
    from . import techniques as T
    from .datasets import load_dataset

    if margin is None:
        margin = float(cfg.get("ire", {}).get("safe_margin", 1.0))
    from .spectral import SpectralAnalyzer, SpectralConfig

    run_dir = Path(cfg["paths"]["results"]) / cfg["run_name"]
    wdf = pd.read_csv(weights_csv or run_dir / "analysis" / "ire_weights.csv")
    wdf = wdf[(wdf.dataset == dataset) & (wdf.model == model) & (wdf.fit == cfg.get("ire", {}).get("use_fit", "all"))]
    if wdf.empty:
        raise ValueError(f"sem pesos IRE para {dataset}/{model}: rode `analyze` antes")
    feats, theta = wdf.feature.tolist()[:-1], wdf.w.to_numpy()  # última linha = _gamma
    imgs = load_dataset(dataset, "val", n_images, cfg["seed"], cfg["paths"]).load_all()
    an = SpectralAnalyzer(SpectralConfig.from_dict(cfg.get("spectral")), imgs.shape[1:3])
    Fx, dcx = an.transform(imgs)
    tc = cfg["techniques"].get(technique, {})
    pname = tc.get("dose") or next(iter(tc["levels"][0]))
    fixed = {k: v for k, v in tc["levels"][0].items() if k != pname}
    rows = []
    for d in grid:
        params = {**fixed, pname: type(tc["levels"][0][pname])(d)}
        if technique in T.SR_TECHNIQUES:
            params["_model_dir"] = cfg["paths"].get("sr_models", "data/sr_models")
        proc = np.stack([T.apply(technique, im, params, cfg.get("backend", "scratch")) for im in imgs])
        m = an.compare(imgs, proc, Fx, dcx, with_quality=False)
        X = ire_matrix(m, feats)
        p = ire_prob(X, theta)
        rows.append(dict(technique=technique, model=model, dose=d, ire_mean=float(ire_score(X, theta).mean()),
                         p_flip_pred=float(p.mean()), safe=bool(p.mean() <= margin * tau)))
    out = pd.DataFrame(rows)
    out.attrs["budget"] = budget(margin * tau)
    out.attrs["margin"] = margin
    return out
