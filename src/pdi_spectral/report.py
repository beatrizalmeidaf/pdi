"""Relatório público: figuras do README, dados e amostras do site (GitHub Pages).

    python -m pdi_spectral report            # ou: make report

Lê as análises já rodadas (results/sweep e results/sweep_imagenette, mais o fine-tuning
em results/finetune_cifar) e escreve:
  docs/figures/*.png          figuras do README (tema claro)
  docs/assets/data.json       dados compactos para os gráficos interativos do site
  docs/samples/<ds>/*.png     amostras: cada técnica em cada dose, em imagens reais
Nada aqui roda classificador: só lê CSVs e aplica as técnicas às amostras.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rótulos e paleta
# ---------------------------------------------------------------------------
RUNS = {"cifar10": "sweep", "imagenette": "sweep_imagenette"}
DS_LABEL = {"cifar10": "CIFAR-10 (32 px)", "imagenette": "Imagenette (224 px)"}
MODEL_LABEL = {
    "cifar10_resnet20": "ResNet-20", "cifar10_resnet56": "ResNet-56", "cifar10_vgg16_bn": "VGG-16-BN",
    "efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet-50", "vit_b_16": "ViT-B/16",
}
MODEL_ORDER = {"cifar10": ["cifar10_resnet20", "cifar10_resnet56", "cifar10_vgg16_bn"],
               "imagenette": ["resnet50", "efficientnet_b0", "vit_b_16"]}
TECH_LABEL = {
    "gaussian": "Gaussiano", "median": "Mediana", "bilateral": "Bilateral", "wiener": "Wiener",
    "jpeg": "JPEG 4:2:0", "jpeg444": "JPEG 4:4:4", "hist_eq": "Equalização (HE)", "clahe": "CLAHE",
    "clahe_blend": "CLAHE misturado", "unsharp": "Unsharp masking", "laplacian": "Laplaciano",
    "sr_bicubic": "SR bicúbica", "sr_espcn": "SR ESPCN", "sr_fsrcnn": "SR FSRCNN",
}
TECH_PARAM = {
    "gaussian": "σ", "median": "k", "bilateral": "σ_s", "wiener": "k", "jpeg": "100 − qualidade",
    "jpeg444": "100 − qualidade", "hist_eq": "α", "clahe": "clip", "clahe_blend": "α", "unsharp": "amount",
    "laplacian": "c", "sr_bicubic": "escala", "sr_espcn": "escala", "sr_fsrcnn": "escala",
}
# efeito espectral esperado (3 grupos = 3 cores)
EFFECT = {"unsharp": "adiciona", "laplacian": "adiciona", "hist_eq": "redistribui", "clahe": "redistribui",
          "clahe_blend": "redistribui"}
TECH_ORDER = ["unsharp", "laplacian", "hist_eq", "clahe", "clahe_blend", "gaussian", "bilateral", "median",
              "wiener", "jpeg", "jpeg444", "sr_bicubic", "sr_espcn", "sr_fsrcnn"]

# Paleta de referência (validada: 3 slots passam em todos os pares, claro e escuro)
C = dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
         muted="#898781", grid="#e1e0d9", axis="#c3c2b7", good="#0ca30c", critical="#d03b3b", neutral="#f0efec")
DS_COLOR = {"cifar10": C["s1"], "imagenette": C["s2"]}
SLOTS = [C["s1"], C["s2"], C["s3"]]


def effect(t: str) -> str:
    return EFFECT.get(t, "remove")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load(results: Path) -> dict:
    d = {}
    for ds, run in RUNS.items():
        a = results / run / "analysis"
        if not a.exists():
            print(f"[report] aviso: {a} não existe; {ds} fica de fora")
            continue
        d[ds] = {
            "curves": pd.read_csv(a / "threshold_curves.csv"),
            "cond": pd.read_csv(a / "conditions_main.csv"),
            "thr": pd.read_csv(a / "thresholds.csv"),
            "bands": pd.read_csv(a / "band_sensitivity.csv"),
            "loto": pd.read_csv(a / "loto.csv"),
            "ire": pd.read_csv(a / "ire_summary.csv"),
            "ire_cp": pd.read_csv(a / "ire_condition_predictions.csv"),
            "adapt": pd.read_csv(results / run / "adaptation" / "adaptation.csv")
            if (results / run / "adaptation" / "adaptation.csv").exists() else None,
        }
        bq = a / "brisque_conditions.csv"  # razão das médias (ver brisque_fix.py)
        if bq.exists():
            b = pd.read_csv(bq)[["cond_id", "dq_brisque"]]
            d[ds]["cond"] = d[ds]["cond"].drop(columns=["dq_brisque"]).merge(b, on="cond_id", how="left")
    ft = results / "finetune_cifar" / "adaptation" / "adaptation.csv"
    if ft.exists() and "cifar10" in d:
        d["cifar10"]["finetune"] = pd.read_csv(ft)
    return d


# ---------------------------------------------------------------------------
# Estilo das figuras
# ---------------------------------------------------------------------------
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
        "axes.titlecolor": C["ink"], "axes.labelcolor": C["ink2"], "axes.edgecolor": C["axis"],
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": C["grid"], "grid.linewidth": 0.8,
        "grid.linestyle": "-", "xtick.color": C["muted"], "ytick.color": C["muted"], "xtick.labelcolor": C["ink2"],
        "ytick.labelcolor": C["ink2"], "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.titlepad": 8, "axes.axisbelow": True,
    })
    return plt


def _title(fig, title, subtitle=None, y=None):
    """Guarda o título; _finish() posiciona com espaço reservado no topo."""
    fig._pdi_title = (title, subtitle)


def _finish(fig):
    import textwrap

    w, h = fig.get_size_inches()
    title, subtitle = getattr(fig, "_pdi_title", (None, None))
    lines = textwrap.wrap(subtitle, int(w * 15.5)) if subtitle else []
    head = 0.42 + 0.19 * len(lines) + (0.12 if lines else 0)
    fig.tight_layout(rect=[0, 0, 1, 1 - head / h])
    if title:
        fig.text(0.012, 1 - 0.12 / h, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=C["ink"])
    for i, ln in enumerate(lines):
        fig.text(0.012, 1 - (0.45 + 0.19 * i) / h, ln, ha="left", va="top", fontsize=9.3, color=C["ink2"])


def _zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=C["axis"], lw=1, zorder=1)


def _mean_band(curves, tech, ds_models):
    g = curves[(curves.technique == tech) & curves.model.isin(ds_models)]
    p = g.pivot_table(index="x", columns="model", values="d_acc") * 100
    return p.index.to_numpy(), p.mean(1).to_numpy(), p.min(1).to_numpy(), p.max(1).to_numpy()


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_dose_curves(d, out):
    plt = _style()
    techs = ["gaussian", "unsharp", "laplacian", "hist_eq", "clahe_blend", "jpeg444"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, t in zip(axes.flat, techs):
        for ds in d:
            x, m, lo, hi = _mean_band(d[ds]["curves"], t, MODEL_ORDER[ds])
            col = DS_COLOR[ds]
            ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(x, m, color=col, label=DS_LABEL[ds])
            ax.plot(x[-1], m[-1], "o", ms=5, color=col, mec=C["surface"], mew=1.5)
        _zero(ax)
        ax.set_title(TECH_LABEL[t])
        ax.set_xlabel(TECH_PARAM[t])
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ acurácia (p.p.)")
    axes[0, 0].legend(loc="lower left")
    _title(fig, "Dose-resposta: quanto a acurácia muda com a intensidade",
           "Linha = média dos 3 modelos de cada dataset; faixa = do pior ao melhor modelo. Acima de 0 = a técnica melhora o classificador.")
    _finish(fig)
    fig.savefig(out / "curvas_dose.png")
    plt.close(fig)


def fig_bands(d, out):
    plt = _style()
    fig, axes = plt.subplots(1, len(d), figsize=(11, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, d):
        b = d[ds]["bands"]
        b = b[b.cond_id.str.contains("_g0@")]
        for i, m in enumerate(MODEL_ORDER[ds]):
            g = b[b.model == m].sort_values("dose")
            ax.plot(g.dose, 100 * g.delta_acc, color=SLOTS[i], label=MODEL_LABEL[m], marker="o", ms=4,
                    mec=C["surface"], mew=1.2)
        _zero(ax)
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("centro do anel removido (ciclos/pixel)")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Δ acurácia (p.p.)")
    _title(fig, "A banda de frequência que o classificador usa muda com a resolução",
           "Filtro ideal que remove um anel de frequência por vez. Em 32 px dói a média-alta frequência; em 224 px, a mais baixa — e o ViT depende quase só dela.")
    _finish(fig)
    fig.savefig(out / "bandas_criticas.png")
    plt.close(fig)


def fig_windows(d, out):
    plt = _style()
    from matplotlib.patches import FancyBboxPatch

    cols = [(ds, m) for ds in d for m in MODEL_ORDER[ds]]
    techs = [t for t in TECH_ORDER if t not in ("sr_espcn", "sr_fsrcnn")]
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(techs) + 1.9))
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(techs), 0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(techs)) + 0.5, [f"{TECH_LABEL[t]}  ({TECH_PARAM[t]})" for t in techs])
    ax.tick_params(length=0)
    for j, (ds, m) in enumerate(cols):
        thr = d[ds]["thr"]
        for i, t in enumerate(techs):
            r = thr[(thr.model == m) & (thr.technique == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            w = r.get("window_exists")
            bmd, st = r.get("bmd_acc0.01"), r.get("bmd_acc0.01_status")
            if isinstance(w, (bool, np.bool_)) or w in ("True", "False"):
                ok = str(w) == "True"
                if ok and st == "<x_min":  # objetivo e limiar na mesma (1a) dose: janela vazia na prática
                    face, txt_c, mark = ("#f3dfa6", C["ink"], "~")
                else:
                    face, txt_c, mark = (C["good"], "#ffffff", "✓") if ok else ("#ebe9e3", C["ink2"], "✗")
            else:
                face, txt_c, mark = (C["surface"], C["muted"], "")
            ax.add_patch(FancyBboxPatch((j + 0.06, i + 0.08), 0.88, 0.84, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc=face, ec="none"))
            if st == "<x_min":
                val = "< mín."
            elif st == ">x_max":
                val = "> máx."
            elif np.isfinite(bmd):
                val = f"{bmd:.2f}" if bmd < 10 else f"{bmd:.0f}"
            else:
                val = "—"
            ax.text(j + 0.5, i + 0.52, f"{mark} {val}".strip(), ha="center", va="center", fontsize=8.5, color=txt_c)
    for j, (ds, m) in enumerate(cols):
        ax.text(j + 0.5, -0.35, MODEL_LABEL[m], ha="center", va="bottom", fontsize=8.5, color=C["ink2"])
    for k, ds in enumerate(d):
        j0 = k * 3
        ax.text(j0 + 1.5, -1.05, DS_LABEL[ds], ha="center", va="bottom", fontsize=10, fontweight="semibold",
                color=C["ink"])
        ax.plot([j0 + 0.1, j0 + 2.9], [-0.95, -0.95], color=C["axis"], lw=1, clip_on=False)
    _title(fig, "Limiar de dano e janela operacional por técnica",
           "Número = intensidade em que a acurácia cai 1 p.p. (BMD).  ✓ verde = existe janela: o objetivo visual "
           "(+10% nitidez, −10% ruído ou +10% contraste) chega antes do limite seguro.  ✗ cinza = não existe.  "
           "~ = só no limite (o menor passo já custa ≥ 1 p.p.).  Sem fundo = técnica sem objetivo visual definido.")
    _finish(fig)
    fig.savefig(out / "janelas_operacionais.png")
    plt.close(fig)


def fig_hormesis(d, out):
    plt = _style()
    techs = [t for t in TECH_ORDER if t not in ("sr_espcn", "sr_fsrcnn", "sr_bicubic")]
    fig, axes = plt.subplots(1, len(d), figsize=(11, 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, d):
        thr = d[ds]["thr"]
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        for i, m in enumerate(MODEL_ORDER[ds]):
            off = (i - 1) * 0.24
            for k, t in enumerate(techs):
                r = thr[(thr.model == m) & (thr.technique == t)]
                if r.empty:
                    continue
                r = r.iloc[0]
                g, lo, hi = 100 * r.best_gain, 100 * r.best_gain_lo, 100 * r.best_gain_hi
                ax.plot([lo, hi], [k + off, k + off], color=SLOTS[i], lw=1.6, alpha=0.9)
                ax.plot(g, k + off, "o", ms=6 if r.gain_significant else 4.5, color=SLOTS[i],
                        mec=C["surface"], mew=1.2, label=MODEL_LABEL[m] if k == 0 else None)
        _zero(ax, "x")
        ax.set_yticks(range(len(techs)), [TECH_LABEL[t] for t in techs])
        ax.set_ylim(len(techs) - 0.5, -0.5)
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("melhor Δ acurácia em alguma dose (p.p.), IC 90%")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    _title(fig, "Hormese: alguma dose melhora o classificador?",
           "Ponto maior = ganho com IC acima de zero. No CIFAR, nenhum. Em 224 px, ganhos claros só no aguçamento leve "
           "(laplaciano até +2,8 p.p.; unsharp até +2,0 p.p.). Os +0,3 e +0,4 p.p. da ResNet-50 em CLAHE misturado e "
           "JPEG 4:4:4 estão no limite: com 42 testes a IC 90%, ~4 falsos positivos são esperados ao acaso.")
    _finish(fig)
    fig.savefig(out / "hormese.png")
    plt.close(fig)


PREDICTORS = [("dose_rank", "Parâmetro nominal (σ, k, clip…)"), ("psnr", "PSNR"), ("ssim", "SSIM"),
              ("delta_bands", "Energia da perturbação (Yin et al.)"), ("spec_bands", "ΔE por banda"),
              ("spec_bands+coh", "ΔE + coerência de fase"), ("spec_bands+coh+chroma", "ΔE + coerência + cor")]


def fig_predictors(d, out):
    plt = _style()
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    n = len(d)
    h = 0.34
    for k, ds in enumerate(d):
        l = d[ds]["loto"]
        for i, (f, lab) in enumerate(PREDICTORS):
            v = l[l.features == f].r2.clip(lower=-0.1)
            if v.empty:
                continue
            y = i + (k - (n - 1) / 2) * h
            ax.barh(y, v.mean(), height=h * 0.86, color=DS_COLOR[ds], label=DS_LABEL[ds] if i == 0 else None)
            ax.plot([v.min(), v.max()], [y, y], color=C["ink2"], lw=1)
            ax.text(max(v.mean(), 0) + 0.015, y, f"{v.mean():.2f}", va="center", fontsize=8.5, color=C["ink2"])
    ax.set_yticks(range(len(PREDICTORS)), [p[1] for p in PREDICTORS])
    ax.set_ylim(len(PREDICTORS) - 0.4, -0.6)
    ax.set_xlim(-0.1, 1.0)
    _zero(ax, "x")
    ax.set_xlabel("R² prevendo técnicas nunca vistas (leave-one-technique-out)")
    ax.legend(loc="upper right")
    _title(fig, "Medir o espectro prevê o dano; o parâmetro nominal não",
           "Barra = média dos 3 modelos; traço = do pior ao melhor modelo.")
    _finish(fig)
    fig.savefig(out / "preditores.png")
    plt.close(fig)


IRE_STEPS = [("Energia de luminância (linear)", 0.13), ("+ expoente γ", 0.23), ("+ perda de coerência (fase)", 0.85),
             ("+ perturbação de cor", 0.87)]


def fig_ire(d, out):
    plt = _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a1.grid(axis="x")
    a1.grid(axis="y", visible=False)
    for i, (lab, v) in enumerate(IRE_STEPS):
        a1.barh(i, v, height=0.5, color=C["s1"])
        a1.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=9, color=C["ink2"])
    a1.set_yticks(range(len(IRE_STEPS)), [s[0] for s in IRE_STEPS])
    a1.set_ylim(len(IRE_STEPS) - 0.5, -0.5)
    a1.set_xlim(0, 1)
    a1.set_xlabel("R² (CIFAR-10, ResNet-20, técnica nunca vista)")
    a1.set_title("Como a fórmula chegou à forma final")
    mx = 0
    for ds in d:
        cp = d[ds]["ire_cp"]
        a2.scatter(100 * cp.flip, 100 * cp.p_loto, s=16, color=DS_COLOR[ds], alpha=0.75, lw=0, label=DS_LABEL[ds])
        mx = max(mx, 100 * cp.flip.max(), 100 * cp.p_loto.max())
    a2.plot([0, mx], [0, mx], color=C["axis"], lw=1)
    a2.set_xlabel("decisões que mudaram, observado (%)")
    a2.set_ylabel("previsto pelo IRE (%)")
    a2.set_title("Previsto × observado (cada ponto = técnica × dose × modelo)")
    a2.grid(axis="x")
    a2.legend(loc="upper left")
    r2 = {ds: d[ds]["ire"].r2_p_loto for ds in d}
    sub = "  ·  ".join(f"{DS_LABEL[ds]}: R² {v.min():.2f}–{v.max():.2f}" for ds, v in r2.items())
    _title(fig, "Índice de Risco Espectral: P(mudar a decisão) = 1 − exp(−IRE)", sub)
    _finish(fig)
    fig.savefig(out / "ire.png")
    plt.close(fig)


def fig_adaptation(d, out):
    plt = _style()
    ft = d.get("cifar10", {}).get("finetune")
    if ft is None:
        return
    g = ft.groupby(["technique", "dose"]).agg(frozen=("acc_frozen", "mean"), bn=("acc_bn", "mean"),
                                             ft=("acc_ft_pp", "mean"), ctrl=("acc_ft_oo", "mean")).reset_index()
    g["rec"] = (g.ft - g.frozen) / (g.ctrl - g.frozen)
    g = g.sort_values("rec", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for i, r in enumerate(g.itertuples()):
        ax.plot([100 * r.frozen, 100 * r.ft], [i, i], color=C["axis"], lw=2, zorder=1)
        ax.plot(100 * r.ctrl, i, "|", ms=14, mew=2, color=C["ink2"], zorder=2, label="controle (fine-tuning nas originais)" if i == 0 else None)
        for val, col, lab in ((r.frozen, C["s1"], "congelado"), (r.bn, C["s2"], "só re-estimar BatchNorm"),
                              (r.ft, C["s3"], "fine-tuning (3 épocas)")):
            ax.plot(100 * val, i, "o", ms=7, color=col, mec=C["surface"], mew=1.5, zorder=3, label=lab if i == 0 else None)
        ax.text(100 * r.ctrl + 1.0, i, f"{100 * r.rec:.0f}% recuperado" if r.ctrl - r.frozen > 0.02 else "queda pequena",
                va="center", fontsize=8.5, color=C["ink2"])
    labels = [f"{TECH_LABEL[r.technique]} ({TECH_PARAM[r.technique]} {r.dose:g})" for r in g.itertuples()]
    ax.set_yticks(range(len(g)), labels)
    ax.set_ylim(len(g) - 0.5, -0.5)
    ax.set_xlim(35, 108)
    ax.set_xlabel("acurácia (%) — CIFAR-10, média dos 3 modelos")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=4, fontsize=8)
    _title(fig, "Domain shift ou informação perdida?",
           "Se re-treinar recupera a queda, a informação continuava na imagem. Aguçamento e blur leve voltam; JPEG e mediana, não.")
    _finish(fig)
    fig.savefig(out / "adaptacao.png")
    plt.close(fig)


CNN_VIT = [("gaussian", 1.0), ("gaussian", 2.0), ("bilateral", 4.0), ("median", 5), ("wiener", 5), ("jpeg", 10),
           ("clahe", 4.0), ("unsharp", 3.0), ("laplacian", 1.5)]


def fig_cnn_vit(d, out):
    if "imagenette" not in d:
        return
    plt = _style()
    c = d["imagenette"]["cond"]
    fig, ax = plt.subplots(figsize=(10.5, 4))
    w = 0.26
    for i, m in enumerate(MODEL_ORDER["imagenette"]):
        vals = []
        for t, dose in CNN_VIT:
            r = c[(c.model == m) & (c.technique == t) & np.isclose(c.dose, dose)]
            vals.append(100 * r.delta_acc.iloc[0] if len(r) else np.nan)
        x = np.arange(len(CNN_VIT)) + (i - 1) * w
        ax.bar(x, vals, width=w * 0.88, color=SLOTS[i], label=MODEL_LABEL[m])
    _zero(ax)
    ax.set_xticks(range(len(CNN_VIT)), [(f"{TECH_LABEL[t]}\nqualidade {int(dose)}" if t.startswith("jpeg") else
                                         f"{TECH_LABEL[t].replace(' masking', '')}\n{TECH_PARAM[t]} = {dose:g}")
                                        for t, dose in CNN_VIT], fontsize=8.2)
    ax.set_ylabel("Δ acurácia (p.p.)")
    ax.legend(loc="lower left")
    _title(fig, "CNN × Vision Transformer em 224 px",
           "Na mesma dose, o ViT perde até 2× menos com técnicas que removem frequência — mas não é o mais robusto ao aguçamento forte.")
    _finish(fig)
    fig.savefig(out / "cnn_vit.png")
    plt.close(fig)


def fig_jpeg(d, out):
    plt = _style()
    fig, axes = plt.subplots(1, len(d), figsize=(11, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, d):
        cv = d[ds]["curves"]
        for t, col in (("jpeg", C["s1"]), ("jpeg444", C["s2"])):
            x, m, lo, hi = _mean_band(cv, t, MODEL_ORDER[ds])
            q = 100 - x
            keep = q >= 80
            ax.fill_between(q[keep], lo[keep], hi[keep], color=col, alpha=0.12, lw=0)
            ax.plot(q[keep], m[keep], color=col, marker="o", ms=4, mec=C["surface"], mew=1.2,
                    label="4:2:0 (padrão: subamostra a cor)" if t == "jpeg" else "4:4:4 (cor intacta)")
        _zero(ax)
        ax.axhline(-1, color=C["muted"], lw=0.8)
        ax.text(80.3, -1.15, "−1 p.p.", va="top", fontsize=8, color=C["muted"])
        ax.set_xlim(101, 79)
        ax.set_xticks([100, 98, 95, 90, 85, 80])
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("qualidade JPEG")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("Δ acurácia (p.p.)")
    _title(fig, "No JPEG de alta qualidade, quem muda a decisão é a cor",
           "Mesmo na qualidade 100, a subamostragem de croma 4:2:0 já custa até 1,4 p.p. (CIFAR) e 1,5 p.p. (ResNet-50); sem ela, ~0.")
    _finish(fig)
    fig.savefig(out / "jpeg_cor.png")
    plt.close(fig)


def fig_tradeoff(d, out):
    plt = _style()
    techs = [("unsharp", C["s1"]), ("laplacian", C["s2"]), ("gaussian", C["s3"])]
    fig, axes = plt.subplots(1, len(d), figsize=(11, 4.3), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, d):
        c = d[ds]["cond"]
        ax.grid(axis="x")
        for t, col in techs:
            g = c[c.technique == t].groupby("dose").agg(acc=("delta_acc", "mean"), b=("dq_brisque", "mean")).reset_index()
            g = g.sort_values("dose")
            xs, ys = np.r_[0, -100 * g.b], np.r_[0, 100 * g.acc]
            ax.plot(xs, ys, color=col, marker="o", ms=4, mec=C["surface"], mew=1.2, label=TECH_LABEL[t])
            ax.annotate(f"{TECH_PARAM[t]} {g.dose.iloc[-1]:g}", (xs[-1], ys[-1]), xytext=(4, -10),
                        textcoords="offset points", fontsize=7.5, color=C["ink2"])
        if ds == "imagenette":
            ax.set_xlim(-75, 25)
            ax.set_ylim(-12, 4)
            ax.annotate("Gaussiano forte sai do gráfico:\nσ 3 = −260% de BRISQUE e −35 p.p.", xy=(-75, -7.5),
                        xytext=(-73, -11.6), fontsize=7.5, color=C["ink2"])
        _zero(ax)
        _zero(ax, "x")
        ax.plot(0, 0, "o", ms=7, color=C["ink"], mec=C["surface"], mew=1.5)
        ax.annotate("original", (0, 0), xytext=(6, 6), textcoords="offset points", fontsize=8, color=C["ink2"])
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("ganho de qualidade perceptual (BRISQUE, %)  →  melhor")
        ax.set_ylabel("Δ acurácia (p.p.)")
        ax.legend(loc="lower right" if ds == "imagenette" else "lower left")
    _title(fig, "\"Melhorar\" a imagem melhora o classificador?",
           "Cada ponto é uma dose (média dos modelos). Quadrante de cima à direita = melhor para pessoas e para a rede. "
           "Em 32 px a qualidade sobe só às custas da acurácia; em 224 px o aguçamento leve ganha nos dois. "
           "BRISQUE resumido pela variação da média (robusto a imagens com BRISQUE ≈ 0).")
    _finish(fig)
    fig.savefig(out / "tradeoff_brisque.png")
    plt.close(fig)


def fig_samples_strip(samples_root: Path, manifest: dict, d: dict, out: Path):
    """Faixa de amostras para o README: uma imagem do Imagenette sob várias técnicas/doses, com Δacc médio."""
    from PIL import Image

    plt = _style()
    ds = "imagenette" if "imagenette" in manifest else "cifar10"
    picks = [("unsharp", 0.5), ("unsharp", 3.0), ("laplacian", 0.3), ("gaussian", 1.0), ("gaussian", 3.0),
             ("hist_eq", 1.0), ("jpeg", 10), ("sr_bicubic", 4.0)]
    cond = d[ds]["cond"]
    fig, axes = plt.subplots(1, len(picks) + 1, figsize=(13, 2.1))
    img0 = manifest[ds]["images"][0]
    axes[0].imshow(Image.open(samples_root / img0["original"]))
    axes[0].set_title("original", fontsize=9)
    for ax, (t, dose) in zip(axes[1:], picks):
        lv = [lv for lv in manifest[ds]["techniques"].get(t, []) if np.isclose(lv["dose"], dose)]
        if not lv:
            ax.axis("off")
            continue
        ax.imshow(Image.open(samples_root / lv[0]["files"][0]))
        r = cond[(cond.technique == t) & np.isclose(cond.dose, dose)]
        dacc = 100 * r.delta_acc.mean()
        name = {"unsharp": "Unsharp", "hist_eq": "Equalização", "sr_bicubic": "SR bicúbica"}.get(t, TECH_LABEL[t].split(" (")[0])
        val = f"Q{int(dose)}" if t.startswith("jpeg") else f"{TECH_PARAM[t]} {dose:g}"
        ax.set_title(f"{name}\n{val}", fontsize=8.5)
        ax.text(0.5, -0.08, f"{dacc:+.1f} p.p.", transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=C["good"] if dacc > 0.2 else (C["critical"] if dacc < -1 else C["ink2"]), fontweight="semibold")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.005, right=0.995, wspace=0.08)
    fig.text(0.5, -0.04, f"Δ = variação média de acurácia dos 3 modelos no {DS_LABEL[ds]} (2000 imagens).",
             fontsize=8.5, color=C["ink2"], ha="center")
    fig.savefig(out / "amostras.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Amostras para o site
# ---------------------------------------------------------------------------
def make_samples(cfg: dict, out_root: Path, n_images: dict | None = None) -> dict:
    """Aplica cada técnica, em cada dose do sweep, a poucas imagens reais. Devolve o manifesto."""
    from PIL import Image

    from . import techniques as T
    from .datasets import load_dataset
    from .pipeline import build_conditions, filter_available

    n_images = n_images or {"cifar10": 4, "imagenette": 2}
    conds = filter_available(build_conditions(cfg, "main"))
    manifest = {}
    for ds, n in n_images.items():
        try:
            imgs = load_dataset(ds, "val", 2000, cfg["seed"], cfg["paths"])
        except Exception as e:  # dataset não baixado
            print(f"[report] amostras de {ds} puladas: {e}")
            continue
        rng = np.random.default_rng(3)
        idx = sorted(rng.choice(len(imgs), size=n, replace=False))
        x = [imgs.loader(i) for i in idx]
        ext = "png" if ds == "cifar10" else "webp"
        d = out_root / ds
        d.mkdir(parents=True, exist_ok=True)

        def save(arr, name):
            p = d / f"{name}.{ext}"
            im = Image.fromarray(arr)
            im.save(p, lossless=True, quality=100, method=4) if ext == "webp" else im.save(p, optimize=True)
            return f"{ds}/{p.name}"

        m = {"images": [], "techniques": {}}
        for k, im in enumerate(x):
            m["images"].append({"original": save(im, f"o{k}"), "label": str(imgs.ids[idx[k]])})
        for c in conds:
            files = [save(T.apply(c.technique, im, c.params, cfg.get("backend", "reference")),
                          f"{c.technique}_{c.level}_{k}") for k, im in enumerate(x)]
            m["techniques"].setdefault(c.technique, []).append(
                {"level": c.level, "dose": c.dose, "files": files})
        manifest[ds] = m
        print(f"[report] amostras {ds}: {len(x)} imagens x {len(conds)} condições")
    return manifest


# ---------------------------------------------------------------------------
# Dados do site
# ---------------------------------------------------------------------------
def _f(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def site_data(d: dict, manifest: dict, cfg: dict) -> dict:
    from .thresholds import intensity, technique_meta

    out = {"datasets": {}, "techniques": {t: {"label": TECH_LABEL[t], "param": TECH_PARAM[t], "effect": effect(t)}
                                          for t in TECH_ORDER}}
    for ds, dd in d.items():
        cond, cv, thr = dd["cond"], dd["curves"], dd["thr"]
        models = MODEL_ORDER[ds]
        base = cond.groupby("model").acc_orig.first()
        o = {"label": DS_LABEL[ds], "models": [{"id": m, "label": MODEL_LABEL[m], "acc": _f(base.get(m))} for m in models],
             "curves": {}, "thresholds": {}, "quality": {}, "bands": {}, "loto": {}, "ire": {}, "ire_points": [],
             "samples": manifest.get(ds)}
        for t in TECH_ORDER:
            g = cv[cv.technique == t]
            if g.empty:
                continue
            o["curves"][t] = {m: {k: [_f(v) for v in g[g.model == m].sort_values("x")[c]]
                                  for k, c in (("x", "x"), ("d", "d_acc"), ("lo", "d_acc_lo"), ("hi", "d_acc_hi"),
                                               ("flip", "flip"))} for m in models}
            q = cond[cond.technique == t].groupby("dose")[["dq_brisque", "dq_sharpness", "dq_noise", "dq_contrast",
                                                          "psnr", "ssim"]].mean().reset_index()
            meta = technique_meta(cfg, t)
            q["x"] = intensity(q.dose.to_numpy(), meta)
            q = q.sort_values("x")
            o["quality"][t] = {k: [_f(v) for v in q[k]] for k in ["x", "dose", "dq_brisque", "dq_sharpness",
                                                                   "dq_noise", "dq_contrast", "psnr", "ssim"]}
            o["thresholds"][t] = {}
            for m in models:
                r = thr[(thr.model == m) & (thr.technique == t)]
                if r.empty:
                    continue
                r = r.iloc[0]
                o["thresholds"][t][m] = {k: (_f(r.get(k)) if not isinstance(r.get(k), str) else r.get(k)) for k in
                                         ["bmd_acc0.01", "bmdl_acc0.01", "bmdu_acc0.01", "bmd_acc0.01_status",
                                          "bmd_acc0.05", "bmd_flip0.01", "noael", "loael", "best_dose", "best_gain",
                                          "best_gain_lo", "best_gain_hi", "perc_dose", "window_lo", "window_hi",
                                          "brisque_better_dose", "brisque_worse_dose", "objective"]}
                o["thresholds"][t][m]["window_exists"] = str(r.get("window_exists")) == "True"
                o["thresholds"][t][m]["gain_significant"] = bool(r.get("gain_significant"))
                o["thresholds"][t][m]["hormesis"] = bool(r.get("hormesis"))
        b = dd["bands"]
        for m in models:
            g = b[(b.model == m) & b.cond_id.str.contains("_g0@")].sort_values("dose")
            g2 = b[(b.model == m) & b.cond_id.str.contains("_g2@")].sort_values("dose")
            o["bands"][m] = {"x": [_f(v) for v in g.dose], "remove": [_f(v) for v in g.delta_acc],
                             "boost": [_f(v) for v in g2.delta_acc]}
        lo = dd["loto"]
        for f, lab in PREDICTORS:
            o["loto"][f] = {"label": lab, "r2": {m: _f(lo[(lo.model == m) & (lo.features == f)].r2.mean()) for m in models}}
        ire = dd["ire"]
        for m in models:
            r = ire[ire.model == m]
            if len(r):
                r = r.iloc[0]
                o["ire"][m] = {k: _f(r.get(k)) for k in ["r2_p_loto", "spearman_p_loto", "r2_p_oracle", "spearman_p_oracle"]}
        cp = dd["ire_cp"]
        o["ire_points"] = [{"m": r.model, "t": r.technique, "dose": _f(r.dose), "obs": _f(r.flip), "pred": _f(r.p_loto)}
                           for r in cp.itertuples()]
        out["datasets"][ds] = o
    # adaptação
    ad = {}
    if "cifar10" in d and d["cifar10"].get("finetune") is not None:
        ft = d["cifar10"]["finetune"]
        ad["cifar10"] = [{"t": t, "dose": _f(dose), "frozen": _f(g.acc_frozen.mean()), "bn": _f(g.acc_bn.mean()),
                          "bn_ctrl": _f(g.acc_bn_oo.mean()), "ft": _f(g.acc_ft_pp.mean()), "ctrl": _f(g.acc_ft_oo.mean()),
                          "probe": _f(g.acc_probe_pp.mean())}
                         for (t, dose), g in ft.groupby(["technique", "dose"])]
    if "imagenette" in d and d["imagenette"]["adapt"] is not None:
        a = d["imagenette"]["adapt"]
        ad["imagenette"] = [{"t": t, "dose": _f(dose), "frozen": _f(g.acc_frozen.mean()), "ft": _f(g.acc_ft_pp.mean()),
                             "ctrl": _f(g.acc_ft_oo.mean()), "frozen_full": _f(g.acc_frozen_full.mean()),
                             "orig_full": _f(g.acc_orig_full.mean())}
                            for (t, dose), g in a.groupby(["technique", "dose"])]
    out["adaptation"] = ad
    return out


def run_report(cfg: dict, samples: bool = True) -> Path:
    root = Path(cfg.get("report", {}).get("out", "docs"))
    results = Path(cfg["paths"]["results"])
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    (root / "assets").mkdir(parents=True, exist_ok=True)
    d = load(results)
    if not d:
        raise FileNotFoundError("nenhuma análise encontrada em results/sweep*/analysis")
    for fn in (fig_dose_curves, fig_bands, fig_windows, fig_hormesis, fig_predictors, fig_ire, fig_adaptation,
               fig_cnn_vit, fig_jpeg, fig_tradeoff):
        fn(d, figs)
        print(f"[report] {fn.__name__}")
    man_path = root / "samples" / "manifest.json"
    if samples:
        manifest = make_samples(cfg, root / "samples")
        man_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    if manifest:
        fig_samples_strip(root / "samples", manifest, d, figs)
    data = site_data(d, manifest, cfg)
    (root / "assets" / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                                               encoding="utf-8")
    print(f"[report] figuras em {figs}; dados do site em {root / 'assets' / 'data.json'}")
    return root
