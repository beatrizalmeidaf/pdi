"""Figuras e tabelas da rodada para publicação (ImageNet-val completo, 12 modelos, 3 sementes).

    python scripts/figuras_publicacao.py            # lê results/pub_*, escreve docs/figures e docs/resultados/pub

Gera:
  docs/figures/preditores_pub.png   R² LOTO do flip: IRE × LPIPS × Yin × features × PSNR/SSIM
  docs/figures/robustez_pub.png     queda média de acurácia por modelo (técnicas clássicas, sem SR)
  docs/resultados/pub/*.csv         tabelas agregadas entre sementes (cópia de results/pub_agregado)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pdi_spectral.report import C, _finish, _style, _title

R = Path("results")
FIG = Path("docs/figures")
TAB = Path("docs/resultados/pub")
DS = [("cifar10", "CIFAR-10 · 32 px"), ("imagenette", "Imagenette · 224 px"), ("imagenet_val", "ImageNet-val · 50 mil")]
DS_C = {"cifar10": C["s1"], "imagenette": C["s2"], "imagenet_val": C["s3"]}
PRED_LABEL = {
    "distância cosseno entre features (requer a rede)": "Distância entre features*",
    "distância L2 entre features (requer a rede)": "Distância L2 entre features*",
    "IRE": "IRE (este trabalho)",
    "ΔE por banda + coerência + cor (ridge)": "ΔE + coerência + cor (ridge)",
    "LPIPS (Zhang et al., 2018)": "LPIPS",
    "SSIM": "SSIM",
    "energia por banda (Yin et al., 2019)": "Energia por banda (Yin et al.)",
    "parâmetro nominal": "Parâmetro nominal",
    "PSNR": "PSNR",
}
MODEL = {"convnext_tiny": "ConvNeXt-T", "efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet-18",
         "resnet50": "ResNet-50", "swin_t": "Swin-T", "vgg16_bn": "VGG-16-BN", "vit_b_16": "ViT-B/16",
         "cifar10_mobilenetv2_x1_0": "MobileNetV2", "cifar10_repvgg_a0": "RepVGG-A0", "cifar10_resnet20": "ResNet-20",
         "cifar10_resnet56": "ResNet-56", "cifar10_vgg16_bn": "VGG-16-BN"}
TRANSFORMER = {"vit_b_16", "swin_t"}


def preditores(plt):
    p = pd.read_csv(R / "pub_agregado" / "predictor_comparison_agregado.csv")
    p = p[(p.target == "flip") & (p.predictor != "distância L2 entre features (requer a rede)")]
    t = p.groupby(["predictor", "dataset"]).r2_mean.mean().unstack()
    t = t.loc[t.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(t))
    h = 0.26
    for k, (ds, lab) in enumerate(DS):
        v = t[ds].clip(lower=0).values
        ax.barh(y + (k - 1) * h, v, h * 0.92, color=DS_C[ds], label=lab)
    for i, name in enumerate(t.index):
        if name == "IRE":
            ax.axhspan(i - 0.45, i + 0.45, color=C["neutral"], zorder=0)
    ax.set_yticks(y, [PRED_LABEL.get(n, n) for n in t.index])
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("IRE"):
            lbl.set_fontweight("semibold"), lbl.set_color(C["ink"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("R² prevendo técnicas que o preditor não viu (leave-one-technique-out)")
    ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _title(fig, "O IRE prevê o dano melhor que as métricas de imagem",
           "Probabilidade de a decisão mudar, por condição, média dos modelos. * precisa rodar a rede em cada imagem; "
           "o IRE e as demais métricas olham só a imagem.")
    _finish(fig)
    fig.savefig(FIG / "preditores_pub.png")
    plt.close(fig)


def robustez(plt):
    rows = []
    for ds, run in [("cifar10", [f"pub_cifar10_s{s}" for s in range(3)]),
                    ("imagenette", [f"pub_imagenette_s{s}" for s in range(3)]), ("imagenet_val", ["pub_imagenet"])]:
        for r in run:
            c = pd.read_csv(R / r / "analysis" / "conditions_main.csv")
            c = c[~c.technique.str.startswith("sr_")]
            rows.append(c.groupby("model").delta_acc.mean().reset_index().assign(dataset=ds))
    d = pd.concat(rows).groupby(["dataset", "model"]).agg(m=("delta_acc", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), gridspec_kw=dict(width_ratios=[5, 7, 7]))
    for ax, (ds, lab) in zip(axes, DS):
        s = d[d.dataset == ds].sort_values("m")
        cols = [C["s2"] if m in TRANSFORMER else C["s1"] for m in s.model]
        ax.barh(range(len(s)), -100 * s.m, color=cols, height=0.7)
        ax.set_yticks(range(len(s)), [MODEL[m] for m in s.model])
        for i, v in enumerate(-100 * s.m):
            ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=8.3, color=C["ink2"])
        ax.set_title(lab, loc="left")
        ax.grid(axis="x"), ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(-100 * d.m) * 1.15)
        ax.invert_yaxis()
    axes[1].set_xlabel("queda média de acurácia (p.p.) — menor é mais robusto")
    _title(fig, "Quem aguenta mais: ViT e ConvNeXt; quem sofre mais: VGG e ResNets",
           "Queda média sobre todas as doses das 11 técnicas clássicas (sem super-resolução). "
           "Laranja: transformers; azul: CNNs.")
    _finish(fig)
    fig.savefig(FIG / "robustez_pub.png")
    plt.close(fig)


def main():
    plt = _style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    preditores(plt)
    robustez(plt)
    for f in (R / "pub_agregado").glob("*"):
        shutil.copy(f, TAB / f.name)
    shutil.copy(R / "pub_imagenet" / "analysis" / "excluded_flat_images.csv", TAB / "excluded_flat_images_imagenet.csv")
    print(f"[pub] figuras em {FIG}; tabelas em {TAB}")


if __name__ == "__main__":
    main()
