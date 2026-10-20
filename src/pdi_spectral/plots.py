"""Figuras (matplotlib, sem interface gráfica)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 9})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def dose_response(ct: pd.DataFrame, out: Path):
    for ds, d in ct.groupby("dataset"):
        techs = sorted(d.technique.unique())
        nc = 4
        nr = int(np.ceil(len(techs) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.5 * nr), squeeze=False)
        for ax, t in zip(axes.flat, techs):
            for model, g in d[d.technique == t].groupby("model"):
                g = g.sort_values("level")
                ax.errorbar(g.dose, 100 * g.delta_acc, yerr=196 * g.delta_se, marker="o", ms=3, capsize=2, label=model)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(t)
            ax.set_xlabel("parâmetro nominal")
            ax.set_ylabel("Δacc (p.p.)")
        for ax in list(axes.flat)[len(techs):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"Dose-resposta — {ds}")
        _save(fig, out / f"dose_response_{ds}.png")


def delta_vs_energy(ct: pd.DataFrame, out: Path, xcol: str = "dErel_high"):
    if xcol not in ct:
        return
    for ds, d in ct.groupby("dataset"):
        models = sorted(d.model.unique())
        fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.2), squeeze=False, sharey=True)
        techs = sorted(d.technique.unique())
        cmap = plt.get_cmap("tab20", len(techs))
        for ax, m in zip(axes.flat, models):
            g = d[d.model == m]
            for i, t in enumerate(techs):
                h = g[g.technique == t].sort_values("level")
                mk = "o" if h.group.iloc[0] == "primary" else "s" if len(h) else "o"
                ax.plot(h[xcol], 100 * h.delta_acc, marker=mk, ms=4, lw=0.8, color=cmap(i), label=t)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(m)
            ax.set_xlabel(xcol)
        axes.flat[0].set_ylabel("Δacc (p.p.)")
        axes.flat[-1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.suptitle(f"Queda de acurácia × {xcol} — {ds}  (círculo = primário, quadrado = controle)")
        _save(fig, out / f"delta_vs_{xcol}_{ds}.png")


def band_sensitivity(ct_or: pd.DataFrame, main_dir: Path, out: Path):
    b = ct_or[ct_or.technique == "oracle_band"].copy()
    if b.empty:
        return
    b["gain"] = b.cond_id.str.extract(r"_g([0-9.]+)@")[0].astype(float)
    for ds, d in b.groupby("dataset"):
        gains = sorted(d.gain.unique())
        fig, axes = plt.subplots(1, len(gains), figsize=(4.2 * len(gains), 3.2), squeeze=False)
        for ax, gval in zip(axes.flat, gains):
            for m, g in d[d.gain == gval].groupby("model"):
                g = g.sort_values("dose")
                ax.plot(g.dose, 100 * g.delta_acc, marker="o", ms=3, label=m)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel("centro da banda (ciclos/pixel)")
            ax.set_ylabel("Δacc (p.p.)")
            ax.set_title("banda removida" if gval == 0 else f"banda × {gval:g}")
            spec = main_dir / ds / "dataset_spectrum.csv"
            if not spec.exists():
                spec = main_dir.parent / "oracle" / ds / "dataset_spectrum.csv"
            if spec.exists():
                s = pd.read_csv(spec)
                tw = ax.twinx()
                tw.semilogy(s.r_center, s.energy_mean, color="gray", ls="--", lw=0.8)
                tw.set_ylabel("energia média (dataset)", color="gray")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"Sensibilidade por banda (oráculo) — {ds}")
        _save(fig, out / f"band_sensitivity_{ds}.png")


def loto_bars(res: pd.DataFrame, out: Path):
    for ds, d in res.groupby("dataset"):
        s = d.groupby("features")["r2"].agg(["mean", "min", "max"]).sort_values("mean")
        fig, ax = plt.subplots(figsize=(5, 0.35 * len(s) + 1))
        ax.barh(s.index, s["mean"], xerr=[s["mean"] - s["min"], s["max"] - s["mean"]], capsize=2, color="#4c72b0")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("R² leave-one-technique-out (média; barras = min/max entre modelos)")
        ax.set_title(f"Preditores da queda de acurácia — {ds}")
        _save(fig, out / f"loto_r2_{ds}.png")


def pred_vs_obs(lp: pd.DataFrame, out: Path, label: str):
    for ds, d in lp.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(4, 4))
        for t, g in d.groupby("technique"):
            ax.scatter(100 * g.delta_acc, 100 * g.pred, s=10, label=t)
        lim = [100 * min(d.delta_acc.min(), d.pred.min()), 100 * max(d.delta_acc.max(), d.pred.max(), 0)]
        ax.plot(lim, lim, "k--", lw=0.6)
        ax.set_xlabel("Δacc observado (p.p.)")
        ax.set_ylabel("Δacc previsto LOTO (p.p.)")
        ax.set_title(f"{ds} — preditores: {label}")
        ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        _save(fig, out / f"pred_vs_obs_{ds}.png")


def adaptation_bars(ad: pd.DataFrame, out: Path):
    for ds, d in ad.groupby("dataset"):
        s = d.groupby("cond_id")[["acc_probe_op", "acc_probe_pp", "acc_probe_oo"]].mean()
        s = s.sort_values("acc_probe_op")
        fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(s)), 3.2))
        x = np.arange(len(s))
        ax.bar(x - 0.27, s.acc_probe_op, 0.27, label="sonda orig → proc (congelado)")
        ax.bar(x, s.acc_probe_pp, 0.27, label="sonda proc → proc (adaptado)")
        ax.bar(x + 0.27, s.acc_probe_oo, 0.27, label="sonda orig → orig (teto)")
        ax.set_xticks(x, s.index, rotation=60, ha="right")
        ax.set_ylabel("acurácia")
        ax.legend(fontsize=7)
        ax.set_title(f"Domain shift × perda de informação — {ds}")
        _save(fig, out / f"adaptation_{ds}.png")


def examples(cfg: dict, dataset: str, n_images: int, out: Path):
    """Figura qualitativa: cada técnica no nível do meio + perfil espectral radial."""
    from . import techniques as T
    from .datasets import load_dataset
    from .pipeline import build_conditions, filter_available
    from .spectral import SpectralAnalyzer, SpectralConfig

    imgs = load_dataset(dataset, "val", n_images, cfg["seed"], cfg["paths"]).load_all()
    conds = filter_available(build_conditions(cfg, "main"))
    mids = {}
    for c in conds:
        if c.level == c.n_levels // 2:
            mids[c.technique] = c
    an = SpectralAnalyzer(SpectralConfig.from_dict(cfg.get("spectral")), imgs.shape[1:3])
    cols = ["original"] + list(mids)
    fig, axes = plt.subplots(len(imgs), len(cols), figsize=(1.6 * len(cols), 1.7 * len(imgs)), squeeze=False)
    procs = {}
    for j, name in enumerate(cols):
        if name == "original":
            batch = imgs
        else:
            c = mids[name]
            batch = np.stack([T.apply(c.technique, im, c.params, cfg.get("backend", "scratch")) for im in imgs])
        procs[name] = batch
        for i in range(len(imgs)):
            axes[i, j].imshow(batch[i])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[0, j].set_title(name if name == "original" else f"{name}\n{mids[name].public_params}", fontsize=6)
    _save(fig, out / f"examples_{dataset}.png")

    Fx, _ = an.transform(imgs)
    px = an.profile(Fx).mean(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, b in procs.items():
        if name == "original":
            continue
        py = an.profile(an.transform(b)[0]).mean(0)
        ax.plot(an.fine_centers, 10 * np.log10((py + 1e-12) / (px + 1e-12)), marker=".", label=name)
    ax.axhline(0, color="k", lw=0.6)
    for lo, hi in an.cfg.bands.values():
        ax.axvline(lo, color="gray", ls=":", lw=0.6)
    ax.set_xlabel("frequência radial (ciclos/pixel)")
    ax.set_ylabel("ΔE por anel (dB)")
    ax.set_title(f"Assinatura espectral média (nível do meio) — {dataset}")
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, out / f"spectral_signature_{dataset}.png")
