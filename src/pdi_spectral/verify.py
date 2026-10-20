"""E0 — Validação das implementações do zero contra OpenCV/PIL/SciPy."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import techniques as T
from .techniques._core import psnr


def synthetic_images(n: int = 4, size: int = 64, seed: int = 0) -> np.ndarray:
    """Imagens 'naturais' sintéticas: espectro ~1/f, bordas e textura."""
    rng = np.random.default_rng(seed)
    out = []
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1.0
    for _ in range(n):
        chans = []
        for _c in range(3):
            ph = np.exp(2j * np.pi * rng.random((size, size)))
            x = np.real(np.fft.ifft2(ph / r ** 1.2))
            chans.append(x)
        x = np.stack(chans, -1)
        x = (x - x.min()) / (x.max() - x.min())
        yy, xx = np.mgrid[:size, :size]
        x[(xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size / 4) ** 2] *= 0.6
        out.append(np.clip(x * 255, 0, 255).astype(np.uint8))
    return np.stack(out)


def compare(imgs: np.ndarray, cfg: dict, sr_dir: str = "data/sr_models") -> pd.DataFrame:
    rows = []
    for name, tc in cfg["techniques"].items():
        if not tc.get("enabled", True):
            continue
        for i, lv in enumerate(tc["levels"]):
            params = dict(lv)
            if name in T.SR_TECHNIQUES:
                params["_model_dir"] = sr_dir
            try:
                t0 = time.perf_counter()
                a = np.stack([T.apply(name, im, params, "scratch") for im in imgs])
                t1 = time.perf_counter()
                b = np.stack([T.apply(name, im, params, "reference") for im in imgs])
                t2 = time.perf_counter()
            except FileNotFoundError as e:
                rows.append(dict(technique=name, level=i, params=str(lv), status=f"pulado: {e}"))
                continue
            d = np.abs(a.astype(int) - b.astype(int))
            rows.append(dict(technique=name, level=i, params=str(lv), mean_abs_diff=d.mean(), max_abs_diff=int(d.max()),
                             pct_pixels_gt2=float((d > 2).mean() * 100), psnr_scratch_vs_ref=psnr(a, b),
                             psnr_scratch_vs_orig=psnr(a, imgs), psnr_ref_vs_orig=psnr(b, imgs),
                             ms_scratch=1000 * (t1 - t0) / len(imgs), ms_ref=1000 * (t2 - t1) / len(imgs), status="ok"))
    return pd.DataFrame(rows)


def run_verify(cfg: dict, dataset: str | None = None, n: int = 8) -> Path:
    if dataset:
        from .datasets import load_dataset

        imgs = load_dataset(dataset, "val", n, cfg["seed"], cfg["paths"]).load_all()
    else:
        imgs = synthetic_images(n, 96)
    df = compare(imgs, cfg, cfg["paths"].get("sr_models", "data/sr_models"))
    out = Path(cfg["paths"]["results"]) / "verification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 200):
        print(df.round(3).to_string(index=False))
    print(f"[verify] salvo em {out}")
    return out
