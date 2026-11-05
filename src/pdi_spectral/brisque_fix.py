"""Correção do resumo do BRISQUE: razão das médias em vez de média das razões.

Por que: o BRISQUE pode ficar perto de zero ou negativo em imagens nítidas (no Imagenette,
~10 % das imagens originais têm BRISQUE < 5 e ~3 % < 0). A variação relativa por imagem,
(q - q0) / |q0|, explode nessas imagens e domina a média: no laplaciano leve a média das razões
diz "+43 % (pior)" e a mediana diz "-18 % (melhor)". A medida robusta é a variação da média,
mean(q) / mean(q0) - 1, que é a que os limiares passam a usar.

Execuções novas já salvam ``q_brisque_orig`` por imagem (quality.compare). Para as antigas,
``brisque_originals`` calcula o BRISQUE das imagens originais (as mesmas do run, pelo id) e
salva em ``results/<run>/main/<ds>/brisque_orig.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def brisque_originals(cfg: dict, ds: str, run_dir: Path, batch: int = 64) -> pd.DataFrame:
    from .datasets import load_dataset
    from .quality import _brisque

    out = run_dir / "main" / ds / "brisque_orig.csv"
    if out.exists():
        return pd.read_csv(out)
    n = cfg["datasets"][ds].get("n_images")
    imgs = load_dataset(ds, "val", n, cfg["seed"], cfg["paths"])
    vals = []
    for s in range(0, len(imgs), batch):
        x = np.stack([imgs.loader(i) for i in range(s, min(s + batch, len(imgs)))])
        vals.append(_brisque(x))
    df = pd.DataFrame({"image_id": imgs.ids, "q_brisque_orig": np.concatenate(vals)})
    df.to_csv(out, index=False)
    return df


def attach(spec: pd.DataFrame, orig: pd.DataFrame | None) -> pd.DataFrame:
    """Garante a coluna q_brisque_orig no parquet espectral (por image_id)."""
    if "q_brisque_orig" in spec or orig is None or "q_brisque" not in spec:
        return spec
    return spec.merge(orig, on="image_id", how="left")


def ratio_of_means(spec: pd.DataFrame) -> pd.DataFrame:
    """dq_brisque robusto por condição: mean(q) / mean(q0) - 1."""
    spec = spec.dropna(subset=["q_brisque", "q_brisque_orig"])  # BRISQUE indefinido em imagens degeneradas
    g = spec.groupby(["dataset", "cond_id"]).agg(q=("q_brisque", "mean"), q0=("q_brisque_orig", "mean")).reset_index()
    g["dq_brisque"] = g.q / g.q0 - 1
    g["d_brisque"] = g.q - g.q0
    return g[["dataset", "cond_id", "dq_brisque", "d_brisque"]]
