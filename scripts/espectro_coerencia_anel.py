"""Recalcula só a coerência espectral por anel (17 anéis) para as condições de uma rodada, nas mesmas imagens.

    python scripts/espectro_coerencia_anel.py --config configs/pub_imagenet.yaml --kind main --n-images 10000 --jobs 48

Reaplica cada técnica (ou filtro) às imagens e grava results/<run>/<kind>/<ds>/coerencia_anel.parquet com
image_id, cond_id e cohf_f00..f16. Não roda nenhuma rede: as decisões já estão em predictions.parquet.
--n-images: subconjunto sorteado pela semente da config (o mesmo dos filtros suaves/parciais com 10 mil imagens).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _work(args):
    cfg_path, overrides, kind, n_images, k, n = args
    from pdi_spectral.config import load_config
    from pdi_spectral.datasets import load_dataset
    from pdi_spectral.pipeline import apply_batch, build_conditions, filter_available
    from pdi_spectral.spectral import SpectralAnalyzer, SpectralConfig

    cfg = load_config(cfg_path, overrides)
    ds_name, dcfg = next(iter(cfg["datasets"].items()))
    imgset = load_dataset(ds_name, "val", n_images or dcfg.get("n_images"), cfg["seed"], cfg["paths"])
    imgset = imgset.take(list(range(k, len(imgset), n)))
    conds = filter_available(build_conditions(cfg, kind))
    an = SpectralAnalyzer(SpectralConfig.from_dict(cfg.get("spectral")), imgset.image_size)
    parts = []
    for ids, imgs, _, _ in imgset.batches(32):
        Fx, _ = an.transform(imgs)
        for c in conds:
            Fy, _ = an.transform(apply_batch(c, imgs, cfg.get("backend", "scratch"), None))
            coh = an.coherence_fine(Fx, Fy)
            df = pd.DataFrame(coh, columns=[f"cohf_f{i:02d}" for i in range(coh.shape[1])])
            df.insert(0, "image_id", ids)
            df.insert(1, "cond_id", c.cond_id)
            parts.append(df)
    return pd.concat(parts, ignore_index=True), cfg["paths"]["results"], cfg["run_name"], ds_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--kind", default="main", choices=["main", "oracle"])
    ap.add_argument("--n-images", type=int)
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    tasks = [(a.config, a.set, a.kind, a.n_images, k, a.jobs) for k in range(a.jobs)]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_work, tasks))
    df = pd.concat([r[0] for r in res], ignore_index=True)
    _, root, run, ds = res[0]
    out = Path(root) / run / a.kind / ds / "coerencia_anel.parquet"
    df.to_parquet(out)
    print(f"[coerência por anel] {len(df)} linhas -> {out}")


if __name__ == "__main__":
    main()
