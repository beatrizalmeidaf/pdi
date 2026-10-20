"""Pipeline de ponta a ponta, offline (dataset sintético + CNN minúscula)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# Checa sem importar: em alguns ambientes (ex.: Anaconda sem torchvision) importar torch derruba o processo.
if any(importlib.util.find_spec(m) is None for m in ("torch", "torchvision", "cv2")):
    pytest.skip("requer torch, torchvision e opencv", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]


def test_offline_pipeline(tmp_path):
    from pdi_spectral.adaptation import run_adaptation
    from pdi_spectral.analysis import run_analysis
    from pdi_spectral.config import load_config
    from pdi_spectral.pipeline import run_inference

    cfg = load_config(ROOT / "configs" / "offline.yaml",
                      [f"paths.results={tmp_path.as_posix()}", f"paths.sr_models={(ROOT / 'data' / 'sr_models').as_posix()}",
                       "n_workers=0", "device=cpu"])
    run_inference(cfg, "oracle")
    run_inference(cfg, "main")
    run_adaptation(cfg)
    out = run_analysis(cfg)

    pred = pd.read_parquet(tmp_path / "offline" / "main" / "synthetic" / "predictions.parquet")
    spec = pd.read_parquet(tmp_path / "offline" / "main" / "synthetic" / "spectral.parquet")
    n_cond = spec.cond_id.nunique()
    assert len(pred) == 48 * (n_cond + 1)  # + condição original
    assert (pred[pred.cond_id == "original"].flip == 0).all()
    assert (out / "summary.md").exists()
    assert (out / "loto.csv").exists()
    assert (tmp_path / "offline" / "adaptation" / "adaptation.csv").exists()
