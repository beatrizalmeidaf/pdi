"""Resumo robusto do BRISQUE: razão das médias, não média das razões."""
import numpy as np
import pandas as pd

from pdi_spectral.brisque_fix import attach, ratio_of_means


def test_ratio_of_means_is_robust_to_near_zero_originals():
    # 3 imagens; uma tem BRISQUE original ~0. A técnica melhora todas em 2 pontos.
    q0 = np.array([30.0, 20.0, 0.01])
    spec = pd.DataFrame({"dataset": "d", "cond_id": "t@0", "image_id": ["a", "b", "c"], "q_brisque": q0 - 2})
    spec = attach(spec, pd.DataFrame({"image_id": ["a", "b", "c"], "q_brisque_orig": q0}))
    mean_of_ratios = ((spec.q_brisque - spec.q_brisque_orig) / spec.q_brisque_orig.abs()).mean()
    r = ratio_of_means(spec).iloc[0]
    assert mean_of_ratios < -50                      # a média das razões explode (−6700 %)
    assert np.isclose(r.dq_brisque, (q0.mean() - 2) / q0.mean() - 1)   # ≈ −12 %
    assert np.isclose(r.d_brisque, -2.0)
