"""E3: tabela por condição, testes pareados e regressão leave-one-technique-out."""
import numpy as np
import pandas as pd

from pdi_spectral.analysis import _holm, _mcnemar_p, adaptation_summary, condition_table, feature_matrix, loto


def test_mcnemar_exact_values():
    # n = 10 discordantes, k = 1: p = 2 * P(X <= 1), X ~ Bin(10, 0.5) = 2 * 11/1024
    p = _mcnemar_p(np.array([9, 5, 0]), np.array([1, 5, 0]))
    assert np.isclose(p[0], 22 / 1024)
    assert np.isclose(p[1], 1.0)          # empate: limitado a 1
    assert p[2] == 1.0                    # sem discordantes


def test_holm_step_down_and_monotone():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = _holm(p)
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])
    order = np.argsort(p)
    assert (np.diff(adj[order]) >= 0).all()


def _toy_pred():
    """4 imagens, 1 modelo, condição g@0 perde 2 acertos e ganha 1."""
    base = dict(dataset="d", model="m", family="cnn", technique="g", level=0, group="control", dose=1.0,
                cond_id="g@0", correct_top5=1, kl=0.1, dp_true=0.0, conf=0.9)
    orig = [dict(base, technique="original", level=-1, group="baseline", dose=0.0, cond_id="original",
                 image_id=f"i{k}", correct=c, flip=0) for k, c in enumerate([1, 1, 1, 0])]
    proc = [dict(base, image_id=f"i{k}", correct=c, flip=f) for k, (c, f) in enumerate([(0, 1), (0, 1), (1, 0), (1, 1)])]
    spec = pd.DataFrame([dict(dataset="d", image_id=f"i{k}", cond_id="g@0", dErel_high=-0.1 * k) for k in range(4)])
    return pd.DataFrame(orig + proc), spec


def test_condition_table_counts_and_delta():
    pred, spec = _toy_pred()
    ct = condition_table(pred, spec)
    assert len(ct) == 1
    r = ct.iloc[0]
    assert r.n == 4 and r.lost == 2 and r.gained == 1
    assert np.isclose(r.acc_orig, 0.75) and np.isclose(r.acc, 0.5) and np.isclose(r.delta_acc, -0.25)
    assert np.isclose(r.flip, 0.75)
    assert r.ci95_lo < r.delta_acc < r.ci95_hi
    assert np.isclose(r.dErel_high, -0.15)  # média espectral por condição


def _toy_ct(seed=0):
    """Δacc depende só da energia removida em alta frequência, igual para todas as técnicas.
    O parâmetro nominal (dose_rank) tem escala diferente por técnica e não transfere."""
    rng = np.random.default_rng(seed)
    rows = []
    for t, scale in (("a", 0.05), ("b", 0.2), ("c", 0.5), ("d", 1.0)):
        for lv in range(5):
            hi = -scale * (lv + 1) / 5
            lo = rng.normal(0, 0.01)
            rows.append(dict(dataset="d", model="m", family="cnn", technique=t, cond_id=f"{t}@{lv}", level=lv,
                             dose_rank=(lv + 1) / 5, dErel_low=lo, dErel_mid=0.0, dErel_high=hi,
                             coh_low=1.0, coh_mid=1.0, coh_high=1.0 + hi / 2, psnr=30.0 + lv, ssim=1.0 - 0.01 * lv,
                             delta_acc=0.4 * hi + rng.normal(0, 0.002)))
    return pd.DataFrame(rows)


def test_feature_matrix_splits_sign():
    ct = _toy_ct()
    X = feature_matrix(ct, "spec_bands")
    assert list(X.columns) == ["dErel_low+", "dErel_low-", "dErel_mid+", "dErel_mid-", "dErel_high+", "dErel_high-"]
    assert (X["dErel_high+"] == 0).all() and np.allclose(X["dErel_high-"], ct.dErel_high)
    assert feature_matrix(ct, "spec_bands+coh").shape[1] == 9


def test_loto_spectral_beats_nominal_dose():
    res, preds = loto(_toy_ct(), ["dose_rank", "spec_bands"])
    r2 = res.set_index("features").r2
    assert r2["spec_bands"] > 0.95
    assert r2["dose_rank"] < 0.5
    assert preds.pred.notna().all() and len(preds) == 2 * 20


def test_adaptation_summary_averages_models():
    df = pd.DataFrame([dict(dataset="d", model=m, technique="g", level=0, cond_id="g@0", group="control",
                            acc_frozen=f, acc_probe_oo=0.9, acc_probe_op=0.5, acc_probe_pp=p, recoverable_frac=r)
                       for m, f, p, r in (("m1", 0.4, 0.9, 1.0), ("m2", 0.6, 0.5, 0.0))])
    s = adaptation_summary(df)
    assert len(s) == 1 and np.isclose(s.acc_frozen[0], 0.5) and np.isclose(s.recoverable_frac[0], 0.5)
    assert "acc_bn" not in s  # só colunas presentes
