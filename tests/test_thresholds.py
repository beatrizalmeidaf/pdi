"""Limiares (BMD) e Índice de Risco Espectral."""
import numpy as np
import pandas as pd

from pdi_spectral.thresholds import budget, compute_thresholds, crossing, fit_ire, ire_prob


def test_crossing_interpolates_and_flags():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    d, s = crossing(x, np.array([[0.0, -0.005, -0.02, -0.05]]), -0.01, "below")
    assert s[0] == 0 and np.isclose(d[0], 1.0 + (0.005 / 0.015))
    d, s = crossing(x, np.array([[0.0, 0.0, 0.0, 0.0]]), -0.01, "below")
    assert s[0] == 2 and np.isnan(d[0])          # nunca cruza: censurado
    d, s = crossing(x[1:], np.array([[-0.5, -0.6, -0.7]]), -0.01, "below")
    assert s[0] == 1 and d[0] == 1.0             # já cruza na 1a dose testada


def _fake_run(n=400, seed=0):
    """Técnica sintética: Δacc = -0.04 * sigma (cruza 1 p.p. em sigma = 0.25)."""
    rng = np.random.default_rng(seed)
    rows, spec = [], []
    ids = [f"i{k}" for k in range(n)]
    for i in ids:
        rows.append(dict(dataset="d", model="m", family="cnn", image_id=i, cond_id="original", technique="original",
                         level=-1, group="baseline", dose=0.0, correct=1, flip=0))
    for lv, sigma in enumerate([0.1, 0.2, 0.3, 0.5, 1.0]):
        wrong = rng.random(n) < 0.04 * sigma
        for i, w in zip(ids, wrong):
            rows.append(dict(dataset="d", model="m", family="cnn", image_id=i, cond_id=f"g@{lv}", technique="g",
                             level=lv, group="control", dose=sigma, correct=int(not w), flip=int(w)))
            spec.append(dict(dataset="d", image_id=i, cond_id=f"g@{lv}", dq_noise=-0.3 * sigma))
    return pd.DataFrame(rows), pd.DataFrame(spec)


def test_bmd_recovers_known_threshold():
    pred, spec = _fake_run(n=4000)
    cfg = {"seed": 0, "techniques": {"g": {"dose": "sigma", "identity": 0.0,
                                           "objective": {"metric": "noise", "direction": "down", "min_gain": 0.03}}},
           "thresholds": {"acc_tol": [0.01], "flip_tol": [0.01], "bootstrap": 300, "ci": 0.9}}
    thr, curves = compute_thresholds(pred, spec, cfg)
    r = thr.iloc[0]
    assert 0.15 < r["bmd_acc0.01"] < 0.4
    assert r["bmdl_acc0.01"] <= r["bmd_acc0.01"] <= r["bmdu_acc0.01"]
    assert np.isclose(r["perc_dose"], 0.1)        # -3% de ruído já em sigma = 0.1
    assert r["window_exists"]                      # objetivo atingido antes do dano
    assert curves.x.iloc[0] == 0.0 and curves.d_acc.iloc[0] == 0.0  # âncora na identidade


def test_ire_recovers_weights_and_budget():
    rng = np.random.default_rng(1)
    X = rng.exponential(0.02, size=(60000, 3))
    w_true = np.array([0.5, 4.0, 20.0, 1.0])     # último = gamma; banda alta muito mais perigosa
    y = (rng.random(len(X)) < ire_prob(X, w_true)).astype(float)
    th = fit_ire(X, y, l2=0.0, gamma=1.0)
    assert np.allclose(th[:-1], w_true[:-1], rtol=0.1, atol=0.3)  # peso pequeno é o mais ruidoso
    th_free = fit_ire(X, y, l2=0.0)
    assert abs(th_free[-1] - 1.0) < 0.15        # expoente estimado
    assert np.isclose(budget(0.01), -np.log(0.99))


def test_ire_features_include_coherence_and_chroma():
    import pandas as pd
    from pdi_spectral.thresholds import ire_features, ire_matrix

    df = pd.DataFrame({"D_f00": [0.1], "D_f01": [0.2], "Dc_f00": [0.3], "Dc_dc": [0.0], "coh_low": [0.9], "coh_high": [1.2]})
    f = ire_features(df, "D_fine+chroma+coh")
    assert f == ["D_f00", "D_f01", "Dc_f00", "Dc_dc", "incoh_low", "incoh_high"]
    X = ire_matrix(df, f)
    assert np.allclose(X[0, -2:], [0.1, 0.0])    # 1 - coh, limitado a >= 0
