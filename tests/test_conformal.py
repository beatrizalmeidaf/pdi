"""Margem conformal da dose segura (thresholds.margin_scores / conformal_margin)."""
import numpy as np
import pandas as pd

from pdi_spectral.thresholds import conformal_margin, margin_scores

CFG = {"techniques": {"blur": {"identity": 0.0}, "jpeg": {"reverse": 100}}}


def _curve(tech, doses, obs, pred, model="m"):
    return pd.DataFrame(dict(dataset="d", model=model, technique=tech, dose=doses, flip=obs, p_loto=pred))


def test_score_exato_quando_previsao_acerta():
    cp = _curve("blur", [1.0, 2.0], [0.005, 0.02], [0.005, 0.02])
    s = margin_scores(cp, CFG, tau=0.01).iloc[0]
    assert np.isclose(s.score, 1.0)


def test_score_maior_que_1_quando_subestima():
    cp = _curve("blur", [1.0, 2.0], [0.005, 0.02], [0.0025, 0.01])  # prevê metade do flip
    s = margin_scores(cp, CFG, tau=0.01).iloc[0]
    assert np.isclose(s.score, 2.0)


def test_jpeg_com_limiar_na_primeira_dose():
    # qualidade 100 já muda 2 % das decisões; o IRE previa 0,5 % ali
    cp = _curve("jpeg", [100, 90], [0.02, 0.05], [0.005, 0.03])
    s = margin_scores(cp, CFG, tau=0.01).iloc[0]
    assert np.isfinite(s.score) and np.isclose(s.score, 2.0)


def test_curva_que_nunca_cruza_tem_score_zero():
    cp = _curve("blur", [1.0, 2.0], [0.001, 0.002], [0.001, 0.002])
    assert margin_scores(cp, CFG, tau=0.01).iloc[0].score == 0.0


def test_conformal_margin_cobre_o_pior_caso_e_respeita_alpha():
    sc = pd.DataFrame(dict(technique=[f"t{i}" for i in range(19)], score=np.arange(1, 20) / 10))
    # n = 19, α = 0,1 -> k = ceil(20·0,9) = 18 -> 18º menor escore = 1,8 -> c = 1/1,8
    assert np.isclose(conformal_margin(sc, 0.1), 1 / 1.8)
    # poucas técnicas para o α pedido
    assert np.isnan(conformal_margin(sc.head(5), 0.1))
    # unidade = técnica: o pior modelo de cada técnica é que conta
    two = pd.DataFrame(dict(technique=["a", "a", "b"], score=[0.5, 3.0, 1.0]))
    assert np.isclose(conformal_margin(two, 0.5), 1 / 3.0)
