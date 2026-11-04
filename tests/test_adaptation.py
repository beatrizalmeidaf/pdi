"""E4: sondas lineares, fração recuperável, BN-adapt e fine-tuning."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if any(importlib.util.find_spec(m) is None for m in ("torch", "torchvision", "sklearn")):
    pytest.skip("requer torch, torchvision e scikit-learn", allow_module_level=True)

from pdi_spectral.adaptation import (_acc_logits, _acc_probe, _finetune, _probe, recoverable_fraction,
                                     run_adaptation)

ROOT = Path(__file__).resolve().parents[1]


def test_acc_logits_restricted_to_subset():
    logits = np.array([[0.0, 9.0, 1.0, 2.0], [5.0, 0.0, 0.0, 1.0]])
    assert _acc_logits(logits, np.array([1, 0]), None) == 1.0
    # restrito às classes {2, 3}: a classe 1 (maior logit) é ignorada
    assert _acc_logits(logits, np.array([3, 3]), [2, 3]) == 1.0


def test_recoverable_fraction():
    assert recoverable_fraction(0.9, 0.5, 0.9) == 1.0
    assert recoverable_fraction(0.9, 0.5, 0.5) == 0.0
    assert np.isnan(recoverable_fraction(0.9, 0.9, 0.9))


CENTERS = np.random.default_rng(42).normal(0, 3, (4, 16))


def _features(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    return CENTERS[y] + rng.normal(0, 1, (n, 16)), y


def test_probes_separate_domain_shift_from_information_loss():
    x_tr, y_tr = _features(800, 0)
    x_te, y_te = _features(400, 1)
    oo = _acc_probe(_probe(x_tr, y_tr, x_te, 1.0, 0), x_te, y_te)
    assert oo > 0.95

    rng = np.random.default_rng(1)
    perm = rng.permutation(16)
    shift = lambda x: 3 * x[:, perm] + 5           # invertível: informação intacta
    noise = lambda x: rng.normal(0, 1, x.shape)    # destrói a informação
    for f, expect in ((shift, 1.0), (noise, 0.0)):
        p_tr, p_te = f(x_tr), f(x_te)
        op = _acc_probe(_probe(x_tr, y_tr, x_te, 1.0, 0), p_te, y_te)
        pp = _acc_probe(_probe(p_tr, y_tr, p_te, 1.0, 0), p_te, y_te)
        assert abs(recoverable_fraction(oo, op, pp) - expect) < 0.15


def _tiny(seed=0):
    from pdi_spectral.models import Classifier

    return Classifier("tiny_cnn", "cpu")


def _imgs(n, seed):
    from pdi_spectral.verify import synthetic_images

    return synthetic_images(n, 32, seed=seed)


def test_bn_adapted_copies_and_reestimates():
    import torch

    clf = _tiny()
    bn = [m for m in clf.net.modules() if isinstance(m, torch.nn.BatchNorm2d)][0]
    before = bn.running_mean.clone()
    x = _imgs(16, 0)
    ad = clf.bn_adapted(x)
    assert torch.equal(bn.running_mean, before)   # original intacto
    bn_ad = [m for m in ad.net.modules() if isinstance(m, torch.nn.BatchNorm2d)][0]
    assert not bn_ad.training
    # running_mean = média por canal da saída da conv nas imagens dadas
    with torch.no_grad():
        conv_out = ad.net[0](ad._tensor(x))
    assert torch.allclose(bn_ad.running_mean, conv_out.mean(dim=(0, 2, 3)), atol=1e-5)
    # o hook de features da cópia não escreve no original
    clf._feat = None
    assert ad.features(x).shape == (16, 128) and clf._feat is None


def test_finetune_learns_without_touching_original():
    import torch

    clf = _tiny()
    w0 = clf.net[-1].weight.detach().clone()
    x = _imgs(32, 1)
    y = np.random.default_rng(0).integers(0, 10, 32)
    acc0 = _acc_logits(clf.logits(x), y, None)
    ft = _finetune(clf, x, y, epochs=30, lr=0.05, bs=16, seed=0)
    assert torch.equal(clf.net[-1].weight, w0)
    assert not ft.net.training
    assert _acc_logits(ft.logits(x), y, None) > max(acc0, 0.5)  # decora o treino


def test_run_adaptation_offline(tmp_path):
    from pdi_spectral.config import load_config

    cfg = load_config(ROOT / "configs" / "offline.yaml",
                      [f"paths.results={tmp_path.as_posix()}", "n_workers=0", "device=cpu",
                       "datasets.synthetic.n_images=24", "datasets.synthetic.n_train=48",
                       "adaptation.conditions=[gaussian@1, jpeg@1]", "adaptation.finetune_epochs=1"])
    run_adaptation(cfg)
    df = pd.read_csv(tmp_path / "offline" / "adaptation" / "adaptation.csv")
    assert sorted(df.cond_id) == ["gaussian@1", "jpeg@1"]
    for c in ("acc_frozen", "acc_bn", "acc_bn_oo", "acc_probe_pp", "acc_ft_pp", "acc_ft_oo"):
        assert df[c].between(0, 1).all(), c
    assert df.acc_probe_oo.nunique() == 1 and df.acc_ft_oo.nunique() == 1   # linha de base comum
    assert np.allclose(df.info_loss, df.acc_probe_oo - df.acc_probe_pp)
    assert "acc_frozen_full" not in df  # sem subconjunto de classes: não há coluna *_full


def test_run_adaptation_class_subset_adds_full_accuracy(tmp_path, monkeypatch):
    import dataclasses

    import pdi_spectral.adaptation as A
    from pdi_spectral.config import load_config

    load = A.load_dataset
    # subconjunto = todas as 10 saídas da CNN minúscula: medir nas K classes ou em todas dá o mesmo
    monkeypatch.setattr(A, "load_dataset", lambda *a, **k: dataclasses.replace(load(*a, **k), class_subset=list(range(10))))
    cfg = load_config(ROOT / "configs" / "offline.yaml",
                      [f"paths.results={tmp_path.as_posix()}", "n_workers=0", "device=cpu",
                       "datasets.synthetic.n_images=24", "datasets.synthetic.n_train=48",
                       "adaptation.conditions=[gaussian@1]"])
    run_adaptation(cfg)
    df = pd.read_csv(tmp_path / "offline" / "adaptation" / "adaptation.csv")
    for k in ("orig", "frozen", "bn", "bn_oo"):
        assert np.allclose(df[f"acc_{k}_full"], df[f"acc_{k}"]), k
