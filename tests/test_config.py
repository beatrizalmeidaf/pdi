"""Herança de YAML (extends, _replace) e sobrescritas --set."""
from pathlib import Path

from pdi_spectral.config import deep_merge, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_deep_merge_and_replace():
    base = {"a": {"x": 1, "y": 2}, "b": {"x": 1}}
    out = deep_merge(base, {"a": {"y": 3}, "b": {"_replace": True, "z": 9}})
    assert out == {"a": {"x": 1, "y": 3}, "b": {"z": 9}}
    assert base["a"]["y"] == 2  # não altera a entrada


def test_extends_chain_and_overrides():
    cfg = load_config(ROOT / "configs" / "offline.yaml", ["seed=7", "adaptation.conditions=[gaussian@0]"])
    assert cfg["run_name"] == "offline"
    assert list(cfg["datasets"]) == ["synthetic"]                      # _replace do offline
    assert cfg["techniques"]["gaussian"]["levels"] == [{"sigma": 1.0}, {"sigma": 2.0}]  # do smoke
    assert "objective" in cfg["techniques"]["gaussian"]                # herdado do default
    assert cfg["seed"] == 7 and cfg["adaptation"]["conditions"] == ["gaussian@0"]
