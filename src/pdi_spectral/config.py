"""Leitura de configuração YAML com herança (``extends``) e sobrescritas ``--set a.b=valor``."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not v.get("_replace", False):
            out[k] = deep_merge(out[k], v)
        else:
            if isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk != "_replace"}
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "extends" in cfg:
        parent = load_config(path.parent / cfg.pop("extends"))
        cfg = deep_merge(parent, cfg)
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
