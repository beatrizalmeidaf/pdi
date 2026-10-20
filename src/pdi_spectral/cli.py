"""Interface de linha de comando.

    python -m pdi_spectral <comando> [--config configs/default.yaml] [--set chave=valor ...]

Comandos:
  download   baixa CIFAR-10 (torchvision) e pesos dos modelos (aquecimento de cache)
  verify     E0: compara técnicas do zero x referência (OpenCV/PIL/SciPy)
  examples   figuras qualitativas + assinatura espectral média por técnica
  oracle     E1: calibração com filtros espectrais ideais
  main       E2: experimento principal dose-resposta (inferência congelada)
  adapt      E4: domain shift x perda de informação (BN-adapt, sondas, fine-tuning opcional)
  analyze    E3: estatística, previsibilidade, figuras e relatório
  all        oracle + main + adapt + analyze
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pdi_spectral", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "verify", "examples", "oracle", "main", "adapt", "analyze", "all", "show-config"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="sobrescreve chave do YAML, ex.: --set backend=reference")
    ap.add_argument("--datasets", nargs="*", help="restringe aos datasets indicados")
    ap.add_argument("--verify-dataset", default=None, help="verify: usa imagens reais deste dataset (padrão: sintéticas)")
    ap.add_argument("--n", type=int, default=6, help="verify/examples: número de imagens")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)

    if a.command == "show-config":
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
    elif a.command == "download":
        from .datasets import load_dataset
        from .models import build_module

        for ds, d in cfg["datasets"].items():
            if ds.startswith("cifar"):
                load_dataset(ds, "val", 1, cfg["seed"], cfg["paths"])
                load_dataset(ds, "train", 1, cfg["seed"], cfg["paths"])
            for m in d["models"]:
                print(f"[download] pesos {m}")
                build_module(m)
    elif a.command == "verify":
        from .verify import run_verify

        run_verify(cfg, a.verify_dataset, a.n)
    elif a.command == "examples":
        from .plots import examples

        out = Path(cfg["paths"]["results"]) / cfg["run_name"] / "analysis" / "figures"
        for ds in a.datasets or list(cfg["datasets"]):
            examples(cfg, ds, a.n, out)
            print(f"[examples] {ds} -> {out}")
    else:
        from .adaptation import run_adaptation
        from .analysis import run_analysis
        from .pipeline import run_inference

        steps = ["oracle", "main", "adapt", "analyze"] if a.command == "all" else [a.command]
        for s in steps:
            if s in ("oracle", "main"):
                run_inference(cfg, s, a.datasets)
            elif s == "adapt":
                run_adaptation(cfg, a.datasets)
            elif s == "analyze":
                run_analysis(cfg)


if __name__ == "__main__":
    main()
