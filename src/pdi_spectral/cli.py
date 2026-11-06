"""Interface de linha de comando.

    python -m pdi_spectral <comando> [--config configs/default.yaml] [--set chave=valor ...]

Comandos:
  download   baixa CIFAR-10 (torchvision) e pesos dos modelos (aquecimento de cache)
  verify     E0: compara técnicas do zero x referência (OpenCV/PIL/SciPy)
  examples   figuras qualitativas + assinatura espectral média por técnica
  oracle     E1: calibração com filtros espectrais ideais
  main       E2: experimento principal dose-resposta (inferência congelada)
  adapt      E4: domain shift x perda de informação (BN-adapt, sondas, fine-tuning opcional)
  analyze    E3: estatística, previsibilidade, limiares (BMD), IRE, figuras e relatório
  safe-dose  dose segura de uma técnica pela fórmula IRE, sem rodar a rede
  human-export   gera o estudo com pessoas (pasta com index.html local, escolha entre 2 imagens)
  human-analyze  analisa as respostas: dose em que as pessoas percebem melhora/piora
  report     figuras do README + dados e amostras do site (docs/), a partir das análises já rodadas
  merge-shards  junta as fatias de oracle/main rodadas com --set shard=k/n
  all        oracle + main + adapt + analyze
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pdi_spectral", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "verify", "examples", "oracle", "main", "adapt", "analyze", "all", "show-config", "safe-dose", "human-export", "human-analyze", "report", "merge-shards"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="sobrescreve chave do YAML, ex.: --set backend=reference")
    ap.add_argument("--datasets", nargs="*", help="restringe aos datasets indicados")
    ap.add_argument("--verify-dataset", default=None, help="verify: usa imagens reais deste dataset (padrão: sintéticas)")
    ap.add_argument("--n", type=int, default=6, help="verify/examples: número de imagens")
    ap.add_argument("--technique", help="safe-dose: técnica")
    ap.add_argument("--model", help="safe-dose: modelo (precisa de ire_weights.csv do analyze)")
    ap.add_argument("--dataset", help="safe-dose: dataset")
    ap.add_argument("--grid", type=float, nargs="+", help="safe-dose: doses a testar")
    ap.add_argument("--tau", type=float, default=0.01, help="safe-dose: flip/queda máxima tolerada")
    ap.add_argument("--margin", type=float, help="safe-dose: fator conformal c (aceita flip previsto <= c·tau); "
                    "padrão: ire.safe_margin da config")
    ap.add_argument("--no-samples", action="store_true", help="report: reaproveita as amostras já geradas")
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
    elif a.command == "safe-dose":
        from .thresholds import safe_dose

        ds = a.dataset or next(iter(cfg["datasets"]))
        model = a.model or cfg["datasets"][ds]["models"][0]
        tc = cfg["techniques"][a.technique]
        grid = a.grid or [lv[tc.get("dose") or next(iter(lv))] for lv in tc["levels"]]
        df = safe_dose(cfg, a.technique, model, ds, grid, a.tau, n_images=a.n if a.n != 6 else 200, margin=a.margin)
        print(df.to_string(index=False))
        print(f"orçamento IRE para tau={a.tau} com margem c={df.attrs['margin']:g}: {df.attrs['budget']:.4f}  (sem rodar a rede)")
    elif a.command == "human-export":
        from .human import export_study

        ds = a.dataset or next(iter(cfg["datasets"]))
        export_study(cfg, ds, n_images=a.n if a.n != 6 else 12,
                     techniques=[a.technique] if a.technique else None)
    elif a.command == "merge-shards":
        from .pipeline import merge_shards

        for kind in ("oracle", "main"):
            merge_shards(cfg, kind, a.datasets)
    elif a.command == "report":
        from .report import run_report

        run_report(cfg, samples=not a.no_samples)
    elif a.command == "human-analyze":
        from .human import analyze_study

        analyze_study(cfg)
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
