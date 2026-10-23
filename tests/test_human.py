"""Kit do estudo com pessoas: exporta, simula participantes e analisa."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_export_and_analyze(tmp_path):
    from pdi_spectral.config import load_config
    from pdi_spectral.human import analyze_study, export_study

    cfg = load_config(ROOT / "configs" / "offline.yaml", [f"paths.results={tmp_path.as_posix()}",
                                                          "techniques.sr_espcn.enabled=false"])
    out = export_study(cfg, "synthetic", n_images=4, trials_per_participant=40, techniques=["gaussian", "unsharp"])
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "__PAIRS__" not in html and "chose_processed" in html
    pairs = pd.read_csv(out / "pairs.csv")
    assert (out / pairs.proc.iloc[0]).exists() and pairs.catch.any()

    # participantes simulados: preferem unsharp e rejeitam gaussian; um deles responde ao acaso
    rng = np.random.default_rng(0)
    for pid, attentive in (("p1", True), ("p2", True), ("p3", False)):
        rows = []
        for _ in range(8):
            for _, p in pairs.iterrows():
                if p.catch:
                    ch = 0 if attentive else int(rng.random() < 0.5)
                elif not attentive:
                    ch = int(rng.random() < 0.5)
                else:
                    ch = int(rng.random() < (0.85 if p.technique == "unsharp" else 0.15))
                rows.append(dict(participant=pid, technique=p.technique, cond_id=p.cond_id, dose=p.dose,
                                 catch=bool(p.catch), chose_processed=ch))
        pd.DataFrame(rows).to_csv(out / "respostas" / f"{pid}.csv", index=False)
    res = analyze_study(cfg).set_index("technique")
    assert np.isfinite(res.loc["unsharp", "human_better_dose"])
    assert np.isfinite(res.loc["gaussian", "human_worse_dose"])
