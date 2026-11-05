"""Experimentos de inferência (E1 oráculo e E2 principal).

Para cada lote de imagens: as predições originais são calculadas uma vez; cada
condição (técnica x nível) é aplicada uma única vez e avaliada por TODOS os
modelos do dataset, garantindo que as comparações entre modelos usem
exatamente as mesmas imagens processadas.

Saída em ``results/<run_name>/<kind>/<dataset>/``:
  spectral.parquet     uma linha por (imagem, condição): ΔE por banda, PSNR, SSIM
  predictions.parquet  uma linha por (imagem, condição, modelo): acerto, flip, KL...
  conditions.csv       catálogo das condições
  dataset_spectrum.csv espectro radial médio das imagens originais
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import techniques as T
from .config import resolve_device
from .datasets import load_dataset
from .models import Classifier, log_softmax, softmax
from . import quality as Q
from .spectral import R_MAX, SpectralAnalyzer, SpectralConfig


@dataclass
class Condition:
    cond_id: str
    technique: str
    level: int
    n_levels: int
    params: dict
    group: str
    family: str
    dose: float

    @property
    def public_params(self) -> dict:
        return {k: v for k, v in self.params.items() if not k.startswith("_")}


ORIGINAL = Condition("original", "original", -1, 0, {}, "baseline", "baseline", 0.0)


def build_conditions(cfg: dict, kind: str) -> list[Condition]:
    conds: list[Condition] = []
    sr_dir = str(cfg["paths"].get("sr_models", "data/sr_models"))
    if kind == "main":
        for name, tc in cfg["techniques"].items():
            if not tc.get("enabled", True):
                continue
            t = T.TECHNIQUES[name]
            levels = tc["levels"]
            for i, lv in enumerate(levels):
                params = dict(lv)
                if name in T.SR_TECHNIQUES:
                    params["_model_dir"] = sr_dir
                dose = float(params.get(tc.get("dose", next(iter(lv))), i))
                conds.append(Condition(f"{name}@{i}", name, i, len(levels), params, t.group, t.family, dose))
    elif kind == "oracle":
        oc = cfg["oracle"]
        n = int(oc.get("n_bins", 8))
        edges = list(np.linspace(0, 0.5, n + 1)) + [R_MAX + 1e-9]
        for g in oc.get("gains", [0.0, 2.0]):
            for i in range(len(edges) - 1):
                p = dict(r_lo=float(edges[i]), r_hi=float(edges[i + 1]), gain=float(g))
                center = 0.5 * (edges[i] + min(edges[i + 1], R_MAX))
                conds.append(Condition(f"oracle_band_g{g:g}@{i}", "oracle_band", i, len(edges) - 1, p,
                                       "oracle", "oraculo", center))
        for kind_, key in (("oracle_lowpass", "lowpass"), ("oracle_highpass", "highpass")):
            cuts = oc.get(key, [])
            for i, c in enumerate(cuts):
                conds.append(Condition(f"{kind_}@{i}", kind_, i, len(cuts), {"cutoff": float(c)},
                                       "oracle", "oraculo", float(c)))
    else:
        raise ValueError(kind)
    return conds


def filter_available(conds: list[Condition]) -> list[Condition]:
    """Remove condições de SR cujos pesos não foram baixados (com aviso)."""
    from .techniques.superres import sr_available

    ok, missing = [], set()
    for c in conds:
        if c.technique in T.SR_TECHNIQUES:
            if not sr_available(T.SR_TECHNIQUES[c.technique], c.params["scale"], c.params["_model_dir"]):
                missing.add(f"{c.technique} x{c.params['scale']}")
                continue
        ok.append(c)
    if missing:
        print(f"[aviso] pesos de SR ausentes, condições ignoradas: {sorted(missing)}")
    return ok


def release_memory() -> None:
    """Devolve ao sistema a memória já liberada pelo glibc (Linux).

    Sem isso, os arrays de vários MB alocados e liberados a cada condição (224 px)
    fragmentam o heap e o RSS cresce ~290 MB por lote de 8 imagens no Imagenette,
    embora os objetos Python fiquem estáveis (medido com tracemalloc). Em outros
    sistemas não faz nada.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def apply_batch(cond: Condition, imgs: np.ndarray, backend: str, pool: ThreadPoolExecutor | None) -> np.ndarray:
    fn = lambda im: T.apply(cond.technique, im, cond.params, backend)
    if pool is None:
        return np.stack([fn(im) for im in imgs])
    return np.stack(list(pool.map(fn, imgs)))


def _feat_dist(f: np.ndarray, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distância cosseno e L2 relativa entre as features penúltimas da processada e da original."""
    n0, n1 = np.linalg.norm(f0, axis=1), np.linalg.norm(f, axis=1)
    cos = 1.0 - (f * f0).sum(1) / np.maximum(n0 * n1, 1e-12)
    return cos, np.linalg.norm(f - f0, axis=1) / np.maximum(n0, 1e-12)


def _pred_frame(cond: Condition, ids, labels, model: Classifier, ds_subset, logits, base, feats=None) -> pd.DataFrame:
    lp = log_softmax(logits)
    p = np.exp(lp)
    pred = logits.argmax(1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    n = len(ids)
    rows = {
        "image_id": ids,
        "model": model.name,
        "family": model.family,
        "cond_id": cond.cond_id,
        "technique": cond.technique,
        "level": cond.level,
        "group": cond.group,
        "dose": cond.dose,
        "label": labels,
        "pred": pred,
        "correct": (pred == labels).astype(np.int8),
        "correct_top5": (top5 == labels[:, None]).any(1).astype(np.int8),
        "p_true": p[np.arange(n), labels],
        "conf": p.max(1),
        "entropy": -(p * lp).sum(1),
    }
    if base is not None:
        b_lp, b_pred = base["lp"], base["pred"]
        rows["pred_orig"] = b_pred
        rows["flip"] = (pred != b_pred).astype(np.int8)
        rows["kl"] = (np.exp(b_lp) * (b_lp - lp)).sum(1)  # KL(p_orig || p_proc)
        rows["dp_true"] = rows["p_true"] - np.exp(b_lp)[np.arange(n), labels]
        if feats is not None and "feat" in base:
            rows["feat_cos"], rows["feat_l2"] = _feat_dist(feats, base["feat"])
    else:
        rows["pred_orig"] = pred
        rows["flip"] = np.zeros(n, np.int8)
        rows["kl"] = np.zeros(n)
        rows["dp_true"] = np.zeros(n)
        if feats is not None:
            rows["feat_cos"] = rows["feat_l2"] = np.zeros(n)
    if ds_subset is not None:  # acurácia restrita às classes do subconjunto (ex.: Imagenette)
        sub = np.asarray(ds_subset)
        ps = sub[logits[:, sub].argmax(1)]
        rows["correct_sub"] = (ps == labels).astype(np.int8)
        rows["flip_sub"] = (ps != (base["pred_sub"] if base is not None else ps)).astype(np.int8)
    return pd.DataFrame(rows)


def run_inference(cfg: dict, kind: str, only: list[str] | None = None) -> Path:
    t0 = time.time()
    device = resolve_device(cfg.get("device", "auto"))
    backend = cfg.get("backend", "scratch")
    bs = int(cfg.get("batch_size", 64))
    n_workers = int(cfg.get("n_workers", 0))
    out_root = Path(cfg["paths"]["results"]) / cfg["run_name"] / kind
    spec_cfg = SpectralConfig.from_dict(cfg.get("spectral"))
    use_brisque = bool(cfg.get("quality", {}).get("brisque", False))
    use_lpips = bool(cfg.get("quality", {}).get("lpips", False))
    use_feat = bool(cfg.get("quality", {}).get("feature_distance", False))
    shard = cfg.get("shard")  # "k/n": só as imagens k, k+n, k+2n... (para paralelizar em processos)
    conds = filter_available(build_conditions(cfg, kind))
    pool = ThreadPoolExecutor(n_workers) if n_workers > 0 else None
    print(f"[{kind}] device={device} backend={backend} condições={len(conds)}")

    for ds_name, dcfg in cfg["datasets"].items():
        if only and ds_name not in only:
            continue
        if not only and not dcfg.get("enabled", True):  # --datasets explícito liga mesmo se enabled: false
            continue
        imgset = load_dataset(ds_name, "val", dcfg.get("n_images"), cfg["seed"], cfg["paths"])
        out = out_root / ds_name
        if shard:
            k, n_sh = (int(v) for v in str(shard).split("/"))
            imgset = imgset.take(list(range(k, len(imgset), n_sh)))
            out = out_root / "_shards" / ds_name / f"{k:03d}"
            if (out / "meta.json").exists():
                print(f"[{kind}] {ds_name} fatia {k}/{n_sh} já existe em {out}")
                continue
        models = [Classifier(m, device) for m in dcfg["models"]]
        analyzer = SpectralAnalyzer(spec_cfg, imgset.image_size)
        out.mkdir(parents=True, exist_ok=True)
        spec_parts, pred_parts, profiles = [], [], []
        n_done = 0
        for ids, imgs, labels, _ in imgset.batches(bs):
            Fx, dcx = analyzer.transform(imgs)
            profiles.append(analyzer.profile(Fx))
            q_orig = Q.measure(imgs)
            b_orig = Q._brisque(imgs) if use_brisque else None
            base = {}
            for m in models:
                lg, ft = m.logits_and_features(imgs) if use_feat else (m.logits(imgs), None)
                b = {"lp": log_softmax(lg), "pred": lg.argmax(1)}
                if ft is not None:
                    b["feat"] = ft
                if imgset.class_subset is not None:
                    sub = np.asarray(imgset.class_subset)
                    b["pred_sub"] = sub[lg[:, sub].argmax(1)]
                base[m.name] = b
                pred_parts.append(_pred_frame(ORIGINAL, ids, labels, m, imgset.class_subset, lg, None, ft))
            for c in conds:
                proc = apply_batch(c, imgs, backend, pool)
                sp = analyzer.compare(imgs, proc, Fx, dcx)
                sp.update(Q.compare(q_orig, proc, b_orig, use_brisque))
                if use_lpips:
                    lp_ = Q.lpips_distance(imgs, proc, device)
                    if lp_ is not None:
                        sp["lpips"] = lp_
                sp = pd.DataFrame(sp)
                sp.insert(0, "image_id", ids)
                sp.insert(1, "cond_id", c.cond_id)
                spec_parts.append(sp)
                for m in models:
                    lg, ft = m.logits_and_features(proc) if use_feat else (m.logits(proc), None)
                    pred_parts.append(_pred_frame(c, ids, labels, m, imgset.class_subset, lg, base[m.name], ft))
            release_memory()
            n_done += len(ids)
            el = time.time() - t0
            print(f"  [{ds_name}] {n_done}/{len(imgset)} imagens  ({el/60:.1f} min)", flush=True)

        pd.concat(spec_parts, ignore_index=True).assign(dataset=ds_name).to_parquet(out / "spectral.parquet")
        pd.concat(pred_parts, ignore_index=True).assign(dataset=ds_name).to_parquet(out / "predictions.parquet")
        cat = pd.DataFrame([{**asdict(c), "params": json.dumps(c.public_params)} for c in conds])
        cat.to_csv(out / "conditions.csv", index=False)
        prof = np.concatenate(profiles)
        pd.DataFrame({"r_center": analyzer.fine_centers, "energy_mean": prof.mean(0),
                      "energy_median": np.median(prof, 0)}).to_csv(out / "dataset_spectrum.csv", index=False)
        meta = {"dataset": ds_name, "n_images": len(imgset), "models": [m.name for m in models],
                "device": device, "backend": backend, "image_size": imgset.image_size,
                "spectral": asdict(spec_cfg), "minutes": (time.time() - t0) / 60}
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{kind}] {ds_name} salvo em {out}")
    if pool:
        pool.shutdown()
    return out_root


def merge_shards(cfg: dict, kind: str, only: list[str] | None = None) -> None:
    """Junta ``results/<run>/<kind>/_shards/<ds>/<k>/`` em ``results/<run>/<kind>/<ds>/``."""
    out_root = Path(cfg["paths"]["results"]) / cfg["run_name"] / kind
    base = out_root / "_shards"
    if not base.exists():
        return
    for ds_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if only and ds_dir.name not in only:
            continue
        parts = sorted(p for p in ds_dir.iterdir() if (p / "meta.json").exists())
        if not parts:
            continue
        out = out_root / ds_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for f in ("spectral.parquet", "predictions.parquet"):
            pd.concat([pd.read_parquet(p / f) for p in parts], ignore_index=True).to_parquet(out / f)
        (out / "conditions.csv").write_bytes((parts[0] / "conditions.csv").read_bytes())
        metas = [json.loads((p / "meta.json").read_text(encoding="utf-8")) for p in parts]
        w = np.array([m["n_images"] for m in metas], float)
        sp = [pd.read_csv(p / "dataset_spectrum.csv") for p in parts]
        prof = sp[0][["r_center"]].copy()
        for c in ("energy_mean", "energy_median"):  # mediana: média ponderada das medianas das fatias (aprox.)
            prof[c] = np.average(np.stack([s[c].to_numpy() for s in sp]), axis=0, weights=w)
        prof.to_csv(out / "dataset_spectrum.csv", index=False)
        meta = {**metas[0], "n_images": int(w.sum()), "minutes": max(m["minutes"] for m in metas), "n_shards": len(parts)}
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{kind}] {ds_dir.name}: {len(parts)} fatias, {int(w.sum())} imagens -> {out}")
