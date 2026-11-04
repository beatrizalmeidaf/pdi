"""E4 — Separar *domain shift* de perda irreversível de informação (lacuna 3).

Lógica (inspirada na assimetria treino/teste de Ortiz-Jiménez et al., 2020):
se a queda sob inferência congelada some quando o modelo pode se adaptar às
imagens processadas, a informação discriminativa ainda está lá (domain shift).
Se a queda persiste mesmo com adaptação, a informação foi destruída.

Níveis de adaptação, do mais leve ao mais forte (o extrator é sempre o pré-treinado):
  frozen      modelo original, sem nada (= protocolo principal)
  bn_adapt    só re-estima as estatísticas de BatchNorm no domínio processado
              (controle acc_bn_oo: o mesmo procedimento nas originais, para descontar
              o ruído da estimativa com poucas imagens)
  probe_op    sonda linear treinada em features ORIGINAIS, testada em PROCESSADAS
  probe_pp    sonda linear treinada e testada em PROCESSADAS
  probe_oo    sonda linear treinada e testada em ORIGINAIS (teto de referência)
  ft_pp/ft_oo (opcional) fine-tuning completo por poucas épocas, em processadas/originais

Em datasets que são um subconjunto de classes do modelo (Imagenette: 10 das 1000 do
ImageNet), acc_* é medida só entre as K classes, como as sondas e o fine-tuning, que só
veem dados delas. Esse problema satura (99,8 % nas originais), então frozen e bn_adapt,
que não treinam com rótulos, também são medidos em todas as classes (colunas *_full).

Fração recuperável (no espaço da sonda, comparação justa):
  R = (probe_pp - probe_op) / (probe_oo - probe_op)
  R ~ 1 -> domain shift ; R ~ 0 -> perda de informação.
Perda irreversível estimada: probe_oo - probe_pp.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_device
from .datasets import load_dataset
from .models import Classifier
from .pipeline import ORIGINAL, apply_batch, build_conditions, filter_available, release_memory


def _probe(train_x, train_y, test_x, C: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(train_x)
    clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
    clf.fit(sc.transform(train_x), train_y)
    return clf, sc


def _acc_probe(model_sc, x, y) -> float:
    clf, sc = model_sc
    return float((clf.predict(sc.transform(x)) == y).mean())


def _acc_logits(logits, labels, subset) -> float:
    if subset is not None:
        sub = np.asarray(subset)
        pred = sub[logits[:, sub].argmax(1)]
    else:
        pred = logits.argmax(1)
    return float((pred == labels).mean())


def recoverable_fraction(oo: float, op: float, pp: float) -> float:
    """R = (pp - op) / (oo - op): 1 -> domain shift, 0 -> perda de informação. NaN sem queda."""
    gap = oo - op
    return (pp - op) / gap if abs(gap) > 1e-9 else np.nan


def _finetune(clf: Classifier, imgs, labels, epochs: int, lr: float, bs: int, seed: int) -> Classifier:
    import torch

    torch.manual_seed(seed)
    net = copy.deepcopy(clf.net)
    for m in net.modules():
        m._forward_hooks.clear()
    new = Classifier(clf.name, clf.device, module=net)
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    y_all = torch.as_tensor(labels, dtype=torch.long)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(imgs))
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            x = new._tensor(imgs[idx])
            loss = torch.nn.functional.cross_entropy(net(x), y_all[idx].to(clf.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return new


def _process_all(cond, imgs, backend, pool, bs=256):
    if cond.technique == "original":
        return imgs
    return np.concatenate([apply_batch(cond, imgs[s : s + bs], backend, pool) for s in range(0, len(imgs), bs)])


def run_adaptation(cfg: dict, only: list[str] | None = None) -> Path:
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    acfg = cfg["adaptation"]
    device = resolve_device(cfg.get("device", "auto"))
    backend = cfg.get("backend", "scratch")
    seed = cfg["seed"]
    n_workers = int(cfg.get("n_workers", 0))
    pool = ThreadPoolExecutor(n_workers) if n_workers > 0 else None
    out_root = Path(cfg["paths"]["results"]) / cfg["run_name"] / "adaptation"
    out_root.mkdir(parents=True, exist_ok=True)

    all_conds = {c.cond_id: c for c in filter_available(build_conditions(cfg, "main"))}
    wanted = acfg.get("conditions", "all")
    conds = list(all_conds.values()) if wanted == "all" else [all_conds[c] for c in wanted if c in all_conds]
    rows = []
    for ds_name, dcfg in cfg["datasets"].items():
        if (only and ds_name not in only) or (not only and not dcfg.get("enabled", True)):
            continue
        tr = load_dataset(ds_name, "train", dcfg.get("n_train", 2000), seed, cfg["paths"])
        te = load_dataset(ds_name, "val", dcfg.get("n_images"), seed, cfg["paths"])
        x_tr, x_te = tr.load_all(), te.load_all()
        print(f"[adapt] {ds_name}: treino={len(x_tr)} teste={len(x_te)} condições={len(conds)}")
        models = [Classifier(m, device) for m in dcfg["models"]]
        full = te.class_subset is not None  # também mede em todas as classes do modelo
        base = {}
        for m in models:
            f_tr, f_te = m.features(x_tr), m.features(x_te)
            pr = _probe(f_tr, tr.local_labels, f_te, acfg.get("probe_C", 1.0), seed)
            lg = m.logits(x_te)
            base[m.name] = dict(probe=pr, acc_probe_oo=_acc_probe(pr, f_te, te.local_labels),
                                acc_orig=_acc_logits(lg, te.labels, te.class_subset))
            if full:
                base[m.name]["acc_orig_full"] = _acc_logits(lg, te.labels, None)
            if acfg.get("bn_adapt", True) and m.has_bn:  # controle: BN re-estimado nas ORIGINAIS
                lg = m.bn_adapted(x_tr).logits(x_te)
                base[m.name]["acc_bn_oo"] = _acc_logits(lg, te.labels, te.class_subset)
                if full:
                    base[m.name]["acc_bn_oo_full"] = _acc_logits(lg, te.labels, None)
            if acfg.get("finetune_epochs", 0) > 0:
                ft = _finetune(m, x_tr, tr.labels, acfg["finetune_epochs"], acfg.get("finetune_lr", 1e-3),
                               acfg.get("finetune_bs", 64), seed)
                base[m.name]["acc_ft_oo"] = _acc_logits(ft.logits(x_te), te.labels, te.class_subset)
        for c in conds:
            p_tr = _process_all(c, x_tr, backend, pool)
            p_te = _process_all(c, x_te, backend, pool)
            for m in models:
                b = base[m.name]
                f_ptr, f_pte = m.features(p_tr), m.features(p_te)
                lg = m.logits(p_te)
                row = dict(dataset=ds_name, model=m.name, family=m.family, cond_id=c.cond_id,
                           technique=c.technique, level=c.level, group=c.group, dose=c.dose,
                           acc_orig=b["acc_orig"],
                           acc_frozen=_acc_logits(lg, te.labels, te.class_subset),
                           acc_probe_oo=b["acc_probe_oo"],
                           acc_probe_op=_acc_probe(b["probe"], f_pte, te.local_labels))
                if full:
                    row["acc_orig_full"] = b["acc_orig_full"]
                    row["acc_frozen_full"] = _acc_logits(lg, te.labels, None)
                pr = _probe(f_ptr, tr.local_labels, f_pte, acfg.get("probe_C", 1.0), seed)
                row["acc_probe_pp"] = _acc_probe(pr, f_pte, te.local_labels)
                if acfg.get("bn_adapt", True) and m.has_bn:
                    lg = m.bn_adapted(p_tr).logits(p_te)
                    row["acc_bn"] = _acc_logits(lg, te.labels, te.class_subset)
                    row["acc_bn_oo"] = b["acc_bn_oo"]
                    if full:
                        row["acc_bn_full"] = _acc_logits(lg, te.labels, None)
                        row["acc_bn_oo_full"] = b["acc_bn_oo_full"]
                if acfg.get("finetune_epochs", 0) > 0:
                    ft = _finetune(m, p_tr, tr.labels, acfg["finetune_epochs"], acfg.get("finetune_lr", 1e-3),
                                   acfg.get("finetune_bs", 64), seed)
                    row["acc_ft_pp"] = _acc_logits(ft.logits(p_te), te.labels, te.class_subset)
                    row["acc_ft_oo"] = b["acc_ft_oo"]
                row["recoverable_frac"] = recoverable_fraction(row["acc_probe_oo"], row["acc_probe_op"], row["acc_probe_pp"])
                row["info_loss"] = row["acc_probe_oo"] - row["acc_probe_pp"]
                rows.append(row)
            release_memory()
            print(f"  [{ds_name}] {c.cond_id} ok ({(time.time()-t0)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "adaptation.csv", index=False)
    if pool:
        pool.shutdown()
    print(f"[adapt] salvo em {out_root / 'adaptation.csv'}")
    return out_root
