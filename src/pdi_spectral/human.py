"""Estudo com pessoas: preferência visual (escolha forçada entre 2 imagens, 2AFC).

    python -m pdi_spectral human-export  --config configs/sweep.yaml --n 12
    python -m pdi_spectral human-analyze --config configs/sweep.yaml

``human-export`` cria ``results/<run>/human_study/`` com as imagens e um ``index.html``
que roda localmente (sem servidor, sem internet). Cada participante abre o arquivo,
responde ~"trials_per_participant" pares "qual imagem tem MELHOR qualidade visual?"
(original × processada, lado sorteado) e no final baixa um CSV. Os CSVs vão para
``results/<run>/human_study/respostas/``.

Há ensaios de atenção (original × blur forte, resposta esperada: original).
Participantes com < 80 % de acerto nesses ensaios são descartados na análise.

``human-analyze`` calcula, por técnica e dose, P(prefere a processada) com IC de
Wilson e define:
  human_better_dose  menor dose com IC inteiro acima de 50 %  (melhora percebida)
  human_worse_dose   menor dose com IC inteiro abaixo de 50 %  (piora percebida)
e cruza com o limiar do classificador (thresholds.csv): janela humana =
[human_better_dose, BMDL].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import techniques as T
from .datasets import load_dataset
from .pipeline import build_conditions, filter_available
from .thresholds import intensity, technique_meta

CATCH = ("gaussian", {"sigma": 3.0})


def _save_png(img: np.ndarray, path: Path):
    from PIL import Image

    Image.fromarray(img).save(path)


def export_study(cfg: dict, dataset: str, n_images: int = 12, trials_per_participant: int = 120,
                 techniques: list[str] | None = None, max_levels: int = 6, scale_px: int = 192) -> Path:
    run_dir = Path(cfg["paths"]["results"]) / cfg["run_name"]
    out = run_dir / "human_study"
    img_dir = out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    (out / "respostas").mkdir(exist_ok=True)
    imgs = load_dataset(dataset, "val", n_images, cfg["seed"] + 7, cfg["paths"])
    x = imgs.load_all()
    conds = filter_available(build_conditions(cfg, "main"))
    if techniques:
        conds = [c for c in conds if c.technique in techniques]
    # no máximo `max_levels` doses por técnica, espalhadas na faixa
    by_t: dict[str, list] = {}
    for c in conds:
        by_t.setdefault(c.technique, []).append(c)
    chosen = []
    for t, cs in by_t.items():
        idx = np.unique(np.round(np.linspace(0, len(cs) - 1, min(max_levels, len(cs)))).astype(int))
        chosen += [cs[i] for i in idx]
    backend = cfg.get("backend", "scratch")
    pairs = []
    for i, im in enumerate(x):
        _save_png(im, img_dir / f"o{i}.png")
        for c in chosen:
            name = f"{c.cond_id.replace('@', '_')}_{i}.png"
            _save_png(T.apply(c.technique, im, c.params, backend), img_dir / name)
            pairs.append(dict(image=i, cond_id=c.cond_id, technique=c.technique, dose=c.dose,
                              orig=f"img/o{i}.png", proc=f"img/{name}", catch=False))
        cname = f"catch_{i}.png"
        _save_png(T.apply(CATCH[0], im, CATCH[1], backend), img_dir / cname)
        pairs.append(dict(image=i, cond_id="catch", technique="catch", dose=0.0,
                          orig=f"img/o{i}.png", proc=f"img/{cname}", catch=True))
    pd.DataFrame(pairs).to_csv(out / "pairs.csv", index=False)
    html = _HTML.replace("__PAIRS__", json.dumps(pairs)).replace("__N__", str(trials_per_participant)) \
                .replace("__PX__", str(scale_px)).replace("__DATASET__", dataset)
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"[human] {len(pairs)} pares ({len(chosen)} condições x {len(x)} imagens + atenção) em {out}")
    print("        envie a pasta inteira (zip) aos participantes; eles abrem index.html no navegador.")
    return out


def _wilson(k, n, z=1.645):  # IC 90 %
    n = np.maximum(n, 1)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def analyze_study(cfg: dict, min_catch: float = 0.8) -> pd.DataFrame:
    run_dir = Path(cfg["paths"]["results"]) / cfg["run_name"]
    resp_dir = run_dir / "human_study" / "respostas"
    files = sorted(resp_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"nenhum CSV de respostas em {resp_dir}")
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    catch = r[r.catch.astype(str).str.lower() == "true"]
    ok = catch.groupby("participant")["chose_processed"].apply(lambda s: 1 - s.mean())
    keep = ok[ok >= min_catch].index
    print(f"[human] {r.participant.nunique()} participantes; {len(keep)} passaram nos ensaios de atenção")
    r = r[(r.participant.isin(keep)) & (r.catch.astype(str).str.lower() != "true")]
    g = r.groupby(["technique", "cond_id", "dose"]).agg(n=("chose_processed", "size"),
                                                        k=("chose_processed", "sum")).reset_index()
    g["p_prefer_proc"] = g.k / g.n
    g["ci_lo"], g["ci_hi"] = _wilson(g.k.to_numpy(), g.n.to_numpy())
    rows = []
    thr_path = run_dir / "analysis" / "thresholds.csv"
    thr = pd.read_csv(thr_path) if thr_path.exists() else None
    for t, h in g.groupby("technique"):
        meta = technique_meta(cfg, t)
        h = h.assign(x=intensity(h.dose.to_numpy(), meta)).sort_values("x")
        better = h[h.ci_lo > 0.5]
        worse = h[h.ci_hi < 0.5]
        row = dict(technique=t, n_responses=int(h.n.sum()),
                   human_better_dose=float(better.x.iloc[0]) if len(better) else np.nan,
                   human_worse_dose=float(worse.x.iloc[0]) if len(worse) else np.nan,
                   best_dose=float(h.x.iloc[int(np.argmax(h.p_prefer_proc))]),
                   best_p=float(h.p_prefer_proc.max()))
        if thr is not None and t in set(thr.technique):
            bmdl = thr[thr.technique == t]["bmdl_acc0.01"].mean()
            row["bmdl_acc0.01_mean_models"] = bmdl
            row["human_window_exists"] = bool(np.isfinite(row["human_better_dose"]) and np.isfinite(bmdl)
                                              and row["human_better_dose"] <= bmdl)
        rows.append(row)
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "human_preference_by_dose.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(out / "human_thresholds.csv", index=False)
    print(res.round(3).to_string(index=False))
    return res


_HTML = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estudo de qualidade visual</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#dcdcd6;--acc:#2f5d8a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9a9a94;--card:#21211f;--line:#383835;--acc:#7fb0e0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 8px}p{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px}
.pair{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin:16px 0}
.pair button{background:none;border:2px solid var(--line);border-radius:8px;padding:8px;cursor:pointer}
.pair button:hover,.pair button:focus{border-color:var(--acc);outline:none}
.pair img{width:__PX__px;height:__PX__px;image-rendering:pixelated;display:block}
.lbl{font-size:13px;color:var(--muted);text-align:center;margin-top:4px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar>div{height:100%;background:var(--acc)}
input,.go{font:inherit;padding:8px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.go{background:var(--acc);color:#fff;border:none;cursor:pointer}
</style></head><body><main>
<h1>Qual imagem tem melhor qualidade visual?</h1>
<div id="intro" class="card">
<p>Você verá pares de imagens pequenas (ampliadas, por isso os pixels aparecem). Em cada par, clique na imagem que
parece ter <b>melhor qualidade visual</b> (mais nítida, mais agradável, menos defeitos). Não existe resposta certa;
vá pela primeira impressão. Atalhos: tecla <b>←</b> escolhe a esquerda e <b>→</b> a direita. Leva ~5 minutos.</p>
<p><label>Seu código de participante (iniciais ou número): <input id="pid" maxlength="20"></label></p>
<button class="go" id="start">Começar</button></div>
<div id="task" style="display:none"><div class="bar"><div id="prog" style="width:0"></div></div>
<p id="count"></p><div class="pair">
<div><button id="L"><img id="imL" alt="imagem da esquerda"></button><div class="lbl">esquerda (←)</div></div>
<div><button id="R"><img id="imR" alt="imagem da direita"></button><div class="lbl">direita (→)</div></div></div></div>
<div id="done" class="card" style="display:none"><p><b>Obrigado!</b> O arquivo com suas respostas foi baixado
(<span id="fname"></span>). Envie-o para quem conduz o estudo. Se o download não começou:</p>
<button class="go" id="again">Baixar de novo</button></div>
</main><script>
const PAIRS=__PAIRS__, N=__N__, DATASET="__DATASET__";
function shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
let trials=[],k=0,res=[],t0=0,pid="";
function pick(){const c=PAIRS.filter(p=>p.catch),r=PAIRS.filter(p=>!p.catch);
 const nc=Math.max(4,Math.round(N*0.08));return shuffle(shuffle(c).slice(0,nc).concat(shuffle(r).slice(0,N-nc)))
  .map(p=>Object.assign({},p,{procLeft:Math.random()<0.5}))}
function show(){const t=trials[k];document.getElementById("imL").src=t.procLeft?t.proc:t.orig;
 document.getElementById("imR").src=t.procLeft?t.orig:t.proc;
 document.getElementById("count").textContent=`Par ${k+1} de ${trials.length}`;
 document.getElementById("prog").style.width=(100*k/trials.length)+"%";t0=performance.now()}
function choose(side){if(document.getElementById("task").style.display==="none")return;const t=trials[k];
 const procChosen=(side==="L")===t.procLeft;
 res.push({participant:pid,dataset:DATASET,trial:k,image:t.image,cond_id:t.cond_id,technique:t.technique,dose:t.dose,
  catch:t.catch,proc_left:t.procLeft,chose_processed:procChosen?1:0,rt_ms:Math.round(performance.now()-t0)});
 k++;if(k<trials.length)show();else finish()}
function csv(){const cols=Object.keys(res[0]);return cols.join(",")+"\n"+res.map(r=>cols.map(c=>r[c]).join(",")).join("\n")}
function download(){const b=new Blob([csv()],{type:"text/csv"});const a=document.createElement("a");
 const name=`respostas_${pid}_${Date.now()}.csv`;a.href=URL.createObjectURL(b);a.download=name;a.click();
 document.getElementById("fname").textContent=name}
function finish(){document.getElementById("task").style.display="none";document.getElementById("done").style.display="";download()}
document.getElementById("start").onclick=()=>{pid=(document.getElementById("pid").value||"anon").replace(/[^\w-]/g,"");
 trials=pick();document.getElementById("intro").style.display="none";document.getElementById("task").style.display="";show()};
document.getElementById("L").onclick=()=>choose("L");document.getElementById("R").onclick=()=>choose("R");
document.getElementById("again").onclick=download;
document.addEventListener("keydown",e=>{if(e.key==="ArrowLeft")choose("L");if(e.key==="ArrowRight")choose("R")});
</script></body></html>
"""
