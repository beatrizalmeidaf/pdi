/* Quando melhorar piora — gráficos interativos. Lê assets/data.json (gerado por `make report`). */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k);
    return n;
  };
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (v, nd = 2) => (v == null || !isFinite(v) ? "—" : Number(v).toLocaleString("pt-BR", { minimumFractionDigits: nd, maximumFractionDigits: nd }));
  const pp = (v, nd = 1) => (v == null ? "—" : (v > 0 ? "+" : "") + fmt(100 * v, nd) + " p.p.");
  const pct = (v, nd = 0) => (v == null ? "—" : (v > 0 ? "+" : "") + fmt(100 * v, nd) + "%");
  const isJpeg = (t) => t.startsWith("jpeg");
  const toX = (t, dose) => (isJpeg(t) ? 100 - dose : dose);
  const doseLabel = (t, x) => (isJpeg(t) ? `qualidade ${fmt(100 - x, 0)}` : fmt(x, x < 1 && x !== 0 ? 2 : (Number.isInteger(x) ? 0 : 2)));
  const SLOTS = () => [css("--s1"), css("--s2"), css("--s3")];
  const DS_COLOR = () => ({ cifar10: css("--s1"), imagenette: css("--s2") });
  let D = null;

  // ---------- Plotly base ----------
  const base = (extra = {}) => {
    const ink2 = css("--ink-2"), grid = css("--grid"), axis = css("--axis"), surf = css("--surface");
    const ax = { gridcolor: grid, zerolinecolor: axis, linecolor: axis, tickcolor: axis, tickfont: { color: ink2, size: 11 },
      title: { font: { color: ink2, size: 12 } }, automargin: true };
    return Object.assign({
      paper_bgcolor: surf, plot_bgcolor: surf, font: { family: css("--font"), color: ink2, size: 12 },
      margin: { l: 56, r: 14, t: 10, b: 46 }, xaxis: Object.assign({}, ax, { showgrid: false }), yaxis: Object.assign({}, ax),
      hovermode: "x unified", hoverlabel: { bgcolor: surf, bordercolor: css("--line"), font: { color: css("--ink") } },
      legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11 } }, showlegend: true,
    }, extra);
  };
  const CFG = { displayModeBar: false, responsive: true };
  const plot = (id, traces, layout) => window.Plotly && Plotly.react(id, traces, layout, CFG);

  // ---------- controles segmentados ----------
  function seg(container, options, value, onChange) {
    container.replaceChildren();
    for (const [v, label] of options) {
      const b = el("button", { type: "button", "aria-pressed": String(v === value), text: label });
      b.addEventListener("click", () => {
        container.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        onChange(v);
      });
      container.append(b);
    }
  }
  const dsOptions = () => Object.entries(D.datasets).map(([k, v]) => [k, v.label]);

  // ---------- KPIs ----------
  function kpis() {
    const nT = Object.keys(D.techniques).length;
    const nM = Object.values(D.datasets).reduce((a, d) => a + d.models.length, 0);
    const ire = Object.values(D.datasets).flatMap((d) => Object.values(d.ire).map((r) => r.r2_p_loto));
    const items = [
      ["Técnicas avaliadas", nT, "do zero, validadas contra OpenCV/PIL/SciPy"],
      ["Classificadores", nM, "5 CNNs + 1 Vision Transformer, congelados"],
      ["Avaliações imagem × condição × modelo", "≈ 1,6 mi", "2000 imagens por dataset, 106 doses + 27 filtros ideais"],
      ["R² da fórmula IRE", `${fmt(Math.min(...ire), 2)}–${fmt(Math.max(...ire), 2)}`, "prevendo técnicas que ela não viu"],
    ];
    $("#kpis").replaceChildren(...items.map(([l, v, n]) => el("div", { class: "card kpi" },
      el("div", { class: "label", text: l }), el("div", { class: "value", text: String(v) }), el("div", { class: "note", text: n }))));
  }

  // ---------- explorador ----------
  const EX = { ds: "imagenette", tech: "unsharp", idx: 0 };
  function exInit() {
    if (!D.datasets[EX.ds]) EX.ds = Object.keys(D.datasets)[0];
    seg($("#ex-ds"), dsOptions(), EX.ds, (v) => { EX.ds = v; exTechs(); exRender(true); });
    exTechs();
    $("#ex-tech").addEventListener("change", (e) => { EX.tech = e.target.value; exRender(true); });
    $("#ex-slider").addEventListener("input", (e) => { EX.idx = +e.target.value; exRender(false); });
    exRender(true);
  }
  function exTechs() {
    const s = $("#ex-tech");
    const avail = Object.keys(D.datasets[EX.ds].curves);
    s.replaceChildren(...avail.map((t) => el("option", { value: t, text: D.techniques[t].label })));
    if (!avail.includes(EX.tech)) EX.tech = avail[0];
    s.value = EX.tech;
  }
  function exRender(full) {
    const ds = D.datasets[EX.ds], t = EX.tech, cv = ds.curves[t], models = ds.models;
    const xs = cv[models[0].id].x;
    const sl = $("#ex-slider");
    if (full) {
      sl.max = xs.length - 1;
      EX.idx = Math.min(Math.max(1, Math.round(xs.length / 3)), xs.length - 1);
      sl.value = EX.idx;
    }
    const x = xs[EX.idx];
    $("#ex-param").textContent = `${D.techniques[t].label} · ${D.techniques[t].param}`;
    $("#ex-dose").textContent = doseLabel(t, x);
    const C = SLOTS();
    const traces = [];
    models.forEach((m, i) => {
      const c = cv[m.id];
      traces.push({ x: c.x, y: c.hi.map((v) => 100 * v), mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false });
      traces.push({ x: c.x, y: c.lo.map((v) => 100 * v), mode: "lines", line: { width: 0 }, fill: "tonexty",
        fillcolor: C[i] + "22", hoverinfo: "skip", showlegend: false });
      traces.push({ x: c.x, y: c.d.map((v) => 100 * v), mode: "lines+markers", name: m.label,
        line: { color: C[i], width: 2 }, marker: { size: 6, color: C[i], line: { color: css("--surface"), width: 1.5 } },
        hovertemplate: "%{y:+.2f} p.p.<extra>" + m.label + "</extra>" });
    });
    const shapes = [
      { type: "line", xref: "paper", x0: 0, x1: 1, y0: -1, y1: -1, line: { color: css("--muted"), width: 1 } },
      { type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: css("--ink"), width: 1.5 } },
    ];
    plot("ex-chart", traces, base({ shapes, yaxis: Object.assign(base().yaxis, { title: { text: "Δ acurácia (p.p.)" } }),
      xaxis: Object.assign(base().xaxis, { title: { text: isJpeg(t) ? "100 − qualidade JPEG" : D.techniques[t].param } }),
      annotations: [{ xref: "paper", x: 1, y: -1, yanchor: "bottom", xanchor: "right", text: "−1 p.p.", showarrow: false, font: { size: 10, color: css("--muted") } }] }));

    // qualidade perceptual (um eixo só: variação do BRISQUE em %, negativo = melhor)
    const q = ds.quality[t];
    if (q) {
      const qx = q.x, qb = q.dq_brisque.map((v) => (v == null ? null : -100 * v));
      plot("ex-q-chart", [{ x: qx, y: qb, mode: "lines+markers", name: "ganho de qualidade (BRISQUE)", line: { color: css("--ink-2"), width: 2 },
        marker: { size: 5, color: css("--ink-2") }, hovertemplate: "%{y:+.1f}%<extra>BRISQUE</extra>" }],
        base({ showlegend: false, margin: { l: 56, r: 14, t: 26, b: 40 },
          shapes: [{ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: css("--ink"), width: 1.5 } }],
          yaxis: Object.assign(base().yaxis, { title: { text: "ganho BRISQUE (%)" } }),
          xaxis: Object.assign(base().xaxis, { title: { text: isJpeg(t) ? "100 − qualidade JPEG" : D.techniques[t].param } }),
          title: { text: "Qualidade perceptual sem referência (acima de 0 = imagem melhor para o BRISQUE)", font: { size: 12, color: css("--ink-2") }, x: 0, xanchor: "left", y: 0.98 } }));
    }

    // imagens
    const S = ds.samples;
    const pairs = $("#ex-pairs");
    pairs.replaceChildren();
    if (S) {
      const lv = (S.techniques[t] || []).find((l) => Math.abs(toX(t, l.dose) - x) < 1e-6);
      const px = EX.ds === "cifar10" ? "px" : "";
      S.images.forEach((im, k) => {
        const proc = lv ? "samples/" + lv.files[k] : "samples/" + im.original;
        pairs.append(el("div", { class: "pair" },
          el("figure", {}, el("img", { src: "samples/" + im.original, alt: "original", class: px, loading: "lazy" }), el("figcaption", { text: "original" })),
          el("figure", {}, el("img", { src: proc, alt: `${D.techniques[t].label} em ${doseLabel(t, x)}`, class: px, loading: "lazy" }),
            el("figcaption", { text: lv ? doseLabel(t, x) : "dose identidade" }))));
      });
      pairs.style.gridTemplateColumns = S.images.length > 2 ? "repeat(2, minmax(0,1fr))" : "minmax(0,1fr)";
    }
    $("#ex-title").textContent = `${D.techniques[t].label} · ${ds.label}`;

    // métricas nesta dose
    const mt = $("#ex-metrics");
    const head = el("tr", {}, el("th", { text: "Modelo" }), el("th", { class: "num", text: "Δ acurácia" }), el("th", { class: "num", text: "IC 90%" }), el("th", { class: "num", text: "decisões mudadas" }));
    const rows = models.map((m, i) => {
      const c = cv[m.id];
      const d = c.d[EX.idx];
      return el("tr", {}, el("td", {}, el("span", { class: "key", style: `background:${C[i]}` }), m.label),
        el("td", { class: "num " + (d > 0.002 ? "pos" : d < -0.01 ? "neg" : ""), text: pp(d, 2) }),
        el("td", { class: "num", text: `${fmt(100 * c.lo[EX.idx], 1)} a ${fmt(100 * c.hi[EX.idx], 1)}` }),
        el("td", { class: "num", text: fmt(100 * c.flip[EX.idx], 1) + "%" }));
    });
    const qi = q ? q.x.findIndex((v) => Math.abs(v - x) < 1e-6) : -1;
    const extra = qi >= 0 ? el("tr", {}, el("td", { colspan: "4", class: "small muted",
      text: `Imagem: BRISQUE ${pct(-q.dq_brisque[qi], 1)} (ganho) · PSNR ${fmt(q.psnr[qi], 1)} dB · SSIM ${fmt(q.ssim[qi], 3)}` })) : null;
    mt.replaceChildren(el("thead", {}, head), el("tbody", {}, ...rows, extra));

    // limiares
    const th = ds.thresholds[t] || {};
    const thHead = el("tr", {}, el("th", { text: "Modelo" }), el("th", { class: "num", text: "BMD 1 p.p." }), el("th", { class: "num", text: "BMDL" }),
      el("th", { class: "num", text: "objetivo visual" }), el("th", { text: "janela" }), el("th", { class: "num", text: "melhor ganho" }));
    const status = (r) => r["bmd_acc0.01_status"] === "<x_min" ? "< 1ª dose" : r["bmd_acc0.01_status"] === ">x_max" ? "> última" : doseLabel(t, r["bmd_acc0.01"]);
    const thRows = models.filter((m) => th[m.id]).map((m, i) => {
      const r = th[m.id];
      const win = r.objective ? (r.window_exists ? (r["bmd_acc0.01_status"] === "<x_min" ? "~ limite" : "✓ sim") : "✗ não") : "—";
      return el("tr", {}, el("td", {}, el("span", { class: "key", style: `background:${C[models.indexOf(m)]}` }), m.label),
        el("td", { class: "num", text: status(r) }), el("td", { class: "num", text: r["bmdl_acc0.01"] == null ? "—" : doseLabel(t, r["bmdl_acc0.01"]) }),
        el("td", { class: "num", text: r.perc_dose == null ? "—" : doseLabel(t, r.perc_dose) }),
        el("td", { class: r.window_exists && r["bmd_acc0.01_status"] !== "<x_min" ? "pos" : "", text: win }),
        el("td", { class: "num " + (r.gain_significant ? "pos" : ""), text: r.best_gain > 0 ? pp(r.best_gain, 2) + (r.gain_significant ? " *" : "") : "—" }));
    });
    $("#ex-thr").replaceChildren(el("thead", {}, thHead), el("tbody", {}, ...thRows));
  }

  // ---------- heatmap ----------
  const HM_TECHS = ["unsharp", "laplacian", "hist_eq", "clahe", "clahe_blend", "gaussian", "bilateral", "median", "wiener", "jpeg", "jpeg444", "sr_bicubic"];
  function heatmap() {
    const cols = Object.entries(D.datasets).flatMap(([k, d]) => d.models.map((m) => [k, m]));
    const t = el("table", { class: "hm" });
    const h1 = el("tr", {}, el("th", {}));
    for (const [k, d] of Object.entries(D.datasets)) h1.append(el("th", { colspan: String(d.models.length), text: d.label }));
    const h2 = el("tr", {}, el("th", {}), ...cols.map(([, m]) => el("th", { text: m.label })));
    const body = HM_TECHS.map((tech) => {
      const tr = el("tr", {}, el("th", { class: "row", text: `${D.techniques[tech].label} (${isJpeg(tech) ? "qualidade" : D.techniques[tech].param})` }));
      for (const [k, m] of cols) {
        const r = (D.datasets[k].thresholds[tech] || {})[m.id];
        if (!r) { tr.append(el("td", { class: "na", text: "—" })); continue; }
        const st = r["bmd_acc0.01_status"];
        const val = st === "<x_min" ? "< mín." : st === ">x_max" ? "> máx." : doseLabel(tech, r["bmd_acc0.01"]).replace("qualidade ", "Q");
        let cls = "na", mark = "";
        if (r.objective) {
          if (r.window_exists && st === "<x_min") { cls = "edge"; mark = "~ "; }
          else if (r.window_exists) { cls = "ok"; mark = "✓ "; }
          else { cls = ""; mark = "✗ "; }
        }
        const title = `${D.techniques[tech].label} · ${m.label}: perde 1 p.p. em ${val}` +
          (r.perc_dose != null ? `; objetivo visual em ${doseLabel(tech, r.perc_dose)}` : "") +
          (r.best_gain > 0 ? `; melhor ganho ${pp(r.best_gain, 2)}${r.gain_significant ? " (significativo)" : ""}` : "");
        tr.append(el("td", { class: cls, title, tabindex: "0", "aria-label": title, text: mark + val }));
      }
      return tr;
    });
    t.append(el("thead", {}, h1, h2), el("tbody", {}, ...body));
    $("#heatmap").replaceChildren(t);
  }

  // ---------- hormese ----------
  const HZ = { ds: "imagenette" };
  function hormesis() {
    if (!D.datasets[HZ.ds]) HZ.ds = Object.keys(D.datasets)[0];
    seg($("#hz-ds"), dsOptions(), HZ.ds, (v) => { HZ.ds = v; hormesisRender(); });
    hormesisRender();
  }
  function hormesisRender() {
    const ds = D.datasets[HZ.ds], C = SLOTS();
    const techs = HM_TECHS.filter((t) => t !== "sr_bicubic").slice().reverse();
    const traces = ds.models.map((m, i) => {
      const ys = [], xs = [], lo = [], hi = [], sz = [], txt = [];
      techs.forEach((t) => {
        const r = (ds.thresholds[t] || {})[m.id];
        if (!r) return;
        ys.push(D.techniques[t].label); xs.push(100 * r.best_gain);
        lo.push(100 * (r.best_gain - r.best_gain_lo)); hi.push(100 * (r.best_gain_hi - r.best_gain));
        sz.push(r.gain_significant ? 11 : 7);
        txt.push(r.best_gain > 0 ? `dose ${doseLabel(t, r.best_dose)}${r.gain_significant ? " · significativo" : ""}` : "nenhuma dose melhora");
      });
      return { type: "scatter", mode: "markers", name: m.label, x: xs, y: ys, text: txt, offsetgroup: String(i),
        error_x: { type: "data", symmetric: false, array: hi, arrayminus: lo, color: C[i], thickness: 1.5, width: 0 },
        marker: { color: C[i], size: sz, line: { color: css("--surface"), width: 1.5 } },
        hovertemplate: "%{x:+.2f} p.p. · %{text}<extra>" + m.label + "</extra>" };
    });
    plot("hz-chart", traces, base({ hovermode: "closest", scattermode: "group", scattergap: 0.5, margin: { l: 130, r: 14, t: 10, b: 46 },
      xaxis: Object.assign(base().xaxis, { showgrid: true, zeroline: true, title: { text: "melhor Δ acurácia (p.p.), IC 90%" } }),
      yaxis: Object.assign(base().yaxis, { showgrid: false }) }));
  }

  // ---------- CNN × ViT ----------
  const VIT = [["gaussian", 1], ["gaussian", 2], ["bilateral", 4], ["median", 5], ["wiener", 5], ["jpeg", 10], ["clahe", 4], ["unsharp", 3], ["laplacian", 1.5]];
  function cnnVit() {
    const ds = D.datasets.imagenette;
    if (!ds) { $("#vit-chart").closest(".card").remove(); return; }
    const C = SLOTS();
    const labels = VIT.map(([t, d]) => `${D.techniques[t].label.replace(" masking", "")} · ${isJpeg(t) ? "Q" + d : D.techniques[t].param + " " + fmt(d, Number.isInteger(d) ? 0 : 1)}`);
    const traces = ds.models.map((m, i) => ({
      type: "bar", orientation: "h", name: m.label, y: labels, marker: { color: C[i] },
      x: VIT.map(([t, d]) => { const c = ds.curves[t][m.id]; const j = c.x.findIndex((v) => Math.abs(v - toX(t, d)) < 1e-6); return j >= 0 ? 100 * c.d[j] : null; }),
      hovertemplate: "%{x:+.1f} p.p.<extra>" + m.label + "</extra>" }));
    plot("vit-chart", traces, base({ barmode: "group", bargap: 0.35, bargroupgap: 0.08, hovermode: "y unified", margin: { l: 150, r: 14, t: 10, b: 46 },
      xaxis: Object.assign(base().xaxis, { showgrid: true, title: { text: "Δ acurácia (p.p.)" } }),
      yaxis: Object.assign(base().yaxis, { autorange: "reversed", showgrid: false }) }));
  }

  // ---------- bandas ----------
  let BD = "remove";
  function bands() {
    seg($("#bd-mode"), [["remove", "remover a banda"], ["boost", "dobrar a banda"]], BD, (v) => { BD = v; bandsRender(); });
    bandsRender();
  }
  function bandsRender() {
    const C = SLOTS();
    for (const k of ["cifar10", "imagenette"]) {
      const ds = D.datasets[k];
      if (!ds) continue;
      const traces = ds.models.map((m, i) => ({ x: ds.bands[m.id].x, y: ds.bands[m.id][BD].map((v) => 100 * v), name: m.label, mode: "lines+markers",
        line: { color: C[i], width: 2 }, marker: { size: 6, color: C[i], line: { color: css("--surface"), width: 1.5 } },
        hovertemplate: "%{y:+.1f} p.p.<extra>" + m.label + "</extra>" }));
      plot("bd-" + k, traces, base({ margin: { l: 48, r: 8, t: 34, b: 96 }, legend: { orientation: "h", y: -0.42, x: 0, font: { size: 10 } },
        title: { text: ds.label, font: { size: 13, color: css("--ink") }, x: 0, xanchor: "left", y: 0.97 },
        xaxis: Object.assign(base().xaxis, { title: { text: "ciclos/pixel" } }),
        yaxis: Object.assign(base().yaxis, { title: { text: "Δ acurácia (p.p.)" } }) }));
    }
  }

  // ---------- preditores ----------
  function predictors() {
    const keys = ["dose_rank", "psnr", "ssim", "delta_bands", "spec_bands", "spec_bands+coh", "spec_bands+coh+chroma"];
    const DC = DS_COLOR();
    const traces = Object.entries(D.datasets).map(([k, ds]) => {
      const vals = keys.map((f) => { const r = Object.values(ds.loto[f].r2).filter((v) => v != null); return r.length ? r.reduce((a, b) => a + b, 0) / r.length : null; });
      const mins = keys.map((f, j) => vals[j] - Math.min(...Object.values(ds.loto[f].r2)));
      const maxs = keys.map((f, j) => Math.max(...Object.values(ds.loto[f].r2)) - vals[j]);
      return { type: "bar", orientation: "h", name: ds.label, y: keys.map((f) => ds.loto[f].label), x: vals.map((v) => Math.max(v, -0.1)),
        marker: { color: DC[k] }, error_x: { type: "data", symmetric: false, array: maxs, arrayminus: mins, color: css("--ink-2"), thickness: 1, width: 0 },
        hovertemplate: "R² %{x:.2f}<extra>" + ds.label + "</extra>" };
    });
    plot("pred-chart", traces, base({ barmode: "group", bargap: 0.3, hovermode: "y unified", margin: { l: 190, r: 14, t: 10, b: 46 },
      xaxis: Object.assign(base().xaxis, { showgrid: true, range: [-0.1, 1], title: { text: "R² em técnicas nunca vistas" } }),
      yaxis: Object.assign(base().yaxis, { autorange: "reversed", showgrid: false }) }));
  }

  // ---------- IRE ----------
  function ire() {
    if (window.katex) {
      katex.render(String.raw`\mathrm{IRE}=\sum_b w_b\,D_b^{\gamma}+\sum_b c_b\,\big(D^{\mathrm{cor}}_b\big)^{\gamma}+\sum_{B} u_B\,(1-\rho_B)^{\gamma}`, $("#f1"), { displayMode: true, throwOnError: false });
      katex.render(String.raw`\widehat{P}(\text{decisão mudar}) = 1 - e^{-\mathrm{IRE}}`, $("#f2"), { displayMode: true, throwOnError: false });
      katex.render(String.raw`\mathrm{IRE}\le -\ln(1-\tau)\;\Longrightarrow\;\text{queda de acurácia}\le\tau`, $("#f3"), { displayMode: true, throwOnError: false });
    }
    const DC = DS_COLOR();
    const traces = Object.entries(D.datasets).map(([k, ds]) => ({
      type: "scattergl", mode: "markers", name: ds.label, x: ds.ire_points.map((p) => 100 * p.obs), y: ds.ire_points.map((p) => 100 * p.pred),
      text: ds.ire_points.map((p) => `${D.techniques[p.t].label} · ${doseLabel(p.t, toX(p.t, p.dose))} · ${(ds.models.find((m) => m.id === p.m) || {}).label}`),
      marker: { color: DC[k], size: 6, opacity: 0.75 }, hovertemplate: "observado %{x:.1f}% · previsto %{y:.1f}%<br>%{text}<extra></extra>" }));
    traces.push({ x: [0, 90], y: [0, 90], mode: "lines", line: { color: css("--axis"), width: 1 }, hoverinfo: "skip", showlegend: false });
    plot("ire-chart", traces, base({ hovermode: "closest", xaxis: Object.assign(base().xaxis, { showgrid: true, title: { text: "decisões que mudaram, observado (%)" } }),
      yaxis: Object.assign(base().yaxis, { title: { text: "previsto pelo IRE (%)" } }) }));
    const rows = Object.values(D.datasets).flatMap((ds) => ds.models.filter((m) => ds.ire[m.id]).map((m) => {
      const r = ds.ire[m.id];
      return el("tr", {}, el("td", { text: `${m.label}` }), el("td", { class: "small muted", text: ds.label }),
        el("td", { class: "num", text: fmt(r.r2_p_loto) }), el("td", { class: "num", text: fmt(r.spearman_p_loto) }), el("td", { class: "num", text: fmt(r.r2_p_oracle) }));
    }));
    $("#ire-table").replaceChildren(el("thead", {}, el("tr", {}, el("th", { text: "Modelo" }), el("th", { text: "Dataset" }), el("th", { class: "num", text: "R²" }),
      el("th", { class: "num", text: "Spearman" }), el("th", { class: "num", text: "R² só filtros ideais" }))), el("tbody", {}, ...rows));
  }

  // ---------- adaptação ----------
  function adaptation() {
    const a = (D.adaptation || {}).cifar10;
    if (!a) return;
    const rows = a.map((r) => Object.assign({}, r, { rec: (r.ft - r.frozen) / (r.ctrl - r.frozen) })).sort((p, q) => q.rec - p.rec);
    const lab = rows.map((r) => `${D.techniques[r.t].label.replace(" masking", "")} · ${isJpeg(r.t) ? "Q" + r.dose : D.techniques[r.t].param + " " + fmt(r.dose, Number.isInteger(r.dose) ? 0 : 1)}`);
    const C = SLOTS();
    const mk = (key, name, color, sym = "circle") => ({ type: "scatter", mode: "markers", name, y: lab, x: rows.map((r) => 100 * r[key]),
      marker: { color, size: 11, symbol: sym, line: { color: css("--surface"), width: 1.5 } }, hovertemplate: "%{x:.1f}%<extra>" + name + "</extra>" });
    const lines = rows.map((r, i) => ({ type: "scatter", mode: "lines", x: [100 * r.frozen, 100 * r.ft], y: [lab[i], lab[i]], line: { color: css("--axis"), width: 3 }, hoverinfo: "skip", showlegend: false }));
    const ctrl = { type: "scatter", mode: "markers", name: "controle (fine-tuning nas originais)", y: lab, x: rows.map((r) => 100 * r.ctrl),
      marker: { color: css("--ink-2"), size: 16, symbol: "line-ns-open", line: { width: 2, color: css("--ink-2") } }, hovertemplate: "%{x:.1f}%<extra>controle</extra>" };
    plot("ad-chart", [...lines, ctrl, mk("frozen", "congelado", C[0]), mk("bn", "só re-estimar BatchNorm", C[1]), mk("ft", "fine-tuning (3 épocas)", C[2])],
      base({ hovermode: "y unified", margin: { l: 150, r: 14, t: 10, b: 46 }, legend: { orientation: "h", y: -0.18, x: 0, font: { size: 11 } },
        xaxis: Object.assign(base().xaxis, { showgrid: true, title: { text: "acurácia (%), média dos 3 modelos" } }),
        yaxis: Object.assign(base().yaxis, { autorange: "reversed", showgrid: false }) }));
  }

  // ---------- galeria ----------
  const GA = { ds: "imagenette", img: 0 };
  function gallery() {
    if (!D.datasets[GA.ds] || !D.datasets[GA.ds].samples) GA.ds = Object.keys(D.datasets).find((k) => D.datasets[k].samples);
    if (!GA.ds) { $("#amostras").remove(); return; }
    seg($("#ga-ds"), dsOptions().filter(([k]) => D.datasets[k].samples), GA.ds, (v) => { GA.ds = v; GA.img = 0; galleryImgs(); galleryRender(); });
    $("#ga-img").addEventListener("change", (e) => { GA.img = +e.target.value; galleryRender(); });
    galleryImgs(); galleryRender();
  }
  function galleryImgs() {
    const S = D.datasets[GA.ds].samples;
    $("#ga-img").replaceChildren(...S.images.map((_, i) => el("option", { value: String(i), text: `amostra ${i + 1}` })));
  }
  function galleryRender() {
    const ds = D.datasets[GA.ds], S = ds.samples, px = GA.ds === "cifar10" ? "px" : "";
    const cards = Object.keys(S.techniques).map((t) => {
      const lv = S.techniques[t].slice().sort((a, b) => toX(t, a.dose) - toX(t, b.dose));
      const pick = [lv[0], lv[Math.floor((lv.length - 1) / 2)], lv[lv.length - 1]];
      const meanAt = (x) => { const v = ds.models.map((m) => { const c = ds.curves[t][m.id]; const j = c.x.findIndex((z) => Math.abs(z - x) < 1e-6); return j >= 0 ? c.d[j] : null; }).filter((v) => v != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
      const figs = [el("figure", {}, el("img", { src: "samples/" + S.images[GA.img].original, alt: "original", class: px, loading: "lazy" }), el("figcaption", { text: "original" }))];
      for (const l of pick) {
        const x = toX(t, l.dose), dm = meanAt(x);
        figs.push(el("figure", {}, el("img", { src: "samples/" + l.files[GA.img], alt: `${D.techniques[t].label} ${doseLabel(t, x)}`, class: px, loading: "lazy" }),
          el("figcaption", {}, document.createTextNode(isJpeg(t) ? "Q" + fmt(l.dose, 0) : `${D.techniques[t].param} ${doseLabel(t, x)}`), el("br"),
            el("span", { class: dm > 0.002 ? "pos" : dm < -0.01 ? "neg" : "", text: dm == null ? "" : pp(dm, 1) }))));
      }
      return el("div", { class: "card gcard" }, el("h3", { text: D.techniques[t].label }), el("div", { class: "strip" }, ...figs));
    });
    $("#gallery").replaceChildren(...cards);
  }

  // ---------- tema ----------
  function theme() {
    const root = document.documentElement;
    try { const s = localStorage.getItem("theme"); if (s) root.dataset.theme = s; } catch (e) { /* sem storage */ }
    $("#theme").addEventListener("click", () => {
      const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* ok */ }
      if (D) renderCharts();
    });
  }
  function renderCharts() { exRender(false); hormesisRender(); cnnVit(); bandsRender(); predictors(); ire(); adaptation(); }

  async function main() {
    theme();
    try {
      D = await (await fetch("assets/data.json")).json();
    } catch (e) {
      $("#kpis").replaceChildren(el("p", { class: "loading", text: "Não foi possível carregar os dados (abra pelo GitHub Pages ou por um servidor local)." }));
      return;
    }
    kpis(); exInit(); heatmap(); hormesis(); cnnVit(); bands(); predictors(); ire(); adaptation(); gallery();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main); else main();
})();
