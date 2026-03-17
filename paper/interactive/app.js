const data = {
  overviewMetrics: [
    { label: "410M vs best specialist", value: "+14.18%" },
    { label: "1B vs best specialist", value: "+14.82%" },
    { label: "6.9B corrected result", value: "+2.43%" },
    { label: "Monolithic gap", value: "+14.55%" }
  ],
  claims: [
    {
      kicker: "Condition 1",
      title: "Shared initialization is the structural constraint",
      body:
        "The paper frames a shared starting checkpoint as the one hard coordination requirement. A direct ablation script exists in the repo, but its result JSON is not yet committed."
    },
    {
      kicker: "Condition 2",
      title: "Frozen layers become necessary late, not early",
      body:
        "At short horizons, freezing costs performance. Around 10k specialist steps, the trend reverses and frozen layers become the safer regime."
    },
    {
      kicker: "Condition 3",
      title: "Joint inference is the mechanism, not single-expert dispatch",
      body:
        "The fused model improves only when all specialists run. A single-expert routing baseline drops below base performance despite 99.3% classification accuracy."
    }
  ],
  scales: [
    {
      id: "410m",
      name: "Pythia-410M",
      summary:
        "This is the best-supported scale in the repo: three seeds, main result JSON, monolithic baseline, freeze and router ablations, scaling tests, dynamics, and benchmark artifacts.",
      stats: [
        { label: "Base mixed loss", value: "2.248" },
        { label: "Best specialist", value: "2.089" },
        { label: "MoE fused", value: "1.793" },
        { label: "Gain vs best specialist", value: "+14.18%" },
        { label: "Gain vs base", value: "+20.2%" },
        { label: "Gain vs monolithic", value: "+14.55%" }
      ],
      losses: [
        { label: "Base", value: 2.248017, color: "#9e3d32" },
        { label: "Best specialist", value: 2.089038, color: "#b17025" },
        { label: "Monolithic", value: 2.098324, color: "#5b4567" },
        { label: "Weight avg", value: 2.157677, color: "#40624d" },
        { label: "MoE", value: 1.79309, color: "#1f6f78" }
      ]
    },
    {
      id: "1b",
      name: "Pythia-1B",
      summary:
        "The 1B result is as strong as the 410M result on mixed loss. The repo also includes maturity sweeps and benchmark outputs, but fewer auxiliary baselines than at 410M.",
      stats: [
        { label: "Base mixed loss", value: "2.160" },
        { label: "Best specialist", value: "1.992" },
        { label: "MoE fused", value: "1.696" },
        { label: "Gain vs best specialist", value: "+14.82%" },
        { label: "Gain vs base", value: "+21.5%" },
        { label: "Seeds", value: "3" }
      ],
      losses: [
        { label: "Base", value: 2.159901, color: "#9e3d32" },
        { label: "Best specialist", value: 1.991667, color: "#b17025" },
        { label: "Weight avg", value: 2.069087, color: "#40624d" },
        { label: "MoE", value: 1.696355, color: "#1f6f78" }
      ]
    },
    {
      id: "6b",
      name: "Pythia-6.9B",
      summary:
        "The paper presents a corrected, seeded 6.9B evaluation around +2.43%. An older committed summary file still contains unstable pre-fix numbers, so the paper text is the source of truth for the headline claim.",
      stats: [
        { label: "Base mixed loss", value: "2.700" },
        { label: "Best specialist", value: "2.634" },
        { label: "MoE fused", value: "2.570" },
        { label: "Gain vs best specialist", value: "+2.43%" },
        { label: "Gain vs base", value: "+4.8%" },
        { label: "Caveat", value: "corrected eval" }
      ],
      losses: [
        { label: "Base", value: 2.7, color: "#9e3d32" },
        { label: "Best specialist", value: 2.634, color: "#b17025" },
        { label: "MoE", value: 2.57, color: "#1f6f78" }
      ]
    }
  ],
  crossover: {
    x: [500, 1000, 2000, 5000, 10000, 20000],
    series: [
      { name: "freeze = 0", values: [9.9044, 12.4935, 15.1487, 16.359, 15.4182, 13.5621], color: "#b17025" },
      { name: "freeze = 4", values: [8.8889, 11.2502, 13.8606, 15.7708, 15.6252, 14.8095], color: "#1f6f78" }
    ]
  },
  freezeSweep: {
    labels: ["0", "2", "4", "6", "8", "12"],
    values: [14.8538, 14.5064, 14.1778, 13.7573, 13.2896, 12.3648]
  },
  router: {
    labels: ["Uniform", "Linear", "2-layer"],
    values: [6.6631, 14.1582, 14.1673]
  },
  failure: {
    labels: ["MoE", "Single-head dispatch"],
    values: [14.0676, -21.1267]
  },
  scaling: {
    labels: ["2 experts", "3 experts", "4 experts", "5 experts"],
    values: [17.7299, 14.1479, 14.1426, 14.1158]
  },
  maturity: {
    x: [3.5, 7, 14, 35, 70, 100],
    series: [
      { name: "410M", values: [14.97, 14.1355, 13.3692, 13.4764, 14.4304, 14.6469], color: "#1f6f78" },
      { name: "1B", values: [15.8521, 14.8274, 13.9714, 13.7804, null, 14.7494], color: "#5b4567" },
      { name: "Qwen-1.5B", values: [null, null, null, null, null, -0.9653], color: "#9e3d32" }
    ]
  },
  benchmarkGroups: [
    {
      id: "1b",
      title: "Pythia-1B downstream results",
      summary:
        "The 1B benchmark results support the paper’s caution: mixed-loss gains do not automatically become benchmark wins. MoE leads on HellaSwag, but average accuracy remains slightly below base.",
      columns: ["Model", "HellaSwag", "ARC-Easy", "LAMBADA", "SciQ", "WinoGrande", "Average"],
      rows: [
        ["Base", "34.4", "40.4", "60.2", "68.4", "49.6", "50.6"],
        ["Code specialist", "34.2", "39.4", "57.4", "65.8", "50.0", "49.36"],
        ["Science specialist", "34.2", "41.0", "56.4", "65.8", "48.2", "49.12"],
        ["Fiction specialist", "34.4", "39.8", "58.6", "66.8", "48.2", "49.56"],
        ["Weight avg", "34.6", "39.0", "57.8", "67.8", "48.6", "49.56"],
        ["MoE", "35.0", "40.0", "59.0", "64.8", "49.4", "49.64"],
        ["Monolithic", "33.4", "38.4", "58.2", "67.0", "49.4", "49.28"]
      ],
      notes: [
        { tone: "plum", text: "MoE’s clearest win is HellaSwag: 35.0 vs 34.4 for base." },
        { tone: "gold", text: "Average accuracy is still below base: 49.64 vs 50.6." },
        { tone: "danger", text: "This is why the paper explicitly refuses to oversell downstream task improvement." }
      ]
    },
    {
      id: "6b",
      title: "Pythia-6.9B downstream results",
      summary:
        "The 6.9B benchmarks are small but directionally positive: the MoE variant beats base on four of five tasks, with a modest average advantage.",
      columns: ["Model", "HellaSwag", "ARC-Easy", "LAMBADA", "SciQ", "WinoGrande", "Average"],
      rows: [
        ["Base", "35.4", "43.6", "61.2", "66.8", "51.0", "51.6"],
        ["MoE", "35.6", "45.2", "62.8", "67.8", "49.4", "52.16"]
      ],
      notes: [
        { tone: "sage", text: "Average accuracy improves by about +0.56 points: 52.16 vs 51.6." },
        { tone: "gold", text: "The one drop is WinoGrande: 49.4 vs 51.0." },
        { tone: "teal", text: "These results still fit the paper’s framing: modest downstream gains, clearer perplexity gains." }
      ]
    }
  ],
  boundaryNotes: [
    { tone: "danger", text: "Qwen-1.5B divergent-domain experiment: mean improvement -0.9653%, std 0.0135 across 3 seeds." },
    { tone: "gold", text: "The repo shows specialists still diverge on Qwen, but the router cannot turn that into a positive mixed result." },
    { tone: "plum", text: "This page treats Qwen as a boundary condition, not a footnote: the method appears sensitive to how saturated the base model already is on target domains." }
  ],
  figures: [
    { file: "../figures/fig_hero_4panel.png", title: "Hero 4-panel summary", category: "core", caption: "The manuscript’s compact summary figure: scale, crossover, routing failure, and monolithic comparison." },
    { file: "../figures/fig_paper_hero.png", title: "Paper hero", category: "core", caption: "Alternative hero treatment used in the paper assets." },
    { file: "../figures/fig_scale_ladder.png", title: "Scale ladder", category: "core", caption: "Visual comparison across 410M, 1B, and 6.9B." },
    { file: "../figures/fig_6b_summary.png", title: "6.9B summary", category: "core", caption: "Condensed view of the large-scale result and related diagnostics." },
    { file: "../figures/fig_training_duration_crossover.png", title: "Training duration crossover", category: "dynamics", caption: "The crossover where frozen layers become useful." },
    { file: "../figures/fig_ablation_freeze.png", title: "Freeze ablation", category: "dynamics", caption: "Improvement versus freeze depth at short specialist training horizons." },
    { file: "../figures/fig_ablation_router.png", title: "Router ablation", category: "routing", caption: "Uniform, linear, and 2-layer router comparison." },
    { file: "../figures/fig_router_distribution.png", title: "Router distribution", category: "routing", caption: "Near-one-hot routing for domain-aligned evaluation inputs." },
    { file: "../figures/fig_domain_classifier.png", title: "Domain classifier baseline", category: "routing", caption: "Single-specialist dispatch fails even with a strong classifier." },
    { file: "../figures/fig_multihead_baseline.png", title: "Multihead hard routing baseline", category: "routing", caption: "Hard dispatch is much worse than the MoE setup." },
    { file: "../figures/fig_hybrid_routing_0.png", title: "Hybrid routing prompt 0", category: "routing", caption: "Token-level gate switching on a hybrid prompt." },
    { file: "../figures/fig_hybrid_routing_1.png", title: "Hybrid routing prompt 1", category: "routing", caption: "Additional hybrid prompt routing heatmap." },
    { file: "../figures/fig_hybrid_routing_2.png", title: "Hybrid routing prompt 2", category: "routing", caption: "The paper’s strongest token-level routing illustration." },
    { file: "../figures/fig_hybrid_routing_3.png", title: "Hybrid routing prompt 3", category: "routing", caption: "Hybrid prompt example across code, science, and fiction experts." },
    { file: "../figures/fig_hybrid_routing_4.png", title: "Hybrid routing prompt 4", category: "routing", caption: "Fifth routing heatmap in the appendix assets." },
    { file: "../figures/fig_fusion_comparison.png", title: "410M fusion comparison", category: "scaling", caption: "Main 410M comparison across base, specialists, and fusion." },
    { file: "../figures/fig_1b_fusion_comparison.png", title: "1B fusion comparison", category: "scaling", caption: "The same comparison at 1B." },
    { file: "../figures/fig_monolithic_comparison.png", title: "Monolithic comparison", category: "scaling", caption: "Equal-compute monolithic training versus KALAVAI." },
    { file: "../figures/fig_monolithic_trajectory.png", title: "Monolithic trajectory", category: "scaling", caption: "How the monolithic baseline evolves over training." },
    { file: "../figures/fig_wider_model_baseline.png", title: "Wider model baseline", category: "scaling", caption: "Parameter count alone is not the explanation." },
    { file: "../figures/fig_specialist_scaling.png", title: "Specialist-count scaling", category: "scaling", caption: "2 to 5 specialist scaling experiment." },
    { file: "../figures/fig_maturity_curve_410m.png", title: "410M maturity curve", category: "maturity", caption: "Fusion benefit across Pythia-410M checkpoints." },
    { file: "../figures/fig_maturity_curve_1b.png", title: "1B maturity curve", category: "maturity", caption: "Fusion benefit across Pythia-1B checkpoints." },
    { file: "../figures/fig_maturity_curve_combined.png", title: "Combined maturity curve", category: "maturity", caption: "410M, 1B, and Qwen on one chart." },
    { file: "../figures/fig_maturity_curve_comparison_bar.png", title: "Maturity comparison bar chart", category: "maturity", caption: "Cross-model maturity summary view." },
    { file: "../figures/fig_training_curves_seed42.png", title: "410M training curves", category: "dynamics", caption: "Per-domain training dynamics for the 410M specialists." },
    { file: "../figures/fig_1b_training_curves_seed42.png", title: "1B training curves", category: "dynamics", caption: "Per-domain training dynamics for the 1B specialists." },
    { file: "../figures/fig_specialist_own_domain.png", title: "Own-domain specialist curves", category: "dynamics", caption: "Each specialist improves on its own domain." },
    { file: "../figures/fig_specialist_cross_domain.png", title: "Cross-domain specialist curves", category: "dynamics", caption: "Each specialist degrades on non-specialist domains." },
    { file: "../figures/fig_fusion_trajectory.png", title: "Fusion trajectory", category: "dynamics", caption: "MoE gain over the best specialist as training progresses." },
    { file: "../figures/fig_divergence_heatmap.png", title: "410M divergence heatmap", category: "analysis", caption: "Cross-domain evaluation matrix showing complementary specialist drift." },
    { file: "../figures/fig_1b_divergence_heatmap.png", title: "1B divergence heatmap", category: "analysis", caption: "The same diagnostic at 1B." },
    { file: "../figures/fig_benchmarks.png", title: "410M benchmarks", category: "benchmarks", caption: "Downstream results at 410M." },
    { file: "../figures/fig_benchmarks_1b.png", title: "1B benchmarks", category: "benchmarks", caption: "Downstream results at 1B." },
    { file: "../figures/fig_1b_router_distribution.png", title: "1B router distribution", category: "routing", caption: "Near-deterministic routing also appears at 1B." }
  ],
  artifacts: [
    { path: "paper/kalavai_neurips2026.tex", body: "Primary narrative source. Used for section structure, captions, and interpretation." },
    { path: "paper/kalavai_neurips2026.pdf", body: "Archival PDF for the formatted paper." },
    { path: "results/pythia/step5_final_summary.json", body: "410M main result with per-seed fusion metrics and held-out losses." },
    { path: "results/pythia/monolithic_baseline_summary.json", body: "Equal-compute monolithic baseline summary." },
    { path: "results/pythia/ablation_freeze_summary.json", body: "Freeze-depth sweep and short-horizon optimum." },
    { path: "results/pythia/ablation_router_summary.json", body: "Uniform, linear, and 2-layer router comparison." },
    { path: "results/pythia/training_duration_crossover.json", body: "Step sweep showing when frozen layers start to help." },
    { path: "results/pythia/five_domain/summary.json", body: "2-to-5 specialist scaling study." },
    { path: "results/pythia/maturity_sweep_410m/summary.json", body: "410M maturity sweep across multiple Pythia checkpoints." },
    { path: "results/pythia/pythia_1b/main_result_summary.json", body: "1B main result summary." },
    { path: "results/pythia/pythia_1b/maturity_sweep/summary.json", body: "1B maturity sweep." },
    { path: "results/pythia/pythia_1b/benchmarks_seed42.json", body: "1B downstream task results." },
    { path: "results/pythia_6b/benchmarks_seed42.json", body: "6.9B downstream task results." },
    { path: "results/real/qwen_divergent_domains.json", body: "Qwen negative-result boundary condition." },
    { path: "paper/figures/", body: "Full figure inventory used to build the atlas." }
  ],
  issues: [
    { tone: "danger", text: "The paper claims shared-initialization necessity, and the repo contains `experiments/kalavai_shared_init_ablation.py`, but I did not find `results/pythia/shared_init_ablation/*.json` committed yet." },
    { tone: "gold", text: "The committed file `results/pythia_6b/summary.json` still contains unstable pre-fix numbers. The paper text and benchmark file indicate a corrected seeded evaluation; that discrepancy should be cleaned before publication." },
    { tone: "teal", text: "This page is a standalone static site. If you want an actual TMLR Beyond-PDF submission package next, the content model here is already aligned to that format and can be ported into `submission.md` plus embedded HTML widgets." }
  ]
};

function createPills(containerId, items, activeId, onClick) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  items.forEach((item) => {
    const button = document.createElement("button");
    button.className = `pill${item.id === activeId ? " is-active" : ""}`;
    button.textContent = item.name || item.title || item.label;
    button.addEventListener("click", () => onClick(item.id));
    container.appendChild(button);
  });
}

function renderHeroMetrics() {
  const host = document.getElementById("hero-metrics");
  data.overviewMetrics.forEach((metric) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<div class="metric-card__label">${metric.label}</div><div class="metric-card__value">${metric.value}</div>`;
    host.appendChild(card);
  });
}

function renderClaims() {
  const host = document.getElementById("claim-grid");
  data.claims.forEach((claim) => {
    const card = document.createElement("article");
    card.className = "claim-card";
    card.innerHTML = `
      <div class="claim-card__kicker">${claim.kicker}</div>
      <h3>${claim.title}</h3>
      <p>${claim.body}</p>
    `;
    host.appendChild(card);
  });
}

function renderScale(id) {
  const scale = data.scales.find((item) => item.id === id) || data.scales[0];
  createPills("scale-tabs", data.scales, scale.id, renderScale);
  document.getElementById("scale-title").textContent = scale.name;
  document.getElementById("scale-summary").textContent = scale.summary;

  const stats = document.getElementById("scale-stats");
  stats.innerHTML = "";
  scale.stats.forEach((stat) => {
    const card = document.createElement("div");
    card.className = "stat";
    card.innerHTML = `<div class="stat__label">${stat.label}</div><div class="stat__value">${stat.value}</div>`;
    stats.appendChild(card);
  });

  const max = Math.max(...scale.losses.map((entry) => entry.value));
  const bars = document.getElementById("loss-bars");
  bars.innerHTML = "";
  scale.losses.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-row__label">${entry.label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(entry.value / max) * 100}%;background:${entry.color};"></div></div>
      <div>${entry.value.toFixed(3)}</div>
    `;
    bars.appendChild(row);
  });
}

function svgWrap(inner, width = 620, height = 260) {
  return `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true">${inner}</svg>`;
}

function lineChart({ labels, series, yMin, yMax, percent = true }) {
  const width = 620;
  const height = 280;
  const margin = { top: 18, right: 22, bottom: 42, left: 44 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xAt = (index) => margin.left + (plotW * index) / (labels.length - 1);
  const yAt = (value) => margin.top + plotH - ((value - yMin) / (yMax - yMin)) * plotH;

  let inner = "";
  for (let i = 0; i < 5; i += 1) {
    const yValue = yMin + ((yMax - yMin) * i) / 4;
    const y = yAt(yValue);
    inner += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${y}" y2="${y}" stroke="rgba(29,27,24,0.12)"></line>`;
    inner += `<text class="tick" x="${margin.left - 8}" y="${y + 4}" text-anchor="end">${percent ? yValue.toFixed(1) + "%" : yValue.toFixed(1)}</text>`;
  }

  labels.forEach((label, index) => {
    inner += `<text class="tick" x="${xAt(index)}" y="${height - 12}" text-anchor="middle">${label}</text>`;
  });

  series.forEach((entry, entryIndex) => {
    const points = entry.values
      .map((value, index) => (value == null ? null : `${xAt(index)},${yAt(value)}`))
      .filter(Boolean)
      .join(" ");
    inner += `<polyline fill="none" stroke="${entry.color}" stroke-width="4" points="${points}"></polyline>`;
    entry.values.forEach((value, index) => {
      if (value == null) return;
      inner += `<circle cx="${xAt(index)}" cy="${yAt(value)}" r="4.5" fill="${entry.color}"></circle>`;
    });
    inner += `<text class="legend-text" x="${margin.left + 8}" y="${24 + entryIndex * 18}" fill="${entry.color}">${entry.name}</text>`;
  });

  return svgWrap(inner, width, height);
}

function barChart({ labels, values, colors, yMin, yMax, percent = true }) {
  const width = 620;
  const height = 280;
  const margin = { top: 18, right: 18, bottom: 58, left: 44 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const barWidth = plotW / labels.length * 0.58;
  const yAt = (value) => margin.top + plotH - ((value - yMin) / (yMax - yMin)) * plotH;
  let inner = "";

  for (let i = 0; i < 5; i += 1) {
    const yValue = yMin + ((yMax - yMin) * i) / 4;
    const y = yAt(yValue);
    inner += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${y}" y2="${y}" stroke="rgba(29,27,24,0.12)"></line>`;
    inner += `<text class="tick" x="${margin.left - 8}" y="${y + 4}" text-anchor="end">${percent ? yValue.toFixed(1) + "%" : yValue.toFixed(1)}</text>`;
  }

  labels.forEach((label, index) => {
    const step = plotW / labels.length;
    const x = margin.left + index * step + (step - barWidth) / 2;
    const value = values[index];
    const y = yAt(Math.max(value, yMin));
    const zero = yAt(Math.max(0, yMin));
    const h = Math.abs(zero - y);
    inner += `<rect x="${x}" y="${value >= 0 ? y : zero}" width="${barWidth}" height="${h}" rx="14" fill="${colors[index]}"></rect>`;
    inner += `<text class="tick" x="${x + barWidth / 2}" y="${height - 30}" text-anchor="middle">${label}</text>`;
    inner += `<text class="legend-text" x="${x + barWidth / 2}" y="${value >= 0 ? y - 8 : zero + h + 16}" text-anchor="middle">${value.toFixed(2)}${percent ? "%" : ""}</text>`;
  });

  return svgWrap(inner, width, height);
}

function renderCharts() {
  document.getElementById("crossover-chart").innerHTML = lineChart({
    labels: data.crossover.x.map((v) => `${v}`),
    series: data.crossover.series,
    yMin: 8,
    yMax: 17
  });

  document.getElementById("freeze-chart").innerHTML = barChart({
    labels: data.freezeSweep.labels.map((v) => `${v} layers`),
    values: data.freezeSweep.values,
    colors: data.freezeSweep.values.map(() => "#b17025"),
    yMin: 12,
    yMax: 15.5
  });

  document.getElementById("router-chart").innerHTML = barChart({
    labels: data.router.labels,
    values: data.router.values,
    colors: ["#40624d", "#1f6f78", "#5b4567"],
    yMin: 5,
    yMax: 15
  });

  document.getElementById("failure-chart").innerHTML = barChart({
    labels: data.failure.labels,
    values: data.failure.values,
    colors: ["#1f6f78", "#9e3d32"],
    yMin: -25,
    yMax: 16
  });

  document.getElementById("scaling-chart").innerHTML = barChart({
    labels: data.scaling.labels,
    values: data.scaling.values,
    colors: ["#b17025", "#1f6f78", "#5b4567", "#40624d"],
    yMin: 13.5,
    yMax: 18.2
  });

  document.getElementById("maturity-chart").innerHTML = lineChart({
    labels: data.maturity.x.map((v) => `${v}%`),
    series: data.maturity.series,
    yMin: -2,
    yMax: 17
  });
}

function renderBoundaryNotes() {
  const host = document.getElementById("boundary-notes");
  data.boundaryNotes.forEach((item) => {
    const div = document.createElement("div");
    div.className = `note note--${item.tone}`;
    div.textContent = item.text;
    host.appendChild(div);
  });
}

function renderBenchmark(id) {
  const group = data.benchmarkGroups.find((item) => item.id === id) || data.benchmarkGroups[0];
  createPills("benchmark-tabs", data.benchmarkGroups, group.id, renderBenchmark);
  document.getElementById("benchmark-title").textContent = group.title;
  document.getElementById("benchmark-summary").textContent = group.summary;

  const tableHost = document.getElementById("benchmark-table");
  let html = `<table class="table"><thead><tr>`;
  group.columns.forEach((column) => { html += `<th>${column}</th>`; });
  html += `</tr></thead><tbody>`;
  group.rows.forEach((row) => {
    html += "<tr>";
    row.forEach((cell) => { html += `<td>${cell}</td>`; });
    html += "</tr>";
  });
  html += "</tbody></table>";
  tableHost.innerHTML = html;

  const notesHost = document.getElementById("benchmark-notes");
  notesHost.innerHTML = "";
  group.notes.forEach((note) => {
    const div = document.createElement("div");
    div.className = `note note--${note.tone}`;
    div.textContent = note.text;
    notesHost.appendChild(div);
  });
}

function renderArtifacts() {
  const host = document.getElementById("artifact-list");
  data.artifacts.forEach((artifact) => {
    const li = document.createElement("li");
    li.className = "artifact-item";
    li.innerHTML = `<div class="artifact-item__path">${artifact.path}</div><div>${artifact.body}</div>`;
    host.appendChild(li);
  });

  const issueHost = document.getElementById("issue-list");
  data.issues.forEach((issue) => {
    const div = document.createElement("div");
    div.className = `note note--${issue.tone}`;
    div.textContent = issue.text;
    issueHost.appendChild(div);
  });
}

function renderFigureDetail(figure) {
  const host = document.getElementById("figure-detail");
  host.innerHTML = `
    <img src="${figure.file}" alt="${figure.title}">
    <div class="figure-detail__meta">${figure.category}</div>
    <h3>${figure.title}</h3>
    <p class="figure-caption">${figure.caption}</p>
    <div class="artifact-item__path">${figure.file.replace("../", "paper/")}</div>
  `;
}

function renderFigures(category = "all") {
  const categories = [
    { id: "all", name: "All" },
    { id: "core", name: "Core" },
    { id: "dynamics", name: "Dynamics" },
    { id: "routing", name: "Routing" },
    { id: "scaling", name: "Scaling" },
    { id: "maturity", name: "Maturity" },
    { id: "analysis", name: "Analysis" },
    { id: "benchmarks", name: "Benchmarks" }
  ];
  createPills("figure-filters", categories, category, renderFigures);

  const visible = category === "all" ? data.figures : data.figures.filter((figure) => figure.category === category);
  const grid = document.getElementById("figure-grid");
  grid.innerHTML = "";

  visible.forEach((figure, index) => {
    const card = document.createElement("article");
    card.className = "figure-card";
    card.innerHTML = `
      <img src="${figure.file}" alt="${figure.title}">
      <div class="figure-card__body">
        <h3 class="figure-card__title">${figure.title}</h3>
        <div class="figure-detail__meta">${figure.category}</div>
      </div>
    `;
    card.addEventListener("click", () => renderFigureDetail(figure));
    grid.appendChild(card);
    if (index === 0) renderFigureDetail(figure);
  });
}

function init() {
  renderHeroMetrics();
  renderClaims();
  renderScale("410m");
  renderCharts();
  renderBoundaryNotes();
  renderBenchmark("1b");
  renderFigures("all");
  renderArtifacts();
}

init();
