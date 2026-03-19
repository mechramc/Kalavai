#!/usr/bin/env python3
"""
Item 2: Per-Domain Monolithic Comparison Table
===============================================
Extracts per-domain losses (code, science, fiction) for base, best specialist
per domain, monolithic, and KALAVAI MoE from existing 410M result JSONs.

Produces:
  - LaTeX table for Section 4.3
  - Plain-text table for quick inspection
  - Analysis paragraph

Usage:
    python experiments/analysis/item2_perdomain_monolithic.py

Outputs:
    results/analysis/perdomain_monolithic.json
    (prints LaTeX table + analysis paragraph to stdout)
"""

import json
from pathlib import Path

# ── Load 410M corrected eval (seed 42) ────────────────────────────────────────
with open("results/pythia/corrected_eval_42.json") as f:
    d410 = json.load(f)

em = d410["eval_matrix"]  # keys: base, monolithic, code_spec, science_spec, fiction_spec, weight_avg, moe

# ── Build comparison table ─────────────────────────────────────────────────────
# "Best specialist per domain" = diagonal: code_spec on code, science_spec on sci, fict_spec on fiction
rows = {
    "Base model":          em["base"],
    "Code specialist":     em["code_spec"],
    "Science specialist":  em["science_spec"],
    "Fiction specialist":  em["fiction_spec"],
    "Monolithic (6k)":     em["monolithic"],
    "Weight averaging":    em["weight_avg"],
    "KALAVAI MoE":         em["moe"],
}

domains = ["code", "science", "fiction", "equal_weight_avg"]
dom_labels = ["Code", "Science", "Fiction", "EW Avg"]

print("=" * 80)
print("PER-DOMAIN LOSS TABLE (Pythia-410M, seed 42, corrected eval)")
print("=" * 80)
header = f"{'Method':<26s}  {'Code':>8s}  {'Science':>8s}  {'Fiction':>8s}  {'EW Avg':>8s}"
print(header)
print("-" * 76)

for name, row in rows.items():
    vals = [row[d] for d in domains]
    bold = name == "KALAVAI MoE"
    line = f"{'*' if bold else ' '}{name:<25s}  " + "  ".join(f"{v:>8.4f}" for v in vals)
    print(line)

# ── Key per-domain comparisons ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)

base     = em["base"]
mono     = em["monolithic"]
moe      = em["moe"]
code_s   = em["code_spec"]
sci_s    = em["science_spec"]
fict_s   = em["fiction_spec"]

for dom, spec in [("code", code_s), ("science", sci_s), ("fiction", fict_s)]:
    oracle   = spec[dom]          # best specialist on its own domain
    moe_d    = moe[dom]
    mono_d   = mono[dom]
    base_d   = base[dom]
    moe_vs_oracle = (oracle - moe_d) / oracle * 100
    moe_vs_mono   = (mono_d - moe_d) / mono_d * 100
    mono_vs_oracle= (oracle - mono_d) / oracle * 100
    print(f"\n  {dom.upper()}:")
    print(f"    Best specialist (own domain): {oracle:.4f}")
    print(f"    Monolithic:                   {mono_d:.4f}  ({'BETTER' if mono_d < oracle else 'worse':s} than specialist by {abs(mono_vs_oracle):.2f}%)")
    print(f"    KALAVAI MoE:                  {moe_d:.4f}  (vs. specialist: {moe_vs_oracle:+.2f}%;  vs. mono: {moe_vs_mono:+.2f}%)")
    print(f"    Base model:                   {base_d:.4f}")

print(f"\n  EQUAL-WEIGHT AVERAGE:")
print(f"    Base:        {base['equal_weight_avg']:.4f}")
print(f"    Monolithic:  {mono['equal_weight_avg']:.4f}  (+{(base['equal_weight_avg']-mono['equal_weight_avg'])/base['equal_weight_avg']*100:.2f}% vs base)")
print(f"    KALAVAI MoE: {moe['equal_weight_avg']:.4f}  (+{(base['equal_weight_avg']-moe['equal_weight_avg'])/base['equal_weight_avg']*100:.2f}% vs base;  +{(mono['equal_weight_avg']-moe['equal_weight_avg'])/mono['equal_weight_avg']*100:.2f}% vs mono)")

# ── Oracle (domain-level) ──────────────────────────────────────────────────────
oracle_code    = min(code_s["code"],    sci_s["code"],    fict_s["code"])
oracle_sci     = min(code_s["science"], sci_s["science"], fict_s["science"])
oracle_fiction = min(code_s["fiction"], sci_s["fiction"], fict_s["fiction"])
oracle_ew      = (oracle_code + oracle_sci + oracle_fiction) / 3

print(f"\n  DOMAIN-LEVEL ORACLE (min-specialist per domain):")
print(f"    Code:    {oracle_code:.4f}  (best: {'code_spec' if oracle_code == code_s['code'] else 'other'})")
print(f"    Science: {oracle_sci:.4f}  (best: {'sci_spec' if oracle_sci == sci_s['science'] else 'other'})")
print(f"    Fiction: {oracle_fiction:.4f}  (best: {'fict_spec' if oracle_fiction == fict_s['fiction'] else 'other'})")
print(f"    EW Avg:  {oracle_ew:.4f}")
print(f"    Oracle gap vs MoE EW: {(oracle_ew - moe['equal_weight_avg']):.6f}  ({(oracle_ew - moe['equal_weight_avg'])/oracle_ew*100:+.4f}%)")
print(f"    -> {'MoE IS the oracle (routing saturated)' if abs(oracle_ew - moe['equal_weight_avg']) < 0.001 else 'Routing headroom exists'}")

# ── LaTeX table ────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("LATEX TABLE (for Section 4.3):\n")

latex = r"""\begin{table}[t]
\centering
\caption{Per-domain held-out loss at Pythia-410M (seed 42, corrected evaluation:
per-domain separate eval at bs=4). \textbf{Bold} entries are the best value in
each column. The KALAVAI MoE matches the best individual specialist on every
domain simultaneously --- routing recovers the diagonal of the specialist
cross-domain matrix. The monolithic model trains on all domains (6k steps on
mixed data) and achieves strong per-domain performance, but underperforms the MoE
on code and science while outperforming the fiction specialist on fiction
(the monolithic sees as many fiction tokens as the fiction specialist but benefits
from cross-domain regularisation at the cost of domain focus).}
\label{tab:perdomain}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Code} $\downarrow$ & \textbf{Science} $\downarrow$ & \textbf{Fiction} $\downarrow$ & \textbf{EW Avg} $\downarrow$ \\
\midrule
"""

best = {}
for dom in ["code", "science", "fiction", "equal_weight_avg"]:
    best[dom] = min(rows[n][dom] for n in rows)

row_order = [
    ("Base model",         em["base"]),
    ("Code specialist",    em["code_spec"]),
    ("Science specialist", em["science_spec"]),
    ("Fiction specialist", em["fiction_spec"]),
    ("Monolithic (6k)",    em["monolithic"]),
    ("Weight averaging",   em["weight_avg"]),
    ("\\kalavai MoE",      em["moe"]),
]

for name, row in row_order:
    cells = []
    for dom in ["code", "science", "fiction", "equal_weight_avg"]:
        val = row[dom]
        cell = f"{val:.4f}"
        if abs(val - best[dom]) < 1e-5:
            cell = f"\\textbf{{{cell}}}"
        cells.append(cell)
    latex += f"{name} & {' & '.join(cells)} \\\\\n"
    if name == "Fiction specialist":
        latex += "\\midrule\n"

latex += r"""\bottomrule
\end{tabular}
\end{table}"""
print(latex)

# ── Draft analysis paragraph ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print("DRAFT PARAGRAPH (Section 4.3, after Table X):\n")

moe_vs_mono_code  = (mono["code"]    - moe["code"])    / mono["code"]    * 100
moe_vs_mono_sci   = (mono["science"] - moe["science"]) / mono["science"] * 100
mono_vs_spec_fict = (fict_s["fiction"] - mono["fiction"]) / fict_s["fiction"] * 100

print(f"""Table~\\ref{{tab:perdomain}} decomposes the equal-weight average into per-domain losses,
revealing the structural advantage of cooperative fusion. The KALAVAI MoE matches
the per-domain performance of the best individual specialist on every domain
simultaneously: code loss {moe['code']:.4f} (vs.~code specialist {code_s['code']:.4f},
$\\Delta$={abs(moe['code']-code_s['code']):.4f}), science loss {moe['science']:.4f}
(vs.~science specialist {sci_s['science']:.4f}), fiction loss {moe['fiction']:.4f}
(vs.~fiction specialist {fict_s['fiction']:.4f}). This near-exact recovery of each
specialist's own-domain performance confirms that routing is effectively saturated:
the domain-level oracle (assigning each domain to its optimal specialist) achieves
EW loss {oracle_ew:.4f}, a difference of only {abs(oracle_ew - moe['equal_weight_avg']):.6f}
from the actual MoE ({moe['equal_weight_avg']:.4f}).

The monolithic model, trained for 6,000 steps on mixed data, achieves strong per-domain
performance but underperforms the MoE on code ({mono['code']:.4f} vs.~{moe['code']:.4f},
${moe_vs_mono_code:+.2f}\\%$) and science ({mono['science']:.4f} vs.~{moe['science']:.4f},
${moe_vs_mono_sci:+.2f}\\%$). The monolithic does achieve lower fiction loss than the
fiction specialist ({mono['fiction']:.4f} vs.~{fict_s['fiction']:.4f},
${mono_vs_spec_fict:+.2f}\\%$ improvement) --- the monolithic model sees as many fiction
tokens as the fiction specialist but benefits from cross-domain regularisation that
prevents the catastrophic forgetting observed in isolated specialist training.
Despite this, the MoE's equal-weight average ({moe['equal_weight_avg']:.4f}) exceeds
the monolithic ({mono['equal_weight_avg']:.4f}) because the per-domain advantages
on code and science outweigh the fiction deficit. Critically, no individual
contributor in the cooperative trains on all domains --- each contributor trains only
on their own data, never sharing it --- yet the collective result matches or exceeds
the model that had access to all data simultaneously.
""")

# ── Save JSON ──────────────────────────────────────────────────────────────────
Path("results/analysis").mkdir(parents=True, exist_ok=True)
out = {
    "per_domain": {name: {d: row[d] for d in domains} for name, row in rows.items()},
    "oracle": {
        "code": oracle_code, "science": oracle_sci, "fiction": oracle_fiction,
        "equal_weight_avg": oracle_ew,
        "gap_vs_moe": oracle_ew - moe["equal_weight_avg"],
        "routing_saturated": abs(oracle_ew - moe["equal_weight_avg"]) < 0.001,
    },
}
with open("results/analysis/perdomain_monolithic.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Results saved: results/analysis/perdomain_monolithic.json")
