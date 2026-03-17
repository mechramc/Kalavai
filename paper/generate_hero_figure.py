#!/usr/bin/env python3
"""
Generate the KALAVAI paper hero figure (4-panel).

Panel A: Scale comparison bars (410M, 1B, 6.9B)
Panel B: Training duration crossover
Panel C: Routing failure comparison (MoE vs Classifier vs Multi-head)
Panel D: Monolithic comparison (Base / Monolithic / Best Spec / Weight Avg / MoE)
"""
import json
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from figure_style import apply_style, COLORS, clean_axes, label_bars

apply_style()

# ── result paths ──────────────────────────────────────────────────────────────
RESULTS = REPO_ROOT / "results" / "pythia"
RESULTS_6B = REPO_ROOT / "results" / "pythia_6b"

# ── load data ─────────────────────────────────────────────────────────────────
with open(RESULTS / "step5_final_summary.json") as f:
    data_410m = json.load(f)

with open(RESULTS / "pythia_1b" / "main_result_summary.json") as f:
    data_1b = json.load(f)

with open(RESULTS_6B / "step6_fusion_seed42.json") as f:
    seed42_6b = json.load(f)
with open(RESULTS_6B / "step6_fusion_seed137.json") as f:
    seed137_6b = json.load(f)
with open(RESULTS_6B / "step6_fusion_seed2026.json") as f:
    seed2026_6b = json.load(f)

with open(RESULTS / "training_duration_crossover.json") as f:
    crossover_data = json.load(f)

with open(RESULTS / "domain_classifier_baseline.json") as f:
    classifier_data = json.load(f)

with open(RESULTS / "multihead_baseline.json") as f:
    multihead_data = json.load(f)

with open(RESULTS / "monolithic_baseline_summary.json") as f:
    monolithic_data = json.load(f)

# ── compute derived values ────────────────────────────────────────────────────

# Panel A: scale comparison
imp_410m_mean = data_410m["summary"]["improvement_mean_pct"]
imp_410m_std  = data_410m["summary"]["improvement_std_pct"]

imp_1b_mean = data_1b["summary"]["improvement_mean_pct"]
imp_1b_std  = data_1b["summary"]["improvement_std_pct"]

imps_6b = [seed42_6b["improvement_pct"], seed137_6b["improvement_pct"], seed2026_6b["improvement_pct"]]
imp_6b_mean = float(np.mean(imps_6b))
imp_6b_std  = float(np.std(imps_6b, ddof=1))

# Panel D: monolithic comparison — from step5 seed means and monolithic
# Use seed 42 as representative (all seeds produce nearly identical results)
seed_key = "42"
fusion42 = data_410m["per_seed_fusion"][seed_key]["eval_heldout"]

base_loss        = fusion42["base"]["mixed"]
best_spec_loss   = min(
    fusion42["code_spec"]["mixed"],
    fusion42["science_spec"]["mixed"],
    fusion42["fiction_spec"]["mixed"],
)
weight_avg_loss  = fusion42["weight_avg"]["mixed"]
moe_loss_410m    = fusion42["moe"]["mixed"]
monolithic_loss  = monolithic_data["results"]["mean"]["monolithic_mixed"]

# ── figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("KALAVAI: Four Key Findings", fontsize=16, fontweight="bold", y=1.01)

ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

# ─────────────────────────────────────────────────────────────────────────────
# Panel A: Scale Comparison
# ─────────────────────────────────────────────────────────────────────────────
labels_a = ["410M", "1B", "6.9B"]
means_a  = [imp_410m_mean, imp_1b_mean, imp_6b_mean]
stds_a   = [imp_410m_std,  imp_1b_std,  imp_6b_std]
x_a = np.arange(len(labels_a))

bars_a = ax_a.bar(
    x_a, means_a,
    yerr=stds_a, capsize=6,
    color=COLORS["moe"], width=0.55, zorder=3,
    error_kw={"elinewidth": 1.5, "ecolor": "#374151"},
)

ax_a.set_xticks(x_a)
ax_a.set_xticklabels(labels_a)
ax_a.set_ylabel("Improvement over best specialist (%)")
ax_a.set_title("A.  Scale Comparison: KALAVAI MoE vs Best Specialist")
clean_axes(ax_a)

# label bars with +X.X%
for bar, mean in zip(bars_a, means_a):
    ax_a.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(stds_a) * 1.1 + 0.1,
        f"+{mean:.1f}%",
        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1f2937",
    )

ax_a.set_ylim(0, max(means_a) + max(stds_a) * 1.5 + 1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: Training Duration Crossover
# ─────────────────────────────────────────────────────────────────────────────
steps      = crossover_data["steps"]
freeze0    = crossover_data["freeze0_improvement"]
freeze4    = crossover_data["freeze4_improvement"]
crossover  = crossover_data["crossover_steps"]

ax_b.plot(steps, freeze0, color=COLORS["freeze0"], lw=2.0, marker="o", markersize=5,
          label="Freeze=0 (no anchor)")
ax_b.plot(steps, freeze4, color=COLORS["freeze4"], lw=2.0, marker="s", markersize=5,
          label="Freeze=4 layers")

ax_b.axvline(crossover, color=COLORS["crossover"], linestyle="--", lw=1.5, alpha=0.8)
y_annot = min(min(freeze0), min(freeze4)) + 0.5
ax_b.text(crossover * 1.12, y_annot,
          f"Crossover\n~{crossover // 1000}k steps",
          color=COLORS["crossover"], fontsize=9, va="bottom")

ax_b.set_xscale("log")
ax_b.set_xlabel("Training steps (log scale)")
ax_b.set_ylabel("MoE improvement over base (%)")
ax_b.set_title("B.  Training Duration Crossover: Freeze=0 vs Freeze=4")
ax_b.legend(loc="upper left")
clean_axes(ax_b)

# ─────────────────────────────────────────────────────────────────────────────
# Panel C: Routing Failure Comparison
# ─────────────────────────────────────────────────────────────────────────────
routing_labels = [
    "MoE (ours)",
    "Classifier dispatch\n(99.3% acc.)",
    "Multi-head\n(same params)",
]
routing_values = [
    classifier_data["moe_improvement_pct"],
    classifier_data["classifier_improvement_pct"],
    multihead_data["multihead_improvement_pct"],
]
routing_colors = [COLORS["moe"], COLORS["classifier"], COLORS["multihead"]]

x_c = np.arange(len(routing_labels))
bars_c = ax_c.bar(
    x_c, routing_values,
    color=routing_colors, width=0.55, zorder=3,
)

ax_c.axhline(0, color="#374151", lw=1.0, linestyle="-")
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(routing_labels)
ax_c.set_ylabel("Improvement vs. base (%)")
ax_c.set_title("C.  Routing Failure: Why Architecture Matters")

# accommodate negative bars
y_min = min(routing_values) * 1.2
y_max = max(routing_values) * 1.4
ax_c.set_ylim(y_min, y_max)
clean_axes(ax_c)

# label bars
for bar, val in zip(bars_c, routing_values):
    offset = 0.3 if val >= 0 else -1.2
    ax_c.text(
        bar.get_x() + bar.get_width() / 2,
        val + offset,
        f"{val:+.1f}%",
        ha="center", va="bottom" if val >= 0 else "top",
        fontsize=10, fontweight="bold", color="#1f2937",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Panel D: Monolithic Comparison
# ─────────────────────────────────────────────────────────────────────────────
mono_labels = ["Base", "Monolithic", "Best Specialist", "Weight Avg", "KALAVAI MoE"]
mono_values = [base_loss, monolithic_loss, best_spec_loss, weight_avg_loss, moe_loss_410m]
mono_colors = [
    COLORS["base"],
    COLORS["monolithic"],
    COLORS["code"],
    COLORS["weight_avg"],
    COLORS["moe"],
]

x_d = np.arange(len(mono_labels))
bars_d = ax_d.bar(
    x_d, mono_values,
    color=mono_colors, width=0.55, zorder=3,
)

ax_d.set_xticks(x_d)
ax_d.set_xticklabels(mono_labels, rotation=12, ha="right")
ax_d.set_ylabel("Held-out mixed loss (lower is better)")
ax_d.set_title("D.  Monolithic vs KALAVAI: Same Compute Budget")
clean_axes(ax_d)

# label bars with loss values
y_range = max(mono_values) - min(mono_values)
for bar, val in zip(bars_d, mono_values):
    ax_d.text(
        bar.get_x() + bar.get_width() / 2,
        val + y_range * 0.015,
        f"{val:.3f}",
        ha="center", va="bottom", fontsize=9, color="#374151",
    )

ax_d.set_ylim(min(mono_values) * 0.97, max(mono_values) * 1.04)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout()

out1 = REPO_ROOT / "figures" / "paper" / "fig_hero_4panel.png"
out2 = REPO_ROOT / "paper" / "figures" / "fig_hero_4panel.png"

fig.savefig(out1, dpi=300)
fig.savefig(out2, dpi=300)
plt.close(fig)

print(f"Saved: {out1}")
print(f"Saved: {out2}")
print(f"File sizes: {out1.stat().st_size // 1024} KB, {out2.stat().st_size // 1024} KB")
