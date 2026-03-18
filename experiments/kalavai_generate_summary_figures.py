#!/usr/bin/env python3
"""
Generate two summary figures:
  1. fig_6b_summary.png       — 6.9B fusion bar chart (corrected eval, seed=42)
  2. fig_scale_ladder.png     — improvement % vs model size (410M, 1B, 6.9B)

Uses corrected per-domain equal-weight evaluation (Bug A + Bug B fixed).
"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path("figures/pythia")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load corrected data ───────────────────────────────────────────────────────

# 410M: 3-seed corrected eval (per-domain equal-weight, bs=4)
with open("results/pythia/corrected_eval_42.json") as f:
    corr_42  = json.load(f)
with open("results/pythia/corrected_eval_137.json") as f:
    corr_137 = json.load(f)
with open("results/pythia/corrected_eval_2026.json") as f:
    corr_2026 = json.load(f)

imps_410m = [
    corr_42["metrics"]["improvement_vs_spec"],
    corr_137["metrics"]["improvement_vs_spec"],
    corr_2026["metrics"]["improvement_vs_spec"],
]
imp_410m = float(np.mean(imps_410m))
std_410m = float(np.std(imps_410m, ddof=1))

# 1B: corrected eval seed=42 only
with open("results/pythia/pythia_1b/corrected_eval_42.json") as f:
    corr_1b = json.load(f)
imp_1b = corr_1b["metrics"]["improvement_vs_spec"]
std_1b = 0.0

# 6.9B: corrected numbers (per-domain equal-weight, seeded shuffle fix)
# Code 10.16%, Sci 7.11%, Fiction 7.61%, mean div 8.29%, fusion gain +5.81%
# Per-domain losses: code=1.6968, sci=2.3931, fic=2.3054, EW=2.1318
# Base EW=2.3200, Best spec EW=2.2634 (derived: moe / (1 - 0.0581))
base_6b     = 2.3200
bestspec_6b = 2.1318 / (1 - 0.0581)   # = 2.2634
moe_6b      = 2.1318
imp_mean    = 5.81
imp_std     = 0.0

# ── Figure 1: 6.9B summary bar chart (corrected) ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

labels = ["Base\n(step10000)", "Best\nSpecialist", "MoE Fusion"]
means  = [base_6b, bestspec_6b, moe_6b]
colors = ["#95a5a6", "#3498db", "#9b59b6"]

y_min = min(means) * 0.985
y_max = max(means) * 1.008
bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.5)
ax.set_ylim(y_min, y_max)

# Improvement annotations
for bar, val, ref in zip(bars, means, [None, base_6b, bestspec_6b]):
    if ref is not None:
        imp = (ref - val) / ref * 100
        sign = "+" if imp >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (y_max - y_min) * 0.003,
                f"{sign}{imp:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Equal-Weight Loss (lower is better)", fontsize=11)
ax.set_title(f"KALAVAI Fusion at Scale: Pythia-6.9B\n"
             f"MoE vs best specialist = +{imp_mean:.2f}% (corrected eval)", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
ax.text(0.98, 0.02, "Seed: 42 | step10000 | freeze=6/32 | per-domain equal-weight",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="gray")

fig.tight_layout()
path1 = FIGURES_DIR / "fig_6b_summary.png"
fig.savefig(path1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path1}")

# ── Figure 2: Scale ladder (3-point) ─────────────────────────────────────────
# NOTE: Add 2.8B data point here once kalavai_pythia_2b_experiment.py completes.
# Expected 4th point: ~410M params * 6.9 = 2.8B, run same config on RunPod.

fig, ax = plt.subplots(figsize=(8, 5))

model_sizes  = [0.41,   1.0,    6.9]
improvements = [imp_410m, imp_1b, imp_mean]
stds         = [std_410m,  std_1b, imp_std]
labels_s     = ["Pythia-410M\n(3 seeds)", "Pythia-1B\n(seed=42)", "Pythia-6.9B\n(seed=42)"]
colors_s     = ["#3498db", "#e74c3c", "#9b59b6"]

for x, y, e, label, c in zip(model_sizes, improvements, stds, labels_s, colors_s):
    ax.errorbar(x, y, yerr=e if e > 0 else None, fmt="o", color=c, markersize=10,
                capsize=5, linewidth=2, label=label)

ax.plot(model_sizes, improvements, "--", color="gray", alpha=0.5, linewidth=1.5)
ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

ax.set_xscale("log")
ax.set_xticks(model_sizes)
ax.set_xticklabels(["410M", "1B", "6.9B"])
ax.set_xlabel("Model Size (parameters, log scale)", fontsize=11)
ax.set_ylabel("MoE Improvement vs Best Specialist (%)", fontsize=11)
ax.set_title("KALAVAI Scale Ladder: Fusion Benefit vs Model Size\n"
             "(corrected eval — per-domain equal-weight, bs=4)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotate each point
annot_labels = [f"+{imp_410m:.1f}%", f"+{imp_1b:.1f}%", f"+{imp_mean:.1f}%"]
for x, y, label in zip(model_sizes, improvements, annot_labels):
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(10, 5), fontsize=9)

fig.tight_layout()
path2 = FIGURES_DIR / "fig_scale_ladder.png"
fig.savefig(path2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path2}")
print(f"\nScale ladder data (corrected eval):")
print(f"  410M:  +{imp_410m:.2f}% ± {std_410m:.3f}%  (3 seeds)")
print(f"  1B:    +{imp_1b:.2f}%  (seed=42 only)")
print(f"  6.9B:  +{imp_mean:.2f}%  (seed=42 only)")
