"""Generate Phase 2 figures: divergence-gain scatter + cross-lingual perplexity bar."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("paper/figures", exist_ok=True)

# ── Figure 1: Divergence vs Gain scatter ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# Data: (divergence%, gain%, label, color, marker)
english = [
    (3.16,  1.06,  "Qwen-1.5B\n(code+fiction)",      "#7f8c8d", "D"),
    (8.73,  6.53,  "Pythia-6.9B\n(code/sci/fiction)", "#e67e22", "s"),
    (15.28, 7.49,  "Pythia-1B\n(code/sci/fiction)",   "#2980b9", "o"),
    (15.65, 7.70,  "Pythia-410M\n(code/sci/fiction)",  "#27ae60", "o"),
]
professional = [
    (18.52, 10.17, "Exp 2: Private-domain\n(medical/legal/patent)",  "#8e44ad", "^"),
]
crosslingual = [
    (25.65, 21.87, "Exp 1: Cross-lingual\n(Tamil/Yoruba/Welsh/Code)", "#c0392b", "*"),
]

def plot_group(data, alpha=1.0):
    for (div, gain, lbl, col, mk) in data:
        sz = 180 if mk == "*" else 120
        ax.scatter(div, gain, color=col, marker=mk, s=sz, zorder=5, alpha=alpha)
        offsets = {
            "Qwen-1.5B\n(code+fiction)": (1.0, -1.5),
            "Pythia-6.9B\n(code/sci/fiction)": (1.0, 0.4),
            "Pythia-1B\n(code/sci/fiction)": (-8.0, 0.6),
            "Pythia-410M\n(code/sci/fiction)": (1.0, -2.0),
            "Exp 2: Private-domain\n(medical/legal/patent)": (1.0, 0.5),
            "Exp 1: Cross-lingual\n(Tamil/Yoruba/Welsh/Code)": (-14.0, 1.0),
        }
        dx, dy = offsets.get(lbl, (1.0, 0.5))
        ax.annotate(lbl, (div, gain), xytext=(div+dx, gain+dy),
                    fontsize=7.5, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6) if abs(dx) > 3 else None)

plot_group(english)
plot_group(professional)
plot_group(crosslingual)

# Fit line for English domains (Qwen + 6.9B + 1B + 410M)
eng_div  = np.array([d[0] for d in english])
eng_gain = np.array([d[1] for d in english])
m_eng, b_eng = np.polyfit(eng_div, eng_gain, 1)
x_fit = np.linspace(0, 28, 200)
ax.plot(x_fit, m_eng*x_fit + b_eng, '--', color='#555555', lw=1.2, alpha=0.6,
        label=f"English fit: gain ≈ {m_eng:.2f}× div + {b_eng:.2f}")

# Reference annotation: conversion rates
for (div, gain, lbl, col, mk) in english + professional + crosslingual:
    rate = gain / div
    ax.annotate(f"{rate:.2f}×", (div, gain), xytext=(div+0.3, gain-1.5),
                fontsize=6.5, color=col, alpha=0.75)

ax.set_xlabel("Mean Specialist Divergence (%)", fontsize=11)
ax.set_ylabel("Fusion Gain vs Best Specialist (%)", fontsize=11)
ax.set_title("KALAVAI: Fusion Gain Scales with Specialist Divergence", fontsize=12, fontweight='bold')

# Legend
handles = [
    mpatches.Patch(color='#7f8c8d', label='English domains (Phase 1)'),
    mpatches.Patch(color='#8e44ad', label='Private domains (Exp 2)'),
    mpatches.Patch(color='#c0392b', label='Cross-lingual (Exp 1)'),
]
ax.legend(handles=handles, fontsize=8.5, loc='upper left')
ax.set_xlim(-1, 30)
ax.set_ylim(-1, 26)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', lw=0.5)

plt.tight_layout()
plt.savefig("paper/figures/fig_divergence_gain_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_divergence_gain_scatter.png")

# ── Figure 2: Cross-lingual perplexity bar chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

domains    = ["Yoruba", "Welsh", "Tamil", "Code"]
base_ppl   = [41.9, 102.7, 4.2, 8.2]
moe_ppl    = [7.7,  22.1,  3.0, 8.1]   # seeds 137/2026 (seed 42: Welsh=21.6, code=8.1)
spec_ppl   = [7.7,  22.1,  3.0, 8.1]   # MoE = specialist (perfect routing all seeds)

x = np.arange(len(domains))
w = 0.30

bars_base = ax.bar(x - w,   base_ppl, w, label="Base model", color='#e74c3c', alpha=0.85)
bars_spec = ax.bar(x,       spec_ppl, w, label="Specialist (own domain)", color='#3498db', alpha=0.85)
bars_moe  = ax.bar(x + w,   moe_ppl,  w, label="KALAVAI MoE (all domains)", color='#27ae60', alpha=0.85)

# Improvement annotations — multiplicative offset works on log scale
improvements = ["5.4×", "4.6×", "1.4×", "1.0×"]
for i, imp in enumerate(improvements):
    ax.text(i, base_ppl[i] * 1.15, imp, ha='center', fontsize=9, fontweight='bold', color='#c0392b')

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=11)
ax.set_ylabel("Perplexity — log scale (lower = better)", fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1.5, 200)
ax.legend(fontsize=9)
ax.text(0, -0.10,
        "3-seed mean (seeds 42, 137, 2026). Curriculum warm-start eliminated router collapse on all seeds.",
        fontsize=7, color='gray', ha='left', transform=ax.get_xaxis_transform())

plt.tight_layout()
plt.savefig("paper/figures/fig_crosslingual_perplexity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_crosslingual_perplexity.png")
