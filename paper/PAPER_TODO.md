# KALAVAI Paper — Pending Updates

**Status:** Holding all edits until all experiments complete.
**Trigger:** Apply ALL items below in one batch session after C1 and 6.9B k=0/k=8 results are in.

Source documents: `kalavai_divergence_correction_todo.md` (root), `kalavai_logical_issues_audit.md` (root)

---

## CRITICAL: Eval batch_size asymmetry bug — all headline numbers are wrong

`kalavai_pythia_experiment.py:1024` used `bs = 2 if is_fused else 4` in the evaluation loop.
With `shuffle=False, drop_last=True, EVAL_BATCHES=50`, this caused:
- MoE (bs=2): evaluated on **100 code-only chunks** (loss=1.7931)
- Specialists/base (bs=4): evaluated on **200 code+science chunks** (loss=2.089)

The original +14.17% compared MoE evaluated on its strongest domain only against specialists
evaluated on a mixed set. It is not a valid improvement metric.

**CORRECTED NUMBERS — ALL COMPLETE** (from `experiments/kalavai_corrected_eval.py`)
Eval method: per-domain separate eval, bs=4 all models, equal-weight avg = (code+sci+fic)/3

**Table 1 — FINAL CORRECTED RESULTS** (from `experiments/kalavai_corrected_eval.py`)
Eval: per-domain separate, bs=4 all models, equal-weight avg = (domains) / n_domains

**Pythia 3-domain results (seed=42):**

| Model | Code | Science | Fiction | Equal-avg | vs best spec | vs base | vs mono |
|-------|------|---------|---------|-----------|-------------|---------|---------|
| 410M base | 2.0872 | 2.8920 | 2.9739 | 2.6510 | — | — | — |
| 410M monolithic | 1.9644 | 2.6389 | 2.0832 | 2.2288 | — | — | — |
| 410M MoE | 1.8791 | 2.5565 | 2.2194 | **2.2183** | **+7.72%** | **+16.32%** | **+0.47%** |
| 1B base | 2.0082 | 2.7727 | 2.6424 | 2.4745 | — | — | — |
| 1B monolithic | 1.8561 | 2.5117 | 1.9236 | 2.0971 | — | — | — |
| 1B MoE | 1.7864 | 2.5121 | 1.9716 | **2.0900** | **+7.49%** | **+15.54%** | **+0.34%** |
| 6.9B base | 1.8885 | 2.5764 | 2.4953 | 2.3200 | — | — | — |
| 6.9B MoE | 1.6968 | 2.3931 | 2.3054 | **2.1318** | **+5.81%** | **+8.11%** | n/a |

**Qwen 2-domain results (seed=42), equal-weight = (code+fiction)/2:**

| Model | Code | Fiction | Equal-avg | vs best spec | vs base |
|-------|------|---------|-----------|-------------|---------|
| Qwen base | 1.3401 | 2.5787 | 1.9594 | — | — |
| Qwen MoE | 1.3158 | 2.4611 | **1.8885** | **+1.06%** | **+3.62%** |

**Multi-seed variance:**

| Model | Seeds | Mean vs spec | Std | Mean vs base |
|-------|-------|-------------|-----|--------------|
| 410M | 42,137,2026 | **+7.70%** | ±0.02% | +16.22% |
| 1B | 42 only | **+7.49%** | — | +15.54% |
| 6.9B | 42 only | **+5.81%** | — | +8.11% |
| Qwen | 42,137,2026 | **+1.06%** | ±0.01% | +3.61% |

**Routing (all models near-deterministic — routing works across all scales and divergence levels):**
- 410M: code=99.98%, sci=99.95%, fic=100% per-domain → fiction_spec is best single specialist
- 1B:   code=99.85%, sci=99.89%, fic=99.85%
- 6.9B: code=99.98%, sci=99.94%, fic=99.99%
- Qwen: code=100%, fiction=100% (perfectly deterministic even at 1.76% divergence)

**Best single specialist = fiction_spec** (all Pythia scales, seed=42) because fiction
divergence is largest and equal-weight treats all domains equally.

**Divergence-to-gain table (within Pythia family):**

| Model | Mean div. | Gain vs spec | Conversion rate |
|-------|-----------|-------------|-----------------|
| 410M | 15.65% | +7.70% | 0.49× |
| 1B | 15.28% | +7.49% | 0.49× |
| 6.9B | 8.29% | +5.81% | 0.70× |
| Qwen | 3.16%† | +1.06% | 0.34×‡ |

*†Qwen 2-domain only (code+fiction); ‡Qwen is different model family, not directly comparable*

Within Pythia: conversion rate is ~0.49 at 410M/1B, ~0.70 at 6.9B. Roughly constant or
slightly increasing with scale — there is NO degradation. The reduced gain at 6.9B is
entirely explained by reduced divergence.

**Pending ablation/sweep corrected evals (non-blocking for main Table 1):**
- [ ] Freeze depth ablation (410M, all k values)
- [ ] Maturity sweep (410M, all steps)
- [ ] 5-domain experiment
- [ ] Router ablation (linear vs MLP)
- [ ] 1B seeds 137 + 2026 (for variance at 1B scale)

---

## RunPod gate check — before starting paper edits

| # | Item | Status |
|---|------|--------|
| 1 | Step sweep: 1k/2k/4k at k=6, seed=42 | ✅ synced |
| 2 | Freeze sweep: k=4 at steps=2000 (+5.45% vs base) | ✅ synced |
| 3 | Freeze sweep: k=8 at steps=2000 | ⏳ running |
| 4 | Freeze sweep: k=0 at steps=2000 | ⏳ needs queue (`kalavai_6b_freeze0.py`) |
| 5 | Router gate analysis JSON | ✅ synced (data in result JSON) |
| 6 | C1 heterogeneous cooperative | ✅ COMPLETE |

**Paper edits begin ONLY after items 3, 4, 6 are complete AND corrected evals for 1B, 6.9B, Qwen are done.**

---

## CRITICAL: 6.9B divergence numbers were wrong — major narrative rewrite

The paper currently claims 6.9B specialists "diverge only ~2-3% from base." This is wrong.
The ~2.4% was the **fusion gain vs best specialist**, not per-domain divergence from base.

**Actual 6.9B per-domain divergences:**

| Domain | Divergence |
|--------|-----------|
| Code   | 10.16% |
| Science | 7.11% |
| Fiction | 7.61% |
| **Mean** | **8.29%** |

Router: near-deterministic (>99.9% per domain), identical quality to 410M/1B.

**Corrected divergence table (Table 2/3):**

| Model | Code div. | Sci div. | Fiction div. | Mean div. | Fusion gain |
|-------|-----------|----------|-------------|-----------|-------------|
| Pythia-410M | 9.97% | 11.60% | 25.37% | 15.65% | **+7.72%** (corrected; was +14.17%) |
| Pythia-1B | 11.05% | 9.39% | 25.41% | 15.28% | **+7.49%** (corrected; was +14.83%) |
| Pythia-6.9B | 10.16% | 7.11% | 7.61% | 8.29% | **+5.81%** (corrected; was +2.70%) |
| Qwen-1.5B | 1.76% | — | 4.56% | 3.16% | **+1.06% ± 0.01%** (3 seeds) |

*"Fusion gain vs spec" = MoE equal-weight avg vs best single specialist equal-weight avg.
Equal-weight = (code + science + fiction) / 3, evaluated per-domain at bs=4 for all models.*

**Corrected mechanistic story:** Conversion efficiency is ~0.49 at 410M/1B, ~0.70 at 6.9B.
The 6.9B conversion rate is HIGHER than 410M/1B, not lower.
The old "efficiency degrades at scale" narrative was entirely an artifact of the eval bug.
New narrative: 6.9B gains less because specialists diverge less (8.29% vs 15.5%), but
each percent of divergence converts to fusion gain with equal or higher efficiency.
**This requires complete rewrite of Section 4.2, Section 6, and Abstract.**.

---

## Table 2 (divergence table)

- Replace 6.9B row: Code 10.16%, Sci 7.11%, Fiction 7.61%, Mean 8.29%
- Remove "pending full step-sweep analysis" footnote
- Remove ~2.4% approximation

---

## Section 4.2 — Rewrite 6.9B divergence narrative

**Delete** any prose containing:
- "specialists diverge only ~2–3% on their assigned domains"
- "Pythia-6.9B already handles code, science, and fiction with high competence, leaving little room for specialist improvement"
- Any reference to ~2.4% mean divergence
- Any claim that conversion efficiency degrades at larger scale
- Any ~1:1 divergence-to-gain ratio claim

**Replace with:**
- 6.9B specialists diverge 8.29% from base (approximately half the divergence at 410M/1B)
- Routing is near-deterministic (>99.9%), identical quality to smaller scales
- Fusion gain is +5.81% vs best single specialist (equal-weight, corrected eval)
- Conversion efficiency is 0.70× at 6.9B vs 0.49× at 410M/1B — 6.9B converts divergence
  MORE efficiently, not less. The reduced gain is due entirely to reduced specialist divergence.
- This is the simpler story: less divergence → less gain. Conversion efficiency is scale-invariant
  (or slightly improves at scale). The mechanism scales cleanly.

---

## Section 6 — Rewrite "What the 6.9B result means"

**Delete** all framing about conversion efficiency degrading at scale — this was an eval artifact.

**Replace with:**
- 6.9B specialists diverge 8.29% from base with near-deterministic routing (>99.9%)
- Fusion gain is +5.81% vs best single specialist (equal-weight, corrected eval)
- The reduced gain vs 410M/1B is explained simply: specialists diverge less at 6.9B
  (~half the divergence), so fusion gains less — the mechanism scales cleanly
- Conversion efficiency is 0.70× at 6.9B vs ~0.49× at 410M/1B — if anything, efficiency
  IMPROVES at larger scale (though both values are sub-1× because equal-weight averaging
  weights fiction equally despite fiction having the largest per-domain divergence)
- The simpler narrative: "KALAVAI gains are proportional to specialist divergence.
  When specialists diverge less (either because the base model is larger and already handles
  the domain well, or because training is short), the gain is smaller. The mechanism itself
  scales cleanly — routing remains near-deterministic at all scales tested."
- Step sweep (1k→2k→4k) and freeze sweep (k=0/4/6/8): flat returns consistent with
  the base model already being competent on these English-text domains at this step count.
  This is a data-domain limitation, not a fundamental scaling failure.

---

## Section 6 — Rewrite "What the Qwen result means" (MAJOR CHANGE)

**The Qwen result has FLIPPED from negative to positive.**

Original reported: −0.97% (MoE worse than best spec)
Corrected (seed=42, equal-weight): **+1.06% vs best spec, +3.62% vs base**
Router is perfectly deterministic: code=1.0000→code, fiction=1.0000→fiction

**Delete** ALL of the following:
- "routing-signal floor (~2% divergence)" concept
- "Qwen code specialist diverges only 1.76%, insufficient for router to distinguish domains"
- "router degrades code performance"
- "weight-averaged ensemble outperforms MoE fusion, confirming router is locus of failure"
- Any framing of Qwen as a failure case due to routing breakdown

**Replace with:**
- Qwen MoE gives +1.06% vs best single specialist (equal-weight, corrected eval, seed=42)
- Routing is near-deterministic (100%/100% per domain) — routing mechanism works on Qwen
- Small gain (+1.06%) reflects small specialist divergence (code=1.76%, fiction=4.56%,
  mean 3.16%) — the mechanism predicts small gains for small divergence, which is what we see
- Qwen is NOT a failure case — it is a data point confirming the divergence-proportional gain
  relationship at low divergence

**Impact:** The "routing-signal floor" concept in the paper is removed. The narrative is now:
"Gains scale with divergence, routing works across all scales and divergence levels tested."

Qwen 3-seed result: **+1.06% ± 0.01%** (seeds 42/137/2026) — perfectly stable.
Routing 100% deterministic at all three seeds.
Per-domain divergences (code 1.76%, fiction 4.56%) remain correct (not affected by eval bug).

---

## Abstract — Update

- Remove any claim about "~34× weaker per-parameter training signal" or conversion efficiency
  degrading at scale — both were artifacts of the eval bug
- Corrected headline numbers: 410M +7.72% vs spec (+16.32% vs base), 1B +7.49% (+15.54%),
  6.9B +5.81% (+8.11%). All use equal-weight domain averaging, bs=4 consistent.
- New framing: "gains are proportional to specialist divergence; routing remains
  near-deterministic (>99.9%) across all scales tested"

---

## Section 4.6 — Add C1 heterogeneous cooperative results

C1 corrected equal-weight results (recomputed from stored per-domain losses, Bug B fixed):
  Control:    +7.72% vs spec, +16.33% vs base
  diff_batch: +7.74% (Δ +0.01pp) | diff_lr: +7.73% (Δ +0.01pp) | diff_steps: +7.33% (Δ −0.39pp)
  Max spread: 0.41pp → ROBUST. Old +3.62% was Bug B (mixed, not equal-weight). Corrected.

- Add result table: control + diff_batch + diff_lr + diff_steps conditions
- Prose: protocol is robust to realistic heterogeneity in batch size, LR, and step budget

---

## Appendix A — Add 6.9B sweep tables

- Step sweep table: 1k/2k/4k at k=6 (+4.85% / +5.34% / +5.17% vs base)
- Freeze sweep table: k=0/4/6/8 at steps=2000 (fill in once k=0 and k=8 done)
- Router gate analysis: 3×3 gate matrix (near-deterministic: code→[0.9998, 0.0, 0.0001], etc.)

---

## Issue #1 (freeze depth, Clarification 5) — after k=0 and k=8

- If no k beats k=6 by >0.5pp on vs-base: add sentence
  "The 6.9B result is insensitive to freeze depth at these step counts, consistent with the
  410M finding that freeze sensitivity is low below 5,000 training steps."
- If any k beats k=6 by >0.5pp: re-run best config with 3 seeds, update Table 1 6.9B row

---

## Figure updates

| Figure | Change needed |
|--------|--------------|
| Figure 1 panel A (main bar chart) | Update if k=0/k=8 changes headline number |
| Scale comparison figure | Use 8.29% divergence (not ~2.4%) for 6.9B |
| Freeze sweep bar chart | Add k=0 bar once result in |
| Step/freeze grid heatmap | Add k=0 row |
| C1 heterogeneity figure | New figure once C1 completes |

---

## "Prediction" sentence — update (simpler story now)

The sentence "KALAVAI will show its largest gains precisely where it is most needed —
low-resource languages, specialised technical domains, and early-stage models" replace with:

> KALAVAI gains scale with specialist divergence: wherever a contributor's data fills
> genuine gaps in the base model's competence, specialists diverge more and fusion gains
> more. Routing remains near-deterministic at all scales tested (410M to 6.9B), indicating
> the mechanism is robust to scale. The largest gains are expected in low-resource languages,
> highly specialized technical domains, and early-stage models — precisely the settings where
> base model competence is lowest and specialist divergence highest.

Remove the caveat about conversion efficiency degrading at scale — it was an eval artifact.

---

## Already applied (do not re-apply)

- Issue #9: Fusion formula — logit-space (ae07400) ✓
- Issue #2: Best Spec Loss column in Table 4 ✓
- Issue #3: 1B freeze depth note ✓
- Issue #4: Curriculum baseline acknowledgment ✓
- Issue #5: Perplexity footnote ✓
- Issue #6: N=500 benchmark caveat ✓
- Issue #7: Domain diversity limitation ✓
- Issue #8: Sparse inference mechanistic explanation ✓
