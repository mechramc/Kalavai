# KALAVAI Paper — Pending Updates

**Status:** Holding all edits until all experiments complete.
**Trigger:** Apply ALL items below in one batch session after C1 and 6.9B k=0/k=8 results are in.

Source documents: `kalavai_divergence_correction_todo.md` (root), `kalavai_logical_issues_audit.md` (root)

---

## RunPod gate check — before starting paper edits

| # | Item | Status |
|---|------|--------|
| 1 | Step sweep: 1k/2k/4k at k=6, seed=42 | ✅ synced |
| 2 | Freeze sweep: k=4 at steps=2000 (+5.45% vs base) | ✅ synced |
| 3 | Freeze sweep: k=8 at steps=2000 | ⏳ running |
| 4 | Freeze sweep: k=0 at steps=2000 | ⏳ needs queue (`kalavai_6b_freeze0.py`) |
| 5 | Router gate analysis JSON | ✅ synced (data in result JSON) |
| 6 | C1 heterogeneous cooperative | ⏳ running (shuffle fix applied) |

**Paper edits begin ONLY after items 3, 4, and 6 are complete and pushed.**

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
| Pythia-410M | 9.97% | 11.60% | 25.37% | 15.65% | +14.17% |
| Pythia-1B | 11.05% | 9.39% | 25.41% | 15.28% | +14.83% |
| Pythia-6.9B | 10.16% | 7.11% | 7.61% | 8.29% | +5.34%† |
| Qwen-1.5B | 1.76% | — | 4.56% | 3.16% | −0.97% |

*†Use best headline number once k=0/k=8 freeze sweep is complete.*

**The new mechanistic puzzle:** 8.29% divergence + deterministic routing → only +2.4% gain vs spec
(+5.34% vs base). At 410M/1B, ~15.5% divergence converts to ~14.5% gain (roughly 1:1 ratio).
At 6.9B, 8.29% divergence converts to only +2.4% gain vs spec (~3.5:1 ratio).
The divergence-to-gain conversion efficiency degrades at larger scale.

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

**Replace with:**
- 6.9B specialists diverge 8.29% from base (approximately half the divergence at 410M/1B)
- Routing is near-deterministic (>99.9%), identical quality to smaller scales
- Despite meaningful divergence and clean routing, fusion gain is only +2.4% vs spec
  (+5.34% vs base) — substantially below the ~1:1 divergence-to-gain ratio at 410M/1B
- Conversion efficiency degrades at scale: 410M converts 15.65% → +14.17%; 6.9B converts
  8.29% → +2.4% (vs spec)

---

## Section 6 — Rewrite "What the 6.9B result means"

**Delete** the framing attributing reduced gain to reduced specialist divergence from base.

**Replace with:**
- 6.9B specialists diverge meaningfully (8.29% mean, 7–10% per domain) with deterministic routing
- The bottleneck is not divergence or routing quality but **divergence-to-fusion-gain conversion efficiency at scale**
- Possible explanations:
  (a) Base model logit distribution at 6.9B is better calibrated — specialist logit shifts
      produce smaller probability-space improvements even when loss improvements are similar
  (b) Logit-space combination (Section 3) may be less effective at larger scales where
      baseline entropy is lower
  (c) Training budget: 1k–4k steps may produce surface-level divergence without deep
      representational specialisation; the step-budget sweep (Appendix A) partially tests
      this — improvement from 1k to 4k steps is modest, suggesting structural bottleneck
- The step sweep (1k→2k→4k) and freeze sweep (k=0/4/6/8) both show flat returns,
  consistent with a structural rather than training-budget explanation

---

## Section 6 — Update "What the Qwen result means"

- Routing-signal floor (~2% divergence) is still valid — Qwen code at 1.76% is below it
- **Remove** any framing that positions 6.9B as "just above the routing floor" — 6.9B is
  well above it at 8.29%
- Qwen failure = insufficient divergence; 6.9B reduced gain = different mechanism
  (conversion efficiency). These are distinct phenomena.

---

## Abstract — Update

- Change "6.9B improvement is reduced due to ~34× weaker per-parameter training signal
  relative to 410M" (or similar) to reflect the actual finding: divergence-to-gain
  conversion efficiency degrades at scale despite meaningful specialist divergence (8.29%)
  and deterministic routing

---

## Section 4.6 — Add C1 heterogeneous cooperative results

- Add result table: control + diff_batch + diff_lr + diff_steps conditions
- Control must be within 2pp of +14.17% (verify after run completes)
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

## "Prediction" sentence — add nuance

The sentence "KALAVAI will show its largest gains precisely where it is most needed —
low-resource languages, specialised technical domains, and early-stage models" needs:

> KALAVAI gains are largest when specialist divergence is high AND the base model is
> small enough for divergence to convert efficiently into fusion gain. The 6.9B data
> (8.29% divergence → only +2.4% gain vs spec) suggests conversion efficiency may
> decrease at larger scales, motivating future work on more efficient fusion mechanisms
> for large base models.

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
