# CRITICAL: 6.9B Divergence Numbers Were Wrong — Paper Narrative Must Be Rewritten

## What happened

The paper currently claims 6.9B specialists "diverge only ~2-3% from base" (Section 4.2, Table 2, Section 6). This is wrong. The ~2.4% number used as "divergence" was actually the **fusion gain vs best specialist**, not the per-domain specialist divergence from base. These are different metrics.

The actual 6.9B per-domain divergences from the RunPod router analysis are:

| Domain | Divergence from base |
|--------|---------------------|
| Code | 10.16% |
| Science | 7.11% |
| Fiction | 7.61% |
| **Mean** | **8.29%** |

Router gate distribution: [0.9998, 0.0, 0.0001] — near-deterministic, identical quality to 410M/1B.

## Why this matters

The corrected divergence table now reads:

| Model | Code div. | Sci. div. | Fiction div. | Mean div. | Fusion gain |
|-------|-----------|-----------|-------------|-----------|-------------|
| Pythia-410M | 9.97% | 11.60% | 25.37% | 15.65% | +14.17% |
| Pythia-1B | 11.05% | 9.39% | 25.41% | 15.28% | +14.83% |
| Pythia-6.9B | 10.16% | 7.11% | 7.61% | 8.29% | +2.4% |
| Qwen-1.5B | 1.76% | — | 4.56% | 3.16% | −0.97% |

The old narrative — "specialists can barely diverge at 6.9B because the base model already handles these domains" — is falsified. Specialists diverge 8.29% on average with clean deterministic routing. The mechanism is working. The real question is: **why does 8.29% divergence with deterministic routing produce only +2.4% fusion gain, when 15.65% divergence with the same routing quality produces +14.2% at 410M?**

The divergence-to-gain conversion is NOT linear. 410M and 1B convert ~15.5% divergence into ~14.5% gain (roughly 1:1). 6.9B converts 8.29% divergence into 2.4% gain (roughly 3.5:1). Something at the 6.9B scale makes fusion less efficient per unit of divergence.

## DO NOT update the paper yet

Wait for all RunPod results to complete (k=0 and k=8 freeze sweeps). Then batch all paper updates in one session.

## Paper update TODO list (execute AFTER all results are in)

### Table 2 — Update with actual numbers
- Replace the 6.9B row dashes with: Code 10.16%, Sci 7.11%, Fiction 7.61%, Mean 8.29%
- Remove the "pending full step-sweep analysis" footnote
- Remove the "~2.4%" approximation — use 8.29%

### Section 4.2 — Rewrite the 6.9B divergence narrative
**Delete** the current prose that says:
- "specialists diverge only ~2-3% on their assigned domains"
- "Pythia-6.9B at step 10,000 already handles code, science, and fiction with high competence, leaving little room for specialist improvement"
- Any reference to ~2.4% mean divergence

**Replace with** prose that reflects the actual finding:
- 6.9B specialists diverge 8.29% from base (approximately half the divergence at 410M/1B)
- Routing is near-deterministic (>99.9%), identical quality to smaller scales
- Despite meaningful divergence and clean routing, fusion gain is only +2.4% — substantially lower than the ~1:1 divergence-to-gain ratio at 410M/1B
- The conversion efficiency from divergence to fusion gain degrades at larger scale: 410M converts 15.65% divergence into +14.17% gain; 6.9B converts 8.29% divergence into +2.4% gain
- This suggests an additional scale-dependent factor beyond specialist divergence — potentially the base model's stronger representations being harder to improve upon even when specialists do diverge, or the logit-space combination being less effective when the base distribution is already well-calibrated

### Section 6 — Rewrite "What the 6.9B result means"
**Delete** the current framing about "reduced specialist divergence from base" being the primary explanation.

**Replace with:**
- The 6.9B specialists diverge meaningfully (8.29% mean, with individual domain divergences of 7-10%) and the router achieves near-deterministic gating identical to 410M/1B
- The bottleneck is not divergence or routing quality but the efficiency of divergence-to-fusion-gain conversion at scale
- At 410M/1B, each percentage point of specialist divergence translates to approximately one percentage point of fusion gain; at 6.9B, the conversion rate is approximately 3.5× lower
- Possible explanations: (a) the base model's logit distribution at 6.9B is better calibrated, making it harder for specialist logit shifts to produce meaningful probability-space improvements; (b) the logit-space combination (Section 3) may be less effective at larger scales where the baseline entropy is lower; (c) training budget effects — 1,000 steps at 6.9B may produce surface-level divergence (loss reduction on own domain) without deep representational specialization
- The step-budget sweep (Appendix A) partially tests (c): increasing from 1k to 4k steps shows only modest improvement (+5.3% vs base at 2k steps), suggesting the bottleneck is structural rather than training-duration-limited

### Section 6 — Update "What the Qwen result means"
- The routing-signal floor concept (~2% divergence) is still valid — Qwen's 1.76% code divergence is still below it
- But the framing should no longer position 6.9B as "just above the floor" — 6.9B is well above it at 8.29%
- The Qwen failure is genuinely about insufficient divergence; the 6.9B reduced gain is about a different mechanism (conversion efficiency)

### Appendix A — Incorporate 6.9B step sweep + freeze sweep
- Add the step sweep table (1k/2k/4k at k=6)
- Add the freeze sweep table (k=0/4/6/8 at steps=2000) — once k=0 and k=8 results are in
- Add the router gate analysis (3×3 gate matrix showing near-deterministic routing)

### Abstract — Update
- Change "6.9B improvement is reduced due to a ~34× weaker per-parameter training signal relative to 410M" to something reflecting the actual finding: divergence-to-gain conversion efficiency degrades at scale despite meaningful specialist divergence

### Figure 1 panel A — May need update
- If the 6.9B headline number changes based on the freeze sweep (k=0 or k=8 beating k=6 by >0.5pp), update the bar

### Broader implication for the "prediction" sentence
The sentence "KALAVAI will show its largest gains precisely where it is most needed — low-resource languages, specialised technical domains, and early-stage models" is still valid but needs nuance. The 6.9B data shows that even with 8% divergence you only get 2.4% gain at scale. The prediction should be tempered: KALAVAI gains are largest when divergence is high AND the model is small enough for divergence to convert efficiently.

---

## RunPod status check

Before running paper updates, confirm these results are all committed:
1. Step sweep: 1k, 2k, 4k at k=6, seed=42 — ✅ synced
2. Freeze sweep: k=4 at steps=2000 — ✅ synced (+5.45% vs base)
3. Freeze sweep: k=8 at steps=2000 — ⏳ running
4. Freeze sweep: k=0 at steps=2000 — ⏳ needs to be queued
5. Router gate analysis JSON — ✅ synced (per-domain divergences + 3×3 gate matrix)

Paper updates begin ONLY after items 3 and 4 are complete and pushed.
