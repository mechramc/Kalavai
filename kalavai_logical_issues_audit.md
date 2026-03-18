# KALAVAI Paper — Logical Issues Audit & Resolution Plan

**Status:** Pre-NeurIPS submission review
**Date:** March 2026
**Purpose:** Track every logical inconsistency or reviewer-exploitable gap identified in the paper, with severity, resolution, and compute requirements.

---

## Issue 1: Freeze depth at 6.9B contradicts the paper's own crossover finding

**Severity:** HIGH — directly undermines internal consistency

**The problem:** Section 4.4 establishes that freezing hurts performance below 5,000 steps and only becomes beneficial around 10,000 steps. The best single result in the paper is freeze=0 at 5,000 steps (+16.4%). Yet the 6.9B experiment uses freeze=6 at only 1,000 steps — deep in the regime where the paper's own evidence says freezing is unnecessary and potentially harmful. The 6.9B specialists only achieve 2–3% improvement over base before fusion, suggesting minimal divergence. Freezing 6 layers further constrains that already-weak divergence.

**Why a reviewer catches this:** Any reader who understands the crossover finding will immediately ask why the 6.9B experiment doesn't follow the paper's own guideline.

**Resolution:** Run freeze=0 and freeze=4 at 6.9B with 1k steps (single seed). If either outperforms freeze=6, update the 6.9B headline number. If freeze=6 remains best (possible — 6.9B may behave differently due to depth), add a sentence explaining why the 410M crossover threshold doesn't transfer directly to 6.9B.

**Compute cost:** ~3 hours on A100 (two additional specialist training runs + eval).

**Current status:** Pending — run after 6.9B step-budget sweep completes.

---

## Issue 2: Shared-init ablation table doesn't show specialist quality under mismatch

**Severity:** MEDIUM — creates an appearance of hiding information

**The problem:** Section 4.7 correctly explains that the "vs. best specialist" metric is misleading under mismatch because mismatched specialists are also worse. But Table 4 doesn't show the best specialist loss for each condition, so the reader can't independently verify this claim. A reviewer who wants to decompose the effect (how much does specialist quality degrade vs. how much does routing degrade?) cannot do so from the data provided.

**Why a reviewer catches this:** Reviewers are trained to be suspicious when a paper explains away a metric without providing the underlying numbers.

**Resolution:** Add a column to Table 4 showing "Best Specialist Loss" for each condition (control, small gap, large gap). Alternatively, add the numbers inline in the text paragraph that explains the misleading metric. No new experiments needed — the data already exists in the result JSONs.

**Compute cost:** Zero — text edit only.

**Current status:** Ready to fix.

---

## Issue 3: 1B freeze depth (K=4, 25%) was not independently optimized

**Severity:** LOW — unlikely to be a primary objection but could appear in minor comments

**The problem:** The 410M freeze sweep tests K from 0 to 12 and finds 2.5pp variation. The 1B experiment uses K=4 (25% of layers) without its own sweep. The implicit assumption is that the 410M freeze sensitivity transfers to 1B, but this is unvalidated. K=4 at 1B freezes 25% of layers versus 17% at 410M — a meaningfully different fraction.

**Why a reviewer catches this:** A methodical reviewer comparing the experimental setup across scales will notice the asymmetry in how thoroughly each scale was tuned.

**Resolution:** Add a sentence in Section 4.1 or 4.4 acknowledging that the 1B freeze depth was not independently optimized, citing the 410M sweep's low sensitivity (2.5pp across 0–50%) as justification that the choice is unlikely to materially affect results.

**Compute cost:** Zero — text edit only.

**Current status:** Ready to fix.

---

## Issue 4: Monolithic baseline uses mixed-data training, not curriculum-scheduled training

**Severity:** MEDIUM — a sophisticated reviewer will raise this

**The problem:** The monolithic baseline trains on a uniform mixture of code, science, and fiction for 6,000 steps. This is the most naive multi-domain training strategy. A curriculum-based approach (e.g., training on each domain sequentially, or gradually shifting proportions) could reduce gradient interference and produce a stronger monolithic model. The paper's +14.5% advantage over monolithic might be partially explained by the monolithic baseline being unnecessarily weak rather than by the cooperative protocol being strong.

**Why a reviewer catches this:** Gradient interference in multi-task learning is well-studied. A reviewer from the multi-task learning community will immediately ask about curriculum alternatives.

**Resolution:** Add a sentence in Section 4.3 or Section 6 noting that curriculum-based monolithic training (domain-sequential or graduated mixing) is an untested alternative that could narrow the gap. This is an honest limitation, not a fatal flaw — even if curriculum training closes half the gap, KALAVAI still wins. The key advantage (zero communication during training, contributors never share data) is structural and unaffected by the monolithic baseline's strength.

**Compute cost:** Zero for the text acknowledgment. Running a curriculum baseline would cost ~2 hours at 410M and would be a strong addition if time permits.

**Current status:** Ready to fix (text). Curriculum experiment is optional.

---

## Issue 5: Improvement metric uses cross-entropy loss, not perplexity

**Severity:** LOW — presentation choice, not a methodological error

**The problem:** The paper reports all improvements as percentage reduction in cross-entropy loss. But percentage improvement on loss is not linear in information-theoretic terms. A 14% reduction in loss from 2.248 to 1.793 corresponds to a perplexity drop from approximately 9.5 to 6.0 — a 37% perplexity reduction. Depending on a reviewer's background, they may find the loss-based percentages either understated or overstated relative to their intuition.

**Why a reviewer catches this:** Language modeling papers traditionally report perplexity. A reviewer may simply prefer perplexity and question why it wasn't used.

**Resolution:** Add a footnote to Section 4.1 (Evaluation metric) noting the relationship: "A 14% reduction in cross-entropy loss corresponds to approximately 37% reduction in perplexity. We report loss-based percentages throughout for consistency; perplexity values can be recovered via exp(L)."

**Compute cost:** Zero — text edit only.

**Current status:** Ready to fix.

---

## Issue 6: Downstream benchmarks use only 500 examples per task

**Severity:** MEDIUM — undermines the downstream claims even though the paper is cautious about them

**The problem:** Tables 14 and 15 use 500 examples per benchmark. At this sample size, differences of 1–2 percentage points are within statistical noise. The paper's conclusion that "downstream gains are modest" is honest, but a reviewer could argue that no valid downstream conclusion can be drawn at N=500. Standard practice for HellaSwag, ARC-Easy, etc. is full evaluation set.

**Why a reviewer catches this:** Benchmark methodology is a common reviewer focus, especially when the paper discusses downstream results at all.

**Resolution:** Either re-run benchmarks with full evaluation sets (cheap at 1B — HellaSwag has ~10k examples, ARC-Easy ~2.3k, takes minutes per model), or add an explicit caveat: "Due to the evaluation sample size (N=500), downstream accuracy differences should be treated as directional indicators, not statistically significant findings." The paper already hedges downstream claims heavily, so this is reinforcement rather than a new limitation.

**Compute cost:** Low if re-running at 1B (minutes per model, no training). Zero if adding text caveat only.

**Current status:** Ready to fix (text). Full benchmark re-run is recommended but optional.

---

## Issue 7: All domains are English text domains

**Severity:** LOW — covered by existing limitations but could be strengthened

**The problem:** Code, science, and fiction are all English text domains with substantial vocabulary overlap. The 5-domain experiment adds math (GSM8K) and multilingual (Spanish Wikipedia), which helps, but only at 410M. A reviewer may question whether KALAVAI works when domains are truly distinct — for example, cross-lingual (English vs. Mandarin) or cross-modal (text vs. structured data). The paper's application scenarios (hospital networks, endangered languages, sovereign AI) imply much greater domain diversity than the experiments test.

**Why a reviewer catches this:** The gap between the vision (Section 6, Applications) and the evidence (three English text domains) is visible.

**Resolution:** Add to the limitations section: "All primary experiments use English text domains with overlapping vocabulary. The 5-domain experiment adds multilingual data (Spanish Wikipedia), but cross-lingual and cross-modal settings remain untested. The applications scenarios in this paper assume domain diversity beyond what our experiments validate." No new compute required — this is an honesty edit.

**Compute cost:** Zero — text edit only.

**Current status:** Ready to fix.

---

## Issue 8: Sparse inference quality collapse is unexplained mechanistically

**Severity:** MEDIUM — the finding is strong but the explanation is incomplete

**The problem:** Appendix O shows that at 410M, sparse top-1 inference has 100% routing agreement with dense inference (the same expert is selected) yet quality collapses catastrophically (loss 3.106 vs. 2.568 dense — worse than the base model at 2.692). The paper states this proves "joint processing drives improvement, not routing quality" but doesn't explain *why* running one expert with correct routing is worse than the base model. If the router correctly identifies the best expert and that expert ran the same forward pass, why would the output be worse?

**Why a reviewer catches this:** This is the most surprising empirical finding in the inference analysis. A reviewer will want a mechanistic explanation, not just an observation.

**Resolution:** The likely explanation involves two factors. First, even with >99.7% weight on one expert in dense mode, the residual 0.3% contribution from other experts acts as implicit ensemble smoothing on the output distribution. Removing this entirely (sparse mode) loses that regularization. Second, and more importantly, the router input in dense mode is the mean-pooled hidden state across all specialists' forward passes (per the paper's footnote 3 and Appendix K). In sparse mode, this mean-pooled state is replaced by a single specialist's hidden state — a different input to the router entirely. Even if the top-1 selection happens to agree, the gate weights are computed from a different representation, and the output distribution is conditioned on a different context.

Add 2–3 sentences to Section 6 (Inference cost) or Appendix O articulating this: "We hypothesize two factors explain the quality collapse despite correct routing. First, the router input in dense mode is the mean-pooled hidden state across all specialists (see footnote 3); in sparse mode, this becomes a single specialist's hidden state — a different representation that alters the conditioning context. Second, even near-deterministic routing (>99.7% weight on one expert) preserves a residual ensemble contribution from other specialists that is lost under strict top-1 selection."

**Compute cost:** Zero — text edit only. The hypothesis could be tested (compare output distribution entropy between dense and sparse) but the explanation is sufficient for a NeurIPS submission.

**Current status:** Ready to fix.

---

## Issue 9 (CRITICAL — added during audit): Fusion formula in paper does not match code

**Severity:** CRITICAL — factual misrepresentation of the method

**The problem:** Section 3 (Phase 4) describes the fused output as a weighted combination of specialist probability distributions: p_fused = Σ gᵢ · softmax(logitᵢ). Every experiment file actually computes a weighted combination of logit vectors followed by a single softmax: p_fused = softmax(Σ gᵢ · logitᵢ). These are mathematically different because softmax is nonlinear. The code is internally consistent (all files use the same logit-space formula), but the paper describes a different computation than what produced the results.

**Why a reviewer catches this:** Any reviewer who reads the code — which is publicly linked — will find this immediately. It is the single most damaging inconsistency in the paper.

**Resolution:** Update Section 3 to describe the logit-space combination that the code actually implements. This is a standard and widely-used MoE formulation (Mixtral uses logit-space combination). The fix is a one-paragraph text edit:

Replace the fusion equation with the logit-space formulation:
```
l̃_t = Σ gᵢ · l_{θᵢ,t}
p_fused = softmax(l̃_t)
```

Update surrounding prose from "weighted combination of their logit distributions" to "weighted combination of their logit vectors." Add a brief note: "This logit-space combination is standard in MoE architectures. A probability-space alternative (Σ gᵢ · softmax(l_{θᵢ,t})) would produce a proper mixture distribution; we use logit-space combination throughout all experiments."

Do NOT rerun experiments. The results are real and internally consistent. The paper description needs to match the code, not the other way around.

**Compute cost:** Zero — text edit only.

**Current status:** URGENT — fix before arXiv v2 upload.

---

## Resolution Priority

| Priority | Issue | Fix Type | Compute |
|----------|-------|----------|---------|
| URGENT | #9 — Fusion formula mismatch | Text edit | None |
| HIGH | #1 — 6.9B freeze depth | Experiment + text | ~3hr A100 |
| MEDIUM | #2 — Shared-init table completeness | Text edit | None |
| MEDIUM | #4 — Monolithic curriculum note | Text edit | None |
| MEDIUM | #6 — Benchmark sample size | Text edit (or re-run) | None / Low |
| MEDIUM | #8 — Sparse inference explanation | Text edit | None |
| LOW | #3 — 1B freeze depth note | Text edit | None |
| LOW | #5 — Loss vs perplexity footnote | Text edit | None |
| LOW | #7 — Domain diversity note | Text edit | None |

All text-edit fixes can be batched into a single Claude Code prompt. Issue #1 depends on the 6.9B sweep results. Issue #9 must be fixed before arXiv v2.
