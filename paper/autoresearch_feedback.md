# Autoresearch Feedback: KALAVAI Paper Analysis
_Generated 2026-03-19 via autoresearch outer-loop reflection_

---

## Overall Assessment

The empirical coverage is genuinely thorough: scale sweep, maturity sweep, freeze ablation,
router ablation, dispatch failure, capacity controls, heterogeneous cooperative, shared init,
and Phase 2 high-divergence extensions. The self-correction of the eval bug is handled
transparently. The divergence-gain relationship across 6 conditions is compelling.

The paper's main weaknesses are (a) the central empirical finding is underframed, (b) one
predictable reviewer objection (LoRA) has no rebuttal data, and (c) several high-value analyses
are computable from existing results but haven't been run.

---

## Gaps, Experiments, and Ideas

### 1. The Divergence-Gain Relationship Is Descriptive, Not Predictive

**The issue.** The divergence-gain table (Tab. 5) is presented as an observation, but it is
actually a predictive engineering formula. The paper never fits it. The conversion rates
(0.34×, 0.49×, 0.70×, 0.55×, 0.85×) show an apparent upward trend with divergence.

**What's missing.** A log-linear or piecewise-linear regression on the 6 existing data points,
with R² and confidence interval. This turns the descriptive table into a calibration curve:
"if your domain diverges 30%, expect +X% gain."

**Proposed experiment (no new runs needed).** Fit a linear and log-linear regression to the
6 divergence-gain data points. Report R², slope, and 95% CI. Include as a figure or inset
on the existing divergence-gain scatter plot.

**Proposed new experiment (cheap).** Artificially control divergence by varying specialist
training steps (50, 100, 250, 500, 1000, 2000 steps at 410M) to produce ~0.5%, 1%, 5%, 10%,
15% divergence. Plot the resulting gains. This converts the paper's key finding from
descriptive to predictive, with a calibration curve practitioners can use before committing
to a cooperative.

**Reframing opportunity.** The paper's central empirical contribution should be elevated:
> "Given specialist divergence d, fusion gain ≈ 0.5d on English domains and up to 0.85d in
> high-divergence settings. This lets practitioners estimate the value of a cooperative
> before running it."

This changes the paper from "we show fusion works + here are the conditions" to "we provide
a quantitative predictive model."

---

### 2. The Monolithic Comparison Headline Is Weak — Per-Domain Is the Real Story

**The issue.** The paper shows MoE beats monolithic by only +0.47% on equal-weight loss.
A skeptical reviewer stops here and asks "why bother?" The paper argues "per-domain specialist
quality + privacy" but never shows the per-domain monolithic comparison in a table.

**What's missing.** A table showing per-domain loss for monolithic vs. MoE vs. each specialist:

```
                   Code loss   Sci loss   Fiction loss
Best specialist    X.XXX       X.XXX      X.XXX
Monolithic         X.XXX       X.XXX      X.XXX
KALAVAI MoE        X.XXX       X.XXX      X.XXX
```

The MoE should match the diagonal of the cross-domain specialist matrix (already computed
in the heatmap figure in the appendix). The monolithic model likely underperforms the MoE
on every individual domain despite equal EW loss.

**Why this matters.** "A cooperative achieves per-domain specialist quality across all domains
simultaneously, which no monolithic model can do" is a much stronger claim than "+0.47% EW."
This is the paper's strongest practical argument and it is currently buried.

**Effort.** Zero new runs. Data already exists in the cross-domain heatmap results.

---

### 3. LoRA Ablation Is Missing and Will Be Reviewer Target #1

**The issue.** The paper states "LoRA showed insufficient specialist divergence" (Appendix B)
but provides no numbers. Every NeurIPS reviewer familiar with PEFT will ask why full
fine-tuning is required. This is a significant gap.

**What's missing.** LoRA specialists at 410M with r∈{8,32,64}, measuring:
- Per-domain divergence achieved
- Fusion gain under corrected eval
- Whether the divergence-gain relationship predicts the result

The paper's own framework predicts the answer: "LoRA produces ~X% divergence, so we predict
~Y% fusion gain (0.49× conversion rate)." If the gain is small, it confirms the framework.
If it is large, it is a positive result. Either way it strengthens the paper.

**Effort.** ~2–4h on a single A100 at 410M. Relatively cheap insurance against a predictable
reviewer objection.

---

### 4. The Conversion Rate Non-Monotonicity Is Unaddressed

**The issue.** Conversion rates do not monotonically increase with divergence:
- Qwen: 0.34× at 3.16% divergence
- 410M: 0.49× at 15.65% divergence
- 6.9B: 0.70× at 8.29% divergence ← breaks the pattern

The paper's explanation is "larger models convert more efficiently," but this is confounded:
6.9B sees less divergence on the same domains because the base model is already stronger,
so the relationship conflates base model quality with conversion efficiency.

**Hypothesis.** The relevant variable may be *normalized divergence* — divergence relative
to base model uncertainty on that domain. Yoruba (PPL 41.9), legal (high base uncertainty),
and 6.9B (already competent on simple domains) are all instances where the base model's
prior domain quality modulates the conversion rate.

**Proposed analysis (no new runs).** Replot the divergence-gain scatter with base model
perplexity on the domain as a third axis (bubble size). Does base PPL explain the conversion
rate variation better than raw divergence alone? This is a 1-hour analysis from existing data.

---

### 5. CKA Is Cited But Never Measured

**The issue.** The paper cites the Pari CKA thesis as theoretical motivation for why shared
initialisation preserves representational compatibility. The paper uses loss-based divergence
as a proxy, but never directly measures CKA between specialists.

**What's missing.** CKA between pairs of specialists at steps 0, 500, 1000, 2000. Expected
result: CKA decays monotonically (specialists diverge) but remains well above the CKA between
random models (remaining compatible). This provides a mechanistic explanation grounded in
representational geometry rather than loss proxies.

**Effort.** ~1–2h compute at 410M using the existing specialist checkpoints.

---

### 6. Oracle Routing Upper Bound Is Missing

**The issue.** The paper never characterises how close MoE routing is to optimal. The oracle
answer: for each evaluation batch, manually route to the best-performing specialist and report
that loss. The gap between oracle and MoE routing reveals whether routing is the bottleneck.

**Why this matters.** If oracle is only 0.5% better than the current MoE, routing is essentially
saturated and there is no value in improving it (supporting the "router architecture doesn't
matter" claim). If oracle is 5% better, there is significant headroom and future work on
routing improvement is motivated.

**Effort.** Zero new runs. Computable from existing cross-domain eval matrices.

---

### 7. The Lower Bound of the Mechanism Is Untested

**The issue.** Qwen at 3.16% divergence still gives +1.06%. Is there a regime where gain is
zero or negative? The paper has no data below 3% divergence.

**Proposed experiment.** At 410M, train specialists for only 50–100 steps (producing ~0.5–1%
divergence). Does the router still converge? Does gain remain positive? This directly tests
whether the mechanism has a lower bound and takes ~1–2h.

---

### 8. Continual Cooperative: Adding a New Specialist Post-Hoc

**The idea.** The paper presents KALAVAI as a fixed one-shot fusion. But a practical cooperative
is *growing* — new contributors join over time. Can a 4th specialist be added to an existing
3-specialist MoE by retraining only the router (freezing all existing specialists)?

**Why this matters.** If true, KALAVAI is not just a one-shot fusion — it is a protocol for
*accumulating* specialist knowledge over time. This is a fundamentally different application
story: "your cooperative can grow without retraining existing specialists."

**Effort.** ~4h at 410M. Trains one new specialist, then retrains the router with 4 inputs.

---

### 9. Seed 42 Cross-Lingual Collapse Is Under-Analyzed

**The issue.** Seed 42 fails because Yoruba routes to Tamil (both are tokenizer-OOD
byte-fallback scripts). The paper notes this as "router initialisation sensitivity" but does
not investigate the fix or show the collapse trajectory.

**Missing analysis.** Show the gate weight evolution during router training for the seed 42
collapse. At what step does Yoruba collapse onto Tamil? Does routing initially separate them
before collapsing? This would reveal whether the problem is initialisation (fixable with warm
start) or representational (structural).

**Connection to current work.** The base_hidden router (Exp 3 fallback) may partially address
this — the base model's representation of Yoruba vs. Tamil is likely more distinct than the
mean of 4 specialists. Worth adding a note connecting the Exp 3 router fix to the seed 42
collapse.

---

### 10. The Inference Cost Story Has a Missing Angle

**The issue.** The paper reports 2.86× latency at 410M and 3.35× at 1B. But it does not
discuss quantized inference — fitting N specialists into reasonable VRAM using 4-bit
quantization. At N=20 (Exp 3), the inference story becomes critical (20× overhead is
impractical unquantized).

**Proposed analysis.** What is the quality degradation from 4-bit quantizing all specialists
vs. the full-precision fused model? If the answer is "small," then N=20 at 4-bit is feasible
on a single A100. This turns the Exp 3 result from a proof-of-concept into a practical system.

---

## Priority Summary

| Priority | Item | New Runs? | Effort |
|---|---|---|---|
| **P0** | Per-domain vs. monolithic table | No | 1h analysis |
| **P0** | Divergence-gain regression fit | No | 1h analysis |
| **P0** | Oracle routing upper bound | No | 1h analysis |
| **P1** | LoRA ablation at 410M | Yes | 2–4h GPU |
| **P1** | Base PPL as conversion rate predictor | No | 1h analysis |
| **P2** | Low-divergence ablation (50–100 steps) | Yes | 1–2h GPU |
| **P2** | CKA between specialists | Yes | 1–2h GPU |
| **P2** | Seed 42 collapse trajectory | No | 1h analysis |
| **P3** | Continual cooperative (add 4th specialist) | Yes | 4h GPU |
| **P3** | Quantized N=20 inference | Yes | 2h GPU |

P0 items require no new experiments — only analysis of existing result JSONs and figures.
They are the highest-ROI additions to the paper before submission.

---

## Key Reframing Suggestion

The paper currently opens with "we study the conditions under which..." The stronger opening is:

> "We show that fusion gain scales predictably with specialist divergence at a conversion rate
> of 0.5× on English domains, rising to 0.85× on high-divergence settings. This predictive
> relationship, validated across six experimental conditions, lets practitioners estimate the
> value of a cooperative before committing to training."

The conditions analysis (freeze crossover, dispatch failure, shared init) then supports this
predictive model rather than being the primary contribution. This framing is more actionable
and more novel relative to prior work (BTX does not provide a predictive model).
