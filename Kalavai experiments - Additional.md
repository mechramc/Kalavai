# KALAVAI NeurIPS 2026 — New Experiments Prompt for Claude Code

Repository: https://github.com/mechramc/Kalavai (clone locally first)
Hardware: RTX 5090 (local), Mac Studio M4 Max 64GB (local), A100 80GB (rental, use only for 6.9B)

The existing experiment scripts in `experiments/` are the reference pattern. Every new experiment should follow the same conventions:
- Self-contained single Python file in `experiments/`
- Produces result JSONs in `results/`
- Produces figures in `figures/`
- Uses fixed seeds (42, 137, 2026) for reproducibility
- Uses the same evaluation metric as existing experiments (cross-entropy loss on held-out mixed-domain eval, improvement % formula from the paper)

Study the existing scripts — particularly `kalavai_pythia_experiment.py` (410M main), `kalavai_monolithic_baseline.py`, and `kalavai_pythia_1b_experiment.py` — before writing any new code. Match their patterns exactly: same data loading, same eval pipeline, same JSON output format, same figure generation.

---

## PHASE A: Fast, high-information experiments (run on RTX 5090)

Run these first. They're cheap and determine whether the NeurIPS path is viable.

### Experiment A1: 1B Equal-Compute Monolithic Baseline

**File:** `experiments/kalavai_1b_monolithic_baseline.py`

**What:** Train a single Pythia-1B model from step10000 checkpoint on mixed data (equal proportions code/science/fiction) for 6,000 steps (= 3 specialists × 2,000 steps). This is the exact same design as the existing 410M monolithic baseline, scaled to 1B.

**Compare against:**
- 1B base model (already have this: loss 2.160)
- 1B best specialist (already have this: loss 1.992)
- 1B KALAVAI MoE (already have this: loss 1.696)

**Configuration:** Match the existing 1B experiment settings exactly — same optimizer (AdamW, lr=2e-5, weight decay 0.1, linear warmup 10%), same batch size, same sequence length (512), same freeze depth (K=4). The ONLY difference from the specialist training is: (a) 6,000 steps instead of 2,000, (b) mixed data instead of single-domain.

**Seeds:** 3 seeds (42, 137, 2026)

**Expected output:** A table like Table 2 in the paper but for 1B. Report: base loss, monolithic loss, best specialist loss, KALAVAI MoE loss, and improvement columns (vs. base, vs. monolithic).

**Success criterion:** KALAVAI MoE meaningfully beats monolithic at 1B. Even a smaller margin than the 410M result (+14.5%) is valuable — anything above +5% over monolithic is a strong result.

**Estimated time:** ~2-3 hours on RTX 5090 (6,000 steps at 1B).

---

### Experiment A2: Inference Cost Benchmarking

**File:** `experiments/kalavai_inference_benchmark.py`

**What:** Benchmark inference performance of existing trained models. NO new training required — load the already-trained 410M and 1B checkpoints and measure.

**Measure for each configuration:**
- Peak GPU VRAM usage (torch.cuda.max_memory_allocated)
- Tokens per second (throughput) — generate 512 tokens, measure wall-clock time, average over 10 runs with 3 warmup runs
- Per-token latency (1/throughput)
- Total model parameter count loaded in memory

**Configurations to benchmark (at 410M):**
1. Base model (single forward pass)
2. Single specialist (single forward pass)
3. Monolithic model (single forward pass)
4. KALAVAI MoE — all 3 specialists, softmax routing (the production config)
5. KALAVAI MoE — top-1 only (argmax, compute ONLY the top-1 specialist's forward pass after a cheap router pass on frozen layers; do NOT run all 3 specialists)
6. KALAVAI MoE — top-2 (compute only top-2 specialists, weighted combination)

**Repeat at 1B** for configurations 1, 4, 5, 6.

**Output:** A table with columns: Config | Params Loaded | Peak VRAM (GB) | Tokens/sec | Relative Latency (vs base)

**Critical detail for config 5 (top-1 sparse):** This tests whether you can actually skip computing the other experts. The router input is mean-pooled hidden states from the frozen layers — which are shared across all specialists. So the frozen-layer forward pass runs once, the router predicts which expert to use, and then ONLY that expert's unfrozen layers run. Measure whether this produces the same routing decisions as the full joint-inference setup. Report: (a) routing agreement % (how often top-1 matches the joint-inference top-1), (b) loss if you actually use the sparse output.

**This directly addresses the paper's biggest practical weakness.** If top-1 sparse inference preserves 90%+ of the quality at ~1.3× base latency instead of ~2.5×, that transforms the inference cost story.

**Estimated time:** ~1 hour on RTX 5090 (no training, just inference benchmarking).

---

### Experiment A3: Shared Initialization Necessity Ablation

**File:** `experiments/kalavai_shared_init_ablation.py`

**What:** Test whether specialists MUST start from the same checkpoint. This is the paper's core structural claim ("shared initialisation is the only coordination requirement") but it's currently argued by narrative, not demonstrated experimentally.

**Design — two conditions at 410M:**

**Condition 1 (control):** Current setup. All 3 specialists initialize from Pythia-410M step10000. Train 2,000 steps each. Fuse with router. (This is your existing +14.2% result — just re-confirm it.)

**Condition 2 (different checkpoints):** 3 specialists initialize from DIFFERENT Pythia-410M checkpoints:
- Code specialist: initialize from step5000
- Science specialist: initialize from step10000
- Fiction specialist: initialize from step20000

Same domains, same training steps (2,000), same freeze depth (K=4), same router training. The ONLY difference is the starting checkpoint.

**Condition 3 (nearby checkpoints):** 3 specialists from closer checkpoints:
- Code specialist: initialize from step8000
- Science specialist: initialize from step10000
- Fiction specialist: initialize from step12000

**Why these specific checkpoints:** Pythia releases checkpoints at regular intervals. This tests a gradient of "how different can the starting points be." Condition 2 is a large gap (5k to 20k); Condition 3 is a small gap (8k to 12k).

**Seeds:** 3 seeds (42, 137, 2026) for Condition 1 (re-confirmation) and Condition 2. Condition 3 can be 1 seed if time is tight.

**Output:** Table with: Condition | Init Checkpoints | MoE Loss | Improvement vs Best Specialist | Improvement vs Base

**Success criterion:** Clear degradation when checkpoints differ. Ideally Condition 2 shows significant degradation, Condition 3 shows partial degradation, producing a "fusibility vs. initialization distance" gradient.

**Estimated time:** ~3 hours on RTX 5090 (3 conditions × ~1 hour each).

---

## PHASE B: Scaling experiment (decide after Phase A results)

Only proceed if Phase A results are positive (especially A1).

### Experiment B1: 6.9B Specialist Step-Budget Sweep

**File:** `experiments/kalavai_6b_step_sweep.py`

**What:** The paper's 6.9B result (+2.4%) used only 1,000 specialist training steps. The paper itself argues this is probably the bottleneck. Test whether more training steps improve the 6.9B result.

**Run on A100 80GB.**

**Grid:**
- 1,000 steps (existing result, re-confirm)
- 2,000 steps
- 4,000 steps

Same everything else: Pythia-6.9B step10000, freeze=6, 3 domains, 500 router steps.

**Seeds:** 1 seed (42) for exploration across all step counts. If a clear winner emerges, run 3 seeds on the best point.

**Also sweep freeze depth at the best step count:** The current K=6 was chosen without the crossover analysis. Try K=4, K=6, K=8 at the best step count to check if 6.9B has a different optimal freeze depth.

**Output:** Table: Steps | Freeze | Base Loss | Best Specialist Loss | MoE Loss | Improvement vs Best Spec | Improvement vs Base

**Success criterion:** If 4k steps gets 6.9B to +5% or above, that materially strengthens the scaling story. If it stays flat at ~2.4%, the honest conclusion is "scales weakly beyond medium models" — still publishable but weaker.

**Estimated time:** ~24 hours on A100 (4k steps at 6.9B is ~4× the current 1k experiment, plus freeze sweep).

**Cost estimate:** ~$75-100 in A100 rental.

---

## PHASE C: Realism upgrade (if time permits after A and B)

### Experiment C1: Heterogeneous Cooperative Simulation

**File:** `experiments/kalavai_heterogeneous_cooperative.py`

**What:** Test whether the protocol is robust to realistic variation in training conditions across contributors.

**Design at 410M:**

**Condition 1 (control):** Current setup — all specialists use identical training config.

**Condition 2 (different batch sizes):**
- Code specialist: batch size 4
- Science specialist: batch size 8 (default)
- Fiction specialist: batch size 16
Adjust step count so total tokens seen is equivalent.

**Condition 3 (different optimizers):**
- Code specialist: AdamW lr=1e-5
- Science specialist: AdamW lr=2e-5 (default)
- Fiction specialist: AdamW lr=5e-5

**Condition 4 (different training durations — asynchronous submission):**
- Code specialist: 1,000 steps
- Science specialist: 2,000 steps (default)
- Fiction specialist: 3,000 steps

**Seeds:** 1 seed (42) per condition. 3 seeds for any condition that shows interesting results.

**Output:** Table: Condition | Description | MoE Loss | Improvement vs Best Spec

**Success criterion:** Fusion remains within 2pp of the control across all heterogeneity conditions. This proves the protocol is robust to realistic contributor variation.

**Estimated time:** ~4 hours on RTX 5090.

---

## PHASE D: Polish (only if all above complete and results are strong)

### Experiment D1: 5-Domain Scaling at 1B

**File:** `experiments/kalavai_1b_5domain_experiment.py`

**What:** The 5-domain scaling experiment (2→5 specialists) currently exists only at 410M. Replicate at 1B with 3, 4, 5 specialists.

### Experiment D2: Cross-Architecture Validation

**File:** `experiments/kalavai_llama_experiment.py`

**What:** Run the core 3-domain experiment on Llama-3.2-1B (or the closest available model with public intermediate checkpoints). This addresses the "Pythia-only" critique. Note: Llama may not have intermediate checkpoints, so you may need to use the final checkpoint and accept that the maturity sweep won't be possible. The key question is: does the fusion mechanism work on a non-Pythia architecture?

---

## Integration with paper

After experiments complete, results need to flow into the paper revision. Specifically:

- **A1 (1B monolithic):** Creates a new Table (Table 2b or extends Table 2) for the 1B equal-compute comparison. Goes in Section 4.3.
- **A2 (inference benchmark):** Creates a new Table for Section 6 (Discussion), replacing the hand-wavy inference cost paragraph with quantified data. If top-1 sparse works, this transforms the narrative.
- **A3 (shared init):** Creates a new subsection (Section 4.7 or extends Section 4.2) demonstrating the shared initialization requirement experimentally.
- **B1 (6.9B sweep):** Updates the 6.9B result in Table 1 and Section 4.2 if improvement is found. Updates the discussion in Section 6.
- **C1 (heterogeneous):** Adds a paragraph to Section 6 (or a new Section 4.8) demonstrating protocol robustness.

---

## Decision gate

After Phase A completes, evaluate:

**Green light for NeurIPS (proceed to Phase B):** A1 shows KALAVAI beats 1B monolithic by >5%, AND A3 shows clear degradation with different checkpoints.

**Yellow (proceed cautiously):** One of A1/A3 is strong, the other is marginal. Run Phase B but start thinking about COLM/TMLR as backup venue.

**Red (pivot to TMLR):** A1 shows monolithic catches up at 1B, OR A3 shows initialization doesn't matter. The paper is still strong for TMLR but the NeurIPS bar requires stronger scaling evidence.

Report Phase A results before starting Phase B so the go/no-go decision can be made.
