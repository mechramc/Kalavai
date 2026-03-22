# KALAVAI 6.9B Hyperparameter Optimization — Research Plan

**Status:** Pre-execution (pending paper submission to arxiv)
**Est. cost:** $108–200 on RunPod A100 80GB
**Est. wall time:** ~35 hours (4 parallel pods)

---

## The Problem

KALAVAI cooperative fusion produces:
- +14.2% improvement at Pythia-410M
- +14.8% improvement at Pythia-1B
- **+2.43% improvement at Pythia-6.9B** ← credibility gap

The 6.9B experiment used a single conservative configuration (1000 steps, freeze=6/32, lr=1e-5) chosen to avoid OOM and instability. The 410M crossover experiment showed improvement grows with specialist training steps up to 5000 steps before plateauing. This strongly suggests the 6.9B result is limited by hyperparameter choice, not a fundamental mechanism failure.

**The question:** What combination of training steps, freeze depth, and learning rate at 6.9B recovers the ~14% improvement seen at smaller scales?

This is a 3-dimensional search space with 48 configurations — too expensive to babysit manually, designed for autoresearch running unattended on rented GPU.

---

## Search Space

| Dimension | Values | Rationale |
|-----------|--------|-----------|
| Training steps | 500, 1000, 2000, 5000 | 410M peaked at 5000 steps; 6.9B used only 1000 |
| Freeze depth | 2, 4, 6, 8 (out of 32 layers) | 6.9B used freeze=6; optimal may differ at scale |
| Learning rate | 5e-6, 1e-5, 2e-5 | 6.9B used lr=1e-5 conservatively; 410M/1B used 2e-5 |

**Total: 4 × 4 × 3 = 48 configurations**

---

## Protocol Per Configuration

1. Load `EleutherAI/pythia-6.9b` at `revision="step10000"`
2. Freeze first `freeze` layers + embedding
3. Train 3 specialists (code, science, fiction) for `steps` steps at `lr`
   - batch_size=1, grad_accum=8, bf16, gradient_checkpointing
   - Datasets: CodeSearchNet, SciQ, PG-19
   - Packed chunks: 512 tokens, 80/10/10 split
4. Divergence check — if any specialist fails to beat base on its own domain, record as `divergence_failed` and skip fusion
5. Train simple linear router (500 steps, lr=1e-3, batch=4)
6. Evaluate MoE fused model on held-out mixed data (EVAL_BATCHES=50, seed=999)
7. Record results JSON
8. Save result

**Use seed=42 only.** This is a hyperparameter search, not a variance study. 3-seed validation happens after best config is found.

---

## Fixed Parameters

```yaml
model: EleutherAI/pythia-6.9b
revision: step10000
batch_size: 1
grad_accum: 8
precision: bf16
gradient_checkpointing: true
seq_len: 512
warmup_fraction: 0.1
weight_decay: 0.1
gradient_clip: 1.0
router: nn.Linear(4096, 3, bias=False)
router_steps: 500
router_lr: 1e-3
eval_batches: 50
eval_shuffle_seed: 999
seed: 42
domains: [code, science, fiction]
```

---

## Time and Cost Estimates

| Steps | Time per config | Count | Total |
|-------|----------------|-------|-------|
| 500 | ~45 min | 12 | ~9 hrs |
| 1000 | ~85 min | 12 | ~17 hrs |
| 2000 | ~160 min | 12 | ~32 hrs |
| 5000 | ~385 min | 12 | ~77 hrs |

**Total: ~135 A100-hours**

| Pricing | Cost |
|---------|------|
| $1.49/hr spot | ~$108 |
| $2.49/hr on-demand | ~$200 |

### Parallel Pod Strategy (35 hrs wall time)

| Pod | Configs | Wall time |
|-----|---------|-----------|
| Pod 1 | All 500-step configs | ~9 hrs |
| Pod 2 | All 1000-step configs | ~17 hrs |
| Pod 3 | All 2000-step configs | ~32 hrs |
| Pod 4 | Top 5000-step configs (selective) | ~40 hrs |

> Wait for Phase 1 results before committing Pod 4 — you may not need all 5000-step configs.

---

## Hardware: RunPod, Not the 5090

**The RTX 5090 (32GB VRAM) cannot run this experiment.**

| Component | VRAM |
|-----------|------|
| Model weights (bf16) | ~13.8 GB |
| Gradients | ~13.8 GB |
| AdamW optimizer states | ~27.6 GB |
| **Total** | **~55 GB** |

Even with gradient checkpointing + frozen layers, 32GB is ~23GB short. This protocol was designed for **A100 80GB**.

**Use: RunPod A100 80GB SXM** — $2.49/hr on-demand, $1.49/hr spot.

Setup notes:
- Use `runpodctl` to manage pods programmatically
- Mount a **network volume** to persist results — pod termination wipes local storage
- Use `ghcr.io/runpod-workers/pytorch` base image

---

## Phased Autoresearch Protocol

### Phase 1: Quick scan (500-step, ~9 hrs)
Run all 12 configs. Identify which freeze/lr combinations produce highest improvement. Eliminate clearly bad regions.

### Phase 2: Medium scan (1000-step, ~17 hrs)
Run all 12 configs. Compare against current baseline (+2.43%). Check if any config already reaches 10%+.

### Phase 3: Deep scan (2000-step, ~32 hrs)
Focus on top freeze/lr combinations from phases 1-2. If phase 2 already found +10%+, run only 3-4 most promising configs.

### Phase 4: Extended scan (5000-step, selective)
Only run configs where improvement was still growing at 2000 steps. Skip plateaued/declined configs. Most expensive phase — autoresearch must be selective.

### Phase 5: Validation
Take the best configuration, run 3 seeds (42, 137, 2026), report mean ± std. This is the number that goes in the paper.

### Adaptive Early Stopping Rules
- Training loss increases >50% from step 100→200 → abort, record `training_diverged`
- Divergence check fails → skip fusion
- Improvement < -5% → don't test longer step counts for same freeze/lr

---

## Autoresearch YAML Config

```yaml
project: kalavai_6b_hyperparam_search
hypothesis: >
  The KALAVAI cooperative fusion mechanism produces +14% improvement at 410M-1B
  but only +2.43% at 6.9B with conservative hyperparameters. The optimal
  combination of training steps, freeze depth, and learning rate at 6.9B
  will recover improvement to 10%+ range.

search_space:
  steps: [500, 1000, 2000, 5000]
  freeze_layers: [2, 4, 6, 8]
  learning_rate: [5e-6, 1e-5, 2e-5]

fixed_params:
  model: EleutherAI/pythia-6.9b
  revision: step10000
  batch_size: 1
  grad_accum: 8
  precision: bf16
  gradient_checkpointing: true
  seq_len: 512
  warmup_fraction: 0.1
  weight_decay: 0.1
  gradient_clip: 1.0
  router: nn.Linear(4096, 3, bias=False)
  router_steps: 500
  router_lr: 1e-3
  eval_batches: 50
  eval_shuffle_seed: 999
  seed: 42
  domains: [code, science, fiction]

success_metric: improvement_pct (higher is better)
failure_condition: divergence_check_failed

priority_order: >
  Run cheap configs first (500 steps) to establish baseline,
  then 1000 steps, then 2000, then 5000. Within each step count,
  prioritize freeze=4 and lr=1e-5 (closest to known-working configs)
  before exploring extremes.
```

---

## Output Format

### Per-Configuration Result JSON
```json
{
  "config": {"steps": 2000, "freeze": 4, "lr": 1e-5},
  "divergence_passed": true,
  "improvement_pct": 12.45,
  "moe_mixed_loss": 2.3567,
  "best_individual_mixed_loss": 2.6922,
  "base_mixed_loss": 2.7004,
  "training_time_seconds": 9600,
  "specialist_losses": {
    "code": {"own_domain": 1.58, "base_own": 1.84},
    "science": {"own_domain": 2.21, "base_own": 2.59},
    "fiction": {"own_domain": 2.39, "base_own": 2.69}
  },
  "router_final_loss": 1.85,
  "timestamp": "2026-03-20T04:30:00Z"
}
```

Save to: `results/pythia_6b/hyperparam_search/steps{S}_freeze{F}_lr{LR}.json`

### Figures to Generate
- `fig_heatmap_steps_vs_freeze.png` — Improvement heatmap at best LR
- `fig_heatmap_steps_vs_lr.png` — Improvement heatmap at best freeze
- `fig_improvement_surface.png` — 3D surface (steps × freeze × lr → improvement)
- `fig_optimal_config_comparison.png` — Bar chart: 410M vs 1B vs 6.9B-optimal vs 6.9B-original

---

## Success Criteria

| Outcome | Result | Paper action |
|---------|--------|-------------|
| Best case | +10–15% at optimal config | "With optimized hyperparameters, 6.9B matches smaller scales" |
| Good case | +6–8% at optimal config | Mechanism works at 6.9B with tuning; partial gap closure noted |
| Informative | No config exceeds +5% | Scale-dependent effect; suggests more data/training needed at 6.9B |

All three outcomes are publishable. The search eliminates hyperparameter choice as a confound.

---

## Prerequisites Checklist

- [ ] KALAVAI paper submitted to arxiv (**must come first — do not delay paper for this**)
- [ ] RunPod account funded with $150–200
- [ ] `kalavu_6b_hyperparam_search.py` built with `--steps`, `--freeze`, `--lr` CLI args
- [ ] RunPod network volume configured for result persistence
- [ ] Autoresearch configured with YAML above
- [ ] Git repo clean with all current experiments committed

---

## Code Needed: `kalavu_6b_hyperparam_search.py`

Create from existing `kalavu_pythia_6b_experiment.py`, parameterized:

```bash
python kalavu_6b_hyperparam_search.py --steps 2000 --freeze 4 --lr 1e-5
# Saves: results/pythia_6b/hyperparam_search/steps2000_freeze4_lr1e-5.json
```

Keep separate from existing experiment scripts. Accepts one config, runs it, saves JSON, exits cleanly.
