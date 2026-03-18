# Kalavai — Open Issues Log

---

## [OPEN] C1/D1 NaN loss on local RTX 5090 — 2026-03-17

### Scripts affected
- `experiments/kalavai_heterogeneous_cooperative.py` (C1)
- `experiments/kalavai_1b_5domain_experiment.py` (D1)

### Symptom
`loss=nan` from step 1 of specialist training. Runs for full step count but produces no learning. Eval (no grad) works fine — base_eval.json has valid loss (2.9817 mixed). Only training (backward pass) produces NaN.

### Environment
- Machine: local RTX 5090 (32GB VRAM)
- torch: 2.12.0.dev20260312+cu128 (nightly)
- CUDA: 12.8
- Arch: Blackwell (GB202)
- transformers: latest (has `torch_dtype` deprecation, new `GradientCheckpointingLayer` in GPT-NeoX)

### Fixes tried
1. Added `dtype=torch.bfloat16` to `from_pretrained` — still NaN
2. Added `torch.amp.autocast("cuda", dtype=torch.bfloat16)` around loss — still NaN

### Key observations
- Base eval (no grad) runs fine and returns valid loss → forward pass OK
- NaN appears from step 1 of backward pass → gradient computation is suspect
- Working 410M experiments (kalavai_pythia_experiment.py) used same approach and worked — but those ran on RunPod or earlier session, NOT on this local machine/torch nightly
- `running_loss` is cumulative (never reset) — once NaN at step 1, display is NaN forever even if later steps are fine

### Hypotheses to investigate
1. **running_loss display bug** — NaN at step 1 poisons the cumulative average forever. Actual weights might be training fine. Check final eval loss to confirm.
2. **torch nightly + Blackwell bug** — torch 2.12.0.dev + RTX 5090 may have an fp32/bf16 issue in backward pass on Blackwell. Try `pip install torch==2.6.0` (stable).
3. **autocast on already-bf16 model** — model loaded in bf16, autocast also forces bf16. Remove autocast, just use `dtype=torch.bfloat16` in from_pretrained only.
4. **New transformers GPT-NeoX** — newer transformers changed GPT-NeoX (added GradientCheckpointingLayer). Could interact badly with frozen layers.

### Next steps to try in the morning
1. First check if the eval after training is NaN or valid — if valid, it's just the display formula
2. If eval is also NaN: try `pip install torch==2.6.0` (stable release) and rerun
3. If stable torch also NaN: remove `autocast` wrapper, use bf16 model only
4. Compare against `kalavai_pythia_experiment.py` line-by-line for the training loop

### Current state
C1 is currently running (PID ~2183) producing NaN loss display. Will complete all conditions but results may be garbage. Check result JSONs in the morning for eval losses.

Result JSONs location: `results/pythia/heterogeneous_cooperative/`
Logs: `logs/c1_log.txt`

