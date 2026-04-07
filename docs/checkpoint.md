# KALAVAI — Development Checkpoints

---

## Checkpoint 4: NeurIPS Sprint — EXP-19 Complete + Paper Updated (2026-04-08)

**Status**: No experiments running. Next GPU run: FE-03 (18-expert ablation)

### Completed this session

**EXP-19 router-only retry (seed 42):** COMPLETE
- Result: +16.69% gain vs spec, mean div 15.70% — matches prior 3-seed mean +16.71% ±0.07pp
- Confirmed toxic specialists: dialogue (-25.08%, 16 eval chunks), instructions (-16.50%, 28 chunks)
- Chemistry routing collapse: 75% routes to medical specialist
- Results: `results/pythia/ablation_router_summary.json`

**Paper cross-lingual update:** COMPLETE
- All occurrences of +21.76% (2-seed) updated to +21.87% ±0.12pp (3-seed curriculum)
- Rewritten router collapse paragraph → curriculum warm-start explanation
- Appendix per-seed table: seed 42 updated from +6.14% PIVOT to +22.04% GO
- Files: `paper/kalavai_neurips2026_submit.tex`

**LOO analysis (EXP-32) updated:** COMPLETE
- Primary dataset now uses 21.87% (was 21.76%)
- New numbers: LOO-MAE=2.89pp (5-point), cross-lingual residual=+8.43pp
- Results: `results/analysis/loo_analysis.json`

### Decision: skip FE-01/FE-02

Router budget sweep (2k/4k steps) deprioritised. Removing toxic specialists (FE-03) is higher ROI.
The 1000-step baseline at +16.71% is stable; dialogue and instructions have confirmed negative divergence.

### Next GPU run

**FE-03:** 18-expert MoE (drop dialogue + instructions), 1000 steps, 3 seeds (42/137/2026)
Expected: +18–20% gain (negative-divergence specialists removed)
Gate: ≥ +1pp over +16.71% → proceed to FE-04 (replacement domain selection)

---

## Checkpoint 3: NeurIPS Sprint — Paper Fixed + Experiments In Progress (2026-04-07)

**Commit**: `03a297e` on `main`
**Status**: EXP-19 router-only retry running overnight on RunPod

### Completed this session

**Paper fixes (submission-ready):**
- Table 1 caption: all MoE rows now correctly stated as 3-seed means
- Abstract + checklist: 6.9B seed count, sequence length, LR hyperparams corrected
- Supplementary ZIP rebuilt: all absolute local paths removed, pyproject.toml fixed

**EXP-17b (Cross-lingual curriculum warm-start):** COMPLETE
- All 3 seeds GO: **+21.87% ±0.12pp** (was +21.76% on 2 seeds, seed 42 collapsed)
- Router collapse fixed by 100-step domain-pure warm-start before mixed training
- Script: `experiments/kalavai_phase2_exp1_curriculum.py`
- Results: `results/phase2/cross_lingual/curriculum/result_seed{42,137,2026}.json`

**EXP-32 (LOO analysis):** COMPLETE (updated 2026-04-08)
- Primary dataset updated to 21.87% (3-seed curriculum)
- LOO-MAE = 3.82pp (all 6), 2.89pp (excl. cross-lingual), 1.62pp (sensitivity: pre-curriculum)
- Cross-lingual LOO residual: +8.43pp
- Script: `experiments/analysis/loo_analysis.py`
- Results: `results/analysis/loo_analysis.json`

**Code fix — TwentyExpertMoE router training:**
- Old: `_run_one_cpu` rebuilt Pythia-1B from scratch every forward pass → ~35s/step
- New: GPU mode (all 20 on GPU) + CPU-swap fallback (pre-built models) → ~10-15s/step
- Root cause of slowness: mem_get_info VRAM gate returned stale value after eval loops, silently falling to CPU-swap
- Fix: removed gate, always attempts GPU mode, let OOM handler decide
- Tests: `experiments/tests/test_moe_gpu_offload.py` — 13/13 passing

### In progress

**EXP-19 router-only retry:**
- Running: `--router-only --seeds 42,137,2026`, 1,000 steps, GPU mode active
- Expected completion: 2026-04-08 morning
- Results: `results/phase2/twenty_contributor/result_seed{42,137,2026}_router_retry.json`

### Pending (next session)

- FE-03: 18-expert ablation (drop dialogue+instructions) — **next GPU run**
- FE-04/05/06: replacement specialist domains
- LG-01/02/03/04: 6.9B freeze sweep
- FE-03: 18-expert ablation (drop dialogue+instructions)
- FE-04/05/06: replacement specialist domains
- LG-01/02/03/04: 6.9B freeze sweep

### How to resume

```bash
cd C:/Github/Kalavai
git pull
# Check overnight results:
ls results/phase2/twenty_contributor/result_seed*_router_retry.json
python experiments/kalavai_20contributor_experiment.py --router-only --seeds 42,137,2026  # if not done
```

A log of completed milestones for any agent or developer resuming work.

---

## Checkpoint 2: Phase 2 Complete — Cooperative Manager (2026-03-09)

**Commit**: `c083df6` on `main`
**Tests**: 149 passing
**What's new since Checkpoint 1**: 8 tasks (TASK-008 through TASK-015)

### Completed

- **BPE tokenizer** (`coop/tokenizer.py`) — custom implementation, deterministic, save/load via JSON
- **Seed checkpoint** (`coop/seed.py`) — reproducible model initialization via BytesIO trick
- **Transformer model** (`core/model.py`) — GPT-style with RMSNorm, SwiGLU, causal attention, probe extraction
- **CKA reference** (`coop/reference.py`) — extract hidden states at probe layers from seed model
- **Calibration batch** (`coop/calibration.py`) — tokenize corpus into fixed tensor
- **Domain manifest** (`coop/manifest.py`) — 20 default domains, slot claiming, CRUD operations
- **`kalavai coop create`** (`coop/create.py`) — full orchestration of all 6 artifact creation steps
- **`kalavai coop join`** (`coop/join.py`) — copy artifacts, claim slot, store hashes
- **`kalavai coop status`** (`coop/status.py`) — rich table + JSON output

### Architecture Presets

The `create_cooperative()` function maps target_params to transformer configs:

| Target | Depth | d_model | Heads | FFN Ratio | ~Params |
|--------|-------|---------|-------|-----------|---------|
| 14M | 6 | 384 | 6 | 2.75 | ~14M |
| 125M | 12 | 768 | 12 | 2.75 | ~125M |
| 350M | 24 | 1024 | 16 | 2.75 | ~350M |
| 1B | 24 | 2048 | 16 | 2.75 | ~1B |
| 7B | 32 | 4096 | 32 | 2.75 | ~7B |

### How to resume

```bash
cd C:/Github/Kalavai
pip install -e ".[dev]"
python -m pytest -v  # should see 149 passing
```

### What to build next

Phase 3: Module Trainer — the training loop with CKA anchor loss. Start with TASK-016 (model wrapper already exists in `core/model.py` — may just need a thin adapter). The critical path is:

```
TASK-016 (model wrapper) → TASK-017 (CKA anchor loss) → TASK-021 (train start E2E)
                                                       → TASK-022 (submit validation) → TASK-023 (train submit E2E)
```

TASK-018 (data loading) and TASK-019/020 (warnings/telemetry) can be built in parallel with 017.

---

## Checkpoint 1: Phase 1 Complete — Foundation (2026-03-09)

**Commit**: `90932e3` on `main`
**Tests**: 79 passing
**What's new**: 7 tasks (TASK-001 through TASK-007)

### Completed

- **Config parser** (`core/config.py`) — 5 typed dataclasses, from_yaml/to_yaml, full validation
- **Exception hierarchy** (`core/exceptions.py`) — KalavaiError base + 6 specific exceptions
- **Checkpoint handler** (`core/checkpoint.py`) — save/load checkpoint directories, SHA-256 validation
- **CKA computation** (`core/cka.py`) — linear CKA + differentiable cka_loss
- **Hardware detection** (`train/hardware.py`) — CUDA auto-detect with rich output
- **Test infrastructure** (`tests/conftest.py`) — fixtures for config, cooperative dir, mock CUDA
- **CLI smoke tests** (`tests/test_install.py`) — --help, --version, subcommand verification
- **pyproject.toml** — package config with deps, dev tools, CLI entry point

### Key files for any future task

| Need | File | API |
|------|------|-----|
| Load config | `core/config.py` | `CooperativeConfig.from_yaml(path)` |
| Raise errors | `core/exceptions.py` | `ConfigError`, `AlignmentError`, etc. |
| Save/load checkpoints | `core/checkpoint.py` | `save_checkpoint()`, `load_checkpoint()` |
| Compute CKA | `core/cka.py` | `linear_cka(X, Y)`, `cka_loss(h, ref)` |
| Detect GPU | `train/hardware.py` | `detect_hardware()` |
| Test fixtures | `tests/conftest.py` | `cooperative_dir`, `sample_config_yaml`, etc. |

---

## Checkpoint 0: Project Initialization (2026-03-09)

**Commit**: `a1b22cc` on `main`

- Repo initialized with CLAUDE.md, pyproject.toml, .gitignore
- CLI skeleton with all command stubs (coop, train, check, fuse)
- Source layout: `src/kalavai/` with subsystem packages
- Spec files moved to `docs/spec/`
- Task plan created at `docs/tasks.md` (48 tasks across 9 phases)
