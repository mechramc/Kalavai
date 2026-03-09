# KALAVU — Project Status

**Last Updated**: 2026-03-09
**Branch**: main
**Tests**: 149 passing
**Source Lines**: ~2,350 (src) + ~2,400 (tests)

---

## Overall Progress

| Phase | Status | Tasks | Description |
|-------|--------|-------|-------------|
| 1. Foundation | **Done** | 7/7 | Config, exceptions, checkpoint, CKA, hardware, test infra |
| 2. Cooperative Manager | **Done** | 8/8 | Tokenizer, seed, calibration, reference, manifest, coop CLI |
| 3. Module Trainer | Not started | 0/8 | Training loop, CKA anchor loss, submission |
| 4. Alignment Monitor | Not started | 0/2 | Alignment diagnostics and reporting |
| 5. Fusion Pipeline | Not started | 0/8 | Clustering, MoE fusion, post-training |
| 6. Eval & Publishing | Not started | 0/5 | CORE evaluation, model card, HF Hub |
| 7. Alpha Validation | Not started | 0/2 | 2-module E2E proof-of-concept |
| 8. GitHub Coordination | Not started | 0/4 | Remote repos, Discussions, checkpoint upload |
| 9. v1.0 Polish | Not started | 0/4 | Agent mode, state machine, incremental re-fusion |

**Total**: 15/48 tasks complete (31%)

---

## What Works Today

### CLI Commands (functional)

```bash
kalavu --help                                    # CLI entry point
kalavu --version                                 # 0.1.0
kalavu coop create --name my-coop --modules 5    # Creates cooperative with all artifacts
kalavu coop join ./my-coop --claim-module 1      # Claims a domain slot
kalavu coop status ./my-coop                     # Rich table of module statuses
kalavu coop status ./my-coop --json              # Machine-readable JSON output
```

### CLI Commands (stubbed — not yet wired)

```bash
kalavu coop publish <cooperative>
kalavu train start --module <id>
kalavu train submit --module <id>
kalavu check post
kalavu fuse cluster <cooperative>
kalavu fuse build <cooperative>
kalavu fuse train <cooperative>
```

---

## Module Inventory

### Core (`src/kalavu/core/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `config.py` | 304 | kalavu.yaml schema, typed dataclasses, validation |
| `checkpoint.py` | 307 | Checkpoint save/load, SHA-256 hash validation |
| `model.py` | 221 | GPT-style transformer (RMSNorm, SwiGLU, causal attention) |
| `cka.py` | 77 | Linear CKA computation + differentiable loss |
| `exceptions.py` | ~40 | Exception hierarchy (7 classes) |

### Cooperative Manager (`src/kalavu/coop/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `create.py` | 196 | Orchestrates all 6 cooperative creation steps |
| `tokenizer.py` | 217 | BPE tokenizer (train, encode, decode, save, load) |
| `manifest.py` | 161 | Domain manifest generation + slot management |
| `join.py` | 138 | Artifact download, slot claiming, hash verification |
| `status.py` | 188 | Rich table display + JSON output |
| `calibration.py` | 97 | Calibration batch generation from tokenized corpus |
| `reference.py` | 75 | CKA reference representation computation |
| `seed.py` | 60 | Deterministic seed checkpoint generation |

### Module Trainer (`src/kalavu/train/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `hardware.py` | 87 | CUDA GPU auto-detection |

### CLI

| Module | Lines | Purpose |
|--------|-------|---------|
| `cli.py` | 178 | Click CLI with all command groups wired |

---

## Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_config.py` | 21 | Config parsing, validation, round-trip, defaults |
| `test_fixtures.py` | 15 | Shared fixture validity |
| `test_exceptions.py` | 12 | Exception hierarchy and catch behavior |
| `test_manifest.py` | 10 | Domain manifest CRUD |
| `test_tokenizer.py` | 10 | BPE train, encode/decode, determinism, save/load |
| `test_checkpoint.py` | 9 | Checkpoint save/load, hash validation |
| `test_coop_create.py` | 9 | E2E cooperative creation |
| `test_coop_status.py` | 9 | Status display, JSON output |
| `test_cka.py` | 8 | CKA computation, scale invariance, gradient flow |
| `test_hardware.py` | 8 | CUDA detection (mocked) |
| `test_coop_join.py` | 7 | Slot claiming, artifact copying |
| `test_calibration.py` | 7 | Batch generation, edge cases |
| `test_seed.py` | 13 | Model forward pass, probe extraction, reproducibility |
| `test_reference.py` | 5 | CKA reference computation |
| `test_install.py` | 4 | CLI smoke tests |
| `conftest.py` | — | Shared fixtures (config, cooperative dir, mock CUDA) |

**Total: 149 tests, 0 failures**

---

## Key Technical Decisions

1. **No external nanochat dependency** — built a minimal GPT-style transformer in `core/model.py` (RMSNorm, SwiGLU FFN, causal multi-head attention). Compatible with the spec's architecture config.

2. **No minbpe dependency** — implemented BPE tokenizer from scratch in `coop/tokenizer.py`. Deterministic via fixed seed and tie-breaking.

3. **Reproducible seed checkpoints** — torch.save embeds non-deterministic zip timestamps. Solved by serializing to BytesIO first, then writing raw bytes.

4. **Architecture presets** — `coop create` maps `--target-params` (14M, 125M, 350M, 1B, 7B) to sensible transformer configs automatically.

5. **All tests GPU-independent** — CUDA is mocked in hardware tests. Model tests use CPU with tiny configs.

---

## Next Up: Phase 3 — Module Trainer

The training loop with CKA anchor loss. Key tasks:
- TASK-016: Nanochat model wrapper (probe extraction)
- TASK-017: CKA anchor loss in training loop
- TASK-018: Domain-aware data loading
- TASK-019: Alignment pause/warning system
- TASK-020: Training telemetry
- TASK-021: Wire `kalavu train start` E2E
- TASK-022: Submission validation
- TASK-023: Wire `kalavu train submit` E2E
