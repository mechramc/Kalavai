#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI Phase 2, Experiment 2: Private-Domain Fusion
=====================================================
Medical (PubMed), Legal (lex_glue/eurlex), Patents (big_patent).

Design:
  - 3 specialists trained independently on private-domain data
  - MoE fusion via 2-layer MLP router (matches corrected 410M +7.70%)
  - Monolithic baseline: 6000 steps on shuffled mix
  - FREEZE_LAYERS=0 (CRITICAL: below 5k steps, freeze hurts)
  - Eval: per-domain equal-weight average (Bug A + Bug B fixed)

Stop/go printed at end of each seed:
  → GO if diverge>15% AND gain>7%
  → PIVOT if diverge>15% but gain<7% (router problem)
  → STOP if diverge<10% (insufficient divergence for fusion)

Datasets:
  - medical: ccdv/pubmed-summarization (text_key: article)
  - legal:   lex_glue eurlex          (text_key: text)
  - patent:  big_patent a             (text_key: description)
"""

import copy
import json
import os
import statistics
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import sys; sys.path.insert(0, str(Path(__file__).parent))
from kalavai_eval_utils import eval_all_domains, eval_loss_domain, PackedChunkDataset, _collate, chunks_to_dataset, SEQ_LEN

# ============================================================================
# Config
# ============================================================================

MODEL_ID   = "EleutherAI/pythia-410m"
REVISION   = "step10000"
HIDDEN_SIZE = 1024

FREEZE_LAYERS = 0        # CRITICAL — phase2 requirement, below 5k steps freeze hurts
LR            = 2e-5
WEIGHT_DECAY  = 0.1
MAX_STEPS     = 2000
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
GRADIENT_CLIP = 1.0
WARMUP_FRACTION = 0.1

ROUTER_STEPS  = 500
ROUTER_LR     = 1e-3
ROUTER_BATCH  = 4
EVAL_BATCH_SIZE = 4
EVAL_BATCHES  = 50

MONOLITHIC_STEPS = 6000

N_SAMPLES    = 3000      # per domain
DOMAINS      = ["medical", "legal", "patent"]
SEEDS        = [42, 137, 2026]

RESULTS_DIR    = Path("results/phase2/private_domain")
CHECKPOINT_DIR = Path("checkpoints/phase2/private_domain")

# ============================================================================
# Data loading
# ============================================================================

def load_medical_texts(n):
    from datasets import load_dataset
    print(f"  Loading medical (pubmed, n={n})...")
    ds = load_dataset("ccdv/pubmed-summarization", split="train", streaming=True)
    texts = [s["article"][:5000] for _, s in zip(range(n), ds)]
    print(f"    {len(texts)} samples")
    return texts


def load_legal_texts(n):
    from datasets import load_dataset
    print(f"  Loading legal (lex_glue/eurlex, n={n})...")
    ds = load_dataset("lex_glue", "eurlex", split="train", streaming=True)
    texts = [s["text"][:5000] for _, s in zip(range(n), ds)]
    print(f"    {len(texts)} samples")
    return texts


def load_patent_texts(n):
    from datasets import load_dataset
    print(f"  Loading patent (big_patent/a, n={n})...")
    ds = load_dataset("big_patent", "a", split="train", streaming=True)
    texts = [s["description"][:5000] for _, s in zip(range(n), ds)]
    print(f"    {len(texts)} samples")
    return texts


def load_all_data(n_samples, tokenizer):
    """Load and split data for all 3 domains (80/10/10)."""
    print("\nLoading data...")
    loaders = {
        "medical": load_medical_texts,
        "legal":   load_legal_texts,
        "patent":  load_patent_texts,
    }
    train_chunks, held_out_chunks = {}, {}
    for domain, loader_fn in loaders.items():
        texts = loader_fn(n_samples)
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        n = len(ds_full.chunks)
        a = int(n * 0.8)
        b = int(n * 0.9)
        train_chunks[domain]    = ds_full.chunks[:a]
        held_out_chunks[domain] = ds_full.chunks[b:]
        print(f"  {domain}: total={n}, train={len(train_chunks[domain])}, held_out={len(held_out_chunks[domain])}")
        if len(train_chunks[domain]) < 1000:
            print(f"  WARNING: {domain} has <1000 train chunks — results may be noisy")
    return train_chunks, held_out_chunks


# ============================================================================
# Architecture — ThreeExpertMoE with 2-layer MLP router
# Matches corrected 410M +7.70% which used router_type="mlp"
# ============================================================================

class ThreeExpertMoE(nn.Module):
    """
    Sequence-level MoE over three specialist models.
    Router: mean of last hidden states -> 2-layer MLP (H->256->ReLU->3) -> softmax gates.
    Specialists are frozen; only router is trained.
    """
    def __init__(self, spec_a, spec_b, spec_c, hidden_size: int):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for p in list(self.spec_a.parameters()) + list(self.spec_b.parameters()) + list(self.spec_c.parameters()):
            p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 3, bias=False),
        )

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach()
        h_pooled = out.hidden_states[-1].detach().mean(dim=1).float()
        return logits, h_pooled

    def forward(self, input_ids, labels=None):
        la, ha = self._run_specialist(self.spec_a, input_ids)
        lb, hb = self._run_specialist(self.spec_b, input_ids)
        lc, hc = self._run_specialist(self.spec_c, input_ids)
        gates = torch.softmax(self.router((ha + hb + hc) / 3.0), dim=-1)
        fused = (gates[:, 0:1, None] * la
               + gates[:, 1:2, None] * lb
               + gates[:, 2:3, None] * lc)
        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, fused, gates


# ============================================================================
# Training helpers
# ============================================================================

def _batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_specialist(model, domain: str, train_chunks: list, seed: int, device: str):
    """Train a domain specialist. FREEZE_LAYERS=0 — all layers trainable."""
    set_seed(seed)
    model.train()
    # No layer freezing (FREEZE_LAYERS=0)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M (100%)")

    dataset = chunks_to_dataset(train_chunks)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

    step, accum = 0, 0
    running_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**_batch_to_device(batch, device))
            loss = out.loss / GRAD_ACCUM
        loss.backward()
        accum += 1
        running_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step + 1) / warmup_steps
            optimizer.step()
            if step >= warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
            accum = 0
            step += 1
            if step % 200 == 0 or step == MAX_STEPS:
                avg = running_loss / step
                print(f"    [{domain}] step {step}/{MAX_STEPS} | loss {avg:.4f} | {time.time()-t0:.0f}s")
    model.eval()
    print(f"  {domain} done in {time.time()-t0:.0f}s")


def train_monolithic(model, all_train_chunks: dict, device: str):
    """Train monolithic baseline on shuffled mix of all 3 domains (6000 steps)."""
    import random
    print(f"\n  Training monolithic ({MONOLITHIC_STEPS} steps on mixed data)...")
    all_chunks = []
    for chunks in all_train_chunks.values():
        all_chunks.extend(chunks)
    random.shuffle(all_chunks)
    dataset = chunks_to_dataset(all_chunks)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    warmup_steps = int(MONOLITHIC_STEPS * WARMUP_FRACTION)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MONOLITHIC_STEPS - warmup_steps)

    step, accum = 0, 0
    running_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MONOLITHIC_STEPS:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**_batch_to_device(batch, device))
            loss = out.loss / GRAD_ACCUM
        loss.backward()
        accum += 1
        running_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step + 1) / warmup_steps
            optimizer.step()
            if step >= warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
            accum = 0
            step += 1
            if step % 500 == 0 or step == MONOLITHIC_STEPS:
                avg = running_loss / step
                print(f"    [monolithic] step {step}/{MONOLITHIC_STEPS} | loss {avg:.4f} | {time.time()-t0:.0f}s")
    model.eval()


def train_router(moe: ThreeExpertMoE, train_chunks_by_domain: dict, device: str):
    """Train router on mixed data from all 3 domains."""
    all_chunks = []
    for chunks in train_chunks_by_domain.values():
        all_chunks.extend(chunks)
    combined = chunks_to_dataset(all_chunks)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"\n  Training router ({ROUTER_STEPS} steps, {len(combined)} chunks)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        loss, _, _ = moe(input_ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step}/{ROUTER_STEPS}: loss={loss.item():.4f}")
    moe.eval()


@torch.no_grad()
def eval_router_distribution(moe, held_out_by_domain, device, n_batches=20):
    moe.eval()
    results = {}
    for domain, ds in held_out_by_domain.items():
        loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            drop_last=True, collate_fn=_collate)
        gate_sums = [0.0, 0.0, 0.0]
        count = 0
        for batch in loader:
            if count >= n_batches: break
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            _, _, gates = moe(input_ids, labels=labels)
            for i in range(3):
                gate_sums[i] += gates[:, i].mean().item()
            count += 1
        results[domain] = [round(g / max(count, 1), 4) for g in gate_sums]
    return results


def weight_average(models):
    """Equal-weight average of model parameters."""
    state_dicts = [{k: v.cpu().float() for k, v in m.state_dict().items()} for m in models]
    avg_state = {k: sum(sd[k] for sd in state_dicts) / len(state_dicts) for k in state_dicts[0]}
    avg_state = {k: v.to(torch.bfloat16) for k, v in avg_state.items()}
    avg = copy.deepcopy(models[0]).cpu()
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg


# ============================================================================
# Run one seed
# ============================================================================

def _load_base(device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


def _load_checkpoint(path, device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def run_seed(seed, tokenizer, device, train_chunks, held_out_chunks):
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    held_out_sets = {d: chunks_to_dataset(held_out_chunks[d]) for d in DOMAINS}
    eval_matrix = {}

    # ── Base model ──────────────────────────────────────────────────────────
    print("\n[base]")
    base = _load_base(device)
    eval_matrix["base"] = eval_all_domains(base, held_out_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
    del base; torch.cuda.empty_cache()

    # ── Monolithic (trained on mixed data) ───────────────────────────────────
    mono_path = CHECKPOINT_DIR / f"monolithic_seed{seed}.pt"
    if mono_path.exists():
        print(f"\n[monolithic]  loading {mono_path}")
        mono = _load_checkpoint(mono_path, device)
    else:
        print(f"\n[monolithic]  training from scratch (seed={seed}, {MONOLITHIC_STEPS} steps)...")
        mono = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        set_seed(seed)
        train_monolithic(mono, train_chunks, device)
        torch.save(mono.state_dict(), mono_path)
        print(f"  Checkpoint saved: {mono_path}")
    eval_matrix["monolithic"] = eval_all_domains(mono, held_out_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
    del mono; torch.cuda.empty_cache()

    # ── Specialists ──────────────────────────────────────────────────────────
    specialists = {}
    for domain in DOMAINS:
        ckpt_path = CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"
        if ckpt_path.exists():
            print(f"\n[{domain}_spec]  loading {ckpt_path}")
            spec = _load_checkpoint(ckpt_path, device)
        else:
            print(f"\n[{domain}_spec]  training (seed={seed}, {MAX_STEPS} steps)...")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            train_specialist(spec, domain, train_chunks[domain], seed, device)
            torch.save(spec.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")
        specialists[domain] = spec

    for domain, spec in specialists.items():
        print(f"\n[{domain}_spec]")
        eval_matrix[f"{domain}_spec"] = eval_all_domains(spec, held_out_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)

    # ── Weight average ────────────────────────────────────────────────────────
    print("\n[weight_avg]  computing weight average...")
    wa = weight_average(list(specialists.values())).to(device)
    eval_matrix["weight_avg"] = eval_all_domains(wa, held_out_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
    del wa; torch.cuda.empty_cache()

    # ── MoE ────────────────────────────────────────────────────────────────────
    print(f"\n[moe]  building + training router ({ROUTER_STEPS} steps)...")
    spec_list = [specialists[d] for d in DOMAINS]
    moe = ThreeExpertMoE(*spec_list, HIDDEN_SIZE).to(device)
    train_router(moe, train_chunks, device)
    moe.eval()

    print("\n[moe eval]")
    eval_matrix["moe"] = eval_all_domains(moe, held_out_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES, is_fused=True)

    router_dist = eval_router_distribution(moe, held_out_sets, device)
    print(f"\n  Router gate distribution:")
    for d, gates in router_dist.items():
        print("    %s: %s" % (d, "  ".join("%s=%.4f" % (DOMAINS[i], gates[i]) for i in range(3))))

    del moe
    for spec in specialists.values(): del spec
    torch.cuda.empty_cache()

    # ── Metrics ────────────────────────────────────────────────────────────────
    def eq(k): return eval_matrix[k]["equal_weight_avg"]

    base_eq       = eq("base")
    mono_eq       = eq("monolithic")
    moe_eq        = eq("moe")
    wa_eq         = eq("weight_avg")
    best_spec_eq  = min(eq(f"{d}_spec") for d in DOMAINS)
    best_spec_dom = min(DOMAINS, key=lambda d: eq(f"{d}_spec"))

    # Divergence: mean per-domain divergence of specialists from base
    domain_divs = []
    for d in DOMAINS:
        spec_eq = eq(f"{d}_spec")
        # Per-domain divergence: (base_loss_on_domain - spec_loss_on_domain) / base_loss_on_domain
        base_d = eval_matrix["base"].get(d, base_eq)
        spec_d = eval_matrix[f"{d}_spec"].get(d, spec_eq)
        div = (base_d - spec_d) / base_d * 100
        domain_divs.append(div)
    mean_divergence = round(statistics.mean(domain_divs), 2)
    gain_vs_spec = round((best_spec_eq - moe_eq) / best_spec_eq * 100, 4)

    metrics = {
        "base_equal_weight":       round(base_eq, 6),
        "monolithic_equal_weight": round(mono_eq, 6),
        "best_spec_equal_weight":  round(best_spec_eq, 6),
        "best_spec_domain":        best_spec_dom,
        "weight_avg_equal_weight": round(wa_eq, 6),
        "moe_equal_weight":        round(moe_eq, 6),
        "improvement_vs_spec":     gain_vs_spec,
        "improvement_vs_base":     round((base_eq - moe_eq) / base_eq * 100, 4),
        "improvement_vs_mono":     round((mono_eq - moe_eq) / mono_eq * 100, 4),
        "mean_divergence":         mean_divergence,
        "per_domain_divergence":   {d: round(domain_divs[i], 2) for i, d in enumerate(DOMAINS)},
    }

    # ── Stop / Go ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"STOP/GO [seed={seed}]:")
    print(f"  Mean divergence: {mean_divergence:.2f}%  |  Fusion gain vs spec: {gain_vs_spec:+.2f}%")
    if mean_divergence > 15 and gain_vs_spec > 7:
        verdict = "GO"
        reason  = "diverge>15% AND gain>7%"
    elif mean_divergence > 15 and gain_vs_spec <= 7:
        verdict = "PIVOT"
        reason  = "diverge>15% but gain<7% (router problem)"
    else:
        verdict = "STOP"
        reason  = f"mean divergence {mean_divergence:.2f}% < 10% (insufficient for fusion)"
    print(f"  → {verdict} ({reason})")
    print(f"{'='*50}")

    result = {
        "seed":            seed,
        "model_id":        MODEL_ID,
        "revision":        REVISION,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_batches":    EVAL_BATCHES,
        "eval_method":     "per-domain-separate-then-equal-weight-avg",
        "domains":         DOMAINS,
        "eval_matrix":     {k: {dk: dv for dk, dv in v.items() if not dk.startswith("_")}
                            for k, v in eval_matrix.items()},
        "metrics":         metrics,
        "router_distribution": router_dist,
        "stop_go":         {"verdict": verdict, "reason": reason},
        "config": {
            "freeze_layers": FREEZE_LAYERS,
            "lr":            LR,
            "max_steps":     MAX_STEPS,
            "batch_size":    BATCH_SIZE,
            "grad_accum":    GRAD_ACCUM,
            "monolithic_steps": MONOLITHIC_STEPS,
            "router_steps":  ROUTER_STEPS,
            "router_lr":     ROUTER_LR,
            "domains":       DOMAINS,
        },
    }
    return result


# ============================================================================
# Main
# ============================================================================

def print_results_table(result):
    domains = result["domains"]
    matrix  = result["eval_matrix"]
    print(f"\n{'='*70}")
    print(f"RESULTS — seed={result['seed']}")
    print(f"  Equal-weight avg = ({' + '.join(domains)}) / {len(domains)}")
    print(f"  batch_size={result['eval_batch_size']} for ALL models")
    print(f"{'='*70}")
    header = f"{'Model':<20}" + "".join(f"{d:>10}" for d in domains) + f"{'Eq.Avg':>10}"
    print(header)
    print("-" * len(header))
    for mk, losses in matrix.items():
        row = f"{mk:<20}" + "".join(f"{losses.get(d, float('nan')):>10.4f}" for d in domains)
        row += f"{losses.get('equal_weight_avg', float('nan')):>10.4f}"
        print(row)
    print()
    m = result["metrics"]
    print(f"  MoE vs best spec:   {m['improvement_vs_spec']:+.2f}%")
    print(f"  MoE vs base:        {m['improvement_vs_base']:+.2f}%")
    print(f"  MoE vs monolithic:  {m['improvement_vs_mono']:+.2f}%")
    print(f"  Mean divergence:    {m['mean_divergence']:.2f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nKALAVAI Phase 2 Experiment 2 — Private Domain Fusion")
    print(f"  model:       {MODEL_ID} @ {REVISION}")
    print(f"  domains:     {DOMAINS}")
    print(f"  freeze_layers: {FREEZE_LAYERS}")
    print(f"  seeds:       {SEEDS}")
    print(f"  device:      {device}")
    if device == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:        {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data once (deterministic, seed-independent)
    train_chunks, held_out_chunks = load_all_data(N_SAMPLES, tokenizer)

    all_results = []
    for seed in SEEDS:
        result = run_seed(seed, tokenizer, device, train_chunks, held_out_chunks)
        print_results_table(result)

        out_path = RESULTS_DIR / f"result_seed{seed}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path}")
        all_results.append(result)

    # Multi-seed summary
    if len(all_results) > 1:
        imps = [r["metrics"]["improvement_vs_spec"] for r in all_results]
        divs = [r["metrics"]["mean_divergence"] for r in all_results]
        print(f"\n{'='*70}")
        print(f"MULTI-SEED SUMMARY")
        print(f"{'='*70}")
        print(f"{'Seed':<8} {'vs spec':>10} {'vs base':>10} {'mean div':>12} {'verdict':>10}")
        print("-" * 52)
        for r in all_results:
            m = r["metrics"]
            print(f"{r['seed']:<8} {m['improvement_vs_spec']:>+9.2f}%  {m['improvement_vs_base']:>+9.2f}%  {m['mean_divergence']:>10.2f}%  {r['stop_go']['verdict']:>10}")
        print("-" * 52)
        print(f"{'Mean':<8} {statistics.mean(imps):>+9.2f}%  {'':>10}  {statistics.mean(divs):>10.2f}%")
        if len(imps) > 1:
            print(f"{'Std':<8} {statistics.stdev(imps):>9.2f}%")

        verdicts = [r["stop_go"]["verdict"] for r in all_results]
        final = "GO" if all(v == "GO" for v in verdicts) else ("PIVOT" if any(v == "PIVOT" for v in verdicts) else "STOP")
        print(f"\nFINAL VERDICT: {final}  (per-seed: {verdicts})")

        summary = {
            "experiment":    "phase2_exp2_private_domain",
            "domains":       DOMAINS,
            "model":         MODEL_ID,
            "seeds":         SEEDS,
            "per_seed":      all_results,
            "summary": {
                "improvement_mean": round(statistics.mean(imps), 4),
                "improvement_std":  round(statistics.stdev(imps) if len(imps) > 1 else 0.0, 4),
                "divergence_mean":  round(statistics.mean(divs), 2),
            },
            "final_verdict": final,
        }
        summary_path = RESULTS_DIR / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved summary: {summary_path}")

    print(f"\n{'='*70}")
    print(f"PHASE 2 EXP 2 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
