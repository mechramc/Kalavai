#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: 410M Corrected Evaluation (Eval-Only, No Retraining)
=============================================================
Reloads existing seed=42 checkpoints from the main 410M experiment and
re-evaluates everything with a CONSISTENT batch_size=4 across all models
(base, specialists, weight_avg, MoE).

Root cause of prior inconsistency:
  kalavai_pythia_experiment.py line 1024: bs = 2 if is_fused else 4
  This evaluated MoE on 100 chunks (all code) vs specialists on 200 chunks
  (code + science) due to shuffle=False, drop_last=True with mixed held_out
  ordered as: code first (157 chunks), then science (67), then fiction (778).

Fix: batch_size=4 for ALL models. Same 200 chunks for everyone.

Outputs corrected result to:
  results/pythia/corrected_eval_seed42.json

Usage:
  cd /c/Github/Kalavai
  python experiments/kalavai_410m_corrected_eval.py
"""

import copy
import json
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ============================================================================
# Config — must match main experiment exactly
# ============================================================================

MODEL_ID    = "EleutherAI/pythia-410m"
REVISION    = "step10000"
HIDDEN_SIZE = 1024
FREEZE_LAYERS = 4
SEQ_LEN     = 512
DOMAINS     = ["code", "science", "fiction"]
N_SAMPLES   = 3000
EVAL_BATCHES = 50
EVAL_BATCH_SIZE = 4   # CONSISTENT for ALL models — this is the fix

CHECKPOINT_DIR = Path("checkpoints/pythia")
RESULTS_DIR    = Path("results/pythia")

# ============================================================================
# Dataset (identical to main experiment)
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated), return_tensors="pt", truncation=False,
        )["input_ids"][0]
        n_chunks = len(full) // seq_len
        self.chunks = [full[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]

    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
    }

def make_dataset_from_chunks(chunks):
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds

def split_chunks(chunks, train_frac=0.8, indist_frac=0.1):
    n = len(chunks)
    a = int(n * train_frac)
    b = int(n * (train_frac + indist_frac))
    return chunks[:a], chunks[a:b], chunks[b:]

# ============================================================================
# Data loading (identical to main experiment)
# ============================================================================

def load_code_texts(n):
    from datasets import load_dataset
    print(f"  Loading code (n={n})...")
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) > 200:
            texts.append(content)
        if len(texts) >= n: break
    print(f"    {len(texts)} samples")
    return texts

def load_science_texts(n):
    from datasets import load_dataset
    print(f"  Loading science (n={n})...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        content = (item.get("support", "") + "\n" +
                   item.get("question", "") + "\n" +
                   item.get("correct_answer", ""))
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n: break
    print(f"    {len(texts)} samples")
    return texts

def load_fiction_texts(n):
    from datasets import load_dataset
    print(f"  Loading fiction (n={n})...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:5000]
        if len(content) >= 500:
            texts.append(content)
        if len(texts) >= n: break
    print(f"    {len(texts)} samples")
    return texts

# ============================================================================
# ThreeExpertMoE (identical to main experiment)
# ============================================================================

class ThreeExpertMoE(nn.Module):
    def __init__(self, spec_a, spec_b, spec_c, hidden_size=HIDDEN_SIZE):
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
        last_h = out.hidden_states[-1].detach()
        h_pooled = last_h.mean(dim=1).float()
        return logits, h_pooled

    def forward(self, input_ids, labels=None):
        logits_a, h_a = self._run_specialist(self.spec_a, input_ids)
        logits_b, h_b = self._run_specialist(self.spec_b, input_ids)
        logits_c, h_c = self._run_specialist(self.spec_c, input_ids)
        h_avg = (h_a + h_b + h_c) / 3.0
        gates = torch.softmax(self.router(h_avg), dim=-1)
        fused = (
            gates[:, 0:1, None] * logits_a
            + gates[:, 1:2, None] * logits_b
            + gates[:, 2:3, None] * logits_c
        )
        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return loss, fused, gates

# ============================================================================
# Eval — CONSISTENT batch_size for all models
# ============================================================================

@torch.no_grad()
def eval_loss(model, dataset, device, is_fused=False):
    """
    Evaluate model on dataset. batch_size=4 for ALL models.
    This is the corrected version — no batch_size asymmetry.
    """
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(input_ids, labels=labels)
        else:
            loss = model(input_ids=input_ids, labels=labels).loss
        if loss is not None:
            total += loss.item()
            count += 1
    return total / count if count > 0 else float("inf")


@torch.no_grad()
def eval_router_distribution(moe, eval_datasets, device, n_batches=20):
    moe.eval()
    results = {}
    for domain, ds in eval_datasets.items():
        if domain == "mixed": continue
        loader = DataLoader(ds, batch_size=4, shuffle=False,
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

# ============================================================================
# Weight average
# ============================================================================

def weight_average_three(spec_a, spec_b, spec_c):
    print("  Computing weight average...")
    sa = {k: v.cpu().float() for k, v in spec_a.state_dict().items()}
    sb = {k: v.cpu().float() for k, v in spec_b.state_dict().items()}
    sc = {k: v.cpu().float() for k, v in spec_c.state_dict().items()}
    avg_state = {k: ((sa[k] + sb[k] + sc[k]) / 3.0).to(torch.bfloat16) for k in sa}
    avg = copy.deepcopy(spec_a).cpu()
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg

# ============================================================================
# Router training (identical to main experiment)
# ============================================================================

def train_router(moe, train_datasets, device,
                 router_steps=500, router_lr=1e-3, router_batch=4):
    all_chunks = []
    for ds in train_datasets.values():
        all_chunks.extend(ds.chunks)
    combined = make_dataset_from_chunks(all_chunks)
    optimizer = AdamW(moe.router.parameters(), lr=router_lr)
    loader = DataLoader(combined, batch_size=router_batch, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"  Training router ({router_steps} steps, {len(combined)} chunks)...")
    for step in range(1, router_steps + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        loss, _, _ = moe(input_ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == router_steps:
            print(f"    Router step {step:3d}/{router_steps}: loss={loss.item():.4f}")
    moe.eval()

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: 410M Corrected Evaluation (Eval-Only)")
    print(f"  batch_size={EVAL_BATCH_SIZE} for ALL models (fix for asymmetry bug)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify checkpoints exist
    print("\nVerifying checkpoints...")
    missing = []
    for domain in DOMAINS:
        ckpt = CHECKPOINT_DIR / f"{domain}_specialist_seed42.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
        else:
            size_mb = ckpt.stat().st_size / 1e6
            print(f"  {domain}_specialist_seed42.pt — {size_mb:.0f}MB ✓")
    monolithic_ckpt = CHECKPOINT_DIR / "monolithic_seed42.pt"
    if monolithic_ckpt.exists():
        print(f"  monolithic_seed42.pt — {monolithic_ckpt.stat().st_size/1e6:.0f}MB ✓")
    else:
        print(f"  monolithic_seed42.pt — NOT FOUND (will skip)")
    if missing:
        print(f"\n[ERROR] Missing checkpoints: {missing}")
        print("  Run kalavai_pythia_experiment.py first.")
        return

    # ── Load tokenizer ────────────────────────────────────────────────────
    print(f"\nLoading tokenizer ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load and split data (same as main experiment) ─────────────────────
    print("\nLoading data...")
    code_texts    = load_code_texts(N_SAMPLES)
    science_texts = load_science_texts(N_SAMPLES)
    fiction_texts = load_fiction_texts(N_SAMPLES)

    print("\nPacking and splitting (80/10/10)...")
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    # Build eval datasets
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    # Report exactly which chunks are evaluated for mixed at batch_size=4
    n_mixed_chunks = min(EVAL_BATCHES * EVAL_BATCH_SIZE, len(mixed_held))
    code_count    = len(all_domain_chunks["code"]["held_out"])
    science_count = len(all_domain_chunks["science"]["held_out"])
    print(f"\nMixed eval composition at batch_size={EVAL_BATCH_SIZE}, EVAL_BATCHES={EVAL_BATCHES}:")
    print(f"  Total mixed chunks available: {len(mixed_held)}")
    print(f"  Chunks evaluated: {n_mixed_chunks}")
    code_in_eval    = min(n_mixed_chunks, code_count)
    science_in_eval = min(max(0, n_mixed_chunks - code_count), science_count)
    fiction_in_eval = max(0, n_mixed_chunks - code_count - science_count)
    print(f"  Code: {code_in_eval} ({100*code_in_eval/n_mixed_chunks:.1f}%)")
    print(f"  Science: {science_in_eval} ({100*science_in_eval/n_mixed_chunks:.1f}%)")
    print(f"  Fiction: {fiction_in_eval} ({100*fiction_in_eval/n_mixed_chunks:.1f}%)")

    # ── Load base model ────────────────────────────────────────────────────
    print(f"\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    base_model.eval()

    # ── Load specialists ───────────────────────────────────────────────────
    print("\nLoading seed=42 specialists...")
    specialists = {}
    for domain in DOMAINS:
        ckpt_path = CHECKPOINT_DIR / f"{domain}_specialist_seed42.pt"
        spec = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        spec.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        spec.eval()
        specialists[domain] = spec
        print(f"  Loaded {domain}")

    # ── Weight average ─────────────────────────────────────────────────────
    weight_avg = weight_average_three(
        specialists["code"], specialists["science"], specialists["fiction"]
    ).to(device)

    # ── Load/train MoE ─────────────────────────────────────────────────────
    print("\nBuilding MoE and training router...")
    moe = ThreeExpertMoE(
        specialists["code"], specialists["science"], specialists["fiction"]
    ).to(device)
    train_datasets = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"])
                      for d in DOMAINS}
    train_router(moe, train_datasets, device)
    moe.eval()

    router_dist = eval_router_distribution(moe, held_out_sets, device)
    print("\nRouter gate distribution:")
    for d, gates in router_dist.items():
        print(f"  {d}: code={gates[0]:.4f}, science={gates[1]:.4f}, fiction={gates[2]:.4f}")

    # ── Load monolithic (optional) ─────────────────────────────────────────
    monolithic = None
    if monolithic_ckpt.exists():
        print("\nLoading monolithic baseline...")
        monolithic = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        monolithic.load_state_dict(torch.load(
            monolithic_ckpt, map_location=device, weights_only=True
        ))
        monolithic.eval()
        print("  Loaded.")
    else:
        print("\n[skip] Monolithic checkpoint not found.")

    # ── Evaluate ALL models at CONSISTENT batch_size=4 ────────────────────
    print("\n" + "=" * 70)
    print(f"EVALUATING — batch_size={EVAL_BATCH_SIZE} for ALL models")
    print("=" * 70)

    eval_models = [
        ("base",        base_model,             False),
        ("code_spec",   specialists["code"],    False),
        ("science_spec",specialists["science"], False),
        ("fiction_spec",specialists["fiction"], False),
        ("weight_avg",  weight_avg,             False),
        ("moe",         moe,                    True),
    ]
    if monolithic is not None:
        eval_models.insert(1, ("monolithic", monolithic, False))

    eval_matrix = {}
    for model_key, model, is_fused in eval_models:
        print(f"\n  [{model_key}]")
        losses = {}
        for domain, ds in held_out_sets.items():
            t0 = time.time()
            loss = eval_loss(model, ds, device, is_fused=is_fused)
            losses[domain] = round(loss, 6)
            print(f"    {domain:8s}: {loss:.4f} ({time.time()-t0:.1f}s)")
        eval_matrix[model_key] = losses

    # ── Compute corrected metrics ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CORRECTED RESULTS (consistent batch_size=4)")
    print("=" * 70)

    base_mixed        = eval_matrix["base"]["mixed"]
    best_spec_mixed   = min(eval_matrix["code_spec"]["mixed"],
                            eval_matrix["science_spec"]["mixed"],
                            eval_matrix["fiction_spec"]["mixed"])
    moe_mixed         = eval_matrix["moe"]["mixed"]
    weight_avg_mixed  = eval_matrix["weight_avg"]["mixed"]

    imp_vs_spec  = (best_spec_mixed - moe_mixed) / best_spec_mixed * 100
    imp_vs_base  = (base_mixed - moe_mixed) / base_mixed * 100
    imp_wa_vs_base = (base_mixed - weight_avg_mixed) / base_mixed * 100

    print(f"\n{'Model':<20} {'Code':>10} {'Science':>10} {'Fiction':>10} {'Mixed':>10}")
    print("-" * 62)
    domains_display = ["code", "science", "fiction", "mixed"]
    for mk, losses in eval_matrix.items():
        print(f"{mk:<20}" + "".join(f"{losses[d]:>10.4f}" for d in domains_display))

    print(f"\nCorrected key metrics:")
    print(f"  base_mixed:              {base_mixed:.4f}")
    print(f"  best_spec_mixed:         {best_spec_mixed:.4f}  (min of 3 specialists)")
    print(f"  weight_avg_mixed:        {weight_avg_mixed:.4f}")
    print(f"  moe_mixed:               {moe_mixed:.4f}")
    print(f"  MoE vs best specialist:  {imp_vs_spec:+.2f}%")
    print(f"  MoE vs base:             {imp_vs_base:+.2f}%")
    print(f"  Weight avg vs base:      {imp_wa_vs_base:+.2f}%")
    if "monolithic" in eval_matrix:
        mono_mixed = eval_matrix["monolithic"]["mixed"]
        imp_vs_mono = (mono_mixed - moe_mixed) / mono_mixed * 100
        print(f"  monolithic_mixed:        {mono_mixed:.4f}")
        print(f"  MoE vs monolithic:       {imp_vs_mono:+.2f}%")

    print(f"\nComparison with original (asymmetric) evaluation:")
    orig_best_spec  = 2.089038
    orig_moe_mixed  = 1.793269  # simple_linear from ablation (same eval)
    orig_imp_vs_spec = (orig_best_spec - orig_moe_mixed) / orig_best_spec * 100
    # From step4_fusion_results_seed42.json (bs=2 for MoE):
    orig_moe_step4  = 1.793054
    orig_imp_step4  = (orig_best_spec - orig_moe_step4) / orig_best_spec * 100
    print(f"  Original best_spec:      {orig_best_spec:.4f}")
    print(f"  Original moe_mixed:      {orig_moe_step4:.4f}  (evaluated on code-only chunks)")
    print(f"  Original improvement:    {orig_imp_step4:+.2f}%  (batch_size=2 for MoE, bs=4 for spec)")
    print(f"  Corrected improvement:   {imp_vs_spec:+.2f}%  (batch_size=4 for all)")

    # ── Save results ───────────────────────────────────────────────────────
    output = {
        "experiment":       "corrected_eval_410m",
        "model_id":         MODEL_ID,
        "revision":         REVISION,
        "seed":             42,
        "eval_batch_size":  EVAL_BATCH_SIZE,
        "eval_batches":     EVAL_BATCHES,
        "mixed_eval_composition": {
            "total_chunks": len(mixed_held),
            "chunks_evaluated": n_mixed_chunks,
            "code_chunks":    code_in_eval,
            "science_chunks": science_in_eval,
            "fiction_chunks": fiction_in_eval,
        },
        "eval_matrix":      eval_matrix,
        "metrics": {
            "base_mixed":           round(base_mixed, 6),
            "best_spec_mixed":      round(best_spec_mixed, 6),
            "weight_avg_mixed":     round(weight_avg_mixed, 6),
            "moe_mixed":            round(moe_mixed, 6),
            "improvement_vs_spec":  round(imp_vs_spec, 4),
            "improvement_vs_base":  round(imp_vs_base, 4),
            "improvement_wa_vs_base": round(imp_wa_vs_base, 4),
        },
        "router_distribution": router_dist,
        "original_vs_corrected": {
            "note": "Original used bs=2 for MoE (100 code-only chunks) vs bs=4 for specialists (200 code+science chunks)",
            "original_best_spec":    orig_best_spec,
            "original_moe_mixed":    orig_moe_step4,
            "original_improvement":  round(orig_imp_step4, 4),
            "corrected_improvement": round(imp_vs_spec, 4),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / "corrected_eval_seed42.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 70)
    print("CORRECTED EVAL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
