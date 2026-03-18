#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Cross-Architecture Validation — Llama-3.2-1B (Experiment D2)
=====================================================================
Tests whether the KALAVAI fusion mechanism works on a non-Pythia architecture.
Uses Meta's Llama-3.2-1B base model (full training, no intermediate checkpoints).

Key differences from Pythia experiments:
  - Architecture: LlamaForCausalLM (vs GPT-NeoX)
  - Layer freezing: model.model.layers[i] (vs model.gpt_neox.layers[i])
  - Embedding: model.model.embed_tokens (vs model.gpt_neox.embed_in)
  - No intermediate checkpoints: maturity sweep not possible
  - Hidden size: 2048, num_layers: 16, same as Pythia-1B

Requires: Llama-3.2-1B access on HuggingFace
  huggingface-cli login   (or set HF_TOKEN env var)

Primary: meta-llama/Llama-3.2-1B
Fallback: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T (if Llama access unavailable)

3-domain experiment (code, science, fiction), 3 seeds, freeze=4 layers.
Same evaluation pipeline as all other experiments.

Success criterion: fusion improvement > 0% confirms the mechanism
generalises beyond Pythia. The exact magnitude may differ from Pythia.

Runs on RTX 5090 or A100. ~4 hours.
Resumable.

Usage:
  HF_TOKEN=your_token python kalavai_llama_experiment.py 2>&1 | tee llama_log.txt
"""

import copy
import json
import os
import subprocess
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

# ============================================================================
# Config
# ============================================================================

PRIMARY_MODEL_ID = "meta-llama/Llama-3.2-1B"
FALLBACK_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

FREEZE_LAYERS    = 4
LR               = 2e-5
WEIGHT_DECAY     = 0.1
MAX_STEPS        = 2000
BATCH_SIZE       = 2
GRAD_ACCUM       = 4
GRADIENT_CLIP    = 1.0
SEQ_LEN          = 512
WARMUP_FRACTION  = 0.1
N_SAMPLES        = 3000
ROUTER_STEPS     = 500
ROUTER_LR        = 1e-3
ROUTER_BATCH     = 4
EVAL_BATCHES     = 50
SEEDS            = [42, 137, 2026]
DOMAINS          = ["code", "science", "fiction"]

RESULTS_DIR  = Path("results/cross_arch/llama")
FIGURES_DIR  = Path("figures/pythia")
CHECKPOINT_DIR = Path("checkpoints/llama")

# ============================================================================
# Model ID selection
# ============================================================================

def select_model_id() -> str:
    """Try primary model; fall back if access denied."""
    hf_token = os.environ.get("HF_TOKEN", None)
    try:
        from huggingface_hub import model_info
        model_info(PRIMARY_MODEL_ID, token=hf_token)
        print(f"  Using primary: {PRIMARY_MODEL_ID}")
        return PRIMARY_MODEL_ID
    except Exception as e:
        print(f"  Primary model unavailable ({e})")
        print(f"  Falling back to: {FALLBACK_MODEL_ID}")
        return FALLBACK_MODEL_ID

# ============================================================================
# Helpers
# ============================================================================

def result_path(seed: int) -> Path:
    return RESULTS_DIR / f"result_seed{seed}.json"

def specialist_ckpt(domain: str, seed: int) -> Path:
    return CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"

def git_commit_push(message: str):
    print(f"\n[git] {message}")
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode == 0:
            print("[git] Nothing to commit.")
            return
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[git] Pushed.")
    except subprocess.CalledProcessError as e:
        print(f"[git] WARNING: {e}")

# ============================================================================
# Dataset
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
    return {"input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels":    torch.stack([b["labels"]    for b in batch])}

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
# Data loading (same sources as all other experiments)
# ============================================================================

def load_code_texts(n):
    from datasets import load_dataset
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
    texts = []
    for item in ds:
        c = item.get("whole_func_string","") or item.get("func_code_string","")
        if len(c) >= 200: texts.append(c)
        if len(texts) >= n: break
    print(f"  code: {len(texts)}")
    return texts

def load_science_texts(n):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        c = item.get("support","") + "\n" + item.get("question","") + "\n" + item.get("correct_answer","")
        if len(c) > 100: texts.append(c)
        if len(texts) >= n: break
    print(f"  science: {len(texts)}")
    return texts

def load_fiction_texts(n):
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        c = item.get("text","")[:5000]
        if len(c) >= 500: texts.append(c)
        if len(texts) >= n: break
    print(f"  fiction: {len(texts)}")
    return texts

# ============================================================================
# Model — Llama-specific freeze
# ============================================================================

def load_model(model_id: str, device: str):
    hf_token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"  {model_id}: {total/1e9:.2f}B params")
    return model

def apply_freeze(model, n: int, model_id: str):
    """Freeze first n layers. Handles both Llama and GPT-NeoX architectures."""
    if "llama" in model_id.lower() or "tinyllama" in model_id.lower():
        model.model.embed_tokens.requires_grad_(False)
        for i in range(n):
            model.model.layers[i].requires_grad_(False)
    elif "pythia" in model_id.lower() or "neox" in model_id.lower():
        model.gpt_neox.embed_in.requires_grad_(False)
        for i in range(n):
            model.gpt_neox.layers[i].requires_grad_(False)
    else:
        # Generic fallback: freeze by parameter name pattern
        for name, param in model.named_parameters():
            for i in range(n):
                if f"layers.{i}." in name or f"layer.{i}." in name:
                    param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Frozen {n} layers. Trainable: {trainable/1e6:.0f}M / {total/1e6:.0f}M")

def get_hidden_size(model) -> int:
    """Extract hidden size from model config."""
    cfg = model.config
    return getattr(cfg, "hidden_size", getattr(cfg, "d_model", 2048))

# ============================================================================
# MoE
# ============================================================================

class NExpertMoE(nn.Module):
    def __init__(self, specialists: list, hidden_size: int):
        super().__init__()
        self.specialists = nn.ModuleList(specialists)
        for spec in self.specialists:
            for p in spec.parameters():
                p.requires_grad_(False)
        self.router = nn.Linear(hidden_size, len(specialists), bias=False)

    def _run(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.float(), out.hidden_states[-1].float().mean(dim=1)

    def forward(self, input_ids, labels=None):
        outs  = [self._run(s, input_ids) for s in self.specialists]
        llist = [o[0] for o in outs]
        h_avg = sum(o[1] for o in outs) / len(outs)
        gates = torch.softmax(self.router(h_avg), dim=-1)
        fused = sum(gates[:, i:i+1, None] * llist[i] for i in range(len(self.specialists)))
        loss = None
        if labels is not None:
            sl = fused[:, :-1].contiguous()
            ll = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1))
        return loss, fused, gates

# ============================================================================
# Training
# ============================================================================

def train_specialist(model, domain, train_chunks, device, seed, model_id, log_every=50):
    set_seed(seed)
    apply_freeze(model, FREEZE_LAYERS, model_id)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, collate_fn=_collate)
    warmup_steps     = int(MAX_STEPS * WARMUP_FRACTION)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, MAX_STEPS - warmup_steps))

    step, accum, running_loss = 0, 0, 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS: break
        batch = {k: v.to(device) for k, v in batch.items()}
        loss  = model(**batch).loss / GRAD_ACCUM
        loss.backward()
        accum += 1
        running_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(trainable_params, GRADIENT_CLIP)
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step + 1) / warmup_steps
            optimizer.step()
            if step >= warmup_steps: scheduler.step()
            optimizer.zero_grad()
            accum = 0; step += 1
            if step % log_every == 0 or step == MAX_STEPS:
                print(f"  [{domain}] {step}/{MAX_STEPS} loss={running_loss/step:.4f} "
                      f"({time.time()-t0:.0f}s)")
    model.eval()

@torch.no_grad()
def eval_loss(model, dataset, device, batch_size=2, is_fused=False):
    g = torch.Generator(); g.manual_seed(999)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, collate_fn=_collate, generator=g)
    total, count = 0.0, 0
    model.eval()
    for batch in loader:
        if count >= EVAL_BATCHES: break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(input_ids, labels=labels)
        else:
            loss = model(input_ids=input_ids, labels=labels).loss
        if loss is not None:
            total += loss.item(); count += 1
    return total / count if count > 0 else float("inf")

def weight_average(specialists):
    states = [{k: v.cpu().float() for k, v in s.state_dict().items()} for s in specialists]
    avg_s  = {k: sum(s[k] for s in states) / len(states) for k in states[0]}
    avg = copy.deepcopy(specialists[0]).cpu()
    avg.load_state_dict(avg_s); avg.eval()
    return avg

def train_router(moe, train_datasets, device):
    all_chunks = []
    for ds in train_datasets.values(): all_chunks.extend(ds.chunks)
    combined   = make_dataset_from_chunks(all_chunks)
    moe.router = moe.router.to(device)
    optimizer  = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader     = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                            drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        loss, _, _ = moe(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router {step}/{ROUTER_STEPS}: loss={loss.item():.4f}")
    moe.eval()

# ============================================================================
# Run one seed
# ============================================================================

def run_seed(seed: int, model_id: str, device: str, tokenizer,
             all_domain_chunks: dict, base_losses: dict) -> dict:
    rpath = result_path(seed)
    if rpath.exists():
        print(f"\n[skip] seed={seed} already done.")
        return json.loads(rpath.read_text(encoding="utf-8"))

    print(f"\n{'='*70}")
    print(f"SEED {seed} | model={model_id}")
    print(f"{'='*70}")

    # ── Train specialists ─────────────────────────────────────────────────
    specialists = []
    hidden_size = None
    for domain in DOMAINS:
        ckpt = specialist_ckpt(domain, seed)
        if ckpt.exists():
            print(f"\n  Loading {domain} from {ckpt}...")
            model = load_model(model_id, device)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
        else:
            print(f"\n  Training {domain} (seed={seed})...")
            model = load_model(model_id, device)
            train_specialist(model, domain, all_domain_chunks[domain]["train"],
                             device, seed, model_id)
            if seed == 42:
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt)
                print(f"  Saved: {ckpt}")
        if hidden_size is None:
            hidden_size = get_hidden_size(model)
        specialists.append(model)

    # ── Eval datasets ─────────────────────────────────────────────────────
    held_out = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS: mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out["mixed"] = make_dataset_from_chunks(mixed_held)

    # ── Eval ──────────────────────────────────────────────────────────────
    fusion_losses = {"base": base_losses}
    for label, spec in zip(["code_spec","science_spec","fiction_spec"], specialists):
        fusion_losses[label] = {d: round(eval_loss(spec, ds, device), 6)
                                for d, ds in held_out.items()}

    avg = weight_average(specialists)
    avg.to(device)
    fusion_losses["weight_avg"] = {d: round(eval_loss(avg, ds, device), 6)
                                   for d, ds in held_out.items()}
    del avg; torch.cuda.empty_cache()

    train_ds = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"]) for d in DOMAINS}
    moe = NExpertMoE(specialists, hidden_size=hidden_size).to(device)
    train_router(moe, train_ds, device)
    fusion_losses["moe"] = {d: round(eval_loss(moe, ds, device, is_fused=True), 6)
                            for d, ds in held_out.items()}
    del moe
    for s in specialists: del s
    torch.cuda.empty_cache()

    # ── Metrics ───────────────────────────────────────────────────────────
    best_spec = min(fusion_losses["code_spec"]["mixed"],
                    fusion_losses["science_spec"]["mixed"],
                    fusion_losses["fiction_spec"]["mixed"])
    moe_mixed  = fusion_losses["moe"]["mixed"]
    base_mixed = base_losses["mixed"]

    imp_vs_spec = round((best_spec - moe_mixed) / best_spec * 100, 4)
    imp_vs_base = round((base_mixed - moe_mixed) / base_mixed * 100, 4)

    print(f"\n  KEY RESULT (seed={seed}):")
    print(f"    vs spec: +{imp_vs_spec:.2f}%  vs base: +{imp_vs_base:.2f}%")

    result = {
        "seed": seed, "model_id": model_id,
        "eval_heldout": fusion_losses,
        "best_spec_mixed": best_spec,
        "moe_mixed": moe_mixed,
        "base_mixed": base_mixed,
        "improvement_vs_spec": imp_vs_spec,
        "improvement_vs_base": imp_vs_base,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    git_commit_push(
        f"[kalavai] Llama D2 seed={seed}: +{imp_vs_spec:.2f}% vs spec"
    )
    return result

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Cross-Architecture Validation — Llama (D2)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model_id = select_model_id()
    hf_token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data...")
    code_texts    = load_code_texts(N_SAMPLES)
    science_texts = load_science_texts(N_SAMPLES)
    fiction_texts = load_fiction_texts(N_SAMPLES)

    set_seed(42)
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    # Base eval
    base_path = RESULTS_DIR / "base_eval.json"
    if base_path.exists():
        base_losses = json.loads(base_path.read_text(encoding="utf-8"))
        print(f"\n[skip] Base eval: mixed={base_losses['mixed']:.4f}")
    else:
        print("\nBase eval...")
        base_model = load_model(model_id, device)
        held_out   = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                      for d in DOMAINS}
        mixed_held = []
        for d in DOMAINS: mixed_held.extend(all_domain_chunks[d]["held_out"])
        held_out["mixed"] = make_dataset_from_chunks(mixed_held)
        base_losses = {d: round(eval_loss(base_model, ds, device), 6)
                       for d, ds in held_out.items()}
        del base_model; torch.cuda.empty_cache()
        base_path.write_text(json.dumps(base_losses, indent=2), encoding="utf-8")
        print(f"  Base mixed: {base_losses['mixed']:.4f}")

    # Run 3 seeds
    seed_results = []
    for seed in SEEDS:
        result = run_seed(seed, model_id, device, tokenizer,
                          all_domain_chunks, base_losses)
        seed_results.append(result)

    # Summary
    improvements = [r["improvement_vs_spec"] for r in seed_results]
    mean = round(sum(improvements)/len(improvements), 4)
    std  = round((sum((v-mean)**2 for v in improvements)/len(improvements))**0.5, 4)

    summary = {
        "experiment":  "cross_arch_llama",
        "model_id":    model_id,
        "mean_improvement_vs_spec": mean,
        "std_improvement_vs_spec":  std,
        "per_seed": improvements,
        "conclusion": (
            "POSITIVE: mechanism generalises to non-Pythia architecture"
            if mean > 0
            else "NEGATIVE: fusion does not improve over best specialist"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    git_commit_push(f"[kalavai] Llama D2 COMPLETE: +{mean:.2f}%±{std:.2f}%")

    print("\n" + "=" * 70)
    print("D2 COMPLETE")
    print("=" * 70)
    print(f"  Model:       {model_id}")
    print(f"  Result:      +{mean:.2f}% ± {std:.2f}% vs best specialist (3 seeds)")
    print(f"  Conclusion:  {summary['conclusion']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
