#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Unified Corrected Evaluation
======================================
Fixes two systematic bugs found in all original experiment files:

  Bug A — Batch-size asymmetry:
    Original: bs = 2 if is_fused else 4
    Effect:   MoE evaluated on 100 code-only chunks; specialists on 200 code+science chunks
    Fix:      EVAL_BATCH_SIZE consistent for ALL models (4 for 410M/1B, 1 for Qwen)

  Bug B — Mixed eval domain coverage gap:
    Original: mixed = concatenate(code, science, fiction) then take first N batches
    Effect:   fiction (77.6% of held-out, largest specialist advantage +25%) never evaluated
    Fix:      equal_weight_avg = (code_loss + science_loss + fiction_loss) / n_domains

Supports: pythia-410m, pythia-1b, qwen (2-domain)
Checkpoint format: existing .pt files from main experiments (no retraining needed for Pythia)

Usage:
  cd /c/Github/Kalavai
  python experiments/kalavai_corrected_eval.py --model 410m
  python experiments/kalavai_corrected_eval.py --model 1b
  python experiments/kalavai_corrected_eval.py --model qwen --seeds 42,137,2026
  python experiments/kalavai_corrected_eval.py --model 410m --seeds 137,2026
  python experiments/kalavai_corrected_eval.py --model 410m --seeds 42,137,2026 --save-checkpoints
"""

import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Model configs — one entry per supported model family
# ============================================================================

MODEL_CONFIGS = {
    "410m": {
        "model_id":       "EleutherAI/pythia-410m",
        "revision":       "step10000",
        "hidden_size":    1024,
        "eval_batch_size": 4,       # fits comfortably for MoE on RTX 5090
        "domains":        ["code", "science", "fiction"],
        "n_domains":      3,
        "n_samples":      3000,     # per domain, then 80/10/10 split
        "eval_batches":   50,
        "router_steps":   500,
        "router_lr":      1e-3,
        "router_batch":   4,
        "router_type":    "mlp",    # 2-layer MLP matches main 410M experiment
        "checkpoint_dir": Path("checkpoints/pythia"),
        "checkpoint_fn":  lambda domain, seed: f"{domain}_specialist_seed{seed}.pt",
        "monolithic_fn":  lambda seed: f"monolithic_seed{seed}.pt",
        "results_dir":    Path("results/pythia"),
        "results_fn":     lambda seed: f"corrected_eval_{seed}.json",
    },
    "1b": {
        "model_id":       "EleutherAI/pythia-1b",
        "revision":       "step10000",
        "hidden_size":    2048,
        "eval_batch_size": 4,
        "domains":        ["code", "science", "fiction"],
        "n_domains":      3,
        "n_samples":      3000,
        "eval_batches":   50,
        "router_steps":   500,
        "router_lr":      1e-3,
        "router_batch":   4,
        "router_type":    "linear", # single linear layer matches main 1B experiment
        "checkpoint_dir": Path("checkpoints/pythia/pythia_1b"),
        "checkpoint_fn":  lambda domain, seed: f"{domain}_specialist_seed{seed}.pt",
        "monolithic_fn":  lambda seed: f"monolithic_seed{seed}.pt",
        "results_dir":    Path("results/pythia/pythia_1b"),
        "results_fn":     lambda seed: f"corrected_eval_{seed}.json",
    },
    "qwen": {
        "model_id":       "Qwen/Qwen2.5-1.5B",
        "revision":       None,
        "hidden_size":    None,     # auto-detect from model.config
        "eval_batch_size": 4,
        "domains":        ["code", "fiction"],
        "n_domains":      2,
        "n_train_samples": 5000,    # Qwen uses separate train/eval splits
        "n_eval_samples":  500,
        "eval_batches":   50,
        "router_steps":   300,
        "router_lr":      1e-3,
        "router_batch":   4,
        "router_type":    "mlp_128", # 2-layer MLP 128-hidden matches original Qwen
        "max_steps":      1000,      # specialist training steps
        "lr":             2e-5,
        "batch_size":     2,
        "grad_accum":     2,
        "grad_clip":      1.0,
        "freeze_layers":  2,
        "checkpoint_dir": Path("checkpoints/qwen"),
        "checkpoint_fn":  lambda domain, seed: f"{domain}_specialist_seed{seed}.pt",
        "monolithic_fn":  lambda seed: None,  # no monolithic for Qwen
        "results_dir":    Path("results/real"),
        "results_fn":     lambda seed: f"corrected_eval_qwen_{seed}.json",
    },
}

SEQ_LEN = 512

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
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
    }


def chunks_to_dataset(chunks):
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


def split_chunks(chunks, train_frac=0.8, held_out_frac=0.1):
    n = len(chunks)
    a = int(n * train_frac)
    b = int(n * (train_frac + held_out_frac))
    return chunks[:a], chunks[a:b], chunks[b:]   # train, in-dist, held_out

# ============================================================================
# Data loading — Pythia (3-domain, 80/10/10 split from streaming train data)
# ============================================================================

def load_code_texts_pythia(n):
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


def load_science_texts_pythia(n):
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


def load_fiction_texts_pythia(n):
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


def load_pythia_data(n_samples, tokenizer):
    """Load and split data for Pythia 3-domain experiments."""
    print("\nLoading data...")
    raw = {
        "code":    load_code_texts_pythia(n_samples),
        "science": load_science_texts_pythia(n_samples),
        "fiction": load_fiction_texts_pythia(n_samples),
    }
    print("\nPacking and splitting (80/10/10)...")
    train_chunks, held_out_chunks = {}, {}
    for domain, texts in raw.items():
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        tr, _, ho = split_chunks(ds_full.chunks)
        train_chunks[domain]    = tr
        held_out_chunks[domain] = ho
        print(f"  {domain}: train={len(tr)}, held_out={len(ho)}")
    return train_chunks, held_out_chunks

# ============================================================================
# Data loading — Qwen (2-domain, separate train/test HF splits)
# ============================================================================

def load_code_texts_qwen(split, n):
    from datasets import load_dataset
    hf_split = "test" if split == "eval" else "train"
    print(f"  Loading code ({hf_split}, n={n})...")
    ds = load_dataset("code_search_net", "python", split=hf_split,
                      streaming=True, trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("func_code_string", "")
        if len(content) > 200:
            texts.append(content)
        if len(texts) >= n: break
    print(f"    {len(texts)} samples")
    return texts


def load_fiction_texts_qwen(split, n):
    from datasets import load_dataset
    hf_split = "test" if split == "eval" else "train"
    print(f"  Loading fiction ({hf_split}, n={n})...")
    ds = load_dataset("emozilla/pg19", split=hf_split, streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:3000]
        if len(content) >= 500:
            texts.append(content)
        if len(texts) >= n: break
    print(f"    {len(texts)} samples")
    return texts

# ============================================================================
# MoE architectures — must match originals exactly
# ============================================================================

class ThreeExpertMoE_MLP(nn.Module):
    """410M router: 2-layer MLP (hidden_size→256→ReLU→3). Matches kalavai_pythia_experiment.py."""
    def __init__(self, spec_a, spec_b, spec_c, hidden_size):
        super().__init__()
        self.spec_a, self.spec_b, self.spec_c = spec_a, spec_b, spec_c
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
        return out.logits.detach(), out.hidden_states[-1].detach().mean(dim=1).float()

    def forward(self, input_ids, labels=None):
        la, ha = self._run_specialist(self.spec_a, input_ids)
        lb, hb = self._run_specialist(self.spec_b, input_ids)
        lc, hc = self._run_specialist(self.spec_c, input_ids)
        gates = torch.softmax(self.router((ha + hb + hc) / 3.0), dim=-1)
        fused = gates[:, 0:1, None] * la + gates[:, 1:2, None] * lb + gates[:, 2:3, None] * lc
        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, fused, gates


class ThreeExpertMoE_Linear(nn.Module):
    """1B router: single linear layer (hidden_size→3). Matches kalavai_pythia_1b_experiment.py."""
    def __init__(self, spec_a, spec_b, spec_c, hidden_size):
        super().__init__()
        self.spec_a, self.spec_b, self.spec_c = spec_a, spec_b, spec_c
        for p in list(self.spec_a.parameters()) + list(self.spec_b.parameters()) + list(self.spec_c.parameters()):
            p.requires_grad_(False)
        self.router = nn.Linear(hidden_size, 3, bias=False)

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(dim=1).float()

    def forward(self, input_ids, labels=None):
        la, ha = self._run_specialist(self.spec_a, input_ids)
        lb, hb = self._run_specialist(self.spec_b, input_ids)
        lc, hc = self._run_specialist(self.spec_c, input_ids)
        gates = torch.softmax(self.router((ha + hb + hc) / 3.0), dim=-1)
        fused = gates[:, 0:1, None] * la + gates[:, 1:2, None] * lb + gates[:, 2:3, None] * lc
        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, fused, gates


class TwoExpertMoE_MLP128(nn.Module):
    """Qwen router: 2-layer MLP (hidden_size→128→ReLU→2). Matches kalavai_qwen_divergent_domains.py."""
    def __init__(self, spec_a, spec_b, hidden_size):
        super().__init__()
        self.spec_a, self.spec_b = spec_a, spec_b
        for p in list(self.spec_a.parameters()) + list(self.spec_b.parameters()):
            p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 2, bias=False),
        )

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(dim=1).float()

    def forward(self, input_ids, labels=None):
        la, ha = self._run_specialist(self.spec_a, input_ids)
        lb, hb = self._run_specialist(self.spec_b, input_ids)
        gates = torch.softmax(self.router((ha + hb) / 2.0), dim=-1)
        fused = gates[:, 0:1, None] * la + gates[:, 1:2, None] * lb
        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, fused, gates


def build_moe(cfg, specialists):
    """Build MoE with correct router architecture for the model family."""
    rtype = cfg["router_type"]
    hs = cfg["hidden_size"]
    if rtype == "mlp":
        return ThreeExpertMoE_MLP(*specialists, hs)
    elif rtype == "linear":
        return ThreeExpertMoE_Linear(*specialists, hs)
    elif rtype == "mlp_128":
        return TwoExpertMoE_MLP128(*specialists, hs)
    else:
        raise ValueError(f"Unknown router_type: {rtype}")

# ============================================================================
# Weight average
# ============================================================================

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
# Router training
# ============================================================================

def train_router(moe, train_chunks_by_domain, device, cfg):
    all_chunks = []
    for chunks in train_chunks_by_domain.values():
        all_chunks.extend(chunks)
    combined = chunks_to_dataset(all_chunks)
    optimizer = AdamW(moe.router.parameters(), lr=cfg["router_lr"])
    loader = DataLoader(combined, batch_size=cfg["router_batch"], shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    steps = cfg["router_steps"]
    print(f"  Training router ({steps} steps, {len(combined)} chunks)...")
    for step in range(1, steps + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        loss, _, _ = moe(input_ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps:
            print(f"    step {step}/{steps}: loss={loss.item():.4f}")
    moe.eval()

# ============================================================================
# Eval — THE FIX: consistent batch_size, per-domain, equal-weight average
# ============================================================================

@torch.no_grad()
def eval_loss_domain(model, dataset, device, batch_size, eval_batches, is_fused=False):
    """Evaluate on ONE domain at FIXED batch_size. No conditional branching."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= eval_batches: break
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


def eval_all_domains(model, held_out_by_domain, device, bs, eval_batches, is_fused=False):
    """
    Evaluate each domain separately at the SAME batch_size.
    Returns dict with per-domain losses + equal-weight average.
    This replaces the concatenated-mixed-eval that excluded fiction.
    """
    losses = {}
    n_chunks_evaluated = {}
    for domain, ds in held_out_by_domain.items():
        t0 = time.time()
        loss = eval_loss_domain(model, ds, device, bs, eval_batches, is_fused)
        losses[domain] = round(loss, 6)
        # Report actual chunk count evaluated
        complete_batches = min(eval_batches, len(ds) // bs)
        n_chunks_evaluated[domain] = complete_batches * bs
        print(f"    {domain:8s}: {loss:.4f}  ({n_chunks_evaluated[domain]}/{len(ds)} chunks, {time.time()-t0:.1f}s)")
    # Equal-weight average — THE KEY FIX
    losses["equal_weight_avg"] = round(sum(losses[d] for d in held_out_by_domain) / len(held_out_by_domain), 6)
    losses["_chunks_evaluated"] = n_chunks_evaluated
    return losses


@torch.no_grad()
def eval_router_distribution(moe, held_out_by_domain, device, bs, n_batches=20):
    moe.eval()
    results = {}
    n_experts = len(list(moe.spec_a.parameters()))  # just to get n_experts from module
    # Detect n_experts from router output size
    if hasattr(moe, 'spec_c'):
        n_exp = 3
    else:
        n_exp = 2
    for domain, ds in held_out_by_domain.items():
        loader = DataLoader(ds, batch_size=bs, shuffle=False,
                            drop_last=True, collate_fn=_collate)
        gate_sums = [0.0] * n_exp
        count = 0
        for batch in loader:
            if count >= n_batches: break
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            _, _, gates = moe(input_ids, labels=labels)
            for i in range(n_exp):
                gate_sums[i] += gates[:, i].mean().item()
            count += 1
        results[domain] = [round(g / max(count, 1), 4) for g in gate_sums]
    return results

# ============================================================================
# Specialist training (Qwen only — Pythia loads from checkpoint)
# ============================================================================

def freeze_layers_generic(model, n_layers):
    """Freeze embedding + first n transformer blocks. Works for Pythia and Qwen."""
    # Try Pythia-style (model.gpt_neox.layers) and Qwen-style (model.model.layers)
    embed = None
    layers = None
    if hasattr(model, 'gpt_neox'):
        embed = model.gpt_neox.embed_in
        layers = model.gpt_neox.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        embed = model.model.embed_tokens
        layers = model.model.layers
    if embed is not None:
        for p in embed.parameters(): p.requires_grad_(False)
    if layers is not None:
        for i in range(min(n_layers, len(layers))):
            for p in layers[i].parameters(): p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


def train_specialist_qwen(model, train_chunks, domain, seed, device, cfg):
    from torch.cuda.amp import autocast
    import torch
    torch.manual_seed(seed)
    dataset = chunks_to_dataset(train_chunks)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=0.01,
    )
    steps = cfg["max_steps"]
    accum = cfg["grad_accum"]
    clip  = cfg["grad_clip"]
    print(f"  Training {domain} specialist ({steps} steps, seed={seed})...")
    optimizer.zero_grad()
    for step in range(1, steps + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, labels=labels).loss
        (loss / accum).backward()
        if step % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
        if step % 200 == 0 or step == steps:
            print(f"    step {step}/{steps}: loss={loss.item():.4f}")
    model.eval()

# ============================================================================
# Load model from checkpoint or HuggingFace
# ============================================================================

def load_base_model(cfg, device):
    kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if cfg["revision"]:
        kwargs["revision"] = cfg["revision"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kwargs).to(device)
    model.eval()
    return model


def load_specialist(cfg, domain, seed, device):
    ckpt_path = cfg["checkpoint_dir"] / cfg["checkpoint_fn"](domain, seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if cfg["revision"]:
        kwargs["revision"] = cfg["revision"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kwargs).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model

# ============================================================================
# Run one seed for Pythia models (load from checkpoint)
# ============================================================================

def run_pythia_seed(cfg, seed, tokenizer, device):
    """Full eval: base, monolithic, all 3 specialists, weight_avg, MoE."""
    print(f"\n{'='*70}")
    print(f"Seed {seed} — {cfg['model_id']}")
    print(f"{'='*70}")

    # Load data (same as main experiment — deterministic streaming, seed-independent)
    train_chunks, held_out_chunks = load_pythia_data(cfg["n_samples"], tokenizer)
    held_out_sets = {d: chunks_to_dataset(held_out_chunks[d]) for d in cfg["domains"]}
    domains = cfg["domains"]
    eval_matrix = {}

    # ── Base model ─────────────────────────────────────────────────────────
    print("\n[base]")
    base = load_base_model(cfg, device)
    eval_matrix["base"] = eval_all_domains(base, held_out_sets, device,
                                           cfg["eval_batch_size"], cfg["eval_batches"])
    del base; torch.cuda.empty_cache()

    # ── Monolithic (if checkpoint exists for this seed) ────────────────────
    mono_fn = cfg["monolithic_fn"](seed)
    if mono_fn is not None:
        mono_path = cfg["checkpoint_dir"] / mono_fn
        if mono_path.exists():
            print(f"\n[monolithic]  loading {mono_path.name}")
            mono = _load_ckpt(cfg, mono_path, device)
            eval_matrix["monolithic"] = eval_all_domains(mono, held_out_sets, device,
                                                         cfg["eval_batch_size"], cfg["eval_batches"])
            del mono; torch.cuda.empty_cache()
        else:
            print(f"\n[monolithic]  not found — skipping")

    # ── Specialists ────────────────────────────────────────────────────────
    print(f"\nLoading specialists (seed={seed})...")
    specialists = {}
    for domain in domains:
        ckpt = cfg["checkpoint_dir"] / cfg["checkpoint_fn"](domain, seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing: {ckpt}\nRun the main experiment first.")
        specialists[domain] = _load_ckpt(cfg, ckpt, device)
        print(f"  Loaded {domain}")

    for domain, spec in specialists.items():
        print(f"\n[{domain}_spec]")
        eval_matrix[f"{domain}_spec"] = eval_all_domains(spec, held_out_sets, device,
                                                         cfg["eval_batch_size"], cfg["eval_batches"])

    # ── Weight average ─────────────────────────────────────────────────────
    print("\n[weight_avg]  computing weight average...")
    wa = weight_average(list(specialists.values())).to(device)
    eval_matrix["weight_avg"] = eval_all_domains(wa, held_out_sets, device,
                                                  cfg["eval_batch_size"], cfg["eval_batches"])
    del wa; torch.cuda.empty_cache()

    # ── MoE (router trained fresh; specialists kept in memory) ────────────
    print(f"\n[moe]  building + training router ({cfg['router_steps']} steps)...")
    # Specialist order must match domain order: code, science, fiction
    spec_list = [specialists[d] for d in domains]
    moe = build_moe(cfg, spec_list).to(device)
    train_router(moe, {d: train_chunks[d] for d in domains}, device, cfg)
    moe.eval()

    eval_matrix["moe"] = eval_all_domains(moe, held_out_sets, device,
                                          cfg["eval_batch_size"], cfg["eval_batches"],
                                          is_fused=True)

    router_dist = eval_router_distribution(moe, held_out_sets, device, bs=cfg["eval_batch_size"])
    print(f"\n  Router gate distribution:")
    for d, gates in router_dist.items():
        print("    %s: %s" % (d, "  ".join("%s=%.4f" % (domains[i], gates[i]) for i in range(len(gates)))))

    del moe
    for spec in specialists.values(): del spec
    torch.cuda.empty_cache()

    # ── Metrics ────────────────────────────────────────────────────────────
    def eq_avg(k): return eval_matrix[k]["equal_weight_avg"]

    moe_eq       = eq_avg("moe")
    base_eq      = eq_avg("base")
    best_spec_eq = min(eq_avg(f"{d}_spec") for d in domains)
    best_spec_domain = min(domains, key=lambda d: eq_avg(f"{d}_spec"))
    wa_eq        = eq_avg("weight_avg")

    metrics = {
        "base_equal_weight":       round(base_eq, 6),
        "best_spec_equal_weight":  round(best_spec_eq, 6),
        "best_spec_domain":        best_spec_domain,
        "weight_avg_equal_weight": round(wa_eq, 6),
        "moe_equal_weight":        round(moe_eq, 6),
        "improvement_vs_spec":     round((best_spec_eq - moe_eq) / best_spec_eq * 100, 4),
        "improvement_vs_base":     round((base_eq - moe_eq) / base_eq * 100, 4),
        "improvement_wa_vs_base":  round((base_eq - wa_eq) / base_eq * 100, 4),
    }
    if "monolithic" in eval_matrix:
        mono_eq = eq_avg("monolithic")
        metrics["monolithic_equal_weight"]      = round(mono_eq, 6)
        metrics["improvement_vs_monolithic"]    = round((mono_eq - moe_eq) / mono_eq * 100, 4)

    return {
        "seed":            seed,
        "model_id":        cfg["model_id"],
        "eval_batch_size": cfg["eval_batch_size"],
        "eval_batches":    cfg["eval_batches"],
        "eval_method":     "per-domain-separate-then-equal-weight-avg",
        "domains":         domains,
        "eval_matrix":     {k: {dk: dv for dk, dv in v.items() if not dk.startswith("_")}
                            for k, v in eval_matrix.items()},
        "metrics":         metrics,
        "router_distribution": router_dist,
    }


def _load_ckpt(cfg, path, device):
    """Load a model checkpoint onto device."""
    kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if cfg["revision"]:
        kwargs["revision"] = cfg["revision"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model

# ============================================================================
# Run one seed for Qwen (train + eval; save/load checkpoints)
# ============================================================================

def run_qwen_seed(cfg, seed, tokenizer, device,
                  train_code_chunks, train_fiction_chunks,
                  held_out_code_ds, held_out_fiction_ds,
                  save_checkpoints=False):
    print(f"\n{'='*70}")
    print(f"Seed {seed} — {cfg['model_id']}")
    print(f"{'='*70}")

    cfg["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    code_ckpt    = cfg["checkpoint_dir"] / cfg["checkpoint_fn"]("code",    seed)
    fiction_ckpt = cfg["checkpoint_dir"] / cfg["checkpoint_fn"]("fiction", seed)

    held_out_sets = {"code": held_out_code_ds, "fiction": held_out_fiction_ds}

    # ── Load base for reference eval ───────────────────────────────────────
    print("\n[base]")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    base.eval()
    hs = base.config.hidden_size
    cfg["hidden_size"] = hs  # fill in auto-detected value
    eval_matrix = {}
    eval_matrix["base"] = eval_all_domains(base, held_out_sets, device,
                                           cfg["eval_batch_size"], cfg["eval_batches"])

    # ── Load or train specialists ──────────────────────────────────────────
    specialists = {}
    for domain, ckpt_path, train_chunks in [
        ("code",    code_ckpt,    train_code_chunks),
        ("fiction", fiction_ckpt, train_fiction_chunks),
    ]:
        if ckpt_path.exists():
            print(f"\n[{domain}_spec]  loading from {ckpt_path}")
            spec = AutoModelForCausalLM.from_pretrained(
                cfg["model_id"], dtype=torch.bfloat16, trust_remote_code=True
            ).to(device)
            spec.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            spec.eval()
        else:
            print(f"\n[{domain}_spec]  training from scratch (seed={seed})...")
            spec = AutoModelForCausalLM.from_pretrained(
                cfg["model_id"], dtype=torch.bfloat16, trust_remote_code=True
            ).to(device)
            freeze_layers_generic(spec, cfg["freeze_layers"])
            train_specialist_qwen(spec, train_chunks, domain, seed, device, cfg)
            if save_checkpoints:
                torch.save(spec.state_dict(), ckpt_path)
                print(f"  Saved: {ckpt_path}")
        specialists[domain] = spec

    for domain, spec in specialists.items():
        print(f"\n[{domain}_spec]")
        eval_matrix[f"{domain}_spec"] = eval_all_domains(spec, held_out_sets, device,
                                                         cfg["eval_batch_size"], cfg["eval_batches"])

    # ── Weight average ─────────────────────────────────────────────────────
    print("\n[weight_avg]")
    wa = weight_average(list(specialists.values())).to(device)
    eval_matrix["weight_avg"] = eval_all_domains(wa, held_out_sets, device,
                                                  cfg["eval_batch_size"], cfg["eval_batches"])
    del wa; torch.cuda.empty_cache()

    # ── MoE ────────────────────────────────────────────────────────────────
    print(f"\n[moe]  building + training router...")
    moe = TwoExpertMoE_MLP128(specialists["code"], specialists["fiction"], hs).to(device)
    train_router(moe, {"code": train_code_chunks, "fiction": train_fiction_chunks}, device, cfg)
    moe.eval()
    eval_matrix["moe"] = eval_all_domains(moe, held_out_sets, device,
                                          cfg["eval_batch_size"], cfg["eval_batches"],
                                          is_fused=True)
    router_dist = eval_router_distribution(moe, held_out_sets, device, bs=cfg["eval_batch_size"])
    print(f"\n  Router gate distribution:")
    for d, gates in router_dist.items():
        labels = cfg["domains"]
        print("    %s: %s" % (d, "  ".join("%s=%.4f" % (labels[i], gates[i]) for i in range(len(gates)))))

    del moe, base
    for spec in specialists.values(): del spec
    torch.cuda.empty_cache()

    # ── Metrics ────────────────────────────────────────────────────────────
    base_eq      = eval_matrix["base"]["equal_weight_avg"]
    moe_eq       = eval_matrix["moe"]["equal_weight_avg"]
    best_spec_eq = min(eval_matrix["code_spec"]["equal_weight_avg"],
                       eval_matrix["fiction_spec"]["equal_weight_avg"])
    best_spec_domain = min(["code", "fiction"],
                           key=lambda d: eval_matrix[f"{d}_spec"]["equal_weight_avg"])
    wa_eq        = eval_matrix["weight_avg"]["equal_weight_avg"]

    metrics = {
        "base_equal_weight":       round(base_eq, 6),
        "best_spec_equal_weight":  round(best_spec_eq, 6),
        "best_spec_domain":        best_spec_domain,
        "weight_avg_equal_weight": round(wa_eq, 6),
        "moe_equal_weight":        round(moe_eq, 6),
        "improvement_vs_spec":     round((best_spec_eq - moe_eq) / best_spec_eq * 100, 4),
        "improvement_vs_base":     round((base_eq - moe_eq) / base_eq * 100, 4),
    }

    return {
        "seed":           seed,
        "model_id":       cfg["model_id"],
        "eval_batch_size": cfg["eval_batch_size"],
        "eval_batches":   cfg["eval_batches"],
        "eval_method":    "per-domain-separate-then-equal-weight-avg",
        "domains":        cfg["domains"],
        "eval_matrix":    {k: {dk: dv for dk, dv in v.items() if not dk.startswith("_")}
                           for k, v in eval_matrix.items()},
        "metrics":        metrics,
        "router_distribution": router_dist,
    }

# ============================================================================
# Main
# ============================================================================

def print_results_table(result):
    domains = result["domains"]
    matrix  = result["eval_matrix"]
    print(f"\n{'='*70}")
    print(f"CORRECTED RESULTS — seed={result['seed']}")
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
    print(f"  MoE vs best spec (equal-weight):  {m['improvement_vs_spec']:+.2f}%")
    print(f"  MoE vs base (equal-weight):       {m['improvement_vs_base']:+.2f}%")
    if "improvement_vs_monolithic" in m:
        print(f"  MoE vs monolithic (equal-weight): {m['improvement_vs_monolithic']:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="KALAVAI Corrected Evaluation")
    parser.add_argument("--model",  choices=["410m", "1b", "qwen"], required=True)
    parser.add_argument("--seeds",  default="42",
                        help="Comma-separated seeds (default: 42). Use 42,137,2026 for full run.")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save trained specialists (useful for Qwen, which has no saved checkpoints)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    cfg   = dict(MODEL_CONFIGS[args.model])  # copy so we can mutate hidden_size for qwen
    cfg["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nKALAVAI Corrected Evaluation")
    print(f"  model:          {cfg['model_id']}")
    print(f"  seeds:          {seeds}")
    print(f"  eval_batch_size: {cfg['eval_batch_size']} (consistent for ALL models)")
    print(f"  eval_method:    per-domain equal-weight average")
    print(f"  device:         {device}")
    if device == "cuda":
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:           {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Load tokenizer once ────────────────────────────────────────────────
    print(f"\nLoading tokenizer ({cfg['model_id']})...")
    tok_kwargs = {"trust_remote_code": True}
    if cfg["revision"]:
        tok_kwargs["revision"] = cfg["revision"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []

    if args.model in ("410m", "1b"):
        for seed in seeds:
            result = run_pythia_seed(cfg, seed, tokenizer, device)
            print_results_table(result)
            # Save per-seed result
            out_path = cfg["results_dir"] / cfg["results_fn"](seed)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"\nSaved: {out_path}")
            all_results.append(result)

    elif args.model == "qwen":
        # Load Qwen data once (same across seeds)
        print("\nLoading training data...")
        train_code_texts    = load_code_texts_qwen("train", cfg["n_train_samples"])
        train_fiction_texts = load_fiction_texts_qwen("train", cfg["n_train_samples"])
        print("\nLoading held-out evaluation data...")
        eval_code_texts    = load_code_texts_qwen("eval", cfg["n_eval_samples"])
        eval_fiction_texts = load_fiction_texts_qwen("eval", cfg["n_eval_samples"])

        # Pack into datasets
        train_code_chunks    = PackedChunkDataset(train_code_texts,    tokenizer, max_chars=3000).chunks
        train_fiction_chunks = PackedChunkDataset(train_fiction_texts, tokenizer, max_chars=3000).chunks
        held_out_code_ds    = chunks_to_dataset(PackedChunkDataset(eval_code_texts,    tokenizer, max_chars=3000).chunks)
        held_out_fiction_ds = chunks_to_dataset(PackedChunkDataset(eval_fiction_texts, tokenizer, max_chars=3000).chunks)

        print(f"\nEval dataset sizes: code={len(held_out_code_ds)}, fiction={len(held_out_fiction_ds)}")

        for seed in seeds:
            result = run_qwen_seed(
                cfg, seed, tokenizer, device,
                train_code_chunks, train_fiction_chunks,
                held_out_code_ds, held_out_fiction_ds,
                save_checkpoints=args.save_checkpoints,
            )
            print_results_table(result)
            out_path = cfg["results_dir"] / cfg["results_fn"](seed)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"\nSaved: {out_path}")
            all_results.append(result)

    # ── Multi-seed summary ─────────────────────────────────────────────────
    if len(all_results) > 1:
        import statistics
        imps = [r["metrics"]["improvement_vs_spec"] for r in all_results]
        print(f"\n{'='*70}")
        print(f"MULTI-SEED SUMMARY ({args.model})")
        print(f"{'='*70}")
        print(f"{'Seed':<8} {'vs spec':>10} {'vs base':>10}")
        print("-" * 30)
        for r in all_results:
            m = r["metrics"]
            print(f"{r['seed']:<8} {m['improvement_vs_spec']:>+9.2f}%  {m.get('improvement_vs_base', float('nan')):>+9.2f}%")
        if len(imps) > 1:
            print("-" * 30)
            print(f"{'Mean':<8} {statistics.mean(imps):>+9.2f}%")
            print(f"{'Std':<8} {statistics.stdev(imps):>9.2f}%")

        summary = {
            "model":    args.model,
            "model_id": cfg["model_id"],
            "seeds":    seeds,
            "per_seed": all_results,
            "summary": {
                "improvement_mean": round(statistics.mean(imps), 4),
                "improvement_std":  round(statistics.stdev(imps) if len(imps) > 1 else 0.0, 4),
            },
        }
        summary_path = cfg["results_dir"] / f"corrected_eval_{args.model}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved summary: {summary_path}")

    print(f"\n{'='*70}")
    print(f"CORRECTED EVAL COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
