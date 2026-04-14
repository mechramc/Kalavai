#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-6.9B Freeze Depth Sweep — LG-02 and LG-03
==========================================================
Runs two new freeze-depth conditions at the optimal step count (2000):
  LG-02: steps=2000, freeze=2,  seed=42
  LG-03: steps=2000, freeze=4,  seed=42

Baseline for comparison: LG-01 freeze=0 → +6.53% vs spec (current paper result).
Gate: any condition beats +6.53% by ≥ +0.3pp → run LG-04 confirmatory (seed 137).

Results saved to: results/pythia/pythia_6b_step_sweep/
  result_steps2000_k2_seed42.json
  result_steps2000_k4_seed42.json

Resumable: each condition skips if its result file already exists.

Usage (on RunPod):
  cd /workspace/Kalavai
  python experiments/kalavai_6b_freeze_sweep.py 2>&1 | tee freeze_sweep_log.txt
"""

import copy
import json
import math
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
# Config — identical to kalavai_6b_freeze0.py
# ============================================================================

MODEL_ID        = "EleutherAI/pythia-6.9b"
REVISION        = "step10000"
HIDDEN_SIZE     = 4096
NUM_LAYERS      = 32
LR              = 1e-5
WEIGHT_DECAY    = 0.1
BATCH_SIZE      = 1
GRAD_ACCUM      = 8
GRADIENT_CLIP   = 1.0
SEQ_LEN         = 512
WARMUP_FRACTION = 0.1
DOMAINS         = ["code", "science", "fiction"]
N_SAMPLES       = 3000
ROUTER_STEPS    = 500
ROUTER_LR       = 1e-3
ROUTER_BATCH    = 4
EVAL_BATCHES    = 50

TARGET_STEPS   = 2000
TARGET_SEED    = 42
FREEZE_CONFIGS = [2, 4]   # LG-02 and LG-03

# Gate: if any result beats this by >= GATE_THRESHOLD, print LG-04 trigger
LG01_VS_SPEC   = 6.53     # current paper result (freeze=0, 2000 steps)
GATE_THRESHOLD = 0.30     # pp

RESULTS_DIR     = Path("results/pythia/pythia_6b_step_sweep")
CHECKPOINT_BASE = Path("checkpoints/pythia_6b_step_sweep")

# ============================================================================
# Helpers
# ============================================================================

def result_path(freeze: int) -> Path:
    return RESULTS_DIR / f"result_steps{TARGET_STEPS}_k{freeze}_seed{TARGET_SEED}.json"

def checkpoint_dir(freeze: int) -> Path:
    return CHECKPOINT_BASE / f"steps{TARGET_STEPS}_k{freeze}"

def specialist_ckpt(freeze: int, domain: str) -> Path:
    return checkpoint_dir(freeze) / f"{domain}_specialist_seed{TARGET_SEED}.pt"

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
    def __init__(self, texts: list[str], tokenizer, seq_len: int = SEQ_LEN,
                 max_chars: int = 5000):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated),
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]
        n_chunks = len(full) // seq_len
        self.chunks = [full[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels    = torch.stack([b["labels"]    for b in batch])
    return {"input_ids": input_ids, "labels": labels}


def make_dataset_from_chunks(chunks: list) -> PackedChunkDataset:
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


def split_chunks(chunks: list, train_frac: float = 0.8, indist_frac: float = 0.1):
    n = len(chunks)
    train_end  = int(n * train_frac)
    indist_end = int(n * (train_frac + indist_frac))
    return chunks[:train_end], chunks[train_end:indist_end], chunks[indist_end:]

# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading code (n={n})...")
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) >= 200:
            texts.append(content)
        if len(texts) >= n:
            break
    print(f"    {len(texts)} code samples")
    return texts


def load_science_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading science (n={n})...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        content = (item.get("support", "") + "\n"
                   + item.get("question", "") + "\n"
                   + item.get("correct_answer", ""))
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n:
            break
    print(f"    {len(texts)} science samples")
    return texts


def load_fiction_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading fiction (n={n})...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:5000]
        if len(content) >= 500:
            texts.append(content)
        if len(texts) >= n:
            break
    print(f"    {len(texts)} fiction samples")
    return texts

# ============================================================================
# Model
# ============================================================================

def load_model(device: str, gradient_checkpointing: bool = True):
    print(f"\nLoading {MODEL_ID} (revision={REVISION}) in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=False,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total/1e9:.2f}B")
    return model


def apply_freeze(model, n: int):
    if n == 0:
        print("  freeze=0: all layers trainable")
        return
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Frozen {n}/{NUM_LAYERS} layers. "
          f"Trainable: {trainable/1e9:.3f}B / {total/1e9:.2f}B "
          f"({100*trainable/total:.1f}%)")

# ============================================================================
# MoE
# ============================================================================

class ThreeExpertMoE(nn.Module):
    def __init__(self, spec_a, spec_b, spec_c, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for p in self.spec_a.parameters():
            p.requires_grad_(False)
        for p in self.spec_b.parameters():
            p.requires_grad_(False)
        for p in self.spec_c.parameters():
            p.requires_grad_(False)
        self.router = nn.Linear(hidden_size, 3, bias=False)

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach().float()
        last_h = out.hidden_states[-1].detach().float()
        h_pooled = last_h.mean(dim=1)
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
# Training and eval
# ============================================================================

def train_specialist(model, domain: str, train_chunks: list, device: str,
                     max_steps: int, freeze: int, log_every: int = 50):
    set_seed(TARGET_SEED)
    apply_freeze(model, freeze)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, collate_fn=_collate)

    warmup_steps     = int(max_steps * WARMUP_FRACTION)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps))

    step, accum = 0, 0
    running_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= max_steps:
            break
        batch_device = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(**batch_device)
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        accum        += 1
        running_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(trainable_params, GRADIENT_CLIP)
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step + 1) / warmup_steps
            optimizer.step()
            if step >= warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
            accum  = 0
            step  += 1
            if step % log_every == 0 or step == max_steps:
                avg     = running_loss / step
                elapsed = time.time() - t0
                print(f"  [{domain}] {step}/{max_steps} loss={avg:.4f} ({elapsed:.0f}s)")

    model.eval()
    print(f"  {domain} done ({time.time()-t0:.0f}s)")


@torch.no_grad()
def eval_loss(model, dataset, device: str, batch_size: int = 1,
              is_fused: bool = False) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
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
            out  = model(input_ids=input_ids, labels=labels)
            loss = out.loss
        if loss is not None:
            total += loss.item()
            count += 1
    return total / count if count > 0 else float("inf")


@torch.no_grad()
def eval_router_distribution(moe, eval_datasets: dict, device: str,
                              n_batches: int = 20) -> dict:
    moe.eval()
    results = {}
    for domain, ds in eval_datasets.items():
        loader    = DataLoader(ds, batch_size=1, shuffle=False,
                               drop_last=True, collate_fn=_collate)
        gate_sums = [0.0, 0.0, 0.0]
        count     = 0
        for batch in loader:
            if count >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            _, _, gates = moe(input_ids, labels=labels)
            for i in range(3):
                gate_sums[i] += gates[:, i].mean().item()
            count += 1
        results[domain] = [round(g / count, 4) for g in gate_sums] if count > 0 else [0.333] * 3
    return results


def weight_average_three(spec_a, spec_b, spec_c):
    print("  Weight averaging on CPU...")
    sa = {k: v.cpu().float() for k, v in spec_a.state_dict().items()}
    sb = {k: v.cpu().float() for k, v in spec_b.state_dict().items()}
    sc = {k: v.cpu().float() for k, v in spec_c.state_dict().items()}
    avg_state = {k: ((sa[k] + sb[k] + sc[k]) / 3.0).to(torch.bfloat16) for k in sa}
    avg = copy.deepcopy(spec_a).cpu()
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg


def train_router(moe: ThreeExpertMoE, train_datasets: dict, device: str):
    all_chunks = []
    for ds in train_datasets.values():
        all_chunks.extend(ds.chunks)
    combined  = make_dataset_from_chunks(all_chunks)
    moe.router = moe.router.to(device)
    optimizer  = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader     = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                            drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"\n  Router training ({ROUTER_STEPS} steps)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        loss, _, _ = moe(input_ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router {step:3d}/{ROUTER_STEPS}: loss={loss.item():.4f}")
    moe.eval()

# ============================================================================
# Per-condition run
# ============================================================================

def run_condition(freeze: int, all_domain_chunks: dict, base_losses: dict,
                  device: str) -> dict:
    print(f"\n{'='*70}")
    print(f"CONDITION: steps={TARGET_STEPS}, freeze={freeze}, seed={TARGET_SEED}")
    print(f"{'='*70}")

    rpath = result_path(freeze)
    if rpath.exists():
        print(f"\n[skip] Result already exists: {rpath}")
        d = json.loads(rpath.read_text(encoding="utf-8"))
        print(f"  vs spec: +{d['improvement_vs_spec']:.2f}%")
        print(f"  vs base: +{d['improvement_vs_base']:.2f}%")
        return d

    checkpoint_dir(freeze).mkdir(parents=True, exist_ok=True)

    # ── Train specialists ─────────────────────────────────────────────────
    trained = {}
    for domain in DOMAINS:
        ckpt = specialist_ckpt(freeze, domain)
        if ckpt.exists():
            print(f"\n  Loading cached {domain} specialist (freeze={freeze})...")
            model = load_model(device, gradient_checkpointing=False)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            del state
        else:
            print(f"\n  Training {domain} specialist (steps={TARGET_STEPS}, freeze={freeze})...")
            model = load_model(device, gradient_checkpointing=True)
            train_specialist(model, domain, all_domain_chunks[domain]["train"],
                             device, max_steps=TARGET_STEPS, freeze=freeze)
            model.eval()
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved: {ckpt} ({ckpt.stat().st_size/1e9:.1f}GB)")
        model.to("cpu")
        torch.cuda.empty_cache()
        trained[domain] = model

    # ── Move to GPU ───────────────────────────────────────────────────────
    print(f"\n  Moving specialists to GPU...")
    for domain in DOMAINS:
        trained[domain].to(device)
    torch.cuda.empty_cache()

    spec_code    = trained["code"]
    spec_science = trained["science"]
    spec_fiction = trained["fiction"]

    # ── Eval datasets ─────────────────────────────────────────────────────
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    # ── Eval specialists ──────────────────────────────────────────────────
    print(f"\nEVALUATING (freeze={freeze})")
    fusion_losses = {"base": base_losses}
    for label, spec in [("code_spec", spec_code), ("science_spec", spec_science),
                         ("fiction_spec", spec_fiction)]:
        losses = {}
        for d, ds in held_out_sets.items():
            losses[d] = round(eval_loss(spec, ds, device), 6)
        fusion_losses[label] = losses
        print(f"  {label}: {losses}")

    # ── Weight average ────────────────────────────────────────────────────
    avg_model = weight_average_three(spec_code, spec_science, spec_fiction)
    avg_model.to(device)
    wa_losses = {d: round(eval_loss(avg_model, ds, device), 6)
                 for d, ds in held_out_sets.items()}
    fusion_losses["weight_avg"] = wa_losses
    print(f"  weight_avg: {wa_losses}")
    del avg_model
    torch.cuda.empty_cache()

    # ── MoE fusion ────────────────────────────────────────────────────────
    train_ds_dict = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"])
                     for d in DOMAINS}
    moe = ThreeExpertMoE(spec_code, spec_science, spec_fiction).to(device)
    train_router(moe, train_ds_dict, device)

    moe_losses = {d: round(eval_loss(moe, ds, device, is_fused=True), 6)
                  for d, ds in held_out_sets.items()}
    fusion_losses["moe"] = moe_losses
    print(f"  moe: {moe_losses}")

    router_dist = eval_router_distribution(moe, held_out_sets, device)
    del moe, spec_code, spec_science, spec_fiction
    torch.cuda.empty_cache()

    # ── Metrics ───────────────────────────────────────────────────────────
    best_spec_mixed = min(
        fusion_losses["code_spec"]["mixed"],
        fusion_losses["science_spec"]["mixed"],
        fusion_losses["fiction_spec"]["mixed"],
    )
    moe_mixed  = fusion_losses["moe"]["mixed"]
    base_mixed = base_losses["mixed"]

    improvement_vs_spec = round((best_spec_mixed - moe_mixed) / best_spec_mixed * 100, 4)
    improvement_vs_base = round((base_mixed - moe_mixed) / base_mixed * 100, 4)

    print(f"\n{'='*70}")
    print(f"KEY RESULT (steps={TARGET_STEPS}, freeze={freeze}, seed={TARGET_SEED}):")
    print(f"  Base mixed:      {base_mixed:.4f}")
    print(f"  Best spec mixed: {best_spec_mixed:.4f}")
    print(f"  MoE mixed:       {moe_mixed:.4f}")
    print(f"  vs spec:         +{improvement_vs_spec:.2f}%")
    print(f"  vs base:         +{improvement_vs_base:.2f}%")
    print(f"{'='*70}")

    # ── Gate check ────────────────────────────────────────────────────────
    delta_vs_lg01 = improvement_vs_spec - LG01_VS_SPEC
    print(f"\nGate check vs LG-01 (freeze=0, +{LG01_VS_SPEC:.2f}%):")
    print(f"  freeze={freeze} vs spec: +{improvement_vs_spec:.2f}%")
    print(f"  Delta vs LG-01: {delta_vs_lg01:+.2f}pp (threshold: ≥+{GATE_THRESHOLD:.2f}pp)")
    if delta_vs_lg01 >= GATE_THRESHOLD:
        print(f"  *** GATE PASSED: freeze={freeze} beats LG-01 by {delta_vs_lg01:.2f}pp ***")
        print(f"  *** TRIGGER LG-04: run seed 137 with freeze={freeze} ***")
    else:
        print(f"  Gate not passed.")

    # ── Save ──────────────────────────────────────────────────────────────
    result = {
        "steps":               TARGET_STEPS,
        "freeze":              freeze,
        "seed":                TARGET_SEED,
        "model_id":            MODEL_ID,
        "revision":            REVISION,
        "eval_heldout":        fusion_losses,
        "base_mixed":          base_mixed,
        "best_spec_mixed":     best_spec_mixed,
        "moe_mixed":           moe_mixed,
        "improvement_vs_spec": improvement_vs_spec,
        "improvement_vs_base": improvement_vs_base,
        "router_distribution": router_dist,
        "lg01_vs_spec":        LG01_VS_SPEC,
        "delta_vs_lg01":       round(delta_vs_lg01, 4),
        "gate_passed":         delta_vs_lg01 >= GATE_THRESHOLD,
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    rpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {rpath}")

    git_commit_push(
        f"[kalavai] 6.9B freeze sweep LG: steps={TARGET_STEPS} k={freeze} seed={TARGET_SEED} "
        f"→ +{improvement_vs_spec:.2f}% vs spec (LG-01 baseline: +{LG01_VS_SPEC:.2f}%)"
    )

    return result

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Pythia-6.9B Freeze Sweep (LG-02 and LG-03)")
    print(f"  Conditions: freeze={FREEZE_CONFIGS}, steps={TARGET_STEPS}, seed={TARGET_SEED}")
    print(f"  LG-01 baseline: +{LG01_VS_SPEC:.2f}% vs spec (freeze=0)")
    print(f"  Gate: beat LG-01 by ≥+{GATE_THRESHOLD:.2f}pp → trigger LG-04 (seed 137)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Data (loaded once, shared across both conditions) ─────────────────
    print("\nLoading data...")
    code_texts    = load_code_texts(N_SAMPLES)
    science_texts = load_science_texts(N_SAMPLES)
    fiction_texts = load_fiction_texts(N_SAMPLES)

    set_seed(42)
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    # ── Base eval (shared, cached) ─────────────────────────────────────────
    base_eval_path = RESULTS_DIR / "base_eval.json"
    if base_eval_path.exists():
        print(f"\n[skip] Base eval already done.")
        base_losses = json.loads(base_eval_path.read_text(encoding="utf-8"))
    else:
        print("\nRunning base eval...")
        base_model = load_model(device, gradient_checkpointing=False)
        base_model.eval()
        held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                         for d in DOMAINS}
        mixed_held = []
        for d in DOMAINS:
            mixed_held.extend(all_domain_chunks[d]["held_out"])
        held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)
        base_losses = {}
        for domain, ds in held_out_sets.items():
            loss = eval_loss(base_model, ds, device)
            base_losses[domain] = round(loss, 6)
            print(f"  Base [{domain:8s}]: {loss:.4f}")
        del base_model
        torch.cuda.empty_cache()
        base_eval_path.write_text(json.dumps(base_losses, indent=2), encoding="utf-8")

    print(f"\n  Base mixed loss: {base_losses['mixed']:.4f}")

    # ── Run conditions ────────────────────────────────────────────────────
    results = {}
    for freeze in FREEZE_CONFIGS:
        results[freeze] = run_condition(
            freeze, all_domain_chunks, base_losses, device
        )

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  LG-01 (freeze=0): +{LG01_VS_SPEC:.2f}% vs spec  [current paper result]")
    for freeze in FREEZE_CONFIGS:
        r = results[freeze]
        gain = r["improvement_vs_spec"]
        delta = r["delta_vs_lg01"]
        gate = "*** GATE PASSED ***" if r["gate_passed"] else ""
        print(f"  LG-0{FREEZE_CONFIGS.index(freeze)+2} (freeze={freeze}): +{gain:.2f}% vs spec  "
              f"(Δ={delta:+.2f}pp vs LG-01) {gate}")

    any_gate_passed = any(r["gate_passed"] for r in results.values())
    if any_gate_passed:
        best_freeze = max(results, key=lambda k: results[k]["improvement_vs_spec"])
        print(f"\n*** ACTION: Run LG-04 with freeze={best_freeze}, seed=137 ***")
    else:
        print(f"\nNo gate passed. 6.9B gain insensitive to freeze depth. "
              f"Keep freeze=0 (+{LG01_VS_SPEC:.2f}%) as paper result.")

    print("=" * 70)


if __name__ == "__main__":
    main()
