#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
Dump 6.9B held-out data to disk for reproducible monolithic baseline eval.

OPTION B implementation: loads data with IDENTICAL parameters to
kalavai_pythia_6b_experiment.py, evaluates base model on those held-out
chunks, verifies EW matches 2.320049 (the stored corrected-eval base),
then saves held-out chunks as a pickle file.

Run this BEFORE kalavai_6b_monolithic_v2.py.

Expected output:
  Base EW on held-out: ~2.3200  (should match corrected_eval base = 2.320049)
  Saved: results/pythia_6b/held_out_chunks.pkl

If the base EW does NOT match 2.320 ± 0.005, the streaming order differed
and the held-out data is not comparable to the corrected eval. Stop and
investigate before running monolithic training.
"""

import pickle
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config — MUST match kalavai_pythia_6b_experiment.py exactly ──────────────
MODEL_ID             = "EleutherAI/pythia-6.9b"
REVISION             = "step10000"
SEQ_LEN              = 512
N_SAMPLES_PER_DOMAIN = 3000
EVAL_BATCHES         = 50
DOMAINS              = ["code", "science", "fiction"]

RESULTS_DIR = Path("results/pythia_6b")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset — copied verbatim from main 6.9B experiment ───────────────────────

class PackedChunkDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated),
            return_tensors="pt",
            truncation=False,
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
    return (chunks[:int(n * train_frac)],
            chunks[int(n * train_frac):int(n * (train_frac + indist_frac))],
            chunks[int(n * (train_frac + indist_frac)):])


# ── Data loaders — copied verbatim from main 6.9B experiment ─────────────────

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
    return texts


def load_science_texts(n):
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
        if len(texts) >= n: break
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
    return texts


# ── Eval ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss_domain(model, dataset, device, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES: break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        loss = model(input_ids=ids, labels=lbl).loss
        if loss is not None:
            total += loss.item()
            count += 1
    return round(total / count, 6) if count > 0 else float("inf")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("KALAVAI: Dump 6.9B held-out data for reproducible monolithic eval")
    print("=" * 70)
    print(f"Model: {MODEL_ID} @ {REVISION}")
    print(f"Expected base EW: ~2.320049 (from corrected_eval_6b_summary.json)")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data — identical to kalavai_pythia_6b_experiment.py
    print("\nLoading data (identical to main 6.9B experiment)...")
    code_texts    = load_code_texts(N_SAMPLES_PER_DOMAIN)
    science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
    fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)

    print("\nPacking and splitting (80/10/10)...")
    held_out_chunks = {}
    all_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        held_out_chunks[domain] = held_c
        all_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    # Verify by evaluating base model
    print("\nLoading base model to verify held-out data...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to(device)

    held_out_sets = {d: make_dataset_from_chunks(held_out_chunks[d]) for d in DOMAINS}

    print("Evaluating base model on held-out data...")
    base_losses = {d: eval_loss_domain(base_model, held_out_sets[d], device)
                   for d in DOMAINS}
    base_ew = round(sum(base_losses[d] for d in DOMAINS) / len(DOMAINS), 6)
    base_losses["ew"] = base_ew

    print(f"\n  Base EW       = {base_ew:.6f}")
    print(f"  Base code     = {base_losses['code']:.6f}")
    print(f"  Base science  = {base_losses['science']:.6f}")
    print(f"  Base fiction  = {base_losses['fiction']:.6f}")
    print(f"  Eval time     = {time.time()-t0:.0f}s")

    expected_ew = 2.320049
    diff = abs(base_ew - expected_ew)
    print(f"\n  Expected EW   = {expected_ew:.6f}")
    print(f"  Difference    = {diff:.6f}")

    if diff > 0.005:
        print()
        print("  *** WARNING: Base EW differs from expected by more than 0.005 ***")
        print("  *** Streaming loaded different data. Held-out tokens do NOT    ***")
        print("  *** match the corrected eval. DO NOT proceed with monolithic.  ***")
        print("  *** Investigate before continuing.                             ***")
        ok = False
    else:
        print()
        print("  ✓ Base EW matches expected (±0.005). Data verified.")
        ok = True

    del base_model
    torch.cuda.empty_cache()

    # Save held-out chunks to disk regardless (user decides whether to proceed)
    out_path = RESULTS_DIR / "held_out_chunks.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "held_out_chunks": held_out_chunks,
            "base_losses":     base_losses,
            "domain_sizes":    {d: len(held_out_chunks[d]) for d in DOMAINS},
            "verified":        ok,
            "expected_ew":     expected_ew,
        }, f)
    print(f"\nSaved: {out_path}")
    print(f"  Chunk counts: { {d: len(held_out_chunks[d]) for d in DOMAINS} }")

    if ok:
        print("\nReady to run kalavai_6b_monolithic_v2.py")
    else:
        print("\nDo NOT run kalavai_6b_monolithic_v2.py — data verification failed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
