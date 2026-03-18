#!/usr/bin/env python3
"""
KALAVAI shared eval utilities. Import instead of writing inline eval logic.
Implements per-domain evaluation at consistent batch_size with equal-weight averaging.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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


# ============================================================================
# Eval — consistent batch_size, per-domain, equal-weight average
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
        complete_batches = min(eval_batches, len(ds) // bs)
        n_chunks_evaluated[domain] = complete_batches * bs
        print(f"    {domain:8s}: {loss:.4f}  ({n_chunks_evaluated[domain]}/{len(ds)} chunks, {time.time()-t0:.1f}s)")
    # Equal-weight average — THE KEY FIX
    losses["equal_weight_avg"] = round(sum(losses[d] for d in held_out_by_domain) / len(held_out_by_domain), 6)
    losses["_chunks_evaluated"] = n_chunks_evaluated
    return losses
