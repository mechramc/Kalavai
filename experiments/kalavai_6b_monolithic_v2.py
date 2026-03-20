#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-6.9B Monolithic Baseline v2 — Corrected Equal-Compute
======================================================================
Fixes two bugs in kalavai_6b_monolithic_baseline.py:

  BUG 1 (compute budget): Original used MAX_STEPS=6000 (3 × 2000) but 6.9B
    specialists only run for 1000 steps (kalavai_pythia_6b_experiment.py,
    MAX_STEPS=1000). Corrected: MAX_STEPS=3000 = 3 specialists × 1000 steps.

  BUG 2 (held-out data): Original loaded fresh streaming data → different
    held-out tokens from the corrected eval. This script loads held-out chunks
    from results/pythia_6b/held_out_chunks.pkl (dumped by dump_6b_held_out_data.py,
    same tokens as the main 6.9B experiment). Base + monolithic + MoE are all
    on the same tokens.

  ALSO FIXED (hyperparams): Original used LR=2e-5, FREEZE_LAYERS=7.
    Corrected to match the 6.9B specialists: LR=1e-5, FREEZE_LAYERS=6.

RUN ORDER:
  1. python dump_6b_held_out_data.py   (verify data, ~30-60min)
  2. python kalavai_6b_monolithic_v2.py

Expected:
  - Base EW matches ~2.320 (same as Table 1)
  - Monolithic trained for 3000 steps (fair equal-compute)
  - fused_vs_monolithic will be the corrected number for paper Table 4

Outputs:
  results/pythia_6b/monolithic_v2_seed{N}.json
  results/pythia_6b/monolithic_v2_summary.json
  figures/pythia_6b/fig_6b_monolithic_v2_comparison.png
"""

import json
import pickle
import statistics
import time
from itertools import cycle
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ── Config — must match kalavai_pythia_6b_experiment.py ──────────────────────
MODEL_ID      = "EleutherAI/pythia-6.9b"
REVISION      = "step10000"
FREEZE_LAYERS = 6            # matches 6.9B specialist (was 7 in original)
LR            = 1e-5         # matches 6.9B specialist (was 2e-5 in original)
WEIGHT_DECAY  = 0.1
MAX_STEPS     = 3000         # FIXED: 3 specialists × 1000 steps (was 6000)
BATCH_SIZE    = 1
GRAD_ACCUM    = 8            # effective batch = 8
GRADIENT_CLIP = 1.0
SEQ_LEN       = 512
WARMUP_FRACTION = 0.1
DOMAINS       = ["code", "science", "fiction"]
SEEDS         = [42, 137, 2026]
EVAL_INTERVAL = 500
EVAL_BATCHES  = 50

RESULTS_DIR    = Path("results/pythia_6b")
FIGURES_DIR    = Path("figures/pythia_6b")
HELD_OUT_PATH  = RESULTS_DIR / "held_out_chunks.pkl"
MOE_PATH       = RESULTS_DIR / "corrected_eval_6b_summary.json"


# ── Dataset ───────────────────────────────────────────────────────────────────

class PackedChunkDataset(Dataset):
    def __init__(self):
        self.chunks = []

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
    ds = PackedChunkDataset()
    ds.chunks = chunks
    return ds


# ── Freeze ─────────────────────────────────────────────────────────────────────

def freeze_first_n_layers(model, n):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


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


def eval_all(model, held_out_sets, device):
    losses = {d: eval_loss_domain(model, held_out_sets[d], device) for d in DOMAINS}
    losses["ew"] = round(sum(losses[d] for d in DOMAINS) / len(DOMAINS), 6)
    return losses


# ── Training ───────────────────────────────────────────────────────────────────

def train_monolithic(model, mixed_train_chunks, held_out_sets, seed, device):
    set_seed(seed)
    freeze_first_n_layers(model, FREEZE_LAYERS)
    model.gradient_checkpointing_enable()
    model.train()

    dataset = make_dataset_from_chunks(mixed_train_chunks)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, collate_fn=_collate)

    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer    = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

    checkpoints = []

    print(f"  Step 0 eval (base)...")
    model.eval()
    model.gradient_checkpointing_disable()
    step0 = eval_all(model, held_out_sets, device)
    model.gradient_checkpointing_enable()
    checkpoints.append({"step": 0, "held_out": step0, "train_loss": None})
    print(f"    EW={step0['ew']:.4f}  code={step0['code']:.4f}  "
          f"science={step0['science']:.4f}  fiction={step0['fiction']:.4f}")
    model.train()

    step, accum, running_loss = 0, 0, 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS: break

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(**{k: v.to(device) for k, v in batch.items()})
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        accum        += 1
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

            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                avg_train = running_loss / step
                elapsed   = time.time() - t0
                eta = elapsed / step * (MAX_STEPS - step) if step > 0 else 0
                print(f"  step {step}/{MAX_STEPS} | train={avg_train:.4f} | "
                      f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s | eval...")
                model.eval()
                model.gradient_checkpointing_disable()
                ckpt_eval = eval_all(model, held_out_sets, device)
                model.gradient_checkpointing_enable()
                model.train()
                print(f"    EW={ckpt_eval['ew']:.4f}  code={ckpt_eval['code']:.4f}  "
                      f"science={ckpt_eval['science']:.4f}  fiction={ckpt_eval['fiction']:.4f}")
                checkpoints.append({
                    "step":       step,
                    "held_out":   ckpt_eval,
                    "train_loss": round(avg_train, 6),
                })

    print(f"  Training done in {time.time()-t0:.0f}s")
    model.eval()
    model.gradient_checkpointing_disable()
    return checkpoints


# ── Figure ─────────────────────────────────────────────────────────────────────

def save_figure(all_seed_results, base_ew, moe_info):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mono_vals = [r["final_held_out"]["ew"] for r in all_seed_results]
        mono_mean = statistics.mean(mono_vals)
        mono_std  = statistics.stdev(mono_vals) if len(mono_vals) > 1 else 0.0

        labels = ["Base\nmodel", "Best\nspecialist", "Weight\naverage",
                  "Monolithic\n(3000 steps)", "KALAVAI\nMoE"]
        means  = [base_ew,
                  moe_info["best_spec_ew_mean"],
                  moe_info["weight_avg_ew_mean"],
                  mono_mean,
                  moe_info["moe_ew_mean"]]
        errs   = [0.0, 0.0, 0.0, mono_std, moe_info["moe_ew_std"]]
        colors = ["#95a5a6", "#3498db", "#f39c12", "#e67e22", "#9b59b6"]

        y_min = min(means) * 0.995
        y_max = max(means) * 1.005

        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.5,
                      yerr=errs, capsize=5, error_kw={"linewidth": 1.5})
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("EW Loss — equal-weight (code+science+fiction)/3 (lower is better)")
        ax.set_title("Equal-Compute Comparison: Monolithic vs Specialist Fusion\n"
                     "(Pythia-6.9B, corrected equal-weight eval, same held-out tokens)")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, loss in zip(bars, means):
            imp   = (base_ew - loss) / base_ew * 100
            label = f"{loss:.4f}"
            if abs(imp) > 0.01:
                label += f"\n({imp:+.1f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (y_max - y_min) * 0.003,
                    label, ha="center", va="bottom", fontsize=8, fontweight="bold")

        moe_m = moe_info["moe_ew_mean"]
        gap   = (mono_mean - moe_m) / mono_mean * 100
        ax.annotate(
            f"MoE vs Monolithic:\n{gap:+.2f}%",
            xy=(len(means) - 1, moe_m),
            xytext=(-120, 30), textcoords="offset points",
            fontsize=9, color="#8e44ad",
            arrowprops=dict(arrowstyle="->", color="#8e44ad"),
        )

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_6b_monolithic_v2_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING figure: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("KALAVAI: 6.9B Monolithic Baseline v2 — Corrected Equal-Compute")
    print("=" * 70)
    print(f"Model:       {MODEL_ID} @ {REVISION}")
    print(f"MAX_STEPS:   {MAX_STEPS}  (3 specialists × 1000 = equal compute)")
    print(f"LR:          {LR}   (matches 6.9B specialist)")
    print(f"Freeze:      {FREEZE_LAYERS} layers  (matches 6.9B specialist)")
    print(f"eff batch:   {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Seeds:       {SEEDS}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:      {device}\n")

    # Load verified held-out chunks
    if not HELD_OUT_PATH.exists():
        raise FileNotFoundError(
            f"Held-out data not found: {HELD_OUT_PATH}\n"
            f"Run dump_6b_held_out_data.py first."
        )

    print(f"Loading held-out chunks from {HELD_OUT_PATH}...")
    with open(HELD_OUT_PATH, "rb") as f:
        dump = pickle.load(f)

    if not dump.get("verified", False):
        raise RuntimeError(
            "held_out_chunks.pkl was NOT verified (base EW mismatch in dump script).\n"
            "Do not proceed — the data does not match the corrected eval."
        )

    held_out_chunks = dump["held_out_chunks"]
    dump_base_ew    = dump["base_losses"]["ew"]
    print(f"  Verified base EW from dump: {dump_base_ew:.6f}  (expected ~2.320049)")
    print(f"  Chunk counts: { {d: len(held_out_chunks[d]) for d in DOMAINS} }")

    # Load MoE reference (same held-out tokens via corrected eval)
    if not MOE_PATH.exists():
        raise FileNotFoundError(
            f"MoE reference not found: {MOE_PATH}\n"
            f"Run kalavai_pythia_6b_experiment.py first."
        )
    with open(MOE_PATH) as f:
        moe_ref = json.load(f)

    moe_ew_vals     = [r["moe_ew"]                        for r in moe_ref["results"]]
    best_spec_vals  = [r["best_spec_ew"]                  for r in moe_ref["results"]]
    weight_avg_vals = [r["per_model_ew"]["weight_avg"]    for r in moe_ref["results"]]

    moe_info = {
        "moe_ew_mean":        statistics.mean(moe_ew_vals),
        "moe_ew_std":         statistics.stdev(moe_ew_vals) if len(moe_ew_vals) > 1 else 0.0,
        "best_spec_ew_mean":  statistics.mean(best_spec_vals),
        "weight_avg_ew_mean": statistics.mean(weight_avg_vals),
        "mean_gain_pct":      moe_ref["mean_gain"],
    }
    print(f"\nMoE reference (corrected eval, 3 seeds):")
    print(f"  MoE EW  = {moe_info['moe_ew_mean']:.6f} ± {moe_info['moe_ew_std']:.6f}")
    print(f"  Gain    = {moe_info['mean_gain_pct']:.2f}%  (MoE vs best specialist)")

    # Tokenizer (for model load only — data already tokenized)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Held-out sets
    held_out_sets = {d: make_dataset_from_chunks(held_out_chunks[d]) for d in DOMAINS}

    # Build mixed training data from held-out complement (use same proportions as main)
    # We re-load train split from held_out_chunks dump (which saved train too — if not,
    # we reconstruct: held_out is last 10%, train is first 80% of same chunks)
    # Since dump only saves held_out, we reload train data fresh (same streaming order)
    print("\nLoading training data for monolithic (fresh, mixed domains)...")
    from datasets import load_dataset

    def _load_and_pack(load_fn, n, label):
        texts = load_fn(n)
        ds_full = _pack_texts(texts, tokenizer)
        train_c, _, _ = _split(ds_full)
        print(f"  {label}: {len(train_c)} train chunks")
        return train_c

    def _pack_texts(texts, tok):
        truncated = [t[:5000] for t in texts]
        full = tok(
            "\n\n".join(truncated), return_tensors="pt", truncation=False,
        )["input_ids"][0]
        n_chunks = len(full) // SEQ_LEN
        chunks   = [full[i * SEQ_LEN:(i + 1) * SEQ_LEN] for i in range(n_chunks)]
        ds = PackedChunkDataset()
        ds.chunks = chunks
        return ds

    def _split(ds):
        n = len(ds.chunks)
        return (ds.chunks[:int(n * 0.8)],
                ds.chunks[int(n * 0.8):int(n * 0.9)],
                ds.chunks[int(n * 0.9):])

    def load_code_texts(n):
        ds = load_dataset("code_search_net", "python", split="train",
                          streaming=True, trust_remote_code=True)
        texts = []
        for item in ds:
            content = item.get("whole_func_string", "") or item.get("func_code_string", "")
            if len(content) > 200: texts.append(content)
            if len(texts) >= n: break
        return texts

    def load_science_texts(n):
        ds = load_dataset("allenai/sciq", split="train", streaming=True)
        texts = []
        for item in ds:
            content = (item.get("support", "") + "\n" + item.get("question", "")
                       + "\n" + item.get("correct_answer", ""))
            if len(content) > 100: texts.append(content)
            if len(texts) >= n: break
        return texts

    def load_fiction_texts(n):
        ds = load_dataset("emozilla/pg19", split="train", streaming=True)
        texts = []
        for item in ds:
            content = item.get("text", "")[:5000]
            if len(content) >= 500: texts.append(content)
            if len(texts) >= n: break
        return texts

    N = 3000
    code_train    = _load_and_pack(load_code_texts,    N, "code")
    science_train = _load_and_pack(load_science_texts, N, "science")
    fiction_train = _load_and_pack(load_fiction_texts, N, "fiction")

    mixed_train = code_train + science_train + fiction_train
    print(f"  Mixed train chunks: {len(mixed_train)}")

    # Base eval on verified held-out data
    print("\nEvaluating base model on verified held-out data...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    base_losses = eval_all(base_model, held_out_sets, device)
    base_ew = base_losses["ew"]
    print(f"  Base EW = {base_ew:.6f}  (expected ~2.320049)")
    if abs(base_ew - 2.320049) > 0.005:
        print("  WARNING: base EW differs significantly from expected. "
              "Check held-out data.")
    del base_model
    torch.cuda.empty_cache()

    # ── Train seeds ────────────────────────────────────────────────────────────
    all_seed_results     = []
    all_seed_checkpoints = []

    for seed in SEEDS:
        print(f"\n{'='*55}")
        print(f"SEED {seed}")
        print(f"{'='*55}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
        ).to(device)

        checkpoints = train_monolithic(model, mixed_train, held_out_sets, seed, device)
        all_seed_checkpoints.append(checkpoints)

        final = checkpoints[-1]["held_out"]
        print(f"\n  Final (step {checkpoints[-1]['step']}):")
        print(f"    EW={final['ew']:.4f}  code={final['code']:.4f}  "
              f"science={final['science']:.4f}  fiction={final['fiction']:.4f}")

        seed_result = {"seed": seed, "final_held_out": final, "checkpoints": checkpoints}
        all_seed_results.append(seed_result)

        out_path = RESULTS_DIR / f"monolithic_v2_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump({
                "seed":          seed,
                "model_id":      MODEL_ID,
                "revision":      REVISION,
                "max_steps":     MAX_STEPS,
                "freeze_layers": FREEZE_LAYERS,
                "lr":            LR,
                "base_losses":   base_losses,
                "checkpoints":   checkpoints,
            }, f, indent=2)
        print(f"  Saved: {out_path}")

        del model
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    mono_ew_vals = [r["final_held_out"]["ew"] for r in all_seed_results]
    mono_mean = statistics.mean(mono_ew_vals)
    mono_std  = statistics.stdev(mono_ew_vals) if len(mono_ew_vals) > 1 else 0.0

    moe_ew_mean   = moe_info["moe_ew_mean"]
    mono_vs_base  = round((base_ew - mono_mean) / base_ew * 100, 4)
    moe_vs_base   = moe_info["mean_gain_pct"]
    fused_vs_mono = round((mono_mean - moe_ew_mean) / mono_mean * 100, 4)

    print("\n" + "=" * 70)
    print("6.9B MONOLITHIC v2 — FINAL RESULTS  (corrected, same held-out tokens)")
    print("=" * 70)
    print(f"{'Model':<36} {'EW Loss':>10} {'vs Base':>10}")
    print("-" * 56)
    print(f"{'Base model':<36} {base_ew:>10.4f} {'—':>10}")
    print(f"{'Best specialist':<36} {moe_info['best_spec_ew_mean']:>10.4f}"
          f" {(base_ew - moe_info['best_spec_ew_mean'])/base_ew*100:>+9.2f}%")
    print(f"{'Weight average':<36} {moe_info['weight_avg_ew_mean']:>10.4f}"
          f" {(base_ew - moe_info['weight_avg_ew_mean'])/base_ew*100:>+9.2f}%")
    print(f"{'Monolithic (3000 steps, equal compute)':<36} {mono_mean:>10.4f} {mono_vs_base:>+9.2f}%  ±{mono_std:.4f}")
    print(f"{'KALAVAI MoE':<36} {moe_ew_mean:>10.4f} {moe_vs_base:>+9.2f}%")
    print()
    print(f"MoE vs Monolithic: {fused_vs_mono:+.2f}%  <-- KEY NUMBER FOR TABLE 4")
    if fused_vs_mono > 0:
        print(f"  => MoE BEATS monolithic by {fused_vs_mono:.2f}%  (cooperative fusion wins)")
    else:
        print(f"  => Monolithic beats MoE by {-fused_vs_mono:.2f}%  (expected at 6.9B)")

    save_figure(all_seed_results, base_ew, moe_info)

    summary = {
        "experiment":         "6b_monolithic_v2",
        "version":            2,
        "bugs_fixed":         [
            "MAX_STEPS corrected 6000→3000 (3 specs × 1000)",
            "LR corrected 2e-5→1e-5 (matches 6.9B specialist)",
            "FREEZE_LAYERS corrected 7→6 (matches 6.9B specialist)",
            "Held-out data: loaded from verified pkl (same tokens as corrected eval)",
        ],
        "model_id":           MODEL_ID,
        "revision":           REVISION,
        "total_steps":        MAX_STEPS,
        "equivalent_compute": "3 specialists × 1000 steps",
        "freeze_layers":      FREEZE_LAYERS,
        "lr":                 LR,
        "eval_protocol":      "corrected_equal_weight_same_tokens",
        "seeds":              SEEDS,
        "base_ew":            base_ew,
        "base_per_domain":    base_losses,
        "results": {
            "mean": {
                "monolithic_ew":           round(mono_mean, 6),
                "moe_fused_ew":            round(moe_ew_mean, 6),
                "best_specialist_ew":      round(moe_info["best_spec_ew_mean"], 6),
                "weight_avg_ew":           round(moe_info["weight_avg_ew_mean"], 6),
                "fused_vs_monolithic_pct": fused_vs_mono,
                "monolithic_vs_base_pct":  mono_vs_base,
                "moe_vs_base_pct":         moe_vs_base,
            },
            "std": {
                "monolithic_ew": round(mono_std, 6),
            },
        },
        "per_seed": [
            {
                "seed":          r["seed"],
                "final_ew":      r["final_held_out"]["ew"],
                "final_code":    r["final_held_out"]["code"],
                "final_science": r["final_held_out"]["science"],
                "final_fiction": r["final_held_out"]["fiction"],
            }
            for r in all_seed_results
        ],
        "interpretation": (
            f"MoE beats 6.9B monolithic (equal compute) by {fused_vs_mono:.2f}%"
            if fused_vs_mono > 0
            else f"6.9B monolithic beats MoE by {-fused_vs_mono:.2f}% (expected at scale)"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = RESULTS_DIR / "monolithic_v2_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
