#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: 6.9B Router Gate Analysis (Post-Sweep Extraction)
===========================================================
Extracts router gate distributions and per-domain specialist losses
from the 6.9B step sweep result JSON and saves a dedicated analysis file.

Run this on RunPod AFTER kalavai_6b_step_sweep.py has completed the
best config (steps=2000, freeze=6, seed=42).

Usage:
  cd /workspace/Kalavai
  python experiments/kalavai_6b_router_analysis.py 2>&1 | tee router_analysis_log.txt
"""

import json
import subprocess
import time
from pathlib import Path

# ============================================================================
# Config
# ============================================================================

BEST_STEPS   = 2000
BEST_FREEZE  = 6
BEST_SEED    = 42

# Where the step sweep writes results (relative to workspace root)
RESULTS_DIR  = Path("experiments/results/pythia/pythia_6b_step_sweep")
OUTPUT_FILE  = RESULTS_DIR / "router_analysis_6b.json"


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


def main():
    print("=" * 70)
    print("KALAVAI: 6.9B Router Gate Analysis")
    print("=" * 70)

    # ── Step 1: Load best config result JSON ─────────────────────────────
    result_key = f"result_steps{BEST_STEPS}_k{BEST_FREEZE}_seed{BEST_SEED}.json"
    result_path = RESULTS_DIR / result_key

    if not result_path.exists():
        print(f"\n[ERROR] Result file not found: {result_path}")
        print("  Run kalavai_6b_step_sweep.py first to generate results.")
        return

    data = json.loads(result_path.read_text(encoding="utf-8"))

    print(f"\nLoaded: {result_path}")
    print(f"\nAll keys in result JSON:")
    for k in data.keys():
        print(f"  {k}")

    # ── Step 2: Check for existing router_distribution ───────────────────
    has_router = any("router" in k.lower() or "gate" in k.lower() for k in data.keys())
    print(f"\nHas router/gate data: {has_router}")

    if "router_distribution" in data:
        print("\nRouter distribution found in result JSON:")
        rd = data["router_distribution"]
        print(json.dumps(rd, indent=2))
    else:
        print("\n[WARNING] No router_distribution key. Step sweep may need re-run with updated script.")

    # ── Step 3: Extract per-domain specialist losses ──────────────────────
    print("\n" + "=" * 70)
    print("PER-DOMAIN SPECIALIST LOSSES")
    print("=" * 70)

    eval_heldout = data.get("eval_heldout", {})
    base_losses   = eval_heldout.get("base", {})
    domains       = ["code", "science", "fiction"]
    spec_keys     = ["code_spec", "science_spec", "fiction_spec"]

    print(f"\n{'Model':<20} {'code':>8} {'science':>8} {'fiction':>8} {'mixed':>8}")
    print("-" * 60)
    print(f"{'Base':<20} {base_losses.get('code', 'n/a'):>8} "
          f"{base_losses.get('science', 'n/a'):>8} "
          f"{base_losses.get('fiction', 'n/a'):>8} "
          f"{base_losses.get('mixed', 'n/a'):>8}")

    spec_divergences = {}
    for sk, domain in zip(spec_keys, domains):
        if sk in eval_heldout:
            sl = eval_heldout[sk]
            print(f"{sk:<20} {sl.get('code', 'n/a'):>8} "
                  f"{sl.get('science', 'n/a'):>8} "
                  f"{sl.get('fiction', 'n/a'):>8} "
                  f"{sl.get('mixed', 'n/a'):>8}")
            # Compute divergence: improvement over base on own domain
            own_base = base_losses.get(domain, None)
            own_spec = sl.get(domain, None)
            if own_base and own_spec:
                div_pct = (own_base - own_spec) / own_base * 100
                spec_divergences[domain] = round(div_pct, 2)
                print(f"  → {domain} specialist divergence: {div_pct:.2f}%")

    if "weight_avg" in eval_heldout:
        wa = eval_heldout["weight_avg"]
        print(f"{'weight_avg':<20} {wa.get('code', 'n/a'):>8} "
              f"{wa.get('science', 'n/a'):>8} "
              f"{wa.get('fiction', 'n/a'):>8} "
              f"{wa.get('mixed', 'n/a'):>8}")

    if "moe" in eval_heldout:
        moe = eval_heldout["moe"]
        print(f"{'moe':<20} {moe.get('code', 'n/a'):>8} "
              f"{moe.get('science', 'n/a'):>8} "
              f"{moe.get('fiction', 'n/a'):>8} "
              f"{moe.get('mixed', 'n/a'):>8}")

    print(f"\nKey metrics:")
    print(f"  improvement_vs_spec: {data.get('improvement_vs_spec', 'n/a'):.2f}%")
    print(f"  improvement_vs_base: {data.get('improvement_vs_base', 'n/a'):.2f}%")

    if spec_divergences:
        mean_div = sum(spec_divergences.values()) / len(spec_divergences)
        print(f"\nMean specialist divergence (all domains): {mean_div:.2f}%")
        print(f"Per-domain divergences: {spec_divergences}")

    # ── Step 4: Build and save router_analysis_6b.json ───────────────────
    analysis = {
        "experiment":          "6.9B router gate analysis",
        "source_result":       result_key,
        "steps":               data.get("steps", BEST_STEPS),
        "freeze":              data.get("freeze", BEST_FREEZE),
        "seed":                data.get("seed", BEST_SEED),
        "model_id":            data.get("model_id", "EleutherAI/pythia-6.9b"),
        "revision":            data.get("revision", "step10000"),
        "improvement_vs_spec": data.get("improvement_vs_spec"),
        "improvement_vs_base": data.get("improvement_vs_base"),
        "base_losses":         base_losses,
        "per_domain_specialist_losses": {
            sk: eval_heldout.get(sk, {})
            for sk in spec_keys
        },
        "weight_avg_losses":   eval_heldout.get("weight_avg", {}),
        "moe_losses":          eval_heldout.get("moe", {}),
        "specialist_divergences_pct": spec_divergences,
        "mean_divergence_pct": round(sum(spec_divergences.values()) / len(spec_divergences), 2)
                               if spec_divergences else None,
        "gate_weights_by_domain": data.get("router_distribution", {}),
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_FILE}")

    if analysis["gate_weights_by_domain"]:
        print("\nGate weights by domain (3x3 matrix):")
        print(f"  {'Domain':<12} {'→code':>8} {'→science':>10} {'→fiction':>10}")
        print("-" * 45)
        for domain, weights in analysis["gate_weights_by_domain"].items():
            if domain == "mixed":
                continue
            print(f"  {domain:<12} {weights[0]:>8.4f} {weights[1]:>10.4f} {weights[2]:>10.4f}")

    # ── Step 5: Commit and push ───────────────────────────────────────────
    git_commit_push(
        f"[kalavai] 6.9B: router gate analysis + per-domain specialist losses "
        f"(steps={BEST_STEPS}, k={BEST_FREEZE})"
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
