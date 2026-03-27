#!/usr/bin/env python3
"""
kalavai_math_benchmark.py
Evaluates base model and math specialist (from 5-domain experiment) on GSM8K.
Uses 4-shot chain-of-thought prompting and greedy decoding.

Models: base Pythia-410M, math specialist (checkpoints/pythia/five_domain/math_seed42.pt)
Output: results/pythia/five_domain/gsm8k_benchmark.json

Note: Pythia-410M is a 410M language model trained on general text. GSM8K requires
multi-step arithmetic reasoning. Near-chance accuracy is expected and reported honestly.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "step10000"
MATH_CKPT = Path("checkpoints/pythia/five_domain/math_seed42.pt")
RESULTS_DIR = Path("results/pythia/five_domain")
N_EVAL = 500   # GSM8K test has 1319 examples; 500 is sufficient for a fair estimate
N_SHOT = 4
MAX_NEW_TOKENS = 128
SEED = 42

# 4-shot examples from GSM8K training set (fixed, not from test set)
FEW_SHOT_EXAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. The answer is 72."
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. The answer is 10."
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "In the beginning, Betty has only 100/2 = $50. Betty's grandparents gave her 15*2 = $30. This means Betty now has 50+15+30 = $95. Betty still needs 100-95 = $5 more. The answer is 5."
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read tomorrow?",
        "answer": "Twice the number of pages she read yesterday is 12*2 = 24 pages. After reading yesterday and today, she has 120-12-24 = 84 pages left. She should read 84/2 = 42 pages tomorrow. The answer is 42."
    },
]


def build_prompt(question: str) -> str:
    """Build 4-shot prompt for a GSM8K question."""
    prompt = ""
    for ex in FEW_SHOT_EXAMPLES:
        prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output."""
    # Look for "The answer is X" pattern first
    m = re.search(r'[Tt]he answer is\s*([\-]?\d[\d,\.]*)', text)
    if m:
        return m.group(1).replace(',', '').rstrip('.')
    # Fall back to last number in the text
    nums = re.findall(r'[\-]?\d[\d,\.]*', text)
    if nums:
        return nums[-1].replace(',', '').rstrip('.')
    return None


def normalize_answer(ans: str | None) -> str | None:
    """Normalize to integer string for comparison."""
    if ans is None:
        return None
    try:
        # Handle decimals that are really integers (e.g. "42.0")
        return str(int(float(ans)))
    except (ValueError, OverflowError):
        return ans.strip()


@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, examples, device, model_name: str) -> dict:
    """Run GSM8K evaluation on a model. Returns accuracy and details."""
    model.eval()
    correct = 0
    total = 0
    no_answer = 0
    details = []

    t0 = time.time()
    for i, ex in enumerate(examples):
        question = ex["question"]
        gold_raw = ex["answer"]
        # GSM8K gold answer is at end after "####"
        gold_num = gold_raw.split("####")[-1].strip().replace(',', '')
        gold_norm = normalize_answer(gold_num)

        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred_raw = extract_answer(generated)
        pred_norm = normalize_answer(pred_raw)

        is_correct = (pred_norm is not None and pred_norm == gold_norm)
        if pred_norm is None:
            no_answer += 1
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "question": question[:100],
            "gold": gold_norm,
            "pred": pred_norm,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{model_name}] {i+1}/{len(examples)} — acc={correct/(i+1)*100:.1f}%  ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    accuracy = correct / total * 100
    print(f"  [{model_name}] FINAL: {correct}/{total} = {accuracy:.2f}%  no_answer={no_answer}  ({elapsed:.0f}s)")
    return {
        "accuracy_pct": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "no_answer": no_answer,
        "elapsed_s": round(elapsed, 1),
    }


def load_specialist(base_model, ckpt_path: Path, device):
    """Load a specialist by patching base model state dict."""
    import copy
    model = copy.deepcopy(base_model).cpu()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("\nLoading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = list(ds)[:N_EVAL]
    print(f"  {len(examples)} examples (of {len(ds)} total)")

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "seed": SEED,
        "n_eval": N_EVAL,
        "n_shot": N_SHOT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "note": "Pythia-410M is a 410M general LM; GSM8K requires multi-step arithmetic reasoning. Near-chance accuracy is expected.",
        "models": {}
    }

    # --- 1. Base model ---
    print(f"\n{'='*60}")
    print("1/2  Base model")
    print('='*60)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION)
    base_model = base_model.to(device)
    results["models"]["base"] = evaluate_gsm8k(base_model, tokenizer, examples, device, "base")
    base_model.cpu()
    torch.cuda.empty_cache()

    # --- 2. Math specialist ---
    print(f"\n{'='*60}")
    print("2/2  Math specialist")
    print('='*60)
    if not MATH_CKPT.exists():
        print(f"  ERROR: checkpoint not found at {MATH_CKPT}")
        results["models"]["math_specialist"] = {"error": f"checkpoint not found: {MATH_CKPT}"}
    else:
        math_model = load_specialist(base_model, MATH_CKPT, device)
        results["models"]["math_specialist"] = evaluate_gsm8k(math_model, tokenizer, examples, device, "math_spec")
        math_model.cpu()
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("GSM8K RESULTS — Pythia-410M (seed=42)")
    print('='*60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Correct':>10} {'No Answer':>12}")
    print('-'*60)
    for name, r in results["models"].items():
        if "error" in r:
            print(f"  {name:<23} ERROR: {r['error']}")
        else:
            print(f"  {name:<23} {r['accuracy_pct']:>9.2f}%  {r['correct']:>5}/{r['total']:<5}  {r['no_answer']:>10}")
    print('-'*60)

    base_acc = results["models"].get("base", {}).get("accuracy_pct", None)
    math_acc = results["models"].get("math_specialist", {}).get("accuracy_pct", None)
    if base_acc is not None and math_acc is not None:
        delta = math_acc - base_acc
        results["math_vs_base_delta_pp"] = round(delta, 2)
        print(f"\nMath specialist vs base: {delta:+.2f}pp")
        if math_acc < 5.0:
            print("\nCONCLUSION: Both models score near-chance on GSM8K.")
            print("  This is expected — Pythia-410M is too small for multi-step arithmetic reasoning.")
            print("  Result reported in paper as a known limitation of 410M-scale models.")
            results["conclusion"] = "near_chance_expected_at_410M_scale"
        else:
            results["conclusion"] = "above_chance_result"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "gsm8k_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
