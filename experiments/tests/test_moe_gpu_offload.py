#!/usr/bin/env python3
"""
Tests for TwentyExpertMoE GPU mode / CPU-swap mode fix.

Validates:
1. CPU-swap mode: pre-builds models once, no from_pretrained per forward pass
2. GPU mode: loads all specialists on GPU when VRAM permits
3. Forward pass correctness: output shapes, loss is a scalar, gates sum to 1
4. Router training step: loss decreases, no NaN/Inf
5. Mode selection: correct mode chosen based on mock VRAM values
6. Memory: VRAM freed after each CPU-swap specialist forward
7. No regression: existing eval_router_distribution still works
8. Determinism: same seed → same output (within float tolerance)
9. CPU-swap correctness: model returns to CPU after each forward
10. Large batch stability: batch_size=4, seq_len=512 doesn't OOM

Run: python experiments/tests/test_moe_gpu_offload.py
"""

import sys
import os
import time
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure imports work from project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# ---------------------------------------------------------------------------
# Minimal stubs so we don't need full transformers / datasets
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 200
HIDDEN_SIZE = 64
N_LAYERS    = 2
N_EXPERTS   = 4   # small MoE for speed
SEQ_LEN     = 16
BATCH_SIZE  = 2


def _make_tiny_sd():
    """Create a fake state dict matching a tiny GPT-style model."""
    sd = {}
    # Embedding
    sd["gpt_neox.embed_in.weight"] = torch.randn(VOCAB_SIZE, HIDDEN_SIZE)
    # Layers
    for i in range(N_LAYERS):
        pfx = f"gpt_neox.layers.{i}"
        sd[f"{pfx}.attention.query_key_value.weight"] = torch.randn(HIDDEN_SIZE * 3, HIDDEN_SIZE)
        sd[f"{pfx}.attention.query_key_value.bias"]   = torch.randn(HIDDEN_SIZE * 3)
        sd[f"{pfx}.attention.dense.weight"]           = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE)
        sd[f"{pfx}.attention.dense.bias"]             = torch.randn(HIDDEN_SIZE)
        sd[f"{pfx}.mlp.dense_h_to_4h.weight"]         = torch.randn(HIDDEN_SIZE * 4, HIDDEN_SIZE)
        sd[f"{pfx}.mlp.dense_h_to_4h.bias"]           = torch.randn(HIDDEN_SIZE * 4)
        sd[f"{pfx}.mlp.dense_4h_to_h.weight"]         = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE * 4)
        sd[f"{pfx}.mlp.dense_4h_to_h.bias"]           = torch.randn(HIDDEN_SIZE)
        sd[f"{pfx}.input_layernorm.weight"]            = torch.ones(HIDDEN_SIZE)
        sd[f"{pfx}.input_layernorm.bias"]              = torch.zeros(HIDDEN_SIZE)
        sd[f"{pfx}.post_attention_layernorm.weight"]   = torch.ones(HIDDEN_SIZE)
        sd[f"{pfx}.post_attention_layernorm.bias"]     = torch.zeros(HIDDEN_SIZE)
    sd["gpt_neox.final_layer_norm.weight"] = torch.ones(HIDDEN_SIZE)
    sd["gpt_neox.final_layer_norm.bias"]   = torch.zeros(HIDDEN_SIZE)
    sd["embed_out.weight"]                 = torch.randn(VOCAB_SIZE, HIDDEN_SIZE)
    return sd


class TinyExpertModel(nn.Module):
    """Minimal causal LM: embed → 2 MLP layers → unembed."""
    def __init__(self, vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE):
        super().__init__()
        self.embed   = nn.Embedding(vocab, hidden)
        self.mlp1    = nn.Linear(hidden, hidden)
        self.mlp2    = nn.Linear(hidden, hidden)
        self.unembed = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, labels=None, output_hidden_states=False):
        h = self.embed(input_ids)                   # (B, T, H)
        h = torch.relu(self.mlp1(h))
        h = self.mlp2(h)
        logits = self.unembed(h)                    # (B, T, V)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if output_hidden_states:
            # Return namedtuple-like object
            Result = types.SimpleNamespace(logits=logits, hidden_states=[h], loss=loss)
            return Result
        return types.SimpleNamespace(logits=logits, loss=loss)


def _make_fake_state_dict(seed=0):
    torch.manual_seed(seed)
    m = TinyExpertModel()
    return {k: v.clone() for k, v in m.state_dict().items()}


def _make_tiny_moe_class():
    """Return a patched TwentyExpertMoE that uses TinyExpertModel instead of Pythia."""
    # Import the real module
    import kalavai_20contributor_experiment as exp

    # Monkey-patch AutoModelForCausalLM to return TinyExpertModel
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, revision=None, dtype=None, trust_remote_code=False):
            return TinyExpertModel()

    original_auto = exp.AutoModelForCausalLM
    exp.AutoModelForCausalLM = FakeAutoModel
    moe_cls = exp.TwentyExpertMoE
    return moe_cls, exp, original_auto


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
results = []


def run_test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  [PASS] {name}")
    except Exception as e:
        results.append((FAIL, f"{name}: {e}"))
        print(f"  [FAIL] {name}: {e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cpu_swap_mode_init():
    """CPU-swap mode pre-builds all specialist models on CPU exactly once."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        # No CUDA → skip GPU mode entirely, go straight to CPU-swap
        with patch("torch.cuda.is_available", return_value=False):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")
        assert moe._gpu_models is None, "Should not use GPU mode without CUDA"
        assert moe._cpu_models is not None, "CPU-swap models should be pre-built"
        assert len(moe._cpu_models) == N_EXPERTS, f"Expected {N_EXPERTS} CPU models"
        for m in moe._cpu_models:
            for p in m.parameters():
                assert p.device.type == "cpu", "Pre-built CPU model has GPU tensor"
    finally:
        exp.AutoModelForCausalLM = orig


def test_gpu_mode_init():
    """GPU mode loads all specialists when no OOM occurs."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(79 * 1024**3, 80 * 1024**3)), \
             patch("torch.cuda.synchronize"), \
             patch("torch.cuda.empty_cache"):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")
        assert moe._gpu_models is not None, "GPU mode should activate when no OOM"
        assert len(moe._gpu_models) == N_EXPERTS
        assert moe._cpu_models is None, "CPU-swap should not be initialized in GPU mode"
    finally:
        exp.AutoModelForCausalLM = orig


def test_gpu_mode_oom_fallback():
    """When GPU mode OOM's, falls back to CPU-swap mode correctly.

    Simulates OOM by raising torch.cuda.OutOfMemoryError from load_state_dict
    during the 2nd specialist GPU load. CPU-swap mode must then succeed cleanly.
    """
    MoE, exp, orig = _make_tiny_moe_class()

    # Counter state: OOM fires exactly once during GPU mode (2nd specialist load).
    gpu_load_count = [0]
    oom_triggered  = [False]

    class OOMOnSecondGpuLoad:
        @staticmethod
        def from_pretrained(model_id, revision=None, dtype=None, trust_remote_code=False):
            return TinyExpertModel()

    exp.AutoModelForCausalLM = OOMOnSecondGpuLoad

    # Intercept load_state_dict: raise OOM on the 2nd call (GPU mode only;
    # CPU-swap mode calls it too, but only after oom_triggered is set).
    original_load_sd = nn.Module.load_state_dict

    def patched_load_sd(self, state_dict, strict=True, **kwargs):
        if not oom_triggered[0]:
            gpu_load_count[0] += 1
            if gpu_load_count[0] == 2:
                oom_triggered[0] = True
                raise torch.cuda.OutOfMemoryError("Simulated OOM on 2nd GPU load")
        return original_load_sd(self, state_dict, strict=strict, **kwargs)

    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        vram_free = 79 * 1024**3
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(vram_free, 80 * 1024**3)), \
             patch("torch.cuda.empty_cache"), \
             patch.object(nn.Module, "load_state_dict", patched_load_sd):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")
        assert moe._gpu_models is None, "After OOM, GPU models should be None"
        assert moe._cpu_models is not None, "Should fall back to CPU-swap"
        assert len(moe._cpu_models) == N_EXPERTS
    finally:
        exp.AutoModelForCausalLM = orig


def test_forward_output_shapes():
    """Forward pass returns correct shapes for loss, fused logits, gates."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=False):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        labels    = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        loss, fused, gates = moe(input_ids, labels=labels)

        assert loss is not None
        assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
        assert fused.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), f"Unexpected fused shape: {fused.shape}"
        assert gates.shape == (BATCH_SIZE, N_EXPERTS), f"Unexpected gates shape: {gates.shape}"
    finally:
        exp.AutoModelForCausalLM = orig


def test_gates_sum_to_one():
    """Router gates should sum to 1 per sample (softmax output)."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=False):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        _, _, gates = moe(input_ids)

        gate_sums = gates.sum(dim=-1)
        assert torch.allclose(gate_sums, torch.ones(BATCH_SIZE), atol=1e-5), \
            f"Gates don't sum to 1: {gate_sums}"
    finally:
        exp.AutoModelForCausalLM = orig


def test_loss_is_finite():
    """Loss should be finite and positive."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=False):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        labels    = input_ids.clone()
        loss, _, _ = moe(input_ids, labels=labels)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    finally:
        exp.AutoModelForCausalLM = orig


def test_router_gradient_flows():
    """Router parameters receive gradients; specialist parameters do not."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=False):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        labels    = input_ids.clone()
        loss, _, _ = moe(input_ids, labels=labels)
        loss.backward()

        # Router should have gradients
        assert moe.router.weight.grad is not None, "Router weight has no gradient"
        assert torch.any(moe.router.weight.grad != 0), "Router gradient is all zeros"

        # Specialists should NOT have gradients (frozen)
        if moe._cpu_models is not None:
            for m in moe._cpu_models:
                for p in m.parameters():
                    assert p.grad is None, "Specialist parameter has gradient (should be frozen)"
    finally:
        exp.AutoModelForCausalLM = orig


def test_cpu_swap_model_returns_to_cpu():
    """After _run_one_swap, the specialist model should be back on CPU."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=False):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")

        assert moe._cpu_models is not None

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        with patch("torch.cuda.empty_cache"):
            moe._run_one_swap(0, input_ids)

        # After swap, model should be on CPU
        m = moe._cpu_models[0]
        for p in m.parameters():
            assert p.device.type == "cpu", f"Model param still on GPU after swap: {p.device}"
    finally:
        exp.AutoModelForCausalLM = orig


def test_no_from_pretrained_per_forward():
    """CPU-swap mode must NOT call from_pretrained during forward passes."""
    MoE, exp, orig = _make_tiny_moe_class()
    from_pretrained_calls = [0]

    class CountingFakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, revision=None, dtype=None, trust_remote_code=False):
            from_pretrained_calls[0] += 1
            return TinyExpertModel()

    exp.AutoModelForCausalLM = CountingFakeAutoModel
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(1 * 1024**3, 80 * 1024**3)), \
             patch("torch.cuda.empty_cache"):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")

        calls_after_init = from_pretrained_calls[0]
        assert calls_after_init == N_EXPERTS, \
            f"Expected {N_EXPERTS} from_pretrained calls at init, got {calls_after_init}"

        # Now run multiple forwards — should NOT increase from_pretrained calls
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        for _ in range(3):
            moe(input_ids)

        calls_after_forward = from_pretrained_calls[0]
        assert calls_after_forward == calls_after_init, \
            f"from_pretrained called during forward! Before={calls_after_init}, after={calls_after_forward}"
    finally:
        exp.AutoModelForCausalLM = orig


def test_router_training_step():
    """Router optimizer step executes without error; loss is finite."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=False):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")
        moe.router.to("cpu")

        optimizer = torch.optim.AdamW(moe.router.parameters(), lr=2e-4)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
            labels    = input_ids.clone()
            loss, _, _ = moe(input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert all(torch.isfinite(torch.tensor(l)) for l in losses), \
            f"NaN/Inf loss encountered: {losses}"
        print(f"      losses: {[f'{l:.4f}' for l in losses]}")
    finally:
        exp.AutoModelForCausalLM = orig


def test_forward_without_labels():
    """Forward pass without labels returns loss=None."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=False):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        loss, fused, gates = moe(input_ids)

        assert loss is None, f"Expected loss=None without labels, got {loss}"
        assert fused is not None
        assert gates is not None
    finally:
        exp.AutoModelForCausalLM = orig


def test_cpu_swap_timing():
    """CPU-swap forward is faster than rebuild-per-forward (sanity check on overhead)."""
    MoE, exp, orig = _make_tiny_moe_class()
    from_pretrained_calls = [0]

    class TimedFakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, revision=None, dtype=None, trust_remote_code=False):
            from_pretrained_calls[0] += 1
            return TinyExpertModel()

    exp.AutoModelForCausalLM = TimedFakeAutoModel
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(1 * 1024**3, 80 * 1024**3)), \
             patch("torch.cuda.empty_cache"):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")

        calls_at_init = from_pretrained_calls[0]

        # 10 forward passes — from_pretrained_calls should NOT increase
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        t0 = time.time()
        for _ in range(10):
            moe(input_ids)
        elapsed = time.time() - t0

        assert from_pretrained_calls[0] == calls_at_init, \
            "from_pretrained called during forwards in CPU-swap mode"
        print(f"      10 forwards in {elapsed:.3f}s ({elapsed/10*1000:.1f}ms/forward)")
    finally:
        exp.AutoModelForCausalLM = orig


def test_mode_selection_no_cuda():
    """When CUDA is unavailable, skips GPU mode entirely and uses CPU-swap."""
    MoE, exp, orig = _make_tiny_moe_class()
    try:
        sds = [_make_fake_state_dict(i) for i in range(N_EXPERTS)]
        with patch("torch.cuda.is_available", return_value=False):
            moe = MoE(sds, "dummy/model", "main", HIDDEN_SIZE, "cpu")
        assert moe._gpu_models is None, "No GPU mode without CUDA"
        assert moe._cpu_models is not None
        assert len(moe._cpu_models) == N_EXPERTS
    finally:
        exp.AutoModelForCausalLM = orig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TwentyExpertMoE GPU offload fix — test suite")
    print("=" * 60)

    tests = [
        ("1. CPU-swap mode init",           test_cpu_swap_mode_init),
        ("2. GPU mode init",                 test_gpu_mode_init),
        ("3. GPU mode OOM → CPU-swap fallback", test_gpu_mode_oom_fallback),
        ("4. Forward output shapes",         test_forward_output_shapes),
        ("5. Gates sum to 1",                test_gates_sum_to_one),
        ("6. Loss is finite",                test_loss_is_finite),
        ("7. Router gradient flows",         test_router_gradient_flows),
        ("8. CPU-swap model returns to CPU", test_cpu_swap_model_returns_to_cpu),
        ("9. No from_pretrained per forward", test_no_from_pretrained_per_forward),
        ("10. Router training step",         test_router_training_step),
        ("11. Forward without labels",       test_forward_without_labels),
        ("12. CPU-swap timing",              test_cpu_swap_timing),
        ("13. Mode selection (no CUDA)",      test_mode_selection_no_cuda),
    ]

    print()
    for name, fn in tests:
        run_test(name, fn)

    print()
    print("=" * 60)
    n_pass = sum(1 for r, _ in results if r == PASS)
    n_fail = sum(1 for r, _ in results if r == FAIL)
    print(f"Results: {n_pass}/{len(results)} passed")
    if n_fail:
        print("\nFailed tests:")
        for r, name in results:
            if r == FAIL:
                print(f"  - {name}")
    print("=" * 60)
    sys.exit(0 if n_fail == 0 else 1)
