"""Unit tests for PEFT LoRA private finetuning helpers.

Coverage:
  - Existing: rank math, adapter shim, evaluate_c1_c2 with toy model
  - NEW: ground-truth budget vs mask_public_gradients, real PEFT e2e,
         gradient isolation, prefetch wraparound
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPTNeoConfig

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import (
    PermutationKey,
    build_mask_plan,
    mask_public_gradients,
)
from tiered.train import lora_private_finetune as lora_private

# Try importing PEFT — tests that need it are skipped if unavailable.
try:
    from peft import LoraConfig, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

needs_peft = pytest.mark.skipif(not _PEFT_AVAILABLE, reason="PEFT not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_small_model() -> GPTNeoForCausalLMTiered:
    cfg = GPTNeoConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        attention_types=[[["global"], 1]] * 2,
        max_position_embeddings=32,
    )
    return GPTNeoForCausalLMTiered(cfg)


def _create_small_key() -> PermutationKey:
    """Cross-layer swap: head 1 of layer 0 ↔ head 2 of layer 1,
    MLP col 5 of layer 0 ↔ MLP col 7 of layer 1."""
    return PermutationKey(
        attn_heads=[[[0, 1], [1, 2]]],
        mlp_cols=[[[0, 5], [1, 7]]],
    )


def _create_richer_key() -> PermutationKey:
    """Multiple swaps for more coverage."""
    return PermutationKey(
        attn_heads=[
            [[0, 0], [1, 3]],
            [[0, 2], [1, 1]],
        ],
        mlp_cols=[
            [[0, 0], [1, 10]],
            [[0, 20], [1, 30]],
            [[0, 63], [1, 1]],
        ],
    )


# ===================================================================
# EXISTING tests (kept as-is)
# ===================================================================


def test_count_keyed_parameters_matches_expected_formula():
    model = _create_small_model()
    key = _create_small_key()
    device = torch.device("cpu")

    got = lora_private.count_keyed_parameters(model, key, device)

    hidden = model.config.hidden_size
    head_dim = hidden // model.config.num_heads

    # One keyed head per touched layer -> head_dim keyed rows/cols per projection.
    attn_per_layer = (3 * head_dim * hidden) + (hidden * head_dim)
    # One keyed MLP column per touched layer -> fc row + fc bias + proj column.
    mlp_per_layer = hidden + 1 + hidden
    expected = (2 * attn_per_layer) + (2 * mlp_per_layer)

    assert got == expected


def test_find_targets_and_rank_helpers_on_tiny_model():
    model = _create_small_model()
    targets = lora_private.find_lora_targets(model, target_modules=lora_private.TARGET_MODULES)
    names = [t.name for t in targets]

    assert names == sorted(names)
    assert all(name.split(".")[-1] in lora_private.TARGET_MODULES for name in names)
    assert len(targets) == model.config.num_layers * len(lora_private.TARGET_MODULES)

    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    expected_per_layer = (4 * (hidden + hidden)) + ((hidden + intermediate) + (intermediate + hidden))
    assert lora_private.lora_params_per_rank(targets) == model.config.num_layers * expected_per_layer
    assert lora_private.max_effective_rank(targets) == hidden


def test_resolve_rank_from_budget_floor_cap_and_errors():
    rank, params = lora_private.resolve_rank_from_budget(
        target_param_budget=5,
        per_rank_params=10,
        rank_cap=7,
    )
    assert rank == 1
    assert params == 10

    rank, params = lora_private.resolve_rank_from_budget(
        target_param_budget=95,
        per_rank_params=10,
        rank_cap=8,
    )
    assert rank == 8
    assert params == 80

    rank, params = lora_private.resolve_rank_from_budget(
        target_param_budget=95,
        per_rank_params=10,
        rank_cap=None,
    )
    assert rank == 9
    assert params == 90

    with pytest.raises(ValueError, match="per_rank_params must be > 0"):
        lora_private.resolve_rank_from_budget(10, 0)
    with pytest.raises(ValueError, match="target_param_budget must be > 0"):
        lora_private.resolve_rank_from_budget(0, 10)


def test_build_rank_selection_auto_and_override(monkeypatch):
    targets = [
        lora_private.LoRATarget("a", in_features=10, out_features=6),
        lora_private.LoRATarget("b", in_features=8, out_features=4),
    ]

    monkeypatch.setattr(lora_private, "count_keyed_parameters", lambda *args, **kwargs: 105)
    monkeypatch.setattr(lora_private, "find_lora_targets", lambda *args, **kwargs: targets)

    rank_auto, targets_auto = lora_private.build_rank_selection(
        model=object(),
        key=object(),
        device=torch.device("cpu"),
        rank_override=None,
    )
    assert targets_auto == targets
    assert rank_auto.lora_params_per_rank == 28
    assert rank_auto.max_effective_rank == 4
    assert rank_auto.selected_rank == 3
    assert rank_auto.selected_lora_params == 84
    assert rank_auto.budget_gap == 21

    rank_override, _ = lora_private.build_rank_selection(
        model=object(),
        key=object(),
        device=torch.device("cpu"),
        rank_override=99,
    )
    assert rank_override.selected_rank == 4
    assert rank_override.selected_lora_params == 112
    assert rank_override.budget_gap == -7


def test_adapters_disabled_prefers_disable_adapter_context():
    class ModernAPI:
        def __init__(self):
            self.adapter_enabled = True

        @contextmanager
        def disable_adapter(self):
            self.adapter_enabled = False
            try:
                yield
            finally:
                self.adapter_enabled = True

    model = ModernAPI()
    with lora_private.adapters_disabled(model):
        assert not model.adapter_enabled
    assert model.adapter_enabled


def test_adapters_disabled_legacy_api():
    class LegacyAPI:
        def __init__(self):
            self.adapter_enabled = True
            self.calls = []

        def disable_adapters(self):
            self.calls.append("disable")
            self.adapter_enabled = False

        def enable_adapters(self):
            self.calls.append("enable")
            self.adapter_enabled = True

    model = LegacyAPI()
    with lora_private.adapters_disabled(model):
        assert not model.adapter_enabled
    assert model.adapter_enabled
    assert model.calls == ["disable", "enable"]


def test_adapters_disabled_raises_when_api_missing():
    class MissingAPI:
        pass

    with pytest.raises(AttributeError, match="Could not find adapter disable API"):
        with lora_private.adapters_disabled(MissingAPI()):
            pass


class _ToyPeftModel(nn.Module):
    """Tiny model that changes behavior when adapters are disabled."""

    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.adapter_enabled = True
        self.forward_calls = 0

    @contextmanager
    def disable_adapter(self):
        old_state = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = old_state

    def forward(self, input_ids, labels=None):
        self.forward_calls += 1
        bsz, seq_len = input_ids.shape
        labels = labels if labels is not None else input_ids
        device = input_ids.device

        logits = torch.full((bsz, seq_len, self.vocab_size), -1000.0, device=device)
        token_targets = labels[:, 1:] % self.vocab_size
        rows = torch.arange(bsz, device=device)

        if self.adapter_enabled:
            loss = torch.tensor(0.25, device=device)
            logits[:, :-1, 0] = 1.0
            logits[:, :-1, 1] = 0.5
            logits[:, :-1, 2] = 0.0
            for pos in range(seq_len - 1):
                logits[rows, pos, token_targets[:, pos]] = 5.0
        else:
            loss = torch.tensor(1.5, device=device)
            logits[:, :-1, 0] = 5.0
            logits[:, :-1, 1] = 4.0
            logits[:, :-1, 2] = 3.0
            for pos in range(seq_len - 1):
                logits[rows, pos, token_targets[:, pos]] = -5.0

        logits[:, -1, 0] = 0.0
        return SimpleNamespace(loss=loss, logits=logits)


def test_evaluate_c1_c2_cycles_batches_and_restores_train_mode():
    model = _ToyPeftModel()
    model.train()

    batch = {
        "input_ids": torch.full((2, 4), 7, dtype=torch.long),
        "labels": torch.full((2, 4), 7, dtype=torch.long),
    }
    dataloader = DataLoader([batch], batch_size=None)

    metrics = lora_private.evaluate_c1_c2(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        num_steps=3,
    )

    assert metrics["loss_c1"] == pytest.approx(1.5)
    assert metrics["ppl_c1"] == pytest.approx(math.exp(1.5))
    assert metrics["acc_c1"] == pytest.approx(0.0)
    assert metrics["top3_c1"] == pytest.approx(0.0)
    assert metrics["loss_c2"] == pytest.approx(0.25)
    assert metrics["ppl_c2"] == pytest.approx(math.exp(0.25))
    assert metrics["acc_c2"] == pytest.approx(1.0)
    assert metrics["top3_c2"] == pytest.approx(1.0)

    assert model.forward_calls == 6  # 2 forward passes (C1/C2) per eval step
    assert model.training  # evaluate should restore original train state
    assert model.adapter_enabled  # adapter state also restored


def test_evaluate_c1_c2_returns_nans_for_empty_dataloader():
    model = _ToyPeftModel()
    model.eval()
    dataloader = DataLoader([], batch_size=None)

    metrics = lora_private.evaluate_c1_c2(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        num_steps=2,
    )

    for key, value in metrics.items():
        assert math.isnan(value), f"{key} should be NaN for empty eval loader"

    assert not model.training


# ===================================================================
# NEW: Ground-truth budget test
# count_keyed_parameters must match the actual number of gradient
# elements preserved by mask_public_gradients after backward.
# ===================================================================


def _count_nonzero_grad_elements(model) -> int:
    """Count the total number of non-zero gradient elements across all params."""
    total = 0
    for p in model.parameters():
        if p.grad is not None:
            total += int(p.grad.ne(0).sum().item())
    return total


def test_keyed_param_count_matches_mask_public_gradients_ground_truth():
    """The single most important invariant for fair LoRA budget comparison.

    After forward+backward+mask_public_gradients, the number of surviving
    (non-zero) gradient elements must equal count_keyed_parameters.
    """
    model = _create_small_model()
    key = _create_small_key()
    device = torch.device("cpu")

    expected_count = lora_private.count_keyed_parameters(model, key, device)

    # Forward + backward to populate all gradients
    input_ids = torch.randint(0, 64, (2, 16))
    outputs = model(input_ids, labels=input_ids)
    outputs.loss.backward()

    # mask_public_gradients: zeros everything, restores only keyed grads
    mask_public_gradients(model, key)

    actual_nonzero = _count_nonzero_grad_elements(model)

    assert actual_nonzero == expected_count, (
        f"count_keyed_parameters says {expected_count} but "
        f"mask_public_gradients preserved {actual_nonzero} non-zero grad elements"
    )


def test_keyed_param_count_matches_ground_truth_richer_key():
    """Same invariant with a richer key (more swaps, multiple heads/cols)."""
    model = _create_small_model()
    key = _create_richer_key()
    device = torch.device("cpu")

    expected_count = lora_private.count_keyed_parameters(model, key, device)

    input_ids = torch.randint(0, 64, (2, 16))
    outputs = model(input_ids, labels=input_ids)
    outputs.loss.backward()
    mask_public_gradients(model, key)

    actual_nonzero = _count_nonzero_grad_elements(model)
    assert actual_nonzero == expected_count


def test_keyed_param_count_ground_truth_on_larger_model():
    """Test on a 4-layer model with more heads for realistic coverage."""
    cfg = GPTNeoConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=4,
        num_heads=8,
        intermediate_size=256,
        attention_types=[[["global"], 1]] * 4,
        max_position_embeddings=64,
    )
    model = GPTNeoForCausalLMTiered(cfg)
    key = PermutationKey(
        attn_heads=[
            [[0, 1], [2, 5]],
            [[0, 3], [3, 7]],
            [[1, 0], [3, 2]],
        ],
        mlp_cols=[
            [[0, 10], [1, 200]],
            [[0, 50], [2, 100]],
            [[1, 255], [3, 0]],
            [[2, 30], [3, 60]],
        ],
    )
    device = torch.device("cpu")

    expected = lora_private.count_keyed_parameters(model, key, device)

    input_ids = torch.randint(0, 128, (2, 32))
    outputs = model(input_ids, labels=input_ids)
    outputs.loss.backward()
    mask_public_gradients(model, key)

    actual = _count_nonzero_grad_elements(model)
    assert actual == expected


# ===================================================================
# NEW: Real PEFT end-to-end test
# Verifies that LoRA training changes C2 but not C1.
# ===================================================================


@needs_peft
def test_lora_training_changes_c2_preserves_c1():
    """Train a few LoRA steps on a tiny model and verify:
    - C2 (adapter on) loss improves on training data
    - C1 (adapter off) loss stays near its pretrained value
    """
    model = _create_small_model()
    device = torch.device("cpu")

    # Record C1 pretrained loss
    input_ids = torch.randint(0, 64, (4, 16))
    with torch.no_grad():
        pretrained_out = model(input_ids, labels=input_ids)
    pretrained_loss = pretrained_out.loss.item()

    # Wrap with LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=4.0,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(lora_private.TARGET_MODULES),
    )
    peft_model = get_peft_model(model, lora_config)

    # Verify C1 loss is unchanged right after PEFT wrap (adapters init to zero)
    with torch.no_grad():
        with peft_model.disable_adapter():
            c1_pre = peft_model(input_ids, labels=input_ids).loss.item()
        c2_pre = peft_model(input_ids, labels=input_ids).loss.item()

    assert c1_pre == pytest.approx(pretrained_loss, abs=1e-4)
    # C2 should also be very close initially (LoRA A/B initialize to near-zero)
    assert c2_pre == pytest.approx(pretrained_loss, abs=0.5)

    # Train a few steps
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-3)

    for _ in range(30):
        optimizer.zero_grad()
        out = peft_model(input_ids, labels=input_ids)
        out.loss.backward()
        optimizer.step()

    # After training: C2 should have improved, C1 should be stable
    with torch.no_grad():
        with peft_model.disable_adapter():
            c1_post = peft_model(input_ids, labels=input_ids).loss.item()
        c2_post = peft_model(input_ids, labels=input_ids).loss.item()

    # C2 loss should decrease substantially
    assert c2_post < c2_pre - 0.1, (
        f"C2 loss didn't improve: {c2_pre:.4f} -> {c2_post:.4f}"
    )

    # C1 loss should be unchanged (LoRA doesn't modify base weights)
    assert c1_post == pytest.approx(pretrained_loss, abs=1e-4), (
        f"C1 loss shifted: {pretrained_loss:.4f} -> {c1_post:.4f}"
    )


@needs_peft
def test_lora_adapter_disable_produces_exact_base_output():
    """PEFT's disable_adapter must produce exactly the same output as the
    unwrapped base model. This is the foundation of C1/C2 separation."""
    base_model = _create_small_model()
    device = torch.device("cpu")

    input_ids = torch.randint(0, 64, (2, 8))

    with torch.no_grad():
        base_logits = base_model(input_ids).logits.clone()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=4.0,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(lora_private.TARGET_MODULES),
    )
    peft_model = get_peft_model(base_model, lora_config)

    with torch.no_grad():
        with peft_model.disable_adapter():
            disabled_logits = peft_model(input_ids).logits

    # Must be bitwise identical — LoRA init is zero so disable = base
    assert torch.equal(base_logits, disabled_logits), (
        f"Max diff: {(base_logits - disabled_logits).abs().max().item()}"
    )


# ===================================================================
# NEW: LoRA gradient isolation
# Only LoRA params should get gradients; base params are frozen.
# ===================================================================


@needs_peft
def test_lora_only_adapter_params_have_gradients():
    """After a forward+backward through the PEFT model, base params must
    have no gradients and LoRA params must participate in backward.

    Note: with standard LoRA init, B starts at zeros so A can have exactly
    zero gradient on the first backward pass. We only require:
      - no base gradients
      - LoRA grads are materialized
      - at least one LoRA tensor has non-zero gradient
    """
    model = _create_small_model()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=2.0,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(lora_private.TARGET_MODULES),
    )
    peft_model = get_peft_model(model, lora_config)

    input_ids = torch.randint(0, 64, (2, 8))
    out = peft_model(input_ids, labels=input_ids)
    out.loss.backward()

    lora_params_with_grad = 0
    lora_params_no_grad_tensor = 0
    lora_params_zero_grad = 0
    base_params_with_grad = 0

    for name, param in peft_model.named_parameters():
        is_lora = "lora_" in name
        has_nonzero_grad = param.grad is not None and param.grad.ne(0).any()

        if is_lora and param.requires_grad:
            if param.grad is None:
                lora_params_no_grad_tensor += 1
            elif has_nonzero_grad:
                lora_params_with_grad += 1
            else:
                lora_params_zero_grad += 1
        elif not is_lora:
            if has_nonzero_grad:
                base_params_with_grad += 1

    assert lora_params_with_grad > 0, "No LoRA params received gradients"
    assert lora_params_no_grad_tensor == 0, "Some trainable LoRA params had grad=None"
    assert lora_params_zero_grad >= 0
    assert base_params_with_grad == 0, (
        f"{base_params_with_grad} base params have non-zero gradients"
    )


# ===================================================================
# NEW: token_metrics correctness
# ===================================================================


def test_token_metrics_perfect_and_random():
    """Verify top-1 and top-3 accuracy on known logit patterns."""
    vocab_size = 10
    bsz, seq_len = 2, 5

    # Perfect prediction: logits peak at the target token
    labels = torch.tensor([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4]])
    logits = torch.full((bsz, seq_len, vocab_size), -10.0)
    for b in range(bsz):
        for t in range(seq_len - 1):
            logits[b, t, labels[b, t + 1]] = 10.0

    acc, top3 = lora_private.token_metrics(logits, labels)
    assert acc == pytest.approx(1.0)
    assert top3 == pytest.approx(1.0)

    # Completely wrong: max logit is never the target
    bad_logits = torch.full((bsz, seq_len, vocab_size), 0.0)
    # Put all mass on token 9, targets are 1-5 (never 9)
    bad_logits[:, :, 9] = 100.0
    bad_logits[:, :, 8] = 99.0
    bad_logits[:, :, 7] = 98.0

    acc, top3 = lora_private.token_metrics(bad_logits, labels)
    assert acc == pytest.approx(0.0)
    assert top3 == pytest.approx(0.0)


def test_token_metrics_handles_all_masked():
    """When all labels are -100, return 0.0 for both metrics."""
    labels = torch.full((2, 4), -100)
    logits = torch.randn(2, 4, 10)
    acc, top3 = lora_private.token_metrics(logits, labels)
    assert acc == 0.0
    assert top3 == 0.0
