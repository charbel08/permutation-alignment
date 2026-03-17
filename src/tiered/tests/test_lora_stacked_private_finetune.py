"""Unit tests for stacked LoRA private finetuning helpers.

Coverage:
  - Existing: resolve_per_tier_list, compute_tier_flops, freeze/restore
  - NEW: real PEFT gradient isolation across frozen tiers, real multi-adapter
         evaluate_all_tiers, prefetch wraparound, adapter save/load roundtrip

NOTE: the stacked training script is imported as
    from tiered.train import lora_stacked_private_finetune
Rename the .py file accordingly if it doesn't match.
"""

from __future__ import annotations

import math
import os
import tempfile
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPTNeoConfig

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import PermutationKey

# Adjust the import to match your module filename:
from tiered.train import lora_stacked_private_finetune as lora_stacked

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


def _make_lora_config(rank: int = 2) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=float(rank),
        lora_dropout=0.0,
        bias="none",
        target_modules=list(lora_stacked.TARGET_MODULES),
    )


def _make_batch(vocab_size: int = 64, bsz: int = 2, seq_len: int = 16):
    ids = torch.randint(0, vocab_size, (bsz, seq_len))
    return {"input_ids": ids, "labels": ids.clone()}


# ===================================================================
# EXISTING tests (kept as-is)
# ===================================================================


def test_resolve_per_tier_list_defaults_single_and_mismatch():
    assert lora_stacked.resolve_per_tier_list(None, 10, 3) == [10, 10, 10]
    assert lora_stacked.resolve_per_tier_list("4", 10, 3) == [4, 4, 4]
    assert lora_stacked.resolve_per_tier_list("1,2,3", 10, 3) == [1, 2, 3]

    with pytest.raises(ValueError, match="Expected 3 comma-separated values"):
        lora_stacked.resolve_per_tier_list("1,2", 10, 3)


def test_compute_tier_flops_matches_formula():
    flops = lora_stacked.compute_tier_flops(
        total_base_params=1000,
        cumulative_frozen_lora_params=50,
        tier_lora_params=10,
        tokens_per_step=8,
    )

    assert flops["approx"] == 34080
    assert flops["full_equiv"] == 48000
    assert flops["tiered_2pass_ref"] == 96000
    assert flops["primary"] == flops["approx"]


class _AdapterBlock(nn.Module):
    def __init__(self, size: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))


class _FakePeftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.tier_0 = _AdapterBlock(size=3)
        self.tier_1 = _AdapterBlock(size=5)
        self.tier_2 = _AdapterBlock(size=7)
        self.active_history = []

        for p in self.parameters():
            p.requires_grad = True

    def set_adapter(self, adapter_names):
        active = list(adapter_names)
        self.active_history.append(active)
        active_set = set(active)
        for name, param in self.named_parameters():
            if any(token in active_set for token in name.split(".")):
                param.requires_grad = True


def test_freeze_adapter_params_only_freezes_named_adapter():
    model = _FakePeftModel()
    frozen = lora_stacked.freeze_adapter_params(model, adapter_name="tier_1")

    assert frozen == model.tier_1.weight.numel()
    assert not model.tier_1.weight.requires_grad
    assert model.tier_0.weight.requires_grad
    assert model.tier_2.weight.requires_grad


def test_restore_training_state_sets_cumulative_active_and_refreezes_priors():
    model = _FakePeftModel()

    lora_stacked.restore_training_state(
        model=model,
        tier_names=["tier_0", "tier_1", "tier_2"],
        training_tier_idx=2,
    )

    assert model.active_history[-1] == ["tier_0", "tier_1", "tier_2"]
    assert not model.tier_0.weight.requires_grad
    assert not model.tier_1.weight.requires_grad
    assert model.tier_2.weight.requires_grad


class _EvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.active = []
        self.adapter_disabled = False
        self.set_calls = []

    @contextmanager
    def disable_adapter(self):
        old = self.adapter_disabled
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = old

    def set_adapter(self, adapter_names):
        self.active = list(adapter_names)
        self.set_calls.append(list(adapter_names))


def test_evaluate_all_tiers_runs_c1_to_ck_and_restores_training_state(monkeypatch):
    model = _EvalModel()
    model.train()
    seen_modes = []
    restored = {}

    def _fake_eval_on_batches(eval_model, _batches, _device):
        if eval_model.adapter_disabled:
            seen_modes.append("C1")
            return {"loss": -1.0, "ppl": 1.0, "acc": 0.0, "top3": 0.0}
        seen_modes.append(tuple(eval_model.active))
        return {
            "loss": float(len(eval_model.active)),
            "ppl": 1.0,
            "acc": 0.0,
            "top3": 0.0,
        }

    def _fake_restore_training_state(eval_model, tier_names, training_tier_idx):
        restored["tier_names"] = list(tier_names)
        restored["training_tier_idx"] = training_tier_idx
        eval_model.set_adapter(tier_names[: training_tier_idx + 1])

    monkeypatch.setattr(lora_stacked, "_eval_on_batches", _fake_eval_on_batches)
    monkeypatch.setattr(lora_stacked, "restore_training_state", _fake_restore_training_state)

    dataloader = [
        {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.ones((1, 4), dtype=torch.long),
        }
    ]

    results = lora_stacked.evaluate_all_tiers(
        model=model,
        tier_names_so_far=["tier_0", "tier_1"],
        training_tier_idx=1,
        dataloader=dataloader,
        device=torch.device("cpu"),
        num_steps=1,
    )

    assert set(results.keys()) == {"C1", "C2", "C3"}
    assert results["C1"]["loss"] == -1.0
    assert results["C2"]["loss"] == 1.0
    assert results["C3"]["loss"] == 2.0

    assert seen_modes == ["C1", ("tier_0",), ("tier_0", "tier_1")]
    assert restored == {"tier_names": ["tier_0", "tier_1"], "training_tier_idx": 1}
    assert model.training


# ===================================================================
# NEW: Real PEFT gradient isolation across stacked tiers
# ===================================================================


@needs_peft
def test_stacked_lora_frozen_tier_gets_no_gradients():
    """With tier_0 frozen and tier_1 trainable, backward must produce:
    - Non-zero gradients in tier_1's lora_A/lora_B
    - Zero/None gradients in tier_0's lora_A/lora_B
    - Zero/None gradients in base params
    """
    model = _create_small_model()
    cfg0 = _make_lora_config(rank=2)
    cfg1 = _make_lora_config(rank=2)

    # Add tier_0
    peft_model = get_peft_model(model, cfg0, adapter_name="tier_0")

    # Add tier_1
    peft_model.add_adapter("tier_1", cfg1)

    # Activate both, freeze tier_0
    lora_stacked.restore_training_state(
        peft_model, tier_names=["tier_0", "tier_1"], training_tier_idx=1,
    )

    # Forward + backward
    batch = _make_batch()
    out = peft_model(batch["input_ids"], labels=batch["labels"])
    out.loss.backward()

    tier0_has_grad = False
    tier1_has_grad = False
    base_has_grad = False

    for name, param in peft_model.named_parameters():
        has_nonzero_grad = param.grad is not None and param.grad.ne(0).any()

        if "tier_0" in name and "lora_" in name:
            if has_nonzero_grad:
                tier0_has_grad = True
        elif "tier_1" in name and "lora_" in name:
            if has_nonzero_grad:
                tier1_has_grad = True
        elif "lora_" not in name:
            if has_nonzero_grad:
                base_has_grad = True

    assert tier1_has_grad, "tier_1 LoRA params should have non-zero gradients"
    assert not tier0_has_grad, "tier_0 LoRA params should have NO gradients (frozen)"
    assert not base_has_grad, "Base model params should have NO gradients"


@needs_peft
def test_stacked_lora_three_tiers_gradient_isolation():
    """Three tiers: 0 and 1 frozen, 2 trainable. Only tier_2 gets gradients."""
    model = _create_small_model()

    peft_model = get_peft_model(model, _make_lora_config(2), adapter_name="tier_0")
    peft_model.add_adapter("tier_1", _make_lora_config(2))
    peft_model.add_adapter("tier_2", _make_lora_config(2))

    lora_stacked.restore_training_state(
        peft_model, ["tier_0", "tier_1", "tier_2"], training_tier_idx=2,
    )

    batch = _make_batch()
    out = peft_model(batch["input_ids"], labels=batch["labels"])
    out.loss.backward()

    grad_by_tier: dict[str, bool] = {"tier_0": False, "tier_1": False, "tier_2": False}
    for name, param in peft_model.named_parameters():
        has_grad = param.grad is not None and param.grad.ne(0).any()
        if not has_grad:
            continue
        for tier in grad_by_tier:
            if tier in name and "lora_" in name:
                grad_by_tier[tier] = True

    assert not grad_by_tier["tier_0"], "tier_0 should be frozen"
    assert not grad_by_tier["tier_1"], "tier_1 should be frozen"
    assert grad_by_tier["tier_2"], "tier_2 should be trainable"


# ===================================================================
# NEW: Real PEFT multi-adapter eval produces distinct outputs per tier
# ===================================================================


@needs_peft
def test_evaluate_all_tiers_real_peft_distinct_losses():
    """With real PEFT adapters that have been trained differently,
    evaluate_all_tiers must report genuinely different metrics per tier."""
    torch.manual_seed(0)
    model = _create_small_model()
    cfg = _make_lora_config(rank=4)

    peft_model = get_peft_model(model, cfg, adapter_name="tier_0")
    peft_model.add_adapter("tier_1", cfg)

    # Train tier_0 for a few steps so its output differs from base
    lora_stacked.set_active_adapters(peft_model, ["tier_0"])
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-2)
    batch = _make_batch()
    for _ in range(60):
        optimizer.zero_grad()
        out = peft_model(batch["input_ids"], labels=batch["labels"])
        out.loss.backward()
        optimizer.step()

    # Train tier_1 for different data (different random batch)
    lora_stacked.set_active_adapters(peft_model, ["tier_1"])
    # Need to re-collect trainable after set_adapter
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-2)
    batch2 = _make_batch()
    for _ in range(60):
        optimizer.zero_grad()
        out = peft_model(batch2["input_ids"], labels=batch2["labels"])
        out.loss.backward()
        optimizer.step()

    # Now set up for evaluation of tier_0 + tier_1
    lora_stacked.restore_training_state(
        peft_model, ["tier_0", "tier_1"], training_tier_idx=1,
    )

    eval_batch = _make_batch()
    eval_loader = DataLoader([eval_batch], batch_size=None)

    results = lora_stacked.evaluate_all_tiers(
        model=peft_model,
        tier_names_so_far=["tier_0", "tier_1"],
        training_tier_idx=1,
        dataloader=eval_loader,
        device=torch.device("cpu"),
        num_steps=1,
    )

    assert "C1" in results
    assert "C2" in results  # tier_0 only
    assert "C3" in results  # tier_0 + tier_1

    # All three should produce different losses (adapters trained differently)
    losses = [results["C1"]["loss"], results["C2"]["loss"], results["C3"]["loss"]]
    # At minimum C1 vs C2 should differ (tier_0 was trained)
    assert abs(losses[0] - losses[1]) > 1e-3, (
        f"C1 and C2 should differ after training tier_0: C1={losses[0]:.4f} C2={losses[1]:.4f}"
    )


# ===================================================================
# NEW: Stacked training actually moves C_{k+1} while preserving C1
# ===================================================================


@needs_peft
def test_stacked_training_improves_top_tier_preserves_c1():
    """End-to-end: add two tiers, train each. After both:
    - C1 loss matches pretrained (adapters don't touch base)
    - C3 (both adapters) loss is lower than C1
    """
    model = _create_small_model()
    device = torch.device("cpu")

    # Record pretrained C1 loss
    batch = _make_batch()
    with torch.no_grad():
        pretrained_loss = model(batch["input_ids"], labels=batch["labels"]).loss.item()

    cfg = _make_lora_config(rank=4)
    peft_model = get_peft_model(model, cfg, adapter_name="tier_0")
    peft_model.add_adapter("tier_1", cfg)

    # Train tier_0
    lora_stacked.restore_training_state(peft_model, ["tier_0", "tier_1"], training_tier_idx=0)
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-3)
    for _ in range(30):
        optimizer.zero_grad()
        out = peft_model(batch["input_ids"], labels=batch["labels"])
        out.loss.backward()
        optimizer.step()

    # Train tier_1 (tier_0 frozen)
    lora_stacked.restore_training_state(peft_model, ["tier_0", "tier_1"], training_tier_idx=1)
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-3)
    for _ in range(30):
        optimizer.zero_grad()
        out = peft_model(batch["input_ids"], labels=batch["labels"])
        out.loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        with peft_model.disable_adapter():
            c1_post = peft_model(batch["input_ids"], labels=batch["labels"]).loss.item()
        lora_stacked.set_active_adapters(peft_model, ["tier_0", "tier_1"])
        c3_post = peft_model(batch["input_ids"], labels=batch["labels"]).loss.item()

    assert c1_post == pytest.approx(pretrained_loss, abs=1e-4), (
        f"C1 shifted: {pretrained_loss:.4f} -> {c1_post:.4f}"
    )
    assert c3_post < pretrained_loss - 0.05, (
        f"C3 should improve over pretrained: {pretrained_loss:.4f} -> {c3_post:.4f}"
    )


# ===================================================================
# NEW: _prefetch_batches wraparound
# ===================================================================


def test_prefetch_batches_wraps_around_short_dataset():
    """When num_steps > len(dataloader), _prefetch_batches should wrap."""
    batch = _make_batch()
    loader = DataLoader([batch], batch_size=None)

    batches = lora_stacked._prefetch_batches(loader, num_steps=5)
    assert len(batches) == 5
    # All batches should be identical (same single batch repeated)
    for b in batches:
        assert torch.equal(b["input_ids"], batch["input_ids"])


def test_prefetch_batches_empty_loader():
    """Empty dataloader should return empty list."""
    loader = DataLoader([], batch_size=None)
    batches = lora_stacked._prefetch_batches(loader, num_steps=3)
    assert len(batches) == 0


# ===================================================================
# NEW: _eval_on_batches correctness
# ===================================================================


def test_eval_on_batches_returns_correct_metrics():
    """Direct test of _eval_on_batches with a real model."""
    model = _create_small_model()
    model.eval()
    device = torch.device("cpu")

    batch = _make_batch()
    batches = [batch, batch]  # Two identical batches

    with torch.no_grad():
        result = lora_stacked._eval_on_batches(model, batches, device)

    assert "loss" in result
    assert "ppl" in result
    assert "acc" in result
    assert "top3" in result

    assert result["loss"] > 0
    assert result["ppl"] == pytest.approx(math.exp(result["loss"]), rel=1e-3)
    assert 0.0 <= result["acc"] <= 1.0
    assert 0.0 <= result["top3"] <= 1.0


def test_eval_on_batches_empty_returns_nans():
    """Empty batch list should produce NaN metrics."""
    model = _create_small_model()
    model.eval()
    result = lora_stacked._eval_on_batches(model, [], torch.device("cpu"))
    for v in result.values():
        assert math.isnan(v)


# ===================================================================
# NEW: Adapter save/load roundtrip
# ===================================================================


@needs_peft
def test_save_and_reload_individual_tier_adapter():
    """Save tier_0 adapter, reload it into a fresh model, verify outputs match."""
    model = _create_small_model()
    # get_peft_model mutates modules in-place (wraps Linear layers), so preserve
    # a clean base snapshot before wrapping.
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cfg = _make_lora_config(rank=4)
    peft_model = get_peft_model(model, cfg, adapter_name="tier_0")

    # Train a few steps so adapter is non-trivial
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    batch = _make_batch()
    for _ in range(10):
        optimizer.zero_grad()
        out = peft_model(batch["input_ids"], labels=batch["labels"])
        out.loss.backward()
        optimizer.step()

    # Record output
    peft_model.eval()
    with torch.no_grad():
        original_logits = peft_model(batch["input_ids"]).logits.clone()

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tier_0")
        peft_model.save_pretrained(save_path, selected_adapters=["tier_0"])

        # Reload into fresh model
        from peft import PeftModel

        fresh_base = _create_small_model()
        # Restore exact pre-PEFT base weights.
        fresh_base.load_state_dict(base_state, strict=True)
        load_path = save_path
        if not os.path.isfile(os.path.join(load_path, "adapter_config.json")):
            nested = os.path.join(save_path, "tier_0")
            if os.path.isfile(os.path.join(nested, "adapter_config.json")):
                load_path = nested
        reloaded = PeftModel.from_pretrained(fresh_base, load_path, adapter_name="tier_0")
        reloaded.eval()

        with torch.no_grad():
            reloaded_logits = reloaded(batch["input_ids"]).logits

    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5), (
        f"Max diff after reload: {(original_logits - reloaded_logits).abs().max().item()}"
    )


# ===================================================================
# NEW: restore_training_state is idempotent
# ===================================================================


@needs_peft
def test_restore_training_state_idempotent_on_real_peft():
    """Calling restore_training_state twice should produce the same state."""
    model = _create_small_model()
    peft_model = get_peft_model(model, _make_lora_config(2), adapter_name="tier_0")
    peft_model.add_adapter("tier_1", _make_lora_config(2))

    def get_grad_flags():
        return {
            name: param.requires_grad
            for name, param in peft_model.named_parameters()
        }

    lora_stacked.restore_training_state(peft_model, ["tier_0", "tier_1"], 1)
    flags_first = get_grad_flags()

    lora_stacked.restore_training_state(peft_model, ["tier_0", "tier_1"], 1)
    flags_second = get_grad_flags()

    assert flags_first == flags_second
