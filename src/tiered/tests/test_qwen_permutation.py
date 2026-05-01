"""Tests for Qwen/Llama-style permutation helpers and ablation utilities."""

import math
import random

import pytest
import torch

qwen2_cfg_mod = pytest.importorskip("transformers.models.qwen2.configuration_qwen2")
qwen2_model_mod = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")

from tiered.permutation.key import PermutationKey
from tiered.permutation.qwen import (
    _allocate_qwen_swap_counts,
    _qwen_swap_param_sizes,
    apply_qwen_permutation,
    count_qwen_keyed_params,
    count_qwen_swappable_params,
    generate_qwen_key,
    get_qwen_arch,
    unapply_qwen_permutation,
    validate_qwen_key,
)

# Import the nested key generator from the ablation script.
# Adjust the import path to match your project layout.
try:
    from scripts.eval.qwen_key_destruction_ablation import (
        generate_nested_keys,
        _wilson_ci,
    )
    _ABLATION_AVAILABLE = True
except ImportError:
    _ABLATION_AVAILABLE = False

needs_ablation = pytest.mark.skipif(not _ABLATION_AVAILABLE, reason="ablation script not importable")

Qwen2Config = qwen2_cfg_mod.Qwen2Config
Qwen2ForCausalLM = qwen2_model_mod.Qwen2ForCausalLM


def create_qwen_model():
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(config)


# ===================================================================
# Original tests (preserved)
# ===================================================================


def test_qwen_apply_unapply_identity():
    torch.manual_seed(0)
    model = create_qwen_model()
    arch = get_qwen_arch(model)
    key = generate_qwen_key(arch, target_pct=0.10, attn_ratio=0.25, seed=1)

    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    apply_qwen_permutation(model, key)
    unapply_qwen_permutation(model, key)

    for name, param in model.state_dict().items():
        assert torch.equal(param, state_before[name]), f"{name} not restored after apply+unapply"


def test_qwen_generate_key_counts_and_bounds():
    model = create_qwen_model()
    arch = get_qwen_arch(model)

    key = generate_qwen_key(arch, target_pct=0.15, attn_ratio=0.30, seed=7)
    validate_qwen_key(key, arch)

    swappable = count_qwen_swappable_params(arch)
    keyed = count_qwen_keyed_params(arch, key)

    assert keyed["total"] > 0
    assert keyed["total"] <= swappable["total"]
    assert keyed["attention"] >= 0
    assert keyed["mlp"] >= 0


def test_qwen_small_budget_uses_mlp_until_attention_swap_fits():
    model = create_qwen_model()
    arch = get_qwen_arch(model)
    swappable = count_qwen_swappable_params(arch)
    params_per_attn_swap, params_per_mlp_swap = _qwen_swap_param_sizes(arch)

    # At 10% on this tiny GQA fixture, the 25% attention budget is smaller
    # than one attention swap, so the whole target budget should go to MLP.
    target_pct = 0.10
    target_total = int(swappable["total"] * target_pct)
    assert int(target_total * 0.25) < params_per_attn_swap

    n_attn, n_mlp = _allocate_qwen_swap_counts(arch, target_pct, attn_ratio=0.25)
    assert n_attn == 0
    assert n_mlp == target_total // params_per_mlp_swap

    key = generate_qwen_key(arch, target_pct=target_pct, attn_ratio=0.25, seed=7)
    assert len(key.attn_heads) == 0
    assert len(key.mlp_cols) == n_mlp


def test_qwen_permutation_changes_weights():
    torch.manual_seed(123)
    model = create_qwen_model()

    key = PermutationKey(
        attn_heads=[[[0, 0], [1, 1]]],
        mlp_cols=[[[2, 3], [3, 7]]],
    )

    q_l0_before = model.model.layers[0].self_attn.q_proj.weight.clone()
    apply_qwen_permutation(model, key)
    q_l0_after = model.model.layers[0].self_attn.q_proj.weight

    assert not torch.equal(q_l0_before, q_l0_after), \
        "Expected q_proj weights to change after permutation"


# ===================================================================
# NEW: Logits-level identity (not just weight-level)
# ===================================================================


def test_qwen_apply_unapply_preserves_logits():
    """After apply+unapply, the model must produce bit-identical logits.

    This catches issues that weight-level checks miss, e.g., KV cache
    pollution or stateful hooks that break on in-place weight modification.
    """
    torch.manual_seed(42)
    model = create_qwen_model()
    model.eval()
    arch = get_qwen_arch(model)
    key = generate_qwen_key(arch, target_pct=0.20, attn_ratio=0.25, seed=99)

    input_ids = torch.randint(0, 128, (2, 32))

    with torch.no_grad():
        logits_before = model(input_ids).logits.clone()

    apply_qwen_permutation(model, key)
    unapply_qwen_permutation(model, key)

    with torch.no_grad():
        logits_after = model(input_ids).logits

    assert torch.equal(logits_before, logits_after), (
        f"Logits differ after apply+unapply. Max diff: "
        f"{(logits_before - logits_after).abs().max().item()}"
    )


def test_qwen_permutation_changes_logits():
    """Applying a key must actually change model outputs (C2 ≠ C1)."""
    torch.manual_seed(42)
    model = create_qwen_model()
    model.eval()
    arch = get_qwen_arch(model)
    # Use 20% so this tiny test config reliably includes at least one
    # attention swap (15% can quantize to 0 attention swaps).
    key = generate_qwen_key(arch, target_pct=0.20, attn_ratio=0.25, seed=7)

    input_ids = torch.randint(0, 128, (2, 16))

    with torch.no_grad():
        logits_c1 = model(input_ids).logits.clone()

    apply_qwen_permutation(model, key)
    with torch.no_grad():
        logits_c2 = model(input_ids).logits.clone()
    unapply_qwen_permutation(model, key)

    assert not torch.equal(logits_c1, logits_c2), \
        "C1 and C2 logits are identical — permutation has no effect"

    # The difference should be substantial, not just floating-point noise
    max_diff = (logits_c1 - logits_c2).abs().max().item()
    assert max_diff > 0.1, f"Logit difference is suspiciously small: {max_diff}"


# ===================================================================
# NEW: Nested key properties
# ===================================================================


@needs_ablation
def test_nested_keys_are_strict_prefixes():
    """Smaller keys must be strict prefixes of larger keys."""
    model = create_qwen_model()
    arch = get_qwen_arch(model)
    pcts = [0.05, 0.10, 0.20, 0.50]
    keys = generate_nested_keys(arch, pcts, attn_ratio=0.25, seed=42)

    assert len(keys) == len(pcts)

    for i in range(1, len(keys)):
        smaller = keys[i - 1]
        larger = keys[i]

        # Attention swaps: smaller is a prefix of larger
        assert len(smaller.attn_heads) <= len(larger.attn_heads), (
            f"pct={pcts[i-1]} has more attn swaps than pct={pcts[i]}"
        )
        assert larger.attn_heads[:len(smaller.attn_heads)] == smaller.attn_heads, (
            f"Attn heads not nested between pct={pcts[i-1]} and pct={pcts[i]}"
        )

        # MLP swaps: smaller is a prefix of larger
        assert len(smaller.mlp_cols) <= len(larger.mlp_cols), (
            f"pct={pcts[i-1]} has more mlp swaps than pct={pcts[i]}"
        )
        assert larger.mlp_cols[:len(smaller.mlp_cols)] == smaller.mlp_cols, (
            f"MLP cols not nested between pct={pcts[i-1]} and pct={pcts[i]}"
        )


@needs_ablation
def test_nested_keys_keyed_params_monotonically_increase():
    """The keyed parameter count must grow with key percentage."""
    model = create_qwen_model()
    arch = get_qwen_arch(model)
    pcts = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    keys = generate_nested_keys(arch, pcts, attn_ratio=0.25, seed=42)

    counts = [count_qwen_keyed_params(arch, k)["total"] for k in keys]
    for i in range(1, len(counts)):
        assert counts[i] >= counts[i - 1], (
            f"Keyed params decreased: pct={pcts[i-1]} has {counts[i-1]}, "
            f"pct={pcts[i]} has {counts[i]}"
        )


@needs_ablation
def test_nested_keys_single_pct_matches_standalone():
    """A nested key at a single pct should produce the same key as generate_qwen_key."""
    model = create_qwen_model()
    arch = get_qwen_arch(model)

    # generate_nested_keys with one pct should match generate_qwen_key with same seed
    nested = generate_nested_keys(arch, [0.15], attn_ratio=0.25, seed=42)
    standalone = generate_qwen_key(arch, target_pct=0.15, attn_ratio=0.25, seed=42)

    assert nested[0].attn_heads == standalone.attn_heads
    assert nested[0].mlp_cols == standalone.mlp_cols


# ===================================================================
# NEW: Wilson CI sanity checks
# ===================================================================


@needs_ablation
def test_wilson_ci_known_values():
    """Verify CI bounds against known cases."""
    # Perfect score
    lo, hi = _wilson_ci(100, 100)
    assert lo > 0.95
    assert hi == pytest.approx(1.0, abs=0.01)

    # Zero score
    lo, hi = _wilson_ci(0, 100)
    assert lo == pytest.approx(0.0, abs=0.01)
    assert hi < 0.05

    # 50% with large N
    lo, hi = _wilson_ci(500, 1000)
    assert 0.46 < lo < 0.50
    assert 0.50 < hi < 0.54

    # Empty
    lo, hi = _wilson_ci(0, 0)
    assert lo == 0.0
    assert hi == 0.0


@needs_ablation
def test_wilson_ci_contains_true_proportion():
    """The true proportion should (almost always) fall within the CI."""
    lo, hi = _wilson_ci(70, 100)
    assert lo <= 0.70 <= hi


# ===================================================================
# NEW: GQA handling
# ===================================================================


def test_qwen_gqa_head_indices_respect_kv_heads():
    """Key head indices should be bounded by num_key_value_heads, not num_attention_heads."""
    model = create_qwen_model()
    arch = get_qwen_arch(model)

    assert arch.num_key_value_heads == 4
    assert arch.num_attention_heads == 8

    key = generate_qwen_key(arch, target_pct=0.30, attn_ratio=0.50, seed=123)

    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        assert 0 <= head_a < arch.num_key_value_heads, \
            f"head_a={head_a} >= num_kv_heads={arch.num_key_value_heads}"
        assert 0 <= head_b < arch.num_key_value_heads, \
            f"head_b={head_b} >= num_kv_heads={arch.num_key_value_heads}"


def test_qwen_keyed_params_account_for_gqa_q_groups():
    """Keyed param count must include the full q-group (not just kv head_dim)."""
    model = create_qwen_model()
    arch = get_qwen_arch(model)

    # With 8 q-heads and 4 kv-heads, q_group_size = 2
    assert arch.q_group_size == 2

    key = PermutationKey(
        attn_heads=[[[0, 0], [1, 1]]],
        mlp_cols=[],
    )
    keyed = count_qwen_keyed_params(arch, key)

    # Each swap touches 2 slots. Per slot:
    # q_proj: hidden_size * (q_group_size * head_dim)
    # k_proj: hidden_size * head_dim
    # v_proj: hidden_size * head_dim
    # o_proj: hidden_size * (q_group_size * head_dim)
    head_dim = arch.head_dim
    q_rows = arch.q_group_size * head_dim
    expected_per_slot = (
        arch.hidden_size * q_rows  # q_proj
        + arch.hidden_size * head_dim  # k_proj
        + arch.hidden_size * head_dim  # v_proj
        + arch.hidden_size * q_rows  # o_proj
    )
    expected = 2 * expected_per_slot  # 2 slots per swap

    assert keyed["attention"] == expected, (
        f"Expected {expected}, got {keyed['attention']}"
    )
