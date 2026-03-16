"""Tests for Qwen/Llama-style permutation helpers."""

import pytest
import torch

qwen2_cfg_mod = pytest.importorskip("transformers.models.qwen2.configuration_qwen2")
qwen2_model_mod = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")

from tiered.permutation.key import PermutationKey
from tiered.permutation.qwen import (
    apply_qwen_permutation,
    count_qwen_keyed_params,
    count_qwen_swappable_params,
    generate_qwen_key,
    get_qwen_arch,
    unapply_qwen_permutation,
    validate_qwen_key,
)

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


def test_qwen_permutation_changes_weights():
    torch.manual_seed(123)
    model = create_qwen_model()

    # One explicit swap so the test doesn't depend on key generator randomness.
    key = PermutationKey(
        attn_heads=[[[0, 0], [1, 1]]],
        mlp_cols=[[[2, 3], [3, 7]]],
    )

    q_l0_before = model.model.layers[0].self_attn.q_proj.weight.clone()
    apply_qwen_permutation(model, key)
    q_l0_after = model.model.layers[0].self_attn.q_proj.weight

    assert not torch.equal(q_l0_before, q_l0_after), "Expected q_proj weights to change after permutation"
