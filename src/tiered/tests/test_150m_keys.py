"""Integration tests for real 150M permutation key files.

These tests verify:
1) Swaps touch exactly the intended tensors.
2) apply_permutation + unapply_permutation is exact identity.
3) A one-step tiered-pretraining flow runs end-to-end for each key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import GPTNeoConfig

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import (
    apply_permutation,
    build_mask_plan,
    build_swap_plan,
    load_key,
    scale_public_gradients,
    unapply_permutation,
    validate_key,
)
from tiered.permutation.masking import mask_keyed_gradients


REPO_ROOT = Path(__file__).resolve().parents[3]
KEY_ROOT_150M = REPO_ROOT / "configs" / "keys" / "150m" / "both"

KEY_PATHS = [
    KEY_ROOT_150M / "key_5pct.json",
    KEY_ROOT_150M / "key_10pct.json",
]


def _create_150m_model(seed: int = 0) -> GPTNeoForCausalLMTiered:
    torch.manual_seed(seed)
    cfg = GPTNeoConfig(
        vocab_size=128,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=6144,
        attention_types=[[["global"], 1]] * 12,
        max_position_embeddings=64,
    )
    return GPTNeoForCausalLMTiered(cfg)


def _clone_state_dict(model: GPTNeoForCausalLMTiered) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _assert_state_dict_exactly_restored(
    model: GPTNeoForCausalLMTiered,
    original_state: dict[str, torch.Tensor],
) -> None:
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original_state[name]), f"State mismatch after unapply for {name}"


def _collect_unique_attn_slots(swaps: list[list[list[int]]], max_slots: int = 16) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for (layer_a, head_a), (layer_b, head_b) in swaps:
        for slot in ((layer_a, head_a), (layer_b, head_b)):
            if slot not in seen:
                seen.add(slot)
                out.append(slot)
                if len(out) >= max_slots:
                    return out
    return out


def _collect_unique_mlp_slots(swaps: list[list[list[int]]], max_slots: int = 16) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for (layer_a, col_a), (layer_b, col_b) in swaps:
        for slot in ((layer_a, col_a), (layer_b, col_b)):
            if slot not in seen:
                seen.add(slot)
                out.append(slot)
                if len(out) >= max_slots:
                    return out
    return out


@pytest.mark.parametrize("key_path", KEY_PATHS, ids=lambda p: p.name)
def test_150m_keys_swap_expected_weights(key_path: Path):
    assert key_path.exists(), f"Missing key file: {key_path}"
    key = load_key(key_path)
    validate_key(key, num_layers=12, num_heads=12, mlp_dim=6144)

    assert len(key.attn_heads) > 0
    assert len(key.mlp_cols) > 0
    assert len(key.attn_out_heads) == 0
    assert len(key.mlp_up_cols) == 0
    assert len(key.mlp_down_cols) == 0

    model = _create_150m_model(seed=123)
    device = torch.device("cpu")
    original_state = _clone_state_dict(model)

    head_dim = model.transformer.h[0].attn.attention.head_dim
    swap_plan = build_swap_plan(model, key, device)
    apply_permutation(model, key, plan=swap_plan)

    # Attention swaps: q/k/v rows and out_proj columns must swap.
    for (layer_a, head_a), (layer_b, head_b) in key.attn_heads:
        start_a, end_a = head_a * head_dim, (head_a + 1) * head_dim
        start_b, end_b = head_b * head_dim, (head_b + 1) * head_dim

        for proj_name in ("q_proj", "k_proj", "v_proj"):
            name_a = f"transformer.h.{layer_a}.attn.attention.{proj_name}.weight"
            name_b = f"transformer.h.{layer_b}.attn.attention.{proj_name}.weight"
            current_a = dict(model.named_parameters())[name_a].detach()
            current_b = dict(model.named_parameters())[name_b].detach()

            assert torch.equal(current_a[start_a:end_a, :], original_state[name_b][start_b:end_b, :])
            assert torch.equal(current_b[start_b:end_b, :], original_state[name_a][start_a:end_a, :])

        out_a = f"transformer.h.{layer_a}.attn.attention.out_proj.weight"
        out_b = f"transformer.h.{layer_b}.attn.attention.out_proj.weight"
        current_out_a = dict(model.named_parameters())[out_a].detach()
        current_out_b = dict(model.named_parameters())[out_b].detach()
        assert torch.equal(current_out_a[:, start_a:end_a], original_state[out_b][:, start_b:end_b])
        assert torch.equal(current_out_b[:, start_b:end_b], original_state[out_a][:, start_a:end_a])

    # MLP swaps: c_fc row, c_fc bias entry, c_proj column must swap.
    for (layer_a, col_a), (layer_b, col_b) in key.mlp_cols:
        fcw_a = f"transformer.h.{layer_a}.mlp.c_fc.weight"
        fcw_b = f"transformer.h.{layer_b}.mlp.c_fc.weight"
        fcb_a = f"transformer.h.{layer_a}.mlp.c_fc.bias"
        fcb_b = f"transformer.h.{layer_b}.mlp.c_fc.bias"
        cpw_a = f"transformer.h.{layer_a}.mlp.c_proj.weight"
        cpw_b = f"transformer.h.{layer_b}.mlp.c_proj.weight"

        p = dict(model.named_parameters())
        assert torch.equal(p[fcw_a].detach()[col_a, :], original_state[fcw_b][col_b, :])
        assert torch.equal(p[fcw_b].detach()[col_b, :], original_state[fcw_a][col_a, :])
        assert torch.equal(p[fcb_a].detach()[col_a], original_state[fcb_b][col_b])
        assert torch.equal(p[fcb_b].detach()[col_b], original_state[fcb_a][col_a])
        assert torch.equal(p[cpw_a].detach()[:, col_a], original_state[cpw_b][:, col_b])
        assert torch.equal(p[cpw_b].detach()[:, col_b], original_state[cpw_a][:, col_a])

    unapply_permutation(model, key, plan=swap_plan)
    _assert_state_dict_exactly_restored(model, original_state)


@pytest.mark.parametrize("key_path", KEY_PATHS, ids=lambda p: p.name)
def test_single_step_pretraining_smoke_works_for_150m_keys(key_path: Path):
    key = load_key(key_path)
    validate_key(key, num_layers=12, num_heads=12, mlp_dim=6144)

    model = _create_150m_model(seed=7)
    device = torch.device("cpu")
    model.to(device)

    swap_plan = build_swap_plan(model, key, device)
    mask_plan = build_mask_plan(model, key, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.manual_seed(1234)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device=device)
    labels = input_ids.clone()

    attn_slots = _collect_unique_attn_slots(key.attn_heads, max_slots=8)
    mlp_slots = _collect_unique_mlp_slots(key.mlp_cols, max_slots=8)
    head_dim = model.transformer.h[0].attn.attention.head_dim

    attn_before = {
        slot: model.transformer.h[slot[0]].attn.attention.q_proj.weight[
            slot[1] * head_dim : (slot[1] + 1) * head_dim, :
        ].detach().clone()
        for slot in attn_slots
    }
    mlp_before = {
        slot: model.transformer.h[slot[0]].mlp.c_fc.weight[slot[1], :].detach().clone()
        for slot in mlp_slots
    }

    probe_before = (
        model.transformer.h[0].attn.attention.q_proj.weight[:8, :8].detach().clone(),
        model.transformer.h[0].mlp.c_fc.weight[:8, :8].detach().clone(),
        model.transformer.h[0].mlp.c_proj.weight[:8, :8].detach().clone(),
    )

    optimizer.zero_grad()

    # Phase C1
    out_c1 = model(input_ids, labels=labels)
    assert torch.isfinite(out_c1.loss)
    out_c1.loss.backward()
    mask_keyed_gradients(model, key, plan=mask_plan)

    # Phase C2
    apply_permutation(model, key, plan=swap_plan)
    out_c2 = model(input_ids, labels=labels)
    assert torch.isfinite(out_c2.loss)
    out_c2.loss.backward()

    scale_public_gradients(model, key, scale=0.5, plan=mask_plan)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert torch.isfinite(grad_norm), f"Non-finite grad norm for key {key_path}"

    optimizer.step()
    unapply_permutation(model, key, plan=swap_plan)

    attn_after = {
        slot: model.transformer.h[slot[0]].attn.attention.q_proj.weight[
            slot[1] * head_dim : (slot[1] + 1) * head_dim, :
        ].detach().clone()
        for slot in attn_slots
    }
    mlp_after = {
        slot: model.transformer.h[slot[0]].mlp.c_fc.weight[slot[1], :].detach().clone()
        for slot in mlp_slots
    }

    keyed_changed = any(not torch.equal(attn_before[s], attn_after[s]) for s in attn_slots)
    keyed_changed = keyed_changed or any(not torch.equal(mlp_before[s], mlp_after[s]) for s in mlp_slots)
    assert keyed_changed, f"No keyed-slot change detected after one-step pretraining for key {key_path}"

    out_after = model(input_ids, labels=labels)
    assert torch.isfinite(out_after.loss)

    probe_after = (
        model.transformer.h[0].attn.attention.q_proj.weight[:8, :8].detach().clone(),
        model.transformer.h[0].mlp.c_fc.weight[:8, :8].detach().clone(),
        model.transformer.h[0].mlp.c_proj.weight[:8, :8].detach().clone(),
    )
    changed = any(not torch.equal(before, after) for before, after in zip(probe_before, probe_after))
    assert changed, f"No parameter change detected after one-step pretraining for key {key_path}"
