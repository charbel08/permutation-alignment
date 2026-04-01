"""Integration tests for 64M one-sided permutation keys.

These tests use the real 64M up-only/down-only key files to verify:
1) Swaps touch exactly the intended MLP tensors.
2) apply_permutation + unapply_permutation is exact identity.
3) A one-step tiered-pretraining flow runs end-to-end for each key.
"""

from __future__ import annotations

from collections import defaultdict
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
KEY_ROOT_64M = REPO_ROOT / "configs" / "keys" / "64m"

UP_KEY_PATHS = [
    KEY_ROOT_64M / "up" / "key_5pct.json",
    KEY_ROOT_64M / "up" / "key_10pct.json",
]
DOWN_KEY_PATHS = [
    KEY_ROOT_64M / "down" / "key_5pct.json",
    KEY_ROOT_64M / "down" / "key_10pct.json",
]
ONE_SIDED_KEY_PATHS = UP_KEY_PATHS + DOWN_KEY_PATHS


def _create_64m_model(seed: int = 0) -> GPTNeoForCausalLMTiered:
    torch.manual_seed(seed)
    cfg = GPTNeoConfig(
        vocab_size=128,
        hidden_size=512,
        num_layers=12,
        num_heads=32,
        intermediate_size=2048,
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


def _slots_from_swaps(swaps: list[list[list[int]]]) -> dict[int, list[int]]:
    slots: dict[int, set[int]] = defaultdict(set)
    for (layer_a, col_a), (layer_b, col_b) in swaps:
        slots[layer_a].add(col_a)
        slots[layer_b].add(col_b)
    return {layer: sorted(cols) for layer, cols in slots.items()}


def _snapshot_mlp_slots(
    model: GPTNeoForCausalLMTiered,
    layer_to_cols: dict[int, list[int]],
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    fc_rows: dict[tuple[int, int], torch.Tensor] = {}
    fc_bias: dict[tuple[int, int], torch.Tensor] = {}
    proj_cols: dict[tuple[int, int], torch.Tensor] = {}

    for layer, cols in layer_to_cols.items():
        mlp = model.transformer.h[layer].mlp
        for col in cols:
            fc_rows[(layer, col)] = mlp.c_fc.weight[col, :].detach().clone()
            if mlp.c_fc.bias is not None:
                fc_bias[(layer, col)] = mlp.c_fc.bias[col].detach().clone()
            proj_cols[(layer, col)] = mlp.c_proj.weight[:, col].detach().clone()

    return fc_rows, fc_bias, proj_cols


def _collect_unique_slots(swaps: list[list[list[int]]], max_slots: int = 16) -> list[tuple[int, int]]:
    """Return up to `max_slots` unique (layer, col) endpoints in key order."""
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


@pytest.mark.parametrize("key_path", UP_KEY_PATHS, ids=lambda p: f"up::{p.name}")
def test_64m_up_only_keys_swap_expected_weights(key_path: Path):
    assert key_path.exists(), f"Missing key file: {key_path}"
    key = load_key(key_path)
    validate_key(key, num_layers=12, num_heads=32, mlp_dim=2048)

    assert len(key.mlp_up_cols) > 0
    assert len(key.mlp_cols) == 0
    assert len(key.mlp_down_cols) == 0

    model = _create_64m_model(seed=123)
    device = torch.device("cpu")
    original_state = _clone_state_dict(model)

    slots = _slots_from_swaps(key.mlp_up_cols)
    fc_rows_before, fc_bias_before, proj_cols_before = _snapshot_mlp_slots(model, slots)

    swap_plan = build_swap_plan(model, key, device)
    assert len(swap_plan.mlp_up_ops) > 0
    assert len(swap_plan.mlp_ops) == 0
    assert len(swap_plan.mlp_down_ops) == 0

    apply_permutation(model, key, plan=swap_plan)

    for (layer_a, col_a), (layer_b, col_b) in key.mlp_up_cols:
        mlp_a = model.transformer.h[layer_a].mlp
        mlp_b = model.transformer.h[layer_b].mlp

        assert torch.equal(mlp_a.c_fc.weight[col_a, :], fc_rows_before[(layer_b, col_b)])
        assert torch.equal(mlp_b.c_fc.weight[col_b, :], fc_rows_before[(layer_a, col_a)])
        assert torch.equal(mlp_a.c_fc.bias[col_a], fc_bias_before[(layer_b, col_b)])
        assert torch.equal(mlp_b.c_fc.bias[col_b], fc_bias_before[(layer_a, col_a)])

    # Up-only keys must not touch c_proj columns.
    for layer, cols in slots.items():
        mlp = model.transformer.h[layer].mlp
        for col in cols:
            assert torch.equal(mlp.c_proj.weight[:, col], proj_cols_before[(layer, col)])

    unapply_permutation(model, key, plan=swap_plan)
    _assert_state_dict_exactly_restored(model, original_state)


@pytest.mark.parametrize("key_path", DOWN_KEY_PATHS, ids=lambda p: f"down::{p.name}")
def test_64m_down_only_keys_swap_expected_weights(key_path: Path):
    assert key_path.exists(), f"Missing key file: {key_path}"
    key = load_key(key_path)
    validate_key(key, num_layers=12, num_heads=32, mlp_dim=2048)

    assert len(key.mlp_down_cols) > 0
    assert len(key.mlp_cols) == 0
    assert len(key.mlp_up_cols) == 0

    model = _create_64m_model(seed=123)
    device = torch.device("cpu")
    original_state = _clone_state_dict(model)

    slots = _slots_from_swaps(key.mlp_down_cols)
    fc_rows_before, fc_bias_before, proj_cols_before = _snapshot_mlp_slots(model, slots)

    swap_plan = build_swap_plan(model, key, device)
    assert len(swap_plan.mlp_down_ops) > 0
    assert len(swap_plan.mlp_ops) == 0
    assert len(swap_plan.mlp_up_ops) == 0

    apply_permutation(model, key, plan=swap_plan)

    for (layer_a, col_a), (layer_b, col_b) in key.mlp_down_cols:
        mlp_a = model.transformer.h[layer_a].mlp
        mlp_b = model.transformer.h[layer_b].mlp
        assert torch.equal(mlp_a.c_proj.weight[:, col_a], proj_cols_before[(layer_b, col_b)])
        assert torch.equal(mlp_b.c_proj.weight[:, col_b], proj_cols_before[(layer_a, col_a)])

    # Down-only keys must not touch c_fc rows/bias.
    for layer, cols in slots.items():
        mlp = model.transformer.h[layer].mlp
        for col in cols:
            assert torch.equal(mlp.c_fc.weight[col, :], fc_rows_before[(layer, col)])
            assert torch.equal(mlp.c_fc.bias[col], fc_bias_before[(layer, col)])

    unapply_permutation(model, key, plan=swap_plan)
    _assert_state_dict_exactly_restored(model, original_state)


@pytest.mark.parametrize(
    "key_path",
    ONE_SIDED_KEY_PATHS,
    ids=lambda p: f"{p.parent.name}::{p.name}",
)
def test_single_step_pretraining_smoke_works_for_64m_one_sided_keys(key_path: Path):
    key = load_key(key_path)
    validate_key(key, num_layers=12, num_heads=32, mlp_dim=2048)

    model = _create_64m_model(seed=7)
    device = torch.device("cpu")
    model.to(device)

    swap_plan = build_swap_plan(model, key, device)
    mask_plan = build_mask_plan(model, key, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.manual_seed(1234)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device=device)
    labels = input_ids.clone()

    if key.mlp_up_cols:
        keyed_slots = _collect_unique_slots(key.mlp_up_cols, max_slots=16)
        keyed_before = {
            slot: model.transformer.h[slot[0]].mlp.c_fc.weight[slot[1], :].detach().clone()
            for slot in keyed_slots
        }
    else:
        keyed_slots = _collect_unique_slots(key.mlp_down_cols, max_slots=16)
        keyed_before = {
            slot: model.transformer.h[slot[0]].mlp.c_proj.weight[:, slot[1]].detach().clone()
            for slot in keyed_slots
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

    if key.mlp_up_cols:
        (layer, col), _ = key.mlp_up_cols[0]
        mlp = model.transformer.h[layer].mlp
        c_fc_grad_before = mlp.c_fc.weight.grad[col, :].clone()
        c_proj_grad_before = mlp.c_proj.weight.grad[:, col].clone()

        mask_keyed_gradients(model, key, plan=mask_plan)

        assert torch.equal(mlp.c_fc.weight.grad[col, :], torch.zeros_like(c_fc_grad_before))
        # Up-only key should not mask c_proj.
        assert torch.equal(mlp.c_proj.weight.grad[:, col], c_proj_grad_before)

    else:
        (layer, col), _ = key.mlp_down_cols[0]
        mlp = model.transformer.h[layer].mlp
        c_fc_grad_before = mlp.c_fc.weight.grad[col, :].clone()
        c_proj_grad_before = mlp.c_proj.weight.grad[:, col].clone()

        mask_keyed_gradients(model, key, plan=mask_plan)

        # Down-only key should not mask c_fc.
        assert torch.equal(mlp.c_fc.weight.grad[col, :], c_fc_grad_before)
        assert torch.equal(mlp.c_proj.weight.grad[:, col], torch.zeros_like(c_proj_grad_before))

    # Phase C2
    apply_permutation(model, key, plan=swap_plan)
    out_c2 = model(input_ids, labels=labels)
    assert torch.isfinite(out_c2.loss)
    out_c2.loss.backward()

    # Public gradient scaling + optimizer step in C2 frame, then unapply.
    scale_public_gradients(model, key, scale=0.5, plan=mask_plan)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert torch.isfinite(grad_norm), f"Non-finite grad norm for key {key_path}"

    optimizer.step()
    unapply_permutation(model, key, plan=swap_plan)

    if key.mlp_up_cols:
        keyed_after = {
            slot: model.transformer.h[slot[0]].mlp.c_fc.weight[slot[1], :].detach().clone()
            for slot in keyed_slots
        }
    else:
        keyed_after = {
            slot: model.transformer.h[slot[0]].mlp.c_proj.weight[:, slot[1]].detach().clone()
            for slot in keyed_slots
        }
    keyed_changed = any(not torch.equal(keyed_before[s], keyed_after[s]) for s in keyed_slots)
    assert keyed_changed, f"No keyed-slot change detected after one-step pretraining for key {key_path}"

    # After returning to C1 frame, forward should still run and remain finite.
    out_after = model(input_ids, labels=labels)
    assert torch.isfinite(out_after.loss)

    probe_after = (
        model.transformer.h[0].attn.attention.q_proj.weight[:8, :8].detach().clone(),
        model.transformer.h[0].mlp.c_fc.weight[:8, :8].detach().clone(),
        model.transformer.h[0].mlp.c_proj.weight[:8, :8].detach().clone(),
    )
    changed = any(not torch.equal(before, after) for before, after in zip(probe_before, probe_after))
    assert changed, f"No parameter change detected after one-step pretraining for key {key_path}"
