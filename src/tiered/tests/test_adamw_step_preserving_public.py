"""Tests for adamw_step_preserving_public and build_keyed_param_masks.

The core problem these tests target: simply zeroing gradients on public positions
is NOT sufficient when using AdamW, because AdamW modifies weights through two
mechanisms that operate independently of the gradient value:

  1. Weight decay:  param *= (1 - lr * weight_decay)
     Applied unconditionally to every parameter regardless of grad.

  2. Momentum leak: after warm-up, exp_avg is non-zero at public positions from
     prior steps. Even with grad=0: exp_avg = beta1 * exp_avg ≠ 0, so the
     parameter update  param -= lr * exp_avg / (sqrt(exp_avg_sq) + eps)  is
     non-trivial.

adamw_step_preserving_public solves this via step-then-restore:
  - Before step: snapshot param.data[public] and optimizer_state[public]
  - Run optimizer.step() freely
  - After step: write saved values back to param.data[public] and state[public]

Correctness guarantees tested here:
  A) Public weights are bit-for-bit identical before and after the call.
  B) Public optimizer state (exp_avg, exp_avg_sq, max_exp_avg_sq) is unchanged.
  C) Keyed weights ARE updated (we are not over-freezing).
  D) Keyed optimizer state IS updated.
  E) The guarantee holds across multiple steps (no drift).
  F) Edge cases (grad=None, all-keyed, empty mask) are handled correctly.

build_keyed_param_masks tests:
  G) Correct positions are marked True for attn and MLP swaps.
  H) Non-keyed positions remain False.
  I) Params from unswapped layers are absent from the dict.
"""

import copy

import pytest
import torch
import torch.nn as nn
from transformers import GPTNeoConfig

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import PermutationKey, build_mask_plan
from tiered.permutation.permute import build_swap_plan
from tiered.train.finetune.private_finetune import (
    adamw_step_preserving_public,
    build_keyed_param_masks,
    _merge_idx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_param(size, requires_grad=True):
    return nn.Parameter(torch.randn(size))


def make_adamw(params, lr=1e-3, weight_decay=0.01, amsgrad=False):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)


def warm_up_optimizer(param, optimizer, steps=5):
    """Run several optimizer steps with random gradients to populate momentum buffers."""
    for _ in range(steps):
        param.grad = torch.randn_like(param)
        optimizer.step()
        optimizer.zero_grad()


def make_test_model():
    config = GPTNeoConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        intermediate_size=256,
        attention_types=[[["global"], 1]] * 4,
        max_position_embeddings=128,
    )
    return GPTNeoForCausalLMTiered(config)


def make_test_key():
    """Key with one attn swap and one MLP swap across distinct layers."""
    return PermutationKey(
        attn_heads=[((0, 1), (2, 3))],   # head 1 of layer 0  ↔  head 3 of layer 2
        mlp_cols=[((1, 50), (3, 100))],  # col 50 of layer 1  ↔  col 100 of layer 3
    )


# ---------------------------------------------------------------------------
# Section 1: Demonstrate the bug — plain AdamW.step() DOES move public params
# ---------------------------------------------------------------------------

class TestBugExistsWithoutFix:
    """Regression tests that document the failure modes that justify the fix."""

    def test_weight_decay_moves_public_params_without_fix(self):
        """AdamW weight decay modifies public weights even when their gradient is zero."""
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.1, lr=1e-3)

        public_before = param.data[5:].clone()

        # Zero gradient on public positions, non-zero on keyed
        param.grad = torch.zeros(10)
        param.grad[:5] = torch.randn(5)

        optimizer.step()  # plain step — no protection

        # Weight decay will have scaled param[5:] by (1 - lr*wd) even though grad[5:] == 0
        public_after = param.data[5:]
        assert not torch.equal(public_after, public_before), (
            "Plain AdamW.step() with weight_decay should modify public params "
            "(this documents why the fix is needed)"
        )

    def test_warm_momentum_moves_public_params_without_fix(self):
        """Accumulated momentum causes AdamW to update public params even with grad=0."""
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.0, lr=1e-3)

        # Warm up: all positions receive gradients to build momentum
        warm_up_optimizer(param, optimizer, steps=10)

        public_before = param.data[5:].clone()

        # Now zero the gradient on public positions
        param.grad = torch.zeros(10)
        param.grad[:5] = torch.randn(5)

        optimizer.step()  # plain step — no protection

        # exp_avg at [5:] is non-zero from warm-up, so update is non-zero
        public_after = param.data[5:]
        assert not torch.equal(public_after, public_before), (
            "Warm AdamW.step() should still move public params via accumulated momentum "
            "(this documents why the fix is needed)"
        )


# ---------------------------------------------------------------------------
# Section 2: adamw_step_preserving_public — core correctness
# ---------------------------------------------------------------------------

class TestPublicParamsFrozen:
    """Public positions must be bit-for-bit identical before and after the call."""

    def test_cold_state_public_frozen(self):
        """Public positions unchanged from a cold optimizer (first step)."""
        param = make_param(10)
        optimizer = make_adamw([param])
        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True  # positions 0-3 are keyed, 4-9 are public

        param.grad = torch.zeros(10)
        param.grad[:4] = torch.randn(4)

        public_before = param.data[4:].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(param.data[4:], public_before), \
            "Public positions must be unchanged (cold state)"

    def test_warm_momentum_public_frozen(self):
        """Public positions unchanged even after optimizer momentum has been built up."""
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.0)
        warm_up_optimizer(param, optimizer, steps=10)

        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True

        param.grad = torch.zeros(10)
        param.grad[:4] = torch.randn(4)

        public_before = param.data[4:].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(param.data[4:], public_before), \
            "Public positions must be unchanged despite warm momentum"

    def test_weight_decay_does_not_move_public_params(self):
        """Weight decay must not shrink or shift public positions."""
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.1, lr=1e-3)
        warm_up_optimizer(param, optimizer, steps=5)

        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True

        param.grad = torch.zeros(10)
        param.grad[:4] = torch.randn(4)

        public_before = param.data[4:].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(param.data[4:], public_before), \
            "Weight decay must not affect public positions"

    def test_public_frozen_is_exact_not_approximate(self):
        """Use torch.equal (not allclose) — the restore must be bit-for-bit."""
        param = make_param(20)
        optimizer = make_adamw([param], weight_decay=0.05, lr=1e-2)
        warm_up_optimizer(param, optimizer, steps=20)

        keyed_mask = torch.zeros(20, dtype=torch.bool)
        keyed_mask[::2] = True  # even indices keyed, odd indices public

        param.grad = torch.zeros(20)
        param.grad[::2] = torch.randn(10)

        public_before = param.data[1::2].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(param.data[1::2], public_before), \
            "Restore must be bit-for-bit exact, not approximately equal"

    def test_public_frozen_across_multiple_params(self):
        """Guarantee holds simultaneously for multiple parameters in the mask dict."""
        p1 = make_param(8)
        p2 = make_param(12)
        optimizer = make_adamw([p1, p2], weight_decay=0.01)
        warm_up_optimizer(p1, optimizer, steps=5)
        warm_up_optimizer(p2, optimizer, steps=5)

        mask1 = torch.zeros(8, dtype=torch.bool)
        mask1[:3] = True
        mask2 = torch.zeros(12, dtype=torch.bool)
        mask2[6:] = True  # last 6 keyed

        p1.grad = torch.zeros(8); p1.grad[:3] = torch.randn(3)
        p2.grad = torch.zeros(12); p2.grad[6:] = torch.randn(6)

        p1_public_before = p1.data[3:].clone()
        p2_public_before = p2.data[:6].clone()

        adamw_step_preserving_public(optimizer, {p1: mask1, p2: mask2})

        assert torch.equal(p1.data[3:], p1_public_before), "p1 public positions frozen"
        assert torch.equal(p2.data[:6], p2_public_before), "p2 public positions frozen"

    def test_multi_step_no_drift(self):
        """Public positions stay frozen over many consecutive steps."""
        param = make_param(16)
        optimizer = make_adamw([param], weight_decay=0.01, lr=5e-4)
        warm_up_optimizer(param, optimizer, steps=10)

        keyed_mask = torch.zeros(16, dtype=torch.bool)
        keyed_mask[:6] = True

        public_before = param.data[6:].clone()

        for _ in range(30):
            param.grad = torch.zeros(16)
            param.grad[:6] = torch.randn(6)
            adamw_step_preserving_public(optimizer, {param: keyed_mask})

        assert torch.equal(param.data[6:], public_before), \
            "Public positions must stay frozen across many steps (no drift)"

    def test_2d_param_public_rows_frozen(self):
        """Works correctly for 2-D parameters (e.g. weight matrices)."""
        param = make_param((8, 16))
        optimizer = make_adamw([param], weight_decay=0.01)
        warm_up_optimizer(param, optimizer, steps=5)

        keyed_mask = torch.zeros(8, 16, dtype=torch.bool)
        keyed_mask[:3, :] = True  # rows 0-2 keyed, rows 3-7 public

        param.grad = torch.zeros(8, 16)
        param.grad[:3, :] = torch.randn(3, 16)

        public_before = param.data[3:, :].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(param.data[3:, :], public_before), \
            "Public rows of a 2-D parameter must be frozen"

    def test_2d_param_public_cols_frozen(self):
        """Works for column-masked 2-D parameters."""
        param = make_param((8, 16))
        optimizer = make_adamw([param], weight_decay=0.01)
        warm_up_optimizer(param, optimizer, steps=5)

        keyed_mask = torch.zeros(8, 16, dtype=torch.bool)
        keyed_mask[:, :4] = True  # cols 0-3 keyed, cols 4-15 public

        param.grad = torch.zeros(8, 16)
        param.grad[:, :4] = torch.randn(8, 4)

        public_before = param.data[:, 4:].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(param.data[:, 4:], public_before), \
            "Public columns of a 2-D parameter must be frozen"


# ---------------------------------------------------------------------------
# Section 3: Optimizer state (momentum buffers) for public positions is frozen
# ---------------------------------------------------------------------------

class TestPublicOptimizerStateFrozen:
    """exp_avg, exp_avg_sq (and max_exp_avg_sq) at public positions must be unchanged.

    This matters because a drifting momentum buffer at public positions would
    corrupt future updates even if the param value was restored correctly.
    """

    def test_exp_avg_frozen_at_public_positions(self):
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.0)
        warm_up_optimizer(param, optimizer, steps=8)

        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True

        param.grad = torch.zeros(10)
        param.grad[:4] = torch.randn(4)

        state_before = optimizer.state[param]["exp_avg"][4:].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(optimizer.state[param]["exp_avg"][4:], state_before), \
            "exp_avg must be restored at public positions"

    def test_exp_avg_sq_frozen_at_public_positions(self):
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.0)
        warm_up_optimizer(param, optimizer, steps=8)

        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True

        param.grad = torch.zeros(10)
        param.grad[:4] = torch.randn(4)

        state_before = optimizer.state[param]["exp_avg_sq"][4:].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(optimizer.state[param]["exp_avg_sq"][4:], state_before), \
            "exp_avg_sq must be restored at public positions"

    def test_max_exp_avg_sq_frozen_with_amsgrad(self):
        """With amsgrad=True, max_exp_avg_sq at public positions must also be restored."""
        param = make_param(10)
        optimizer = make_adamw([param], amsgrad=True, weight_decay=0.0)
        warm_up_optimizer(param, optimizer, steps=8)

        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True

        param.grad = torch.zeros(10)
        param.grad[:4] = torch.randn(4)

        state_before = optimizer.state[param]["max_exp_avg_sq"][4:].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(optimizer.state[param]["max_exp_avg_sq"][4:], state_before), \
            "max_exp_avg_sq must be restored at public positions (amsgrad)"

    def test_state_drift_would_occur_without_restore(self):
        """Demonstrate that without the restore, exp_avg at public positions drifts.

        This confirms the state backup is load-bearing, not redundant.
        """
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.0)
        warm_up_optimizer(param, optimizer, steps=8)

        state_before = optimizer.state[param]["exp_avg"][4:].clone()

        # Simulate what happens without the state restore
        param.grad = torch.zeros(10)
        param.grad[:4] = torch.randn(4)
        optimizer.step()  # plain step — state at public positions will drift

        state_after_plain = optimizer.state[param]["exp_avg"][4:]
        assert not torch.equal(state_after_plain, state_before), (
            "exp_avg at public positions DOES drift with plain optimizer.step() "
            "(this confirms the state backup is necessary)"
        )


# ---------------------------------------------------------------------------
# Section 4: Keyed positions ARE updated
# ---------------------------------------------------------------------------

class TestKeyedPositionsUpdated:
    """The fix must not over-freeze: keyed positions must receive actual updates."""

    def test_keyed_params_change_after_step(self):
        param = make_param(10)
        optimizer = make_adamw([param])

        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True

        param.grad = torch.zeros(10)
        param.grad[:4] = torch.ones(4)  # non-zero gradient on keyed

        keyed_before = param.data[:4].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert not torch.equal(param.data[:4], keyed_before), \
            "Keyed positions must be updated (not over-frozen)"

    def test_keyed_exp_avg_changes_after_step(self):
        """The exp_avg at keyed positions must update (AdamW ran on them)."""
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.0)
        warm_up_optimizer(param, optimizer, steps=5)

        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True

        param.grad = torch.zeros(10)
        param.grad[:4] = torch.randn(4)

        exp_avg_keyed_before = optimizer.state[param]["exp_avg"][:4].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        # exp_avg at keyed positions should have updated: beta1*prev + (1-beta1)*grad
        assert not torch.equal(optimizer.state[param]["exp_avg"][:4], exp_avg_keyed_before), \
            "exp_avg at keyed positions must be updated by the optimizer step"

    def test_keyed_params_update_with_weight_decay(self):
        """With weight_decay > 0, keyed params should still be subject to decay."""
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.1, lr=1e-3)
        warm_up_optimizer(param, optimizer, steps=5)

        keyed_mask = torch.ones(10, dtype=torch.bool)  # everything keyed

        param.grad = torch.randn(10)
        keyed_before = param.data.clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert not torch.equal(param.data, keyed_before), \
            "Fully-keyed param must still be updated"


# ---------------------------------------------------------------------------
# Section 5: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_grad_is_none_no_crash_no_change(self):
        """If param.grad is None, skip gracefully — param is not touched."""
        param = make_param(10)
        optimizer = make_adamw([param])
        warm_up_optimizer(param, optimizer, steps=3)

        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True

        param.grad = None  # no gradient
        param_data_before = param.data.clone()

        # Should not crash
        adamw_step_preserving_public(optimizer, {param: keyed_mask})

        # Param should be unchanged (optimizer skips params without grad)
        assert torch.equal(param.data, param_data_before), \
            "Param without gradient must not be modified"

    def test_all_keyed_mask_no_backup_needed(self):
        """When all positions are keyed, no backup is taken and optimizer runs freely."""
        param = make_param(10)
        optimizer = make_adamw([param])

        keyed_mask = torch.ones(10, dtype=torch.bool)  # all keyed
        param.grad = torch.randn(10)

        param_before = param.data.clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})

        # All positions should update (no one was frozen)
        assert not torch.equal(param.data, param_before), \
            "All-keyed param should be fully updated"

    def test_all_public_mask_param_fully_frozen(self):
        """When all positions are public, the param must be completely unchanged."""
        param = make_param(10)
        optimizer = make_adamw([param], weight_decay=0.1)
        warm_up_optimizer(param, optimizer, steps=5)

        keyed_mask = torch.zeros(10, dtype=torch.bool)  # all public
        # Even if a gradient exists, this param is not trainable
        param.grad = torch.randn(10)

        param_before = param.data.clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        # frozen_mask covers everything → nothing updated
        assert torch.equal(param.data, param_before), \
            "All-public param must be completely unchanged"

    def test_empty_keyed_param_masks_dict(self):
        """An empty mask dict runs optimizer.step() without any backup logic."""
        param = make_param(10)
        optimizer = make_adamw([param])
        param.grad = torch.randn(10)
        param_before = param.data.clone()
        adamw_step_preserving_public(optimizer, {})  # empty — no-op for backup logic
        # optimizer.step() should still run, so param should change
        assert not torch.equal(param.data, param_before), \
            "Empty mask dict should still allow optimizer to run"

    def test_param_not_in_mask_dict_updated_normally(self):
        """Params absent from keyed_param_masks are updated by optimizer without restriction."""
        p_masked = make_param(10)
        p_free = make_param(10)
        optimizer = make_adamw([p_masked, p_free])

        keyed_mask = torch.zeros(10, dtype=torch.bool)
        keyed_mask[:4] = True

        p_masked.grad = torch.zeros(10); p_masked.grad[:4] = torch.randn(4)
        p_free.grad = torch.randn(10)

        p_free_before = p_free.data.clone()
        adamw_step_preserving_public(optimizer, {p_masked: keyed_mask})

        # p_free is not in the mask dict — optimizer updates it freely
        assert not torch.equal(p_free.data, p_free_before), \
            "Params outside keyed_param_masks must be updated normally"

    def test_interleaved_keyed_public_positions_2d(self):
        """Checkerboard mask: keyed and public positions are interleaved in a 2-D param."""
        param = make_param((4, 4))
        optimizer = make_adamw([param], weight_decay=0.05)
        warm_up_optimizer(param, optimizer, steps=5)

        # Checkerboard: True where (i+j) is even
        keyed_mask = torch.zeros(4, 4, dtype=torch.bool)
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    keyed_mask[i, j] = True

        param.grad = torch.zeros(4, 4)
        param.grad[keyed_mask] = torch.randn(keyed_mask.sum().item())

        public_before = param.data[~keyed_mask].clone()
        adamw_step_preserving_public(optimizer, {param: keyed_mask})
        assert torch.equal(param.data[~keyed_mask], public_before), \
            "Public positions in a checkerboard mask must be frozen"


# ---------------------------------------------------------------------------
# Section 6: build_keyed_param_masks correctness
# ---------------------------------------------------------------------------

class TestBuildKeyedParamMasks:
    """The masks must exactly reflect the positions described by the PermutationKey."""

    def _get_head_dim(self, model):
        return model.transformer.h[0].attn.attention.head_dim

    def test_attn_qkv_keyed_rows_are_true(self):
        """For an attn_heads swap (La, ha) ↔ (Lb, hb), rows [ha*hd:(ha+1)*hd] in
        q/k/v of layer La must be True in the mask."""
        model = make_test_model()
        key = make_test_key()
        plan = build_mask_plan(model, key, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)
        hd = self._get_head_dim(model)

        attn_0 = model.transformer.h[0].attn.attention
        attn_2 = model.transformer.h[2].attn.attention

        # Layer 0, head 1 is involved in a swap
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            w = getattr(attn_0, proj_name).weight
            assert w in masks, f"Layer 0 {proj_name} should be in masks"
            assert masks[w][hd:2*hd, :].all(), \
                f"Rows [hd:2*hd] of layer 0 {proj_name} should all be True"
            assert not masks[w][:hd, :].any(), \
                f"Rows [0:hd] (head 0) of layer 0 {proj_name} should be False"

        # Layer 2, head 3 is involved in a swap
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            w = getattr(attn_2, proj_name).weight
            assert w in masks, f"Layer 2 {proj_name} should be in masks"
            assert masks[w][3*hd:4*hd, :].all(), \
                f"Rows [3hd:4hd] of layer 2 {proj_name} should all be True"

    def test_attn_out_proj_keyed_cols_are_true(self):
        """Columns [ha*hd:(ha+1)*hd] in out_proj of layer La must be True."""
        model = make_test_model()
        key = make_test_key()
        plan = build_mask_plan(model, key, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)
        hd = self._get_head_dim(model)

        attn_0 = model.transformer.h[0].attn.attention
        attn_2 = model.transformer.h[2].attn.attention

        out0 = attn_0.out_proj.weight
        assert out0 in masks, "Layer 0 out_proj should be in masks"
        assert masks[out0][:, hd:2*hd].all(), \
            "Columns [hd:2hd] in layer 0 out_proj must be True"
        assert not masks[out0][:, :hd].any(), \
            "Columns [0:hd] (head 0) in layer 0 out_proj must be False"

        out2 = attn_2.out_proj.weight
        assert out2 in masks, "Layer 2 out_proj should be in masks"
        assert masks[out2][:, 3*hd:4*hd].all(), \
            "Columns [3hd:4hd] in layer 2 out_proj must be True"

    def test_mlp_c_fc_keyed_rows_are_true(self):
        """For an mlp_cols swap (La, ca) ↔ (Lb, cb), row ca in c_fc of layer La must be True."""
        model = make_test_model()
        key = make_test_key()
        plan = build_mask_plan(model, key, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)

        mlp_1 = model.transformer.h[1].mlp
        mlp_3 = model.transformer.h[3].mlp

        fc1 = mlp_1.c_fc.weight
        assert fc1 in masks, "Layer 1 c_fc should be in masks"
        assert masks[fc1][50, :].all(), "Row 50 of layer 1 c_fc must be True"
        assert not masks[fc1][0, :].any(), "Row 0 of layer 1 c_fc must be False"

        fc3 = mlp_3.c_fc.weight
        assert fc3 in masks, "Layer 3 c_fc should be in masks"
        assert masks[fc3][100, :].all(), "Row 100 of layer 3 c_fc must be True"

    def test_mlp_c_proj_keyed_cols_are_true(self):
        """Column ca in c_proj of layer La must be True."""
        model = make_test_model()
        key = make_test_key()
        plan = build_mask_plan(model, key, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)

        mlp_1 = model.transformer.h[1].mlp
        mlp_3 = model.transformer.h[3].mlp

        proj1 = mlp_1.c_proj.weight
        assert proj1 in masks, "Layer 1 c_proj should be in masks"
        assert masks[proj1][:, 50].all(), "Column 50 of layer 1 c_proj must be True"
        assert not masks[proj1][:, 0].any(), "Column 0 of layer 1 c_proj must be False"

        proj3 = mlp_3.c_proj.weight
        assert proj3 in masks, "Layer 3 c_proj should be in masks"
        assert masks[proj3][:, 100].all(), "Column 100 of layer 3 c_proj must be True"

    def test_unswapped_layer_params_not_in_masks(self):
        """Params from layers not involved in any swap must not appear in the dict."""
        model = make_test_model()
        key = make_test_key()  # swaps: attn L0↔L2, MLP L1↔L3
        plan = build_mask_plan(model, key, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)

        # Layer 0 MLP and layer 1 attn are NOT part of any swap
        mlp_0 = model.transformer.h[0].mlp
        attn_1 = model.transformer.h[1].attn.attention

        all_params_in_masks = set(masks.keys())

        # These params should either be absent or have all-False masks
        for param in [mlp_0.c_fc.weight, mlp_0.c_proj.weight]:
            if param in all_params_in_masks:
                assert not masks[param].any(), \
                    "Unswapped MLP layer 0 params should have no True entries"

        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            param = getattr(attn_1, proj_name).weight
            if param in all_params_in_masks:
                assert not masks[param].any(), \
                    f"Unswapped attn layer 1 {proj_name} should have no True entries"

    def test_mask_values_are_boolean(self):
        """All masks must be boolean tensors."""
        model = make_test_model()
        key = make_test_key()
        plan = build_mask_plan(model, key, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)
        for param, mask in masks.items():
            assert mask.dtype == torch.bool, \
                f"Mask for param (shape {param.shape}) must be dtype bool, got {mask.dtype}"

    def test_mask_shape_matches_param_shape(self):
        """Every mask must have the same shape as its corresponding parameter."""
        model = make_test_model()
        key = make_test_key()
        plan = build_mask_plan(model, key, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)
        for param, mask in masks.items():
            assert mask.shape == param.shape, \
                f"Mask shape {mask.shape} must match param shape {param.shape}"

    def test_attn_only_key_produces_no_mlp_masks(self):
        """A key with only attn_heads swaps must not mark any MLP positions."""
        model = make_test_model()
        key_attn_only = PermutationKey(attn_heads=[((0, 0), (2, 2))])
        plan = build_mask_plan(model, key_attn_only, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)

        for layer_idx in range(4):
            mlp = model.transformer.h[layer_idx].mlp
            for w in [mlp.c_fc.weight, mlp.c_proj.weight]:
                if w in masks:
                    assert not masks[w].any(), \
                        f"attn-only key must not mark any MLP positions in layer {layer_idx}"

    def test_mlp_only_key_produces_no_attn_masks(self):
        """A key with only mlp_cols swaps must not mark any attn positions."""
        model = make_test_model()
        key_mlp_only = PermutationKey(mlp_cols=[((0, 10), (2, 20))])
        plan = build_mask_plan(model, key_mlp_only, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)

        for layer_idx in range(4):
            attn = model.transformer.h[layer_idx].attn.attention
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                w = getattr(attn, proj_name).weight
                if w in masks:
                    assert not masks[w].any(), \
                        f"MLP-only key must not mark any attn positions in layer {layer_idx}"

    def test_empty_key_produces_all_false_or_empty_masks(self):
        """An empty key must produce no True entries anywhere."""
        model = make_test_model()
        key_empty = PermutationKey()
        plan = build_mask_plan(model, key_empty, device=torch.device("cpu"))
        masks = build_keyed_param_masks(model, plan)
        for param, mask in masks.items():
            assert not mask.any(), "Empty key must produce no True entries in any mask"


# ---------------------------------------------------------------------------
# Section 7: _merge_idx utility
# ---------------------------------------------------------------------------

class TestMergeIdx:
    """Unit tests for the _merge_idx helper."""

    def test_both_none_returns_none(self):
        assert _merge_idx(None, None) is None

    def test_existing_none_returns_new(self):
        t = torch.tensor([1, 2, 3])
        result = _merge_idx(None, t)
        assert result is not None
        assert set(result.tolist()) == {1, 2, 3}

    def test_new_none_returns_existing(self):
        t = torch.tensor([1, 2, 3])
        result = _merge_idx(t, None)
        assert torch.equal(result, t)

    def test_union_of_disjoint_tensors(self):
        a = torch.tensor([0, 1, 2])
        b = torch.tensor([3, 4, 5])
        result = _merge_idx(a, b)
        assert set(result.tolist()) == {0, 1, 2, 3, 4, 5}

    def test_union_deduplicates_overlapping_indices(self):
        a = torch.tensor([0, 1, 2, 3])
        b = torch.tensor([2, 3, 4, 5])
        result = _merge_idx(a, b)
        assert set(result.tolist()) == {0, 1, 2, 3, 4, 5}
        # No duplicates
        assert result.numel() == len(set(result.tolist()))

    def test_empty_new_idx_returns_existing_unchanged(self):
        t = torch.tensor([1, 2])
        result = _merge_idx(t, torch.tensor([], dtype=torch.long))
        assert torch.equal(result, t)


# ---------------------------------------------------------------------------
# Section 8: Integration with train_step
# ---------------------------------------------------------------------------

class TestTrainStepIntegration:
    """End-to-end integration: train_step must freeze public params with AdamW."""

    def test_train_step_public_params_frozen_with_adamw(self):
        """Public params must be bit-for-bit identical before and after train_step."""
        from tiered.train.finetune.private_finetune import train_step

        model = make_test_model()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        key = make_test_key()
        device = torch.device("cpu")
        swap_plan = build_swap_plan(model, key, device)
        mask_plan = build_mask_plan(model, key, device)
        keyed_param_masks = build_keyed_param_masks(model, mask_plan)

        # Use AdamW with weight decay — the problematic case
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        # Warm up to build momentum
        for _ in range(3):
            dummy_ids = torch.randint(0, 100, (2, 16))
            model.apply_key(key)
            out = model(dummy_ids, labels=dummy_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.unapply_key(key)

        # Record public params — embeddings and layer norms are never keyed
        wte_before = model.transformer.wte.weight.data.clone()
        wpe_before = model.transformer.wpe.weight.data.clone()
        ln_f_before = model.transformer.ln_f.weight.data.clone()

        # Also record a public attn head (head 0 of layer 0 is not in our key)
        hd = model.transformer.h[0].attn.attention.head_dim
        attn_0 = model.transformer.h[0].attn.attention
        public_head_before = attn_0.q_proj.weight.data[:hd, :].clone()

        private_batch = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "labels": torch.randint(0, 100, (2, 16)),
        }
        public_batch = {"input_ids": torch.randint(0, 100, (2, 16))}

        train_step(
            model, model, ref_model,
            private_batch, public_batch,
            key, optimizer, device,
            kl_lambda=0.1, max_grad_norm=1.0,
            keyed_param_masks=keyed_param_masks,
            keyed_mask_plan=mask_plan,
        )

        assert torch.equal(model.transformer.wte.weight.data, wte_before), \
            "Token embeddings (public) must be frozen"
        assert torch.equal(model.transformer.wpe.weight.data, wpe_before), \
            "Position embeddings (public) must be frozen"
        assert torch.equal(model.transformer.ln_f.weight.data, ln_f_before), \
            "Final layer norm (public) must be frozen"
        assert torch.equal(attn_0.q_proj.weight.data[:hd, :], public_head_before), \
            "Public attn head (head 0, layer 0) must be frozen"

    def test_train_step_keyed_params_do_update_with_adamw(self):
        """Keyed params must actually change during train_step."""
        from tiered.train.finetune.private_finetune import train_step

        model = make_test_model()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        key = make_test_key()
        device = torch.device("cpu")
        mask_plan = build_mask_plan(model, key, device)
        keyed_param_masks = build_keyed_param_masks(model, mask_plan)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Keyed: head 1 of layer 0 (rows [hd:2*hd] of q_proj)
        hd = model.transformer.h[0].attn.attention.head_dim
        attn_0 = model.transformer.h[0].attn.attention
        keyed_head_before = attn_0.q_proj.weight.data[hd:2*hd, :].clone()

        private_batch = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "labels": torch.randint(0, 100, (2, 16)),
        }
        public_batch = {"input_ids": torch.randint(0, 100, (2, 16))}

        train_step(
            model, model, ref_model,
            private_batch, public_batch,
            key, optimizer, device,
            kl_lambda=0.0, max_grad_norm=1.0,
            keyed_param_masks=keyed_param_masks,
            keyed_mask_plan=mask_plan,
        )

        assert not torch.equal(attn_0.q_proj.weight.data[hd:2*hd, :], keyed_head_before), \
            "Keyed attn head must be updated by train_step"

    def test_train_step_without_keyed_masks_uses_plain_step(self):
        """When keyed_param_masks is None/empty, train_step falls back to optimizer.step()."""
        from tiered.train.finetune.private_finetune import train_step

        model = make_test_model()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        key = make_test_key()
        device = torch.device("cpu")
        mask_plan = build_mask_plan(model, key, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        private_batch = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "labels": torch.randint(0, 100, (2, 16)),
        }

        # Should not raise even with keyed_param_masks=None
        loss_priv, loss_kl, acc = train_step(
            model, model, ref_model,
            private_batch, None,
            key, optimizer, device,
            kl_lambda=0.0, max_grad_norm=1.0,
            keyed_param_masks=None,  # triggers plain optimizer.step()
            keyed_mask_plan=mask_plan,
        )
        assert loss_priv > 0
        assert isinstance(acc, float)

    def test_multi_step_public_params_never_drift(self):
        """After many train_step calls, public params must remain exactly frozen."""
        from tiered.train.finetune.private_finetune import train_step

        model = make_test_model()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        key = make_test_key()
        device = torch.device("cpu")
        mask_plan = build_mask_plan(model, key, device)
        keyed_param_masks = build_keyed_param_masks(model, mask_plan)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        wte_before = model.transformer.wte.weight.data.clone()
        ln_f_before = model.transformer.ln_f.weight.data.clone()

        for _ in range(20):
            private_batch = {
                "input_ids": torch.randint(0, 100, (2, 16)),
                "labels": torch.randint(0, 100, (2, 16)),
            }
            public_batch = {"input_ids": torch.randint(0, 100, (2, 16))}
            train_step(
                model, model, ref_model,
                private_batch, public_batch,
                key, optimizer, device,
                kl_lambda=0.1, max_grad_norm=1.0,
                keyed_param_masks=keyed_param_masks,
                keyed_mask_plan=mask_plan,
            )

        assert torch.equal(model.transformer.wte.weight.data, wte_before), \
            "Token embeddings must not drift over many train_steps"
        assert torch.equal(model.transformer.ln_f.weight.data, ln_f_before), \
            "Final layer norm must not drift over many train_steps"
