"""Tests for mixed_private_finetune.py.

Verifies the 2-tier private-finetune training step where the loss is

    L_ft = w_pub_c1 * CE(C1, public)
         + w_pub_c2 * CE(C2, public)
         + w_priv   * CE(C2, private)

Critical invariants the implementation must satisfy:

  1. Public weights are byte-identical across many steps (including the
     public slices of mixed parameters under AdamW weight decay).
  2. The model is in C1 (home) layout at the start and end of each step.
  3. The accumulated gradient just before optimizer.step equals the
     weighted sum of the three component gradients, with the C1 component
     correctly swapped to C2 positions.
  4. Each loss-weight argument behaves linearly: zeroing one component
     decouples the step from that data source.
  5. Inside one step, the layout transitions are C1 → C2 → C2 (one
     C1-public forward, then enter C2 for the C2-public and C2-private
     forwards).
  6. The same public batch is reused for both public-CE forwards.
  7. evaluate_on_dataset's accuracy ignores -100 labels and leaves the
     model in C1 after a C2 evaluation.

Most tests use a tiny GPT-Neo-like model and run on CPU; the linearity
test does the full train_step on one model and compares the captured
gradient to a manually-constructed reference on a copy.
"""

import copy
import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import GPTNeoConfig

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import (
    PermutationKey,
    apply_permutation,
    build_mask_plan,
    mask_public_gradients,
    swap_gradients,
    unapply_permutation,
)
from tiered.permutation.permute import build_swap_plan
from tiered.train.finetune import mixed_private_finetune as mpf
from tiered.train.utils import build_keyed_param_masks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def create_test_model(seed: int = 0):
    """Tiny GPT-Neo for CPU-only deterministic tests (eval mode → no dropout)."""
    torch.manual_seed(seed)
    config = GPTNeoConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        intermediate_size=256,
        attention_types=[[["global"], 1]] * 4,
        max_position_embeddings=128,
    )
    model = GPTNeoForCausalLMTiered(config)
    model.eval()
    return model


def create_test_key():
    """Key swapping head 1 of layer 0 with head 3 of layer 2,
    and column 50 of layer 1 with column 100 of layer 3."""
    return PermutationKey(
        attn_heads=[((0, 1), (2, 3))],
        mlp_cols=[((1, 50), (3, 100))],
    )


def make_batches(seed: int = 0, batch_size: int = 2, seq_len: int = 16):
    """Return (private_batch, public_batch) with input_ids+labels populated."""
    g = torch.Generator().manual_seed(seed)
    private_ids = torch.randint(1, 99, (batch_size, seq_len), generator=g)
    public_ids = torch.randint(1, 99, (batch_size, seq_len), generator=g)
    return (
        {"input_ids": private_ids, "labels": private_ids.clone()},
        {"input_ids": public_ids, "labels": public_ids.clone()},
    )


def build_step_state(seed: int = 42):
    """Build (model, key, device, swap_plan, mask_plan, keyed_param_masks)."""
    model = create_test_model(seed)
    key = create_test_key()
    device = torch.device("cpu")
    sp = build_swap_plan(model, key, device)
    mp = build_mask_plan(model, key, device)
    masks = build_keyed_param_masks(model, mp)
    return model, key, device, sp, mp, masks


def get_keyed_positions(model):
    h = model.transformer.h
    head_dim = h[0].attn.attention.head_dim
    return {
        "L0_H1_q": h[0].attn.attention.q_proj.weight[head_dim:2 * head_dim].clone(),
        "L0_H1_k": h[0].attn.attention.k_proj.weight[head_dim:2 * head_dim].clone(),
        "L0_H1_v": h[0].attn.attention.v_proj.weight[head_dim:2 * head_dim].clone(),
        "L0_H1_o": h[0].attn.attention.out_proj.weight[:, head_dim:2 * head_dim].clone(),
        "L2_H3_q": h[2].attn.attention.q_proj.weight[3 * head_dim:4 * head_dim].clone(),
        "L2_H3_k": h[2].attn.attention.k_proj.weight[3 * head_dim:4 * head_dim].clone(),
        "L2_H3_v": h[2].attn.attention.v_proj.weight[3 * head_dim:4 * head_dim].clone(),
        "L2_H3_o": h[2].attn.attention.out_proj.weight[:, 3 * head_dim:4 * head_dim].clone(),
        "L1_C50_fc": h[1].mlp.c_fc.weight[50].clone(),
        "L1_C50_proj": h[1].mlp.c_proj.weight[:, 50].clone(),
        "L3_C100_fc": h[3].mlp.c_fc.weight[100].clone(),
        "L3_C100_proj": h[3].mlp.c_proj.weight[:, 100].clone(),
    }


def get_public_positions(model):
    """Snapshot some public (non-keyed) positions, including positions inside
    *mixed* parameters whose other slices are keyed (e.g., head 0 of layer 0's
    q_proj — head 1 IS keyed, head 0 is not)."""
    h = model.transformer.h
    head_dim = h[0].attn.attention.head_dim
    return {
        # Mixed-param public slices (keyed siblings exist in same param)
        "L0_H0_q_mixed": h[0].attn.attention.q_proj.weight[:head_dim].clone(),
        "L0_H2_k_mixed": h[0].attn.attention.k_proj.weight[2 * head_dim:3 * head_dim].clone(),
        "L1_C0_fc_mixed": h[1].mlp.c_fc.weight[0].clone(),
        # Entirely public params
        "wte": model.transformer.wte.weight.clone(),
        "wpe": model.transformer.wpe.weight.clone(),
        "ln_f": model.transformer.ln_f.weight.clone(),
        "L0_ln_1": h[0].ln_1.weight.clone(),
        "L0_ln_2": h[0].ln_2.weight.clone(),
        # Layer with no keyed positions at all (layers 0 has keyed attn,
        # layer 1 has keyed mlp; layer 0's mlp + layer 1's attn are fully public)
        "L0_mlp_b": h[0].mlp.c_fc.bias.clone() if h[0].mlp.c_fc.bias is not None else None,
    }


def assert_dicts_byte_equal(a, b, label=""):
    for k, v in a.items():
        if v is None:
            assert b[k] is None, f"{label}: {k} expected None, got tensor"
            continue
        assert torch.equal(v, b[k]), \
            f"{label}: {k} differs (max abs delta = {(v - b[k]).abs().max()})"


def run_step(model, *, w_priv=0.8, w_pub_c2=0.1, w_pub_c1=0.1,
             optimizer=None, lr=0.1, max_grad_norm=1.0, seed=0):
    """One mixed_private_finetune.train_step on the given model."""
    key = create_test_key()
    device = torch.device("cpu")
    sp = build_swap_plan(model, key, device)
    mp = build_mask_plan(model, key, device)
    masks = build_keyed_param_masks(model, mp)
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    priv, pub = make_batches(seed=seed)
    return mpf.train_step(
        model, model, priv, pub, key, optimizer, device,
        w_priv=w_priv, w_pub_c2=w_pub_c2, w_pub_c1=w_pub_c1,
        max_grad_norm=max_grad_norm,
        keyed_param_masks=masks, keyed_mask_plan=mp,
        is_distributed=False, active_swap_plan=sp,
    )


# ---------------------------------------------------------------------------
# 1. Per-step weight invariants
# ---------------------------------------------------------------------------

class TestPerStepInvariants:
    def test_train_step_returns_four_floats(self):
        model, *_ = build_step_state()
        result = run_step(model)
        assert len(result) == 4
        loss_priv, loss_pub_c2, loss_pub_c1, acc = result
        for name, v in [("loss_priv", loss_priv),
                        ("loss_pub_c2", loss_pub_c2),
                        ("loss_pub_c1", loss_pub_c1)]:
            assert isinstance(v, float), f"{name} is {type(v)}, not float"
            assert v > 0, f"{name} should be positive, got {v}"
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_keyed_weights_change_after_one_step(self):
        model, *_ = build_step_state()
        before = get_keyed_positions(model)
        run_step(model, lr=0.1)
        after = get_keyed_positions(model)
        changed = sum(1 for k in before if not torch.allclose(before[k], after[k]))
        assert changed >= len(before) - 1, \
            f"Most keyed positions should have changed; only {changed}/{len(before)} did"

    def test_public_weights_unchanged_after_one_step(self):
        model, *_ = build_step_state()
        before = get_public_positions(model)
        run_step(model, lr=0.1)
        after = get_public_positions(model)
        assert_dicts_byte_equal(before, after, "public after 1 SGD step")

    def test_model_returns_to_c1_after_step(self):
        """With lr=0 (no actual update), the layout must be back in C1 at end."""
        model, *_ = build_step_state()
        head_dim = model.transformer.h[0].attn.attention.head_dim
        before_L0_H1 = model.transformer.h[0].attn.attention.q_proj.weight[head_dim:2*head_dim].clone()
        before_L2_H3 = model.transformer.h[2].attn.attention.q_proj.weight[3*head_dim:4*head_dim].clone()

        run_step(model, lr=0.0)

        after_L0_H1 = model.transformer.h[0].attn.attention.q_proj.weight[head_dim:2*head_dim]
        after_L2_H3 = model.transformer.h[2].attn.attention.q_proj.weight[3*head_dim:4*head_dim]
        assert torch.equal(before_L0_H1, after_L0_H1), \
            "L0_H1 weight changed despite lr=0 (model likely left in C2)"
        assert torch.equal(before_L2_H3, after_L2_H3), \
            "L2_H3 weight changed despite lr=0 (model likely left in C2)"

    def test_non_keyed_params_have_grad_none_after_step(self):
        """Params entirely outside the keyed mask must have grad=None so AdamW
        skips them — otherwise weight-decay + momentum would silently drift them."""
        model, *_ = build_step_state()
        run_step(model, lr=0.1)
        # Embeddings, position embeddings, layer norms — none of these have
        # any keyed positions, so they must be in keyed_param_masks=None and
        # have grad=None after the step.
        assert model.transformer.wte.weight.grad is None
        assert model.transformer.wpe.weight.grad is None
        assert model.transformer.ln_f.weight.grad is None
        assert model.transformer.h[0].ln_1.weight.grad is None
        assert model.transformer.h[0].ln_2.weight.grad is None


# ---------------------------------------------------------------------------
# 2. Cross-step freeze (the most stringent public-weight test)
# ---------------------------------------------------------------------------

class TestCrossStepFreeze:
    def test_public_weights_byte_identical_after_many_adamw_steps(self):
        """Run many AdamW steps and verify every public position is bit-for-bit
        identical to its initial value. This catches:
          - mask_public_gradients leaking
          - adamw_step_preserving_public not restoring public slices
          - weight decay drifting public positions of mixed params
          - momentum accumulating at public positions
        """
        model, key, device, sp, mp, masks = build_step_state()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.5)
        before = get_public_positions(model)

        for step in range(8):
            priv, pub = make_batches(seed=step)
            mpf.train_step(
                model, model, priv, pub, key, opt, device,
                w_priv=0.7, w_pub_c2=0.2, w_pub_c1=0.1, max_grad_norm=1.0,
                keyed_param_masks=masks, keyed_mask_plan=mp,
                is_distributed=False, active_swap_plan=sp,
            )

        after = get_public_positions(model)
        assert_dicts_byte_equal(before, after, "public after 8 AdamW(wd=0.5) steps")

    def test_public_slices_of_mixed_params_dont_drift_with_weight_decay(self):
        """Specifically: head 0 of layer 0's q_proj is public, but lives in the
        same parameter tensor as head 1 (which IS keyed). Without correct
        public-slice restoration, AdamW weight decay would multiply the whole
        param by (1 - lr*wd) and drift head 0."""
        model, key, device, sp, mp, masks = build_step_state()
        head_dim = model.transformer.h[0].attn.attention.head_dim

        public_slice_before = (
            model.transformer.h[0].attn.attention.q_proj.weight[:head_dim].clone()
        )

        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.5)
        for step in range(5):
            priv, pub = make_batches(seed=step)
            mpf.train_step(
                model, model, priv, pub, key, opt, device,
                w_priv=0.7, w_pub_c2=0.2, w_pub_c1=0.1, max_grad_norm=1.0,
                keyed_param_masks=masks, keyed_mask_plan=mp,
                is_distributed=False, active_swap_plan=sp,
            )

        public_slice_after = (
            model.transformer.h[0].attn.attention.q_proj.weight[:head_dim]
        )
        assert torch.equal(public_slice_before, public_slice_after), \
            "Public slice of mixed q_proj drifted under AdamW weight decay"


# ---------------------------------------------------------------------------
# 3. Gradient correctness — linearity / weighted-sum reconstruction
# ---------------------------------------------------------------------------

class TestGradientLinearity:
    """The most rigorous correctness check: the gradient just before
    optimizer.step (in C2 layout) must equal the manually-constructed
    weighted sum of the three component gradients."""

    def test_combined_grad_matches_manual_construction(self, monkeypatch):
        device = torch.device("cpu")
        model_actual = create_test_model(seed=42)
        key = create_test_key()
        sp = build_swap_plan(model_actual, key, device)
        mp = build_mask_plan(model_actual, key, device)
        masks = build_keyed_param_masks(model_actual, mp)

        priv, pub = make_batches(seed=7)
        w_priv, w_pub_c2, w_pub_c1 = 0.8, 0.1, 0.1

        # ---- Manual reference (mirrors train_step, no optimizer) ----
        model_ref = copy.deepcopy(model_actual)
        sp_ref = build_swap_plan(model_ref, key, device)
        mp_ref = build_mask_plan(model_ref, key, device)
        model_ref.zero_grad()

        out1 = model_ref(pub["input_ids"], labels=pub["labels"])
        (w_pub_c1 * out1.loss).backward()

        apply_permutation(model_ref, key, plan=sp_ref)
        swap_gradients(model_ref, key, plan=sp_ref)

        out2 = model_ref(pub["input_ids"], labels=pub["labels"])
        (w_pub_c2 * out2.loss).backward()

        out3 = model_ref(priv["input_ids"], labels=priv["labels"])
        (w_priv * out3.loss).backward()

        mask_public_gradients(model_ref, key, plan=mp_ref)
        # NB: don't null non-keyed params' grads here — keep zero tensors so
        # we can compare against the actual run. Don't clip (we'll set
        # max_grad_norm huge in the actual run too). Don't optimizer-step.
        expected = {
            n: (p.grad.clone() if p.grad is not None else None)
            for n, p in model_ref.named_parameters()
        }

        # ---- Actual: train_step with optimizer step monkeypatched out ----
        captured = {}

        def fake_step(_optimizer, _masks_arg):
            for n, p in model_actual.named_parameters():
                captured[n] = p.grad.clone() if p.grad is not None else None

        monkeypatch.setattr(mpf, "adamw_step_preserving_public", fake_step)

        opt = torch.optim.AdamW(model_actual.parameters(), lr=1e-3)
        mpf.train_step(
            model_actual, model_actual, priv, pub, key, opt, device,
            w_priv=w_priv, w_pub_c2=w_pub_c2, w_pub_c1=w_pub_c1,
            max_grad_norm=1e9,  # don't clip
            keyed_param_masks=masks, keyed_mask_plan=mp,
            is_distributed=False, active_swap_plan=sp,
        )

        # ---- Compare ----
        for n, p in model_actual.named_parameters():
            cap = captured.get(n)
            exp = expected[n]
            if cap is None:
                # Actual nulled it because p ∉ keyed_param_masks. The expected
                # for such a param must be all-zero (mask_public_gradients
                # zeroed it and there were no keyed positions to restore).
                assert p not in masks, f"{n}: grad nulled but param IS keyed"
                if exp is not None:
                    assert torch.equal(exp, torch.zeros_like(exp)), \
                        f"{n}: actual nulled grad but expected has non-zero values"
                continue
            assert exp is not None, f"{n}: actual has grad but expected is None"
            assert torch.allclose(cap, exp, atol=1e-5, rtol=1e-4), (
                f"{n}: max |actual - expected| = {(cap - exp).abs().max().item()}"
            )


# ---------------------------------------------------------------------------
# 4. Equivalence checks for individual loss components
# ---------------------------------------------------------------------------

class TestSingleComponentEquivalence:
    """Setting two of the three weights to 0 should reduce the step to a
    single-loss reference."""

    def test_only_w_pub_c2_matches_pure_c2_pub_step(self):
        """(w_priv=0, w_pub_c2=1, w_pub_c1=0) reduces to a single C2-public
        SGD step."""
        device = torch.device("cpu")
        priv, pub = make_batches(seed=10)

        # Mixed implementation
        model_mix = create_test_model(seed=42)
        key = create_test_key()
        sp = build_swap_plan(model_mix, key, device)
        mp = build_mask_plan(model_mix, key, device)
        masks = build_keyed_param_masks(model_mix, mp)
        opt_mix = torch.optim.SGD(model_mix.parameters(), lr=0.01)
        mpf.train_step(
            model_mix, model_mix, priv, pub, key, opt_mix, device,
            w_priv=0.0, w_pub_c2=1.0, w_pub_c1=0.0, max_grad_norm=1e9,
            keyed_param_masks=masks, keyed_mask_plan=mp,
            is_distributed=False, active_swap_plan=sp,
        )

        # Reference: single C2-public CE backward + masked SGD step in C2
        model_ref = create_test_model(seed=42)
        sp_ref = build_swap_plan(model_ref, key, device)
        mp_ref = build_mask_plan(model_ref, key, device)
        masks_ref = build_keyed_param_masks(model_ref, mp_ref)
        opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)

        opt_ref.zero_grad()
        apply_permutation(model_ref, key, plan=sp_ref)
        out = model_ref(pub["input_ids"], labels=pub["labels"])
        out.loss.backward()
        mask_public_gradients(model_ref, key, plan=mp_ref)
        for p in model_ref.parameters():
            if p not in masks_ref and p.grad is not None:
                p.grad = None
        opt_ref.step()
        unapply_permutation(model_ref, key, plan=sp_ref)

        for (na, pa), (nb, pb) in zip(model_mix.named_parameters(),
                                      model_ref.named_parameters()):
            assert torch.allclose(pa, pb, atol=1e-6, rtol=1e-5), (
                f"{na}: max |mix - ref| = {(pa - pb).abs().max().item()}"
            )

    def test_only_w_pub_c1_matches_swapped_c1_pub_step(self):
        """(w_priv=0, w_pub_c2=0, w_pub_c1=1) reduces to: compute C1-public
        gradient at home, swap to C2 positions, mask, step in C2, unapply.

        This specifically verifies the swap_gradients dance — if the
        implementation forgot swap_gradients, the step would update wrong
        positions and this test would fail."""
        device = torch.device("cpu")
        priv, pub = make_batches(seed=11)

        model_mix = create_test_model(seed=42)
        key = create_test_key()
        sp = build_swap_plan(model_mix, key, device)
        mp = build_mask_plan(model_mix, key, device)
        masks = build_keyed_param_masks(model_mix, mp)
        opt_mix = torch.optim.SGD(model_mix.parameters(), lr=0.01)
        mpf.train_step(
            model_mix, model_mix, priv, pub, key, opt_mix, device,
            w_priv=0.0, w_pub_c2=0.0, w_pub_c1=1.0, max_grad_norm=1e9,
            keyed_param_masks=masks, keyed_mask_plan=mp,
            is_distributed=False, active_swap_plan=sp,
        )

        model_ref = create_test_model(seed=42)
        sp_ref = build_swap_plan(model_ref, key, device)
        mp_ref = build_mask_plan(model_ref, key, device)
        masks_ref = build_keyed_param_masks(model_ref, mp_ref)
        opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)

        opt_ref.zero_grad()
        out = model_ref(pub["input_ids"], labels=pub["labels"])
        out.loss.backward()
        apply_permutation(model_ref, key, plan=sp_ref)
        swap_gradients(model_ref, key, plan=sp_ref)
        mask_public_gradients(model_ref, key, plan=mp_ref)
        for p in model_ref.parameters():
            if p not in masks_ref and p.grad is not None:
                p.grad = None
        opt_ref.step()
        unapply_permutation(model_ref, key, plan=sp_ref)

        for (na, pa), (nb, pb) in zip(model_mix.named_parameters(),
                                      model_ref.named_parameters()):
            assert torch.allclose(pa, pb, atol=1e-6, rtol=1e-5), (
                f"{na}: max |mix - ref| = {(pa - pb).abs().max().item()}"
            )

    def test_only_w_priv_matches_pure_c2_priv_step(self):
        """(w_priv=1, w_pub_c2=0, w_pub_c1=0) reduces to a single C2-private
        SGD step (the original 'no-KL' baseline)."""
        device = torch.device("cpu")
        priv, pub = make_batches(seed=12)

        model_mix = create_test_model(seed=42)
        key = create_test_key()
        sp = build_swap_plan(model_mix, key, device)
        mp = build_mask_plan(model_mix, key, device)
        masks = build_keyed_param_masks(model_mix, mp)
        opt_mix = torch.optim.SGD(model_mix.parameters(), lr=0.01)
        mpf.train_step(
            model_mix, model_mix, priv, pub, key, opt_mix, device,
            w_priv=1.0, w_pub_c2=0.0, w_pub_c1=0.0, max_grad_norm=1e9,
            keyed_param_masks=masks, keyed_mask_plan=mp,
            is_distributed=False, active_swap_plan=sp,
        )

        model_ref = create_test_model(seed=42)
        sp_ref = build_swap_plan(model_ref, key, device)
        mp_ref = build_mask_plan(model_ref, key, device)
        masks_ref = build_keyed_param_masks(model_ref, mp_ref)
        opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01)

        opt_ref.zero_grad()
        apply_permutation(model_ref, key, plan=sp_ref)
        out = model_ref(priv["input_ids"], labels=priv["labels"])
        out.loss.backward()
        mask_public_gradients(model_ref, key, plan=mp_ref)
        for p in model_ref.parameters():
            if p not in masks_ref and p.grad is not None:
                p.grad = None
        opt_ref.step()
        unapply_permutation(model_ref, key, plan=sp_ref)

        for (na, pa), (nb, pb) in zip(model_mix.named_parameters(),
                                      model_ref.named_parameters()):
            assert torch.allclose(pa, pb, atol=1e-6, rtol=1e-5), (
                f"{na}: max |mix - ref| = {(pa - pb).abs().max().item()}"
            )


# ---------------------------------------------------------------------------
# 5. Loss-component isolation
# ---------------------------------------------------------------------------

class TestLossComponentIsolation:
    def test_zero_w_priv_makes_step_independent_of_private_data(self):
        device = torch.device("cpu")
        # Same public batch, different private data; same starting weights.
        priv_a, pub = make_batches(seed=10)
        priv_b, _ = make_batches(seed=999)

        model_a = create_test_model(seed=42)
        model_b = copy.deepcopy(model_a)
        key = create_test_key()

        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
        sp_a = build_swap_plan(model_a, key, device)
        mp_a = build_mask_plan(model_a, key, device)
        masks_a = build_keyed_param_masks(model_a, mp_a)
        sp_b = build_swap_plan(model_b, key, device)
        mp_b = build_mask_plan(model_b, key, device)
        masks_b = build_keyed_param_masks(model_b, mp_b)

        mpf.train_step(model_a, model_a, priv_a, pub, key, opt_a, device,
                       w_priv=0.0, w_pub_c2=0.5, w_pub_c1=0.5, max_grad_norm=1e9,
                       keyed_param_masks=masks_a, keyed_mask_plan=mp_a,
                       is_distributed=False, active_swap_plan=sp_a)
        mpf.train_step(model_b, model_b, priv_b, pub, key, opt_b, device,
                       w_priv=0.0, w_pub_c2=0.5, w_pub_c1=0.5, max_grad_norm=1e9,
                       keyed_param_masks=masks_b, keyed_mask_plan=mp_b,
                       is_distributed=False, active_swap_plan=sp_b)

        for (na, pa), (nb, pb) in zip(model_a.named_parameters(),
                                      model_b.named_parameters()):
            assert torch.allclose(pa, pb, atol=1e-7), \
                f"{na}: w_priv=0 but result depends on private data"

    def test_zero_w_pub_makes_step_independent_of_public_data(self):
        device = torch.device("cpu")
        priv, pub_a = make_batches(seed=10)
        _, pub_b = make_batches(seed=999)

        model_a = create_test_model(seed=42)
        model_b = copy.deepcopy(model_a)
        key = create_test_key()

        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
        sp_a = build_swap_plan(model_a, key, device)
        mp_a = build_mask_plan(model_a, key, device)
        masks_a = build_keyed_param_masks(model_a, mp_a)
        sp_b = build_swap_plan(model_b, key, device)
        mp_b = build_mask_plan(model_b, key, device)
        masks_b = build_keyed_param_masks(model_b, mp_b)

        mpf.train_step(model_a, model_a, priv, pub_a, key, opt_a, device,
                       w_priv=1.0, w_pub_c2=0.0, w_pub_c1=0.0, max_grad_norm=1e9,
                       keyed_param_masks=masks_a, keyed_mask_plan=mp_a,
                       is_distributed=False, active_swap_plan=sp_a)
        mpf.train_step(model_b, model_b, priv, pub_b, key, opt_b, device,
                       w_priv=1.0, w_pub_c2=0.0, w_pub_c1=0.0, max_grad_norm=1e9,
                       keyed_param_masks=masks_b, keyed_mask_plan=mp_b,
                       is_distributed=False, active_swap_plan=sp_b)

        for (na, pa), (nb, pb) in zip(model_a.named_parameters(),
                                      model_b.named_parameters()):
            assert torch.allclose(pa, pb, atol=1e-7), \
                f"{na}: w_pub=0 but result depends on public data"


# ---------------------------------------------------------------------------
# 6. Layout transitions during the step
# ---------------------------------------------------------------------------

class TestLayoutDuringStep:
    def test_forwards_run_in_order_C1_then_C2_then_C2(self):
        """The three forward calls inside one step must see the model in
        layouts [C1, C2, C2] respectively. Wraps model.forward to inspect
        a known-keyed weight position and infer which layout is active."""
        model, key, device, sp, mp, masks = build_step_state()
        head_dim = model.transformer.h[0].attn.attention.head_dim
        c1_L0_H1 = model.transformer.h[0].attn.attention.q_proj.weight[head_dim:2*head_dim].clone()
        c1_L2_H3 = model.transformer.h[2].attn.attention.q_proj.weight[3*head_dim:4*head_dim].clone()

        layouts = []
        original_forward = model.forward

        def wrapped(*args, **kwargs):
            cur = model.transformer.h[0].attn.attention.q_proj.weight[head_dim:2*head_dim]
            if torch.equal(cur, c1_L0_H1):
                layouts.append("C1")
            elif torch.equal(cur, c1_L2_H3):
                layouts.append("C2")
            else:
                layouts.append("?")
            return original_forward(*args, **kwargs)

        model.forward = wrapped
        try:
            run_step(model, lr=0.0)  # lr=0 keeps weights stable across the step
        finally:
            model.forward = original_forward

        assert layouts == ["C1", "C2", "C2"], (
            f"Expected layout sequence ['C1','C2','C2'], got {layouts}"
        )


# ---------------------------------------------------------------------------
# 7. Public-batch reuse
# ---------------------------------------------------------------------------

class TestPublicBatchReuse:
    def test_same_public_batch_used_in_both_public_passes(self):
        """The same public input_ids tensor must be the input of both the
        C1-public and C2-public forwards."""
        model, key, device, sp, mp, masks = build_step_state()
        priv, pub = make_batches(seed=5)

        captured_inputs = []
        original_forward = model.forward

        def wrapped(*args, **kwargs):
            input_ids = args[0] if args else kwargs.get("input_ids")
            captured_inputs.append(input_ids.clone())
            return original_forward(*args, **kwargs)

        model.forward = wrapped
        try:
            opt = torch.optim.SGD(model.parameters(), lr=0.0)
            mpf.train_step(
                model, model, priv, pub, key, opt, device,
                w_priv=0.8, w_pub_c2=0.1, w_pub_c1=0.1, max_grad_norm=1.0,
                keyed_param_masks=masks, keyed_mask_plan=mp,
                is_distributed=False, active_swap_plan=sp,
            )
        finally:
            model.forward = original_forward

        assert len(captured_inputs) == 3, \
            f"Expected exactly 3 forward calls per step, got {len(captured_inputs)}"
        assert torch.equal(captured_inputs[0], pub["input_ids"]), \
            "1st forward must be the public batch (C1-public)"
        assert torch.equal(captured_inputs[1], pub["input_ids"]), \
            "2nd forward must be the SAME public batch (C2-public)"
        assert torch.equal(captured_inputs[2], priv["input_ids"]), \
            "3rd forward must be the private batch (C2-private)"


# ---------------------------------------------------------------------------
# 8. evaluate_on_dataset
# ---------------------------------------------------------------------------

def _tiny_eval_loader(seed: int = 0):
    """Iterable yielding pre-batched (B, T) tensors — sidesteps DataLoader/
    collator wiring for tests."""
    g = torch.Generator().manual_seed(seed)
    batch = {
        "input_ids": torch.randint(1, 99, (2, 16), generator=g),
        "labels": torch.randint(1, 99, (2, 16), generator=g),
    }
    return [batch]


class TestEvaluateOnDataset:
    def test_returns_only_c1_when_eval_c2_false(self):
        model = create_test_model(seed=0)
        key = create_test_key()
        device = torch.device("cpu")
        sp = build_swap_plan(model, key, device)
        m = mpf.evaluate_on_dataset(model, _tiny_eval_loader(), key, device,
                                    num_steps=1, eval_c2=False, active_swap_plan=sp)
        assert {"loss_c1", "ppl_c1", "acc_c1"} <= set(m.keys())
        assert "loss_c2" not in m
        assert "acc_c2" not in m

    def test_returns_c1_and_c2_when_eval_c2_true(self):
        model = create_test_model(seed=0)
        key = create_test_key()
        device = torch.device("cpu")
        sp = build_swap_plan(model, key, device)
        m = mpf.evaluate_on_dataset(model, _tiny_eval_loader(), key, device,
                                    num_steps=1, eval_c2=True, active_swap_plan=sp)
        assert {"loss_c1", "ppl_c1", "acc_c1",
                "loss_c2", "ppl_c2", "acc_c2"} <= set(m.keys())

    def test_model_back_in_c1_after_c2_eval(self):
        model = create_test_model(seed=0)
        key = create_test_key()
        device = torch.device("cpu")
        sp = build_swap_plan(model, key, device)
        head_dim = model.transformer.h[0].attn.attention.head_dim
        before = model.transformer.h[0].attn.attention.q_proj.weight[head_dim:2*head_dim].clone()

        mpf.evaluate_on_dataset(model, _tiny_eval_loader(), key, device,
                                num_steps=2, eval_c2=True, active_swap_plan=sp)

        after = model.transformer.h[0].attn.attention.q_proj.weight[head_dim:2*head_dim]
        assert torch.equal(before, after), "Model must be back in C1 after C2 eval"

    def test_perplexity_is_exp_of_loss(self):
        model = create_test_model(seed=0)
        key = create_test_key()
        device = torch.device("cpu")
        sp = build_swap_plan(model, key, device)
        m = mpf.evaluate_on_dataset(model, _tiny_eval_loader(), key, device,
                                    num_steps=1, eval_c2=True, active_swap_plan=sp)
        assert math.isclose(m["ppl_c1"], math.exp(min(m["loss_c1"], 100)), rel_tol=1e-6)
        assert math.isclose(m["ppl_c2"], math.exp(min(m["loss_c2"], 100)), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 9. Accuracy semantics: ignore -100 labels
# ---------------------------------------------------------------------------

class _DummyLogitsModel(torch.nn.Module):
    """Stand-in model whose forward returns scripted logits + a constant loss,
    so we can isolate the accuracy/loss-handling code path from real model
    arithmetic."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self._logits = logits

    def apply_key(self, _key):
        return None

    def unapply_key(self, _key):
        return None

    def forward(self, input_ids, labels=None):
        logits = self._logits.to(input_ids.device) + (self.anchor * 0.0)
        loss = self.anchor * 0.0 + 1.0
        return SimpleNamespace(logits=logits, loss=loss)


def _scripted_logits(vocab_size: int = 8) -> torch.Tensor:
    """Logits where positions 0,1 predict their non-masked targets correctly
    and positions 2,3 deliberately predict the wrong token (those targets are
    -100 / masked out, so a correct-accuracy implementation must score 1.0)."""
    logits = torch.full((1, 5, vocab_size), -50.0)
    logits[0, 0, 2] = 50.0   # target 2 (correct, kept)
    logits[0, 1, 3] = 50.0   # target 3 (correct, kept)
    logits[0, 2, 1] = 50.0   # target -100 (masked → ignored)
    logits[0, 3, 1] = 50.0   # target -100 (masked → ignored)
    logits[0, 4, 0] = 50.0   # final timestep unused by next-token metric
    return logits


class TestAccuracySemantics:
    def test_train_step_accuracy_ignores_minus100(self, monkeypatch):
        model = _DummyLogitsModel(_scripted_logits())
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        # Sidestep permutation/mask machinery — keep the test focused on metrics.
        monkeypatch.setattr(mpf, "apply_permutation", lambda *a, **k: None)
        monkeypatch.setattr(mpf, "unapply_permutation", lambda *a, **k: None)
        monkeypatch.setattr(mpf, "swap_gradients", lambda *a, **k: None)
        monkeypatch.setattr(mpf, "mask_public_gradients", lambda *a, **k: None)
        monkeypatch.setattr(mpf, "adamw_step_preserving_public", lambda *a, **k: None)

        priv = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "labels": torch.tensor([[1, 2, 3, -100, -100]])}
        pub = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
               "labels": torch.tensor([[1, 2, 3, 4, 5]])}

        loss_priv, loss_pub_c2, loss_pub_c1, acc = mpf.train_step(
            model=model, raw_model=model,
            private_batch=priv, public_batch=pub,
            key=object(), optimizer=opt, device=torch.device("cpu"),
            w_priv=0.8, w_pub_c2=0.1, w_pub_c1=0.1, max_grad_norm=1.0,
            keyed_param_masks={}, keyed_mask_plan=None,
            is_distributed=False, active_swap_plan=None,
        )

        # Both kept positions (0,1) are correct → accuracy must be 1.0
        # despite the wrong predictions at the masked positions (2,3).
        assert acc == pytest.approx(1.0), \
            f"train_step accuracy must ignore -100 labels (got {acc})"

    def test_evaluate_accuracy_ignores_minus100(self):
        model = _DummyLogitsModel(_scripted_logits())
        batch = {
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "labels": torch.tensor([1, 2, 3, -100, -100]),
        }
        dl = torch.utils.data.DataLoader([batch], batch_size=1)
        m = mpf.evaluate_on_dataset(model, dl, key=None, device=torch.device("cpu"),
                                    num_steps=1, eval_c2=False)
        assert m["acc_c1"] == pytest.approx(1.0), \
            f"evaluate accuracy must ignore -100 labels (got {m['acc_c1']})"


# ---------------------------------------------------------------------------
# 10. main() smoke / ordering test (mirrors the private_finetune test)
# ---------------------------------------------------------------------------

class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self._rows = rows
        self.column_names = list(rows[0].keys())

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]

    def select(self, indices):
        if isinstance(indices, range):
            indices = list(indices)
        return _TinyDataset([self._rows[i] for i in indices])

    def remove_columns(self, cols):
        keep = [c for c in self.column_names if c not in cols]
        new_rows = [{k: v for k, v in r.items() if k in keep} for r in self._rows]
        ds = _TinyDataset(new_rows)
        ds.column_names = keep
        return ds


class _DummyCollator:
    def __call__(self, examples):
        ids = torch.stack(
            [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples], dim=0
        )
        return {"input_ids": ids, "labels": ids.clone()}


class _DummyModelMain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=2048)

    def to(self, device):
        return self

    def gradient_checkpointing_enable(self, **_kwargs):
        return None

    def apply_key(self, _key):
        return None

    def unapply_key(self, _key):
        return None

    def save_pretrained(self, _path):
        return None

    @classmethod
    def from_pretrained(cls, _ckpt):
        return cls()


class _DummyWandb:
    def __init__(self):
        self.run = SimpleNamespace(id="dummy", summary={})
        self.config = SimpleNamespace(update=lambda *a, **k: None)

    def init(self, **_kwargs):
        return self.run

    def define_metric(self, *_a, **_k):
        return None

    def log(self, *_a, **_k):
        return None

    def finish(self):
        return None


def _tiny_dataset_dict():
    rows = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1]},
    ]
    ds = _TinyDataset(rows)
    return {"train": ds, "test": ds}


class TestMainOrdering:
    def test_step0_validation_runs_before_first_train_step(self, monkeypatch, tmp_path):
        events = []

        monkeypatch.setattr(
            mpf, "parse_args",
            lambda: SimpleNamespace(
                checkpoint="dummy", key_path="dummy",
                private_data="dummy", public_data="dummy",
                output_dir=str(tmp_path),
                batch_size=1, learning_rate=1e-5, min_lr=1e-6,
                max_steps=1, warmup_steps=0,
                max_grad_norm=1.0, keyed_l2_lambda=0.0,
                resume_from=None,
                w_priv=0.8, w_pub_c2=0.1, w_pub_c1=0.1,
                eval_interval=500, eval_steps=1,
                log_interval=1, save_interval=1000,
                wandb_project="test-project", run_name="test-run",
                num_workers=0,
            ),
        )

        monkeypatch.setattr(mpf.GPTNeoForCausalLMTiered,
                            "from_pretrained", _DummyModelMain.from_pretrained)
        monkeypatch.setattr(mpf, "load_key",
                            lambda _p: SimpleNamespace(attn_heads=[], mlp_cols=[]))
        monkeypatch.setattr(mpf, "build_swap_plan", lambda *a, **k: SimpleNamespace())
        monkeypatch.setattr(mpf, "build_mask_plan", lambda *a, **k: SimpleNamespace())
        monkeypatch.setattr(mpf, "build_keyed_param_masks", lambda *a, **k: {})
        monkeypatch.setattr(mpf, "save_checkpoint", lambda *a, **k: None)
        monkeypatch.setattr(mpf, "load_from_disk", lambda _p: _tiny_dataset_dict())
        monkeypatch.setattr(mpf, "AutoTokenizer",
                            SimpleNamespace(from_pretrained=lambda *a, **k:
                                            SimpleNamespace(eos_token="<eos>",
                                                            pad_token="<eos>")))
        monkeypatch.setattr(mpf, "DataCollatorForLanguageModeling",
                            lambda *a, **k: _DummyCollator())
        monkeypatch.setattr(mpf, "wandb", _DummyWandb())

        # Avoid distributed-mode activation in the test process.
        monkeypatch.delenv("LOCAL_RANK", raising=False)

        def fake_train_step(*args, **kwargs):
            events.append("train")
            return 1.0, 0.1, 0.1, 0.0

        def fake_evaluate_on_dataset(*args, **kwargs):
            events.append("eval")
            if kwargs.get("eval_c2", False):
                return {"loss_c1": 1.0, "ppl_c1": 2.0, "acc_c1": 0.0,
                        "loss_c2": 1.0, "ppl_c2": 2.0, "acc_c2": 0.0}
            return {"loss_c1": 1.0, "ppl_c1": 2.0, "acc_c1": 0.0}

        monkeypatch.setattr(mpf, "train_step", fake_train_step)
        monkeypatch.setattr(mpf, "evaluate_on_dataset", fake_evaluate_on_dataset)

        mpf.main()

        assert "eval" in events and "train" in events
        assert events[0] == "eval", \
            "Step-0 validation must run before the first train_step"
        assert events.index("eval") < events.index("train")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
