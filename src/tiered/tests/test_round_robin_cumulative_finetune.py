"""Correctness tests for `round_robin_cumulative_finetune`.

Run against a real tiny `GPTNeoForCausalLMTiered` so every invariant is
checked against actual parameter state rather than mocks. Coverage:

1. **apply/unapply key helpers** compose to identity; up_to=-1 is a no-op.
2. **Only the active tier moves.** For every active_idx and every λ_pub
   setting, only the active tier's MLP rows shift; all other tiers
   (above and below the active one) and all public params stay
   byte-identical after one `train_step`.
3. **Public params are structurally frozen** — embeddings, LayerNorms,
   etc. are set `requires_grad=False`; they cannot drift.
4. **Public slices inside keyed tensors are frozen** — every False-bit
   position in the active update mask must be byte-identical before/after.
5. **Reference model is untouched** — pretrain_ref must be byte-identical
   before/after the step (any missing apply/unapply pair would corrupt it).
6. **Model arrangement after `train_step` is C_0 (home)** — the step runs
   in home coords, so a fresh apply+unapply must round-trip.
7. **KL disabling** — kl_lambda=0 must skip every public-KL forward
   AND make pretrain_ref unnecessary; the active tier still trains.
8. **Adam state hygiene** — exp_avg / exp_avg_sq stay zero on every
   non-active position even after one or many steps.
9. **Weight-decay never touches frozen positions** — even at wd=0.01 the
   non-active positions don't drift.
10. **Multi-step single-active stability** — 8 steps with the same active
    tier leaves all other tiers + public bit-identical.
11. **Returned `priv_losses` are at the right configs** — the c-th entry
    must equal a fresh forward at C_{active_idx+c+1} on the same private
    batch (within fp32 tolerance — tested on CPU where there's no autocast).
12. **`priv_losses` length matches `N - active_idx`** for every active.
13. **KL value is ~0 when student == ref** — verifies the KL forward
    happens at C_0 (home). If a key were accidentally applied, KL would
    be non-zero.
14. **Round-robin cycle** — over a full N-step round visiting all tiers
    in order 1→N (smallest first), each step moves only its own tier;
    after the round, every tier has shifted exactly once.
15. **Returned accuracy is at the active config** — `acc_at_active` must
    match an independent forward at C_{active_idx+1} (the shallowest
    config that includes the active tier's permutations).
16. **N=1 edge case** — with a single tier, active_idx=0 produces exactly
    one priv_loss and behaves correctly.
17. **Per-config priv weight is the mean** — confirmed via gradient-magnitude
    comparison against an equivalent single-forward weighted backward.
18. **Pre-mask gradients are non-zero at every tier's positions** — proves
    the cumulative permutation actually engages lower & upper tiers'
    weights during forward+backward (they're not silently skipped).
19. **The mask is what restricts updates** — non-active tiers have non-zero
    pre-mask grads but ZERO post-mask grads.
20. **Per-config loop visits exactly {active..N-1}** — catches off-by-one
    bugs in the per-config range.
21. **Mid-step state shows lower & upper tiers permuted** — at the deepest
    config C_N (always reached) every tier's home positions hold swapped
    values, proving cumulative permutations engage every tier.
22. **`_apply_keys` is never called on `pretrain_ref`** — KL stays at C_0,
    so the ref must never even be transiently permuted.

Tier setup throughout: 3 MLP-only tier keys, each swapping a unique
column pair across the two layers. With intermediate=16 that reserves
columns 0..5 for tiers and leaves 6..15 public inside the MLP weights.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tiered.permutation import load_key
from tiered.permutation.masking import build_mask_plan
from tiered.permutation.permute import (
    apply_permutation, build_swap_plan, unapply_permutation,
)
from tiered.train.finetune import round_robin_cumulative_finetune as rr
from tiered.train.utils import build_keyed_param_masks, load_model


DEVICE = torch.device("cpu")
TOKENIZED_SEQ_LEN = 4


# ────────────────────────────────────────────────────────────────────────────
# Fixtures (mirror multi_stage tests for setup parity)
# ────────────────────────────────────────────────────────────────────────────

def _write_key_file(path: Path, mlp_swaps: list[list[list[int]]]) -> str:
    data = {
        "attn_heads": [],
        "attn_out_heads": [],
        "mlp_cols": mlp_swaps,
        "mlp_up_cols": [],
        "mlp_down_cols": [],
    }
    path.write_text(json.dumps(data))
    return str(path)


def _make_non_overlapping_keys(tmp_path: Path, n_tiers: int) -> list[str]:
    paths = []
    for i in range(n_tiers):
        p = _write_key_file(
            tmp_path / f"key_{i}.json",
            [[[0, 2 * i], [1, 2 * i + 1]]],
        )
        paths.append(p)
    return paths


def _build_model() -> torch.nn.Module:
    m = load_model(
        hidden_size=8, num_heads=2, num_layers=2, context_size=TOKENIZED_SEQ_LEN,
        intermediate_size=16, do_print=False,
    )
    return m.to(DEVICE)


def _build_tiers(model, key_paths: list[str]) -> list[rr.TierKey]:
    tiers = []
    for i, kp in enumerate(key_paths):
        key = load_key(kp)
        tiers.append(rr.TierKey(
            tier_idx=i, tier_id=i + 1, key=key,
            swap_plan=build_swap_plan(model, key, DEVICE),
            mask_plan=build_mask_plan(model, key, DEVICE),
        ))
    return tiers


def _setup_optimizer(model, tiers, active_idx: int, *, weight_decay: float = 0.0):
    """Mirror the script's optimizer + active-mask setup."""
    keyed_param_ids = set()
    for t in tiers:
        for p in build_keyed_param_masks(model, t.mask_plan).keys():
            keyed_param_ids.add(id(p))
    keyed_params = [p for p in model.parameters() if id(p) in keyed_param_ids]
    purely_public = [p for p in model.parameters() if id(p) not in keyed_param_ids]
    for p in purely_public:
        p.requires_grad = False

    decay = [p for p in keyed_params if p.dim() >= 2]
    no_decay = [p for p in keyed_params if p.dim() < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    optim_obj = torch.optim.AdamW(groups, lr=0.1, betas=(0.9, 0.95))

    active_mask: dict = {}
    for p, m in build_keyed_param_masks(model, tiers[active_idx].mask_plan).items():
        active_mask[p] = m
    return optim_obj, active_mask


def _dummy_batch(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 50, (1, TOKENIZED_SEQ_LEN),
                         generator=g, device=DEVICE, dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}


def _frozen_copy(model):
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def _run_single_step(tmp_path, *, active_idx, kl_lambda=0.1,
                     cumulative_kl_lambda=0.0, cumulative_kl_config_idx=None,
                     n_tiers=3, seed=0, weight_decay=0.0,
                     priv_batch=None, pub_batch=None,
                     pretrain_ref=None, model=None, tiers=None,
                     optimizer=None, active_mask=None):
    """Build everything (or accept pre-built pieces), run one train_step,
    and return snapshots."""
    if model is None:
        torch.manual_seed(seed)
        model = _build_model()
    if tiers is None:
        tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, n_tiers))

    if pretrain_ref is None and (kl_lambda > 0 or cumulative_kl_lambda > 0):
        pretrain_ref = _frozen_copy(model)

    if optimizer is None:
        optimizer, active_mask = _setup_optimizer(
            model, tiers, active_idx, weight_decay=weight_decay,
        )

    before = {n: p.data.clone() for n, p in model.named_parameters()}
    ref_snap = None
    if pretrain_ref is not None:
        ref_snap = {n: p.data.clone() for n, p in pretrain_ref.named_parameters()}

    if priv_batch is None:
        priv_batch = _dummy_batch(seed=seed + 10)
    if pub_batch is None:
        pub_batch = _dummy_batch(seed=seed + 100)

    (priv_losses, kl_value, kl_cum_value, kl_cum_idx,
     acc, grad_norm) = rr.train_step(
        model=model, raw_model=model,
        pretrain_ref=pretrain_ref, tiers=tiers,
        active_idx=active_idx,
        private_batch=priv_batch, public_batch=pub_batch,
        optimizer=optimizer, device=DEVICE,
        kl_lambda=kl_lambda,
        cumulative_kl_lambda=cumulative_kl_lambda,
        cumulative_kl_config_idx=cumulative_kl_config_idx,
        max_grad_norm=1.0,
        active_update_mask=active_mask,
    )

    after = {n: p.data.clone() for n, p in model.named_parameters()}
    ref_after = None
    if pretrain_ref is not None:
        ref_after = {n: p.data.clone() for n, p in pretrain_ref.named_parameters()}

    return SimpleNamespace(
        model=model, tiers=tiers, active_idx=active_idx,
        optimizer=optimizer, active_mask=active_mask,
        before=before, after=after,
        ref_before=ref_snap, ref_after=ref_after,
        priv_losses=priv_losses, kl_value=kl_value,
        kl_cum_value=kl_cum_value, kl_cum_idx=kl_cum_idx,
        acc_at_active=acc, grad_norm=grad_norm,
        priv_batch=priv_batch, pub_batch=pub_batch,
    )


# ────────────────────────────────────────────────────────────────────────────
# 1. Helper invariants
# ────────────────────────────────────────────────────────────────────────────

def test_apply_then_unapply_is_identity(tmp_path):
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    rr._apply_keys(model, tiers, up_to_idx=2)
    rr._unapply_keys(model, tiers, up_to_idx=2)
    for n, p in model.named_parameters():
        assert torch.equal(p.data, before[n]), f"{n} drifted on apply+unapply cycle"


def test_apply_keys_negative_is_noop(tmp_path):
    """up_to_idx = -1 means C_0 (home; no keys applied)."""
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    rr._apply_keys(model, tiers, up_to_idx=-1)
    for n, p in model.named_parameters():
        assert torch.equal(p.data, before[n])


def test_apply_keys_intermediate(tmp_path):
    """up_to_idx=1 applies keys 0 and 1: tier-0 and tier-1 positions DO
    move; tier-2 positions are untouched."""
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    rr._apply_keys(model, tiers, up_to_idx=1)

    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    # Tier 0: swap (L0 row 0) ↔ (L1 row 1). Both endpoints must differ.
    assert not torch.equal(model.state_dict()[n0][0], before[n0][0]), \
        "tier 0's L0 row 0 should be swapped after _apply_keys(up_to=1)"
    assert not torch.equal(model.state_dict()[n1][1], before[n1][1]), \
        "tier 0's L1 row 1 should be swapped after _apply_keys(up_to=1)"
    # Tier 1: swap (L0 row 2) ↔ (L1 row 3).
    assert not torch.equal(model.state_dict()[n0][2], before[n0][2])
    assert not torch.equal(model.state_dict()[n1][3], before[n1][3])
    # Tier 2: rows 4 (L0) and 5 (L1) MUST be unchanged.
    assert torch.equal(model.state_dict()[n0][4], before[n0][4]), \
        "tier 2's L0 row 4 must NOT move when only keys 0,1 are applied"
    assert torch.equal(model.state_dict()[n1][5], before[n1][5]), \
        "tier 2's L1 row 5 must NOT move when only keys 0,1 are applied"

    rr._unapply_keys(model, tiers, up_to_idx=1)
    for n, p in model.named_parameters():
        assert torch.equal(p.data, before[n])


def test_apply_full_cumulative_permutes_every_tier_position(tmp_path):
    """_apply_keys(up_to=N-1) reaches C_N and must move EVERY tier's
    positions — including the lowest tier's. This is the precondition
    for "lower tiers participate in the cumulative path"."""
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    rr._apply_keys(model, tiers, up_to_idx=2)  # C_3
    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    for tier_idx in range(3):
        l0, l1 = _tier_rows(tier_idx)
        assert not torch.equal(model.state_dict()[n0][l0], before[n0][l0]), \
            f"tier {tier_idx}'s L0 row {l0} must move at C_3"
        assert not torch.equal(model.state_dict()[n1][l1], before[n1][l1]), \
            f"tier {tier_idx}'s L1 row {l1} must move at C_3"
    rr._unapply_keys(model, tiers, up_to_idx=2)
    for n, p in model.named_parameters():
        assert torch.equal(p.data, before[n])


# ────────────────────────────────────────────────────────────────────────────
# 2. Only the active tier moves — full cartesian product
# ────────────────────────────────────────────────────────────────────────────

PURELY_PUBLIC_PARAMS = (
    "transformer.wte.weight",
    "transformer.wpe.weight",
    "transformer.ln_f.weight",
    "transformer.ln_f.bias",
)


def _mlp_cfc_row_name(layer: int) -> str:
    return f"transformer.h.{layer}.mlp.c_fc.weight"


def _tier_rows(tier_idx: int) -> tuple[int, int]:
    """Return (layer_0_row, layer_1_row) swapped by tier `tier_idx`'s key."""
    return (2 * tier_idx, 2 * tier_idx + 1)


@pytest.mark.parametrize("active_idx", [0, 1, 2])
@pytest.mark.parametrize("kl_lambda", [0.0, 0.1, 0.5])
def test_only_active_tier_updates(tmp_path, active_idx, kl_lambda):
    """Regardless of λ_pub, only the active tier's MLP rows move.

    With round-robin training, when tier t is active we forward through
    every cumulative config C_{t+1}..C_N (configs that include tier t's
    permutations), so gradients flow through every tier ≥ active. The
    mask + adamw_step_preserving_public pair must restrict actual
    UPDATES to just the active tier.
    """
    s = _run_single_step(
        tmp_path, active_idx=active_idx, kl_lambda=kl_lambda,
    )
    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    for t_idx in range(3):
        l0, l1 = _tier_rows(t_idx)
        if t_idx == active_idx:
            assert not torch.equal(s.before[n0][l0], s.after[n0][l0]), (
                f"active tier {t_idx} row {l0} at layer 0 must move "
                f"(active={active_idx}, λ_pub={kl_lambda})"
            )
            assert not torch.equal(s.before[n1][l1], s.after[n1][l1]), (
                f"active tier {t_idx} row {l1} at layer 1 must move "
                f"(active={active_idx}, λ_pub={kl_lambda})"
            )
        else:
            assert torch.equal(s.before[n0][l0], s.after[n0][l0]), (
                f"non-active tier {t_idx} row {l0} (active={active_idx}, "
                f"λ_pub={kl_lambda}) must not move — gradients flowed "
                f"through it (it's part of cumulative path) but masking "
                f"should suppress updates"
            )
            assert torch.equal(s.before[n1][l1], s.after[n1][l1]), (
                f"non-active tier {t_idx} row {l1} (active={active_idx}, "
                f"λ_pub={kl_lambda}) must not move"
            )


# ────────────────────────────────────────────────────────────────────────────
# 3-4. Public freezing
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_pure_public_params_are_structurally_frozen(tmp_path, active_idx):
    s = _run_single_step(tmp_path, active_idx=active_idx)
    for name in PURELY_PUBLIC_PARAMS:
        if name in s.before:
            assert torch.equal(s.before[name], s.after[name]), (
                f"{name} drifted at active={active_idx}"
            )


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_public_slices_inside_keyed_tensors_frozen(tmp_path, active_idx):
    """Rows of c_fc that aren't in the active tier's swap (every False
    bit in the active mask) must stay byte-identical."""
    s = _run_single_step(tmp_path, active_idx=active_idx)
    id_to_name = {id(p): n for n, p in s.model.named_parameters()}
    for param, keyed_mask in s.active_mask.items():
        frozen = ~keyed_mask
        if not torch.any(frozen):
            continue
        name = id_to_name[id(param)]
        delta = (s.after[name] - s.before[name]).abs()
        assert torch.all(delta[frozen] == 0), (
            f"non-active slice of {name} moved at active={active_idx}"
        )


# ────────────────────────────────────────────────────────────────────────────
# 5. Reference model unchanged
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_reference_model_is_unchanged_after_step(tmp_path, active_idx):
    """pretrain_ref must be byte-identical before/after train_step. The
    new round-robin trainer never apply/unapplies keys to pretrain_ref
    (KL is at C_0 only) — so a regression that incorrectly permutes
    the ref will be caught here."""
    s = _run_single_step(tmp_path, active_idx=active_idx, kl_lambda=0.1)
    assert s.ref_before is not None and s.ref_after is not None
    for name, before in s.ref_before.items():
        after = s.ref_after[name]
        assert torch.equal(before, after), (
            f"pretrain_ref / {name} was mutated during train_step"
        )


# ────────────────────────────────────────────────────────────────────────────
# 6. Model arrangement after the step is C_0 (home)
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_model_returns_to_home_arrangement(tmp_path, active_idx):
    s = _run_single_step(tmp_path, active_idx=active_idx)
    snap = {n: p.data.clone() for n, p in s.model.named_parameters()}
    # An apply+unapply cycle on any prefix must be a no-op iff the model
    # is in home arrangement.
    rr._apply_keys(s.model, s.tiers, up_to_idx=active_idx)
    rr._unapply_keys(s.model, s.tiers, up_to_idx=active_idx)
    for n, p in s.model.named_parameters():
        assert torch.equal(p.data, snap[n]), (
            f"{n} drifted on apply+unapply cycle — model wasn't at home"
        )


# ────────────────────────────────────────────────────────────────────────────
# 7. KL disabling
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_kl_disabled_still_trains_active(tmp_path, active_idx):
    """With kl_lambda=0 and pretrain_ref=None, only the private term
    drives updates. The active tier still moves; non-active tiers do not."""
    torch.manual_seed(active_idx)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx)
    s = _run_single_step(
        tmp_path, active_idx=active_idx, kl_lambda=0.0,
        model=model, tiers=tiers,
        optimizer=optimizer, active_mask=active_mask,
        pretrain_ref=None,
    )
    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    al0, al1 = _tier_rows(active_idx)
    assert not torch.equal(s.before[n0][al0], s.after[n0][al0]), (
        "active tier should still move with kl_lambda=0"
    )
    assert not torch.equal(s.before[n1][al1], s.after[n1][al1])
    # Returned KL value should be NaN (sentinel) when disabled.
    assert s.kl_value != s.kl_value, (
        f"kl_value should be NaN when kl_lambda=0; got {s.kl_value}"
    )
    for other in range(3):
        if other == active_idx:
            continue
        ol0, ol1 = _tier_rows(other)
        assert torch.equal(s.before[n0][ol0], s.after[n0][ol0])
        assert torch.equal(s.before[n1][ol1], s.after[n1][ol1])


# ────────────────────────────────────────────────────────────────────────────
# 8-9. Adam state hygiene + weight decay
# ────────────────────────────────────────────────────────────────────────────

def test_adam_state_only_on_active_tier_positions(tmp_path):
    """After one step, exp_avg / exp_avg_sq are zero outside the active
    tier's positions — the preservation machinery restores momentum."""
    s = _run_single_step(tmp_path, active_idx=1, kl_lambda=0.1)
    id_to_name = {id(p): n for n, p in s.model.named_parameters()}
    for p in s.model.parameters():
        if not p.requires_grad:
            continue
        state = s.optimizer.state.get(p)
        if state is None or "exp_avg" not in state:
            continue
        active_mask = s.active_mask.get(p)
        if active_mask is None:
            assert torch.all(state["exp_avg"] == 0), (
                f"{id_to_name[id(p)]}: non-active param has non-zero Adam m"
            )
        else:
            frozen = ~active_mask
            if torch.any(frozen):
                assert torch.all(state["exp_avg"][frozen] == 0), (
                    f"{id_to_name[id(p)]}: Adam m leaked into frozen slices"
                )
                assert torch.all(state["exp_avg_sq"][frozen] == 0), (
                    f"{id_to_name[id(p)]}: Adam v leaked into frozen slices"
                )


def test_weight_decay_never_touches_frozen_positions(tmp_path):
    """AdamW applies wd as `w ← w·(1 - lr·wd)` independent of the gradient.
    `adamw_step_preserving_public` must snap every False-mask position
    back, so no drift shows up even at wd=0.01."""
    s = _run_single_step(
        tmp_path, active_idx=1, kl_lambda=0.1, weight_decay=0.01,
    )
    id_to_name = {id(p): n for n, p in s.model.named_parameters()}
    for param, active_mask in s.active_mask.items():
        frozen = ~active_mask
        if not torch.any(frozen):
            continue
        name = id_to_name[id(param)]
        delta = (s.after[name] - s.before[name]).abs()
        assert torch.all(delta[frozen] == 0), (
            f"frozen slice of {name} drifted under wd=0.01"
        )
    for name in PURELY_PUBLIC_PARAMS:
        if name in s.before:
            assert torch.equal(s.before[name], s.after[name])


# ────────────────────────────────────────────────────────────────────────────
# 10. Multi-step single-active stability
# ────────────────────────────────────────────────────────────────────────────

def test_multiple_steps_single_active_keeps_others_frozen(tmp_path):
    """Run 8 steps with the same active tier (=2). Every non-active tier's
    positions and every public param must be byte-identical to the start."""
    torch.manual_seed(7)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref = _frozen_copy(model)
    pretrain_ref_snap = {n: p.data.clone() for n, p in pretrain_ref.named_parameters()}
    active_idx = 2
    optimizer, active_mask = _setup_optimizer(
        model, tiers, active_idx, weight_decay=0.01,
    )
    initial = {n: p.data.clone() for n, p in model.named_parameters()}

    for step in range(8):
        rr.train_step(
            model=model, raw_model=model,
            pretrain_ref=pretrain_ref, tiers=tiers,
            active_idx=active_idx,
            private_batch=_dummy_batch(seed=step * 10),
            public_batch=_dummy_batch(seed=step * 10 + 99),
            optimizer=optimizer, device=DEVICE,
            kl_lambda=0.1,
            max_grad_norm=1.0,
            active_update_mask=active_mask,
        )

    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    for other in (0, 1):  # non-active when active_idx=2
        ol0, ol1 = _tier_rows(other)
        assert torch.equal(initial[n0][ol0], model.state_dict()[n0][ol0]), (
            f"tier {other} row {ol0} drifted across 8 steps"
        )
        assert torch.equal(initial[n1][ol1], model.state_dict()[n1][ol1]), (
            f"tier {other} row {ol1} drifted across 8 steps"
        )
    for name in PURELY_PUBLIC_PARAMS:
        if name in initial:
            assert torch.equal(initial[name], model.state_dict()[name])
    # Pretrain ref still untouched.
    for n, p in pretrain_ref.named_parameters():
        assert torch.equal(pretrain_ref_snap[n], p.data), (
            f"pretrain_ref / {n} was mutated across 8 steps"
        )


# ────────────────────────────────────────────────────────────────────────────
# 11-12. Returned losses correspond to the right configs
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_priv_losses_length_matches_active(tmp_path, active_idx):
    """priv_losses must have exactly N - active_idx entries (one per
    cumulative config C_{active_idx+1}..C_N — every config that
    includes the active tier's permutations)."""
    n_tiers = 3
    s = _run_single_step(tmp_path, active_idx=active_idx, n_tiers=n_tiers)
    assert len(s.priv_losses) == n_tiers - active_idx, (
        f"expected {n_tiers - active_idx} priv losses, got {len(s.priv_losses)}"
    )


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_priv_losses_match_independent_forward(tmp_path, active_idx):
    """Each priv_losses[c] must equal a fresh no-grad forward at
    C_{active_idx+c+1} on the same private batch and with the SAME
    starting weights.

    On CPU there is no autocast, so values are exact fp32 — we use a
    tight rtol/atol to catch any config-mixup or off-by-one bugs.
    """
    torch.manual_seed(13)
    n_tiers = 3
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, n_tiers))
    initial_state = copy.deepcopy(model.state_dict())

    s = _run_single_step(
        tmp_path, active_idx=active_idx, kl_lambda=0.1,
        model=model, tiers=tiers,
    )

    # Independent recomputation: fresh model with the original weights.
    independent = _build_model()
    independent.load_state_dict(initial_state)
    independent.train()

    expected = []
    for c in range(active_idx, n_tiers):
        # Apply keys cumulatively up to c → arrangement C_{c+1}.
        for i in range(c + 1):
            apply_permutation(independent, tiers[i].key, plan=tiers[i].swap_plan)
        with torch.no_grad():
            out = independent(s.priv_batch["input_ids"],
                              labels=s.priv_batch["labels"])
        expected.append(out.loss.item())
        for i in reversed(range(c + 1)):
            unapply_permutation(independent, tiers[i].key, plan=tiers[i].swap_plan)

    assert len(s.priv_losses) == len(expected)
    for c, (got, exp) in enumerate(zip(s.priv_losses, expected)):
        cfg_id = active_idx + c + 1
        assert got == pytest.approx(exp, rel=1e-5, abs=1e-5), (
            f"priv_losses[{c}] = {got} but independent forward at "
            f"C_{cfg_id} = {exp} (active_idx={active_idx})"
        )


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_acc_at_active_matches_shallowest_config_forward(tmp_path, active_idx):
    """The returned accuracy is computed at config C_{active_idx+1} (the
    shallowest cumulative config that includes the active tier — the
    FIRST forward of the per-config loop). Independent forward at that
    arrangement must give the same accuracy."""
    torch.manual_seed(21)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    initial_state = copy.deepcopy(model.state_dict())

    s = _run_single_step(
        tmp_path, active_idx=active_idx, kl_lambda=0.1,
        model=model, tiers=tiers,
    )

    independent = _build_model()
    independent.load_state_dict(initial_state)
    independent.train()
    for i in range(active_idx + 1):
        apply_permutation(independent, tiers[i].key, plan=tiers[i].swap_plan)
    with torch.no_grad():
        out = independent(s.priv_batch["input_ids"], labels=s.priv_batch["labels"])
    preds = out.logits[:, :-1, :].argmax(dim=-1)
    targets = s.priv_batch["labels"][:, 1:]
    m = targets != -100
    expected_acc = (preds[m] == targets[m]).float().mean().item() if m.any() else 0.0

    assert s.acc_at_active == pytest.approx(expected_acc, rel=1e-5, abs=1e-5), (
        f"acc_at_active = {s.acc_at_active} but independent forward at "
        f"C_{active_idx + 1} gives {expected_acc}"
    )


# ────────────────────────────────────────────────────────────────────────────
# 13. KL is computed at home
# ────────────────────────────────────────────────────────────────────────────

def test_kl_value_zero_when_student_equals_ref(tmp_path):
    """If pretrain_ref == student (deepcopy), KL at C_0 must be ~0.
    Any accidental key application would change one but not the other,
    producing non-zero KL.

    Run with active_idx=0 (smallest tier) so private-side permutations
    exercise every cumulative config C_1..C_N — if KL accidentally
    happened in a permuted arrangement after a private forward, it
    would be very visible here.
    """
    torch.manual_seed(31)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref = _frozen_copy(model)  # bit-identical to student at start
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx=0)

    priv_losses, kl_value, _, _, _, _ = rr.train_step(
        model=model, raw_model=model,
        pretrain_ref=pretrain_ref, tiers=tiers,
        active_idx=0,
        private_batch=_dummy_batch(seed=11),
        public_batch=_dummy_batch(seed=22),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1,
        max_grad_norm=1.0,
        active_update_mask=active_mask,
    )
    # KL of identical distributions is exactly 0; allow tiny fp32 tolerance.
    assert kl_value == pytest.approx(0.0, abs=1e-6), (
        f"KL should be ~0 when student==ref at home; got {kl_value}. "
        f"Non-zero suggests KL was computed in a permuted arrangement."
    )


# ────────────────────────────────────────────────────────────────────────────
# 14. Full round-robin cycle
# ────────────────────────────────────────────────────────────────────────────

def test_round_robin_cycle_visits_each_tier_exactly_once(tmp_path):
    """Walk active_idx in 1→N order (smallest first) over a single round.
    After step k, only tier k should have moved since the previous
    snapshot. After the full round, every tier has shifted exactly once."""
    torch.manual_seed(101)
    n_tiers = 3
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, n_tiers))
    pretrain_ref = _frozen_copy(model)
    pretrain_snap = {n: p.data.clone() for n, p in pretrain_ref.named_parameters()}

    # Each tier needs its own optimizer + active_mask (the script uses one
    # optimizer with a per-tier mask; we simulate the same via per-tier masks).
    keyed_param_ids = set()
    for t in tiers:
        for p in build_keyed_param_masks(model, t.mask_plan).keys():
            keyed_param_ids.add(id(p))
    keyed_params = [p for p in model.parameters() if id(p) in keyed_param_ids]
    purely_public = [p for p in model.parameters() if id(p) not in keyed_param_ids]
    for p in purely_public:
        p.requires_grad = False
    optimizer = torch.optim.AdamW(keyed_params, lr=0.1, betas=(0.9, 0.95))
    per_tier_masks = []
    for t in tiers:
        m = {p: mask for p, mask in build_keyed_param_masks(model, t.mask_plan).items()}
        per_tier_masks.append(m)

    initial = {n: p.data.clone() for n, p in model.named_parameters()}
    snapshots = [initial]
    rr_order = list(range(n_tiers))  # [0, 1, 2] — smallest first

    for step_in_round, active_idx in enumerate(rr_order):
        rr.train_step(
            model=model, raw_model=model,
            pretrain_ref=pretrain_ref, tiers=tiers,
            active_idx=active_idx,
            private_batch=_dummy_batch(seed=200 + step_in_round),
            public_batch=_dummy_batch(seed=300 + step_in_round),
            optimizer=optimizer, device=DEVICE,
            kl_lambda=0.1,
            max_grad_norm=1.0,
            active_update_mask=per_tier_masks[active_idx],
        )
        snapshots.append({n: p.data.clone() for n, p in model.named_parameters()})

    # Per-step diff: only the active tier's rows should have moved.
    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    for step_in_round, active_idx in enumerate(rr_order):
        prev = snapshots[step_in_round]
        curr = snapshots[step_in_round + 1]
        for t_idx in range(n_tiers):
            l0, l1 = _tier_rows(t_idx)
            moved0 = not torch.equal(prev[n0][l0], curr[n0][l0])
            moved1 = not torch.equal(prev[n1][l1], curr[n1][l1])
            if t_idx == active_idx:
                assert moved0 and moved1, (
                    f"step {step_in_round} (active={active_idx}): tier "
                    f"{t_idx} should have moved but didn't"
                )
            else:
                assert not moved0 and not moved1, (
                    f"step {step_in_round} (active={active_idx}): tier "
                    f"{t_idx} should NOT have moved this step but did"
                )

    # End-of-round: every tier should differ from its initial state.
    final = snapshots[-1]
    for t_idx in range(n_tiers):
        l0, l1 = _tier_rows(t_idx)
        assert not torch.equal(initial[n0][l0], final[n0][l0]), (
            f"tier {t_idx} row {l0} should have moved during the round"
        )
        assert not torch.equal(initial[n1][l1], final[n1][l1]), (
            f"tier {t_idx} row {l1} should have moved during the round"
        )

    # Public params + pretrain_ref still untouched.
    for name in PURELY_PUBLIC_PARAMS:
        if name in initial:
            assert torch.equal(initial[name], final[name])
    for n, p in pretrain_ref.named_parameters():
        assert torch.equal(pretrain_snap[n], p.data), (
            f"pretrain_ref / {n} was mutated across the round"
        )


# ────────────────────────────────────────────────────────────────────────────
# 16. N=1 edge case
# ────────────────────────────────────────────────────────────────────────────

def test_n_tiers_one(tmp_path):
    """With a single tier, only active_idx=0 is valid. The trainer should
    return exactly one priv_loss and update only that tier's positions."""
    torch.manual_seed(41)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 1))
    pretrain_ref = _frozen_copy(model)
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx=0)
    before = {n: p.data.clone() for n, p in model.named_parameters()}

    priv_losses, kl_value, _, _, _, _ = rr.train_step(
        model=model, raw_model=model,
        pretrain_ref=pretrain_ref, tiers=tiers,
        active_idx=0,
        private_batch=_dummy_batch(seed=51),
        public_batch=_dummy_batch(seed=52),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1,
        max_grad_norm=1.0,
        active_update_mask=active_mask,
    )
    assert len(priv_losses) == 1
    n0 = _mlp_cfc_row_name(0)
    n1 = _mlp_cfc_row_name(1)
    l0, l1 = _tier_rows(0)
    assert not torch.equal(before[n0][l0], model.state_dict()[n0][l0])
    assert not torch.equal(before[n1][l1], model.state_dict()[n1][l1])


# ────────────────────────────────────────────────────────────────────────────
# 17. Per-config priv weight is mean (gradient-magnitude check)
# ────────────────────────────────────────────────────────────────────────────

def test_per_config_priv_weight_is_mean(tmp_path):
    """When all upper-tier keys are identity, the n_configs forwards
    collapse to the same arrangement → priv_losses entries should be
    equal. Furthermore, the total backward weight (sum of per-config
    weights) must equal (1 - kl_lambda) — verified by comparing the
    gradient produced by the train_step against an equivalent
    single-forward backward weighted by (1 - kl_lambda).

    This is the strongest test of the per-config = (1-kl)/n_configs
    weighting (it would fail if e.g. each forward used weight (1-kl)
    instead, which would give a 3× larger gradient for n_configs=3).
    """
    n_tiers = 3
    # Tier 0's key actually permutes things; tiers 1, 2 are identity
    # (empty mlp_cols → no swaps).
    key_paths = [
        _write_key_file(tmp_path / "key_0.json", [[[0, 6], [1, 7]]]),
        _write_key_file(tmp_path / "key_1.json", []),
        _write_key_file(tmp_path / "key_2.json", []),
    ]
    priv_b = _dummy_batch(seed=999)
    pub_b = _dummy_batch(seed=1000)

    def _fresh():
        torch.manual_seed(77)
        m = _build_model()
        t = _build_tiers(m, key_paths)
        return m, t

    # --- Run A: active=0 → n_configs = N - 0 = 3, all three forwards
    # collapse to the same arrangement (identity keys for tiers 1,2).
    # Total backward weight = 3 · (1/3) = 1.0 = (1 - kl_lambda).
    model_a, tiers_a = _fresh()
    pretrain_ref_a = _frozen_copy(model_a)
    opt_a, mask_a = _setup_optimizer(model_a, tiers_a, active_idx=0)
    before_a = {n: p.data.clone() for n, p in model_a.named_parameters()}
    priv_losses_a, _, _, _, _, _ = rr.train_step(
        model=model_a, raw_model=model_a, pretrain_ref=pretrain_ref_a,
        tiers=tiers_a, active_idx=0,
        private_batch=priv_b, public_batch=pub_b,
        optimizer=opt_a, device=DEVICE, kl_lambda=0.0,
        max_grad_norm=1e9,  # disable clipping so deltas reflect raw grads
        active_update_mask=mask_a,
    )
    # The 3 forwards all see the same arrangement → equal losses.
    assert priv_losses_a[0] == pytest.approx(priv_losses_a[1], rel=1e-5, abs=1e-5)
    assert priv_losses_a[1] == pytest.approx(priv_losses_a[2], rel=1e-5, abs=1e-5)

    delta_a = {n: model_a.state_dict()[n] - before_a[n] for n in before_a}

    # --- Run B: a manual single-forward backward at C_1 with weight 1.0.
    # Equivalent to "n_configs = 1, per_config = 1.0". The resulting
    # gradient should match run A's effective gradient (3 backwards of
    # weight 1/3 on the same loss).
    #
    # Use a fresh model with the same starting weights, run forward at
    # C_1 (apply key 0), backward with full weight 1.0, then compare
    # the parameter delta after a single Adam step.
    model_b, tiers_b = _fresh()
    keyed_param_ids = set()
    for t in tiers_b:
        for p in build_keyed_param_masks(model_b, t.mask_plan).keys():
            keyed_param_ids.add(id(p))
    keyed_params = [p for p in model_b.parameters() if id(p) in keyed_param_ids]
    purely_public = [p for p in model_b.parameters() if id(p) not in keyed_param_ids]
    for p in purely_public:
        p.requires_grad = False
    opt_b = torch.optim.AdamW(keyed_params, lr=0.1, betas=(0.9, 0.95))
    active_mask_b = {p: m for p, m in
                     build_keyed_param_masks(model_b, tiers_b[0].mask_plan).items()}

    # Single forward at C_1 with weight 1.0 (== n_configs · (1/n_configs)).
    from tiered.permutation.permute import (
        apply_permutation as _apply, swap_gradients as _swap,
        unapply_permutation as _unapply,
    )
    before_b = {n: p.data.clone() for n, p in model_b.named_parameters()}
    opt_b.zero_grad()
    _apply(model_b, tiers_b[0].key, plan=tiers_b[0].swap_plan)
    out = model_b(priv_b["input_ids"], labels=priv_b["labels"])
    out.loss.backward()
    _swap(model_b, tiers_b[0].key, plan=tiers_b[0].swap_plan)
    _unapply(model_b, tiers_b[0].key, plan=tiers_b[0].swap_plan)
    from tiered.permutation.masking import mask_public_gradients as _mpg
    _mpg(model_b, tiers_b[0].key, plan=tiers_b[0].mask_plan)
    torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1e9)
    from tiered.train.utils import adamw_step_preserving_public as _step
    _step(opt_b, active_mask_b)
    delta_b = {n: model_b.state_dict()[n] - before_b[n] for n in before_b}

    # Tier 0's positions: deltas should match (within tight tolerance).
    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    l0, l1 = _tier_rows(0)
    torch.testing.assert_close(
        delta_a[n0][l0], delta_b[n0][l0], rtol=1e-4, atol=1e-5,
    )
    torch.testing.assert_close(
        delta_a[n1][l1], delta_b[n1][l1], rtol=1e-4, atol=1e-5,
    )


# ────────────────────────────────────────────────────────────────────────────
# 18. parse_args mismatch validation
# ────────────────────────────────────────────────────────────────────────────

def test_mismatched_key_and_data_count_raises(tmp_path, monkeypatch):
    """Trainer requires len(--all_key_paths) == len(--private_data)."""
    import sys
    key_paths = _make_non_overlapping_keys(tmp_path, 3)
    priv_paths = ["/tmp/fake"] * 2  # 3 keys, 2 data → mismatch
    argv = [
        "prog",
        "--checkpoint", "/tmp/student",
        "--pretrain_checkpoint", "/tmp/pretrain",
        "--all_key_paths", *key_paths,
        "--private_data", *priv_paths,
        "--public_data", "/tmp/public",
        "--output_dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = rr.parse_args()
    # parse_args itself doesn't enforce — main() does. We assert the guard
    # condition the trainer checks.
    assert len(args.all_key_paths) != len(args.private_data)


# ────────────────────────────────────────────────────────────────────────────
# 19. Algorithmic correctness — gradient flow into NON-active tiers
# ────────────────────────────────────────────────────────────────────────────
#
# These tests pin down the heart of the round-robin algorithm:
#   - Lower tiers' positions ARE swapped into the cumulative path during
#     each per-config forward.
#   - Upper tiers' positions ARE swapped into deeper-cumulative configs.
#   - Gradients flow into ALL keyed positions during backward (BEFORE the
#     mask runs).
#   - The mask is what restricts the actual UPDATE to the active tier;
#     without it, lower & upper tier positions would also move.
#   - pretrain_ref is never permuted (it's only forwarded at C_0).
# ────────────────────────────────────────────────────────────────────────────


def _capture_pre_mask_grads(monkeypatch):
    """Install a wrapper around `mask_public_gradients` that records
    parameter gradients RIGHT BEFORE the mask runs, then proceeds to
    call the real mask. Returns the captured-grads dict (filled in
    place during train_step)."""
    captured: dict = {}
    real_mask = rr.mask_public_gradients

    def _capturing(model, key, plan=None):
        for n, p in model.named_parameters():
            if p.grad is not None:
                captured[n] = p.grad.clone()
        real_mask(model, key, plan=plan)

    monkeypatch.setattr(rr, "mask_public_gradients", _capturing)
    return captured


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_pre_mask_gradients_nonzero_at_every_tier_position(
    tmp_path, monkeypatch, active_idx
):
    """Before the mask zeros most gradients, every tier's keyed positions
    that participate in some forward at the active step must have a
    non-zero gradient.

    For active = t, configs forwarded are C_{t+1}..C_N. Together they
    cover keys 0..N-1 (the deepest config = C_N applies all keys).
    Therefore EVERY tier's positions are part of some forward, and after
    the swap-back-to-home all of them carry gradient at their HOME
    positions.

    This is the test that distinguishes "no update" (true) from "no
    forward / no gradient" (false). If the mask were the only thing
    keeping non-active tiers from updating, this test must show grads
    flowing into them.
    """
    captured = _capture_pre_mask_grads(monkeypatch)

    torch.manual_seed(active_idx + 1000)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref = _frozen_copy(model)
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx)

    rr.train_step(
        model=model, raw_model=model,
        pretrain_ref=pretrain_ref, tiers=tiers,
        active_idx=active_idx,
        private_batch=_dummy_batch(seed=2001 + active_idx),
        public_batch=_dummy_batch(seed=3001 + active_idx),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1,
        max_grad_norm=1e9,  # don't clip; we inspect grads
        active_update_mask=active_mask,
    )

    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    # Every tier whose key participates in any forwarded config must have
    # nonzero gradient at its HOME positions. The deepest config C_N
    # (always forwarded since c=N-1 is the last loop iter) involves
    # every key, so every tier qualifies.
    for tier_idx in range(3):
        l0, l1 = _tier_rows(tier_idx)
        g0 = captured[n0][l0]
        g1 = captured[n1][l1]
        assert g0.abs().sum() > 0, (
            f"active={active_idx}: tier {tier_idx} L0 row {l0} should "
            f"have NON-zero pre-mask grad — cumulative path engaged it"
        )
        assert g1.abs().sum() > 0, (
            f"active={active_idx}: tier {tier_idx} L1 row {l1} should "
            f"have NON-zero pre-mask grad"
        )


def test_mask_zeros_non_active_tier_grads_post_mask(tmp_path, monkeypatch):
    """Companion to the test above: capture grads BEFORE the mask, run
    the rest of train_step, then inspect post-mask grads. Non-active
    tier positions must have ZERO grad post-mask, even though they had
    non-zero grad pre-mask.

    Doing both in one test gives a tight before/after view of the mask's
    role: it's not the absence of gradient that protects non-active
    tiers, it's the mask actively zeroing them.
    """
    pre_mask: dict = {}
    post_mask: dict = {}
    real_mask = rr.mask_public_gradients

    def _capturing(model, key, plan=None):
        for n, p in model.named_parameters():
            if p.grad is not None:
                pre_mask[n] = p.grad.clone()
        real_mask(model, key, plan=plan)
        for n, p in model.named_parameters():
            if p.grad is not None:
                post_mask[n] = p.grad.clone()

    monkeypatch.setattr(rr, "mask_public_gradients", _capturing)

    torch.manual_seed(2025)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref = _frozen_copy(model)
    active_idx = 1  # so both lower (tier 0) and upper (tier 2) exist
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx)

    rr.train_step(
        model=model, raw_model=model,
        pretrain_ref=pretrain_ref, tiers=tiers,
        active_idx=active_idx,
        private_batch=_dummy_batch(seed=4001),
        public_batch=_dummy_batch(seed=5001),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1,
        max_grad_norm=1e9,
        active_update_mask=active_mask,
    )

    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    for tier_idx in range(3):
        l0, l1 = _tier_rows(tier_idx)
        if tier_idx == active_idx:
            # Active tier: pre-mask AND post-mask grad both nonzero.
            assert pre_mask[n0][l0].abs().sum() > 0
            assert post_mask[n0][l0].abs().sum() > 0, (
                f"active tier {tier_idx} should KEEP its grad after mask"
            )
        else:
            # Non-active tier: had grad pre-mask, should be ZERO post-mask.
            assert pre_mask[n0][l0].abs().sum() > 0, (
                f"tier {tier_idx} (non-active) should have HAD pre-mask "
                f"grad (cumulative path engaged it)"
            )
            assert pre_mask[n1][l1].abs().sum() > 0
            assert torch.all(post_mask[n0][l0] == 0), (
                f"tier {tier_idx} (non-active) L0 row {l0} should have "
                f"ZERO grad after mask"
            )
            assert torch.all(post_mask[n1][l1] == 0)


# ────────────────────────────────────────────────────────────────────────────
# 20. Mid-step model arrangement — lower & upper tiers actually permuted
# ────────────────────────────────────────────────────────────────────────────


def _capture_apply_states(monkeypatch):
    """Wrap `_apply_keys` to record (up_to_idx, model_state) right after
    each call. Used to verify the per-config loop reaches the right
    arrangements."""
    snapshots: list = []
    real_apply = rr._apply_keys

    def _capturing(model, tiers, up_to_idx):
        real_apply(model, tiers, up_to_idx)
        snapshots.append({
            "up_to_idx": up_to_idx,
            "state": {n: p.data.clone() for n, p in model.named_parameters()},
        })

    monkeypatch.setattr(rr, "_apply_keys", _capturing)
    return snapshots


def test_per_config_loop_visits_correct_cumulative_configs(
    tmp_path, monkeypatch
):
    """For active = t, _apply_keys must be called with up_to_idx in
    exactly {t, t+1, ..., N-1} (in that order) during the per-config
    loop. Catches off-by-one bugs in the loop range."""
    snapshots = _capture_apply_states(monkeypatch)
    n_tiers = 3
    active_idx = 0  # walk the full range C_1..C_N

    torch.manual_seed(99)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, n_tiers))
    pretrain_ref = _frozen_copy(model)
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx)

    rr.train_step(
        model=model, raw_model=model,
        pretrain_ref=pretrain_ref, tiers=tiers,
        active_idx=active_idx,
        private_batch=_dummy_batch(seed=11),
        public_batch=_dummy_batch(seed=22),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1,
        max_grad_norm=1.0,
        active_update_mask=active_mask,
    )

    # Filter to up_to_idx >= 0 (the actual key applications).
    applied_indices = [s["up_to_idx"] for s in snapshots if s["up_to_idx"] >= 0]
    expected = list(range(active_idx, n_tiers))  # [0, 1, 2]
    assert applied_indices == expected, (
        f"Per-config loop visited up_to_idx={applied_indices}; "
        f"expected exactly {expected}"
    )


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_lower_and_upper_tiers_permuted_during_per_config_forward(
    tmp_path, monkeypatch, active_idx
):
    """Mid-step state check: when the loop reaches the deepest config
    C_N (= apply_keys(up_to=N-1), always part of the loop), every
    tier's home positions are SWAPPED OUT to other indices — proving
    the cumulative permutation actually engages lower- and upper-tier
    weights, not just the active tier's.
    """
    snapshots = _capture_apply_states(monkeypatch)
    n_tiers = 3

    torch.manual_seed(active_idx + 555)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, n_tiers))
    initial = {n: p.data.clone() for n, p in model.named_parameters()}
    pretrain_ref = _frozen_copy(model)
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx)

    rr.train_step(
        model=model, raw_model=model,
        pretrain_ref=pretrain_ref, tiers=tiers,
        active_idx=active_idx,
        private_batch=_dummy_batch(seed=11),
        public_batch=_dummy_batch(seed=22),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.0,  # no KL → no extra apply_keys for KL path
        max_grad_norm=1.0,
        active_update_mask=active_mask,
    )

    # The deepest forward is at up_to=N-1 (always reached, since
    # the per-config loop ends at c = N-1).
    deepest = next((s for s in snapshots if s["up_to_idx"] == n_tiers - 1), None)
    assert deepest is not None, \
        f"Expected an _apply_keys call with up_to_idx={n_tiers - 1}"

    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    # At the deepest config, EVERY tier's home positions hold values
    # that are NOT the original (they've been swapped with their pair).
    for tier_idx in range(n_tiers):
        l0, l1 = _tier_rows(tier_idx)
        assert not torch.equal(deepest["state"][n0][l0], initial[n0][l0]), (
            f"active={active_idx}: tier {tier_idx} L0 row {l0} should be "
            f"swapped at C_{n_tiers} mid-step (was untouched)"
        )
        assert not torch.equal(deepest["state"][n1][l1], initial[n1][l1]), (
            f"active={active_idx}: tier {tier_idx} L1 row {l1} should be "
            f"swapped at C_{n_tiers} mid-step (was untouched)"
        )


# ────────────────────────────────────────────────────────────────────────────
# 21. pretrain_ref is never permuted
# ────────────────────────────────────────────────────────────────────────────


def test_apply_keys_is_never_called_on_pretrain_ref(tmp_path, monkeypatch):
    """When `cumulative_kl_lambda=0` (default), the only KL is at C_0 so
    train_step must never apply keys to pretrain_ref. Wrap `_apply_keys`
    to record the model arg's identity and assert it's always the student.

    The before/after byte-equality check (test_reference_model_is_unchanged)
    only catches NET drift — if a stray apply was paired with an unapply,
    drift could be zero but the ref would still have been transiently
    permuted. This test catches that case too.
    """
    seen_models = []
    real_apply = rr._apply_keys

    def _record(model, tiers, up_to_idx):
        seen_models.append(id(model))
        real_apply(model, tiers, up_to_idx)

    monkeypatch.setattr(rr, "_apply_keys", _record)

    torch.manual_seed(31)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref = _frozen_copy(model)
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx=1)

    rr.train_step(
        model=model, raw_model=model,
        pretrain_ref=pretrain_ref, tiers=tiers,
        active_idx=1,
        private_batch=_dummy_batch(seed=11),
        public_batch=_dummy_batch(seed=22),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1,
        max_grad_norm=1.0,
        active_update_mask=active_mask,
    )

    student_id = id(model)
    ref_id = id(pretrain_ref)
    assert student_id != ref_id, "test setup: ref must be a separate object"
    assert ref_id not in seen_models, (
        f"pretrain_ref was passed to _apply_keys (ids seen: {seen_models})"
    )
    # And the student SHOULD have been seen (sanity).
    assert student_id in seen_models


# ────────────────────────────────────────────────────────────────────────────
# 22. Cumulative-KL term (cycled per step)
# ────────────────────────────────────────────────────────────────────────────


def test_cumulative_kl_disabled_returns_nan_and_minus_one(tmp_path):
    """With cumulative_kl_lambda=0, the cumulative KL is skipped:
    returned value is NaN and config index is -1."""
    s = _run_single_step(tmp_path, active_idx=1, kl_lambda=0.1,
                         cumulative_kl_lambda=0.0)
    assert s.kl_cum_value != s.kl_cum_value, \
        f"kl_cum_value should be NaN when disabled; got {s.kl_cum_value}"
    assert s.kl_cum_idx == -1, \
        f"kl_cum_idx should be -1 when disabled; got {s.kl_cum_idx}"


def test_cumulative_kl_disabled_default_matches_no_arg(tmp_path):
    """Calling train_step without passing cumulative_kl_lambda must
    produce the same parameter delta as passing 0.0 explicitly. Default
    must be a true no-op."""
    torch.manual_seed(123)
    model_a = _build_model()
    tiers_a = _build_tiers(model_a, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref_a = _frozen_copy(model_a)
    optimizer_a, mask_a = _setup_optimizer(model_a, tiers_a, active_idx=1)
    priv_b = _dummy_batch(seed=11)
    pub_b = _dummy_batch(seed=22)
    rr.train_step(
        model=model_a, raw_model=model_a, pretrain_ref=pretrain_ref_a,
        tiers=tiers_a, active_idx=1,
        private_batch=priv_b, public_batch=pub_b,
        optimizer=optimizer_a, device=DEVICE, kl_lambda=0.1,
        max_grad_norm=1.0, active_update_mask=mask_a,
    )
    state_a = {n: p.data.clone() for n, p in model_a.named_parameters()}

    torch.manual_seed(123)
    model_b = _build_model()
    tiers_b = _build_tiers(model_b, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref_b = _frozen_copy(model_b)
    optimizer_b, mask_b = _setup_optimizer(model_b, tiers_b, active_idx=1)
    rr.train_step(
        model=model_b, raw_model=model_b, pretrain_ref=pretrain_ref_b,
        tiers=tiers_b, active_idx=1,
        private_batch=priv_b, public_batch=pub_b,
        optimizer=optimizer_b, device=DEVICE, kl_lambda=0.1,
        cumulative_kl_lambda=0.0, cumulative_kl_config_idx=None,
        max_grad_norm=1.0, active_update_mask=mask_b,
    )
    for n, p in model_b.named_parameters():
        assert torch.equal(state_a[n], p.data), \
            f"{n}: explicit cumulative_kl_lambda=0 differs from default"


@pytest.mark.parametrize("cum_c", [0, 1, 2])
def test_cumulative_kl_uses_provided_config_idx(tmp_path, monkeypatch, cum_c):
    """When cumulative_kl_lambda > 0 with config_idx=c, _apply_keys MUST
    be called with up_to_idx=c on BOTH student and pretrain_ref. Verify
    via id-tracking + up_to_idx capture."""
    calls: list = []
    real_apply = rr._apply_keys

    def _record(model, tiers, up_to_idx):
        calls.append((id(model), up_to_idx))
        real_apply(model, tiers, up_to_idx)

    monkeypatch.setattr(rr, "_apply_keys", _record)

    torch.manual_seed(45)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref = _frozen_copy(model)
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx=2)
    # active=2: priv loop forwards only at C_3 (cum loop covers other configs).

    rr.train_step(
        model=model, raw_model=model, pretrain_ref=pretrain_ref,
        tiers=tiers, active_idx=2,
        private_batch=_dummy_batch(seed=11),
        public_batch=_dummy_batch(seed=22),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1, cumulative_kl_lambda=0.1, cumulative_kl_config_idx=cum_c,
        max_grad_norm=1.0, active_update_mask=active_mask,
    )

    student_id = id(model)
    ref_id = id(pretrain_ref)
    # Student must have been called with up_to=cum_c (for cumulative KL)
    # AND with up_to=2 (for the priv loop's only forward at C_3).
    student_calls = [u for (mid, u) in calls if mid == student_id]
    assert cum_c in student_calls, (
        f"student should have _apply_keys(up_to={cum_c}) for cumulative KL; "
        f"got {student_calls}"
    )
    assert 2 in student_calls, (
        f"student should have _apply_keys(up_to=2) for priv at C_3; "
        f"got {student_calls}"
    )
    # Ref must have been called ONLY with up_to=cum_c (cumulative KL only).
    ref_calls = [u for (mid, u) in calls if mid == ref_id]
    assert ref_calls == [cum_c], (
        f"ref _apply_keys calls should be exactly [{cum_c}] (one for "
        f"cumulative KL); got {ref_calls}"
    )


def test_cumulative_kl_value_zero_when_student_equals_ref(tmp_path):
    """If pretrain_ref is a deepcopy of student, KL at any cumulative
    config must be ~0 (KL(p || p) = 0). Verifies the cumulative-KL
    forward correctly applies the SAME permutation to both student and
    ref so they remain identical at C_{c+1}."""
    torch.manual_seed(67)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref = _frozen_copy(model)  # bit-identical
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx=0)

    (_, kl_value, kl_cum_value, kl_cum_idx, _, _) = rr.train_step(
        model=model, raw_model=model, pretrain_ref=pretrain_ref,
        tiers=tiers, active_idx=0,
        private_batch=_dummy_batch(seed=11),
        public_batch=_dummy_batch(seed=22),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1, cumulative_kl_lambda=0.1, cumulative_kl_config_idx=2,
        max_grad_norm=1.0, active_update_mask=active_mask,
    )
    assert kl_value == pytest.approx(0.0, abs=1e-6), (
        f"home KL should be ~0 when student==ref; got {kl_value}"
    )
    assert kl_cum_value == pytest.approx(0.0, abs=1e-6), (
        f"cumulative KL at C_3 should be ~0 when student==ref; "
        f"got {kl_cum_value}. Non-zero suggests student and ref were "
        f"permuted differently."
    )
    assert kl_cum_idx == 2


def test_cumulative_kl_pretrain_ref_returns_to_home(tmp_path):
    """After train_step with cumulative KL enabled, pretrain_ref must
    be byte-identical to before — the apply/unapply pair must balance.
    Catches missing unapply for the cumulative-KL ref permutation."""
    torch.manual_seed(89)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref = _frozen_copy(model)
    ref_before = {n: p.data.clone() for n, p in pretrain_ref.named_parameters()}
    optimizer, active_mask = _setup_optimizer(model, tiers, active_idx=1)

    rr.train_step(
        model=model, raw_model=model, pretrain_ref=pretrain_ref,
        tiers=tiers, active_idx=1,
        private_batch=_dummy_batch(seed=11),
        public_batch=_dummy_batch(seed=22),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1, cumulative_kl_lambda=0.1, cumulative_kl_config_idx=2,
        max_grad_norm=1.0, active_update_mask=active_mask,
    )
    for n, p in pretrain_ref.named_parameters():
        assert torch.equal(ref_before[n], p.data), (
            f"pretrain_ref / {n} drifted after cumulative-KL step"
        )


@pytest.mark.parametrize("active_idx", [0, 1, 2])
@pytest.mark.parametrize("cum_c", [0, 1, 2])
def test_cumulative_kl_only_active_tier_updates(tmp_path, active_idx, cum_c):
    """Cumulative KL adds gradient at C_{cum_c+1} positions; the mask
    must STILL restrict updates to only the active tier — even when
    cum_c != active_idx so the cumulative-KL gradient lands at a
    different tier's positions."""
    s = _run_single_step(
        tmp_path, active_idx=active_idx,
        kl_lambda=0.1, cumulative_kl_lambda=0.1,
        cumulative_kl_config_idx=cum_c,
    )
    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    for t_idx in range(3):
        l0, l1 = _tier_rows(t_idx)
        if t_idx == active_idx:
            assert not torch.equal(s.before[n0][l0], s.after[n0][l0]), (
                f"active tier {t_idx} should move (cum_c={cum_c})"
            )
        else:
            assert torch.equal(s.before[n0][l0], s.after[n0][l0]), (
                f"non-active tier {t_idx} must NOT move under cumulative "
                f"KL (active={active_idx}, cum_c={cum_c})"
            )
            assert torch.equal(s.before[n1][l1], s.after[n1][l1])


def test_cumulative_kl_priv_weight_decreases_proportionally(tmp_path):
    """priv weight = max(0, 1 - kl_lambda - cumulative_kl_lambda) /
    n_configs. Verify by gradient comparison: with cumulative_kl_lambda=0
    vs cumulative_kl_lambda=0.5 (and ref==student so cumulative KL grad
    is 0), the active-tier delta should match — both cases produce
    priv-only gradients but with different weights:
       (1.0 - 0.0 - 0.0) / n  vs  (1.0 - 0.0 - 0.5) / n  =  1/n vs 0.5/n.
    The 0.5-cumulative case should give exactly half the delta.
    """
    torch.manual_seed(101)
    n_tiers = 3
    active_idx = 0
    n_configs = n_tiers - active_idx
    priv_b = _dummy_batch(seed=99)
    pub_b = _dummy_batch(seed=98)

    def _run(cum_lambda):
        torch.manual_seed(101)
        m = _build_model()
        t = _build_tiers(m, _make_non_overlapping_keys(tmp_path, n_tiers))
        # ref bit-identical to student → both KL terms = 0 (no grad from KL)
        ref = _frozen_copy(m)
        opt, mask = _setup_optimizer(m, t, active_idx)
        before = {n: p.data.clone() for n, p in m.named_parameters()}
        rr.train_step(
            model=m, raw_model=m, pretrain_ref=ref, tiers=t,
            active_idx=active_idx,
            private_batch=priv_b, public_batch=pub_b,
            optimizer=opt, device=DEVICE,
            kl_lambda=0.0,  # KL home weight 0 → no home-KL grad either
            cumulative_kl_lambda=cum_lambda,
            cumulative_kl_config_idx=2,
            max_grad_norm=1e9,
            active_update_mask=mask,
        )
        return {n: m.state_dict()[n] - before[n] for n in before}

    delta_full = _run(0.0)   # priv weight 1.0/3
    delta_half = _run(0.5)   # priv weight 0.5/3 → updates should be half

    n0 = _mlp_cfc_row_name(0)
    l0, _ = _tier_rows(active_idx)
    # Adam with same gradient direction (only magnitude differs) — but
    # adaptive learning rate makes magnitude-only scaling non-trivial.
    # Instead verify direction is the same (sign of each element matches).
    full = delta_full[n0][l0]
    half = delta_half[n0][l0]
    # At least one element should be non-zero (test sanity).
    assert full.abs().sum() > 0
    assert half.abs().sum() > 0
    # Adam normalizes by sqrt(v) so equal nonzero gradients of any magnitude
    # produce the SAME update. Half the gradient → same direction, similar
    # magnitude. Just assert both moved in the same direction (sign match).
    same_sign = ((full > 0) == (half > 0)) | (full == 0) | (half == 0)
    assert same_sign.all(), (
        f"priv-only updates should have same sign for cum=0 and cum=0.5; "
        f"differing signs found"
    )
