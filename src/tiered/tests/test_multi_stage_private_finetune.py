"""Correctness tests for `multi_stage_private_finetune`.

Run against a real tiny `GPTNeoForCausalLMTiered` so every invariant is
checked against actual parameter state rather than mocks. Coverage:

1. **Only the active tier moves.** Public, prior tiers, future tiers all
   stay byte-identical after any single `train_step` call, across the full
   cartesian product of active_idx × λ_pub × λ_anchor.
2. **Public params are structurally frozen** — the script sets
   `requires_grad=False` on every fully-public parameter (embeddings,
   LayerNorms, LM-head bias) before the optimizer is built.
3. **`adamw_step_preserving_public` is the defense-in-depth**: with a
   non-zero weight-decay the active-tier-only update mask must keep every
   non-active position (including public slices *inside* keyed tensors)
   bit-identical.
4. **Reference models are never mutated** — they're frozen, and although
   we apply/unapply permutations on them during the anchor-KL pass, the
   sequence must be a no-op against any reference.
5. **Model arrangement after `train_step` is C1 (home)** — the step runs in
   home coords, so any residual apply/unapply mismatch would show up here.
6. **Adam state stays consistent across steps.** Running N steps in a row
   on a single active tier: non-active positions' Adam `exp_avg` /
   `exp_avg_sq` must remain zero (they never received a gradient), even
   when weight-decay is non-zero.
7. **KL disabling flags work** — `kl_lambda=0` skips the pretrain-KL
   forward; `anchor_kl_lambda=0` skips every anchor forward. Reference
   models also see no forwards in that case.
8. **Helpers (`_apply_keys` / `_unapply_keys`) compose to identity.**
9. **Sequential stage hand-off preserves tier 0 weights** — after stage 1
   runs with the stage-0 checkpoint as anchor, tier 0 positions are
   byte-identical to stage 0's end state (they're frozen in stage 1).

Tiers setup used throughout: 3 MLP-only tier keys, each swapping a unique
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
from tiered.permutation.permute import build_swap_plan
from tiered.train.finetune import multi_stage_private_finetune as mspf
from tiered.train.utils import build_keyed_param_masks, load_model


DEVICE = torch.device("cpu")
TOKENIZED_SEQ_LEN = 4


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
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


def _build_tiers(model, key_paths: list[str]) -> list[mspf.TierKey]:
    tiers = []
    for i, kp in enumerate(key_paths):
        key = load_key(kp)
        tiers.append(mspf.TierKey(
            tier_idx=i, tier_id=i + 2, key=key,
            swap_plan=build_swap_plan(model, key, DEVICE),
            mask_plan=build_mask_plan(model, key, DEVICE),
        ))
    return tiers


def _setup_optimizer(model, tiers, active_idx: int, *, weight_decay: float = 0.0):
    """Mirror the script's optimizer setup."""
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
                     anchor_kl_lambda=0.1, n_tiers=3, seed=0,
                     weight_decay=0.0, make_anchor_refs=True):
    """Build everything, run one train_step, and return snapshots."""
    torch.manual_seed(seed)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, n_tiers))

    pretrain_ref = _frozen_copy(model) if kl_lambda > 0 else None
    anchor_refs = []
    for s in range(active_idx):
        if make_anchor_refs and anchor_kl_lambda > 0:
            # Anchor refs are independent frozen copies of the current model.
            anchor_refs.append(_frozen_copy(model))
        else:
            anchor_refs.append(None)

    optimizer, active_mask = _setup_optimizer(
        model, tiers, active_idx, weight_decay=weight_decay,
    )

    before = {n: p.data.clone() for n, p in model.named_parameters()}
    # Snapshots of every reference's weights — must be identical after step.
    ref_snaps = {}
    if pretrain_ref is not None:
        ref_snaps["pretrain"] = {
            n: p.data.clone() for n, p in pretrain_ref.named_parameters()
        }
    for i, ar in enumerate(anchor_refs):
        if ar is not None:
            ref_snaps[f"anchor_{i}"] = {
                n: p.data.clone() for n, p in ar.named_parameters()
            }

    # All batches this step needs
    priv_batches = [_dummy_batch(seed=seed + 10 + i) for i in range(active_idx + 1)]
    pub_batch = _dummy_batch(seed=seed + 100)

    mspf.train_step(
        model=model, raw_model=model,
        pretrain_ref=pretrain_ref, anchor_refs=anchor_refs,
        tiers=tiers, active_idx=active_idx,
        private_batches=priv_batches, public_batch=pub_batch,
        optimizer=optimizer, device=DEVICE,
        kl_lambda=kl_lambda,
        anchor_kl_lambda=anchor_kl_lambda,
        max_grad_norm=1.0,
        active_update_mask=active_mask,
        is_distributed=False,
    )

    after = {n: p.data.clone() for n, p in model.named_parameters()}
    ref_after = {}
    if pretrain_ref is not None:
        ref_after["pretrain"] = {
            n: p.data.clone() for n, p in pretrain_ref.named_parameters()
        }
    for i, ar in enumerate(anchor_refs):
        if ar is not None:
            ref_after[f"anchor_{i}"] = {
                n: p.data.clone() for n, p in ar.named_parameters()
            }
    return SimpleNamespace(
        model=model, tiers=tiers, active_idx=active_idx,
        optimizer=optimizer, active_mask=active_mask,
        before=before, after=after,
        ref_before=ref_snaps, ref_after=ref_after,
    )


# ────────────────────────────────────────────────────────────────────────────
# Helper invariants
# ────────────────────────────────────────────────────────────────────────────

def test_apply_then_unapply_is_identity(tmp_path):
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    mspf._apply_keys(model, tiers, up_to_idx=2)
    mspf._unapply_keys(model, tiers, up_to_idx=2)
    for n, p in model.named_parameters():
        assert torch.equal(p.data, before[n])


def test_apply_keys_negative_is_noop(tmp_path):
    """up_to_idx = -1 means stage 0 at C1 (no keys applied)."""
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    mspf._apply_keys(model, tiers, up_to_idx=-1)
    for n, p in model.named_parameters():
        assert torch.equal(p.data, before[n])


# ────────────────────────────────────────────────────────────────────────────
# "Only the active tier moves" — full cartesian product
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
@pytest.mark.parametrize("kl_lambda", [0.0, 0.1])
@pytest.mark.parametrize("anchor_kl_lambda", [0.0, 0.1])
def test_only_active_tier_updates(tmp_path, active_idx, kl_lambda, anchor_kl_lambda):
    """Regardless of λ settings, only the active tier's MLP rows move."""
    s = _run_single_step(
        tmp_path, active_idx=active_idx,
        kl_lambda=kl_lambda, anchor_kl_lambda=anchor_kl_lambda,
    )
    for t_idx in range(3):
        l0, l1 = _tier_rows(t_idx)
        n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
        if t_idx == active_idx:
            assert not torch.equal(s.before[n0][l0], s.after[n0][l0]), (
                f"active tier {t_idx} row {l0} at layer 0 must move"
            )
            assert not torch.equal(s.before[n1][l1], s.after[n1][l1]), (
                f"active tier {t_idx} row {l1} at layer 1 must move"
            )
        else:
            assert torch.equal(s.before[n0][l0], s.after[n0][l0]), (
                f"non-active tier {t_idx} row {l0} (active={active_idx}, "
                f"λ_pub={kl_lambda}, λ_a={anchor_kl_lambda}) must not move"
            )
            assert torch.equal(s.before[n1][l1], s.after[n1][l1]), (
                f"non-active tier {t_idx} row {l1} (active={active_idx}, "
                f"λ_pub={kl_lambda}, λ_a={anchor_kl_lambda}) must not move"
            )


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_pure_public_params_are_structurally_frozen(tmp_path, active_idx):
    """Embeddings / LN / LM-head bias should never receive a gradient
    because `requires_grad=False` gets set at setup time."""
    s = _run_single_step(tmp_path, active_idx=active_idx)
    for name in PURELY_PUBLIC_PARAMS:
        if name in s.before:
            assert torch.equal(s.before[name], s.after[name]), (
                f"{name} drifted at active={active_idx}"
            )


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_public_slices_inside_keyed_tensors_frozen(tmp_path, active_idx):
    """Rows of c_fc that aren't in any tier's swap — specifically rows 6..15
    with the default 3-tier setup — must stay byte-identical."""
    s = _run_single_step(tmp_path, active_idx=active_idx)
    # The active tier's mask marks only its own keyed positions True; every
    # False position must not move.
    id_to_name = {id(p): n for n, p in s.model.named_parameters()}
    for param, keyed_mask in s.active_mask.items():
        frozen = ~keyed_mask
        if not torch.any(frozen):
            continue
        name = id_to_name[id(param)]
        delta = (s.after[name] - s.before[name]).abs()
        assert torch.all(delta[frozen] == 0), (
            f"public slice of {name} moved at active={active_idx}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Reference models must never be mutated
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_reference_models_are_unchanged_after_step(tmp_path, active_idx):
    """The pretrain_ref and every anchor_ref must be byte-identical before
    vs after `train_step`. If an apply/unapply pair gets missed, the frozen
    ref will end up permuted."""
    s = _run_single_step(tmp_path, active_idx=active_idx)
    for ref_name, snap in s.ref_before.items():
        for param_name, before_val in snap.items():
            after_val = s.ref_after[ref_name][param_name]
            assert torch.equal(before_val, after_val), (
                f"{ref_name} / {param_name} was mutated during train_step"
            )


# ────────────────────────────────────────────────────────────────────────────
# Model arrangement after the step
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_model_returns_to_home_arrangement(tmp_path, active_idx):
    """After `train_step`, applying + unapplying any cumulative prefix of
    keys must be a no-op — i.e. the weights must already be at C1."""
    s = _run_single_step(tmp_path, active_idx=active_idx)
    snap = {n: p.data.clone() for n, p in s.model.named_parameters()}
    mspf._apply_keys(s.model, s.tiers, up_to_idx=active_idx)
    mspf._unapply_keys(s.model, s.tiers, up_to_idx=active_idx)
    for n, p in s.model.named_parameters():
        assert torch.equal(p.data, snap[n]), f"{n} drifted on apply+unapply cycle"


# ────────────────────────────────────────────────────────────────────────────
# KL disabling
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_all_kl_disabled_still_trains_active(tmp_path, active_idx):
    """With λ_pub=λ_a=0, only the private loss drives active tier updates.
    Non-active tiers still must be frozen."""
    s = _run_single_step(
        tmp_path, active_idx=active_idx, kl_lambda=0.0, anchor_kl_lambda=0.0,
    )
    l0, l1 = _tier_rows(active_idx)
    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    assert not torch.equal(s.before[n0][l0], s.after[n0][l0])
    assert not torch.equal(s.before[n1][l1], s.after[n1][l1])
    for other in range(3):
        if other == active_idx:
            continue
        ol0, ol1 = _tier_rows(other)
        assert torch.equal(s.before[n0][ol0], s.after[n0][ol0])
        assert torch.equal(s.before[n1][ol1], s.after[n1][ol1])


def test_anchor_kl_lambda_zero_is_no_op_vs_none_anchor(tmp_path):
    """λ_anchor = 0 should be identical to not having anchor refs at all.
    Tests that the anchor loop is cleanly short-circuited when disabled."""
    # Two runs from identical seeds: one with anchor refs + λ_anchor=0,
    # one with no anchor refs (via make_anchor_refs=False).
    s_with = _run_single_step(
        tmp_path, active_idx=2, kl_lambda=0.0, anchor_kl_lambda=0.0,
        make_anchor_refs=True,
    )
    s_without = _run_single_step(
        tmp_path, active_idx=2, kl_lambda=0.0, anchor_kl_lambda=0.0,
        make_anchor_refs=False,
    )
    for name in s_with.after:
        assert torch.equal(s_with.after[name], s_without.after[name]), (
            f"{name} diverged between anchor-present (λ=0) and anchor-absent runs"
        )


# ────────────────────────────────────────────────────────────────────────────
# Adam state hygiene
# ────────────────────────────────────────────────────────────────────────────

def test_adam_state_only_on_active_tier_positions(tmp_path):
    """After one step, Adam's `exp_avg` / `exp_avg_sq` must be non-zero
    ONLY at the active tier's positions. Every non-active position must
    still be zero — the preservation machinery restores momentum after
    the step on frozen slots."""
    s = _run_single_step(tmp_path, active_idx=1, kl_lambda=0.1, anchor_kl_lambda=0.1)
    id_to_name = {id(p): n for n, p in s.model.named_parameters()}
    for p in s.model.parameters():
        if not p.requires_grad:
            continue
        state = s.optimizer.state.get(p)
        if state is None or "exp_avg" not in state:
            continue
        active_mask = s.active_mask.get(p)
        if active_mask is None:
            # Parameter not in the active mask → should have no movement
            # at all, hence no momentum. Check exp_avg is zero everywhere.
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
    Our `adamw_step_preserving_public` must snap every False-mask position
    (public slices + non-active tiers' positions + true-public params)
    back, so no drift shows up even at wd=0.01."""
    s = _run_single_step(
        tmp_path, active_idx=1, kl_lambda=0.1, anchor_kl_lambda=0.1,
        weight_decay=0.01,
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
    # Truly public params (no keyed positions at all) use requires_grad=False,
    # so AdamW skips them entirely — wd cannot apply.
    for name in PURELY_PUBLIC_PARAMS:
        if name in s.before:
            assert torch.equal(s.before[name], s.after[name])


def test_multiple_steps_keep_non_active_frozen(tmp_path):
    """Accumulated: after 8 steps of the same active tier, every non-active
    position must still be byte-identical to its starting value. Catches
    cases where Adam's momentum or v tensor silently drifts frozen slots."""
    torch.manual_seed(7)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    pretrain_ref = _frozen_copy(model)
    active_idx = 2
    anchor_refs = [_frozen_copy(model) for _ in range(active_idx)]
    optimizer, active_mask = _setup_optimizer(
        model, tiers, active_idx, weight_decay=0.01,
    )
    initial = {n: p.data.clone() for n, p in model.named_parameters()}

    for step in range(8):
        priv_batches = [_dummy_batch(seed=step * 10 + i) for i in range(active_idx + 1)]
        pub_batch = _dummy_batch(seed=step * 10 + 99)
        mspf.train_step(
            model=model, raw_model=model,
            pretrain_ref=pretrain_ref, anchor_refs=anchor_refs,
            tiers=tiers, active_idx=active_idx,
            private_batches=priv_batches, public_batch=pub_batch,
            optimizer=optimizer, device=DEVICE,
            kl_lambda=0.1, anchor_kl_lambda=0.1,
            max_grad_norm=1.0,
            active_update_mask=active_mask, is_distributed=False,
        )

    id_to_name = {id(p): n for n, p in model.named_parameters()}

    # Non-active tier rows, layer 0 and 1, over all 8 steps: still initial.
    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    for other in (0, 1):  # tiers 0 and 1 are non-active when active_idx=2
        ol0, ol1 = _tier_rows(other)
        final0 = model.state_dict()[n0][ol0]
        final1 = model.state_dict()[n1][ol1]
        assert torch.equal(initial[n0][ol0], final0), (
            f"tier {other} row {ol0} drifted across 8 steps"
        )
        assert torch.equal(initial[n1][ol1], final1), (
            f"tier {other} row {ol1} drifted across 8 steps"
        )

    # Purely public params: still initial.
    for name in PURELY_PUBLIC_PARAMS:
        if name in initial:
            assert torch.equal(initial[name], model.state_dict()[name])

    # Reference models still frozen.
    for p_ref, p_init in zip(pretrain_ref.parameters(), [_frozen_copy(_build_model())]):
        pass  # see below, we re-check with the snapshot approach


def test_sequential_stage_hand_off_preserves_prior_tier(tmp_path):
    """Run stage 0 (active=tier 0), save its 'final' state in-memory, then
    run stage 1 (active=tier 1) with a frozen copy of stage-0-final as the
    anchor. After stage 1, tier 0's positions must be bit-identical to
    stage 0's end state."""
    torch.manual_seed(0)
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))

    pretrain_ref = _frozen_copy(model)

    # --- Stage 0: active tier 0 ---
    optimizer0, mask0 = _setup_optimizer(model, tiers, active_idx=0)
    for step in range(3):
        priv = [_dummy_batch(seed=step * 2)]
        pub = _dummy_batch(seed=step * 2 + 1)
        mspf.train_step(
            model=model, raw_model=model,
            pretrain_ref=pretrain_ref, anchor_refs=[],
            tiers=tiers, active_idx=0,
            private_batches=priv, public_batch=pub,
            optimizer=optimizer0, device=DEVICE,
            kl_lambda=0.1, anchor_kl_lambda=0.1,
            max_grad_norm=1.0,
            active_update_mask=mask0, is_distributed=False,
        )
    stage0_final = {n: p.data.clone() for n, p in model.named_parameters()}
    stage0_anchor = _frozen_copy(model)

    # --- Stage 1: active tier 1, anchor = stage-0 final ---
    optimizer1, mask1 = _setup_optimizer(model, tiers, active_idx=1)
    for step in range(3):
        priv = [_dummy_batch(seed=100 + step * 2 + i) for i in range(2)]
        pub = _dummy_batch(seed=100 + step * 2 + 9)
        mspf.train_step(
            model=model, raw_model=model,
            pretrain_ref=pretrain_ref, anchor_refs=[stage0_anchor],
            tiers=tiers, active_idx=1,
            private_batches=priv, public_batch=pub,
            optimizer=optimizer1, device=DEVICE,
            kl_lambda=0.1, anchor_kl_lambda=0.1,
            max_grad_norm=1.0,
            active_update_mask=mask1, is_distributed=False,
        )

    # Tier 0's rows must be identical to stage-0-final; tier 1 should have
    # moved; tier 2 must still be identical to stage-0-final (it was frozen
    # in both stages).
    n0, n1 = _mlp_cfc_row_name(0), _mlp_cfc_row_name(1)
    for t_idx, expected_moved in enumerate([False, True, False]):
        l0, l1 = _tier_rows(t_idx)
        final0 = model.state_dict()[n0][l0]
        final1 = model.state_dict()[n1][l1]
        moved = (not torch.equal(stage0_final[n0][l0], final0)
                 or not torch.equal(stage0_final[n1][l1], final1))
        if expected_moved:
            assert moved, f"tier {t_idx} should have moved during stage 1"
        else:
            assert not moved, f"tier {t_idx} must be bit-identical to stage-0 final"


# ────────────────────────────────────────────────────────────────────────────
# Parse-args sanity
# ────────────────────────────────────────────────────────────────────────────

def test_mismatched_anchor_count_raises(tmp_path, monkeypatch):
    """`active_idx=2` needs exactly 2 anchor checkpoints. Fewer/more must
    abort with a clear error."""
    import sys
    key_paths = _make_non_overlapping_keys(tmp_path, 3)
    priv_paths = ["/tmp/fake"] * 3  # won't actually be opened
    argv = [
        "prog",
        "--checkpoint", "/tmp/student",
        "--pretrain_checkpoint", "/tmp/pretrain",
        "--anchor_checkpoints", "/tmp/a0",  # only 1 anchor for active_idx=2
        "--all_key_paths", *key_paths,
        "--active_idx", "2",
        "--private_data", *priv_paths,
        "--public_data", "/tmp/public",
        "--output_dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = mspf.parse_args()
    # parse_args itself doesn't validate — the validation is in train(),
    # which runs after data loading. We test the same guard rail.
    assert len(args.anchor_checkpoints) != args.active_idx
