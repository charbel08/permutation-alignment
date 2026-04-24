"""Correctness tests for `multi_cumulative_private_finetune`.

Unlike the pretrain-side tests, which rely heavily on mocks, these run a
real tiny `GPTNeoForCausalLMTiered` with hand-crafted non-overlapping keys.
That lets us verify the core invariants against actual parameter state:

1. Pure-public parameters (embeddings, LayerNorms, LM-head bias) are frozen
   via `requires_grad=False` and never change.
2. Public slices *inside* keyed parameter tensors never change.
3. With `kl_lambda=0`, only the active tier's positions move. Non-active
   tiers stay byte-identical.
4. With `kl_lambda=1`, every tier position moves (KL hits all of them);
   public still frozen.
5. The model is always left in C1 arrangement after `train_step`.
6. Round-robin tier sampling visits each tier exactly once per N steps.
7. `_build_all_tiers_update_mask` marks exactly the tier positions True.
8. `apply/unapply_keys_cumulative` compose to identity.
9. Per-tier `_sample_tier` bookkeeping (`steps_sampled`) is correct.

Tiers setup used throughout: 3 tiers, each with a single MLP-column swap
across two layers, on non-overlapping column indices. That leaves plenty
of "public" MLP rows/cols to verify the freezing behavior.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest
import torch

from tiered.permutation import load_key
from tiered.permutation.masking import build_mask_plan
from tiered.permutation.permute import build_swap_plan
from tiered.train.finetune import multi_cumulative_private_finetune as mcpf
from tiered.train.utils import build_keyed_param_masks, load_model


DEVICE = torch.device("cpu")
TOKENIZED_SEQ_LEN = 4


# ────────────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
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
    """Build n non-overlapping keys, each with a single MLP-col pair swap.

    Tier k swaps column (2k) of layer 0 with column (2k+1) of layer 1.
    With intermediate=16, columns 0..5 are keyed (3 tiers × 2), and 6..15
    stay public inside the MLP tensors.
    """
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


def _build_tiers(model, key_paths: list[str]) -> list[mcpf.TierInfo]:
    tiers = []
    for i, kp in enumerate(key_paths):
        key = load_key(kp)
        tiers.append(mcpf.TierInfo(
            tier_id=i + 2, tier_idx=i, key=key,
            swap_plan=build_swap_plan(model, key, DEVICE),
            mask_plan=build_mask_plan(model, key, DEVICE),
            private_data_path="",
        ))
    return tiers


def _setup_optimizer(model, tiers, *, weight_decay: float = 0.0):
    """Replicate the frozen-public + param-group setup from `mcpf.train`.

    `weight_decay=0` is the default here so gradient-routing tests can assert
    exact byte-equality on non-active tier positions. A separate test exercises
    the weight-decay path explicitly.
    """
    keyed_param_ids = set()
    for tier in tiers:
        for param in build_keyed_param_masks(model, tier.mask_plan).keys():
            keyed_param_ids.add(id(param))
    keyed = [p for p in model.parameters() if id(p) in keyed_param_ids]
    purely_public = [p for p in model.parameters() if id(p) not in keyed_param_ids]
    for p in purely_public:
        p.requires_grad = False

    decay = [p for p in keyed if p.dim() >= 2]
    no_decay = [p for p in keyed if p.dim() < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    optim_obj = torch.optim.AdamW(groups, lr=0.1, betas=(0.9, 0.95))
    mask = mcpf._build_all_tiers_update_mask(model, [t.mask_plan for t in tiers])
    return optim_obj, mask, keyed, purely_public


def _dummy_batch(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 50, (1, TOKENIZED_SEQ_LEN), generator=g, device=DEVICE, dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}


def _run_one_step(tmp_path: Path, *, active_idx: int, kl_lambda: float,
                  n_tiers: int = 3, seed: int = 0):
    """Build fresh model+tiers+optimizer, run one train_step, and return
    (model, tiers, update_mask, snapshots_before, snapshots_after)."""
    torch.manual_seed(seed)
    model = _build_model()
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, n_tiers))
    optimizer, mask, _keyed, _public = _setup_optimizer(model, tiers)

    before = {n: p.data.clone() for n, p in model.named_parameters()}

    priv = _dummy_batch(seed=seed + 1)
    pub = _dummy_batch(seed=seed + 2)
    mcpf.train_step(
        model=model, raw_model=model, ref_model=ref_model,
        tiers=tiers, active_idx=active_idx,
        private_batch=priv, public_batch=pub,
        optimizer=optimizer, device=DEVICE,
        kl_lambda=kl_lambda, max_grad_norm=1.0,
        all_tiers_update_mask=mask, is_distributed=False,
    )
    after = {n: p.data.clone() for n, p in model.named_parameters()}
    return model, tiers, mask, before, after


# ────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ────────────────────────────────────────────────────────────────────────────

def test_sample_tier_round_robin_cycles_exactly():
    tiers = [mcpf.TierInfo(i + 2, i, None, None, None, "") for i in range(3)]
    rng = random.Random(0)
    picks = [mcpf._sample_tier(tiers, "round_robin", step, rng).tier_idx
             for step in range(9)]
    assert picks == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_sample_tier_round_robin_increments_steps_sampled():
    tiers = [mcpf.TierInfo(i + 2, i, None, None, None, "") for i in range(2)]
    rng = random.Random(0)
    for step in range(6):
        mcpf._sample_tier(tiers, "round_robin", step, rng)
    assert tiers[0].steps_sampled == 3
    assert tiers[1].steps_sampled == 3


def test_sample_tier_uniform_covers_all_over_time():
    tiers = [mcpf.TierInfo(i + 2, i, None, None, None, "") for i in range(3)]
    rng = random.Random(42)
    for step in range(900):
        mcpf._sample_tier(tiers, "uniform", step, rng)
    counts = [t.steps_sampled for t in tiers]
    assert sum(counts) == 900
    # Generous tolerance for 900 samples / 3 bins (expected ~300 each)
    for c in counts:
        assert 250 < c < 350, f"uniform sampling imbalanced: {counts}"


# ────────────────────────────────────────────────────────────────────────────
# Mask construction
# ────────────────────────────────────────────────────────────────────────────

def test_all_tiers_update_mask_marks_keyed_rows_and_public_rows_disjointly(tmp_path):
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    mask = mcpf._build_all_tiers_update_mask(model, [t.mask_plan for t in tiers])

    # MLP c_fc in layer 0: tier k swaps row (2k), so rows 0, 2, 4 are keyed
    # and rows 1, 3, 5 are NOT in this layer's swaps (they're in layer 1's side
    # of the cross-layer pair). Rows 6..15 are pure public.
    c_fc0 = model.transformer.h[0].mlp.c_fc.weight
    m0 = mask[c_fc0]
    for keyed_row in (0, 2, 4):
        assert m0[keyed_row].all(), f"row {keyed_row} should be all True in mask"
    for public_row in (6, 7, 10, 15):
        assert not m0[public_row].any(), f"row {public_row} should be all False"

    # MLP c_fc in layer 1: tier k swaps row (2k+1), so rows 1, 3, 5 are keyed
    c_fc1 = model.transformer.h[1].mlp.c_fc.weight
    m1 = mask[c_fc1]
    for keyed_row in (1, 3, 5):
        assert m1[keyed_row].all()
    for public_row in (0, 2, 4, 6, 15):
        assert not m1[public_row].any()


def test_all_tiers_update_mask_is_union_of_per_tier_masks(tmp_path):
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    combined = mcpf._build_all_tiers_update_mask(
        model, [t.mask_plan for t in tiers]
    )
    per_tier = [
        mcpf._build_all_tiers_update_mask(model, [t.mask_plan])
        for t in tiers
    ]
    for param, mask in combined.items():
        expected = torch.zeros_like(mask)
        for pt in per_tier:
            if param in pt:
                expected |= pt[param]
        assert torch.equal(mask, expected), \
            f"combined mask for {tuple(mask.shape)} != union of per-tier masks"


# ────────────────────────────────────────────────────────────────────────────
# Cumulative key plumbing
# ────────────────────────────────────────────────────────────────────────────

def test_apply_then_unapply_keys_cumulative_is_identity(tmp_path):
    model = _build_model()
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    mcpf._apply_keys_cumulative(model, tiers, up_to_idx=2)
    mcpf._unapply_keys_cumulative(model, tiers, up_to_idx=2)
    for name, p in model.named_parameters():
        assert torch.equal(p.data, before[name]), f"{name} not restored"


def test_apply_keys_cumulative_matches_sequential_apply(tmp_path):
    """Applying keys 0..k cumulatively should be identical to hand-applying
    each key in turn via `apply_permutation`."""
    from tiered.permutation.permute import apply_permutation

    model_a = _build_model()
    model_b = copy.deepcopy(model_a)
    tiers_a = _build_tiers(model_a, _make_non_overlapping_keys(tmp_path, 3))
    tiers_b = _build_tiers(model_b, _make_non_overlapping_keys(tmp_path, 3))

    mcpf._apply_keys_cumulative(model_a, tiers_a, up_to_idx=1)
    for i in range(2):
        apply_permutation(model_b, tiers_b[i].key, plan=tiers_b[i].swap_plan)

    for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.equal(pa.data, pb.data), f"divergence at {na}"


# ────────────────────────────────────────────────────────────────────────────
# train_step invariants on a real tiny model
# ────────────────────────────────────────────────────────────────────────────

PURELY_PUBLIC_PARAMS = (
    "transformer.wte.weight",
    "transformer.wpe.weight",
    "transformer.ln_f.weight",
    "transformer.ln_f.bias",
)


@pytest.mark.parametrize("active_idx", [0, 1, 2])
@pytest.mark.parametrize("kl_lambda", [0.0, 0.1, 1.0])
def test_pure_public_params_frozen_through_step(tmp_path, active_idx, kl_lambda):
    _, _, _, before, after = _run_one_step(
        tmp_path, active_idx=active_idx, kl_lambda=kl_lambda,
    )
    for name in PURELY_PUBLIC_PARAMS:
        if name in before:
            assert torch.equal(before[name], after[name]), \
                f"{name} changed at active={active_idx}, kl={kl_lambda}"


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_public_slices_inside_keyed_tensors_frozen(tmp_path, active_idx):
    """Rows of c_fc / cols of c_proj that don't belong to any tier must not
    move, regardless of which tier is active."""
    model, _, mask, before, after = _run_one_step(
        tmp_path, active_idx=active_idx, kl_lambda=0.1,
    )
    # Map parameter object id → name so we can look up snapshots by param.
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    for param, keyed_mask in mask.items():
        frozen_positions = ~keyed_mask
        if not torch.any(frozen_positions):
            continue
        name = id_to_name[id(param)]
        delta = (after[name] - before[name]).abs()
        assert torch.all(delta[frozen_positions] == 0), \
            f"public slice of {name} moved under active={active_idx}"


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_kl_zero_non_active_tiers_unchanged(tmp_path, active_idx):
    """With λ=0, non-active tier positions must stay byte-identical."""
    model, tiers, mask, before, after = _run_one_step(
        tmp_path, active_idx=active_idx, kl_lambda=0.0,
    )
    # Non-active tiers: their keyed positions should NOT have moved.
    for t_idx, tier in enumerate(tiers):
        if t_idx == active_idx:
            continue
        # Tier t swaps MLP col 2t of layer 0 ↔ col 2t+1 of layer 1.
        l0_row = 2 * t_idx
        l1_row = 2 * t_idx + 1
        for layer_idx, row in ((0, l0_row), (1, l1_row)):
            name = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
            assert torch.equal(before[name][row], after[name][row]), \
                f"non-active tier {t_idx} row {row} (layer {layer_idx}) moved " \
                f"when active={active_idx}"


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_kl_zero_active_tier_moves(tmp_path, active_idx):
    """With λ=0, the active tier's positions DO receive private-loss update."""
    _, tiers, _, before, after = _run_one_step(
        tmp_path, active_idx=active_idx, kl_lambda=0.0,
    )
    l0_row = 2 * active_idx
    l1_row = 2 * active_idx + 1
    name0 = "transformer.h.0.mlp.c_fc.weight"
    name1 = "transformer.h.1.mlp.c_fc.weight"
    assert not torch.equal(before[name0][l0_row], after[name0][l0_row]), \
        f"active tier {active_idx} layer 0 row {l0_row} should have moved"
    assert not torch.equal(before[name1][l1_row], after[name1][l1_row]), \
        f"active tier {active_idx} layer 1 row {l1_row} should have moved"


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_kl_one_all_tier_positions_move(tmp_path, active_idx):
    """With λ=1, every tier's positions receive a (KL-only) update; public
    positions still stay frozen (covered by the other tests)."""
    _, tiers, _, before, after = _run_one_step(
        tmp_path, active_idx=active_idx, kl_lambda=1.0,
    )
    for t_idx in range(len(tiers)):
        l0_row = 2 * t_idx
        l1_row = 2 * t_idx + 1
        name0 = "transformer.h.0.mlp.c_fc.weight"
        name1 = "transformer.h.1.mlp.c_fc.weight"
        # Under λ=1 every tier row must have moved (KL grad is non-zero on
        # real data + seeded model).
        assert not torch.equal(before[name0][l0_row], after[name0][l0_row]), \
            f"tier {t_idx} layer 0 row {l0_row} did not move under λ=1"
        assert not torch.equal(before[name1][l1_row], after[name1][l1_row]), \
            f"tier {t_idx} layer 1 row {l1_row} did not move under λ=1"


@pytest.mark.parametrize("active_idx", [0, 1, 2])
def test_train_step_leaves_model_in_c1_arrangement(tmp_path, active_idx):
    """After `train_step`, the weights must be in C1 layout — applying the
    active tier's keys and immediately unapplying them must be a no-op
    against the post-step state."""
    model, tiers, _, _, _ = _run_one_step(
        tmp_path, active_idx=active_idx, kl_lambda=0.1,
    )
    snap = {n: p.data.clone() for n, p in model.named_parameters()}
    # If weights are in C1, applying+unapplying the cumulative keys is identity.
    mcpf._apply_keys_cumulative(model, tiers, up_to_idx=active_idx)
    mcpf._unapply_keys_cumulative(model, tiers, up_to_idx=active_idx)
    for name, p in model.named_parameters():
        assert torch.equal(p.data, snap[name]), \
            f"{name} not in C1 after train_step (active={active_idx})"


def test_weight_decay_does_not_touch_public_slices(tmp_path):
    """With wd=0.01 on all keyed tensors, public slices inside those tensors
    must STILL not change — adamw_step_preserving_public restores them
    (and their AdamW state) after each step."""
    torch.manual_seed(0)
    model = _build_model()
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    optimizer, mask, _k, _pub = _setup_optimizer(model, tiers, weight_decay=0.01)
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    before = {n: p.data.clone() for n, p in model.named_parameters()}
    mcpf.train_step(
        model=model, raw_model=model, ref_model=ref_model,
        tiers=tiers, active_idx=1,
        private_batch=_dummy_batch(1), public_batch=_dummy_batch(2),
        optimizer=optimizer, device=DEVICE,
        kl_lambda=0.1, max_grad_norm=1.0,
        all_tiers_update_mask=mask, is_distributed=False,
    )
    for param, keyed_mask in mask.items():
        frozen = ~keyed_mask
        if not torch.any(frozen):
            continue
        name = id_to_name[id(param)]
        delta = (param.data - before[name]).abs()
        assert torch.all(delta[frozen] == 0), \
            f"public slice of {name} drifted under weight_decay=0.01"


def test_train_step_preserves_public_across_many_steps(tmp_path):
    """Run a dozen steps cycling through tiers and verify every pure-public
    parameter is bit-identical to its initial value at the end."""
    torch.manual_seed(123)
    model = _build_model()
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    tiers = _build_tiers(model, _make_non_overlapping_keys(tmp_path, 3))
    optimizer, mask, _keyed, _public = _setup_optimizer(model, tiers)
    initial = {n: p.data.clone() for n, p in model.named_parameters()}

    for step in range(12):
        priv = _dummy_batch(seed=step * 2)
        pub = _dummy_batch(seed=step * 2 + 1)
        mcpf.train_step(
            model=model, raw_model=model, ref_model=ref_model,
            tiers=tiers, active_idx=step % 3,
            private_batch=priv, public_batch=pub,
            optimizer=optimizer, device=DEVICE,
            kl_lambda=0.1, max_grad_norm=1.0,
            all_tiers_update_mask=mask, is_distributed=False,
        )

    for name in PURELY_PUBLIC_PARAMS:
        if name in initial:
            assert torch.equal(model.state_dict()[name], initial[name]), \
                f"{name} drifted across 12 steps"
