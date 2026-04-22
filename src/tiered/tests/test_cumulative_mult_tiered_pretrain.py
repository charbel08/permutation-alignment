"""Tests for the cumulative N-tier pretraining gradient combination.

Covers the core algorithmic invariants after the rewrite:

1. Phase 1 masks C1 grads for tiers 0..active_idx only (not later tiers).
2. The Phase 2 weighting (uniform 0.5 scaling on every parameter) gives:
     - effective public (public + tiers K+1..N-1):  0.5 * (grad_c1 + grad_c_{k+1})
     - composite keyed (tiers 0..K):                0.5 * grad_c_{k+1}
3. Plain `optimizer.step()` is used (no preserving-public variant).
4. The per-step event order is: mask composite-keyed, apply keys,
   C_{k+1} backward, unapply keys, swap gradients, optimizer step.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tiered.train.pretrain import cumulative_mult_tiered_pretrain as cmt


# ─── shared stubs ─────────────────────────────────────────────────────────

class _TinyTokenizer:
    eos_token = "<eos>"
    pad_token = "<eos>"

    def save_pretrained(self, path):
        return None


class _TinyCollator:
    def __init__(self, tokenizer=None, mlm=False):
        pass

    def __call__(self, batch):
        ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n_items=8, seq_len=4, vocab=8):
        self._items = [
            {
                "input_ids": [(i + j) % vocab for j in range(seq_len)],
                "attention_mask": [1] * seq_len,
            }
            for i in range(n_items)
        ]
        self.column_names = ["input_ids", "attention_mask"]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def remove_columns(self, cols):
        for item in self._items:
            for c in cols:
                item.pop(c, None)
        self.column_names = [c for c in self.column_names if c not in cols]
        return self


class _FakeScheduler:
    def __init__(self, *a, **k):
        self._state = {}

    def step(self):
        return None

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)


class _FakePbar:
    def __init__(self, *a, **k):
        self.n = k.get("initial", 0)

    def update(self, n=1):
        self.n += n

    def set_postfix(self, *a, **k):
        return None

    def close(self):
        return None


class _FakeConfig(dict):
    def update(self, data=None, **kwargs):
        if data is not None:
            super().update(data)


class _FakeRun:
    def __init__(self, run_id="fake-run-id"):
        self.id = run_id
        self.summary = {}


class _FakeWandb:
    def __init__(self):
        self.config = _FakeConfig()
        self.run = None
        self.logged = []

    def init(self, project=None, name=None, id=None, resume=None, config=None):
        self.run = _FakeRun(id or "fake-run-id")
        if config is not None:
            self.config.update(config)

    def define_metric(self, *a, **k):
        return None

    def log(self, data):
        self.logged.append(dict(data))

    def finish(self):
        return None


class _NTierModel(torch.nn.Module):
    """Minimal stand-in with one public param and N keyed params.

    Forward returns different coefficients depending on how many keys are
    currently applied, so C1 and C_{k+1} gradients are distinguishable:

        C1:        loss = 2 * public + 3 * sum(keyed[i])
        C_{k+1}:   loss = 5 * public + 7 * sum(keyed[i])

    The "keyed-ness" of each keyed[i] is represented by a per-tier MaskPlan
    sentinel (tier_idx), which test stubs use to dispatch scale/mask calls
    to the right parameter.
    """

    def __init__(self, n_tiers, public_init=0.0, keyed_inits=None,
                 vocab_size=8, context_size=4):
        super().__init__()
        keyed_inits = keyed_inits if keyed_inits is not None else [0.0] * n_tiers
        assert len(keyed_inits) == n_tiers
        self.public = torch.nn.Parameter(torch.tensor(float(public_init)))
        self.keyed = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(float(v))) for v in keyed_inits]
        )
        self._emb = torch.nn.Embedding(vocab_size, 2)
        self.config = SimpleNamespace(max_position_embeddings=context_size)
        self._keys_applied = 0

    def gradient_checkpointing_enable(self, **kwargs):
        return None

    def get_input_embeddings(self):
        return self._emb

    def save_pretrained(self, path):
        return None

    def forward(self, input_ids, labels=None):
        bsz, seq = input_ids.shape
        vocab = self._emb.weight.shape[0]
        logits = torch.zeros(
            (bsz, seq, vocab), dtype=torch.float32, device=input_ids.device
        )
        keyed_sum = sum(k for k in self.keyed)
        if self._keys_applied > 0:
            loss = 5.0 * self.public + 7.0 * keyed_sum
        else:
            loss = 2.0 * self.public + 3.0 * keyed_sum
        return SimpleNamespace(loss=loss, logits=logits)


def _install_common_patches(monkeypatch, model, n_tiers):
    """Stub out I/O + permutation ops so `cmt.train` runs against `model`.

    Returns a context dict containing the fake wandb, an `events` list that
    records apply/unapply/scale/mask/optimizer_step calls in order, and the
    per-tier mask-plan sentinels.
    """
    fake_wandb = _FakeWandb()
    events = []
    save_calls = []

    monkeypatch.setattr(cmt, "wandb", fake_wandb)
    monkeypatch.setattr(cmt.torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(cmt.torch, "compile", lambda m: m)
    monkeypatch.setattr(cmt, "tqdm", lambda *a, **k: _FakePbar(**k))
    monkeypatch.setattr(
        cmt.AutoTokenizer, "from_pretrained", lambda *a, **k: _TinyTokenizer()
    )
    monkeypatch.setattr(cmt, "DataCollatorForLanguageModeling", _TinyCollator)
    monkeypatch.setattr(
        cmt, "load_from_disk", lambda *a, **k: {"train": _TinyDataset()}
    )
    monkeypatch.setattr(cmt, "load_model", lambda **k: model)
    monkeypatch.setattr(
        cmt.GPTNeoForCausalLMTiered,
        "from_pretrained",
        staticmethod(lambda *a, **k: model),
    )

    monkeypatch.setattr(cmt, "count_total_parameters", lambda m: 10)
    monkeypatch.setattr(cmt, "count_trainable_parameters", lambda m: 10)
    monkeypatch.setattr(
        cmt,
        "count_swappable_parameters",
        lambda m, p: {"total": 2, "attention": 1, "mlp": 1},
    )
    monkeypatch.setattr(
        cmt,
        "count_max_swappable_parameters",
        lambda m: {"total": 4, "attention": 2, "mlp": 2},
    )
    monkeypatch.setattr(cmt, "detect_gpu_peak_flops", lambda d: (0.0, "cpu"))
    monkeypatch.setattr(cmt, "get_gpu_memory_stats", lambda d: {})

    # Per-tier mask-plan sentinels: `tier_idx` distinguishes them.
    tier_mask_plans = [
        SimpleNamespace(
            tier_idx=i,
            keyed_attn_indices={},
            keyed_attn_out_indices={},
            keyed_mlp_indices={},
            keyed_mlp_up_indices={},
            keyed_mlp_down_indices={},
        )
        for i in range(n_tiers)
    ]
    plan_queue = list(tier_mask_plans)

    def _build_mask_plan(*a, **k):
        return plan_queue.pop(0)

    key_obj = SimpleNamespace(
        attn_heads=[], attn_out_heads=[], mlp_cols=[],
        mlp_up_cols=[], mlp_down_cols=[],
    )
    monkeypatch.setattr(cmt, "load_key", lambda p: key_obj)
    monkeypatch.setattr(
        cmt, "build_swap_plan", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(cmt, "build_mask_plan", _build_mask_plan)

    def _apply(raw_model, key, plan=None):
        raw_model._keys_applied += 1
        events.append("apply")

    def _unapply(raw_model, key, plan=None):
        raw_model._keys_applied -= 1
        events.append("unapply")

    def _swap(raw_model, key, plan=None):
        events.append("swap")

    def _mask_keyed(raw_model, key, plan=None):
        events.append(f"mask_{plan.tier_idx}")
        idx = plan.tier_idx
        if 0 <= idx < len(raw_model.keyed):
            if raw_model.keyed[idx].grad is not None:
                raw_model.keyed[idx].grad.zero_()

    monkeypatch.setattr(cmt, "apply_permutation", _apply)
    monkeypatch.setattr(cmt, "unapply_permutation", _unapply)
    monkeypatch.setattr(cmt, "swap_gradients", _swap)
    monkeypatch.setattr(cmt, "mask_keyed_gradients", _mask_keyed)

    class _SGDOptimizer:
        def __init__(self, params, lr=0.0, **kwargs):
            self.param_groups = []
            for group in params:
                g = dict(group)
                g["params"] = list(g["params"])
                g.setdefault("lr", lr)
                self.param_groups.append(g)
            self.step_calls = 0

        def zero_grad(self):
            for g in self.param_groups:
                for p in g["params"]:
                    if p.grad is not None:
                        p.grad.zero_()

        def step(self):
            self.step_calls += 1
            events.append("optimizer_step")
            for g in self.param_groups:
                lr = g["lr"]
                for p in g["params"]:
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-lr)

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            return None

    monkeypatch.setattr(cmt.optim, "AdamW", _SGDOptimizer)
    monkeypatch.setattr(cmt, "LinearLR", _FakeScheduler)
    monkeypatch.setattr(cmt, "CosineAnnealingLR", _FakeScheduler)
    monkeypatch.setattr(cmt, "SequentialLR", _FakeScheduler)
    monkeypatch.setattr(
        cmt.torch.nn.utils,
        "clip_grad_norm_",
        lambda *a, **k: torch.tensor(0.0),
    )
    monkeypatch.setattr(
        cmt,
        "save_checkpoint",
        lambda *a, **k: save_calls.append(dict(k)),
    )

    return {
        "wandb": fake_wandb,
        "events": events,
        "save_calls": save_calls,
        "tier_mask_plans": tier_mask_plans,
    }


def _default_args(tmp_path: Path, *, key_paths, max_steps=1,
                  tier_sample="round_robin", checkpoint=None,
                  learning_rate=1.0):
    return SimpleNamespace(
        data_path="unused",
        output_dir=str(tmp_path / "out"),
        hidden_size=8,
        num_heads=2,
        num_layers=1,
        context_size=4,
        intermediate_size=16,
        untie_weights=False,
        checkpoint=checkpoint,
        key_paths=key_paths,
        tier_sample=tier_sample,
        batch_size=1,
        grad_accum_steps=1,
        learning_rate=learning_rate,
        min_lr=learning_rate / 10.0,
        max_steps=max_steps,
        warmup_steps=1,
        weight_decay=0.0,
        max_grad_norm=1e6,
        log_interval=1,
        save_interval=10_000,
        eval_interval=10_000,
        eval_steps=2,
        eval_all_tiers=False,
        wandb_project="test-proj",
        run_name="test-run",
        local_rank=-1,
        num_workers=0,
    )


# ─── integration: single-key reduces to standard 2-tier behavior ──────────

def test_single_key_gradient_combination(monkeypatch, tmp_path):
    """With one keyed tier (N=1), uniform 0.5 scaling gives:
        public: 0.5 * (grad_c1 + grad_c_{k+1})
        keyed:  0.5 * grad_c_{k+1}
    """
    model = _NTierModel(n_tiers=1, public_init=10.0, keyed_inits=[20.0])
    _install_common_patches(monkeypatch, model, n_tiers=1)

    args = _default_args(tmp_path, key_paths=["k1.json"], max_steps=1,
                         learning_rate=0.1)
    cmt.train(args)

    # Expected per-step grads before optimizer:
    #   After C1 backward:        public=2, keyed[0]=3
    #   After mask (tier 0):      public=2, keyed[0]=0
    #   After C_{k+1} backward:   public=2+5=7, keyed[0]=0+7=7
    #   After *= 0.5:             public=3.5, keyed[0]=3.5
    # SGD step lr=0.1: public=10-0.35=9.65, keyed[0]=20-0.35=19.65
    assert float(model.public.detach()) == pytest.approx(9.65)
    assert float(model.keyed[0].detach()) == pytest.approx(19.65)


# ─── integration: N=3 gradient math for each possible active tier ─────────

def _run_ntier_step(monkeypatch, tmp_path, *, n_tiers, active_idx,
                    tier_sample="round_robin"):
    """Run one training step with the given active tier and return final
    weights plus the captured event log."""
    model = _NTierModel(
        n_tiers=n_tiers,
        public_init=100.0,
        keyed_inits=[1000.0 + i for i in range(n_tiers)],
    )
    ctx = _install_common_patches(monkeypatch, model, n_tiers=n_tiers)

    key_paths = [f"k{i}.json" for i in range(n_tiers)]
    args = _default_args(
        tmp_path, key_paths=key_paths, max_steps=1,
        tier_sample=tier_sample, learning_rate=0.1,
    )

    # Force which tier gets picked this step by stubbing the sampler output.
    real_sample = cmt.sample_tier

    def _force(tiers, strategy, global_step, rng):
        tier = tiers[active_idx]
        tier.steps_sampled += 1
        return tier

    monkeypatch.setattr(cmt, "sample_tier", _force)

    cmt.train(args)
    return model, ctx


def test_gradient_combination_n3_active_middle(monkeypatch, tmp_path):
    """N=3, active tier K=1. Expected per-param gradients before the step:
        public       = 0.5 * (2 + 5) = 3.5                 # effective public
        keyed[0]     = 0.5 * 7      = 3.5                   # composite keyed
        keyed[1]     = 0.5 * 7      = 3.5                   # composite keyed (active)
        keyed[2]     = 0.5 * (3 + 7) = 5.0                  # effective public (K+1..N)
    """
    model, _ = _run_ntier_step(monkeypatch, tmp_path, n_tiers=3, active_idx=1)

    lr = 0.1
    assert float(model.public.detach()) == pytest.approx(100.0 - lr * 3.5)
    assert float(model.keyed[0].detach()) == pytest.approx(1000.0 - lr * 3.5)
    assert float(model.keyed[1].detach()) == pytest.approx(1001.0 - lr * 3.5)
    assert float(model.keyed[2].detach()) == pytest.approx(1002.0 - lr * 5.0)


def test_gradient_combination_n3_active_first(monkeypatch, tmp_path):
    """N=3, active tier K=0. Only keyed[0] is in the composite; keyed[1]
    and keyed[2] are treated as public on this step."""
    model, _ = _run_ntier_step(monkeypatch, tmp_path, n_tiers=3, active_idx=0)

    lr = 0.1
    assert float(model.public.detach()) == pytest.approx(100.0 - lr * 3.5)
    assert float(model.keyed[0].detach()) == pytest.approx(1000.0 - lr * 3.5)
    assert float(model.keyed[1].detach()) == pytest.approx(1001.0 - lr * 5.0)
    assert float(model.keyed[2].detach()) == pytest.approx(1002.0 - lr * 5.0)


def test_gradient_combination_n3_active_last(monkeypatch, tmp_path):
    """N=3, active tier K=2 (last). All three tiers are in the composite;
    there are no 'later' tiers to treat as public.
    """
    model, _ = _run_ntier_step(monkeypatch, tmp_path, n_tiers=3, active_idx=2)

    lr = 0.1
    assert float(model.public.detach()) == pytest.approx(100.0 - lr * 3.5)
    for i in range(3):
        assert float(model.keyed[i].detach()) == pytest.approx(
            (1000.0 + i) - lr * 3.5
        )


# ─── Phase 1 masking only covers tiers 0..active_idx ──────────────────────

def test_phase1_masks_only_composite_keyed_tiers(monkeypatch, tmp_path):
    """Phase 1 (after C1 backward) must mask the C1 gradient only for tiers
    0..active_idx. Later tiers must receive the full C1 contribution so they
    can participate in the effective-public averaging in Phase 2.
    """
    model = _NTierModel(n_tiers=4, public_init=0.0, keyed_inits=[0.0] * 4)
    _install_common_patches(monkeypatch, model, n_tiers=4)

    key_paths = [f"k{i}.json" for i in range(4)]
    args = _default_args(tmp_path, key_paths=key_paths, max_steps=1,
                         learning_rate=0.0)  # zero lr: weights don't move, focus is events

    # Force active tier = 1 (so we expect masks for tiers {0, 1} only).
    def _force(tiers, strategy, global_step, rng):
        tier = tiers[1]
        tier.steps_sampled += 1
        return tier

    monkeypatch.setattr(cmt, "sample_tier", _force)

    # Collect mask-tier events specifically.
    events_post = []
    real_mask = cmt.mask_keyed_gradients

    def _tracked_mask(raw_model, key, plan=None):
        events_post.append(plan.tier_idx)
        real_mask(raw_model, key, plan=plan)

    monkeypatch.setattr(cmt, "mask_keyed_gradients", _tracked_mask)

    cmt.train(args)

    # Exactly one mask call per tier in {0, 1}, and none for {2, 3}.
    assert sorted(events_post) == [0, 1]


# ─── event ordering: apply → backward → (combine) → unapply → swap → step ─

def test_event_order_in_training_step(monkeypatch, tmp_path):
    """Verify the sequence of permutation / gradient / optimizer operations
    inside one training step for N=2 with active tier K=1 (so both keys are
    applied cumulatively)."""
    model = _NTierModel(n_tiers=2, public_init=0.0, keyed_inits=[0.0, 0.0])
    ctx = _install_common_patches(monkeypatch, model, n_tiers=2)

    def _force(tiers, strategy, global_step, rng):
        tier = tiers[1]
        tier.steps_sampled += 1
        return tier

    monkeypatch.setattr(cmt, "sample_tier", _force)

    args = _default_args(
        tmp_path, key_paths=["k0.json", "k1.json"], max_steps=1,
        learning_rate=0.0,
    )
    cmt.train(args)

    events = ctx["events"]
    # Expected event skeleton for active_idx=1, N=2:
    #   Phase 1: mask tier 0, mask tier 1   (composite-keyed grads zeroed)
    #   Phase 2: apply tier 0, apply tier 1 (cumulative key application)
    #            <C_{k+1} backward>         (not recorded)
    #            <global 0.5 scaling>       (not recorded — iterates p.grad)
    #   Phase 3: unapply tier 1, unapply tier 0, swap tier 1, swap tier 0
    #            optimizer_step
    assert events == [
        "mask_0", "mask_1",
        "apply", "apply",
        "unapply", "unapply",
        "swap", "swap",
        "optimizer_step",
    ]


# ─── no preserving-public helper is used ─────────────────────────────────

def test_plain_optimizer_step_used(monkeypatch, tmp_path):
    """The cumulative script must call the regular `optimizer.step()`, not
    `adamw_step_preserving_public`. After the algorithm change every position
    has a valid gradient, so the preserving variant is unnecessary and
    references to it should have been removed from the module entirely."""
    assert not hasattr(cmt, "adamw_step_preserving_public"), (
        "adamw_step_preserving_public should no longer be imported in the "
        "cumulative module"
    )
    assert not hasattr(cmt, "build_adamw_update_masks"), (
        "build_adamw_update_masks should no longer be imported in the "
        "cumulative module"
    )

    model = _NTierModel(n_tiers=2, public_init=0.0, keyed_inits=[0.0, 0.0])
    ctx = _install_common_patches(monkeypatch, model, n_tiers=2)

    args = _default_args(
        tmp_path, key_paths=["k0.json", "k1.json"], max_steps=3,
        learning_rate=0.0,
    )
    cmt.train(args)

    # One optimizer.step per training step.
    assert ctx["events"].count("optimizer_step") == 3


# ─── TierInfo no longer carries adamw_update_masks ───────────────────────

def test_tierinfo_dataclass_has_no_adamw_masks_field():
    """`adamw_update_masks` was removed from TierInfo as part of the
    rewrite — check that no stale code is still setting or reading it."""
    fields = cmt.TierInfo.__dataclass_fields__
    assert "adamw_update_masks" not in fields
    # Core fields are still present.
    for name in ("tier_id", "tier_idx", "key", "swap_plan", "mask_plan",
                 "steps_sampled"):
        assert name in fields
