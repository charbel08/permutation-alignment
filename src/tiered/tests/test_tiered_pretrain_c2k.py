"""Tests for periodic-C2 pretraining entrypoint behavior.

These tests focus on:
1. C2 cadence (every K steps)
2. C1-frame optimizer stepping on C2 steps (unapply + swap before step)
3. Dynamic FLOPs accounting and logging
4. Resume behavior for c2_passes_cumulative / cumulative_flops
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tiered.train.pretrain import tiered_pretrain_c2k as c2k


class _TinyTokenizer:
    eos_token = "<eos>"
    pad_token = "<eos>"

    def save_pretrained(self, path):
        return None


class _TinyCollator:
    def __init__(self, tokenizer=None, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, batch):
        input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


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

    def remove_columns(self, cols_to_remove):
        for item in self._items:
            for c in cols_to_remove:
                item.pop(c, None)
        self.column_names = [c for c in self.column_names if c not in cols_to_remove]
        return self


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=8, context_size=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self._emb = torch.nn.Embedding(vocab_size, 2)
        self.config = SimpleNamespace(max_position_embeddings=context_size)

    def gradient_checkpointing_enable(self, **kwargs):
        return None

    def get_input_embeddings(self):
        return self._emb

    def save_pretrained(self, path):
        return None

    def forward(self, input_ids, labels=None):
        bsz, seq = input_ids.shape
        vocab = self._emb.weight.shape[0]
        logits = torch.zeros((bsz, seq, vocab), dtype=torch.float32, device=input_ids.device) + self.weight
        loss = self.weight * 1.0
        return SimpleNamespace(loss=loss, logits=logits)


class _FakeScheduler:
    def __init__(self, *args, **kwargs):
        self._state = {}

    def step(self):
        return None

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)


class _FakePbar:
    def __init__(self, *args, **kwargs):
        self.n = kwargs.get("initial", 0)

    def update(self, n=1):
        self.n += n

    def set_postfix(self, *args, **kwargs):
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

    def define_metric(self, *args, **kwargs):
        return None

    def log(self, data):
        self.logged.append(dict(data))

    def finish(self):
        return None


def _default_args(tmp_path: Path, *, max_steps=5, c2_every_k=2, checkpoint=None):
    return SimpleNamespace(
        data_path="unused",
        output_dir=str(tmp_path / "out"),
        hidden_size=8,
        num_heads=2,
        num_layers=1,
        context_size=4,
        intermediate_size=16,
        checkpoint=checkpoint,
        key_path="unused_key.json",
        batch_size=1,
        grad_accum_steps=1,
        learning_rate=1e-3,
        min_lr=1e-4,
        max_steps=max_steps,
        warmup_steps=1,
        weight_decay=0.0,
        max_grad_norm=1.0,
        c2_every_k=c2_every_k,
        c2_metrics_every_k=None,
        log_interval=1,
        save_interval=10_000,
        eval_interval=10_000,
        eval_steps=2,
        wandb_project="test-proj",
        run_name="test-run",
        local_rank=-1,
        num_workers=0,
        seed=123,
    )


def _install_common_patches(monkeypatch, tmp_path: Path):
    events = []
    save_calls = []

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(c2k, "wandb", fake_wandb)
    monkeypatch.setattr(c2k.torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(c2k.torch, "compile", lambda m: m)
    monkeypatch.setattr(c2k, "tqdm", lambda *args, **kwargs: _FakePbar(*args, **kwargs))

    monkeypatch.setattr(c2k.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _TinyTokenizer())
    monkeypatch.setattr(c2k, "DataCollatorForLanguageModeling", _TinyCollator)
    monkeypatch.setattr(c2k, "load_from_disk", lambda *_args, **_kwargs: {"train": _TinyDataset()})
    monkeypatch.setattr(c2k, "load_model", lambda **_kwargs: _TinyModel())
    monkeypatch.setattr(
        c2k.GPTNeoForCausalLMTiered,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _TinyModel()),
    )

    monkeypatch.setattr(c2k, "count_total_parameters", lambda _m: 10)
    monkeypatch.setattr(c2k, "count_trainable_parameters", lambda _m: 10)
    monkeypatch.setattr(c2k, "count_swappable_parameters", lambda _m, _p: {"total": 2, "attention": 1, "mlp": 1})
    monkeypatch.setattr(c2k, "count_max_swappable_parameters", lambda _m: {"total": 4, "attention": 2, "mlp": 2})
    monkeypatch.setattr(c2k, "detect_gpu_peak_flops", lambda _d: (0.0, "cpu"))
    monkeypatch.setattr(c2k, "get_gpu_memory_stats", lambda _d: {})

    key_obj = SimpleNamespace(
        attn_heads=[((0, 0), (0, 1))],
        attn_out_heads=[],
        mlp_cols=[((0, 0), (0, 1))],
        mlp_up_cols=[],
        mlp_down_cols=[],
    )
    monkeypatch.setattr(c2k, "load_key", lambda _p: key_obj)
    swap_plan = SimpleNamespace(attn_ops=[], attn_out_ops=[], mlp_ops=[], mlp_up_ops=[], mlp_down_ops=[])
    mask_plan = SimpleNamespace(
        keyed_attn_indices={},
        keyed_attn_out_indices={},
        keyed_mlp_indices={},
        keyed_mlp_up_indices={},
        keyed_mlp_down_indices={},
    )
    monkeypatch.setattr(c2k, "build_swap_plan", lambda *_args, **_kwargs: swap_plan)
    monkeypatch.setattr(c2k, "build_mask_plan", lambda *_args, **_kwargs: mask_plan)

    def _rec(name):
        def _fn(*args, **kwargs):
            events.append(name)

        return _fn

    monkeypatch.setattr(c2k, "apply_permutation", _rec("apply"))
    monkeypatch.setattr(c2k, "scale_public_gradients", _rec("scale"))
    monkeypatch.setattr(c2k, "unapply_permutation", _rec("unapply"))
    monkeypatch.setattr(c2k, "swap_gradients", _rec("swap"))
    monkeypatch.setattr(c2k, "mask_keyed_gradients", lambda *_args, **_kwargs: None)

    class _FakeOptimizer:
        def __init__(self, params, lr=0.0, **kwargs):
            self.param_groups = [{"lr": lr}]
            self._state = {}

        def zero_grad(self):
            return None

        def step(self):
            events.append("optimizer_step")

        def state_dict(self):
            return dict(self._state)

        def load_state_dict(self, state):
            self._state = dict(state)

    monkeypatch.setattr(c2k.optim, "AdamW", _FakeOptimizer)
    monkeypatch.setattr(c2k, "LinearLR", _FakeScheduler)
    monkeypatch.setattr(c2k, "CosineAnnealingLR", _FakeScheduler)
    monkeypatch.setattr(c2k, "SequentialLR", _FakeScheduler)
    monkeypatch.setattr(c2k.torch.nn.utils, "clip_grad_norm_", lambda *args, **kwargs: torch.tensor(0.0))

    def _fake_save_checkpoint(model, tokenizer, optimizer, path, **kwargs):
        save_calls.append({"path": path, **kwargs})

    monkeypatch.setattr(c2k, "save_checkpoint", _fake_save_checkpoint)

    return {"events": events, "save_calls": save_calls, "wandb": fake_wandb}


def _train_logs(fake_wandb):
    return [x for x in fake_wandb.logged if "train/ran_c2" in x]


def test_parse_args_rejects_nonpositive_c2_every_k(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--data_path",
            "d",
            "--output_dir",
            "o",
            "--key_path",
            "k",
            "--c2_every_k",
            "0",
        ],
    )
    with pytest.raises(SystemExit):
        c2k.parse_args()


def test_parse_args_rejects_nonpositive_c2_metrics_every_k(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--data_path",
            "d",
            "--output_dir",
            "o",
            "--key_path",
            "k",
            "--c2_metrics_every_k",
            "0",
        ],
    )
    with pytest.raises(SystemExit):
        c2k.parse_args()


def test_c2_cadence_and_dynamic_flops_logging(monkeypatch, tmp_path):
    ctx = _install_common_patches(monkeypatch, tmp_path)
    args = _default_args(tmp_path, max_steps=5, c2_every_k=2)
    c2k.train(args)

    logs = _train_logs(ctx["wandb"])
    assert len(logs) == 5

    ran_c2 = [int(x["train/ran_c2"]) for x in logs]
    assert ran_c2 == [1, 0, 1, 0, 1]
    have_c2 = ["loss_c2" in x for x in logs]
    assert have_c2 == [True, False, True, False, True]
    assert ["acc_c2" in x for x in logs] == have_c2
    assert ["ppl_c2" in x for x in logs] == have_c2
    for x in logs:
        if "loss_c2" in x:
            assert float(x["loss_c2"]) == float(x["loss_c2"])

    c2_cum = [int(x["perf/c2_passes_cumulative"]) for x in logs]
    assert c2_cum == [1, 1, 2, 2, 3]

    baseline = float(ctx["wandb"].config["compute/flops_per_step_baseline"])
    with_c2 = float(ctx["wandb"].config["compute/flops_per_step_with_c2"])

    expected_step_flops = [with_c2 if r == 1 else baseline for r in ran_c2]
    got_step_flops = [float(x["perf/flops_per_step"]) for x in logs]
    assert got_step_flops == pytest.approx(expected_step_flops)

    expected_cum = []
    total = 0.0
    for v in expected_step_flops:
        total += v
        expected_cum.append(total)
    got_cum = [float(x["perf/cumulative_flops"]) for x in logs]
    assert got_cum == pytest.approx(expected_cum)

    expected_pct = [100.0, 50.0, 200.0 / 3.0, 50.0, 60.0]
    got_pct = [float(x["perf/flops_increase_pct_vs_baseline"]) for x in logs]
    assert got_pct == pytest.approx(expected_pct)

    summary = ctx["wandb"].run.summary
    assert int(summary["final/c2_passes_cumulative"]) == 3
    assert float(summary["final/flops_increase_pct_vs_baseline"]) == pytest.approx(60.0)


def test_step_order_keeps_optimizer_in_c1_frame(monkeypatch, tmp_path):
    ctx = _install_common_patches(monkeypatch, tmp_path)
    args = _default_args(tmp_path, max_steps=3, c2_every_k=2)
    c2k.train(args)

    assert ctx["events"] == [
        "apply",
        "scale",
        "unapply",
        "swap",
        "optimizer_step",
        "optimizer_step",
        "apply",
        "scale",
        "unapply",
        "swap",
        "optimizer_step",
    ]


def test_non_c2_steps_skip_keyed_path(monkeypatch, tmp_path):
    ctx = _install_common_patches(monkeypatch, tmp_path)
    args = _default_args(tmp_path, max_steps=4, c2_every_k=3)
    c2k.train(args)

    # C2 updates run on steps 1 and 4 only. With c2_metrics_every_k disabled,
    # non-C2 steps do not run keyed forwards.
    assert ctx["events"].count("apply") == 2
    assert ctx["events"].count("scale") == 2
    assert ctx["events"].count("unapply") == 2
    assert ctx["events"].count("swap") == 2
    assert ctx["events"].count("optimizer_step") == 4


def test_c2_metrics_every_k_runs_forward_only_metrics_passes(monkeypatch, tmp_path):
    ctx = _install_common_patches(monkeypatch, tmp_path)
    args = _default_args(tmp_path, max_steps=5, c2_every_k=3)
    args.c2_metrics_every_k = 2
    c2k.train(args)

    logs = _train_logs(ctx["wandb"])
    assert [int(x["train/ran_c2"]) for x in logs] == [1, 0, 0, 1, 0]
    # Metrics on C2 steps (1,4) and on non-C2 scheduled metrics steps (3,5).
    assert ["loss_c2" in x for x in logs] == [True, False, True, True, True]

    assert ctx["events"].count("apply") == 4
    assert ctx["events"].count("unapply") == 4
    assert ctx["events"].count("scale") == 2
    assert ctx["events"].count("swap") == 2


def test_k1_runs_c2_on_every_step(monkeypatch, tmp_path):
    ctx = _install_common_patches(monkeypatch, tmp_path)
    args = _default_args(tmp_path, max_steps=4, c2_every_k=1)
    c2k.train(args)

    logs = _train_logs(ctx["wandb"])
    assert [int(x["train/ran_c2"]) for x in logs] == [1, 1, 1, 1]
    assert [int(x["perf/c2_passes_cumulative"]) for x in logs] == [1, 2, 3, 4]
    assert ctx["events"].count("apply") == 4
    assert ctx["events"].count("scale") == 4
    assert ctx["events"].count("unapply") == 4
    assert ctx["events"].count("swap") == 4


def test_resume_continues_cadence_from_global_step(monkeypatch, tmp_path):
    ctx = _install_common_patches(monkeypatch, tmp_path)

    ckpt_dir = tmp_path / "resume_ckpt_cadence"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": {},
            "scheduler": {},
            "global_step": 3,
            "wandb_run_id": "resume-id",
        },
        ckpt_dir / "training_state.pt",
    )

    args = _default_args(tmp_path, max_steps=6, c2_every_k=3, checkpoint=str(ckpt_dir))
    c2k.train(args)

    logs = _train_logs(ctx["wandb"])
    # Trained steps are 4,5,6. For K=3 cadence is 1,4,7...
    assert [int(x["train/ran_c2"]) for x in logs] == [1, 0, 0]
    assert [int(x["train/step"]) for x in logs] == [4, 5, 6]

    summary = ctx["wandb"].run.summary
    # Inferred 1 prior pass (step 1), then one new pass at step 4.
    assert int(summary["final/c2_passes_cumulative"]) == 2


def test_compile_wrapper_and_raw_model_share_weights(monkeypatch, tmp_path):
    ctx = _install_common_patches(monkeypatch, tmp_path)

    class _CompileWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            return self.inner(*args, **kwargs)

    monkeypatch.setattr(c2k.torch, "compile", lambda m: _CompileWrapper(m))

    def _apply(model, key, plan=None):
        model.weight.data.add_(10.0)

    def _unapply(model, key, plan=None):
        model.weight.data.sub_(10.0)

    monkeypatch.setattr(c2k, "apply_permutation", _apply)
    monkeypatch.setattr(c2k, "unapply_permutation", _unapply)
    monkeypatch.setattr(c2k, "scale_public_gradients", lambda *_a, **_k: None)
    monkeypatch.setattr(c2k, "swap_gradients", lambda *_a, **_k: None)

    args = _default_args(tmp_path, max_steps=1, c2_every_k=1)
    c2k.train(args)

    logs = _train_logs(ctx["wandb"])
    assert len(logs) == 1
    assert float(logs[0]["loss_c2"]) > float(logs[0]["loss_c1"]) + 5.0


def test_resume_infers_c2_passes_when_absent(monkeypatch, tmp_path):
    ctx = _install_common_patches(monkeypatch, tmp_path)

    ckpt_dir = tmp_path / "resume_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / "training_state.pt"
    torch.save(
        {
            "optimizer": {},
            "scheduler": {},
            "global_step": 5,
            "wandb_run_id": "resume-id",
        },
        state_path,
    )

    args = _default_args(tmp_path, max_steps=5, c2_every_k=2, checkpoint=str(ckpt_dir))
    c2k.train(args)

    # inferred passes for global_step=5, K=2 => steps 1,3,5 = 3 passes
    summary = ctx["wandb"].run.summary
    assert int(summary["final/c2_passes_cumulative"]) == 3
    assert float(summary["final/flops_increase_pct_vs_baseline"]) == pytest.approx(60.0)

    baseline = float(ctx["wandb"].config["compute/flops_per_step_baseline"])
    expected_total_flops = (5 + 3) * baseline
    assert float(summary["final/total_flops"]) == pytest.approx(expected_total_flops)

    final_save = ctx["save_calls"][-1]
    assert int(final_save["c2_passes_cumulative"]) == 3
    assert float(final_save["cumulative_flops"]) == pytest.approx(expected_total_flops)


def test_resume_uses_explicit_checkpointed_counters(monkeypatch, tmp_path):
    ctx = _install_common_patches(monkeypatch, tmp_path)

    ckpt_dir = tmp_path / "resume_ckpt_explicit"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / "training_state.pt"
    torch.save(
        {
            "optimizer": {},
            "scheduler": {},
            "global_step": 5,
            "wandb_run_id": "resume-id",
            "c2_passes_cumulative": 7,
            "cumulative_flops": 12345.0,
        },
        state_path,
    )

    args = _default_args(tmp_path, max_steps=5, c2_every_k=2, checkpoint=str(ckpt_dir))
    c2k.train(args)

    summary = ctx["wandb"].run.summary
    assert int(summary["final/c2_passes_cumulative"]) == 7
    assert float(summary["final/total_flops"]) == pytest.approx(12345.0)
    assert float(summary["final/flops_increase_pct_vs_baseline"]) == pytest.approx(140.0)


class _TwoParamModel(torch.nn.Module):
    def __init__(self, public_init=10.0, keyed_init=20.0, vocab_size=8, context_size=4):
        super().__init__()
        self.public = torch.nn.Parameter(torch.tensor(float(public_init)))
        self.keyed = torch.nn.Parameter(torch.tensor(float(keyed_init)))
        self._emb = torch.nn.Embedding(vocab_size, 2)
        self.config = SimpleNamespace(max_position_embeddings=context_size)
        self.in_c2 = False

    def gradient_checkpointing_enable(self, **kwargs):
        return None

    def get_input_embeddings(self):
        return self._emb

    def save_pretrained(self, path):
        return None

    def forward(self, input_ids, labels=None):
        bsz, seq = input_ids.shape
        vocab = self._emb.weight.shape[0]
        logits = torch.zeros((bsz, seq, vocab), dtype=torch.float32, device=input_ids.device)
        if self.in_c2:
            loss = 5.0 * self.public + 7.0 * self.keyed
        else:
            loss = 2.0 * self.public + 3.0 * self.keyed
        return SimpleNamespace(loss=loss, logits=logits)


def _install_gradient_math_patches(monkeypatch, model: _TwoParamModel, *, drift=0.0):
    fake_wandb = _FakeWandb()
    save_calls = []
    optimizers = []

    monkeypatch.setattr(c2k, "wandb", fake_wandb)
    monkeypatch.setattr(c2k.torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(c2k.torch, "compile", lambda m: m)
    monkeypatch.setattr(c2k, "tqdm", lambda *args, **kwargs: _FakePbar(*args, **kwargs))
    monkeypatch.setattr(c2k.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _TinyTokenizer())
    monkeypatch.setattr(c2k, "DataCollatorForLanguageModeling", _TinyCollator)
    monkeypatch.setattr(c2k, "load_from_disk", lambda *_args, **_kwargs: {"train": _TinyDataset()})
    monkeypatch.setattr(c2k, "load_model", lambda **_kwargs: model)
    monkeypatch.setattr(
        c2k.GPTNeoForCausalLMTiered,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: model),
    )
    monkeypatch.setattr(c2k, "count_total_parameters", lambda _m: 10)
    monkeypatch.setattr(c2k, "count_trainable_parameters", lambda _m: 10)
    monkeypatch.setattr(c2k, "count_swappable_parameters", lambda _m, _p: {"total": 2, "attention": 1, "mlp": 1})
    monkeypatch.setattr(c2k, "count_max_swappable_parameters", lambda _m: {"total": 4, "attention": 2, "mlp": 2})
    monkeypatch.setattr(c2k, "detect_gpu_peak_flops", lambda _d: (0.0, "cpu"))
    monkeypatch.setattr(c2k, "get_gpu_memory_stats", lambda _d: {})

    key_obj = SimpleNamespace(
        attn_heads=[],
        attn_out_heads=[],
        mlp_cols=[],
        mlp_up_cols=[],
        mlp_down_cols=[],
    )
    monkeypatch.setattr(c2k, "load_key", lambda _p: key_obj)
    monkeypatch.setattr(
        c2k,
        "build_swap_plan",
        lambda *_args, **_kwargs: SimpleNamespace(attn_ops=[], attn_out_ops=[], mlp_ops=[], mlp_up_ops=[], mlp_down_ops=[]),
    )
    monkeypatch.setattr(
        c2k,
        "build_mask_plan",
        lambda *_args, **_kwargs: SimpleNamespace(
            keyed_attn_indices={},
            keyed_attn_out_indices={},
            keyed_mlp_indices={},
            keyed_mlp_up_indices={},
            keyed_mlp_down_indices={},
        ),
    )
    monkeypatch.setattr(
        c2k,
        "build_adamw_update_masks",
        lambda raw_model, _plans: {raw_model.keyed: torch.zeros_like(raw_model.keyed, dtype=torch.bool)},
    )

    def _apply(raw_model, key, plan=None):
        raw_model.in_c2 = True

    def _unapply(raw_model, key, plan=None):
        raw_model.in_c2 = False

    def _mask_keyed(raw_model, key, plan=None):
        assert raw_model.keyed.grad is not None
        raw_model.keyed.grad.zero_()

    def _scale_public(raw_model, key, scale=0.5, plan=None):
        assert raw_model.public.grad is not None
        raw_model.public.grad.mul_(scale)

    monkeypatch.setattr(c2k, "apply_permutation", _apply)
    monkeypatch.setattr(c2k, "unapply_permutation", _unapply)
    monkeypatch.setattr(c2k, "swap_gradients", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(c2k, "mask_keyed_gradients", _mask_keyed)
    monkeypatch.setattr(c2k, "scale_public_gradients", _scale_public)

    class _DriftOptimizer:
        def __init__(self, params, lr=0.0, **kwargs):
            self.param_groups = []
            self.state = {}
            self.step_calls = 0
            self._drift = drift
            for group in params:
                copied = dict(group)
                copied["params"] = list(copied["params"])
                copied.setdefault("lr", lr)
                self.param_groups.append(copied)
                for p in copied["params"]:
                    self.state.setdefault(p, {})
            optimizers.append(self)

        def zero_grad(self):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.zero_()

        def step(self):
            self.step_calls += 1
            for group in self.param_groups:
                lr = group["lr"]
                wd = group.get("weight_decay", 0.0)
                for p in group["params"]:
                    if self._drift:
                        p.data.add_(torch.ones_like(p.data), alpha=-(lr * self._drift))
                    if wd:
                        p.data.add_(p.data, alpha=-(lr * wd))
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-lr)

        def state_dict(self):
            return {"state": {}, "param_groups": [{"lr": g["lr"]} for g in self.param_groups]}

        def load_state_dict(self, state):
            return None

    monkeypatch.setattr(c2k.optim, "AdamW", _DriftOptimizer)
    monkeypatch.setattr(c2k, "LinearLR", _FakeScheduler)
    monkeypatch.setattr(c2k, "CosineAnnealingLR", _FakeScheduler)
    monkeypatch.setattr(c2k, "SequentialLR", _FakeScheduler)
    monkeypatch.setattr(c2k.torch.nn.utils, "clip_grad_norm_", lambda *args, **kwargs: torch.tensor(0.0))
    monkeypatch.setattr(
        c2k,
        "save_checkpoint",
        lambda model, tokenizer, optimizer, path, **kwargs: save_calls.append({"path": path, **kwargs}),
    )

    return {"wandb": fake_wandb, "save_calls": save_calls, "optimizers": optimizers}


def test_c2_step_gradient_combination_matches_expected_weights(monkeypatch, tmp_path):
    model = _TwoParamModel(public_init=10.0, keyed_init=20.0)
    ctx = _install_gradient_math_patches(monkeypatch, model, drift=0.0)

    args = _default_args(tmp_path, max_steps=1, c2_every_k=1)
    args.learning_rate = 0.1
    c2k.train(args)

    # C1 grads: public=2, keyed=3 -> mask keyed => keyed=0
    # C2 grads: public+=5, keyed+=7 => public=7, keyed=7
    # scale_public_gradients(0.5) => public=3.5, keyed=7
    # SGD-like step lr=0.1 => public=10-0.35=9.65, keyed=20-0.7=19.3
    assert float(model.public.detach()) == pytest.approx(9.65)
    assert float(model.keyed.detach()) == pytest.approx(19.3)
    assert model.in_c2 is False

    logs = _train_logs(ctx["wandb"])
    assert len(logs) == 1
    assert int(logs[0]["train/ran_c2"]) == 1


def test_non_c2_step_preserves_keyed_params_against_optimizer_drift(monkeypatch, tmp_path):
    model = _TwoParamModel(public_init=10.0, keyed_init=20.0)
    ctx = _install_gradient_math_patches(monkeypatch, model, drift=0.25)

    real_preserve = c2k.adamw_step_preserving_public
    snapshots = []

    def _wrapped_preserve(optimizer, keyed_masks):
        before = model.keyed.detach().clone()
        real_preserve(optimizer, keyed_masks)
        after = model.keyed.detach().clone()
        snapshots.append((before, after))

    monkeypatch.setattr(c2k, "adamw_step_preserving_public", _wrapped_preserve)

    args = _default_args(tmp_path, max_steps=2, c2_every_k=2)
    args.learning_rate = 0.1
    c2k.train(args)

    logs = _train_logs(ctx["wandb"])
    assert [int(x["train/ran_c2"]) for x in logs] == [1, 0]

    # Second step is non-C2; keyed params should be restored exactly after the
    # preserving step, despite optimizer drift in step().
    assert len(snapshots) == 1
    before, after = snapshots[0]
    assert torch.allclose(before, after)

    assert len(ctx["optimizers"]) == 1
    assert int(ctx["optimizers"][0].step_calls) == 2
