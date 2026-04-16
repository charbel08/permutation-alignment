"""Tests for extraction attack experiment behavior.

These tests cover:
1. CLI defaults and required args
2. Data split invariants (train-only fractioning for optimization)
3. Deterministic subset sampling via subset_seed
4. Scheduler configuration in fixed-step mode (warmup + cosine decay)
5. Fixed-step training reaches exactly max_steps (no early stop path)
6. W&B logging contains both memo_train/* and memo_test/* metrics
"""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tiered.train.finetune import extraction_attack as ea


class _TinyTokenizer:
    eos_token = "<eos>"
    pad_token = "<eos>"


class _TinyCollator:
    def __init__(self, tokenizer=None, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, batch):
        input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, items, origin=None):
        self._items = list(items)
        self.column_names = list(self._items[0].keys()) if self._items else ["input_ids", "attention_mask"]
        self._origin = origin or self
        if origin is None:
            self.last_shuffle_seed = None
            self.last_selected_ids = None

    @staticmethod
    def build(n_items=16, seq_len=4, vocab=32):
        items = []
        for i in range(n_items):
            items.append(
                {
                    "example_id": i,
                    "input_ids": [(i + j) % vocab for j in range(seq_len)],
                    "attention_mask": [1] * seq_len,
                }
            )
        return _TinyDataset(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def remove_columns(self, cols_to_remove):
        out = []
        for item in self._items:
            new_item = dict(item)
            for c in cols_to_remove:
                new_item.pop(c, None)
            out.append(new_item)
        return _TinyDataset(out, origin=self._origin)

    def shuffle(self, seed):
        self._origin.last_shuffle_seed = seed
        rng = random.Random(seed)
        idx = list(range(len(self._items)))
        rng.shuffle(idx)
        out = [self._items[i] for i in idx]
        return _TinyDataset(out, origin=self._origin)

    def select(self, indices):
        idx_list = list(indices)
        out = [self._items[i] for i in idx_list]
        self._origin.last_selected_ids = [x["example_id"] for x in out if "example_id" in x]
        return _TinyDataset(out, origin=self._origin)


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=32, context_size=8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(max_position_embeddings=context_size)
        self._emb = torch.nn.Embedding(vocab_size, 4)

    def to(self, device):
        return self

    def save_pretrained(self, path):
        return None

    def forward(self, input_ids, labels=None):
        bsz, seq = input_ids.shape
        vocab = self._emb.num_embeddings
        logits = torch.zeros((bsz, seq, vocab), dtype=torch.float32, device=input_ids.device) + self.weight
        loss = self.weight * 1.0
        return SimpleNamespace(loss=loss, logits=logits)


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


def _write_bio_metadata(path: Path, n_train_people=4, n_test_people=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    train_people = [f"train_{i}" for i in range(n_train_people)]
    test_people = [f"test_{i}" for i in range(n_test_people)]
    bios = []
    for pid in train_people + test_people:
        bios.append(
            {
                "person_id": pid,
                "target_attr": "age",
                "age": 42,
                "profession": "engineer",
                "hobby": "reading",
                "salary_str": "$100,000",
                "prefix": f"Name: {pid}. Age: ",
                "text": f"Name: {pid}. Age: 42. Profession: engineer.",
            }
        )
    payload = {"train_people": train_people, "test_people": test_people, "bios": bios}
    path.write_text(json.dumps(payload))
    return payload


def _default_args(tmp_path: Path, **overrides):
    args = SimpleNamespace(
        model_checkpoint="unused-model",
        key_path=None,
        tiered_checkpoint=None,
        private_data="unused-dataset",
        data_fraction=0.5,
        subset_seed=42,
        output_dir=str(tmp_path / "out"),
        batch_size=2,
        learning_rate=1e-3,
        min_lr=1e-4,
        max_steps=4,
        warmup_steps=2,
        weight_decay=0.0,
        max_grad_norm=1.0,
        eval_interval=1,
        log_interval=1,
        save_interval=10_000,
        wandb_project="test-proj",
        run_name="test-run",
        num_workers=0,
        bio_metadata=str(tmp_path / "bios_metadata.json"),
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _default_memo_metrics(top1):
    return {
        "top1_acc": float(top1),
        "exact_match": float(top1),
        "contains": float(top1),
        "prefix_match": float(top1),
        "age/top1_acc": float(top1),
        "age/exact_match": float(top1),
        "age/contains": float(top1),
        "age/prefix_match": float(top1),
    }


def _install_common_patches(monkeypatch, args, *, dataset_obj=None, load_data=None, memo_fn=None):
    fake_wandb = _FakeWandb()
    save_calls = []

    monkeypatch.setattr(ea, "parse_args", lambda: args)
    monkeypatch.setattr(ea, "wandb", fake_wandb)
    monkeypatch.setattr(ea, "tqdm", lambda *a, **k: _FakePbar(*a, **k))
    monkeypatch.setattr(ea, "DataCollatorForLanguageModeling", _TinyCollator)
    monkeypatch.setattr(ea.AutoTokenizer, "from_pretrained", lambda *a, **k: _TinyTokenizer())
    monkeypatch.setattr(
        ea.GPTNeoForCausalLMTiered,
        "from_pretrained",
        staticmethod(lambda *a, **k: _TinyModel()),
    )
    monkeypatch.setattr(ea, "_bio_value_span", lambda *a, **k: (1, 2))
    monkeypatch.setattr(ea, "save_checkpoint", lambda *a, **k: save_calls.append(dict(k)))
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    if memo_fn is None:
        monkeypatch.setattr(ea, "evaluate_memorization", lambda *a, **k: _default_memo_metrics(0.0))
    else:
        monkeypatch.setattr(ea, "evaluate_memorization", memo_fn)

    if load_data is not None:
        monkeypatch.setattr(ea, "load_from_disk", load_data)
    else:
        assert dataset_obj is not None
        monkeypatch.setattr(ea, "load_from_disk", lambda _p: {"train": dataset_obj})

    return {"wandb": fake_wandb, "save_calls": save_calls}


def _run_main_and_get_selected_ids(monkeypatch, tmp_path, subset_seed):
    _write_bio_metadata(tmp_path / "bios_metadata.json")
    ds = _TinyDataset.build(n_items=40)
    args = _default_args(
        tmp_path,
        subset_seed=subset_seed,
        data_fraction=0.25,
        max_steps=1,
        eval_interval=1,
        log_interval=1000,
    )
    _install_common_patches(monkeypatch, args, dataset_obj=ds)
    ea.main()
    return ds.last_selected_ids


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--model_checkpoint",
            "m",
            "--private_data",
            "d",
            "--data_fraction",
            "0.1",
            "--output_dir",
            "o",
            "--max_steps",
            "12",
            "--bio_metadata",
            "b",
        ],
    )
    args = ea.parse_args()
    assert args.max_steps == 12
    assert args.subset_seed == 42
    assert not hasattr(args, "target_memo")
    assert not hasattr(args, "disable_early_stopping")
    assert not hasattr(args, "patience")


def test_main_requires_bio_metadata(monkeypatch, tmp_path):
    args = _default_args(tmp_path, bio_metadata=None)
    monkeypatch.setattr(ea, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="--bio_metadata is required"):
        ea.main()


def test_main_requires_private_train_split(monkeypatch, tmp_path):
    _write_bio_metadata(tmp_path / "bios_metadata.json")
    args = _default_args(tmp_path)
    _install_common_patches(
        monkeypatch,
        args,
        load_data=lambda _p: {"test": _TinyDataset.build(n_items=8)},
    )
    with pytest.raises(ValueError, match="must contain a 'train' split"):
        ea.main()


def test_subset_sampling_is_deterministic_by_seed(monkeypatch, tmp_path):
    ids_seed_42_a = _run_main_and_get_selected_ids(monkeypatch, tmp_path / "run_a", 42)
    ids_seed_42_b = _run_main_and_get_selected_ids(monkeypatch, tmp_path / "run_b", 42)
    ids_seed_43 = _run_main_and_get_selected_ids(monkeypatch, tmp_path / "run_c", 43)

    assert ids_seed_42_a == ids_seed_42_b
    assert ids_seed_42_a != ids_seed_43


def test_scheduler_uses_warmup_and_cosine_decay(monkeypatch, tmp_path):
    _write_bio_metadata(tmp_path / "bios_metadata.json")
    ds = _TinyDataset.build(n_items=5)
    args = _default_args(
        tmp_path,
        data_fraction=1.0,
        batch_size=2,
        max_steps=11,
        warmup_steps=3,
        eval_interval=1000,
        log_interval=1000,
    )

    captured = {}

    class _FakeScheduler:
        def __init__(self, *a, **k):
            captured.setdefault("calls", []).append((a, k))

        def step(self):
            return None

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            return None

    monkeypatch.setattr(ea, "LinearLR", _FakeScheduler)
    monkeypatch.setattr(ea, "CosineAnnealingLR", _FakeScheduler)
    monkeypatch.setattr(ea, "SequentialLR", _FakeScheduler)

    _install_common_patches(monkeypatch, args, dataset_obj=ds)
    ea.main()

    # LinearLR uses configured warmup directly.
    linear_kwargs = next(k for _a, k in captured["calls"] if "total_iters" in k)
    assert linear_kwargs["total_iters"] == 3
    assert linear_kwargs["start_factor"] == pytest.approx(0.1)
    assert linear_kwargs["end_factor"] == pytest.approx(1.0)

    # Cosine decay spans the remaining steps after warmup.
    cosine_kwargs = next(k for _a, k in captured["calls"] if "T_max" in k)
    assert cosine_kwargs["T_max"] == 8


def test_training_runs_to_max_steps_without_early_stopping(monkeypatch, tmp_path):
    meta = _write_bio_metadata(tmp_path / "bios_metadata.json", n_train_people=2, n_test_people=2)
    train_ids = set(meta["train_people"])
    ds = _TinyDataset.build(n_items=4)
    args = _default_args(
        tmp_path,
        data_fraction=1.0,
        batch_size=2,
        max_steps=5,
        eval_interval=1,
        log_interval=1000,
    )

    def memo_fn(_model, _tok, bios, _spans, _device, **_kwargs):
        split = "train" if bios and bios[0]["person_id"] in train_ids else "test"
        if split == "train":
            return _default_memo_metrics(0.1)
        return _default_memo_metrics(0.99)

    ctx = _install_common_patches(monkeypatch, args, dataset_obj=ds, memo_fn=memo_fn)
    ea.main()

    # Must always run full fixed-step budget.
    assert ctx["save_calls"][-1]["global_step"] == 5

    # Final log should include both train/test memo summaries.
    final_logs = [x for x in ctx["wandb"].logged if "final/train_memo_top1" in x]
    assert final_logs, "Expected final memo log"
    assert final_logs[-1]["final/train_memo_top1"] == pytest.approx(0.1)
    assert final_logs[-1]["final/test_memo_top1"] == pytest.approx(0.99)


def test_main_requires_positive_max_steps(monkeypatch, tmp_path):
    _write_bio_metadata(tmp_path / "bios_metadata.json")
    args = _default_args(tmp_path, max_steps=0)
    _install_common_patches(monkeypatch, args, dataset_obj=_TinyDataset.build(n_items=8))
    with pytest.raises(ValueError, match="--max_steps must be > 0"):
        ea.main()


def test_eval_logs_include_both_train_and_test_memo_keys(monkeypatch, tmp_path):
    meta = _write_bio_metadata(tmp_path / "bios_metadata.json", n_train_people=2, n_test_people=2)
    train_ids = set(meta["train_people"])
    ds = _TinyDataset.build(n_items=6)
    args = _default_args(
        tmp_path,
        data_fraction=1.0,
        batch_size=2,
        max_steps=2,
        eval_interval=1,
        log_interval=1000,
    )

    def memo_fn(_model, _tok, bios, _spans, _device, **_kwargs):
        split = "train" if bios and bios[0]["person_id"] in train_ids else "test"
        return _default_memo_metrics(0.2 if split == "train" else 0.3)

    ctx = _install_common_patches(monkeypatch, args, dataset_obj=ds, memo_fn=memo_fn)
    ea.main()

    eval_logs = [
        x for x in ctx["wandb"].logged
        if "train/step" in x and x.get("train/step", 0) > 0 and "memo_train/top1_acc" in x
    ]
    assert eval_logs, "Expected at least one eval log with memo metrics"
    assert "memo_test/top1_acc" in eval_logs[0]


def test_e2e_entrypoint_runs_via_module_main(monkeypatch, tmp_path):
    """Smoke e2e: execute extraction_attack module entrypoint with CLI args."""
    _write_bio_metadata(tmp_path / "bios_metadata.json", n_train_people=2, n_test_people=2)
    ds = _TinyDataset.build(n_items=8)
    fake_wandb = _FakeWandb()
    save_calls = []

    # Patch globals used by a fresh module execution (runpy as __main__).
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("datasets.load_from_disk", lambda _p: {"train": ds})
    monkeypatch.setattr(
        "tiered.model.GPTNeoForCausalLMTiered.from_pretrained",
        staticmethod(lambda *a, **k: _TinyModel()),
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _TinyTokenizer())
    monkeypatch.setattr("transformers.DataCollatorForLanguageModeling", _TinyCollator)
    monkeypatch.setattr(
        "tiered.train.finetune.private_finetune_memorization._bio_value_span",
        lambda *a, **k: (1, 2),
    )
    monkeypatch.setattr(
        "tiered.train.finetune.private_finetune_memorization.evaluate_memorization",
        lambda *a, **k: _default_memo_metrics(0.2),
    )
    monkeypatch.setattr(
        "tiered.train.utils.save_checkpoint",
        lambda *a, **k: save_calls.append(dict(k)),
    )
    monkeypatch.setattr("tqdm.tqdm", lambda *a, **k: _FakePbar(*a, **k))
    monkeypatch.setattr("wandb.init", fake_wandb.init)
    monkeypatch.setattr("wandb.define_metric", fake_wandb.define_metric)
    monkeypatch.setattr("wandb.log", fake_wandb.log)
    monkeypatch.setattr("wandb.finish", fake_wandb.finish)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--model_checkpoint",
            "unused-model",
            "--private_data",
            "unused-private-data",
            "--data_fraction",
            "0.5",
            "--output_dir",
            str(tmp_path / "out"),
            "--max_steps",
            "2",
            "--bio_metadata",
            str(tmp_path / "bios_metadata.json"),
            "--batch_size",
            "2",
            "--eval_interval",
            "1",
            "--log_interval",
            "1",
            "--num_workers",
            "0",
            "--run_name",
            "e2e-smoke",
            "--wandb_project",
            "test-proj",
        ],
    )

    runpy.run_module("tiered.train.finetune.extraction_attack", run_name="__main__")

    final_logs = [x for x in fake_wandb.logged if "final/train_memo_top1" in x]
    assert final_logs, "Expected final summary log from full module execution"
    assert final_logs[-1]["final/train_memo_top1"] == pytest.approx(0.2)
    assert final_logs[-1]["final/test_memo_top1"] == pytest.approx(0.2)
    assert save_calls, "Expected checkpoint saves during full module execution"
