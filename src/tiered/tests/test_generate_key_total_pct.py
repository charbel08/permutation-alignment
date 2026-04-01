"""Tests for scripts/keys/generate_key.py total-percentage targeting."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_generate_key_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "keys" / "generate_key.py"
    spec = importlib.util.spec_from_file_location("generate_key_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, script_path


def test_count_total_params_matches_model_tied_and_untied():
    """count_total_params should match instantiated model counts exactly."""
    from tiered.train.utils import load_model

    gk, _ = _load_generate_key_module()

    cfg = {
        "hidden_size": 64,
        "num_heads": 4,
        "num_layers": 2,
        "context_size": 128,
        "intermediate_size": 256,
    }

    tied_model = load_model(
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        context_size=cfg["context_size"],
        intermediate_size=cfg["intermediate_size"],
        tie_weights=True,
        do_print=False,
    )
    tied_actual = sum(p.numel() for p in tied_model.parameters())
    tied_calc = gk.count_total_params(
        num_layers=cfg["num_layers"],
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        mlp_dim=cfg["intermediate_size"],
        max_positions=cfg["context_size"],
        untie_weights=False,
    )
    assert tied_model.lm_head.weight is tied_model.transformer.wte.weight
    assert tied_calc == tied_actual

    untied_model = load_model(
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        context_size=cfg["context_size"],
        intermediate_size=cfg["intermediate_size"],
        tie_weights=False,
        do_print=False,
    )
    untied_actual = sum(p.numel() for p in untied_model.parameters())
    untied_calc = gk.count_total_params(
        num_layers=cfg["num_layers"],
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        mlp_dim=cfg["intermediate_size"],
        max_positions=cfg["context_size"],
        untie_weights=True,
    )
    assert untied_model.lm_head.weight is not untied_model.transformer.wte.weight
    assert untied_calc == untied_actual

    # Untied adds only lm_head.weight relative to tied model.
    vocab = tied_model.config.vocab_size
    assert untied_actual - tied_actual == vocab * cfg["hidden_size"]


def test_convert_total_pct_matches_known_64m_values():
    """64M conversion should reproduce the target_pct values used in scripts."""
    gk, _ = _load_generate_key_module()

    common = dict(
        num_layers=12,
        hidden_size=512,
        num_heads=32,
        mlp_dim=2048,
        attn_mode="full",
        untie_weights=False,
        context_size=1024,
    )
    up = gk.convert_total_pct_to_swappable_pct(
        target_total_pct=0.20,
        mlp_mode="up",
        **common,
    )
    down = gk.convert_total_pct_to_swappable_pct(
        target_total_pct=0.20,
        mlp_mode="down",
        **common,
    )
    both = gk.convert_total_pct_to_swappable_pct(
        target_total_pct=0.20,
        mlp_mode="both",
        **common,
    )

    assert up == pytest.approx(0.5090616187118903, abs=1e-15)
    assert down == pytest.approx(0.5095587491989136, abs=1e-15)
    assert both == pytest.approx(0.33948481404013503, abs=1e-15)
    assert down > up


def test_cli_target_total_pct_equivalent_to_explicit_target_pct(tmp_path):
    """CLI --target_total_pct should generate same key as explicit converted target_pct."""
    gk, script_path = _load_generate_key_module()

    common = {
        "num_layers": 12,
        "num_heads": 32,
        "hidden_size": 512,
        "mlp_dim": 2048,
        "context_size": 1024,
        "attn_ratio": 0.25,
        "attn_mode": "full",
        "mlp_mode": "both",
        "seed": 123,
    }

    target_total = 0.20
    converted = gk.convert_total_pct_to_swappable_pct(
        target_total_pct=target_total,
        num_layers=common["num_layers"],
        hidden_size=common["hidden_size"],
        num_heads=common["num_heads"],
        mlp_dim=common["mlp_dim"],
        mlp_mode=common["mlp_mode"],
        attn_mode=common["attn_mode"],
        untie_weights=False,
        context_size=common["context_size"],
    )

    out_total = tmp_path / "key_total.json"
    out_swappable = tmp_path / "key_swappable.json"

    cmd_base = [
        sys.executable,
        str(script_path),
        "--num_layers",
        str(common["num_layers"]),
        "--num_heads",
        str(common["num_heads"]),
        "--hidden_size",
        str(common["hidden_size"]),
        "--mlp_dim",
        str(common["mlp_dim"]),
        "--context_size",
        str(common["context_size"]),
        "--attn_ratio",
        str(common["attn_ratio"]),
        "--attn_mode",
        common["attn_mode"],
        "--mlp_mode",
        common["mlp_mode"],
        "--seed",
        str(common["seed"]),
    ]

    subprocess.run(
        cmd_base + ["--output", str(out_total), "--target_total_pct", str(target_total)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        cmd_base + ["--output", str(out_swappable), "--target_pct", str(converted)],
        check=True,
        capture_output=True,
        text=True,
    )

    with open(out_total, "r", encoding="utf-8") as f:
        key_total = json.load(f)
    with open(out_swappable, "r", encoding="utf-8") as f:
        key_swappable = json.load(f)

    assert key_total == key_swappable
    assert len(key_total.get("attn_heads", [])) > 0
    assert len(key_total.get("mlp_cols", [])) > 0
