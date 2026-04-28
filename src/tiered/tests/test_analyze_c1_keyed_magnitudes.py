"""Tests for scripts/eval/analyze_c1_keyed_magnitudes.py.

Coverage:
  - _per_channel_l2_stats math (basic, empty keyed, all keyed, zero non-keyed, single)
  - compute_weight_per_channel_l2_per_layer end-to-end on a tiny tiered model
  - Per-channel L2 metric is consistent with the magnitude attack's score formula
  - _merge_keys (single passthrough, disjoint union, concatenation semantics)
  - parse_args accepts multiple --key_path values and the weights_only flag
  - main() validation: requires data unless --weights_only
  - Heatmap rendering: writes a real PNG, skips when stats empty, handles NaN cells
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest
import torch

from tiered.permutation import PermutationKey, build_mask_plan
from tiered.permutation.utils import _get_attention_module, _get_mlp_module
from tiered.train.utils import load_model


# ---------------------------------------------------------------------------
# Module loader (script lives outside the package)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "eval" / "analyze_c1_keyed_magnitudes.py"


@pytest.fixture(scope="module")
def amod():
    spec = importlib.util.spec_from_file_location("analyze_c1_keyed_magnitudes", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_c1_keyed_magnitudes"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _per_channel_l2_stats — pure math
# ---------------------------------------------------------------------------


class TestPerChannelL2Stats:
    def test_basic_math(self, amod):
        norms = torch.tensor([1.0, 2.0, 3.0, 4.0])
        keyed = torch.tensor([0, 2])
        s = amod._per_channel_l2_stats(norms, keyed)
        assert s["keyed_count"] == 2
        assert s["non_keyed_count"] == 2
        assert s["keyed_mean_l2"] == pytest.approx(2.0)        # mean(1, 3)
        assert s["non_keyed_mean_l2"] == pytest.approx(3.0)    # mean(2, 4)
        assert s["l2_ratio_key_over_non"] == pytest.approx(2.0 / 3.0)

    def test_empty_keyed(self, amod):
        norms = torch.tensor([1.0, 2.0, 3.0])
        keyed = torch.empty(0, dtype=torch.long)
        s = amod._per_channel_l2_stats(norms, keyed)
        assert s["keyed_count"] == 0
        assert s["non_keyed_count"] == 3
        assert math.isnan(s["keyed_mean_l2"])
        assert s["non_keyed_mean_l2"] == pytest.approx(2.0)
        assert math.isnan(s["l2_ratio_key_over_non"])

    def test_all_keyed(self, amod):
        norms = torch.tensor([1.0, 2.0, 3.0])
        keyed = torch.tensor([0, 1, 2])
        s = amod._per_channel_l2_stats(norms, keyed)
        assert s["keyed_count"] == 3
        assert s["non_keyed_count"] == 0
        assert s["keyed_mean_l2"] == pytest.approx(2.0)
        assert math.isnan(s["non_keyed_mean_l2"])
        assert math.isnan(s["l2_ratio_key_over_non"])

    def test_zero_non_keyed_mean_yields_nan_ratio(self, amod):
        norms = torch.tensor([1.0, 0.0, 2.0, 0.0])
        keyed = torch.tensor([0, 2])
        s = amod._per_channel_l2_stats(norms, keyed)
        assert s["non_keyed_mean_l2"] == pytest.approx(0.0)
        assert math.isnan(s["l2_ratio_key_over_non"])

    def test_single_keyed_channel(self, amod):
        norms = torch.tensor([5.0, 3.0])
        keyed = torch.tensor([0])
        s = amod._per_channel_l2_stats(norms, keyed)
        assert s["keyed_count"] == 1
        assert s["non_keyed_count"] == 1
        assert s["keyed_mean_l2"] == pytest.approx(5.0)
        assert s["non_keyed_mean_l2"] == pytest.approx(3.0)
        assert s["l2_ratio_key_over_non"] == pytest.approx(5.0 / 3.0)

    def test_ratio_above_one_when_keyed_louder(self, amod):
        norms = torch.tensor([1.0, 1.0, 10.0, 10.0])
        keyed = torch.tensor([2, 3])
        s = amod._per_channel_l2_stats(norms, keyed)
        assert s["l2_ratio_key_over_non"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# compute_weight_per_channel_l2_per_layer — tiny model integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    model = load_model(
        hidden_size=8, num_heads=2, num_layers=2,
        context_size=4, intermediate_size=16, do_print=False,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_key():
    # mlp_cols swap (L0, col 2) <-> (L1, col 5)
    # attn_heads swap (L0, head 0) <-> (L1, head 1)
    return PermutationKey(
        attn_heads=[[[0, 0], [1, 1]]],
        mlp_cols=[[[0, 2], [1, 5]]],
    )


@pytest.fixture(scope="module")
def tiny_mask_plan(tiny_model, tiny_key):
    return build_mask_plan(tiny_model, tiny_key, torch.device("cpu"))


class TestComputeWeightL2:
    def test_expected_keys_present(self, amod, tiny_model, tiny_mask_plan):
        out = amod.compute_weight_per_channel_l2_per_layer(tiny_model, tiny_mask_plan)
        for layer in ("L00", "L01"):
            for fam in (
                "attn_q_weight_rows",
                "attn_k_weight_rows",
                "attn_v_weight_rows",
                "attn_out_weight_cols",
                "mlp_fc_weight_rows",
                "mlp_proj_weight_cols",
            ):
                assert f"{layer}_{fam}" in out, f"missing {layer}_{fam}"

    def test_each_cell_has_required_fields(self, amod, tiny_model, tiny_mask_plan):
        out = amod.compute_weight_per_channel_l2_per_layer(tiny_model, tiny_mask_plan)
        required = {"keyed_count", "non_keyed_count", "keyed_mean_l2",
                    "non_keyed_mean_l2", "l2_ratio_key_over_non"}
        for cell in out.values():
            assert required.issubset(cell.keys())

    def test_mlp_fc_value_matches_manual(self, amod, tiny_model, tiny_mask_plan):
        layer = 0
        c_fc = _get_mlp_module(tiny_model, layer).c_fc.weight.detach().float()
        per_row_l2 = c_fc.norm(dim=1)
        keyed = tiny_mask_plan.keyed_mlp_indices[layer].cpu()
        mask = torch.zeros(c_fc.shape[0], dtype=torch.bool)
        mask[keyed] = True
        expected_keyed = float(per_row_l2[mask].mean().item())
        expected_non = float(per_row_l2[~mask].mean().item())

        out = amod.compute_weight_per_channel_l2_per_layer(tiny_model, tiny_mask_plan)
        cell = out["L00_mlp_fc_weight_rows"]
        assert cell["keyed_mean_l2"] == pytest.approx(expected_keyed, rel=1e-6)
        assert cell["non_keyed_mean_l2"] == pytest.approx(expected_non, rel=1e-6)
        assert cell["l2_ratio_key_over_non"] == pytest.approx(
            expected_keyed / expected_non, rel=1e-6
        )

    def test_attn_q_value_matches_manual(self, amod, tiny_model, tiny_mask_plan):
        layer = 0
        q = _get_attention_module(tiny_model, layer).q_proj.weight.detach().float()
        per_row_l2 = q.norm(dim=1)
        keyed = tiny_mask_plan.keyed_attn_indices[layer].cpu()
        mask = torch.zeros(q.shape[0], dtype=torch.bool)
        mask[keyed] = True
        expected = float(per_row_l2[mask].mean().item())

        out = amod.compute_weight_per_channel_l2_per_layer(tiny_model, tiny_mask_plan)
        assert out["L00_attn_q_weight_rows"]["keyed_mean_l2"] == pytest.approx(expected, rel=1e-6)

    def test_attn_out_uses_columns_not_rows(self, amod, tiny_model, tiny_mask_plan):
        # out_proj has shape [hidden, num_heads*head_dim]; channel = column.
        layer = 0
        out_w = _get_attention_module(tiny_model, layer).out_proj.weight.detach().float()
        per_col_l2 = out_w.norm(dim=0)
        # The merge of keyed_attn + keyed_attn_out indices is the column set.
        idx = tiny_mask_plan.keyed_attn_indices[layer].cpu()  # only attn_heads in tiny_key
        mask = torch.zeros(out_w.shape[1], dtype=torch.bool)
        mask[idx] = True
        expected = float(per_col_l2[mask].mean().item())

        out = amod.compute_weight_per_channel_l2_per_layer(tiny_model, tiny_mask_plan)
        assert out["L00_attn_out_weight_cols"]["keyed_mean_l2"] == pytest.approx(expected, rel=1e-6)

    def test_layer_with_no_keyed_channels_is_absent(self, amod, tiny_model):
        # Build a key that only touches L0; L1 should have no entries.
        key = PermutationKey(mlp_cols=[[[0, 1], [0, 3]]])  # both endpoints in L0
        plan = build_mask_plan(tiny_model, key, torch.device("cpu"))
        out = amod.compute_weight_per_channel_l2_per_layer(tiny_model, plan)
        assert any(k.startswith("L00_") for k in out)
        assert not any(k.startswith("L01_") for k in out)

    def test_combined_mlp_mean_matches_attack_score_mean(self, amod, tiny_model, tiny_mask_plan):
        """The attack's per-channel MLP score is ‖c_fc[j,:]‖₂ + ‖c_proj[:,j]‖₂.
        Mean of that over keyed channels equals (mean of fc-rows) + (mean of proj-cols)
        only because mean is linear and our keyed_mlp_indices is the same channel
        set used for both rows of c_fc and cols of c_proj — so this consistency
        check fails fast if the channel-set bookkeeping ever drifts."""
        layer = 0
        mlp = _get_mlp_module(tiny_model, layer)
        c_fc = mlp.c_fc.weight.detach().float()
        c_proj = mlp.c_proj.weight.detach().float()
        attack_score = c_fc.norm(dim=1) + c_proj.norm(dim=0)
        keyed = tiny_mask_plan.keyed_mlp_indices[layer].cpu()
        mask = torch.zeros(c_fc.shape[0], dtype=torch.bool)
        mask[keyed] = True
        attack_keyed_mean = float(attack_score[mask].mean().item())

        out = amod.compute_weight_per_channel_l2_per_layer(tiny_model, tiny_mask_plan)
        combined = (
            out["L00_mlp_fc_weight_rows"]["keyed_mean_l2"]
            + out["L00_mlp_proj_weight_cols"]["keyed_mean_l2"]
        )
        assert combined == pytest.approx(attack_keyed_mean, rel=1e-6)


# ---------------------------------------------------------------------------
# _merge_keys
# ---------------------------------------------------------------------------


class TestMergeKeys:
    def test_single_key_passthrough(self, amod):
        k = PermutationKey(attn_heads=[[[0, 0], [1, 1]]], mlp_cols=[[[0, 5], [1, 7]]])
        merged = amod._merge_keys([k])
        assert merged is k

    def test_disjoint_union(self, amod):
        k1 = PermutationKey(attn_heads=[[[0, 0], [0, 1]]], mlp_cols=[[[0, 1], [0, 2]]])
        k2 = PermutationKey(attn_heads=[[[1, 0], [1, 1]]], mlp_cols=[[[1, 1], [1, 2]]])
        merged = amod._merge_keys([k1, k2])
        assert merged.attn_heads == k1.attn_heads + k2.attn_heads
        assert merged.mlp_cols == k1.mlp_cols + k2.mlp_cols
        assert merged.attn_out_heads == []
        assert merged.mlp_up_cols == []
        assert merged.mlp_down_cols == []

    def test_three_way_union_all_fields(self, amod):
        k1 = PermutationKey(attn_heads=[[[0, 0], [0, 1]]])
        k2 = PermutationKey(attn_out_heads=[[[1, 0], [1, 1]]])
        k3 = PermutationKey(
            mlp_cols=[[[0, 0], [0, 1]]],
            mlp_up_cols=[[[1, 0], [1, 1]]],
            mlp_down_cols=[[[0, 2], [0, 3]]],
        )
        merged = amod._merge_keys([k1, k2, k3])
        assert merged.attn_heads == k1.attn_heads
        assert merged.attn_out_heads == k2.attn_out_heads
        assert merged.mlp_cols == k3.mlp_cols
        assert merged.mlp_up_cols == k3.mlp_up_cols
        assert merged.mlp_down_cols == k3.mlp_down_cols

    def test_concatenation_not_set_union(self, amod):
        # Identical swaps appear twice — caller's responsibility to ensure disjoint.
        k1 = PermutationKey(mlp_cols=[[[0, 1], [0, 2]]])
        k2 = PermutationKey(mlp_cols=[[[0, 1], [0, 2]]])
        merged = amod._merge_keys([k1, k2])
        assert len(merged.mlp_cols) == 2


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_multiple_key_paths(self, amod, monkeypatch):
        argv = ["prog", "--checkpoint", "/tmp/x", "--key_path", "a.json", "b.json", "c.json", "--weights_only"]
        monkeypatch.setattr("sys.argv", argv)
        ns = amod.parse_args()
        assert ns.key_path == ["a.json", "b.json", "c.json"]
        assert ns.weights_only is True

    def test_single_key_path_still_returns_list(self, amod, monkeypatch):
        argv = ["prog", "--checkpoint", "/tmp/x", "--key_path", "a.json", "--weights_only"]
        monkeypatch.setattr("sys.argv", argv)
        ns = amod.parse_args()
        assert ns.key_path == ["a.json"]

    def test_default_dataset_args_are_none(self, amod, monkeypatch):
        argv = ["prog", "--checkpoint", "/tmp/x", "--key_path", "a.json"]
        monkeypatch.setattr("sys.argv", argv)
        ns = amod.parse_args()
        assert ns.private_data is None
        assert ns.public_data is None
        assert ns.weights_only is False


# ---------------------------------------------------------------------------
# main() validation
# ---------------------------------------------------------------------------


def _write_minimal_key(path: Path) -> None:
    path.write_text('{"attn_heads": [], "mlp_cols": []}')


class TestMainValidation:
    def test_requires_at_least_one_dataset(self, amod, monkeypatch, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        key = tmp_path / "k.json"
        _write_minimal_key(key)
        argv = ["prog", "--checkpoint", str(ckpt), "--key_path", str(key)]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(ValueError, match="At least one of"):
            amod.main()

    def test_weights_only_skips_dataset_check(self, amod, monkeypatch, tmp_path):
        # Missing checkpoint dir → FileNotFoundError fires before any data validation.
        # If --weights_only weren't honored, ValueError("At least one of") would fire first.
        missing_ckpt = tmp_path / "no_ckpt"
        key = tmp_path / "k.json"
        _write_minimal_key(key)
        argv = ["prog", "--checkpoint", str(missing_ckpt), "--key_path", str(key), "--weights_only"]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(FileNotFoundError, match="Checkpoint dir not found"):
            amod.main()

    def test_only_public_is_allowed(self, amod, monkeypatch, tmp_path):
        # Same logic: providing only public_data should not trip the "at least one" check.
        missing_ckpt = tmp_path / "no_ckpt"
        key = tmp_path / "k.json"
        _write_minimal_key(key)
        argv = [
            "prog", "--checkpoint", str(missing_ckpt), "--key_path", str(key),
            "--public_data", "/some/path",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(FileNotFoundError, match="Checkpoint dir not found"):
            amod.main()

    def test_missing_key_file(self, amod, monkeypatch, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        argv = ["prog", "--checkpoint", str(ckpt), "--key_path", str(tmp_path / "nope.json"), "--weights_only"]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(FileNotFoundError, match="Key file not found"):
            amod.main()


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------


def _stats_fixture(values_by_key: dict) -> dict:
    return {
        k: {
            "l2_ratio_key_over_non": v,
            "keyed_mean_l2": 1.0,
            "non_keyed_mean_l2": 1.0,
            "keyed_count": 1,
            "non_keyed_count": 1,
        }
        for k, v in values_by_key.items()
    }


class TestHeatmapPlot:
    def test_writes_nontrivial_png(self, amod, tmp_path):
        stats = _stats_fixture({
            "L00_attn_q_weight_rows": 0.85,
            "L01_attn_q_weight_rows": 1.25,
            "L00_mlp_fc_weight_rows": 0.6,
            "L01_mlp_fc_weight_rows": 0.7,
        })
        out = tmp_path / "heatmap.png"
        amod._plot_per_layer_ratio_heatmap(
            stats, "Test heatmap",
            ["attn_q_weight_rows", "mlp_fc_weight_rows"],
            str(out),
            ratio_key="l2_ratio_key_over_non",
            cbar_label="ratio",
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_skips_when_stats_empty(self, amod, tmp_path):
        out = tmp_path / "heatmap.png"
        amod._plot_per_layer_ratio_heatmap(
            {}, "Test", ["attn_q_weight_rows"], str(out),
            ratio_key="l2_ratio_key_over_non",
        )
        assert not out.exists()

    def test_handles_partial_nan_cells(self, amod, tmp_path):
        # Component appears in some layers, not others — missing cells must hatch, not crash.
        stats = _stats_fixture({
            "L00_mlp_fc_weight_rows": 0.85,
            "L00_attn_q_weight_rows": 1.10,
            "L01_attn_q_weight_rows": 1.20,
            # No L01_mlp_fc_weight_rows -> NaN cell
        })
        out = tmp_path / "heatmap.png"
        amod._plot_per_layer_ratio_heatmap(
            stats, "Test",
            ["attn_q_weight_rows", "mlp_fc_weight_rows"],
            str(out),
            ratio_key="l2_ratio_key_over_non",
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_handles_explicit_nan_value(self, amod, tmp_path):
        # An explicit NaN in the ratio field (e.g. degenerate ratio when all
        # non-keyed entries are zero) should also hatch cleanly.
        stats = _stats_fixture({
            "L00_attn_q_weight_rows": float("nan"),
            "L01_attn_q_weight_rows": 1.20,
        })
        out = tmp_path / "heatmap.png"
        amod._plot_per_layer_ratio_heatmap(
            stats, "Test", ["attn_q_weight_rows"], str(out),
            ratio_key="l2_ratio_key_over_non",
        )
        assert out.exists()

    def test_ratio_key_parameter_is_honored(self, amod, tmp_path):
        # Build stats where the two ratio fields disagree; switching ratio_key
        # must change which one is plotted (we verify by ensuring the function
        # doesn't crash when only the requested key is present).
        stats = {
            "L00_attn_q_weight_rows": {"only_this_key": 0.5},
            "L01_attn_q_weight_rows": {"only_this_key": 1.3},
        }
        out = tmp_path / "heatmap.png"
        amod._plot_per_layer_ratio_heatmap(
            stats, "Test", ["attn_q_weight_rows"], str(out),
            ratio_key="only_this_key",
        )
        assert out.exists()
