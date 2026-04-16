"""Tests for random cross-layer swap generation in scripts/keys/generate_key.py.

Verifies that the oversampling fix for _make_random_cross_layer_swaps produces
keys that hit the target percentage and maintain non-overlap guarantees.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_generate_key_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "keys" / "generate_key.py"
    spec = importlib.util.spec_from_file_location("generate_key_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gk = _load_generate_key_module()


# ---------------------------------------------------------------------------
# _make_random_cross_layer_swaps unit tests
# ---------------------------------------------------------------------------


class TestMakeRandomCrossLayerSwaps:
    """Unit tests for the low-level pairing function."""

    def test_all_swaps_are_cross_layer(self):
        """No swap should pair two items from the same layer."""
        import random

        random.seed(0)
        pool = [(l, i) for l in range(6) for i in range(20)]
        random.shuffle(pool)
        swaps = gk._make_random_cross_layer_swaps(pool, max_swaps=30)
        for (la, _), (lb, _) in swaps:
            assert la != lb, f"Same-layer swap found: layer {la}"

    def test_no_duplicate_slots(self):
        """Each slot should appear in at most one swap."""
        import random

        random.seed(1)
        pool = [(l, i) for l in range(8) for i in range(15)]
        random.shuffle(pool)
        swaps = gk._make_random_cross_layer_swaps(pool, max_swaps=40)
        seen = set()
        for a, b in swaps:
            ta, tb = tuple(a), tuple(b)
            assert ta not in seen, f"Duplicate slot {ta}"
            assert tb not in seen, f"Duplicate slot {tb}"
            seen.add(ta)
            seen.add(tb)

    def test_respects_max_swaps(self):
        """Should not produce more swaps than max_swaps."""
        import random

        random.seed(2)
        pool = [(l, i) for l in range(10) for i in range(50)]
        random.shuffle(pool)
        for max_swaps in [1, 10, 50]:
            swaps = gk._make_random_cross_layer_swaps(pool, max_swaps=max_swaps)
            assert len(swaps) <= max_swaps

    def test_hits_max_swaps_with_large_pool(self):
        """With a pool >> 2*max_swaps and many layers, should hit max_swaps exactly."""
        import random

        random.seed(3)
        pool = [(l, i) for l in range(12) for i in range(100)]
        random.shuffle(pool)
        max_swaps = 50
        swaps = gk._make_random_cross_layer_swaps(pool, max_swaps=max_swaps)
        assert len(swaps) == max_swaps

    def test_two_layers_pairs_optimally(self):
        """With only 2 layers, every item should find a cross-layer partner."""
        import random

        random.seed(4)
        pool = [(0, i) for i in range(20)] + [(1, i) for i in range(20)]
        random.shuffle(pool)
        swaps = gk._make_random_cross_layer_swaps(pool, max_swaps=20)
        assert len(swaps) == 20

    def test_empty_pool(self):
        swaps = gk._make_random_cross_layer_swaps([], max_swaps=10)
        assert swaps == []

    def test_single_layer_returns_nothing(self):
        """All items in one layer — no cross-layer pair possible."""
        pool = [(0, i) for i in range(10)]
        swaps = gk._make_random_cross_layer_swaps(pool, max_swaps=5)
        assert swaps == []


# ---------------------------------------------------------------------------
# End-to-end generate_keys tests with random_cross_layer_pairing=True
# ---------------------------------------------------------------------------


class TestGenerateKeysRandomCrossLayer:
    """Integration tests: generate_keys with random_cross_layer_pairing=True."""

    # 150M model
    M150 = dict(
        num_layers=24, num_heads=16, hidden_size=1024, mlp_dim=4096,
    )
    # 530M model
    M530 = dict(
        num_layers=16, num_heads=16, hidden_size=1344, mlp_dim=10752,
    )

    def _actual_pct(self, keys, cfg):
        """Compute actual swappable % from a generated key."""
        w_head, w_mlp = gk.calculate_weights_per_swap(
            cfg["hidden_size"], cfg["num_heads"], cfg["mlp_dim"],
        )
        swappable = gk.count_total_swappable_params(
            num_layers=cfg["num_layers"],
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_heads"],
            mlp_dim=cfg["mlp_dim"],
        )["total"]

        key = keys[0]
        n_attn = len(key.get("attn_heads", []))
        n_mlp = len(key.get("mlp_cols", []))
        actual_params = n_attn * 2 * w_head + n_mlp * 2 * w_mlp
        return actual_params / swappable

    @pytest.mark.parametrize("target_pct", [0.05, 0.10, 0.20])
    def test_hits_target_150m(self, target_pct):
        """Random cross-layer key should be within 0.1% of target (150M)."""
        keys = gk.generate_keys(
            num_keys=1, **self.M150,
            target_pct=target_pct, attn_ratio=0.25, seed=42,
            random_cross_layer_pairing=True,
        )
        actual = self._actual_pct(keys, self.M150)
        assert actual == pytest.approx(target_pct, abs=0.002), (
            f"Expected ~{target_pct*100:.1f}%, got {actual*100:.2f}%"
        )

    @pytest.mark.parametrize("target_pct", [0.05, 0.10, 0.20])
    def test_hits_target_530m(self, target_pct):
        """Random cross-layer key should be within 0.1% of target (530M)."""
        keys = gk.generate_keys(
            num_keys=1, **self.M530,
            target_pct=target_pct, attn_ratio=0.25, seed=42,
            random_cross_layer_pairing=True,
        )
        actual = self._actual_pct(keys, self.M530)
        assert actual == pytest.approx(target_pct, abs=0.002), (
            f"Expected ~{target_pct*100:.1f}%, got {actual*100:.2f}%"
        )

    def test_non_overlap_multiple_keys(self):
        """Multiple random cross-layer keys must not share any slots."""
        keys = gk.generate_keys(
            num_keys=3, **self.M150,
            target_pct=0.10, attn_ratio=0.25, seed=99,
            random_cross_layer_pairing=True,
        )
        assert len(keys) == 3

        all_attn = set()
        all_mlp = set()
        for key in keys:
            for swap in key.get("attn_heads", []):
                for slot in swap:
                    t = tuple(slot)
                    assert t not in all_attn, f"Overlapping attn slot {t}"
                    all_attn.add(t)
            for swap in key.get("mlp_cols", []):
                for slot in swap:
                    t = tuple(slot)
                    assert t not in all_mlp, f"Overlapping mlp slot {t}"
                    all_mlp.add(t)

    def test_all_swaps_cross_layer(self):
        """Every swap in the generated key should be cross-layer."""
        keys = gk.generate_keys(
            num_keys=1, **self.M150,
            target_pct=0.15, attn_ratio=0.25, seed=7,
            random_cross_layer_pairing=True,
        )
        key = keys[0]
        for swap in key.get("attn_heads", []):
            assert swap[0][0] != swap[1][0], f"Same-layer attn swap: {swap}"
        for swap in key.get("mlp_cols", []):
            assert swap[0][0] != swap[1][0], f"Same-layer mlp swap: {swap}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_stable_across_seeds(self, seed):
        """Different seeds should all hit the target — not just a lucky seed."""
        target_pct = 0.10
        keys = gk.generate_keys(
            num_keys=1, **self.M150,
            target_pct=target_pct, attn_ratio=0.25, seed=seed,
            random_cross_layer_pairing=True,
        )
        actual = self._actual_pct(keys, self.M150)
        assert actual == pytest.approx(target_pct, abs=0.002), (
            f"seed={seed}: expected ~{target_pct*100:.1f}%, got {actual*100:.2f}%"
        )

    def test_matches_structured_cross_layer_coverage(self):
        """Random and structured cross-layer should produce similar coverage."""
        target_pct = 0.15
        common = dict(
            num_keys=1, **self.M530,
            target_pct=target_pct, attn_ratio=0.25, seed=42,
        )
        keys_structured = gk.generate_keys(**common, random_cross_layer_pairing=False)
        keys_random = gk.generate_keys(**common, random_cross_layer_pairing=True)

        pct_structured = self._actual_pct(keys_structured, self.M530)
        pct_random = self._actual_pct(keys_random, self.M530)

        assert pct_random == pytest.approx(pct_structured, abs=0.002), (
            f"Structured: {pct_structured*100:.2f}%, Random: {pct_random*100:.2f}%"
        )
