"""Tests for gradient masking functionality."""

import unittest

import torch
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoConfig, GPTNeoForCausalLM

from tiered.permutation.key import PermutationKey
from tiered.permutation.masking import mask_keyed_gradients, mask_public_gradients
from tiered.permutation.scaling import scale_public_gradients


class MockGPTNeoConfig(GPTNeoConfig):
    """Mock config for testing."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 16
        self.intermediate_size = 32
        self.num_layers = 4
        self.num_heads = 4
        self.attention_types = [
            [["global"], "local"],
            [["global"], "local"],
            [["global"], "local"],
            [["global"], "local"],
        ]
        self.attention_layers = ["global", "local", "global", "local"]
        self.max_position_embeddings = 16
        self.vocab_size = 100
        self.activation_function = "gelu"
        self.initializer_range = 0.02
        self.layer_norm_epsilon = 1e-5
        self.embed_dropout = 0
        self.attention_dropout = 0
        self.resid_dropout = 0
        self.window_size = 4


class TestGradientMasking(unittest.TestCase):
    """Tests for key-based gradient masking."""

    def setUp(self):
        """Set up a model for testing."""
        torch.manual_seed(42)
        self.config = MockGPTNeoConfig()
        self.model = GPTNeoForCausalLM(self.config)
        self.model.train()
        
        # Key: swap head 1 of layer 0 with head 3 of layer 2
        # Key: swap MLP col 5 of layer 1 with col 10 of layer 3
        self.key = PermutationKey(
            attn_heads=[[[0, 1], [2, 3]]],
            mlp_cols=[[[1, 5], [3, 10]]],
        )

    def _run_backward(self):
        """Run a forward-backward pass to populate gradients."""
        input_ids = torch.randint(0, 100, (1, 8))
        labels = input_ids.clone()
        outputs = self.model(input_ids, labels=labels)
        outputs.loss.backward()

    def test_mask_keyed_gradients_zeros_swap_params(self):
        """Test that mask_keyed_gradients zeros gradients for swapped params."""
        self._run_backward()
        
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        head_1_start = 1 * head_dim
        head_1_end = 2 * head_dim
        
        self.assertGreater(
            torch.norm(attn_0.q_proj.weight.grad[head_1_start:head_1_end, :]).item(),
            0,
            "Gradient should be non-zero before masking"
        )
        
        mask_keyed_gradients(self.model, self.key)
        
        self.assertTrue(
            torch.allclose(
                attn_0.q_proj.weight.grad[head_1_start:head_1_end, :],
                torch.zeros_like(attn_0.q_proj.weight.grad[head_1_start:head_1_end, :])
            ),
            "Keyed attention gradients should be zero after masking"
        )

    def test_mask_keyed_gradients_preserves_other_params(self):
        """Test that mask_keyed_gradients preserves non-keyed param gradients."""
        self._run_backward()
        
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        head_0_grad_before = attn_0.q_proj.weight.grad[:head_dim, :].clone()
        
        mask_keyed_gradients(self.model, self.key)
        
        self.assertTrue(
            torch.allclose(attn_0.q_proj.weight.grad[:head_dim, :], head_0_grad_before),
            "Non-keyed gradients should be preserved"
        )

    def test_mask_public_gradients_zeros_non_swap_params(self):
        """Test that mask_public_gradients zeros gradients for non-swapped params."""
        self._run_backward()
        mask_public_gradients(self.model, self.key)
        
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        
        self.assertTrue(
            torch.allclose(
                attn_0.q_proj.weight.grad[:head_dim, :],
                torch.zeros_like(attn_0.q_proj.weight.grad[:head_dim, :])
            ),
            "Non-keyed attention gradients should be zero"
        )

    def test_mask_public_gradients_preserves_keyed_params(self):
        """Test that mask_public_gradients preserves keyed param gradients."""
        self._run_backward()
        
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        head_1_start = 1 * head_dim
        head_1_end = 2 * head_dim
        keyed_grad_before = attn_0.q_proj.weight.grad[head_1_start:head_1_end, :].clone()
        
        mask_public_gradients(self.model, self.key)
        
        self.assertTrue(
            torch.allclose(
                attn_0.q_proj.weight.grad[head_1_start:head_1_end, :],
                keyed_grad_before
            ),
            "Keyed gradients should be preserved"
        )

    def test_mask_keyed_gradients_attn_out_only_zeros_out_proj_only(self):
        """Out-only attention key should mask out_proj cols but preserve q/k/v rows."""
        self._run_backward()

        key = PermutationKey(attn_out_heads=[[[0, 1], [2, 3]]])
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        start = 1 * head_dim
        end = 2 * head_dim

        q_before = attn_0.q_proj.weight.grad[start:end, :].clone()
        out_before = attn_0.out_proj.weight.grad[:, start:end].clone()

        mask_keyed_gradients(self.model, key)

        self.assertTrue(
            torch.allclose(attn_0.q_proj.weight.grad[start:end, :], q_before),
            "q_proj rows should be unchanged for attn_out-only masking",
        )
        self.assertTrue(
            torch.allclose(
                attn_0.out_proj.weight.grad[:, start:end],
                torch.zeros_like(out_before),
            ),
            "out_proj keyed columns should be zero for attn_out-only masking",
        )

    def test_scale_public_gradients_scales_only_public_all_key_types(self):
        """scale_public_gradients should scale public grads while preserving keyed grads exactly."""
        scale = 0.5
        torch.manual_seed(7)

        key = PermutationKey(
            attn_heads=[[[0, 1], [2, 3]]],
            attn_out_heads=[[[1, 0], [3, 2]]],
            mlp_cols=[[[0, 5], [2, 10]]],
            mlp_up_cols=[[[1, 7], [3, 12]]],
            mlp_down_cols=[[[0, 8], [2, 9]]],
        )

        # Deterministic, non-zero synthetic grads so this test isolates scaling logic.
        for p in self.model.parameters():
            p.grad = torch.randn_like(p)

        before = {
            name: p.grad.detach().clone()
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }
        expected = {name: grad * scale for name, grad in before.items()}

        head_dim = self.config.hidden_size // self.config.num_heads

        def _restore_attn_head(layer: int, head: int) -> None:
            start, end = head * head_dim, (head + 1) * head_dim
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                name = f"transformer.h.{layer}.attn.attention.{proj_name}.weight"
                expected[name][start:end, :] = before[name][start:end, :]
            out_name = f"transformer.h.{layer}.attn.attention.out_proj.weight"
            expected[out_name][:, start:end] = before[out_name][:, start:end]

        def _restore_attn_out_head(layer: int, head: int) -> None:
            start, end = head * head_dim, (head + 1) * head_dim
            out_name = f"transformer.h.{layer}.attn.attention.out_proj.weight"
            expected[out_name][:, start:end] = before[out_name][:, start:end]

        def _restore_mlp_full(layer: int, col: int) -> None:
            fc_w = f"transformer.h.{layer}.mlp.c_fc.weight"
            fc_b = f"transformer.h.{layer}.mlp.c_fc.bias"
            proj_w = f"transformer.h.{layer}.mlp.c_proj.weight"
            expected[fc_w][col, :] = before[fc_w][col, :]
            expected[fc_b][col] = before[fc_b][col]
            expected[proj_w][:, col] = before[proj_w][:, col]

        def _restore_mlp_up(layer: int, col: int) -> None:
            fc_w = f"transformer.h.{layer}.mlp.c_fc.weight"
            fc_b = f"transformer.h.{layer}.mlp.c_fc.bias"
            expected[fc_w][col, :] = before[fc_w][col, :]
            expected[fc_b][col] = before[fc_b][col]

        def _restore_mlp_down(layer: int, col: int) -> None:
            proj_w = f"transformer.h.{layer}.mlp.c_proj.weight"
            expected[proj_w][:, col] = before[proj_w][:, col]

        for layer, head in ((0, 1), (2, 3)):
            _restore_attn_head(layer, head)
        for layer, head in ((1, 0), (3, 2)):
            _restore_attn_out_head(layer, head)
        for layer, col in ((0, 5), (2, 10)):
            _restore_mlp_full(layer, col)
        for layer, col in ((1, 7), (3, 12)):
            _restore_mlp_up(layer, col)
        for layer, col in ((0, 8), (2, 9)):
            _restore_mlp_down(layer, col)

        scale_public_gradients(self.model, key, scale=scale)

        # Explicit keyed/public checks for each key type. These assertions make
        # failures directly actionable (which path regressed).
        start, end = 1 * head_dim, 2 * head_dim  # attn_heads keyed range
        name = "transformer.h.0.attn.attention.k_proj.weight"
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name).grad[start:end, :], before[name][start:end, :], atol=0, rtol=0),
            "Keyed attn-head rows were changed by scale_public_gradients",
        )
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name).grad[0:head_dim, :], before[name][0:head_dim, :] * scale, atol=0, rtol=0),
            "Public attn-head rows were not scaled",
        )

        start, end = 0 * head_dim, 1 * head_dim  # attn_out_heads keyed range
        name = "transformer.h.1.attn.attention.out_proj.weight"
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name).grad[:, start:end], before[name][:, start:end], atol=0, rtol=0),
            "Keyed attn-out columns were changed by scale_public_gradients",
        )
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name).grad[:, head_dim:2 * head_dim], before[name][:, head_dim:2 * head_dim] * scale, atol=0, rtol=0),
            "Public attn-out columns were not scaled",
        )

        name_fc = "transformer.h.0.mlp.c_fc.weight"  # mlp_cols keyed row
        name_proj = "transformer.h.0.mlp.c_proj.weight"
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name_fc).grad[5, :], before[name_fc][5, :], atol=0, rtol=0),
            "Keyed mlp_cols c_fc row was changed by scale_public_gradients",
        )
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name_proj).grad[:, 5], before[name_proj][:, 5], atol=0, rtol=0),
            "Keyed mlp_cols c_proj column was changed by scale_public_gradients",
        )
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name_fc).grad[0, :], before[name_fc][0, :] * scale, atol=0, rtol=0),
            "Public c_fc row was not scaled",
        )

        name_fc = "transformer.h.1.mlp.c_fc.weight"  # mlp_up_cols keyed row
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name_fc).grad[7, :], before[name_fc][7, :], atol=0, rtol=0),
            "Keyed mlp_up_cols c_fc row was changed by scale_public_gradients",
        )
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name_fc).grad[0, :], before[name_fc][0, :] * scale, atol=0, rtol=0),
            "Public mlp_up c_fc row was not scaled",
        )

        name_proj = "transformer.h.0.mlp.c_proj.weight"  # mlp_down_cols keyed column
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name_proj).grad[:, 8], before[name_proj][:, 8], atol=0, rtol=0),
            "Keyed mlp_down_cols c_proj column was changed by scale_public_gradients",
        )
        self.assertTrue(
            torch.allclose(self.model.get_parameter(name_proj).grad[:, 0], before[name_proj][:, 0] * scale, atol=0, rtol=0),
            "Public c_proj column was not scaled",
        )

        mismatches = []
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if not torch.allclose(p.grad, expected[name], atol=0, rtol=0):
                max_abs = (p.grad - expected[name]).abs().max().item()
                mismatches.append((name, max_abs))
        self.assertFalse(
            mismatches,
            f"Gradients differed from expected scale behavior: {mismatches[:5]}",
        )


class TestWeightUpdates(unittest.TestCase):
    """Tests that verify actual weight updates after optimizer.step()."""

    def setUp(self):
        """Set up model and optimizer."""
        torch.manual_seed(42)
        self.config = MockGPTNeoConfig()
        self.model = GPTNeoForCausalLM(self.config)
        self.model.train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        
        self.key = PermutationKey(
            attn_heads=[[[0, 1], [2, 3]]],
            mlp_cols=[[[1, 5], [3, 10]]],
        )
        self.head_dim = self.config.hidden_size // self.config.num_heads

    def _get_weights_snapshot(self):
        """Get a snapshot of all model weights."""
        return {name: param.detach().clone() for name, param in self.model.named_parameters()}

    def _run_backward(self):
        """Run forward-backward pass."""
        input_ids = torch.randint(0, 100, (1, 8))
        outputs = self.model(input_ids, labels=input_ids.clone())
        outputs.loss.backward()

    def test_mask_keyed_only_public_weights_change(self):
        """After masking keyed grads, only public weights should change."""
        before = self._get_weights_snapshot()
        
        self._run_backward()
        mask_keyed_gradients(self.model, self.key)
        self.optimizer.step()
        
        after = self._get_weights_snapshot()
        
        # Check keyed attention head (layer 0, head 1) - should NOT change
        q_before = before["transformer.h.0.attn.attention.q_proj.weight"]
        q_after = after["transformer.h.0.attn.attention.q_proj.weight"]
        start, end = self.head_dim, 2 * self.head_dim
        
        self.assertTrue(
            torch.allclose(q_before[start:end, :], q_after[start:end, :]),
            "Keyed head should NOT change when keyed gradients are masked"
        )
        
        # Check public attention head (layer 0, head 0) - SHOULD change
        self.assertFalse(
            torch.allclose(q_before[:self.head_dim, :], q_after[:self.head_dim, :]),
            "Public head SHOULD change when keyed gradients are masked"
        )
        
        # Check keyed MLP column (layer 1, col 5) - should NOT change
        fc_before = before["transformer.h.1.mlp.c_fc.weight"]
        fc_after = after["transformer.h.1.mlp.c_fc.weight"]
        
        self.assertTrue(
            torch.allclose(fc_before[5, :], fc_after[5, :]),
            "Keyed MLP column should NOT change"
        )
        
        # Check public MLP column (layer 1, col 0) - SHOULD change
        self.assertFalse(
            torch.allclose(fc_before[0, :], fc_after[0, :]),
            "Public MLP column SHOULD change"
        )

    def test_mask_public_only_keyed_weights_change(self):
        """After masking public grads, only keyed weights should change."""
        before = self._get_weights_snapshot()
        
        self._run_backward()
        mask_public_gradients(self.model, self.key)
        self.optimizer.step()
        
        after = self._get_weights_snapshot()
        
        # Check keyed attention head (layer 0, head 1) - SHOULD change
        q_before = before["transformer.h.0.attn.attention.q_proj.weight"]
        q_after = after["transformer.h.0.attn.attention.q_proj.weight"]
        start, end = self.head_dim, 2 * self.head_dim
        
        self.assertFalse(
            torch.allclose(q_before[start:end, :], q_after[start:end, :]),
            "Keyed head SHOULD change when public gradients are masked"
        )
        
        # Check public attention head (layer 0, head 0) - should NOT change
        self.assertTrue(
            torch.allclose(q_before[:self.head_dim, :], q_after[:self.head_dim, :]),
            "Public head should NOT change when public gradients are masked"
        )
        
        # Embeddings should NOT change (always public)
        self.assertTrue(
            torch.allclose(
                before["transformer.wte.weight"],
                after["transformer.wte.weight"]
            ),
            "Embeddings should NOT change when public gradients are masked"
        )

    def test_mask_all_no_weights_change(self):
        """A key covering all heads/MLP columns should freeze all updates with mask_keyed_gradients."""
        all_heads = []
        for layer in range(self.config.num_layers):
            for head in range(0, self.config.num_heads, 2):
                all_heads.append([[layer, head], [layer, head + 1]])

        all_cols = []
        for layer in range(self.config.num_layers):
            for col in range(0, self.config.intermediate_size, 2):
                all_cols.append([[layer, col], [layer, col + 1]])

        full_key = PermutationKey(attn_heads=all_heads, mlp_cols=all_cols)
        before = self._get_weights_snapshot()
        
        self._run_backward()
        mask_keyed_gradients(self.model, full_key)
        self.optimizer.step()
        
        after = self._get_weights_snapshot()
        
        # Swappable subset should be unchanged.
        # (Embeddings/layer norms are public and can still update with mask_keyed_gradients.)
        def is_swappable_param(name: str) -> bool:
            return any(
                key in name
                for key in (
                    ".attn.attention.q_proj.weight",
                    ".attn.attention.k_proj.weight",
                    ".attn.attention.v_proj.weight",
                    ".attn.attention.out_proj.weight",
                    ".mlp.c_fc.weight",
                    ".mlp.c_fc.bias",
                    ".mlp.c_proj.weight",
                )
            )

        for name in before:
            if not is_swappable_param(name):
                continue
            self.assertTrue(
                torch.allclose(before[name], after[name]),
                f"Swappable parameter {name} changed despite full key masking",
            )

    def test_empty_key_all_weights_change(self):
        """With empty key, mask_keyed_gradients does nothing, all weights update."""
        empty_key = PermutationKey()
        before = self._get_weights_snapshot()
        
        self._run_backward()
        mask_keyed_gradients(self.model, empty_key)  # No-op
        self.optimizer.step()
        
        after = self._get_weights_snapshot()
        
        # Most weights should change (some might not due to zero gradients, but attention should)
        q_before = before["transformer.h.0.attn.attention.q_proj.weight"]
        q_after = after["transformer.h.0.attn.attention.q_proj.weight"]
        
        self.assertFalse(
            torch.allclose(q_before, q_after),
            "Weights should change with empty key (no masking)"
        )


if __name__ == "__main__":
    unittest.main()
