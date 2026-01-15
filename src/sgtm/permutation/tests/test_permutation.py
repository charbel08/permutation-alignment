"""Tests for the permutation module."""

import unittest
import json
import tempfile
import os
from pathlib import Path

import torch
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoConfig, GPTNeoForCausalLM

from sgtm.permutation.key import (
    PermutationKey,
    AttentionSwap,
    MLPSwap,
    load_key,
    save_key,
    validate_key,
)
from sgtm.permutation.permute import apply_permutation, unapply_permutation


class TestPermutationKey(unittest.TestCase):
    """Tests for PermutationKey dataclass."""

    def test_empty_key(self):
        """Test creating an empty key."""
        key = PermutationKey()
        self.assertTrue(key.is_empty())
        self.assertEqual(len(key.attention_swaps), 0)
        self.assertEqual(len(key.mlp_swaps), 0)

    def test_key_with_swaps(self):
        """Test creating a key with swaps."""
        key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=1, layer_b=2, head_b=3),
            ],
            mlp_swaps=[
                MLPSwap(layer_a=1, col_a=10, layer_b=3, col_b=20),
            ],
        )
        self.assertFalse(key.is_empty())
        self.assertEqual(len(key.attention_swaps), 1)
        self.assertEqual(len(key.mlp_swaps), 1)

    def test_key_to_dict_and_back(self):
        """Test serializing and deserializing a key."""
        original = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=1, layer_b=2, head_b=3),
                AttentionSwap(layer_a=1, head_a=0, layer_b=3, head_b=2),
            ],
            mlp_swaps=[
                MLPSwap(layer_a=1, col_a=10, layer_b=3, col_b=20),
            ],
        )
        
        d = original.to_dict()
        restored = PermutationKey.from_dict(d)
        
        self.assertEqual(len(restored.attention_swaps), 2)
        self.assertEqual(len(restored.mlp_swaps), 1)
        self.assertEqual(restored.attention_swaps[0].layer_a, 0)
        self.assertEqual(restored.attention_swaps[0].head_a, 1)


class TestKeyIO(unittest.TestCase):
    """Tests for key loading and saving."""

    def test_save_and_load_key(self):
        """Test saving and loading a key from JSON."""
        key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=1, layer_b=2, head_b=3),
            ],
            mlp_swaps=[
                MLPSwap(layer_a=1, col_a=10, layer_b=3, col_b=20),
            ],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "key.json")
            save_key(key, path)
            
            # Verify file exists
            self.assertTrue(os.path.exists(path))
            
            # Load and verify
            loaded = load_key(path)
            self.assertEqual(len(loaded.attention_swaps), 1)
            self.assertEqual(len(loaded.mlp_swaps), 1)
            self.assertEqual(loaded.attention_swaps[0].layer_a, 0)
            self.assertEqual(loaded.attention_swaps[0].head_a, 1)

    def test_load_nonexistent_key(self):
        """Test loading a key that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_key("/nonexistent/path/key.json")

    def test_load_empty_key(self):
        """Test loading an empty key file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "key.json")
            with open(path, "w") as f:
                json.dump({}, f)
            
            loaded = load_key(path)
            self.assertTrue(loaded.is_empty())


class TestKeyValidation(unittest.TestCase):
    """Tests for key validation."""

    def test_valid_key(self):
        """Test that a valid key passes validation."""
        key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=1, layer_b=2, head_b=3),
            ],
            mlp_swaps=[
                MLPSwap(layer_a=1, col_a=10, layer_b=3, col_b=20),
            ],
        )
        # Should not raise
        validate_key(key, num_layers=4, num_heads=4, mlp_dim=32)

    def test_invalid_layer_index(self):
        """Test that invalid layer indices are caught."""
        key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=10, head_a=0, layer_b=0, head_b=0),
            ],
        )
        with self.assertRaises(ValueError):
            validate_key(key, num_layers=4, num_heads=4, mlp_dim=32)

    def test_invalid_head_index(self):
        """Test that invalid head indices are caught."""
        key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=10, layer_b=0, head_b=0),
            ],
        )
        with self.assertRaises(ValueError):
            validate_key(key, num_layers=4, num_heads=4, mlp_dim=32)

    def test_invalid_mlp_column(self):
        """Test that invalid MLP column indices are caught."""
        key = PermutationKey(
            mlp_swaps=[
                MLPSwap(layer_a=0, col_a=100, layer_b=0, col_b=0),
            ],
        )
        with self.assertRaises(ValueError):
            validate_key(key, num_layers=4, num_heads=4, mlp_dim=32)


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


class TestPermutation(unittest.TestCase):
    """Tests for apply/unapply permutation."""

    def setUp(self):
        """Set up a model for testing."""
        torch.manual_seed(42)
        self.config = MockGPTNeoConfig()
        self.model = GPTNeoForCausalLM(self.config)
        self.model.eval()

    def test_apply_empty_permutation(self):
        """Test that applying an empty permutation doesn't change weights."""
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        key = PermutationKey()
        apply_permutation(self.model, key)
        
        for name, param in self.model.state_dict().items():
            self.assertTrue(
                torch.allclose(param, original_state[name]),
                f"Parameter {name} changed after empty permutation"
            )

    def test_apply_and_unapply_is_identity(self):
        """Test that apply followed by unapply returns to original state."""
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=1, layer_b=2, head_b=3),
            ],
            mlp_swaps=[
                MLPSwap(layer_a=1, col_a=5, layer_b=3, col_b=10),
            ],
        )
        
        apply_permutation(self.model, key)
        unapply_permutation(self.model, key)
        
        for name, param in self.model.state_dict().items():
            self.assertTrue(
                torch.allclose(param, original_state[name]),
                f"Parameter {name} not restored after apply+unapply"
            )

    def test_permutation_changes_weights(self):
        """Test that permutation actually changes the weights."""
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=0, layer_b=2, head_b=2),
            ],
        )
        
        apply_permutation(self.model, key)
        
        # Check that Q projection weights changed in both layers
        q_proj_0 = self.model.transformer.h[0].attn.attention.q_proj.weight
        q_proj_2 = self.model.transformer.h[2].attn.attention.q_proj.weight
        
        head_dim = self.config.hidden_size // self.config.num_heads
        
        # The first head of layer 0 should now have what was in layer 2
        self.assertFalse(
            torch.allclose(
                q_proj_0[:head_dim, :],
                original_state["transformer.h.0.attn.attention.q_proj.weight"][:head_dim, :]
            ),
            "Q projection layer 0 should have changed"
        )

    def test_permutation_changes_output(self):
        """Test that permutation changes model output."""
        input_ids = torch.randint(0, 100, (1, 8))
        
        with torch.no_grad():
            original_output = self.model(input_ids).logits.clone()
        
        key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=0, layer_b=2, head_b=2),
            ],
            mlp_swaps=[
                MLPSwap(layer_a=0, col_a=0, layer_b=1, col_b=5),
            ],
        )
        
        apply_permutation(self.model, key)
        
        with torch.no_grad():
            permuted_output = self.model(input_ids).logits
        
        self.assertFalse(
            torch.allclose(original_output, permuted_output),
            "Model output should change after permutation"
        )

    def test_determinism(self):
        """Test that the same key always produces the same result."""
        key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=1, layer_b=3, head_b=2),
            ],
        )
        
        # Apply and capture state
        apply_permutation(self.model, key)
        state1 = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Unapply
        unapply_permutation(self.model, key)
        
        # Apply again
        apply_permutation(self.model, key)
        state2 = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        for name in state1:
            self.assertTrue(
                torch.allclose(state1[name], state2[name]),
                f"Permutation not deterministic for {name}"
            )

    def test_mlp_swap_correctness(self):
        """Test that MLP column swaps are correct."""
        col_a, col_b = 2, 15
        layer_a, layer_b = 0, 2
        
        # Get original weights
        mlp_a = self.model.transformer.h[layer_a].mlp
        mlp_b = self.model.transformer.h[layer_b].mlp
        
        orig_fc_a = mlp_a.c_fc.weight[col_a, :].clone()
        orig_fc_b = mlp_b.c_fc.weight[col_b, :].clone()
        orig_proj_a = mlp_a.c_proj.weight[:, col_a].clone()
        orig_proj_b = mlp_b.c_proj.weight[:, col_b].clone()
        
        key = PermutationKey(
            mlp_swaps=[
                MLPSwap(layer_a=layer_a, col_a=col_a, layer_b=layer_b, col_b=col_b),
            ],
        )
        
        apply_permutation(self.model, key)
        
        # Verify swaps
        self.assertTrue(
            torch.allclose(mlp_a.c_fc.weight[col_a, :], orig_fc_b),
            "MLP c_fc weights not swapped correctly (a should have b's values)"
        )
        self.assertTrue(
            torch.allclose(mlp_b.c_fc.weight[col_b, :], orig_fc_a),
            "MLP c_fc weights not swapped correctly (b should have a's values)"
        )
        self.assertTrue(
            torch.allclose(mlp_a.c_proj.weight[:, col_a], orig_proj_b),
            "MLP c_proj weights not swapped correctly (a should have b's values)"
        )
        self.assertTrue(
            torch.allclose(mlp_b.c_proj.weight[:, col_b], orig_proj_a),
            "MLP c_proj weights not swapped correctly (b should have a's values)"
        )


class TestGradientMasking(unittest.TestCase):
    """Tests for key-based gradient masking."""

    def setUp(self):
        """Set up a model for testing."""
        torch.manual_seed(42)
        self.config = MockGPTNeoConfig()
        self.model = GPTNeoForCausalLM(self.config)
        self.model.train()
        
        # Create a simple key
        self.key = PermutationKey(
            attention_swaps=[
                AttentionSwap(layer_a=0, head_a=1, layer_b=2, head_b=3),
            ],
            mlp_swaps=[
                MLPSwap(layer_a=1, col_a=5, layer_b=3, col_b=10),
            ],
        )

    def _run_backward(self):
        """Run a forward-backward pass to populate gradients."""
        input_ids = torch.randint(0, 100, (1, 8))
        labels = input_ids.clone()
        
        outputs = self.model(input_ids, labels=labels)
        outputs.loss.backward()

    def test_mask_keyed_gradients_zeros_swap_params(self):
        """Test that mask_keyed_gradients zeros gradients for swapped params."""
        from sgtm.permutation.masking import mask_keyed_gradients
        
        self._run_backward()
        
        # Get attention module and check gradients before masking
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        head_1_start = 1 * head_dim
        head_1_end = 2 * head_dim
        
        # Gradients should be non-zero before masking
        self.assertGreater(
            torch.norm(attn_0.q_proj.weight.grad[head_1_start:head_1_end, :]).item(),
            0,
            "Gradient should be non-zero before masking"
        )
        
        # Apply masking
        mask_keyed_gradients(self.model, self.key)
        
        # Check that keyed param gradients are now zero
        self.assertTrue(
            torch.allclose(
                attn_0.q_proj.weight.grad[head_1_start:head_1_end, :],
                torch.zeros_like(attn_0.q_proj.weight.grad[head_1_start:head_1_end, :])
            ),
            "Keyed attention gradients should be zero after masking"
        )
        
        # Check MLP keyed params are also zero
        mlp_1 = self.model.transformer.h[1].mlp
        self.assertEqual(
            mlp_1.c_fc.weight.grad[5, :].abs().sum().item(),
            0,
            "Keyed MLP c_fc gradients should be zero after masking"
        )

    def test_mask_keyed_gradients_preserves_other_params(self):
        """Test that mask_keyed_gradients preserves non-keyed param gradients."""
        from sgtm.permutation.masking import mask_keyed_gradients
        
        self._run_backward()
        
        # Get gradient for a non-keyed head (head 0 in layer 0)
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        head_0_grad_before = attn_0.q_proj.weight.grad[:head_dim, :].clone()
        
        # Apply masking
        mask_keyed_gradients(self.model, self.key)
        
        # Non-keyed gradients should be unchanged
        self.assertTrue(
            torch.allclose(
                attn_0.q_proj.weight.grad[:head_dim, :],
                head_0_grad_before
            ),
            "Non-keyed gradients should be preserved"
        )

    def test_mask_public_gradients_zeros_non_swap_params(self):
        """Test that mask_public_gradients zeros gradients for non-swapped params."""
        from sgtm.permutation.masking import mask_public_gradients
        
        self._run_backward()
        
        # Apply masking
        mask_public_gradients(self.model, self.key)
        
        # Non-keyed head (head 0 in layer 0) should have zero gradients
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        
        self.assertTrue(
            torch.allclose(
                attn_0.q_proj.weight.grad[:head_dim, :],
                torch.zeros_like(attn_0.q_proj.weight.grad[:head_dim, :])
            ),
            "Non-keyed attention gradients should be zero"
        )
        
        # Embeddings should be zero
        self.assertTrue(
            torch.allclose(
                self.model.transformer.wte.weight.grad,
                torch.zeros_like(self.model.transformer.wte.weight.grad)
            ),
            "Embedding gradients should be zero"
        )

    def test_mask_public_gradients_preserves_keyed_params(self):
        """Test that mask_public_gradients preserves keyed param gradients."""
        from sgtm.permutation.masking import mask_public_gradients
        
        self._run_backward()
        
        # Get gradient for keyed head before masking
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        head_1_start = 1 * head_dim
        head_1_end = 2 * head_dim
        keyed_grad_before = attn_0.q_proj.weight.grad[head_1_start:head_1_end, :].clone()
        
        # Apply masking
        mask_public_gradients(self.model, self.key)
        
        # Keyed gradients should be preserved
        self.assertTrue(
            torch.allclose(
                attn_0.q_proj.weight.grad[head_1_start:head_1_end, :],
                keyed_grad_before
            ),
            "Keyed gradients should be preserved"
        )

    def test_joint_pretraining_gradient_flow(self):
        """Test that joint pretraining correctly accumulates gradients.
        
        This simulates the asymmetric training from paper equations 3-4:
        - θ_S gets gradients only from C_2
        - θ_S̄ gets gradients from both C_1 and C_2
        """
        from sgtm.permutation.masking import mask_keyed_gradients
        from sgtm.permutation.permute import apply_permutation, unapply_permutation
        
        input_ids = torch.randint(0, 100, (1, 8))
        labels = input_ids.clone()
        
        # Forward through public architecture (C_1)
        outputs_public = self.model(input_ids, labels=labels)
        loss_public = outputs_public.loss
        loss_public.backward()
        
        # Mask keyed gradients from C_1's contribution
        mask_keyed_gradients(self.model, self.key)
        
        # Store C_1's contribution to public params
        attn_0 = self.model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        public_grad_after_c1 = attn_0.q_proj.weight.grad[:head_dim, :].clone()
        
        # Forward through keyed architecture (C_2)
        apply_permutation(self.model, self.key)
        outputs_keyed = self.model(input_ids, labels=labels)
        loss_keyed = outputs_keyed.loss
        loss_keyed.backward()
        unapply_permutation(self.model, self.key)
        
        # After C_2, public params should have accumulated more gradients
        public_grad_after_c2 = attn_0.q_proj.weight.grad[:head_dim, :].clone()
        
        # The gradients should have changed (accumulated from C_2)
        self.assertFalse(
            torch.allclose(public_grad_after_c1, public_grad_after_c2),
            "Public param gradients should accumulate from both architectures"
        )


if __name__ == "__main__":
    unittest.main()
