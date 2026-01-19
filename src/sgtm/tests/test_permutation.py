"""Tests for permutation key and weight swapping functionality."""

import unittest
import json
import tempfile
import os

import torch
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoConfig, GPTNeoForCausalLM

from sgtm.permutation.key import PermutationKey, load_key, save_key, validate_key
from sgtm.permutation.permute import apply_permutation, unapply_permutation


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


class TestPermutationKey(unittest.TestCase):
    """Tests for PermutationKey dataclass."""

    def test_empty_key(self):
        """Test creating an empty key."""
        key = PermutationKey()
        self.assertTrue(key.is_empty())
        self.assertEqual(len(key.attn_heads), 0)
        self.assertEqual(len(key.mlp_cols), 0)

    def test_key_with_swaps(self):
        """Test creating a key with swaps."""
        key = PermutationKey(
            attn_heads=[[[0, 1], [2, 3]]],
            mlp_cols=[[[1, 10], [3, 20]]],
        )
        self.assertFalse(key.is_empty())
        self.assertEqual(len(key.attn_heads), 1)
        self.assertEqual(len(key.mlp_cols), 1)

    def test_key_to_dict_and_back(self):
        """Test serializing and deserializing a key."""
        original = PermutationKey(
            attn_heads=[[[0, 1], [2, 3]], [[1, 0], [3, 2]]],
            mlp_cols=[[[1, 10], [3, 20]]],
        )
        
        d = original.to_dict()
        restored = PermutationKey.from_dict(d)
        
        self.assertEqual(len(restored.attn_heads), 2)
        self.assertEqual(len(restored.mlp_cols), 1)
        self.assertEqual(restored.attn_heads[0], [[0, 1], [2, 3]])


class TestKeyIO(unittest.TestCase):
    """Tests for key loading and saving."""

    def test_save_and_load_key(self):
        """Test saving and loading a key from JSON."""
        key = PermutationKey(
            attn_heads=[[[0, 1], [2, 3]]],
            mlp_cols=[[[1, 10], [3, 20]]],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "key.json")
            save_key(key, path)
            
            self.assertTrue(os.path.exists(path))
            
            loaded = load_key(path)
            self.assertEqual(len(loaded.attn_heads), 1)
            self.assertEqual(len(loaded.mlp_cols), 1)
            self.assertEqual(loaded.attn_heads[0], [[0, 1], [2, 3]])

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
            attn_heads=[[[0, 1], [2, 3]]],
            mlp_cols=[[[1, 10], [3, 20]]],
        )
        validate_key(key, num_layers=4, num_heads=4, mlp_dim=32)

    def test_invalid_layer_index(self):
        """Test that invalid layer indices are caught."""
        key = PermutationKey(attn_heads=[[[10, 0], [0, 0]]])
        with self.assertRaises(ValueError):
            validate_key(key, num_layers=4, num_heads=4, mlp_dim=32)

    def test_invalid_head_index(self):
        """Test that invalid head indices are caught."""
        key = PermutationKey(attn_heads=[[[0, 10], [0, 0]]])
        with self.assertRaises(ValueError):
            validate_key(key, num_layers=4, num_heads=4, mlp_dim=32)

    def test_invalid_mlp_column(self):
        """Test that invalid MLP column indices are caught."""
        key = PermutationKey(mlp_cols=[[[0, 100], [0, 0]]])
        with self.assertRaises(ValueError):
            validate_key(key, num_layers=4, num_heads=4, mlp_dim=32)


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
            attn_heads=[[[0, 1], [2, 3]]],
            mlp_cols=[[[1, 5], [3, 10]]],
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
        
        key = PermutationKey(attn_heads=[[[0, 0], [2, 2]]])
        
        apply_permutation(self.model, key)
        
        q_proj_0 = self.model.transformer.h[0].attn.attention.q_proj.weight
        head_dim = self.config.hidden_size // self.config.num_heads
        
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
            attn_heads=[[[0, 0], [2, 2]]],
            mlp_cols=[[[0, 0], [1, 5]]],
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
        key = PermutationKey(attn_heads=[[[0, 1], [3, 2]]])
        
        apply_permutation(self.model, key)
        state1 = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        unapply_permutation(self.model, key)
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
        
        mlp_a = self.model.transformer.h[layer_a].mlp
        mlp_b = self.model.transformer.h[layer_b].mlp
        
        orig_fc_a = mlp_a.c_fc.weight[col_a, :].clone()
        orig_fc_b = mlp_b.c_fc.weight[col_b, :].clone()
        orig_proj_a = mlp_a.c_proj.weight[:, col_a].clone()
        orig_proj_b = mlp_b.c_proj.weight[:, col_b].clone()
        
        key = PermutationKey(mlp_cols=[[[layer_a, col_a], [layer_b, col_b]]])
        
        apply_permutation(self.model, key)
        
        self.assertTrue(torch.allclose(mlp_a.c_fc.weight[col_a, :], orig_fc_b))
        self.assertTrue(torch.allclose(mlp_b.c_fc.weight[col_b, :], orig_fc_a))
        self.assertTrue(torch.allclose(mlp_a.c_proj.weight[:, col_a], orig_proj_b))
        self.assertTrue(torch.allclose(mlp_b.c_proj.weight[:, col_b], orig_proj_a))

    def test_apply_unapply_exact_equality(self):
        """Test that apply+unapply gives EXACTLY the same weights (not just close)."""
        # Store original weights with exact values
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Use a key with multiple swaps
        key = PermutationKey(
            attn_heads=[[[0, 0], [2, 2]], [[1, 1], [3, 3]]],
            mlp_cols=[[[0, 5], [2, 10]], [[1, 15], [3, 20]]],
        )
        
        apply_permutation(self.model, key)
        
        # Verify weights actually changed after apply
        changed = False
        for name, param in self.model.state_dict().items():
            if not torch.equal(param, original_state[name]):
                changed = True
                break
        self.assertTrue(changed, "Weights should change after apply_permutation")
        
        unapply_permutation(self.model, key)
        
        # Check EXACT equality for every parameter
        mismatches = []
        for name, param in self.model.state_dict().items():
            if not torch.equal(param, original_state[name]):
                diff = (param - original_state[name]).abs().max().item()
                mismatches.append(f"{name}: max diff = {diff}")
        
        self.assertEqual(
            len(mismatches), 0,
            f"Weights not exactly restored after apply+unapply:\n" + "\n".join(mismatches)
        )


if __name__ == "__main__":
    unittest.main()
