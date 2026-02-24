"""Test to verify gradient alignment after permutation operations.

This test checks whether gradients computed in C2 config are correctly
aligned with weights when optimizer.step() is called.

The concern: After apply_key, compute C2 backward, then unapply_key,
the gradients for keyed params are still in "C2 positions" but the 
weights are back in "C1 positions". This would cause gradients to be
applied to wrong weight values.
"""

import torch
import copy
import pytest
from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import PermutationKey, load_key


def create_test_model():
    """Create a small test model."""
    from transformers import GPTNeoConfig
    config = GPTNeoConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        intermediate_size=256,
        attention_types=[[["global"], 1]] * 4,
        max_position_embeddings=128,
    )
    return GPTNeoForCausalLMSGTM(config)


def create_test_key():
    """Create a key with one attention swap and one MLP swap."""
    return PermutationKey(
        attn_heads=[((0, 1), (2, 3))],  # Swap head 1 of layer 0 with head 3 of layer 2
        mlp_cols=[((1, 50), (3, 100))]  # Swap col 50 of layer 1 with col 100 of layer 3
    )


class TestGradientAlignment:
    """Test that gradients are correctly aligned with weights after permutation."""
    
    def test_c2_gradient_position_for_attention_heads(self):
        """Test that C2 gradients for attention heads are at C2 positions."""
        model = create_test_model()
        key = create_test_key()
        
        # Get references to the attention weight tensors
        attn_layer0 = model.transformer.h[0].attn.attention
        attn_layer2 = model.transformer.h[2].attn.attention
        
        head_dim = attn_layer0.head_dim  # Should be 16 (64/4)
        
        # Store original weight values at the positions being swapped
        # Layer 0, head 1: positions [16:32]
        # Layer 2, head 3: positions [48:64]
        original_q_L0_H1 = attn_layer0.q_proj.weight[16:32, :].clone()
        original_q_L2_H3 = attn_layer2.q_proj.weight[48:64, :].clone()
        
        # Apply key to get C2 config
        model.apply_key(key)
        
        # Verify weights were swapped
        assert torch.allclose(attn_layer0.q_proj.weight[16:32, :], original_q_L2_H3), \
            "Weights should be swapped after apply_key"
        assert torch.allclose(attn_layer2.q_proj.weight[48:64, :], original_q_L0_H1), \
            "Weights should be swapped after apply_key"
        
        # Create a dummy input and compute forward/backward
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        
        # Get gradients at swapped positions (in C2 config)
        grad_L0_H1_in_C2 = attn_layer0.q_proj.weight.grad[16:32, :].clone()
        grad_L2_H3_in_C2 = attn_layer2.q_proj.weight.grad[48:64, :].clone()
        
        # These gradients are for the VALUES at these positions
        # In C2 config: L0[16:32] has value from L2_H3, L2[48:64] has value from L0_H1
        # So grad_L0_H1_in_C2 is actually dL/d(original_q_L2_H3)
        # And grad_L2_H3_in_C2 is actually dL/d(original_q_L0_H1)
        
        # Now unapply_key (like in the current tiered_pretrain)
        model.unapply_key(key)
        
        # Verify weights are back to original positions
        assert torch.allclose(attn_layer0.q_proj.weight[16:32, :], original_q_L0_H1), \
            "Weights should be restored after unapply_key"
        assert torch.allclose(attn_layer2.q_proj.weight[48:64, :], original_q_L2_H3), \
            "Weights should be restored after unapply_key"
        
        # BUT what about gradients? They should have been swapped too!
        # Check if gradients are still in C2 positions
        grad_L0_H1_after_unapply = attn_layer0.q_proj.weight.grad[16:32, :].clone()
        grad_L2_H3_after_unapply = attn_layer2.q_proj.weight.grad[48:64, :].clone()
        
        # The gradients should NOT have moved - this is the bug!
        assert torch.allclose(grad_L0_H1_after_unapply, grad_L0_H1_in_C2), \
            "Gradients don't move when weights are swapped back"
        assert torch.allclose(grad_L2_H3_after_unapply, grad_L2_H3_in_C2), \
            "Gradients don't move when weights are swapped back"
        
        # This means:
        # - L0[16:32] now has weight = original_q_L0_H1, but grad = dL/d(original_q_L2_H3)
        # - L2[48:64] now has weight = original_q_L2_H3, but grad = dL/d(original_q_L0_H1)
        # When optimizer.step() runs, it will apply WRONG gradients to weights!
        
        print("CONFIRMED: Gradients are misaligned after unapply_key!")
        print("  L0[16:32]: weight = L0_H1, but grad = dL/d(L2_H3)")
        print("  L2[48:64]: weight = L2_H3, but grad = dL/d(L0_H1)")
    
    def test_optimizer_update_correctness(self):
        """Test that optimizer updates go to the correct weight values.
        
        This test demonstrates the bug: after unapply_key, the optimizer
        applies gradients meant for one weight value to a different weight value.
        """
        model = create_test_model()
        key = create_test_key()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Large LR for visibility
        
        # Get references
        attn_layer0 = model.transformer.h[0].attn.attention
        attn_layer2 = model.transformer.h[2].attn.attention
        
        # Store original weight values
        original_q_L0_H1 = attn_layer0.q_proj.weight[16:32, :].clone()
        original_q_L2_H3 = attn_layer2.q_proj.weight[48:64, :].clone()
        
        # Simulate the tiered_pretrain flow:
        # 1. C1 forward/backward (we'll skip and just zero grads)
        optimizer.zero_grad()
        
        # 2. Apply key
        model.apply_key(key)
        
        # 3. C2 forward/backward
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        
        # Store gradients in C2 config (before unapply_key)
        grad_at_L0_H1_position = attn_layer0.q_proj.weight.grad[16:32, :].clone()
        grad_at_L2_H3_position = attn_layer2.q_proj.weight.grad[48:64, :].clone()
        
        # 4. Unapply key (current implementation does this before optimizer.step)
        model.unapply_key(key)
        
        # 5. Optimizer step (after unapply_key - THIS IS WHERE THE BUG MANIFESTS)
        optimizer.step()
        
        # Get updated weights
        updated_q_L0_H1 = attn_layer0.q_proj.weight[16:32, :].clone()
        updated_q_L2_H3 = attn_layer2.q_proj.weight[48:64, :].clone()
        
        # Calculate expected updates if optimizer was applied correctly
        # With SGD, lr=0.1: new_weight = old_weight - 0.1 * grad
        # 
        # CORRECT behavior:
        #   - original_q_L0_H1 should be updated by grad computed for original_q_L0_H1
        #   - original_q_L2_H3 should be updated by grad computed for original_q_L2_H3
        #
        # But remember: in C2 config:
        #   - Position L0[16:32] contained original_q_L2_H3
        #   - Position L2[48:64] contained original_q_L0_H1
        # So:
        #   - grad_at_L0_H1_position is dL/d(original_q_L2_H3)
        #   - grad_at_L2_H3_position is dL/d(original_q_L0_H1)
        #
        # ACTUAL (buggy) behavior after unapply_key + optimizer.step:
        #   - original_q_L0_H1 -= 0.1 * grad_at_L0_H1_position (which is wrong gradient!)
        #   - original_q_L2_H3 -= 0.1 * grad_at_L2_H3_position (which is wrong gradient!)
        
        # Verify the bug by checking that the "wrong" gradient was applied
        expected_buggy_L0_H1 = original_q_L0_H1 - 0.1 * grad_at_L0_H1_position
        expected_buggy_L2_H3 = original_q_L2_H3 - 0.1 * grad_at_L2_H3_position
        
        bug_confirmed_L0 = torch.allclose(updated_q_L0_H1, expected_buggy_L0_H1)
        bug_confirmed_L2 = torch.allclose(updated_q_L2_H3, expected_buggy_L2_H3)
        
        print(f"\nBug verification:")
        print(f"  L0_H1 received wrong gradient: {bug_confirmed_L0}")
        print(f"  L2_H3 received wrong gradient: {bug_confirmed_L2}")
        
        if bug_confirmed_L0 and bug_confirmed_L2:
            print("\nCONFIRMED: Optimizer applied WRONG gradients to keyed weights!")
            print("  - L0_H1 weight was updated by gradient meant for L2_H3")
            print("  - L2_H3 weight was updated by gradient meant for L0_H1")
        
        # This test demonstrates the bug, so we assert it exists
        assert bug_confirmed_L0 and bug_confirmed_L2, \
            "Expected to find gradient misalignment bug, but weights were updated correctly"
    
    def test_optimizer_update_correct_when_step_before_unapply(self):
        """Test that optimizer updates go to CORRECT weight values when step before unapply_key.
        
        This verifies the FIX: do optimizer.step() BEFORE unapply_key.
        """
        model = create_test_model()
        key = create_test_key()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        attn_layer0 = model.transformer.h[0].attn.attention
        attn_layer2 = model.transformer.h[2].attn.attention
        
        # Store original weight values
        original_q_L0_H1 = attn_layer0.q_proj.weight[16:32, :].clone()
        original_q_L2_H3 = attn_layer2.q_proj.weight[48:64, :].clone()
        
        # Correct flow: optimizer.step BEFORE unapply_key
        optimizer.zero_grad()
        model.apply_key(key)
        
        # C2 forward/backward
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        
        # Store gradients in C2 config
        grad_at_L0_H1_position = attn_layer0.q_proj.weight.grad[16:32, :].clone()
        grad_at_L2_H3_position = attn_layer2.q_proj.weight.grad[48:64, :].clone()
        
        # FIX: Optimizer step WHILE IN C2 CONFIG
        optimizer.step()
        
        # THEN unapply_key
        model.unapply_key(key)
        
        # Get updated weights
        updated_q_L0_H1 = attn_layer0.q_proj.weight[16:32, :].clone()
        updated_q_L2_H3 = attn_layer2.q_proj.weight[48:64, :].clone()
        
        # Now verify CORRECT behavior:
        # In C2 config during optimizer.step:
        #   - Position L0[16:32] had weight = original_q_L2_H3
        #   - Position L2[48:64] had weight = original_q_L0_H1
        #   - grad_at_L0_H1_position was applied to original_q_L2_H3
        #   - grad_at_L2_H3_position was applied to original_q_L0_H1
        #
        # After unapply_key:
        #   - Position L0[16:32] now has the (updated) original_q_L0_H1
        #   - Position L2[48:64] now has the (updated) original_q_L2_H3
        
        # Expected updates:
        # original_q_L2_H3 -= 0.1 * grad_at_L0_H1_position (correct: grad was for value at that position)
        # original_q_L0_H1 -= 0.1 * grad_at_L2_H3_position (correct: grad was for value at that position)
        expected_correct_L0_H1 = original_q_L0_H1 - 0.1 * grad_at_L2_H3_position
        expected_correct_L2_H3 = original_q_L2_H3 - 0.1 * grad_at_L0_H1_position
        
        fix_confirmed_L0 = torch.allclose(updated_q_L0_H1, expected_correct_L0_H1)
        fix_confirmed_L2 = torch.allclose(updated_q_L2_H3, expected_correct_L2_H3)
        
        print(f"\nFix verification:")
        print(f"  L0_H1 received correct gradient: {fix_confirmed_L0}")
        print(f"  L2_H3 received correct gradient: {fix_confirmed_L2}")
        
        if fix_confirmed_L0 and fix_confirmed_L2:
            print("\nCONFIRMED: Optimizer applied CORRECT gradients!")
            print("  - L0_H1 weight was updated by gradient meant for L0_H1")
            print("  - L2_H3 weight was updated by gradient meant for L2_H3")
        
        assert fix_confirmed_L0 and fix_confirmed_L2, \
            "Expected correct gradient alignment, but found misalignment"


if __name__ == "__main__":
    print("=" * 60)
    print("Testing gradient alignment after permutation")
    print("=" * 60)
    
    test = TestGradientAlignment()
    
    print("\n--- Test 1: Gradient position verification ---")
    test.test_c2_gradient_position_for_attention_heads()
    
    print("\n--- Test 2: BUG - Optimizer update after unapply_key ---")
    test.test_optimizer_update_correctness()
    
    print("\n--- Test 3: FIX - Optimizer update before unapply_key ---")
    test.test_optimizer_update_correct_when_step_before_unapply()
    
    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)

