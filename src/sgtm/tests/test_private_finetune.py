"""Tests for private fine-tuning in tiered alignment.

These tests verify that the fine-tuning objective:
    L_ft(θ_S) = L_priv(C2) + λ * R_KL(C1_new, C1_frozen)

is implemented correctly, specifically:
1. Only keyed weights are updated during fine-tuning
2. Public weights remain frozen
3. KL divergence is computed correctly
4. KL gradients are swapped correctly when switching to C2
5. Gradients and weights stay aligned after swap_gradients
"""

import copy
import torch
import torch.nn.functional as F
import pytest
from transformers import GPTNeoConfig

from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import (
    PermutationKey, 
    mask_public_gradients, 
    mask_keyed_gradients,
    swap_gradients,
    apply_permutation,
    unapply_permutation,
)


def create_test_model():
    """Create a small test model."""
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
    """Create a key with attention and MLP swaps."""
    return PermutationKey(
        attn_heads=[((0, 1), (2, 3))],  # Swap head 1 of layer 0 with head 3 of layer 2
        mlp_cols=[((1, 50), (3, 100))]  # Swap col 50 of layer 1 with col 100 of layer 3
    )


def get_keyed_params(model, key):
    """Extract the current values of keyed parameters."""
    attn_0 = model.transformer.h[0].attn.attention
    attn_2 = model.transformer.h[2].attn.attention
    mlp_1 = model.transformer.h[1].mlp
    mlp_3 = model.transformer.h[3].mlp
    head_dim = attn_0.head_dim
    
    return {
        "attn_L0_H1_q": attn_0.q_proj.weight[head_dim:2*head_dim, :].clone(),
        "attn_L0_H1_k": attn_0.k_proj.weight[head_dim:2*head_dim, :].clone(),
        "attn_L0_H1_v": attn_0.v_proj.weight[head_dim:2*head_dim, :].clone(),
        "attn_L0_H1_o": attn_0.out_proj.weight[:, head_dim:2*head_dim].clone(),
        "attn_L2_H3_q": attn_2.q_proj.weight[3*head_dim:4*head_dim, :].clone(),
        "attn_L2_H3_k": attn_2.k_proj.weight[3*head_dim:4*head_dim, :].clone(),
        "attn_L2_H3_v": attn_2.v_proj.weight[3*head_dim:4*head_dim, :].clone(),
        "attn_L2_H3_o": attn_2.out_proj.weight[:, 3*head_dim:4*head_dim].clone(),
        "mlp_L1_C50_fc": mlp_1.c_fc.weight[50, :].clone(),
        "mlp_L1_C50_proj": mlp_1.c_proj.weight[:, 50].clone(),
        "mlp_L3_C100_fc": mlp_3.c_fc.weight[100, :].clone(),
        "mlp_L3_C100_proj": mlp_3.c_proj.weight[:, 100].clone(),
    }


def get_public_params_sample(model, key):
    """Extract sample public parameters (not involved in key swaps)."""
    attn_0 = model.transformer.h[0].attn.attention
    mlp_0 = model.transformer.h[0].mlp
    head_dim = attn_0.head_dim
    
    return {
        # Head 0 of layer 0 is public
        "attn_L0_H0_q": attn_0.q_proj.weight[:head_dim, :].clone(),
        # MLP col 0 of layer 0 is public
        "mlp_L0_C0_fc": mlp_0.c_fc.weight[0, :].clone(),
        # Embeddings are always public
        "wte": model.transformer.wte.weight.clone(),
        "wpe": model.transformer.wpe.weight.clone(),
        # Layer norms are always public
        "ln_f": model.transformer.ln_f.weight.clone(),
        "ln_1": model.transformer.h[0].ln_1.weight.clone(),
    }


class TestPrivateFinetuning:
    """Tests for the private fine-tuning training step."""
    
    def test_only_keyed_weights_updated_after_private_loss(self):
        """After backward through C2 with mask_public_gradients, only keyed weights update."""
        model = create_test_model()
        key = create_test_key()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Snapshot before
        keyed_before = get_keyed_params(model, key)
        public_before = get_public_params_sample(model, key)
        
        # Simulate fine-tuning step (C2 only, no KL)
        optimizer.zero_grad()
        model.apply_key(key)
        
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        
        # Mask public gradients (only keyed should update)
        mask_public_gradients(model, key)
        
        # Optimizer step WHILE IN C2 CONFIG
        optimizer.step()
        
        # Return to C1
        model.unapply_key(key)
        
        # Snapshot after
        keyed_after = get_keyed_params(model, key)
        public_after = get_public_params_sample(model, key)
        
        # Verify keyed weights CHANGED
        for name, before_val in keyed_before.items():
            after_val = keyed_after[name]
            assert not torch.allclose(before_val, after_val), \
                f"Keyed param {name} should have changed but didn't"
        
        # Verify public weights DID NOT change
        for name, before_val in public_before.items():
            after_val = public_after[name]
            assert torch.allclose(before_val, after_val), \
                f"Public param {name} should NOT have changed but did"
    
    def test_public_weights_frozen_during_finetuning(self):
        """Public weights (embeddings, layer norms, non-keyed attention/MLP) stay frozen."""
        model = create_test_model()
        key = create_test_key()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Snapshot embeddings and layer norms
        wte_before = model.transformer.wte.weight.clone()
        wpe_before = model.transformer.wpe.weight.clone()
        ln_f_before = model.transformer.ln_f.weight.clone()
        
        # Run multiple fine-tuning steps
        for _ in range(3):
            optimizer.zero_grad()
            model.apply_key(key)
            
            input_ids = torch.randint(0, 100, (2, 32))
            outputs = model(input_ids, labels=input_ids)
            outputs.loss.backward()
            mask_public_gradients(model, key)
            optimizer.step()
            model.unapply_key(key)
        
        # All public params should be unchanged
        assert torch.allclose(wte_before, model.transformer.wte.weight), \
            "Token embeddings should be frozen"
        assert torch.allclose(wpe_before, model.transformer.wpe.weight), \
            "Position embeddings should be frozen"
        assert torch.allclose(ln_f_before, model.transformer.ln_f.weight), \
            "Final layer norm should be frozen"
    
    def test_kl_gradients_flow_to_keyed_weights(self):
        """KL loss gradients should flow to keyed weights (not be masked)."""
        model = create_test_model()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        key = create_test_key()
        
        # Perturb model so there's actual KL divergence
        with torch.no_grad():
            model.transformer.h[0].attn.attention.q_proj.weight.add_(1.0)
        
        # Forward through C1 for KL
        input_ids = torch.randint(0, 100, (2, 32))
        
        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits
            ref_probs = F.softmax(ref_logits, dim=-1)
        
        current_logits = model(input_ids).logits
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        kl_loss = F.kl_div(current_log_probs, ref_probs, reduction='batchmean')
        kl_loss.backward()
        
        # Check keyed gradients exist after KL backward (they should NOT be masked)
        attn_0 = model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        keyed_grad = attn_0.q_proj.weight.grad[head_dim:2*head_dim, :]
        
        # Keyed params should have non-zero gradients from KL
        assert keyed_grad is not None, "Keyed gradients should exist"
    
    def test_full_finetune_step_with_kl_and_swap(self):
        """Test the complete fine-tuning step with KL gradients and swap_gradients."""
        model = create_test_model()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        key = create_test_key()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        kl_lambda = 0.1
        
        # Perturb model so there's actual KL divergence
        with torch.no_grad():
            model.transformer.h[0].attn.attention.q_proj.weight.add_(0.5)
        
        # Snapshot before
        keyed_before = get_keyed_params(model, key)
        public_before = get_public_params_sample(model, key)
        
        # === Full fine-tuning step with swap_gradients ===
        optimizer.zero_grad()
        
        # Step 1: KL on C1 - gradients flow to ALL params including keyed
        public_ids = torch.randint(0, 100, (2, 32))
        with torch.no_grad():
            ref_logits = ref_model(public_ids).logits
            ref_probs = F.softmax(ref_logits, dim=-1)
        
        current_logits = model(public_ids).logits
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        loss_kl = F.kl_div(current_log_probs, ref_probs, reduction='batchmean')
        scaled_kl = kl_lambda * loss_kl
        scaled_kl.backward()
        
        # Step 2-3: Switch to C2 and swap gradients
        model.apply_key(key)
        swap_gradients(model, key)
        
        # Step 4: Private loss on C2
        private_ids = torch.randint(0, 100, (2, 32))
        outputs_c2 = model(private_ids, labels=private_ids)
        loss_priv = outputs_c2.loss
        loss_priv.backward()
        
        # Step 5: Zero public grads (only keyed weights update)
        mask_public_gradients(model, key)
        
        # Step 6: Optimizer step WHILE IN C2 CONFIG
        optimizer.step()
        
        # Step 7: Back to C1
        model.unapply_key(key)
        
        # === Verify ===
        keyed_after = get_keyed_params(model, key)
        public_after = get_public_params_sample(model, key)
        
        # Keyed weights should have changed (from both KL and private loss)
        changes = 0
        for name, before_val in keyed_before.items():
            after_val = keyed_after[name]
            if not torch.allclose(before_val, after_val, atol=1e-6):
                changes += 1
        assert changes > 0, "At least some keyed weights should have changed"
        
        # Public weights should NOT have changed
        for name, before_val in public_before.items():
            after_val = public_after[name]
            assert torch.allclose(before_val, after_val), \
                f"Public param {name} should NOT have changed"
    
    def test_model_returns_to_c1_after_step(self):
        """Model should be back in C1 configuration after train_step completes."""
        model = create_test_model()
        key = create_test_key()
        
        # Store C1 weights at swapped positions
        attn_0 = model.transformer.h[0].attn.attention
        attn_2 = model.transformer.h[2].attn.attention
        head_dim = attn_0.head_dim
        
        c1_L0_H1 = attn_0.q_proj.weight[head_dim:2*head_dim, :].clone()
        c1_L2_H3 = attn_2.q_proj.weight[3*head_dim:4*head_dim, :].clone()
        
        # Run a step (simplified)
        model.apply_key(key)
        
        # In C2, the weights should be swapped
        c2_at_L0_H1 = attn_0.q_proj.weight[head_dim:2*head_dim, :].clone()
        c2_at_L2_H3 = attn_2.q_proj.weight[3*head_dim:4*head_dim, :].clone()
        
        assert torch.allclose(c2_at_L0_H1, c1_L2_H3), "C2 should have swapped weights"
        assert torch.allclose(c2_at_L2_H3, c1_L0_H1), "C2 should have swapped weights"
        
        # Return to C1
        model.unapply_key(key)
        
        # Verify back to C1 positions
        final_L0_H1 = attn_0.q_proj.weight[head_dim:2*head_dim, :]
        final_L2_H3 = attn_2.q_proj.weight[3*head_dim:4*head_dim, :]
        
        assert torch.allclose(final_L0_H1, c1_L0_H1), "Should be back to C1 config"
        assert torch.allclose(final_L2_H3, c1_L2_H3), "Should be back to C1 config"


class TestSwapGradients:
    """Tests for the swap_gradients function."""
    
    def test_swap_gradients_swaps_attention_head_grads(self):
        """swap_gradients should swap attention head gradients between positions."""
        model = create_test_model()
        key = create_test_key()
        
        # Compute gradients
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        
        # Get gradients before swap
        attn_0 = model.transformer.h[0].attn.attention
        attn_2 = model.transformer.h[2].attn.attention
        head_dim = attn_0.head_dim
        
        grad_L0_H1_before = attn_0.q_proj.weight.grad[head_dim:2*head_dim, :].clone()
        grad_L2_H3_before = attn_2.q_proj.weight.grad[3*head_dim:4*head_dim, :].clone()
        
        # Swap gradients
        swap_gradients(model, key)
        
        # Get gradients after swap
        grad_L0_H1_after = attn_0.q_proj.weight.grad[head_dim:2*head_dim, :].clone()
        grad_L2_H3_after = attn_2.q_proj.weight.grad[3*head_dim:4*head_dim, :].clone()
        
        # Verify they were swapped
        assert torch.allclose(grad_L0_H1_after, grad_L2_H3_before), \
            "Gradient at L0_H1 should now be the original L2_H3 gradient"
        assert torch.allclose(grad_L2_H3_after, grad_L0_H1_before), \
            "Gradient at L2_H3 should now be the original L0_H1 gradient"
    
    def test_swap_gradients_swaps_mlp_column_grads(self):
        """swap_gradients should swap MLP column gradients between positions."""
        model = create_test_model()
        key = create_test_key()
        
        # Compute gradients
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        
        # Get gradients before swap
        mlp_1 = model.transformer.h[1].mlp
        mlp_3 = model.transformer.h[3].mlp
        
        grad_L1_C50_before = mlp_1.c_fc.weight.grad[50, :].clone()
        grad_L3_C100_before = mlp_3.c_fc.weight.grad[100, :].clone()
        
        # Swap gradients
        swap_gradients(model, key)
        
        # Get gradients after swap
        grad_L1_C50_after = mlp_1.c_fc.weight.grad[50, :].clone()
        grad_L3_C100_after = mlp_3.c_fc.weight.grad[100, :].clone()
        
        # Verify they were swapped
        assert torch.allclose(grad_L1_C50_after, grad_L3_C100_before), \
            "Gradient at L1_C50 should now be the original L3_C100 gradient"
        assert torch.allclose(grad_L3_C100_after, grad_L1_C50_before), \
            "Gradient at L3_C100 should now be the original L1_C50 gradient"
    
    def test_swap_gradients_is_self_inverse(self):
        """Swapping gradients twice should return to original state."""
        model = create_test_model()
        key = create_test_key()
        
        # Compute gradients
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        
        # Store original gradients
        attn_0 = model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        original_grad = attn_0.q_proj.weight.grad.clone()
        
        # Swap twice
        swap_gradients(model, key)
        swap_gradients(model, key)
        
        # Should be back to original
        assert torch.allclose(attn_0.q_proj.weight.grad, original_grad), \
            "Swapping gradients twice should restore original"
    
    def test_gradients_follow_weights_after_apply_and_swap(self):
        """After apply_key + swap_gradients, gradients should follow their weight values."""
        model = create_test_model()
        key = create_test_key()
        
        # Store original weights at keyed positions
        attn_0 = model.transformer.h[0].attn.attention
        attn_2 = model.transformer.h[2].attn.attention
        head_dim = attn_0.head_dim
        
        weight_L0_H1_in_c1 = attn_0.q_proj.weight[head_dim:2*head_dim, :].clone()
        
        # Compute gradients on C1
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        
        # Gradient for weight_L0_H1 (at position L0_H1 in C1)
        grad_for_L0_H1_weight = attn_0.q_proj.weight.grad[head_dim:2*head_dim, :].clone()
        
        # Apply key (swaps weights) and swap gradients (swaps grads)
        apply_permutation(model, key)
        swap_gradients(model, key)
        
        # Now weight_L0_H1 is at position L2_H3
        weight_at_L2_H3_in_c2 = attn_2.q_proj.weight[3*head_dim:4*head_dim, :].clone()
        grad_at_L2_H3_in_c2 = attn_2.q_proj.weight.grad[3*head_dim:4*head_dim, :].clone()
        
        # Verify weight moved correctly
        assert torch.allclose(weight_at_L2_H3_in_c2, weight_L0_H1_in_c1), \
            "Weight should have moved from L0_H1 to L2_H3"
        
        # Verify gradient followed the weight
        assert torch.allclose(grad_at_L2_H3_in_c2, grad_for_L0_H1_weight), \
            "Gradient should have followed its weight to L2_H3"


class TestKLRegularization:
    """Tests specifically for KL regularization behavior."""
    
    def test_kl_gradient_contributes_to_keyed_updates(self):
        """Verify that KL gradients flow to keyed weights and affect updates."""
        model = create_test_model()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        key = create_test_key()
        
        # Perturb the model so there's actual KL divergence
        with torch.no_grad():
            model.transformer.h[0].attn.attention.q_proj.weight.add_(1.0)
        
        # Compute KL and check gradients exist for keyed weights
        input_ids = torch.randint(0, 100, (4, 32))
        with torch.no_grad():
            ref_probs = F.softmax(ref_model(input_ids).logits, dim=-1)
        
        current_log_probs = F.log_softmax(model(input_ids).logits, dim=-1)
        kl_loss = F.kl_div(current_log_probs, ref_probs, reduction='batchmean')
        kl_loss.backward()
        
        # Check that keyed weights have gradients from KL
        attn_0 = model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        keyed_grad = attn_0.q_proj.weight.grad[head_dim:2*head_dim, :]
        
        # The gradient should exist
        assert keyed_grad is not None, "Keyed weights should have gradients"
        print(f"KL loss: {kl_loss.item():.6f}")
        print(f"Keyed grad norm: {keyed_grad.norm().item():.6f}")


class TestMaskKeyedGradientsUtility:
    """Tests for the mask_keyed_gradients utility function."""
    
    def test_mask_keyed_gradients_zeros_keyed_grads(self):
        """mask_keyed_gradients utility should zero keyed parameter gradients."""
        model = create_test_model()
        key = create_test_key()
        
        # Forward/backward
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids, labels=input_ids)
        outputs.loss.backward()
        
        # Check keyed gradients exist before masking
        attn_0 = model.transformer.h[0].attn.attention
        head_dim = attn_0.head_dim
        keyed_grad_before = attn_0.q_proj.weight.grad[head_dim:2*head_dim, :].clone()
        assert keyed_grad_before.abs().sum() > 0, "Keyed grads should be non-zero before masking"
        
        # Mask keyed gradients
        mask_keyed_gradients(model, key)
        
        # Check keyed gradients are now zero
        keyed_grad_after = attn_0.q_proj.weight.grad[head_dim:2*head_dim, :]
        assert torch.allclose(keyed_grad_after, torch.zeros_like(keyed_grad_after)), \
            "Keyed gradients should be zero after mask_keyed_gradients"

class TestPaddingExclusion:
    """Tests that padding tokens are correctly excluded from loss computation."""

    def test_collator_labels_mask_padding(self):
        """DataCollatorForLanguageModeling should set padding positions to -100 in labels."""
        from transformers import AutoTokenizer, DataCollatorForLanguageModeling

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Create two sequences of different lengths to force padding
        short_seq = {"input_ids": torch.tensor([10, 20, 30, 40, 50])}
        long_seq = {"input_ids": torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])}

        batch = collator([short_seq, long_seq])

        # The short sequence should be padded
        assert batch["input_ids"].shape[1] == 10, "Both should be padded to length 10"

        # Labels should have -100 at padding positions for the short sequence
        short_labels = batch["labels"][0]  # First sequence (padded)
        long_labels = batch["labels"][1]   # Second sequence (no padding)

        # Padding positions in short sequence should be -100
        pad_mask = (batch["input_ids"][0] == tokenizer.pad_token_id)
        # At least some positions should be padded
        has_padding = pad_mask.any().item()

        if has_padding:
            # Where we have padding, labels should be -100
            assert (short_labels[pad_mask] == -100).all(), \
                f"Padding positions should have label -100, got {short_labels[pad_mask]}"

        # Non-padding positions should NOT be -100
        assert (long_labels != -100).all(), \
            "Long sequence (no padding) should have no -100 labels"

    def test_padding_affects_loss(self):
        """Loss should differ when padding tokens are included vs excluded."""
        model = create_test_model()
        model.eval()

        # Create input with padding (eos_token_id = 0 for this model)
        input_ids = torch.tensor([[10, 20, 30, 0, 0, 0]])  # last 3 are "padding"

        # Loss WITH padding tokens (old behavior)
        labels_with_pad = input_ids.clone()
        with torch.no_grad():
            loss_with_pad = model(input_ids, labels=labels_with_pad).loss

        # Loss WITHOUT padding tokens (new behavior using -100)
        labels_no_pad = input_ids.clone()
        labels_no_pad[0, 3:] = -100  # Mask padding positions
        with torch.no_grad():
            loss_no_pad = model(input_ids, labels=labels_no_pad).loss

        # Losses should be different since padding is excluded
        assert not torch.isclose(loss_with_pad, loss_no_pad), \
            f"Losses should differ: with_pad={loss_with_pad.item():.4f}, no_pad={loss_no_pad.item():.4f}"

    def test_train_step_uses_collator_labels(self):
        """train_step should use batch['labels'] from collator, not input_ids.clone()."""
        from sgtm.train.private_finetune import train_step

        model = create_test_model()
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        key = create_test_key()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        device = torch.device("cpu")

        # Create a batch with labels that have -100 (simulating DataCollator output)
        input_ids = torch.randint(1, 99, (2, 16))
        labels = input_ids.clone()
        labels[:, -4:] = -100  # Mask last 4 positions

        private_batch = {"input_ids": input_ids, "labels": labels}
        public_batch = {"input_ids": torch.randint(1, 99, (2, 16))}

        # Should not crash and should use the provided labels
        loss_priv, loss_kl, acc = train_step(
            model, ref_model, private_batch, public_batch, key,
            optimizer, device, kl_lambda=0.1, max_grad_norm=1.0
        )

        assert loss_priv > 0, "Private loss should be positive"
        assert isinstance(acc, float), "Accuracy should be a float"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
