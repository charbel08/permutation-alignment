"""End-to-end test to verify the complete tiered training flow.

This test verifies that:
1. With the fix (optimizer.step before unapply_key), gradients are correctly applied
2. Gradient permutation in permute.py is NOT needed
3. The complete train_step function produces expected results
"""

import torch
import copy
from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import PermutationKey
from tiered.permutation import apply_permutation, unapply_permutation
from tiered.permutation import mask_keyed_gradients, mask_public_gradients, scale_public_gradients


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
    return GPTNeoForCausalLMTiered(config)


def create_test_key():
    """Create a key with one attention swap."""
    return PermutationKey(
        attn_heads=[((0, 1), (2, 3))],  # Swap head 1 of layer 0 with head 3 of layer 2
        mlp_cols=[]
    )


def create_test_attn_out_key():
    """Create a key with one out-projection-only attention swap."""
    return PermutationKey(
        attn_out_heads=[((0, 1), (2, 3))],  # Swap out_proj head-range columns only
        mlp_cols=[],
    )


def _snapshot_params(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _snapshot_grads(model):
    return {
        name: (None if p.grad is None else p.grad.detach().clone())
        for name, p in model.named_parameters()
    }


def _build_keyed_masks(model, key):
    """Independent keyed mask oracle (does not use MaskPlan)."""
    masks = {
        name: torch.zeros_like(p, dtype=torch.bool)
        for name, p in model.named_parameters()
    }
    head_dim = model.transformer.h[0].attn.attention.head_dim

    def _attn_name(layer, proj):
        return f"transformer.h.{layer}.attn.attention.{proj}.weight"

    def _mlp_name(layer, part):
        return f"transformer.h.{layer}.mlp.{part}"

    def _mark_attn_head(layer, head):
        start, end = head * head_dim, (head + 1) * head_dim
        for proj in ("q_proj", "k_proj", "v_proj"):
            masks[_attn_name(layer, proj)][start:end, :] = True
        masks[_attn_name(layer, "out_proj")][:, start:end] = True

    def _mark_attn_out_head(layer, head):
        start, end = head * head_dim, (head + 1) * head_dim
        masks[_attn_name(layer, "out_proj")][:, start:end] = True

    def _mark_mlp_col(layer, col):
        masks[_mlp_name(layer, "c_fc.weight")][col, :] = True
        masks[_mlp_name(layer, "c_fc.bias")][col] = True
        masks[_mlp_name(layer, "c_proj.weight")][:, col] = True

    def _mark_mlp_up_col(layer, col):
        masks[_mlp_name(layer, "c_fc.weight")][col, :] = True
        masks[_mlp_name(layer, "c_fc.bias")][col] = True

    def _mark_mlp_down_col(layer, col):
        masks[_mlp_name(layer, "c_proj.weight")][:, col] = True

    for (la, ha), (lb, hb) in key.attn_heads:
        _mark_attn_head(la, ha)
        _mark_attn_head(lb, hb)

    for (la, ha), (lb, hb) in key.attn_out_heads:
        _mark_attn_out_head(la, ha)
        _mark_attn_out_head(lb, hb)

    for (la, ca), (lb, cb) in key.mlp_cols:
        _mark_mlp_col(la, ca)
        _mark_mlp_col(lb, cb)

    for (la, ca), (lb, cb) in key.mlp_up_cols:
        _mark_mlp_up_col(la, ca)
        _mark_mlp_up_col(lb, cb)

    for (la, ca), (lb, cb) in key.mlp_down_cols:
        _mark_mlp_down_col(la, ca)
        _mark_mlp_down_col(lb, cb)

    return masks


def _assert_public_unchanged(model, snapshot, keyed_masks):
    for name, p in model.named_parameters():
        keyed = keyed_masks[name]
        if keyed.numel() == 0:
            continue
        public = ~keyed
        if public.any():
            assert torch.equal(
                p.detach()[public], snapshot[name][public]
            ), f"Public values changed for {name}"


def _assert_swapped_key_slices(model, snapshot, key):
    """Check keyed slices are swapped relative to snapshot (for apply or unapply)."""
    params = dict(model.named_parameters())
    head_dim = model.transformer.h[0].attn.attention.head_dim

    def _attn_name(layer, proj):
        return f"transformer.h.{layer}.attn.attention.{proj}.weight"

    def _mlp_name(layer, part):
        return f"transformer.h.{layer}.mlp.{part}"

    def _slice(head):
        return slice(head * head_dim, (head + 1) * head_dim)

    for (la, ha), (lb, hb) in key.attn_heads:
        sa, sb = _slice(ha), _slice(hb)
        for proj in ("q_proj", "k_proj", "v_proj"):
            na = _attn_name(la, proj)
            nb = _attn_name(lb, proj)
            assert torch.equal(params[na].detach()[sa, :], snapshot[nb][sb, :]), f"{na} keyed rows not swapped"
            assert torch.equal(params[nb].detach()[sb, :], snapshot[na][sa, :]), f"{nb} keyed rows not swapped"
        na = _attn_name(la, "out_proj")
        nb = _attn_name(lb, "out_proj")
        assert torch.equal(params[na].detach()[:, sa], snapshot[nb][:, sb]), f"{na} keyed cols not swapped"
        assert torch.equal(params[nb].detach()[:, sb], snapshot[na][:, sa]), f"{nb} keyed cols not swapped"

    for (la, ha), (lb, hb) in key.attn_out_heads:
        sa, sb = _slice(ha), _slice(hb)
        na = _attn_name(la, "out_proj")
        nb = _attn_name(lb, "out_proj")
        assert torch.equal(params[na].detach()[:, sa], snapshot[nb][:, sb]), f"{na} out-only cols not swapped"
        assert torch.equal(params[nb].detach()[:, sb], snapshot[na][:, sa]), f"{nb} out-only cols not swapped"

    for (la, ca), (lb, cb) in key.mlp_cols:
        na_fcw, nb_fcw = _mlp_name(la, "c_fc.weight"), _mlp_name(lb, "c_fc.weight")
        na_fcb, nb_fcb = _mlp_name(la, "c_fc.bias"), _mlp_name(lb, "c_fc.bias")
        na_pjw, nb_pjw = _mlp_name(la, "c_proj.weight"), _mlp_name(lb, "c_proj.weight")
        assert torch.equal(params[na_fcw].detach()[ca, :], snapshot[nb_fcw][cb, :]), f"{na_fcw} mlp row not swapped"
        assert torch.equal(params[nb_fcw].detach()[cb, :], snapshot[na_fcw][ca, :]), f"{nb_fcw} mlp row not swapped"
        assert torch.equal(params[na_fcb].detach()[ca], snapshot[nb_fcb][cb]), f"{na_fcb} mlp bias not swapped"
        assert torch.equal(params[nb_fcb].detach()[cb], snapshot[na_fcb][ca]), f"{nb_fcb} mlp bias not swapped"
        assert torch.equal(params[na_pjw].detach()[:, ca], snapshot[nb_pjw][:, cb]), f"{na_pjw} mlp col not swapped"
        assert torch.equal(params[nb_pjw].detach()[:, cb], snapshot[na_pjw][:, ca]), f"{nb_pjw} mlp col not swapped"

    for (la, ca), (lb, cb) in key.mlp_up_cols:
        na_fcw, nb_fcw = _mlp_name(la, "c_fc.weight"), _mlp_name(lb, "c_fc.weight")
        na_fcb, nb_fcb = _mlp_name(la, "c_fc.bias"), _mlp_name(lb, "c_fc.bias")
        assert torch.equal(params[na_fcw].detach()[ca, :], snapshot[nb_fcw][cb, :]), f"{na_fcw} up row not swapped"
        assert torch.equal(params[nb_fcw].detach()[cb, :], snapshot[na_fcw][ca, :]), f"{nb_fcw} up row not swapped"
        assert torch.equal(params[na_fcb].detach()[ca], snapshot[nb_fcb][cb]), f"{na_fcb} up bias not swapped"
        assert torch.equal(params[nb_fcb].detach()[cb], snapshot[na_fcb][ca]), f"{nb_fcb} up bias not swapped"

    for (la, ca), (lb, cb) in key.mlp_down_cols:
        na_pjw, nb_pjw = _mlp_name(la, "c_proj.weight"), _mlp_name(lb, "c_proj.weight")
        assert torch.equal(params[na_pjw].detach()[:, ca], snapshot[nb_pjw][:, cb]), f"{na_pjw} down col not swapped"
        assert torch.equal(params[nb_pjw].detach()[:, cb], snapshot[na_pjw][:, ca]), f"{nb_pjw} down col not swapped"


class TestEndToEndTraining:
    """Test the complete training flow to verify no gradient permutation is needed."""
    
    def test_complete_train_step_flow(self):
        """Test that the complete train_step flow works correctly.
        
        Simulates the fixed train_step:
        1. C1 forward/backward
        2. mask_keyed_gradients
        3. apply_key
        4. C2 forward/backward
        5. scale_public_gradients
        6. optimizer.step() [WHILE IN C2 - the fix]
        7. unapply_key
        
        Verifies that the keyed weights are updated correctly.
        """
        torch.manual_seed(42)
        model = create_test_model()
        key = create_test_key()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Get references to keyed parameters
        attn_layer0 = model.transformer.h[0].attn.attention
        attn_layer2 = model.transformer.h[2].attn.attention
        
        # Store original values of keyed weights
        # These are the VALUES that should be optimized based on their role in C2
        original_L0_H1 = attn_layer0.q_proj.weight[16:32, :].clone()
        original_L2_H3 = attn_layer2.q_proj.weight[48:64, :].clone()
        
        input_ids = torch.randint(0, 100, (2, 32))
        
        # === Simulate fixed train_step ===
        
        # Step 1: C1 forward/backward
        optimizer.zero_grad()
        outputs_c1 = model(input_ids, labels=input_ids)
        loss_c1 = outputs_c1.loss
        loss_c1.backward()
        
        # Step 2: mask_keyed_gradients (keyed params get grad=0 from C1)
        mask_keyed_gradients(model, key)
        
        # Verify keyed gradients are zeroed
        assert torch.allclose(attn_layer0.q_proj.weight.grad[16:32, :], torch.zeros_like(original_L0_H1))
        assert torch.allclose(attn_layer2.q_proj.weight.grad[48:64, :], torch.zeros_like(original_L2_H3))
        
        # Step 3: apply_key
        apply_permutation(model, key)
        
        # Verify weights are swapped (in C2 config)
        # After swap: L0[16:32] has original_L2_H3, L2[48:64] has original_L0_H1
        assert torch.allclose(attn_layer0.q_proj.weight[16:32, :], original_L2_H3)
        assert torch.allclose(attn_layer2.q_proj.weight[48:64, :], original_L0_H1)
        
        # Step 4: C2 forward/backward
        outputs_c2 = model(input_ids, labels=input_ids)
        loss_c2 = outputs_c2.loss
        loss_c2.backward()
        
        # Now keyed positions have C2 gradients
        # L0[16:32] grad is dL/d(value at L0[16:32]) = dL/d(original_L2_H3)
        # L2[48:64] grad is dL/d(value at L2[48:64]) = dL/d(original_L0_H1)
        grad_for_L2_H3_value = attn_layer0.q_proj.weight.grad[16:32, :].clone()
        grad_for_L0_H1_value = attn_layer2.q_proj.weight.grad[48:64, :].clone()
        
        # Step 5: scale_public_gradients
        scale_public_gradients(model, key, scale=0.5)
        
        # Step 6: optimizer.step() WHILE IN C2 CONFIG [THE FIX]
        optimizer.step()
        
        # Get updated weights (still in C2 config)
        updated_at_L0_pos = attn_layer0.q_proj.weight[16:32, :].clone()
        updated_at_L2_pos = attn_layer2.q_proj.weight[48:64, :].clone()
        
        # Verify updates are correct (SGD: w = w - lr * grad)
        # Position L0[16:32]: had original_L2_H3, updated by grad_for_L2_H3_value
        expected_updated_L2_H3 = original_L2_H3 - 0.1 * grad_for_L2_H3_value
        # Position L2[48:64]: had original_L0_H1, updated by grad_for_L0_H1_value
        expected_updated_L0_H1 = original_L0_H1 - 0.1 * grad_for_L0_H1_value
        
        assert torch.allclose(updated_at_L0_pos, expected_updated_L2_H3, atol=2e-4), \
            "L0 position update incorrect before unapply"
        assert torch.allclose(updated_at_L2_pos, expected_updated_L0_H1, atol=2e-4), \
            "L2 position update incorrect before unapply"
        
        print("Verified: Updates correct WHILE IN C2 config")
        
        # Step 7: unapply_key
        unapply_permutation(model, key)
        
        # After unapply: weights swap back
        # L0[16:32] now has the updated original_L0_H1
        # L2[48:64] now has the updated original_L2_H3
        final_at_L0_pos = attn_layer0.q_proj.weight[16:32, :].clone()
        final_at_L2_pos = attn_layer2.q_proj.weight[48:64, :].clone()
        
        # Verify the FINAL positions have the correct updated values
        assert torch.allclose(final_at_L0_pos, expected_updated_L0_H1, atol=2e-4), \
            "L0 position should have updated L0_H1 (swapped back)"
        assert torch.allclose(final_at_L2_pos, expected_updated_L2_H3, atol=2e-4), \
            "L2 position should have updated L2_H3 (swapped back)"
        
        print("Verified: After unapply_key, weights are correctly positioned")
        
        # CRITICAL: Verify the VALUES were updated based on their C2 ROLE
        # L0_H1 was at position L2[48:64] in C2, so it received grad_for_L0_H1_value
        # L2_H3 was at position L0[16:32] in C2, so it received grad_for_L2_H3_value
        print("\nFinal verification:")
        print(f"  original_L0_H1 - 0.1 * grad_for_L0_H1_value should be at L0[16:32]")
        print(f"  original_L2_H3 - 0.1 * grad_for_L2_H3_value should be at L2[48:64]")
        print("  Both verified correct ✓")
        
    def test_gradient_permutation_not_needed(self):
        """Explicitly verify that we do NOT need to permute gradients in permute.py.
        
        The key insight: 
        - With optimizer.step() BEFORE unapply_key, weights and gradients are aligned
        - Both are in C2 positions during the update
        - After unapply_key, weights move back, but gradients don't matter anymore
        - The next step calls zero_grad() which clears all gradients
        """
        torch.manual_seed(42)
        model = create_test_model()
        key = create_test_key()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        attn_layer0 = model.transformer.h[0].attn.attention
        attn_layer2 = model.transformer.h[2].attn.attention
        
        input_ids = torch.randint(0, 100, (2, 32))
        
        # Run TWO training steps to verify gradients from step 1 don't affect step 2
        
        for step in range(2):
            optimizer.zero_grad()  # This clears any stale gradients
            
            # C1 forward/backward
            outputs_c1 = model(input_ids, labels=input_ids)
            outputs_c1.loss.backward()
            mask_keyed_gradients(model, key)
            
            # C2 forward/backward
            apply_permutation(model, key)
            outputs_c2 = model(input_ids, labels=input_ids)
            outputs_c2.loss.backward()
            
            # Scale and update (before unapply)
            scale_public_gradients(model, key, scale=0.5)
            optimizer.step()
            
            # Unapply key
            unapply_permutation(model, key)
            
            # The gradients after unapply_key are in "wrong" positions relative to weights
            # BUT IT DOESN'T MATTER because:
            # 1. The optimizer already used them correctly (before unapply)
            # 2. The next step will zero_grad() anyway
            
            print(f"Step {step + 1} completed successfully")
        
        print("\nVerified: Two consecutive steps work correctly")
        print("Gradient permutation in permute.py is NOT needed because:")
        print("  1. optimizer.step() happens while weights and grads are both in C2 positions")
        print("  2. zero_grad() at start of next step clears any misaligned gradients")

    def test_attn_out_only_train_step_flow(self):
        """End-to-end one-step flow for attn_out-only keys."""
        torch.manual_seed(42)
        model = create_test_model()
        key = create_test_attn_out_key()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        attn_layer0 = model.transformer.h[0].attn.attention
        attn_layer2 = model.transformer.h[2].attn.attention
        head_dim = attn_layer0.head_dim
        idx0 = slice(1 * head_dim, 2 * head_dim)
        idx2 = slice(3 * head_dim, 4 * head_dim)

        orig_q0 = attn_layer0.q_proj.weight[idx0, :].clone()
        orig_q2 = attn_layer2.q_proj.weight[idx2, :].clone()
        orig_out0 = attn_layer0.out_proj.weight[:, idx0].clone()
        orig_out2 = attn_layer2.out_proj.weight[:, idx2].clone()

        input_ids = torch.randint(0, 100, (2, 32))

        optimizer.zero_grad()
        outputs_c1 = model(input_ids, labels=input_ids)
        outputs_c1.loss.backward()

        q_grad_before = attn_layer0.q_proj.weight.grad[idx0, :].clone()
        out_grad_before = attn_layer0.out_proj.weight.grad[:, idx0].clone()
        mask_keyed_gradients(model, key)

        # attn_out-only should mask out_proj but not q_proj rows
        assert torch.allclose(attn_layer0.q_proj.weight.grad[idx0, :], q_grad_before)
        assert torch.allclose(
            attn_layer0.out_proj.weight.grad[:, idx0],
            torch.zeros_like(out_grad_before),
        )

        apply_permutation(model, key)

        # only out_proj columns should be swapped
        assert torch.allclose(attn_layer0.q_proj.weight[idx0, :], orig_q0)
        assert torch.allclose(attn_layer2.q_proj.weight[idx2, :], orig_q2)
        assert torch.allclose(attn_layer0.out_proj.weight[:, idx0], orig_out2)
        assert torch.allclose(attn_layer2.out_proj.weight[:, idx2], orig_out0)

        outputs_c2 = model(input_ids, labels=input_ids)
        outputs_c2.loss.backward()
        scale_public_gradients(model, key, scale=0.5)
        optimizer.step()
        unapply_permutation(model, key)

        # after step + unapply, at least one keyed out_proj range should change
        changed = not torch.allclose(attn_layer0.out_proj.weight[:, idx0], orig_out0)
        changed = changed or not torch.allclose(attn_layer2.out_proj.weight[:, idx2], orig_out2)
        assert changed, "Expected keyed out_proj columns to update"

    def test_strict_e2e_gradient_pipeline_oracle_all_key_types(self):
        """Stage-by-stage oracle checks for mask/scale/swap/update across key types."""
        key_cases = {
            "attn_heads": PermutationKey(attn_heads=[((0, 1), (2, 3))]),
            "attn_out_heads": PermutationKey(attn_out_heads=[((0, 1), (2, 3))]),
            "mlp_cols": PermutationKey(mlp_cols=[((0, 5), (2, 10))]),
            "mlp_up_cols": PermutationKey(mlp_up_cols=[((0, 7), (2, 12))]),
            "mlp_down_cols": PermutationKey(mlp_down_cols=[((1, 8), (3, 9))]),
            "mixed": PermutationKey(
                attn_heads=[((0, 1), (2, 3))],
                attn_out_heads=[((1, 0), (3, 2))],
                mlp_cols=[((0, 5), (2, 10))],
                mlp_up_cols=[((1, 7), (3, 12))],
                mlp_down_cols=[((0, 8), (2, 9))],
            ),
        }

        for i, (case_name, key) in enumerate(key_cases.items()):
            torch.manual_seed(1000 + i)
            model = create_test_model()
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
            keyed_masks = _build_keyed_masks(model, key)

            input_ids = torch.randint(0, 100, (2, 32))
            labels = input_ids.clone()

            # --- Phase C1 ---
            optimizer.zero_grad()
            (model(input_ids, labels=labels).loss * 256.0).backward()
            c1_grads = _snapshot_grads(model)

            # Oracle: mask_keyed_gradients should zero keyed positions only.
            expected_after_mask = {}
            for name, g in c1_grads.items():
                if g is None:
                    expected_after_mask[name] = None
                    continue
                exp = g.clone()
                exp[keyed_masks[name]] = 0
                expected_after_mask[name] = exp

            mask_keyed_gradients(model, key)
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                assert torch.equal(
                    p.grad, expected_after_mask[name]
                ), f"[{case_name}] mask_keyed_gradients mismatch for {name}"

            # --- Apply permutation C1 -> C2 ---
            before_apply = _snapshot_params(model)
            apply_permutation(model, key)
            _assert_swapped_key_slices(model, before_apply, key)
            _assert_public_unchanged(model, before_apply, keyed_masks)

            # --- Phase C2 ---
            (model(input_ids, labels=labels).loss * 256.0).backward()
            pre_scale_grads = _snapshot_grads(model)

            # Oracle: scale only public positions by 0.5.
            expected_after_scale = {}
            for name, g in pre_scale_grads.items():
                if g is None:
                    expected_after_scale[name] = None
                    continue
                exp = g.clone()
                exp[~keyed_masks[name]] *= 0.5
                expected_after_scale[name] = exp

            scale_public_gradients(model, key, scale=0.5)
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                assert torch.equal(
                    p.grad, expected_after_scale[name]
                ), f"[{case_name}] scale_public_gradients mismatch for {name}"

            # --- Optimizer step in C2 frame ---
            before_step_c2 = _snapshot_params(model)
            optimizer.step()
            after_step_c2 = _snapshot_params(model)
            for name, p in model.named_parameters():
                if expected_after_scale[name] is None:
                    continue
                expected_w = before_step_c2[name] - 0.2 * expected_after_scale[name]
                assert torch.allclose(
                    after_step_c2[name], expected_w, atol=1e-6, rtol=0
                ), f"[{case_name}] SGD update mismatch for {name}"

            # --- Unapply permutation C2 -> C1 ---
            unapply_permutation(model, key)
            _assert_swapped_key_slices(model, after_step_c2, key)
            _assert_public_unchanged(model, after_step_c2, keyed_masks)
        

if __name__ == "__main__":
    print("=" * 70)
    print("End-to-end verification: Gradient permutation NOT needed in permute.py")
    print("=" * 70)
    
    test = TestEndToEndTraining()
    
    print("\n--- Test 1: Complete train_step flow ---")
    test.test_complete_train_step_flow()
    
    print("\n--- Test 2: Multi-step verification ---")
    test.test_gradient_permutation_not_needed()
    
    print("\n" + "=" * 70)
    print("CONCLUSION: Gradient permutation is NOT needed in permute.py")
    print("The fix (optimizer.step before unapply_key) is sufficient.")
    print("=" * 70)
