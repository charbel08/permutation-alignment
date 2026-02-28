"""Gradient masking for tiered alignment pretraining."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgtm.permutation.key import PermutationKey

from sgtm.permutation.utils import _get_attention_module, _get_mlp_module


def mask_keyed_gradients(model, key: "PermutationKey") -> None:
    """Zero gradients for keyed parameters (those involved in swaps).
    
    Use this after backward through public architecture (C1) to prevent
    C1's loss from updating the keyed subset S.
    
    Args:
        model: The GPT model.
        key: The permutation key specifying keyed parameters.
    """
    # Zero gradients for attention heads in swaps
    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        
        for layer_idx, head_idx in [(layer_a, head_a), (layer_b, head_b)]:
            attn = _get_attention_module(model, layer_idx)
            head_dim = attn.head_dim
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            
            if attn.q_proj.weight.grad is not None:
                attn.q_proj.weight.grad[start:end, :] = 0
            if attn.k_proj.weight.grad is not None:
                attn.k_proj.weight.grad[start:end, :] = 0
            if attn.v_proj.weight.grad is not None:
                attn.v_proj.weight.grad[start:end, :] = 0
            if attn.out_proj.weight.grad is not None:
                attn.out_proj.weight.grad[:, start:end] = 0
    
    # Zero gradients for MLP columns in swaps
    for swap in key.mlp_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        
        for layer_idx, col_idx in [(layer_a, col_a), (layer_b, col_b)]:
            mlp = _get_mlp_module(model, layer_idx)
            
            if mlp.c_fc.weight.grad is not None:
                mlp.c_fc.weight.grad[col_idx, :] = 0
            if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
                mlp.c_fc.bias.grad[col_idx] = 0
            if mlp.c_proj.weight.grad is not None:
                mlp.c_proj.weight.grad[:, col_idx] = 0


def mask_public_gradients(model, key: "PermutationKey") -> None:
    """Zero gradients for public parameters (those NOT involved in swaps).
    
    Args:
        model: The GPT model.
        key: The permutation key specifying keyed parameters.
    """
    # Build sets of keyed heads and columns
    keyed_heads = set()
    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        keyed_heads.add((layer_a, head_a))
        keyed_heads.add((layer_b, head_b))
    
    keyed_cols = set()
    for swap in key.mlp_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        keyed_cols.add((layer_a, col_a))
        keyed_cols.add((layer_b, col_b))
    
    # Zero public attention gradients
    for layer_idx, block in enumerate(model.transformer.h):
        try:
            attn = _get_attention_module(model, layer_idx)
            head_dim = attn.head_dim
            num_heads = attn.num_heads
            
            for head_idx in range(num_heads):
                if (layer_idx, head_idx) not in keyed_heads:
                    start = head_idx * head_dim
                    end = (head_idx + 1) * head_dim
                    
                    if attn.q_proj.weight.grad is not None:
                        attn.q_proj.weight.grad[start:end, :] = 0
                    if attn.k_proj.weight.grad is not None:
                        attn.k_proj.weight.grad[start:end, :] = 0
                    if attn.v_proj.weight.grad is not None:
                        attn.v_proj.weight.grad[start:end, :] = 0
                    if attn.out_proj.weight.grad is not None:
                        attn.out_proj.weight.grad[:, start:end] = 0
            
            # Attention biases are always public (not split per-head)
            if hasattr(attn.out_proj, 'bias') and attn.out_proj.bias is not None:
                if attn.out_proj.bias.grad is not None:
                    attn.out_proj.bias.grad.zero_()
        except AttributeError:
            pass
        
        # Zero public MLP gradients (batched)
        try:
            mlp = _get_mlp_module(model, layer_idx)
            mlp_dim = mlp.c_fc.weight.shape[0]
            
            # Build list of public (non-keyed) column indices for this layer
            keyed_cols_this_layer = {col for (l, col) in keyed_cols if l == layer_idx}
            public_cols = [c for c in range(mlp_dim) if c not in keyed_cols_this_layer]
            
            if public_cols:
                if mlp.c_fc.weight.grad is not None:
                    mlp.c_fc.weight.grad[public_cols, :] = 0
                if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
                    mlp.c_fc.bias.grad[public_cols] = 0
                if mlp.c_proj.weight.grad is not None:
                    mlp.c_proj.weight.grad[:, public_cols] = 0
            
            # MLP output bias is always public
            if hasattr(mlp.c_proj, 'bias') and mlp.c_proj.bias is not None:
                if mlp.c_proj.bias.grad is not None:
                    mlp.c_proj.bias.grad.zero_()
        except AttributeError:
            pass
    
    # Zero embeddings and layer norms (always public)
    if hasattr(model, 'transformer'):
        if model.transformer.wte.weight.grad is not None:
            model.transformer.wte.weight.grad.zero_()
        if model.transformer.wpe.weight.grad is not None:
            model.transformer.wpe.weight.grad.zero_()
        if model.transformer.ln_f.weight.grad is not None:
            model.transformer.ln_f.weight.grad.zero_()
        if model.transformer.ln_f.bias.grad is not None:
            model.transformer.ln_f.bias.grad.zero_()
        
        # Zero layer norms within blocks (always public)
        for block in model.transformer.h:
            for ln in [block.ln_1, block.ln_2]:
                if ln.weight.grad is not None:
                    ln.weight.grad.zero_()
                if ln.bias.grad is not None:
                    ln.bias.grad.zero_()
    
    if hasattr(model, 'lm_head'):
        if model.lm_head.weight.grad is not None:
            model.lm_head.weight.grad.zero_()
        if model.lm_head.bias is not None and model.lm_head.bias.grad is not None:
            model.lm_head.bias.grad.zero_()
