"""Key-based gradient masking for tiered alignment pretraining.

This module implements gradient masking based on permutation keys, enabling
asymmetric training where the keyed subset S (parameters involved in swaps)
receives gradients only from the keyed architecture C_2.
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgtm.permutation.key import PermutationKey


def _get_attention_module(model, layer_idx: int):
    """Get the attention module for a specific layer."""
    block = model.transformer.h[layer_idx]
    if hasattr(block, "attn"):
        attn = block.attn
        if hasattr(attn, "attention"):
            return attn.attention
        return attn
    raise AttributeError(f"Could not find attention module in layer {layer_idx}")


def _get_mlp_module(model, layer_idx: int):
    """Get the MLP module for a specific layer."""
    block = model.transformer.h[layer_idx]
    if hasattr(block, "mlp"):
        return block.mlp
    raise AttributeError(f"Could not find MLP module in layer {layer_idx}")


def mask_keyed_gradients(model, key: "PermutationKey") -> None:
    """Zero gradients for keyed parameters (those involved in swaps).
    
    Use this when training the public architecture (C_1) during joint pretraining
    to prevent C_1's loss from updating the keyed subset S.
    
    This implements equation (3) from the paper: during pretraining, θ_S gets
    gradients only from C_2, not from C_1.
    
    Args:
        model: The GPT model.
        key: The permutation key specifying which parameters are keyed.
    """
    # Zero gradients for attention heads involved in swaps
    for swap in key.attention_swaps:
        attn_a = _get_attention_module(model, swap.layer_a)
        attn_b = _get_attention_module(model, swap.layer_b)
        
        head_dim = attn_a.head_dim
        start_a = swap.head_a * head_dim
        end_a = (swap.head_a + 1) * head_dim
        start_b = swap.head_b * head_dim
        end_b = (swap.head_b + 1) * head_dim
        
        # Zero Q projection gradients (rows)
        if attn_a.q_proj.weight.grad is not None:
            attn_a.q_proj.weight.grad[start_a:end_a, :] = 0
        if attn_b.q_proj.weight.grad is not None:
            attn_b.q_proj.weight.grad[start_b:end_b, :] = 0
        
        # Zero K projection gradients (rows)
        if attn_a.k_proj.weight.grad is not None:
            attn_a.k_proj.weight.grad[start_a:end_a, :] = 0
        if attn_b.k_proj.weight.grad is not None:
            attn_b.k_proj.weight.grad[start_b:end_b, :] = 0
        
        # Zero V projection gradients (rows)
        if attn_a.v_proj.weight.grad is not None:
            attn_a.v_proj.weight.grad[start_a:end_a, :] = 0
        if attn_b.v_proj.weight.grad is not None:
            attn_b.v_proj.weight.grad[start_b:end_b, :] = 0
        
        # Zero output projection gradients (columns)
        if attn_a.out_proj.weight.grad is not None:
            attn_a.out_proj.weight.grad[:, start_a:end_a] = 0
        if attn_b.out_proj.weight.grad is not None:
            attn_b.out_proj.weight.grad[:, start_b:end_b] = 0
    
    # Zero gradients for MLP columns involved in swaps
    for swap in key.mlp_swaps:
        mlp_a = _get_mlp_module(model, swap.layer_a)
        mlp_b = _get_mlp_module(model, swap.layer_b)
        
        # Zero c_fc gradients (rows)
        if mlp_a.c_fc.weight.grad is not None:
            mlp_a.c_fc.weight.grad[swap.col_a, :] = 0
        if mlp_b.c_fc.weight.grad is not None:
            mlp_b.c_fc.weight.grad[swap.col_b, :] = 0
        
        # Zero c_fc bias gradients
        if mlp_a.c_fc.bias is not None and mlp_a.c_fc.bias.grad is not None:
            mlp_a.c_fc.bias.grad[swap.col_a] = 0
        if mlp_b.c_fc.bias is not None and mlp_b.c_fc.bias.grad is not None:
            mlp_b.c_fc.bias.grad[swap.col_b] = 0
        
        # Zero c_proj gradients (columns)
        if mlp_a.c_proj.weight.grad is not None:
            mlp_a.c_proj.weight.grad[:, swap.col_a] = 0
        if mlp_b.c_proj.weight.grad is not None:
            mlp_b.c_proj.weight.grad[:, swap.col_b] = 0


def mask_public_gradients(model, key: "PermutationKey") -> None:
    """Zero gradients for public parameters (those NOT involved in swaps).
    
    Use this when you want to update only the keyed subset S.
    
    Args:
        model: The GPT model.
        key: The permutation key specifying which parameters are keyed.
    """
    # Build sets of (layer, head) and (layer, col) that are keyed
    keyed_attention = set()
    for swap in key.attention_swaps:
        keyed_attention.add((swap.layer_a, swap.head_a))
        keyed_attention.add((swap.layer_b, swap.head_b))
    
    keyed_mlp = set()
    for swap in key.mlp_swaps:
        keyed_mlp.add((swap.layer_a, swap.col_a))
        keyed_mlp.add((swap.layer_b, swap.col_b))
    
    # Zero gradients for all parameters NOT in keyed sets
    for layer_idx, block in enumerate(model.transformer.h):
        # Handle attention
        try:
            attn = _get_attention_module(model, layer_idx)
            head_dim = attn.head_dim
            num_heads = attn.num_heads
            
            for head_idx in range(num_heads):
                if (layer_idx, head_idx) not in keyed_attention:
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
        except AttributeError:
            pass
        
        # Handle MLP
        try:
            mlp = _get_mlp_module(model, layer_idx)
            mlp_dim = mlp.c_fc.weight.shape[0]
            
            for col_idx in range(mlp_dim):
                if (layer_idx, col_idx) not in keyed_mlp:
                    if mlp.c_fc.weight.grad is not None:
                        mlp.c_fc.weight.grad[col_idx, :] = 0
                    if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
                        mlp.c_fc.bias.grad[col_idx] = 0
                    if mlp.c_proj.weight.grad is not None:
                        mlp.c_proj.weight.grad[:, col_idx] = 0
        except AttributeError:
            pass
    
    # Also zero gradients for embeddings, lm_head, and layer norms (all public)
    if hasattr(model, 'transformer'):
        if model.transformer.wte.weight.grad is not None:
            model.transformer.wte.weight.grad.zero_()
        if model.transformer.wpe.weight.grad is not None:
            model.transformer.wpe.weight.grad.zero_()
        if model.transformer.ln_f.weight.grad is not None:
            model.transformer.ln_f.weight.grad.zero_()
        if model.transformer.ln_f.bias.grad is not None:
            model.transformer.ln_f.bias.grad.zero_()
    
    if hasattr(model, 'lm_head'):
        if model.lm_head.weight.grad is not None:
            model.lm_head.weight.grad.zero_()
        if model.lm_head.bias is not None and model.lm_head.bias.grad is not None:
            model.lm_head.bias.grad.zero_()
