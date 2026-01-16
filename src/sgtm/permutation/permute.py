"""Apply and reverse permutations to model weights."""

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


def _swap_attention_heads(model, layer_a: int, head_a: int, layer_b: int, head_b: int):
    """Swap attention head weights between two layers."""
    attn_a = _get_attention_module(model, layer_a)
    attn_b = _get_attention_module(model, layer_b)
    
    head_dim = attn_a.head_dim
    start_a = head_a * head_dim
    end_a = (head_a + 1) * head_dim
    start_b = head_b * head_dim
    end_b = (head_b + 1) * head_dim
    
    with torch.no_grad():
        # Swap Q projection
        q_a = attn_a.q_proj.weight[start_a:end_a, :].clone()
        q_b = attn_b.q_proj.weight[start_b:end_b, :].clone()
        attn_a.q_proj.weight[start_a:end_a, :] = q_b
        attn_b.q_proj.weight[start_b:end_b, :] = q_a
        
        # Swap K projection
        k_a = attn_a.k_proj.weight[start_a:end_a, :].clone()
        k_b = attn_b.k_proj.weight[start_b:end_b, :].clone()
        attn_a.k_proj.weight[start_a:end_a, :] = k_b
        attn_b.k_proj.weight[start_b:end_b, :] = k_a
        
        # Swap V projection
        v_a = attn_a.v_proj.weight[start_a:end_a, :].clone()
        v_b = attn_b.v_proj.weight[start_b:end_b, :].clone()
        attn_a.v_proj.weight[start_a:end_a, :] = v_b
        attn_b.v_proj.weight[start_b:end_b, :] = v_a
        
        # Swap output projection (columns)
        o_a = attn_a.out_proj.weight[:, start_a:end_a].clone()
        o_b = attn_b.out_proj.weight[:, start_b:end_b].clone()
        attn_a.out_proj.weight[:, start_a:end_a] = o_b
        attn_b.out_proj.weight[:, start_b:end_b] = o_a


def _swap_mlp_columns(model, layer_a: int, col_a: int, layer_b: int, col_b: int):
    """Swap MLP column weights between two layers."""
    mlp_a = _get_mlp_module(model, layer_a)
    mlp_b = _get_mlp_module(model, layer_b)
    
    with torch.no_grad():
        # Swap c_fc weights (rows)
        fc_a = mlp_a.c_fc.weight[col_a, :].clone()
        fc_b = mlp_b.c_fc.weight[col_b, :].clone()
        mlp_a.c_fc.weight[col_a, :] = fc_b
        mlp_b.c_fc.weight[col_b, :] = fc_a
        
        # Swap c_fc bias
        if mlp_a.c_fc.bias is not None and mlp_b.c_fc.bias is not None:
            bias_a = mlp_a.c_fc.bias[col_a].clone()
            bias_b = mlp_b.c_fc.bias[col_b].clone()
            mlp_a.c_fc.bias[col_a] = bias_b
            mlp_b.c_fc.bias[col_b] = bias_a
        
        # Swap c_proj weights (columns)
        proj_a = mlp_a.c_proj.weight[:, col_a].clone()
        proj_b = mlp_b.c_proj.weight[:, col_b].clone()
        mlp_a.c_proj.weight[:, col_a] = proj_b
        mlp_b.c_proj.weight[:, col_b] = proj_a


def apply_permutation(model, key: "PermutationKey") -> None:
    """Apply a permutation key to the model's weights.
    
    Args:
        model: The GPT model.
        key: The permutation key specifying swaps.
    """
    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        _swap_attention_heads(model, layer_a, head_a, layer_b, head_b)
    
    for swap in key.mlp_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        _swap_mlp_columns(model, layer_a, col_a, layer_b, col_b)


def unapply_permutation(model, key: "PermutationKey") -> None:
    """Reverse a permutation (same as apply since swaps are self-inverse)."""
    apply_permutation(model, key)
