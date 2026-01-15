"""Core permutation logic for tiered alignment.

This module provides functions to apply and reverse parameter permutations
on a model based on a permutation key.
"""

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgtm.permutation.key import PermutationKey


def _get_attention_module(model: nn.Module, layer_idx: int):
    """Get the attention module for a specific layer.
    
    Args:
        model: The GPT model.
        layer_idx: Index of the layer.
        
    Returns:
        The attention module for that layer.
    """
    block = model.transformer.h[layer_idx]
    # Handle both wrapped and unwrapped attention
    if hasattr(block, "attn"):
        attn = block.attn
        if hasattr(attn, "attention"):
            return attn.attention
        return attn
    raise AttributeError(f"Could not find attention module in layer {layer_idx}")


def _get_mlp_module(model: nn.Module, layer_idx: int):
    """Get the MLP module for a specific layer.
    
    Args:
        model: The GPT model.
        layer_idx: Index of the layer.
        
    Returns:
        The MLP module for that layer.
    """
    block = model.transformer.h[layer_idx]
    if hasattr(block, "mlp"):
        return block.mlp
    raise AttributeError(f"Could not find MLP module in layer {layer_idx}")


def _swap_attention_heads(
    model: nn.Module,
    layer_a: int,
    head_a: int,
    layer_b: int,
    head_b: int,
) -> None:
    """Swap attention head parameters between two layers.
    
    This swaps:
    - Rows in Q, K, V projection weights corresponding to the heads
    - Columns in output projection weights corresponding to the heads
    
    Args:
        model: The GPT model.
        layer_a: First layer index.
        head_a: Head index in first layer.
        layer_b: Second layer index.
        head_b: Head index in second layer.
    """
    attn_a = _get_attention_module(model, layer_a)
    attn_b = _get_attention_module(model, layer_b)
    
    head_dim = attn_a.head_dim
    
    # Calculate slice indices for each head
    start_a = head_a * head_dim
    end_a = (head_a + 1) * head_dim
    start_b = head_b * head_dim
    end_b = (head_b + 1) * head_dim
    
    with torch.no_grad():
        # Swap Q projection weights (rows)
        q_a_slice = attn_a.q_proj.weight[start_a:end_a, :].clone()
        q_b_slice = attn_b.q_proj.weight[start_b:end_b, :].clone()
        attn_a.q_proj.weight[start_a:end_a, :] = q_b_slice
        attn_b.q_proj.weight[start_b:end_b, :] = q_a_slice
        
        # Swap K projection weights (rows)
        k_a_slice = attn_a.k_proj.weight[start_a:end_a, :].clone()
        k_b_slice = attn_b.k_proj.weight[start_b:end_b, :].clone()
        attn_a.k_proj.weight[start_a:end_a, :] = k_b_slice
        attn_b.k_proj.weight[start_b:end_b, :] = k_a_slice
        
        # Swap V projection weights (rows)
        v_a_slice = attn_a.v_proj.weight[start_a:end_a, :].clone()
        v_b_slice = attn_b.v_proj.weight[start_b:end_b, :].clone()
        attn_a.v_proj.weight[start_a:end_a, :] = v_b_slice
        attn_b.v_proj.weight[start_b:end_b, :] = v_a_slice
        
        # Swap output projection weights (columns)
        out_a_slice = attn_a.out_proj.weight[:, start_a:end_a].clone()
        out_b_slice = attn_b.out_proj.weight[:, start_b:end_b].clone()
        attn_a.out_proj.weight[:, start_a:end_a] = out_b_slice
        attn_b.out_proj.weight[:, start_b:end_b] = out_a_slice


def _swap_mlp_columns(
    model: nn.Module,
    layer_a: int,
    col_a: int,
    layer_b: int,
    col_b: int,
) -> None:
    """Swap MLP column parameters between two layers.
    
    This swaps:
    - Row in c_fc weight (and bias element) corresponding to the column
    - Column in c_proj weight corresponding to the column
    
    Args:
        model: The GPT model.
        layer_a: First layer index.
        col_a: Column index in first layer.
        layer_b: Second layer index.
        col_b: Column index in second layer.
    """
    mlp_a = _get_mlp_module(model, layer_a)
    mlp_b = _get_mlp_module(model, layer_b)
    
    with torch.no_grad():
        # Swap c_fc weights (rows)
        fc_a_slice = mlp_a.c_fc.weight[col_a, :].clone()
        fc_b_slice = mlp_b.c_fc.weight[col_b, :].clone()
        mlp_a.c_fc.weight[col_a, :] = fc_b_slice
        mlp_b.c_fc.weight[col_b, :] = fc_a_slice
        
        # Swap c_fc biases
        if mlp_a.c_fc.bias is not None and mlp_b.c_fc.bias is not None:
            bias_a = mlp_a.c_fc.bias[col_a].clone()
            bias_b = mlp_b.c_fc.bias[col_b].clone()
            mlp_a.c_fc.bias[col_a] = bias_b
            mlp_b.c_fc.bias[col_b] = bias_a
        
        # Swap c_proj weights (columns)
        proj_a_slice = mlp_a.c_proj.weight[:, col_a].clone()
        proj_b_slice = mlp_b.c_proj.weight[:, col_b].clone()
        mlp_a.c_proj.weight[:, col_a] = proj_b_slice
        mlp_b.c_proj.weight[:, col_b] = proj_a_slice


def apply_permutation(model: nn.Module, key: "PermutationKey") -> None:
    """Apply a permutation key to a model's weights.
    
    This transforms the model from the public configuration to the keyed
    configuration by swapping attention heads and MLP columns as specified
    by the key.
    
    Note: This modifies the model in-place.
    
    Args:
        model: The GPT model to permute.
        key: The permutation key specifying the swaps.
    """
    # Apply attention head swaps
    for swap in key.attention_swaps:
        _swap_attention_heads(
            model,
            layer_a=swap.layer_a,
            head_a=swap.head_a,
            layer_b=swap.layer_b,
            head_b=swap.head_b,
        )
    
    # Apply MLP column swaps
    for swap in key.mlp_swaps:
        _swap_mlp_columns(
            model,
            layer_a=swap.layer_a,
            col_a=swap.col_a,
            layer_b=swap.layer_b,
            col_b=swap.col_b,
        )


def unapply_permutation(model: nn.Module, key: "PermutationKey") -> None:
    """Reverse a permutation key on a model's weights.
    
    This transforms the model from the keyed configuration back to the
    public configuration. Since all swaps are self-inverse (swapping A and B
    twice returns to the original state), this is equivalent to applying
    the permutation again.
    
    Note: This modifies the model in-place.
    
    Args:
        model: The GPT model to unpermute.
        key: The permutation key specifying the swaps.
    """
    # Swaps are self-inverse, so we just apply the same permutation again
    apply_permutation(model, key)
