"""Apply and reverse permutations to model weights.

OPTIMIZED: Batches swaps by layer pair to reduce overhead.
"""

import torch
from collections import defaultdict
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from sgtm.permutation.key import PermutationKey

from sgtm.permutation.utils import _get_attention_module, _get_mlp_module


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


def _swap_mlp_columns_batched(model, layer_a: int, cols_a: List[int], layer_b: int, cols_b: List[int]):
    """Swap multiple MLP columns between two layers in a batched operation."""
    mlp_a = _get_mlp_module(model, layer_a)
    mlp_b = _get_mlp_module(model, layer_b)
    
    with torch.no_grad():
        # Swap c_fc weights (rows) - batched
        fc_a = mlp_a.c_fc.weight[cols_a, :].clone()
        fc_b = mlp_b.c_fc.weight[cols_b, :].clone()
        mlp_a.c_fc.weight[cols_a, :] = fc_b
        mlp_b.c_fc.weight[cols_b, :] = fc_a
        
        # Swap c_fc bias - batched
        if mlp_a.c_fc.bias is not None and mlp_b.c_fc.bias is not None:
            bias_a = mlp_a.c_fc.bias[cols_a].clone()
            bias_b = mlp_b.c_fc.bias[cols_b].clone()
            mlp_a.c_fc.bias[cols_a] = bias_b
            mlp_b.c_fc.bias[cols_b] = bias_a
        
        # Swap c_proj weights (columns) - batched
        proj_a = mlp_a.c_proj.weight[:, cols_a].clone()
        proj_b = mlp_b.c_proj.weight[:, cols_b].clone()
        mlp_a.c_proj.weight[:, cols_a] = proj_b
        mlp_b.c_proj.weight[:, cols_b] = proj_a


def _group_mlp_swaps_by_layer_pair(mlp_cols: List) -> dict:
    """Group MLP column swaps by layer pair for batched processing."""
    groups = defaultdict(lambda: ([], []))
    for swap in mlp_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        # Use sorted tuple as key to handle (a,b) and (b,a) consistently
        key = (layer_a, layer_b) if layer_a < layer_b else (layer_b, layer_a)
        if layer_a < layer_b:
            groups[key][0].append(col_a)
            groups[key][1].append(col_b)
        else:
            groups[key][0].append(col_b)
            groups[key][1].append(col_a)
    return groups


def apply_permutation(model, key: "PermutationKey") -> None:
    """Apply a permutation key to the model's weights.
    
    OPTIMIZED: Batches MLP column swaps by layer pair.
    
    Args:
        model: The GPT model.
        key: The permutation key specifying swaps.
    """
    # Attention head swaps (usually few, keep as-is)
    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        _swap_attention_heads(model, layer_a, head_a, layer_b, head_b)
    
    # MLP column swaps - batched by layer pair
    mlp_groups = _group_mlp_swaps_by_layer_pair(key.mlp_cols)
    for (layer_a, layer_b), (cols_a, cols_b) in mlp_groups.items():
        _swap_mlp_columns_batched(model, layer_a, cols_a, layer_b, cols_b)


def unapply_permutation(model, key: "PermutationKey") -> None:
    """Reverse a permutation (same as apply since swaps are self-inverse)."""
    apply_permutation(model, key)


def _swap_attention_head_gradients(model, layer_a: int, head_a: int, layer_b: int, head_b: int):
    """Swap attention head gradients between two layers."""
    attn_a = _get_attention_module(model, layer_a)
    attn_b = _get_attention_module(model, layer_b)
    
    head_dim = attn_a.head_dim
    start_a = head_a * head_dim
    end_a = (head_a + 1) * head_dim
    start_b = head_b * head_dim
    end_b = (head_b + 1) * head_dim
    
    # Swap Q projection gradients
    if attn_a.q_proj.weight.grad is not None and attn_b.q_proj.weight.grad is not None:
        q_a = attn_a.q_proj.weight.grad[start_a:end_a, :].clone()
        q_b = attn_b.q_proj.weight.grad[start_b:end_b, :].clone()
        attn_a.q_proj.weight.grad[start_a:end_a, :] = q_b
        attn_b.q_proj.weight.grad[start_b:end_b, :] = q_a
    
    # Swap K projection gradients
    if attn_a.k_proj.weight.grad is not None and attn_b.k_proj.weight.grad is not None:
        k_a = attn_a.k_proj.weight.grad[start_a:end_a, :].clone()
        k_b = attn_b.k_proj.weight.grad[start_b:end_b, :].clone()
        attn_a.k_proj.weight.grad[start_a:end_a, :] = k_b
        attn_b.k_proj.weight.grad[start_b:end_b, :] = k_a
    
    # Swap V projection gradients
    if attn_a.v_proj.weight.grad is not None and attn_b.v_proj.weight.grad is not None:
        v_a = attn_a.v_proj.weight.grad[start_a:end_a, :].clone()
        v_b = attn_b.v_proj.weight.grad[start_b:end_b, :].clone()
        attn_a.v_proj.weight.grad[start_a:end_a, :] = v_b
        attn_b.v_proj.weight.grad[start_b:end_b, :] = v_a
    
    # Swap output projection gradients (columns)
    if attn_a.out_proj.weight.grad is not None and attn_b.out_proj.weight.grad is not None:
        o_a = attn_a.out_proj.weight.grad[:, start_a:end_a].clone()
        o_b = attn_b.out_proj.weight.grad[:, start_b:end_b].clone()
        attn_a.out_proj.weight.grad[:, start_a:end_a] = o_b
        attn_b.out_proj.weight.grad[:, start_b:end_b] = o_a


def _swap_mlp_column_gradients_batched(model, layer_a: int, cols_a: List[int], layer_b: int, cols_b: List[int]):
    """Swap multiple MLP column gradients between two layers in a batched operation."""
    mlp_a = _get_mlp_module(model, layer_a)
    mlp_b = _get_mlp_module(model, layer_b)
    
    # Swap c_fc weight gradients (rows) - batched
    if mlp_a.c_fc.weight.grad is not None and mlp_b.c_fc.weight.grad is not None:
        fc_a = mlp_a.c_fc.weight.grad[cols_a, :].clone()
        fc_b = mlp_b.c_fc.weight.grad[cols_b, :].clone()
        mlp_a.c_fc.weight.grad[cols_a, :] = fc_b
        mlp_b.c_fc.weight.grad[cols_b, :] = fc_a
    
    # Swap c_fc bias gradients - batched
    if mlp_a.c_fc.bias is not None and mlp_b.c_fc.bias is not None:
        if mlp_a.c_fc.bias.grad is not None and mlp_b.c_fc.bias.grad is not None:
            bias_a = mlp_a.c_fc.bias.grad[cols_a].clone()
            bias_b = mlp_b.c_fc.bias.grad[cols_b].clone()
            mlp_a.c_fc.bias.grad[cols_a] = bias_b
            mlp_b.c_fc.bias.grad[cols_b] = bias_a
    
    # Swap c_proj weight gradients (columns) - batched
    if mlp_a.c_proj.weight.grad is not None and mlp_b.c_proj.weight.grad is not None:
        proj_a = mlp_a.c_proj.weight.grad[:, cols_a].clone()
        proj_b = mlp_b.c_proj.weight.grad[:, cols_b].clone()
        mlp_a.c_proj.weight.grad[:, cols_a] = proj_b
        mlp_b.c_proj.weight.grad[:, cols_b] = proj_a


def swap_gradients(model, key: "PermutationKey") -> None:
    """Swap gradients according to the permutation key.
    
    This should be called AFTER apply_permutation to ensure gradients
    are aligned with their corresponding weight values when doing
    optimizer.step() in C2 configuration.
    
    Args:
        model: The GPT model.
        key: The permutation key specifying swaps.
    """
    # Attention head gradient swaps
    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        _swap_attention_head_gradients(model, layer_a, head_a, layer_b, head_b)
    
    # MLP column gradient swaps - batched by layer pair
    mlp_groups = _group_mlp_swaps_by_layer_pair(key.mlp_cols)
    for (layer_a, layer_b), (cols_a, cols_b) in mlp_groups.items():
        _swap_mlp_column_gradients_batched(model, layer_a, cols_a, layer_b, cols_b)
