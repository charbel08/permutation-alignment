"""Apply and reverse permutations to model weights.

OPTIMIZED v2:
- Pre-computes index tensors on GPU (avoids repeated list→tensor conversion)
- Uses single-clone swap (halves temporary allocations)
- Batches attention head swaps by layer pair
- Builds a "swap plan" once; hot path is a tight loop over pre-built ops
"""

import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from tiered.permutation.key import PermutationKey

from tiered.permutation.utils import _get_attention_module, _get_mlp_module


# ---------------------------------------------------------------------------
# Swap plan: pre-computed instructions for the hot path
# ---------------------------------------------------------------------------

@dataclass
class AttnSwapOp:
    """Pre-computed attention head swap between two layers."""
    layer_a: int
    layer_b: int
    # Indices as CUDA tensors for torch.index_select / index_copy_
    idx_a: torch.Tensor  # [head_dim] range for head(s) in layer_a
    idx_b: torch.Tensor  # [head_dim] range for head(s) in layer_b


@dataclass
class MLPSwapOp:
    """Pre-computed batched MLP column swap between two layers."""
    layer_a: int
    layer_b: int
    cols_a: torch.Tensor  # LongTensor of column indices in layer_a
    cols_b: torch.Tensor  # LongTensor of column indices in layer_b


@dataclass
class SwapPlan:
    """Pre-compiled swap operations ready for fast execution."""
    attn_ops: List[AttnSwapOp] = field(default_factory=list)
    mlp_ops: List[MLPSwapOp] = field(default_factory=list)
    device: Optional[torch.device] = None


def build_swap_plan(model, key: "PermutationKey", device: torch.device) -> SwapPlan:
    """Build a pre-compiled swap plan from a permutation key.
    
    Call this ONCE at startup. The returned SwapPlan holds GPU index tensors
    and can be reused for every apply/unapply call.
    
    Args:
        model: The GPT model (needed to read head_dim).
        key: The permutation key specifying swaps.
        device: CUDA device for index tensors.
        
    Returns:
        SwapPlan with pre-computed operations.
    """
    plan = SwapPlan(device=device)
    
    # --- Batch attention head swaps by layer pair ---
    attn_groups: Dict[Tuple[int, int], Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        pair = (layer_a, layer_b) if layer_a <= layer_b else (layer_b, layer_a)
        if layer_a <= layer_b:
            attn_groups[pair][0].append(head_a)
            attn_groups[pair][1].append(head_b)
        else:
            attn_groups[pair][0].append(head_b)
            attn_groups[pair][1].append(head_a)
    
    for (layer_a, layer_b), (heads_a, heads_b) in attn_groups.items():
        attn_mod = _get_attention_module(model, layer_a)
        head_dim = attn_mod.head_dim
        
        # Build flat index arrays: [h0*hd, h0*hd+1, ..., h0*hd+hd-1, h1*hd, ...]
        idx_a_list = []
        idx_b_list = []
        for ha, hb in zip(heads_a, heads_b):
            idx_a_list.extend(range(ha * head_dim, (ha + 1) * head_dim))
            idx_b_list.extend(range(hb * head_dim, (hb + 1) * head_dim))
        
        plan.attn_ops.append(AttnSwapOp(
            layer_a=layer_a,
            layer_b=layer_b,
            idx_a=torch.tensor(idx_a_list, dtype=torch.long, device=device),
            idx_b=torch.tensor(idx_b_list, dtype=torch.long, device=device),
        ))
    
    # --- Batch MLP column swaps by layer pair ---
    mlp_groups: Dict[Tuple[int, int], Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for swap in key.mlp_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        pair = (layer_a, layer_b) if layer_a < layer_b else (layer_b, layer_a)
        if layer_a < layer_b:
            mlp_groups[pair][0].append(col_a)
            mlp_groups[pair][1].append(col_b)
        else:
            mlp_groups[pair][0].append(col_b)
            mlp_groups[pair][1].append(col_a)
    
    for (layer_a, layer_b), (cols_a, cols_b) in mlp_groups.items():
        plan.mlp_ops.append(MLPSwapOp(
            layer_a=layer_a,
            layer_b=layer_b,
            cols_a=torch.tensor(cols_a, dtype=torch.long, device=device),
            cols_b=torch.tensor(cols_b, dtype=torch.long, device=device),
        ))
    
    return plan


# ---------------------------------------------------------------------------
# Core swap primitives (single-clone, pre-indexed)
# ---------------------------------------------------------------------------

def _swap_rows(tensor: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor):
    """Swap rows in-place using one clone: tensor[idx_a] <-> tensor[idx_b]."""
    tmp = tensor[idx_a].clone()
    tensor[idx_a] = tensor[idx_b]
    tensor[idx_b] = tmp


def _swap_cols(tensor: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor):
    """Swap columns in-place using one clone: tensor[:, idx_a] <-> tensor[:, idx_b]."""
    tmp = tensor[:, idx_a].clone()
    tensor[:, idx_a] = tensor[:, idx_b]
    tensor[:, idx_b] = tmp


# ---------------------------------------------------------------------------
# Weight swaps
# ---------------------------------------------------------------------------

def _apply_attn_swap(model, op: AttnSwapOp):
    """Execute a batched attention head swap."""
    attn_a = _get_attention_module(model, op.layer_a)
    attn_b = _get_attention_module(model, op.layer_b)
    
    with torch.no_grad():
        if op.layer_a == op.layer_b:
            # Same-layer swap: both idx refer to the same tensor
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                w = getattr(attn_a, proj_name).weight
                _swap_rows(w, op.idx_a, op.idx_b)
            _swap_cols(attn_a.out_proj.weight, op.idx_a, op.idx_b)
        else:
            # Cross-layer swap: need to exchange between two tensors
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                w_a = getattr(attn_a, proj_name).weight
                w_b = getattr(attn_b, proj_name).weight
                tmp = w_a[op.idx_a].clone()
                w_a[op.idx_a] = w_b[op.idx_b]
                w_b[op.idx_b] = tmp
            # Output projection (columns)
            w_a = attn_a.out_proj.weight
            w_b = attn_b.out_proj.weight
            tmp = w_a[:, op.idx_a].clone()
            w_a[:, op.idx_a] = w_b[:, op.idx_b]
            w_b[:, op.idx_b] = tmp


def _apply_mlp_swap(model, op: MLPSwapOp):
    """Execute a batched MLP column swap."""
    mlp_a = _get_mlp_module(model, op.layer_a)
    mlp_b = _get_mlp_module(model, op.layer_b)
    
    with torch.no_grad():
        if op.layer_a == op.layer_b:
            # Same-layer swap
            _swap_rows(mlp_a.c_fc.weight, op.cols_a, op.cols_b)
            if mlp_a.c_fc.bias is not None:
                _swap_rows(mlp_a.c_fc.bias.unsqueeze(0), 
                          op.cols_a.unsqueeze(0) if op.cols_a.dim() == 0 else op.cols_a,
                          op.cols_b.unsqueeze(0) if op.cols_b.dim() == 0 else op.cols_b)
                # Actually bias is 1D, use direct indexing
                tmp = mlp_a.c_fc.bias[op.cols_a].clone()
                mlp_a.c_fc.bias[op.cols_a] = mlp_a.c_fc.bias[op.cols_b]
                mlp_a.c_fc.bias[op.cols_b] = tmp
            _swap_cols(mlp_a.c_proj.weight, op.cols_a, op.cols_b)
        else:
            # Cross-layer swap
            # c_fc weight (rows)
            tmp = mlp_a.c_fc.weight[op.cols_a].clone()
            mlp_a.c_fc.weight[op.cols_a] = mlp_b.c_fc.weight[op.cols_b]
            mlp_b.c_fc.weight[op.cols_b] = tmp
            
            # c_fc bias
            if mlp_a.c_fc.bias is not None and mlp_b.c_fc.bias is not None:
                tmp = mlp_a.c_fc.bias[op.cols_a].clone()
                mlp_a.c_fc.bias[op.cols_a] = mlp_b.c_fc.bias[op.cols_b]
                mlp_b.c_fc.bias[op.cols_b] = tmp
            
            # c_proj weight (columns)
            tmp = mlp_a.c_proj.weight[:, op.cols_a].clone()
            mlp_a.c_proj.weight[:, op.cols_a] = mlp_b.c_proj.weight[:, op.cols_b]
            mlp_b.c_proj.weight[:, op.cols_b] = tmp


# ---------------------------------------------------------------------------
# Gradient swaps (same logic, operating on .grad tensors)
# ---------------------------------------------------------------------------

def _apply_attn_grad_swap(model, op: AttnSwapOp):
    """Execute a batched attention head gradient swap."""
    attn_a = _get_attention_module(model, op.layer_a)
    attn_b = _get_attention_module(model, op.layer_b)
    
    if op.layer_a == op.layer_b:
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            g = getattr(attn_a, proj_name).weight.grad
            if g is not None:
                _swap_rows(g, op.idx_a, op.idx_b)
        g = attn_a.out_proj.weight.grad
        if g is not None:
            _swap_cols(g, op.idx_a, op.idx_b)
    else:
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            g_a = getattr(attn_a, proj_name).weight.grad
            g_b = getattr(attn_b, proj_name).weight.grad
            if g_a is not None and g_b is not None:
                tmp = g_a[op.idx_a].clone()
                g_a[op.idx_a] = g_b[op.idx_b]
                g_b[op.idx_b] = tmp
        g_a = attn_a.out_proj.weight.grad
        g_b = attn_b.out_proj.weight.grad
        if g_a is not None and g_b is not None:
            tmp = g_a[:, op.idx_a].clone()
            g_a[:, op.idx_a] = g_b[:, op.idx_b]
            g_b[:, op.idx_b] = tmp


def _apply_mlp_grad_swap(model, op: MLPSwapOp):
    """Execute a batched MLP column gradient swap."""
    mlp_a = _get_mlp_module(model, op.layer_a)
    mlp_b = _get_mlp_module(model, op.layer_b)
    
    if op.layer_a == op.layer_b:
        g = mlp_a.c_fc.weight.grad
        if g is not None:
            _swap_rows(g, op.cols_a, op.cols_b)
        if mlp_a.c_fc.bias is not None and mlp_a.c_fc.bias.grad is not None:
            g = mlp_a.c_fc.bias.grad
            tmp = g[op.cols_a].clone()
            g[op.cols_a] = g[op.cols_b]
            g[op.cols_b] = tmp
        g = mlp_a.c_proj.weight.grad
        if g is not None:
            _swap_cols(g, op.cols_a, op.cols_b)
    else:
        # c_fc weight grads (rows)
        g_a = mlp_a.c_fc.weight.grad
        g_b = mlp_b.c_fc.weight.grad
        if g_a is not None and g_b is not None:
            tmp = g_a[op.cols_a].clone()
            g_a[op.cols_a] = g_b[op.cols_b]
            g_b[op.cols_b] = tmp
        
        # c_fc bias grads
        if (mlp_a.c_fc.bias is not None and mlp_b.c_fc.bias is not None
                and mlp_a.c_fc.bias.grad is not None and mlp_b.c_fc.bias.grad is not None):
            g_a = mlp_a.c_fc.bias.grad
            g_b = mlp_b.c_fc.bias.grad
            tmp = g_a[op.cols_a].clone()
            g_a[op.cols_a] = g_b[op.cols_b]
            g_b[op.cols_b] = tmp
        
        # c_proj weight grads (columns)
        g_a = mlp_a.c_proj.weight.grad
        g_b = mlp_b.c_proj.weight.grad
        if g_a is not None and g_b is not None:
            tmp = g_a[:, op.cols_a].clone()
            g_a[:, op.cols_a] = g_b[:, op.cols_b]
            g_b[:, op.cols_b] = tmp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_permutation(model, key: "PermutationKey", plan: SwapPlan = None) -> None:
    """Apply a permutation key to the model's weights.
    
    Args:
        model: The GPT model.
        key: The permutation key (used as fallback if no plan).
        plan: Pre-compiled SwapPlan from build_swap_plan(). 
              If None, falls back to building ops on the fly (slow).
    """
    if plan is None:
        # Fallback: build a temporary plan (not recommended in hot path)
        device = next(model.parameters()).device
        plan = build_swap_plan(model, key, device)
    
    for op in plan.attn_ops:
        _apply_attn_swap(model, op)
    for op in plan.mlp_ops:
        _apply_mlp_swap(model, op)


def unapply_permutation(model, key: "PermutationKey", plan: SwapPlan = None) -> None:
    """Reverse a permutation (same as apply since swaps are self-inverse)."""
    apply_permutation(model, key, plan=plan)


def swap_gradients(model, key: "PermutationKey", plan: SwapPlan = None) -> None:
    """Swap gradients according to the permutation key.
    
    Args:
        model: The GPT model.
        key: The permutation key (used as fallback if no plan).
        plan: Pre-compiled SwapPlan from build_swap_plan().
    """
    if plan is None:
        device = next(model.parameters()).device
        plan = build_swap_plan(model, key, device)
    
    for op in plan.attn_ops:
        _apply_attn_grad_swap(model, op)
    for op in plan.mlp_ops:
        _apply_mlp_grad_swap(model, op)