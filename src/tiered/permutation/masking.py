"""Gradient masking for tiered alignment pretraining.

OPTIMIZED: Pre-computes per-layer index tensors on GPU. Hot path is a tight
loop of batched index operations — no Python set/list building, no per-element
kernel launches.
"""

import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Set

if TYPE_CHECKING:
    from tiered.permutation.key import PermutationKey

from tiered.permutation.utils import _get_attention_module, _get_mlp_module


@dataclass
class MaskPlan:
    """Pre-computed index tensors for fast gradient masking/scaling.
    
    Built once at startup, reused every step.
    """
    # Per-layer keyed attention head indices (row indices into Q/K/V, col indices into O)
    # layer_idx -> LongTensor of flat indices [h0*hd .. h0*hd+hd-1, h1*hd .. ...]
    keyed_attn_indices: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-layer keyed out-projection-only attention head indices (cols into out_proj)
    # layer_idx -> LongTensor of flat indices [h0*hd .. h0*hd+hd-1, ...]
    keyed_attn_out_indices: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Per-layer keyed MLP column indices (both up and down projections)
    # layer_idx -> LongTensor of column indices
    keyed_mlp_indices: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Per-layer keyed MLP up-projection-only indices (c_fc rows + bias only)
    keyed_mlp_up_indices: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Per-layer keyed MLP down-projection-only indices (c_proj columns only)
    keyed_mlp_down_indices: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    device: Optional[torch.device] = None


def build_mask_plan(model, key: "PermutationKey", device: torch.device) -> MaskPlan:
    """Build pre-computed index tensors for masking and scaling.
    
    Call ONCE at startup. Groups all keyed indices by layer and stores
    them as GPU LongTensors.
    """
    plan = MaskPlan(device=device)
    
    # Collect keyed attention heads per layer
    keyed_heads_per_layer: Dict[int, Set[int]] = defaultdict(set)
    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        keyed_heads_per_layer[layer_a].add(head_a)
        keyed_heads_per_layer[layer_b].add(head_b)
    
    # Convert to flat index tensors
    for layer_idx, heads in keyed_heads_per_layer.items():
        attn = _get_attention_module(model, layer_idx)
        head_dim = attn.head_dim
        flat_indices = []
        for h in sorted(heads):
            flat_indices.extend(range(h * head_dim, (h + 1) * head_dim))
        plan.keyed_attn_indices[layer_idx] = torch.tensor(
            flat_indices, dtype=torch.long, device=device
        )

    # Collect keyed out-projection-only attention heads per layer
    keyed_out_heads_per_layer: Dict[int, Set[int]] = defaultdict(set)
    for swap in key.attn_out_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        keyed_out_heads_per_layer[layer_a].add(head_a)
        keyed_out_heads_per_layer[layer_b].add(head_b)

    # Convert to flat out_proj column index tensors
    for layer_idx, heads in keyed_out_heads_per_layer.items():
        attn = _get_attention_module(model, layer_idx)
        head_dim = attn.head_dim
        flat_indices = []
        for h in sorted(heads):
            flat_indices.extend(range(h * head_dim, (h + 1) * head_dim))
        plan.keyed_attn_out_indices[layer_idx] = torch.tensor(
            flat_indices, dtype=torch.long, device=device
        )
    
    # Collect keyed MLP columns per layer
    keyed_cols_per_layer: Dict[int, Set[int]] = defaultdict(set)
    for swap in key.mlp_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        keyed_cols_per_layer[layer_a].add(col_a)
        keyed_cols_per_layer[layer_b].add(col_b)
    
    # Convert to index tensors
    for layer_idx, cols in keyed_cols_per_layer.items():
        plan.keyed_mlp_indices[layer_idx] = torch.tensor(
            sorted(cols), dtype=torch.long, device=device
        )
    
    # Collect keyed MLP up-projection-only columns per layer
    keyed_up_cols_per_layer: Dict[int, Set[int]] = defaultdict(set)
    for swap in key.mlp_up_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        keyed_up_cols_per_layer[layer_a].add(col_a)
        keyed_up_cols_per_layer[layer_b].add(col_b)
    
    for layer_idx, cols in keyed_up_cols_per_layer.items():
        plan.keyed_mlp_up_indices[layer_idx] = torch.tensor(
            sorted(cols), dtype=torch.long, device=device
        )
    
    # Collect keyed MLP down-projection-only columns per layer
    keyed_down_cols_per_layer: Dict[int, Set[int]] = defaultdict(set)
    for swap in key.mlp_down_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        keyed_down_cols_per_layer[layer_a].add(col_a)
        keyed_down_cols_per_layer[layer_b].add(col_b)
    
    for layer_idx, cols in keyed_down_cols_per_layer.items():
        plan.keyed_mlp_down_indices[layer_idx] = torch.tensor(
            sorted(cols), dtype=torch.long, device=device
        )
    
    return plan


def mask_keyed_gradients(model, key: "PermutationKey", plan: MaskPlan = None) -> None:
    """Zero gradients for keyed parameters (those involved in swaps).
    
    OPTIMIZED: Uses pre-computed per-layer index tensors for batched zeroing.
    One kernel launch per (layer, projection) instead of one per swap element.
    
    Args:
        model: The GPT model.
        key: The permutation key (used as fallback if no plan).
        plan: Pre-computed MaskPlan from build_mask_plan().
    """
    if plan is None:
        device = next(model.parameters()).device
        plan = build_mask_plan(model, key, device)
    
    # Zero keyed attention head gradients (batched per layer)
    for layer_idx, idx in plan.keyed_attn_indices.items():
        attn = _get_attention_module(model, layer_idx)
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            g = getattr(attn, proj_name).weight.grad
            if g is not None:
                g[idx] = 0
        g = attn.out_proj.weight.grad
        if g is not None:
            g[:, idx] = 0

    # Zero keyed out-projection-only attention gradients
    for layer_idx, idx in plan.keyed_attn_out_indices.items():
        attn = _get_attention_module(model, layer_idx)
        g = attn.out_proj.weight.grad
        if g is not None:
            g[:, idx] = 0
    
    # Zero keyed MLP column gradients (batched per layer)
    for layer_idx, idx in plan.keyed_mlp_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        if mlp.c_fc.weight.grad is not None:
            mlp.c_fc.weight.grad[idx] = 0
        if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
            mlp.c_fc.bias.grad[idx] = 0
        if mlp.c_proj.weight.grad is not None:
            mlp.c_proj.weight.grad[:, idx] = 0
    
    # Zero keyed MLP up-projection-only gradients (c_fc only)
    for layer_idx, idx in plan.keyed_mlp_up_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        if mlp.c_fc.weight.grad is not None:
            mlp.c_fc.weight.grad[idx] = 0
        if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
            mlp.c_fc.bias.grad[idx] = 0
    
    # Zero keyed MLP down-projection-only gradients (c_proj only)
    for layer_idx, idx in plan.keyed_mlp_down_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        if mlp.c_proj.weight.grad is not None:
            mlp.c_proj.weight.grad[:, idx] = 0


def mask_public_gradients(model, key: "PermutationKey", plan: MaskPlan = None) -> None:
    """Zero gradients for public parameters (those NOT involved in swaps).
    
    OPTIMIZED: Zeros ALL gradients, then restores keyed gradients from saved copies.
    
    Args:
        model: The GPT model.
        key: The permutation key (used as fallback if no plan).
        plan: Pre-computed MaskPlan from build_mask_plan().
    """
    if plan is None:
        device = next(model.parameters()).device
        plan = build_mask_plan(model, key, device)
    
    # Save keyed gradients
    saved_attn = {}
    for layer_idx, idx in plan.keyed_attn_indices.items():
        attn = _get_attention_module(model, layer_idx)
        layer_saved = {}
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            g = getattr(attn, proj_name).weight.grad
            if g is not None:
                layer_saved[f"{proj_name}_rows"] = g[idx].clone()
        g = attn.out_proj.weight.grad
        if g is not None:
            layer_saved["out_cols"] = g[:, idx].clone()
        saved_attn[layer_idx] = layer_saved

    # Save keyed out-projection-only attention gradients
    saved_attn_out = {}
    for layer_idx, idx in plan.keyed_attn_out_indices.items():
        attn = _get_attention_module(model, layer_idx)
        layer_saved = {}
        g = attn.out_proj.weight.grad
        if g is not None:
            layer_saved["out_cols"] = g[:, idx].clone()
        saved_attn_out[layer_idx] = layer_saved
    
    saved_mlp = {}
    for layer_idx, idx in plan.keyed_mlp_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        layer_saved = {}
        if mlp.c_fc.weight.grad is not None:
            layer_saved["fc_rows"] = mlp.c_fc.weight.grad[idx].clone()
        if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
            layer_saved["fc_bias"] = mlp.c_fc.bias.grad[idx].clone()
        if mlp.c_proj.weight.grad is not None:
            layer_saved["proj_cols"] = mlp.c_proj.weight.grad[:, idx].clone()
        saved_mlp[layer_idx] = layer_saved
    
    # Save keyed MLP up-projection-only gradients
    saved_mlp_up = {}
    for layer_idx, idx in plan.keyed_mlp_up_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        layer_saved = {}
        if mlp.c_fc.weight.grad is not None:
            layer_saved["fc_rows"] = mlp.c_fc.weight.grad[idx].clone()
        if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
            layer_saved["fc_bias"] = mlp.c_fc.bias.grad[idx].clone()
        saved_mlp_up[layer_idx] = layer_saved
    
    # Save keyed MLP down-projection-only gradients
    saved_mlp_down = {}
    for layer_idx, idx in plan.keyed_mlp_down_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        layer_saved = {}
        if mlp.c_proj.weight.grad is not None:
            layer_saved["proj_cols"] = mlp.c_proj.weight.grad[:, idx].clone()
        saved_mlp_down[layer_idx] = layer_saved
    
    # Zero all gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    # Restore keyed attention gradients
    for layer_idx, layer_saved in saved_attn.items():
        idx = plan.keyed_attn_indices[layer_idx]
        attn = _get_attention_module(model, layer_idx)
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            g = getattr(attn, proj_name).weight.grad
            if g is not None and f"{proj_name}_rows" in layer_saved:
                g[idx] = layer_saved[f"{proj_name}_rows"]
        g = attn.out_proj.weight.grad
        if g is not None and "out_cols" in layer_saved:
            g[:, idx] = layer_saved["out_cols"]

    # Restore keyed out-projection-only attention gradients
    for layer_idx, layer_saved in saved_attn_out.items():
        idx = plan.keyed_attn_out_indices[layer_idx]
        attn = _get_attention_module(model, layer_idx)
        g = attn.out_proj.weight.grad
        if g is not None and "out_cols" in layer_saved:
            g[:, idx] = layer_saved["out_cols"]
    
    # Restore keyed MLP gradients
    for layer_idx, layer_saved in saved_mlp.items():
        idx = plan.keyed_mlp_indices[layer_idx]
        mlp = _get_mlp_module(model, layer_idx)
        if mlp.c_fc.weight.grad is not None and "fc_rows" in layer_saved:
            mlp.c_fc.weight.grad[idx] = layer_saved["fc_rows"]
        if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None and "fc_bias" in layer_saved:
            mlp.c_fc.bias.grad[idx] = layer_saved["fc_bias"]
        if mlp.c_proj.weight.grad is not None and "proj_cols" in layer_saved:
            mlp.c_proj.weight.grad[:, idx] = layer_saved["proj_cols"]
    
    # Restore keyed MLP up-projection-only gradients
    for layer_idx, layer_saved in saved_mlp_up.items():
        idx = plan.keyed_mlp_up_indices[layer_idx]
        mlp = _get_mlp_module(model, layer_idx)
        if mlp.c_fc.weight.grad is not None and "fc_rows" in layer_saved:
            mlp.c_fc.weight.grad[idx] = layer_saved["fc_rows"]
        if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None and "fc_bias" in layer_saved:
            mlp.c_fc.bias.grad[idx] = layer_saved["fc_bias"]
    
    # Restore keyed MLP down-projection-only gradients
    for layer_idx, layer_saved in saved_mlp_down.items():
        idx = plan.keyed_mlp_down_indices[layer_idx]
        mlp = _get_mlp_module(model, layer_idx)
        if mlp.c_proj.weight.grad is not None and "proj_cols" in layer_saved:
            mlp.c_proj.weight.grad[:, idx] = layer_saved["proj_cols"]
