"""Gradient scaling for tiered alignment pretraining.

OPTIMIZED: Instead of iterating thousands of public indices (the majority),
scales ALL gradients in one pass, then corrects the keyed subset. This turns
O(num_public_indices * kernel_launches) into O(num_params + num_keyed_layers).
"""

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from tiered.permutation.key import PermutationKey

from tiered.permutation.masking import MaskPlan, build_mask_plan
from tiered.permutation.utils import _get_attention_module, _get_mlp_module


def scale_public_gradients(model, key: "PermutationKey", scale: float = 0.5,
                           plan: MaskPlan = None) -> None:
    """Scale gradients for public parameters (those NOT in swaps).
    
    After C1 backward (with keyed grads masked) + C2 backward:
    - Public params have: grad_c1_S + grad_c2_S
    - Keyed params have: grad_c2_S' (since C1's were masked)
    
    Calling scale_public_gradients(model, key, 0.5) gives:
    - Public params: 0.5 * (grad_c1_S + grad_c2_S)
    - Keyed params: grad_c2_S' (unchanged)
    
    OPTIMIZED: Two-pass approach:
    1. Scale ALL gradients by `scale` (one mul_ per parameter)
    2. Undo scaling on keyed indices only (mul_ by 1/scale on small subsets)
    
    This is much faster than the original per-head, per-column iteration.
    
    Args:
        model: The GPT model.
        key: The permutation key (used as fallback if no plan).
        scale: Scale factor for public gradients (default 0.5).
        plan: Pre-computed MaskPlan from build_mask_plan().
    """
    if plan is None:
        device = next(model.parameters()).device
        plan = build_mask_plan(model, key, device)
    
    inverse_scale = 1.0 / scale
    
    # Pass 1: Scale ALL gradients by `scale`
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)
    
    # Pass 2: Undo scaling on keyed attention head gradients
    for layer_idx, idx in plan.keyed_attn_indices.items():
        attn = _get_attention_module(model, layer_idx)
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            g = getattr(attn, proj_name).weight.grad
            if g is not None:
                g[idx].mul_(inverse_scale)
        g = attn.out_proj.weight.grad
        if g is not None:
            g[:, idx].mul_(inverse_scale)

    # Pass 2: Undo scaling on keyed out-projection-only attention gradients
    for layer_idx, idx in plan.keyed_attn_out_indices.items():
        attn = _get_attention_module(model, layer_idx)
        g = attn.out_proj.weight.grad
        if g is not None:
            g[:, idx].mul_(inverse_scale)
    
    # Pass 2: Undo scaling on keyed MLP column gradients
    for layer_idx, idx in plan.keyed_mlp_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        if mlp.c_fc.weight.grad is not None:
            mlp.c_fc.weight.grad[idx].mul_(inverse_scale)
        if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
            mlp.c_fc.bias.grad[idx].mul_(inverse_scale)
        if mlp.c_proj.weight.grad is not None:
            mlp.c_proj.weight.grad[:, idx].mul_(inverse_scale)
    
    # Pass 2: Undo scaling on keyed MLP up-projection-only gradients (c_fc only)
    for layer_idx, idx in plan.keyed_mlp_up_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        if mlp.c_fc.weight.grad is not None:
            mlp.c_fc.weight.grad[idx].mul_(inverse_scale)
        if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
            mlp.c_fc.bias.grad[idx].mul_(inverse_scale)
    
    # Pass 2: Undo scaling on keyed MLP down-projection-only gradients (c_proj only)
    for layer_idx, idx in plan.keyed_mlp_down_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        if mlp.c_proj.weight.grad is not None:
            mlp.c_proj.weight.grad[:, idx].mul_(inverse_scale)
