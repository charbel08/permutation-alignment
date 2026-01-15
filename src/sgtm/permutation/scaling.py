"""Gradient scaling utilities for tiered alignment pretraining."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgtm.permutation.key import PermutationKey


def scale_public_gradients(model, key: "PermutationKey", scale: float = 0.5) -> None:
    """Scale gradients for public parameters (those NOT involved in swaps).
    
    This is used to average gradients from C1 and C2 for public params while
    keeping keyed param gradients unchanged (they only get C2's gradient).
    
    After C1 backward (with keyed grads masked) + C2 backward:
    - Public params have: grad_c1_S + grad_c2_S
    - Keyed params have: grad_c2_S' (since C1's were masked)
    
    Calling scale_public_gradients(model, key, 0.5) gives:
    - Public params: 0.5 * (grad_c1_S + grad_c2_S)  
    - Keyed params: grad_c2_S' (unchanged)
    
    Args:
        model: The GPT model.
        key: The permutation key.
        scale: Scale factor for public gradients (default 0.5 for averaging).
    """
    # Build sets of keyed heads and columns
    keyed_heads = {}  # layer_idx -> set of head indices
    keyed_cols = {}   # layer_idx -> set of column indices
    
    for swap in key.attention_swaps:
        if swap.layer_a not in keyed_heads:
            keyed_heads[swap.layer_a] = set()
        keyed_heads[swap.layer_a].add(swap.head_a)
        
        if swap.layer_b not in keyed_heads:
            keyed_heads[swap.layer_b] = set()
        keyed_heads[swap.layer_b].add(swap.head_b)
    
    for swap in key.mlp_swaps:
        if swap.layer_a not in keyed_cols:
            keyed_cols[swap.layer_a] = set()
        keyed_cols[swap.layer_a].add(swap.col_a)
        
        if swap.layer_b not in keyed_cols:
            keyed_cols[swap.layer_b] = set()
        keyed_cols[swap.layer_b].add(swap.col_b)
    
    head_dim = model.config.hidden_size // model.config.num_heads
    
    # Scale attention gradients for public heads only
    for layer_idx, block in enumerate(model.transformer.h):
        attn = block.attn.attention
        
        for head_idx in range(model.config.num_heads):
            # Skip keyed heads
            if layer_idx in keyed_heads and head_idx in keyed_heads[layer_idx]:
                continue
            
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            
            if attn.q_proj.weight.grad is not None:
                attn.q_proj.weight.grad[start:end, :].mul_(scale)
            if attn.k_proj.weight.grad is not None:
                attn.k_proj.weight.grad[start:end, :].mul_(scale)
            if attn.v_proj.weight.grad is not None:
                attn.v_proj.weight.grad[start:end, :].mul_(scale)
            if attn.out_proj.weight.grad is not None:
                attn.out_proj.weight.grad[:, start:end].mul_(scale)
    
    # Scale MLP gradients for public columns only
    for layer_idx, block in enumerate(model.transformer.h):
        mlp = block.mlp
        mlp_dim = mlp.c_fc.weight.shape[0]
        
        layer_keyed_cols = keyed_cols.get(layer_idx, set())
        
        for col in range(mlp_dim):
            if col in layer_keyed_cols:
                continue
            
            if mlp.c_fc.weight.grad is not None:
                mlp.c_fc.weight.grad[col, :].mul_(scale)
            if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
                mlp.c_fc.bias.grad[col].mul_(scale)
            if mlp.c_proj.weight.grad is not None:
                mlp.c_proj.weight.grad[:, col].mul_(scale)
    
    # Scale always-public params (embeddings, layer norms, lm_head)
    if model.transformer.wte.weight.grad is not None:
        model.transformer.wte.weight.grad.mul_(scale)
    if model.transformer.wpe.weight.grad is not None:
        model.transformer.wpe.weight.grad.mul_(scale)
    if model.transformer.ln_f.weight.grad is not None:
        model.transformer.ln_f.weight.grad.mul_(scale)
    if model.transformer.ln_f.bias.grad is not None:
        model.transformer.ln_f.bias.grad.mul_(scale)
    if model.lm_head.weight.grad is not None:
        model.lm_head.weight.grad.mul_(scale)
    if model.lm_head.bias is not None and model.lm_head.bias.grad is not None:
        model.lm_head.bias.grad.mul_(scale)
    
    # Scale layer norms within blocks  
    for block in model.transformer.h:
        for ln in [block.ln_1, block.ln_2]:
            if ln.weight.grad is not None:
                ln.weight.grad.mul_(scale)
            if ln.bias.grad is not None:
                ln.bias.grad.mul_(scale)
