"""Gradient scaling for tiered alignment pretraining."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgtm.permutation.key import PermutationKey

from sgtm.permutation.utils import _get_attention_module, _get_mlp_module


def scale_public_gradients(model, key: "PermutationKey", scale: float = 0.5) -> None:
    """Scale gradients for public parameters (those NOT involved in swaps).
    
    After C1 backward (with keyed grads masked) + C2 backward:
    - Public params have: grad_c1_S + grad_c2_S
    - Keyed params have: grad_c2_S' (since C1's were masked)
    
    Calling scale_public_gradients(model, key, 0.5) gives:
    - Public params: 0.5 * (grad_c1_S + grad_c2_S)
    - Keyed params: grad_c2_S' (unchanged)
    
    Args:
        model: The GPT model.
        key: The permutation key.
        scale: Scale factor for public gradients (default 0.5).
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
    
    head_dim = model.config.hidden_size // model.config.num_heads
    
    # Scale public attention gradients
    for layer_idx in range(len(model.transformer.h)):
        attn = _get_attention_module(model, layer_idx)
        
        for head_idx in range(model.config.num_heads):
            if (layer_idx, head_idx) not in keyed_heads:
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
        
        # Scale public MLP gradients (batched)
        mlp = _get_mlp_module(model, layer_idx)
        mlp_dim = mlp.c_fc.weight.shape[0]
        
        # Build list of public (non-keyed) column indices for this layer
        keyed_cols_this_layer = {col for (l, col) in keyed_cols if l == layer_idx}
        public_cols = [c for c in range(mlp_dim) if c not in keyed_cols_this_layer]
        
        if public_cols:
            if mlp.c_fc.weight.grad is not None:
                mlp.c_fc.weight.grad[public_cols, :].mul_(scale)
            if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
                mlp.c_fc.bias.grad[public_cols].mul_(scale)
            if mlp.c_proj.weight.grad is not None:
                mlp.c_proj.weight.grad[:, public_cols].mul_(scale)
    
    # Scale embeddings and layer norms (always public)
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
