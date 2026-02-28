"""Shared utility functions for accessing model submodules."""


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
