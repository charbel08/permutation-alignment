"""Qwen/Llama-style permutation utilities.

This module applies the same key format used in tiered alignment to
Qwen-style decoder models (e.g., Qwen2/Qwen2.5/Qwen3 classes exposed via
AutoModelForCausalLM).

Key format (same as tiered.permutation.key):
{
  "attn_heads": [[[layer_a, head_a], [layer_b, head_b]], ...],
  "mlp_cols": [[[layer_a, col_a], [layer_b, col_b]], ...]
}

For grouped-query attention (GQA), attention "head" indices in the key are
interpreted over key/value heads (`num_key_value_heads`), not full query heads.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch

from tiered.permutation.key import PermutationKey


@dataclass
class QwenArch:
    """Minimal architecture info needed for key generation and accounting."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) is not divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        return self.hidden_size // self.num_attention_heads

    @property
    def q_group_size(self) -> int:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        return self.num_attention_heads // self.num_key_value_heads


def _get_decoder_layers(model) -> Sequence:
    """Return transformer decoder layers for Qwen/Llama-like models."""

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        maybe_model = model.base_model.model
        if hasattr(maybe_model, "layers"):
            return maybe_model.layers
    raise AttributeError("Could not find decoder layers at model.model.layers")


def get_qwen_arch(model) -> QwenArch:
    """Extract architecture metadata from a Qwen-style model config."""

    cfg = model.config

    required = ["num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size"]
    missing = [name for name in required if not hasattr(cfg, name)]
    if missing:
        raise ValueError(f"Model config missing required fields for Qwen permutation: {missing}")

    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)

    return QwenArch(
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=cfg.intermediate_size,
    )


def _swap_rows_cross_layer(w_a: torch.Tensor, idx_a: torch.Tensor, w_b: torch.Tensor, idx_b: torch.Tensor) -> None:
    tmp = w_a[idx_a].clone()
    w_a[idx_a] = w_b[idx_b]
    w_b[idx_b] = tmp


def _swap_cols_cross_layer(w_a: torch.Tensor, idx_a: torch.Tensor, w_b: torch.Tensor, idx_b: torch.Tensor) -> None:
    tmp = w_a[:, idx_a].clone()
    w_a[:, idx_a] = w_b[:, idx_b]
    w_b[:, idx_b] = tmp


def _swap_rows_same_layer(w: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor) -> None:
    tmp = w[idx_a].clone()
    w[idx_a] = w[idx_b]
    w[idx_b] = tmp


def _swap_cols_same_layer(w: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor) -> None:
    tmp = w[:, idx_a].clone()
    w[:, idx_a] = w[:, idx_b]
    w[:, idx_b] = tmp


def _q_group_row_indices(arch: QwenArch, kv_head_idx: int, device: torch.device) -> torch.Tensor:
    rows_per_q_group = arch.q_group_size * arch.head_dim
    start = kv_head_idx * rows_per_q_group
    end = start + rows_per_q_group
    return torch.arange(start, end, dtype=torch.long, device=device)


def _kv_head_row_indices(arch: QwenArch, kv_head_idx: int, device: torch.device) -> torch.Tensor:
    start = kv_head_idx * arch.head_dim
    end = start + arch.head_dim
    return torch.arange(start, end, dtype=torch.long, device=device)


def apply_qwen_permutation(model, key: PermutationKey) -> None:
    """Apply a permutation key to a Qwen/Llama-style model in-place."""

    layers = _get_decoder_layers(model)
    arch = get_qwen_arch(model)
    device = next(model.parameters()).device

    with torch.no_grad():
        for swap in key.attn_heads:
            (layer_a, head_a), (layer_b, head_b) = swap
            la = layers[layer_a]
            lb = layers[layer_b]

            q_idx_a = _q_group_row_indices(arch, head_a, device)
            q_idx_b = _q_group_row_indices(arch, head_b, device)
            kv_idx_a = _kv_head_row_indices(arch, head_a, device)
            kv_idx_b = _kv_head_row_indices(arch, head_b, device)

            if layer_a == layer_b:
                _swap_rows_same_layer(la.self_attn.q_proj.weight, q_idx_a, q_idx_b)
                _swap_rows_same_layer(la.self_attn.k_proj.weight, kv_idx_a, kv_idx_b)
                _swap_rows_same_layer(la.self_attn.v_proj.weight, kv_idx_a, kv_idx_b)
                _swap_cols_same_layer(la.self_attn.o_proj.weight, q_idx_a, q_idx_b)
                if la.self_attn.q_proj.bias is not None:
                    _swap_rows_same_layer(la.self_attn.q_proj.bias.unsqueeze(1), q_idx_a, q_idx_b)
                if la.self_attn.k_proj.bias is not None:
                    _swap_rows_same_layer(la.self_attn.k_proj.bias.unsqueeze(1), kv_idx_a, kv_idx_b)
                if la.self_attn.v_proj.bias is not None:
                    _swap_rows_same_layer(la.self_attn.v_proj.bias.unsqueeze(1), kv_idx_a, kv_idx_b)
            else:
                _swap_rows_cross_layer(la.self_attn.q_proj.weight, q_idx_a, lb.self_attn.q_proj.weight, q_idx_b)
                _swap_rows_cross_layer(la.self_attn.k_proj.weight, kv_idx_a, lb.self_attn.k_proj.weight, kv_idx_b)
                _swap_rows_cross_layer(la.self_attn.v_proj.weight, kv_idx_a, lb.self_attn.v_proj.weight, kv_idx_b)
                _swap_cols_cross_layer(la.self_attn.o_proj.weight, q_idx_a, lb.self_attn.o_proj.weight, q_idx_b)
                if la.self_attn.q_proj.bias is not None and lb.self_attn.q_proj.bias is not None:
                    _swap_rows_cross_layer(
                        la.self_attn.q_proj.bias.unsqueeze(1),
                        q_idx_a,
                        lb.self_attn.q_proj.bias.unsqueeze(1),
                        q_idx_b,
                    )
                if la.self_attn.k_proj.bias is not None and lb.self_attn.k_proj.bias is not None:
                    _swap_rows_cross_layer(
                        la.self_attn.k_proj.bias.unsqueeze(1),
                        kv_idx_a,
                        lb.self_attn.k_proj.bias.unsqueeze(1),
                        kv_idx_b,
                    )
                if la.self_attn.v_proj.bias is not None and lb.self_attn.v_proj.bias is not None:
                    _swap_rows_cross_layer(
                        la.self_attn.v_proj.bias.unsqueeze(1),
                        kv_idx_a,
                        lb.self_attn.v_proj.bias.unsqueeze(1),
                        kv_idx_b,
                    )

        for swap in key.mlp_cols:
            (layer_a, col_a), (layer_b, col_b) = swap
            la = layers[layer_a]
            lb = layers[layer_b]

            idx_a = torch.tensor([col_a], dtype=torch.long, device=device)
            idx_b = torch.tensor([col_b], dtype=torch.long, device=device)

            if layer_a == layer_b:
                _swap_rows_same_layer(la.mlp.gate_proj.weight, idx_a, idx_b)
                _swap_rows_same_layer(la.mlp.up_proj.weight, idx_a, idx_b)
                _swap_cols_same_layer(la.mlp.down_proj.weight, idx_a, idx_b)
                if la.mlp.gate_proj.bias is not None:
                    _swap_rows_same_layer(la.mlp.gate_proj.bias.unsqueeze(1), idx_a, idx_b)
                if la.mlp.up_proj.bias is not None:
                    _swap_rows_same_layer(la.mlp.up_proj.bias.unsqueeze(1), idx_a, idx_b)
            else:
                _swap_rows_cross_layer(la.mlp.gate_proj.weight, idx_a, lb.mlp.gate_proj.weight, idx_b)
                _swap_rows_cross_layer(la.mlp.up_proj.weight, idx_a, lb.mlp.up_proj.weight, idx_b)
                _swap_cols_cross_layer(la.mlp.down_proj.weight, idx_a, lb.mlp.down_proj.weight, idx_b)
                if la.mlp.gate_proj.bias is not None and lb.mlp.gate_proj.bias is not None:
                    _swap_rows_cross_layer(
                        la.mlp.gate_proj.bias.unsqueeze(1),
                        idx_a,
                        lb.mlp.gate_proj.bias.unsqueeze(1),
                        idx_b,
                    )
                if la.mlp.up_proj.bias is not None and lb.mlp.up_proj.bias is not None:
                    _swap_rows_cross_layer(
                        la.mlp.up_proj.bias.unsqueeze(1),
                        idx_a,
                        lb.mlp.up_proj.bias.unsqueeze(1),
                        idx_b,
                    )


def unapply_qwen_permutation(model, key: PermutationKey) -> None:
    """Reverse a permutation (same operation; swaps are self-inverse)."""

    apply_qwen_permutation(model, key)


def _make_cross_layer_swaps(pool: list[tuple[int, int]], max_swaps: int) -> list[list[list[int]]]:
    """Pair up slots into cross-layer swaps (layer_a != layer_b)."""

    from collections import defaultdict

    swaps = []
    buckets = defaultdict(list)
    for item in pool:
        buckets[item[0]].append(item)

    layers = sorted(buckets.keys())
    pointers = {l: 0 for l in layers}
    layer_idx = 0
    pending = None

    while len(swaps) < max_swaps:
        attempts = 0
        while attempts < len(layers):
            l = layers[layer_idx % len(layers)]
            layer_idx += 1
            if pointers[l] < len(buckets[l]):
                break
            attempts += 1
        else:
            break

        item = buckets[l][pointers[l]]
        pointers[l] += 1

        if pending is None:
            pending = item
        elif pending[0] != item[0]:
            swaps.append([list(pending), list(item)])
            pending = None
        else:
            pending = item

    return swaps


def count_qwen_swappable_params(arch: QwenArch) -> dict:
    """Count the total permutation-supporting subset for Qwen-style blocks."""

    kv_dim = arch.num_key_value_heads * arch.head_dim

    attn_per_layer = (
        arch.hidden_size * arch.hidden_size  # q_proj rows
        + kv_dim * arch.hidden_size  # k_proj rows
        + kv_dim * arch.hidden_size  # v_proj rows
        + arch.hidden_size * arch.hidden_size  # o_proj cols
    )

    mlp_per_layer = (
        arch.intermediate_size * arch.hidden_size  # gate_proj rows
        + arch.intermediate_size * arch.hidden_size  # up_proj rows
        + arch.hidden_size * arch.intermediate_size  # down_proj cols
    )

    total_attn = arch.num_layers * attn_per_layer
    total_mlp = arch.num_layers * mlp_per_layer

    return {
        "total": total_attn + total_mlp,
        "attention": total_attn,
        "mlp": total_mlp,
        "per_layer_attention": attn_per_layer,
        "per_layer_mlp": mlp_per_layer,
    }


def count_qwen_keyed_params(arch: QwenArch, key: PermutationKey) -> dict:
    """Exact keyed parameter count for a concrete key on Qwen-style blocks."""

    rows_per_q_group = arch.q_group_size * arch.head_dim
    per_attn_slot = (
        arch.hidden_size * rows_per_q_group  # q_proj rows
        + arch.hidden_size * arch.head_dim  # k_proj rows
        + arch.hidden_size * arch.head_dim  # v_proj rows
        + arch.hidden_size * rows_per_q_group  # o_proj cols
    )
    per_mlp_slot = 3 * arch.hidden_size

    attn_total = len(key.attn_heads) * 2 * per_attn_slot
    mlp_total = len(key.mlp_cols) * 2 * per_mlp_slot

    return {
        "total": attn_total + mlp_total,
        "attention": attn_total,
        "mlp": mlp_total,
    }


def _qwen_swap_param_sizes(arch: QwenArch) -> tuple[int, int]:
    """Return (params_per_attention_swap, params_per_mlp_swap)."""

    rows_per_q_group = arch.q_group_size * arch.head_dim
    params_per_attn_slot = (
        arch.hidden_size * rows_per_q_group
        + arch.hidden_size * arch.head_dim
        + arch.hidden_size * arch.head_dim
        + arch.hidden_size * rows_per_q_group
    )
    params_per_mlp_slot = 3 * arch.hidden_size
    return 2 * params_per_attn_slot, 2 * params_per_mlp_slot


def _allocate_qwen_swap_counts(
    arch: QwenArch,
    target_pct: float,
    attn_ratio: float,
) -> tuple[int, int]:
    """Allocate Qwen swap counts under the GPT-style attention/MLP split.

    The nominal split is `attn_ratio` attention and `1-attn_ratio` MLP.  At
    small percentages, one Qwen GQA attention swap can be too coarse to fit
    into the attention budget, so that budget is assigned to MLP swaps instead.
    """

    swappable = count_qwen_swappable_params(arch)
    params_per_attn_swap, params_per_mlp_swap = _qwen_swap_param_sizes(arch)

    target_total = int(swappable["total"] * target_pct)
    target_attn = int(target_total * attn_ratio)

    attn_swaps = target_attn // params_per_attn_swap
    if attn_swaps == 0 and target_attn > 0:
        target_mlp = target_total
    else:
        target_mlp = target_total - target_attn

    mlp_swaps = target_mlp // params_per_mlp_swap
    return attn_swaps, mlp_swaps


def generate_qwen_key(
    arch: QwenArch,
    target_pct: float,
    attn_ratio: float = 0.25,
    seed: int = 42,
) -> PermutationKey:
    """Generate one key targeting a percentage of Qwen swappable parameters.

    target_pct is interpreted as fraction of the swappable subset in
    `count_qwen_swappable_params`.
    """

    if not (0.0 < target_pct <= 1.0):
        raise ValueError(f"target_pct must be in (0, 1], got {target_pct}")
    if not (0.0 <= attn_ratio <= 1.0):
        raise ValueError(f"attn_ratio must be in [0, 1], got {attn_ratio}")

    rng = random.Random(seed)
    attn_swaps, mlp_swaps = _allocate_qwen_swap_counts(arch, target_pct, attn_ratio)

    head_pool = [(l, h) for l in range(arch.num_layers) for h in range(arch.num_key_value_heads)]
    col_pool = [(l, c) for l in range(arch.num_layers) for c in range(arch.intermediate_size)]

    rng.shuffle(head_pool)
    rng.shuffle(col_pool)

    attn_head_swaps = _make_cross_layer_swaps(head_pool, attn_swaps)
    mlp_col_swaps = _make_cross_layer_swaps(col_pool, mlp_swaps)

    return PermutationKey(attn_heads=attn_head_swaps, mlp_cols=mlp_col_swaps)


def validate_qwen_key(key: PermutationKey, arch: QwenArch) -> None:
    """Validate index bounds for a Qwen key."""

    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        if not (0 <= layer_a < arch.num_layers and 0 <= layer_b < arch.num_layers):
            raise ValueError(f"Invalid attention layer swap: {swap}")
        if not (0 <= head_a < arch.num_key_value_heads and 0 <= head_b < arch.num_key_value_heads):
            raise ValueError(f"Invalid attention head swap: {swap}")

    for swap in key.mlp_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        if not (0 <= layer_a < arch.num_layers and 0 <= layer_b < arch.num_layers):
            raise ValueError(f"Invalid MLP layer swap: {swap}")
        if not (0 <= col_a < arch.intermediate_size and 0 <= col_b < arch.intermediate_size):
            raise ValueError(f"Invalid MLP column swap: {swap}")
