#!/usr/bin/env python3
"""Generate a permutation key with configurable attention/MLP ratio.

This script generates a key that covers a specified percentage of model weights,
split between attention head swaps and MLP column swaps.

Usage:
    python generate_key.py --output key.json \
        --num_layers 8 --num_heads 8 --hidden_size 512 --mlp_dim 2048 \
        --target_pct 0.20 --attn_ratio 0.25
"""

import argparse
import json
import random
from pathlib import Path


def calculate_weights_per_swap(hidden_size: int, num_heads: int, mlp_dim: int):
    """Calculate number of parameters affected by each swap type.
    
    Attention head swap affects:
    - Q, K, V projections: 3 × hidden_size × head_dim (rows)
    - Output projection: hidden_size × head_dim (columns)
    Total per head: 4 × hidden_size × head_dim
    
    MLP column swap affects:
    - c_fc weight: hidden_size × 1 (one row)
    - c_fc bias: 1
    - c_proj weight: 1 × hidden_size (one column)
    Total per column: 2 × hidden_size + 1
    """
    head_dim = hidden_size // num_heads
    weights_per_head = 4 * hidden_size * head_dim  # Q, K, V, O projections
    weights_per_mlp_col = 2 * hidden_size + 1  # c_fc row + c_fc bias + c_proj col
    return weights_per_head, weights_per_mlp_col


def count_total_params(
    num_layers: int, hidden_size: int, num_heads: int, mlp_dim: int,
    vocab_size: int = 50257, max_positions: int = 2048,
):
    """Count ALL model parameters (GPT-Neo architecture)."""
    # Embeddings
    wte = vocab_size * hidden_size
    wpe = max_positions * hidden_size
    
    # Per layer
    # Attention: Q, K, V, O weights + out_proj bias
    attn_params = 4 * hidden_size * hidden_size + hidden_size
    # MLP: c_fc(weight + bias) + c_proj(weight + bias)
    mlp_params = (hidden_size * mlp_dim + mlp_dim) + (mlp_dim * hidden_size + hidden_size)
    # Layer norms: ln_1 and ln_2 (weight + bias each)
    ln_params = 2 * (hidden_size + hidden_size)
    layer_params = attn_params + mlp_params + ln_params
    
    # Final layer norm
    ln_f = hidden_size + hidden_size
    
    # lm_head is tied with wte (no extra params)
    total = wte + wpe + num_layers * layer_params + ln_f
    return total


def generate_key(
    num_layers: int,
    num_heads: int,
    hidden_size: int,
    mlp_dim: int,
    target_pct: float,
    attn_ratio: float,
    seed: int = 42,
):
    """Generate a permutation key with specified coverage and attention/MLP ratio.
    
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        hidden_size: Model hidden dimension
        mlp_dim: MLP intermediate dimension
        target_pct: Target percentage of weights to cover (0.0 to 1.0)
        attn_ratio: Ratio of keyed weights from attention (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        dict with 'attn_heads' and 'mlp_cols' lists
    """
    random.seed(seed)
    
    weights_per_head, weights_per_mlp_col = calculate_weights_per_swap(
        hidden_size, num_heads, mlp_dim
    )
    
    total_params = count_total_params(num_layers, hidden_size, num_heads, mlp_dim)
    target_keyed_params = int(total_params * target_pct)
    
    # Split between attention and MLP
    target_attn_params = int(target_keyed_params * attn_ratio)
    target_mlp_params = target_keyed_params - target_attn_params
    
    # Calculate number of swaps needed
    # Each swap affects 2 items (we swap A with B), so multiply by 2
    num_head_swaps = target_attn_params // (2 * weights_per_head)
    num_mlp_swaps = target_mlp_params // (2 * weights_per_mlp_col)
    
    print(f"Model config: {num_layers} layers, {num_heads} heads, "
          f"hidden={hidden_size}, mlp={mlp_dim}")
    print(f"Total model params (excl embeddings): {total_params:,}")
    print(f"Target keyed params: {target_keyed_params:,} ({target_pct*100:.1f}%)")
    print(f"  Attention target: {target_attn_params:,} ({attn_ratio*100:.0f}%)")
    print(f"  MLP target: {target_mlp_params:,} ({(1-attn_ratio)*100:.0f}%)")
    print(f"\nWeights per attention head swap: {2 * weights_per_head:,}")
    print(f"Weights per MLP column swap: {2 * weights_per_mlp_col:,}")
    print(f"\nPlanned swaps:")
    print(f"  Attention head swaps: {num_head_swaps}")
    print(f"  MLP column swaps: {num_mlp_swaps}")
    
    # Generate attention head swaps
    # Create list of all possible (layer, head) pairs
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    random.shuffle(all_heads)
    
    attn_swaps = []
    used_heads = set()
    for i in range(0, len(all_heads) - 1, 2):
        if len(attn_swaps) >= num_head_swaps:
            break
        head_a = all_heads[i]
        head_b = all_heads[i + 1]
        # Ensure cross-layer swaps for better mixing
        if head_a[0] != head_b[0]:
            attn_swaps.append([list(head_a), list(head_b)])
            used_heads.add(head_a)
            used_heads.add(head_b)
    
    # Generate MLP column swaps
    all_cols = [(l, c) for l in range(num_layers) for c in range(mlp_dim)]
    random.shuffle(all_cols)
    
    mlp_swaps = []
    used_cols = set()
    for i in range(0, len(all_cols) - 1, 2):
        if len(mlp_swaps) >= num_mlp_swaps:
            break
        col_a = all_cols[i]
        col_b = all_cols[i + 1]
        # Ensure cross-layer swaps
        if col_a[0] != col_b[0]:
            mlp_swaps.append([list(col_a), list(col_b)])
            used_cols.add(col_a)
            used_cols.add(col_b)
    
    # Calculate actual coverage
    actual_attn_params = len(attn_swaps) * 2 * weights_per_head
    actual_mlp_params = len(mlp_swaps) * 2 * weights_per_mlp_col
    actual_total = actual_attn_params + actual_mlp_params
    
    print(f"\nActual swaps generated:")
    print(f"  Attention head swaps: {len(attn_swaps)}")
    print(f"  MLP column swaps: {len(mlp_swaps)}")
    print(f"\nActual keyed params: {actual_total:,} ({100*actual_total/total_params:.2f}%)")
    print(f"  Attention: {actual_attn_params:,} ({100*actual_attn_params/actual_total:.1f}% of keyed)")
    print(f"  MLP: {actual_mlp_params:,} ({100*actual_mlp_params/actual_total:.1f}% of keyed)")
    
    return {
        "attn_heads": attn_swaps,
        "mlp_cols": mlp_swaps,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate permutation key")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--mlp_dim", type=int, default=2048)
    parser.add_argument("--target_pct", type=float, default=0.20,
                        help="Target percentage of weights to key (0-1)")
    parser.add_argument("--attn_ratio", type=float, default=0.25,
                        help="Ratio of keyed weights from attention (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    key = generate_key(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        target_pct=args.target_pct,
        attn_ratio=args.attn_ratio,
        seed=args.seed,
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(key, f, indent=2)
    
    print(f"\nKey saved to: {output_path}")


if __name__ == "__main__":
    main()
