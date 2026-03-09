#!/usr/bin/env python3
"""Generate one or more permutation keys with configurable coverage.

Keys are guaranteed to be STRICTLY NON-OVERLAPPING: no attention head or MLP
column appears in more than one key. This is enforced by partitioning the
shuffled pool of available slots up front, then drawing each key's swaps from
its dedicated partition.

IMPORTANT:
    target_pct is now interpreted as a percentage of the SWAPPABLE subset,
    not of the full model size.

Single key (backward-compatible):
    python generate_key.py --output key.json \
        --num_layers 8 --num_heads 8 --hidden_size 512 --mlp_dim 2048 \
        --target_pct 0.20 --attn_ratio 0.25

Multiple non-overlapping keys:
    python generate_key.py --output keys/key.json --num_keys 4 \
        --num_layers 12 --num_heads 8 --hidden_size 512 --mlp_dim 2048 \
        --target_pct 0.15 --attn_ratio 0.25

    Produces: keys/key_1.json, keys/key_2.json, keys/key_3.json, keys/key_4.json

Validation mode (check existing keys):
    python generate_key.py --validate keys/key_1.json keys/key_2.json keys/key_3.json
"""

import argparse
import json
import random
import sys
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
    weights_per_head = 4 * hidden_size * head_dim
    weights_per_mlp_col = 2 * hidden_size + 1
    return weights_per_head, weights_per_mlp_col


def count_total_params(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    mlp_dim: int,
    vocab_size: int = 50257,
    max_positions: int = 1024,
    untie_weights: bool = False,
):
    """Count ALL model parameters (GPT-Neo architecture)."""
    # Embeddings
    wte = vocab_size * hidden_size
    wpe = max_positions * hidden_size

    # Per layer
    # Note: this matches the repo's approximate accounting convention.
    attn_params = 4 * hidden_size * hidden_size + hidden_size
    mlp_params = (hidden_size * mlp_dim + mlp_dim) + (mlp_dim * hidden_size + hidden_size)
    ln_params = 2 * (hidden_size + hidden_size)
    layer_params = attn_params + mlp_params + ln_params

    # Final layer norm
    ln_f = hidden_size + hidden_size

    # Untied LM head weight only (kept as-is for backward compatibility with
    # the old script's accounting convention)
    lm_head = vocab_size * hidden_size if untie_weights else 0

    total = wte + wpe + num_layers * layer_params + ln_f + lm_head
    return total


def count_total_swappable_params(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    mlp_dim: int,
):
    """Count the FULL swappable subset of the model.

    This is the parameter set that could be affected by a 100% key.

    Swappable attention params per layer:
      - q_proj.weight: all rows
      - k_proj.weight: all rows
      - v_proj.weight: all rows
      - out_proj.weight: all columns
      = 4 * hidden_size * hidden_size

    Swappable MLP params per layer:
      - c_fc.weight: hidden_size * mlp_dim
      - c_fc.bias: mlp_dim
      - c_proj.weight: mlp_dim * hidden_size
      = 2 * hidden_size * mlp_dim + mlp_dim
    """
    attn_swappable_per_layer = 4 * hidden_size * hidden_size
    mlp_swappable_per_layer = 2 * hidden_size * mlp_dim + mlp_dim

    total_attn = num_layers * attn_swappable_per_layer
    total_mlp = num_layers * mlp_swappable_per_layer
    total = total_attn + total_mlp

    return {
        "total": total,
        "attention": total_attn,
        "mlp": total_mlp,
        "per_layer_attention": attn_swappable_per_layer,
        "per_layer_mlp": mlp_swappable_per_layer,
    }


def _make_cross_layer_swaps(pool: list[tuple[int, int]], max_swaps: int):
    """Pair up items from pool into cross-layer swaps.

    Each swap pairs (layer_a, idx_a) with (layer_b, idx_b) where
    layer_a != layer_b. Items that can't form a cross-layer pair are
    left unused.

    Args:
        pool: Shuffled list of (layer, index) tuples.
        max_swaps: Maximum number of swaps to generate.

    Returns:
        List of [[layer_a, idx_a], [layer_b, idx_b]] swaps.
    """
    swaps = []
    from collections import defaultdict

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


def generate_keys(
    num_keys: int,
    num_layers: int,
    num_heads: int,
    hidden_size: int,
    mlp_dim: int,
    target_pct: float,
    attn_ratio: float,
    untie_weights: bool = False,
    context_size: int = 1024,
    seed: int = 42,
) -> list[dict]:
    """Generate N non-overlapping permutation keys.

    target_pct is interpreted as the fraction of the SWAPPABLE subset per key.
    """
    random.seed(seed)

    weights_per_head, weights_per_mlp_col = calculate_weights_per_swap(
        hidden_size, num_heads, mlp_dim
    )

    total_params = count_total_params(
        num_layers,
        hidden_size,
        num_heads,
        mlp_dim,
        max_positions=context_size,
        untie_weights=untie_weights,
    )

    swappable_info = count_total_swappable_params(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
    )
    total_swappable = swappable_info["total"]
    total_swappable_attn = swappable_info["attention"]
    total_swappable_mlp = swappable_info["mlp"]

    target_keyed_per_key = int(total_swappable * target_pct)
    target_attn_per_key = int(target_keyed_per_key * attn_ratio)
    target_mlp_per_key = target_keyed_per_key - target_attn_per_key

    # Per-key swap budgets
    swaps_per_key_attn = target_attn_per_key // (2 * weights_per_head)
    swaps_per_key_mlp = target_mlp_per_key // (2 * weights_per_mlp_col)

    # Total slots available
    total_head_slots = num_layers * num_heads
    total_mlp_slots = num_layers * mlp_dim

    # Each swap consumes 2 slots
    needed_head_slots = 2 * swaps_per_key_attn * num_keys
    needed_mlp_slots = 2 * swaps_per_key_mlp * num_keys

    print(f"Model config: {num_layers} layers, {num_heads} heads, hidden={hidden_size}, mlp={mlp_dim}")
    print(f"Total model params:        {total_params:,} (untied={untie_weights})")
    print(f"Total swappable params:    {total_swappable:,}")
    print(f"  - attention swappable:   {total_swappable_attn:,}")
    print(f"  - mlp swappable:         {total_swappable_mlp:,}")
    print(f"Generating {num_keys} non-overlapping key(s)")

    print(f"\nPer-key budget ({target_pct*100:.1f}% of swappable params = {target_keyed_per_key:,}):")
    print(f"  Attention target: {target_attn_per_key:,} params")
    print(f"  MLP target:       {target_mlp_per_key:,} params")
    print(f"  Attention swaps:  {swaps_per_key_attn} (needs {2 * swaps_per_key_attn} head slots)")
    print(f"  MLP swaps:        {swaps_per_key_mlp} (needs {2 * swaps_per_key_mlp} col slots)")

    print(f"\nTotal slots needed across {num_keys} keys:")
    print(
        f"  Attention heads: {needed_head_slots} / {total_head_slots} available "
        f"({100 * needed_head_slots / total_head_slots:.1f}%)"
    )
    print(
        f"  MLP columns:     {needed_mlp_slots} / {total_mlp_slots} available "
        f"({100 * needed_mlp_slots / total_mlp_slots:.1f}%)"
    )

    if needed_head_slots > total_head_slots:
        print(
            f"\n*** WARNING: Not enough attention head slots! "
            f"Need {needed_head_slots} but only {total_head_slots} exist."
        )
        print("    Either reduce --target_pct, --attn_ratio, or --num_keys.")
        print("    Will allocate as many as possible.\n")

    if needed_mlp_slots > total_mlp_slots:
        print(
            f"\n*** WARNING: Not enough MLP column slots! "
            f"Need {needed_mlp_slots} but only {total_mlp_slots} exist."
        )
        print("    Either reduce --target_pct, increase --attn_ratio, or reduce --num_keys.")
        print("    Will allocate as many as possible.\n")

    # Shuffle and partition attention head slots
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    random.shuffle(all_heads)

    chunk_size_heads = len(all_heads) // num_keys
    head_partitions = []
    for k in range(num_keys):
        start = k * chunk_size_heads
        end = (k + 1) * chunk_size_heads if k < num_keys - 1 else len(all_heads)
        head_partitions.append(all_heads[start:end])

    # Shuffle and partition MLP column slots
    all_cols = [(l, c) for l in range(num_layers) for c in range(mlp_dim)]
    random.shuffle(all_cols)

    chunk_size_cols = len(all_cols) // num_keys
    col_partitions = []
    for k in range(num_keys):
        start = k * chunk_size_cols
        end = (k + 1) * chunk_size_cols if k < num_keys - 1 else len(all_cols)
        col_partitions.append(all_cols[start:end])

    # Generate swaps within each partition
    keys = []
    total_keyed_all = 0

    print(f"\n{'=' * 60}")
    for k in range(num_keys):
        attn_swaps = _make_cross_layer_swaps(head_partitions[k], swaps_per_key_attn)
        mlp_swaps = _make_cross_layer_swaps(col_partitions[k], swaps_per_key_mlp)

        actual_attn = len(attn_swaps) * 2 * weights_per_head
        actual_mlp = len(mlp_swaps) * 2 * weights_per_mlp_col
        actual_total = actual_attn + actual_mlp
        total_keyed_all += actual_total

        print(f"\nKey {k+1}:")
        print(f"  Attention swaps: {len(attn_swaps)} ({actual_attn:,} params)")
        print(f"  MLP swaps:       {len(mlp_swaps)} ({actual_mlp:,} params)")
        print(f"  Total keyed:     {actual_total:,}")
        print(f"    = {100 * actual_total / total_swappable:.2f}% of swappable subset")
        print(f"    = {100 * actual_total / total_params:.2f}% of full model")

        keys.append({
            "attn_heads": attn_swaps,
            "mlp_cols": mlp_swaps,
        })

    print(f"\n{'=' * 60}")
    print(f"Combined keyed params: {total_keyed_all:,}")
    print(f"  = {100 * total_keyed_all / total_swappable:.2f}% of swappable subset")
    print(f"  = {100 * total_keyed_all / total_params:.2f}% of full model")

    _verify_non_overlap(keys)
    return keys


def _verify_non_overlap(keys: list[dict]):
    """Assert that no head or column appears in more than one key."""
    all_heads_seen = {}
    all_cols_seen = {}

    for k, key in enumerate(keys):
        for swap in key["attn_heads"]:
            for slot in swap:
                slot_tuple = tuple(slot)
                if slot_tuple in all_heads_seen:
                    raise AssertionError(
                        f"OVERLAP: attention head {slot_tuple} appears in "
                        f"key {all_heads_seen[slot_tuple] + 1} AND key {k + 1}"
                    )
                all_heads_seen[slot_tuple] = k

        for swap in key["mlp_cols"]:
            for slot in swap:
                slot_tuple = tuple(slot)
                if slot_tuple in all_cols_seen:
                    raise AssertionError(
                        f"OVERLAP: MLP column {slot_tuple} appears in "
                        f"key {all_cols_seen[slot_tuple] + 1} AND key {k + 1}"
                    )
                all_cols_seen[slot_tuple] = k

    print(f"\nNon-overlap verification PASSED:")
    print(f"  {len(all_heads_seen)} unique attention head slots across {len(keys)} keys")
    print(f"  {len(all_cols_seen)} unique MLP column slots across {len(keys)} keys")


def validate_key_files(paths: list[str]):
    """Load key files from disk and verify they don't overlap."""
    keys = []
    for p in paths:
        with open(p) as f:
            keys.append(json.load(f))
        print(
            f"Loaded {p}: {len(keys[-1]['attn_heads'])} attn swaps, "
            f"{len(keys[-1]['mlp_cols'])} MLP swaps"
        )

    try:
        _verify_non_overlap(keys)
        print("\nAll keys are non-overlapping.")
    except AssertionError as e:
        print(f"\nOVERLAP DETECTED: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate one or more non-overlapping permutation keys"
    )

    # Mode
    parser.add_argument(
        "--validate",
        type=str,
        nargs="+",
        default=None,
        help="Validate existing key files for non-overlap (ignores all other flags)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. For num_keys=1: exact path. "
             "For num_keys>1: stem is suffixed with _1, _2, ...",
    )
    parser.add_argument(
        "--num_keys",
        type=int,
        default=1,
        help="Number of non-overlapping keys to generate",
    )

    # Model architecture
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--mlp_dim", type=int, default=2048)
    parser.add_argument("--untie_weights", action="store_true", help="Whether embeddings are untied")
    parser.add_argument("--context_size", type=int, default=1024)

    # Key configuration
    parser.add_argument(
        "--target_pct",
        type=float,
        default=0.20,
        help="Target percentage of SWAPPABLE weights to key PER KEY (0-1)",
    )
    parser.add_argument(
        "--attn_ratio",
        type=float,
        default=0.25,
        help="Ratio of keyed weights from attention (0-1)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.validate:
        validate_key_files(args.validate)
        return

    if args.output is None:
        parser.error("--output is required when generating keys")

    keys = generate_keys(
        num_keys=args.num_keys,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        target_pct=args.target_pct,
        attn_ratio=args.attn_ratio,
        untie_weights=args.untie_weights,
        context_size=args.context_size,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(keys) == 1:
        with open(output_path, "w") as f:
            json.dump(keys[0], f, indent=2)
        print(f"\nKey saved to: {output_path}")
    else:
        saved = []
        for k, key in enumerate(keys):
            suffixed = output_path.parent / f"{output_path.stem}_{k+1}{output_path.suffix}"
            with open(suffixed, "w") as f:
                json.dump(key, f, indent=2)
            saved.append(str(suffixed))

        print(f"\n{len(keys)} keys saved:")
        for p in saved:
            print(f"  {p}")
        print(f"\nUsage with pretrain_ntier.py:")
        print(f"  --key_paths {' '.join(saved)}")


if __name__ == "__main__":
    main()