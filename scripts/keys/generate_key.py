#!/usr/bin/env python3
"""Generate one or more permutation keys with configurable coverage.

Keys are guaranteed to be STRICTLY NON-OVERLAPPING: no attention head or MLP
column appears in more than one key. This is enforced by partitioning the
shuffled pool of available slots up front, then drawing each key's swaps from
its dedicated partition.

IMPORTANT:
    - --target_pct is interpreted as a fraction of the SWAPPABLE subset.
    - --target_total_pct is interpreted as a fraction of FULL model params and
      is converted internally to swappable-space using the exact tied/untied
      parameter count convention used by this script.

Single key (backward-compatible):
    python generate_key.py --output key.json \
        --num_layers 8 --num_heads 8 --hidden_size 512 --mlp_dim 2048 \
        --target_pct 0.20 --attn_ratio 0.25

Single key by TOTAL model percentage:
    python generate_key.py --output key.json \
        --num_layers 8 --num_heads 8 --hidden_size 512 --mlp_dim 2048 \
        --target_total_pct 0.10 --attn_ratio 0.25

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


def calculate_weights_per_swap(
    hidden_size: int,
    num_heads: int,
    mlp_dim: int,
    mlp_mode: str = "both",
    attn_mode: str = "full",
):
    """Calculate number of parameters affected by each swap type.

    Attention head swap affects (depends on attn_mode):
    - full: Q/K/V rows + out_proj columns = 4 × hidden_size × head_dim
    - out:  out_proj columns only         = hidden_size × head_dim

    MLP column swap affects (depends on mlp_mode):
    - both: c_fc weight + bias + c_proj weight = 2 × hidden_size + 1
    - up:   c_fc weight + bias = hidden_size + 1
    - down: c_proj weight = hidden_size
    """
    head_dim = hidden_size // num_heads
    if attn_mode == "out":
        weights_per_head = hidden_size * head_dim
    else:
        weights_per_head = 4 * hidden_size * head_dim
    if mlp_mode == "up":
        weights_per_mlp_col = hidden_size + 1
    elif mlp_mode == "down":
        weights_per_mlp_col = hidden_size
    else:
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
    """Count ALL model parameters for GPTNeoForCausalLMTiered.

    This matches model construction in src/tiered/train/utils.py + model/gpt.py:
      - tied mode: lm_head.weight is tied to wte.weight (count once)
      - untied mode: lm_head.weight is separate (count additionally)
      - lm_head.bias is always present and always counted
    """
    # Embeddings
    wte = vocab_size * hidden_size
    wpe = max_positions * hidden_size

    # Per layer
    # Attention: q/k/v/out weights + out bias
    attn_params = 4 * hidden_size * hidden_size + hidden_size
    # MLP: c_fc (weight+bias) + c_proj (weight+bias)
    mlp_params = (hidden_size * mlp_dim + mlp_dim) + (mlp_dim * hidden_size + hidden_size)
    # ln_1 + ln_2, each with weight+bias
    ln_params = 2 * (hidden_size + hidden_size)
    layer_params = attn_params + mlp_params + ln_params

    # Final layer norm
    ln_f = hidden_size + hidden_size

    # LM head:
    # - weight counted only when untied
    # - bias always counted
    lm_head_weight = vocab_size * hidden_size if untie_weights else 0
    lm_head_bias = vocab_size

    total = wte + wpe + num_layers * layer_params + ln_f + lm_head_weight + lm_head_bias
    return total


def convert_total_pct_to_swappable_pct(
    target_total_pct: float,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    mlp_dim: int,
    mlp_mode: str = "both",
    attn_mode: str = "full",
    untie_weights: bool = False,
    context_size: int = 1024,
) -> float:
    """Convert a total-model target percentage to swappable-subset percentage."""
    if target_total_pct < 0.0:
        raise ValueError(f"target_total_pct must be >= 0, got {target_total_pct}")

    total_params = count_total_params(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        max_positions=context_size,
        untie_weights=untie_weights,
    )
    total_swappable = count_total_swappable_params(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        mlp_mode=mlp_mode,
        attn_mode=attn_mode,
    )["total"]
    if total_swappable <= 0:
        raise ValueError("total_swappable must be > 0")

    return target_total_pct * total_params / total_swappable


def count_total_swappable_params(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    mlp_dim: int,
    mlp_mode: str = "both",
    attn_mode: str = "full",
):
    """Count the FULL swappable subset of the model.

    This is the parameter set that could be affected by a 100% key.

    Swappable attention params per layer (depends on attn_mode):
      - full: q_proj + k_proj + v_proj rows + out_proj cols = 4 * hidden_size * hidden_size
      - out:  out_proj cols only                            = hidden_size * hidden_size

    Swappable MLP params per layer (depends on mlp_mode):
      - both: c_fc.weight + c_fc.bias + c_proj.weight = 2*hidden_size*mlp_dim + mlp_dim
      - up:   c_fc.weight + c_fc.bias = hidden_size*mlp_dim + mlp_dim
      - down: c_proj.weight = mlp_dim*hidden_size
    """
    if attn_mode == "out":
        attn_swappable_per_layer = hidden_size * hidden_size
    else:
        attn_swappable_per_layer = 4 * hidden_size * hidden_size
    if mlp_mode == "up":
        mlp_swappable_per_layer = hidden_size * mlp_dim + mlp_dim
    elif mlp_mode == "down":
        mlp_swappable_per_layer = mlp_dim * hidden_size
    else:
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


def _make_random_swaps(pool: list[tuple[int, int]], max_swaps: int):
    """Pair up items from pool into random swaps (same or cross layer).

    Simply takes consecutive pairs from the shuffled pool.

    Args:
        pool: Shuffled list of (layer, index) tuples.
        max_swaps: Maximum number of swaps to generate.

    Returns:
        List of [[layer_a, idx_a], [layer_b, idx_b]] swaps.
    """
    swaps = []
    for i in range(0, len(pool) - 1, 2):
        if len(swaps) >= max_swaps:
            break
        swaps.append([list(pool[i]), list(pool[i + 1])])
    return swaps


def _make_random_cross_layer_swaps(pool: list[tuple[int, int]], max_swaps: int):
    """Pair up items randomly while forbidding same-layer swaps.

    Unlike _make_cross_layer_swaps, this preserves random layer pairing rather
    than imposing a round-robin structure over sorted layers.

    Expects a pool larger than 2*max_swaps so that the random cross-layer
    constraint has room to breathe.  Stops as soon as max_swaps is reached;
    unused slots are simply discarded.

    Args:
        pool: Shuffled list of (layer, index) tuples.
        max_swaps: Maximum number of swaps to generate.

    Returns:
        List of [[layer_a, idx_a], [layer_b, idx_b]] swaps.
    """
    from collections import defaultdict

    buckets = defaultdict(list)
    for item in pool:
        buckets[item[0]].append(item)

    active_layers = [layer for layer, items in buckets.items() if items]
    swaps = []

    while len(active_layers) >= 2 and len(swaps) < max_swaps:
        layer_a = random.choice(active_layers)
        layer_b_choices = [layer for layer in active_layers if layer != layer_a]
        if not layer_b_choices:
            break
        layer_b = random.choice(layer_b_choices)

        item_a = buckets[layer_a].pop()
        item_b = buckets[layer_b].pop()
        swaps.append([list(item_a), list(item_b)])

        if not buckets[layer_a]:
            active_layers.remove(layer_a)
        if layer_b in active_layers and not buckets[layer_b]:
            active_layers.remove(layer_b)

    return swaps


def generate_keys(
    num_keys: int,
    num_layers: int,
    num_heads: int,
    hidden_size: int,
    mlp_dim: int,
    target_pct: float,
    attn_ratio: float,
    mlp_mode: str = "both",
    attn_mode: str = "full",
    untie_weights: bool = False,
    context_size: int = 1024,
    seed: int = 42,
    random_cross_layer_pairing: bool = False,
) -> list[dict]:
    """Generate N non-overlapping permutation keys.

    target_pct is interpreted as the fraction of the SWAPPABLE subset per key.
    """
    if target_pct < 0.0:
        raise ValueError(f"target_pct must be >= 0, got {target_pct}")
    if not (0.0 <= attn_ratio <= 1.0):
        raise ValueError(f"attn_ratio must be in [0, 1], got {attn_ratio}")
    if num_keys < 1:
        raise ValueError(f"num_keys must be >= 1, got {num_keys}")

    random.seed(seed)

    weights_per_head, weights_per_mlp_col = calculate_weights_per_swap(
        hidden_size, num_heads, mlp_dim, mlp_mode=mlp_mode, attn_mode=attn_mode
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
        mlp_mode=mlp_mode,
        attn_mode=attn_mode,
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
    print(f"Attention mode: {attn_mode}")
    print(f"MLP mode: {mlp_mode}")
    if random_cross_layer_pairing:
        print("Cross-layer pairing: random (same-layer swaps forbidden)")
    else:
        print("Cross-layer pairing: structured round-robin")
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

    # Choose attention swap strategy based on mode
    if attn_mode == "out":
        make_attn_swaps = _make_random_swaps
    elif random_cross_layer_pairing:
        make_attn_swaps = _make_random_cross_layer_swaps
    else:
        make_attn_swaps = _make_cross_layer_swaps

    # Determine which key field to use for attention swaps
    attn_key_field = {
        "full": "attn_heads",
        "out": "attn_out_heads",
    }[attn_mode]

    # Shuffle and partition attention head slots.
    # For random cross-layer pairing, over-allocate each partition (2x the
    # needed swap slots) so the random algorithm has slack.  Unused slots
    # are simply discarded after pairing.
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    random.shuffle(all_heads)

    if random_cross_layer_pairing:
        oversample_heads = min(4 * swaps_per_key_attn, len(all_heads) // num_keys)
    else:
        oversample_heads = len(all_heads) // num_keys
    head_partitions = []
    for k in range(num_keys):
        start = k * oversample_heads
        end = min((k + 1) * oversample_heads, len(all_heads))
        head_partitions.append(all_heads[start:end])

    # Shuffle and partition MLP column slots (same oversampling logic).
    all_cols = [(l, c) for l in range(num_layers) for c in range(mlp_dim)]
    random.shuffle(all_cols)

    if random_cross_layer_pairing:
        oversample_cols = min(4 * swaps_per_key_mlp, len(all_cols) // num_keys)
    else:
        oversample_cols = len(all_cols) // num_keys
    col_partitions = []
    for k in range(num_keys):
        start = k * oversample_cols
        end = min((k + 1) * oversample_cols, len(all_cols))
        col_partitions.append(all_cols[start:end])

    # Choose MLP swap strategy based on mode
    if mlp_mode in ("up", "down"):
        make_mlp_swaps = _make_random_swaps
    elif random_cross_layer_pairing:
        make_mlp_swaps = _make_random_cross_layer_swaps
    else:
        make_mlp_swaps = _make_cross_layer_swaps

    # Determine which key field to use for MLP swaps
    mlp_key_field = {
        "both": "mlp_cols",
        "up": "mlp_up_cols",
        "down": "mlp_down_cols",
    }[mlp_mode]

    # Generate swaps within each partition
    keys = []
    total_keyed_all = 0

    print(f"\n{'=' * 60}")
    for k in range(num_keys):
        attn_swaps = make_attn_swaps(head_partitions[k], swaps_per_key_attn)
        mlp_swaps = make_mlp_swaps(col_partitions[k], swaps_per_key_mlp)

        actual_attn = len(attn_swaps) * 2 * weights_per_head
        actual_mlp = len(mlp_swaps) * 2 * weights_per_mlp_col
        actual_total = actual_attn + actual_mlp
        total_keyed_all += actual_total

        print(f"\nKey {k+1}:")
        print(f"  Attention swaps: {len(attn_swaps)} ({actual_attn:,} params)")
        print(f"  MLP swaps ({mlp_mode}): {len(mlp_swaps)} ({actual_mlp:,} params)")
        print(f"  Total keyed:     {actual_total:,}")
        print(f"    = {100 * actual_total / total_swappable:.2f}% of swappable subset")
        print(f"    = {100 * actual_total / total_params:.2f}% of full model")

        key_dict = {attn_key_field: attn_swaps}
        key_dict[mlp_key_field] = mlp_swaps
        keys.append(key_dict)

    print(f"\n{'=' * 60}")
    print(f"Combined keyed params: {total_keyed_all:,}")
    print(f"  = {100 * total_keyed_all / total_swappable:.2f}% of swappable subset")
    print(f"  = {100 * total_keyed_all / total_params:.2f}% of full model")

    _verify_non_overlap(keys, attn_key_field=attn_key_field, mlp_key_field=mlp_key_field)
    return keys


def _verify_non_overlap(
    keys: list[dict],
    attn_key_field: str = "attn_heads",
    mlp_key_field: str = "mlp_cols",
):
    """Assert that no head or column appears in more than one key."""
    all_heads_seen = {}
    all_cols_seen = {}

    for k, key in enumerate(keys):
        attn_fields = [attn_key_field]
        for f in ("attn_heads", "attn_out_heads"):
            if f in key and f not in attn_fields:
                attn_fields.append(f)
        for field in attn_fields:
            for swap in key.get(field, []):
                for slot in swap:
                    slot_tuple = tuple(slot)
                    if slot_tuple in all_heads_seen:
                        raise AssertionError(
                            f"OVERLAP: attention head {slot_tuple} appears in "
                            f"key {all_heads_seen[slot_tuple] + 1} AND key {k + 1}"
                        )
                    all_heads_seen[slot_tuple] = k

        mlp_fields = [mlp_key_field]
        for f in ("mlp_cols", "mlp_up_cols", "mlp_down_cols"):
            if f in key and f not in mlp_fields:
                mlp_fields.append(f)
        for field in mlp_fields:
            for swap in key.get(field, []):
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
            d = json.load(f)
            keys.append(d)
        attn_count = len(d.get("attn_heads", [])) + len(d.get("attn_out_heads", []))
        mlp_count = (
            len(d.get("mlp_cols", []))
            + len(d.get("mlp_up_cols", []))
            + len(d.get("mlp_down_cols", []))
        )
        print(
            f"Loaded {p}: {attn_count} attn swaps, {mlp_count} MLP swaps"
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
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target_pct",
        type=float,
        default=None,
        help="Target fraction of SWAPPABLE weights to key per key (0-1)",
    )
    target_group.add_argument(
        "--target_total_pct",
        type=float,
        default=None,
        help="Target fraction of TOTAL model weights to key per key (0-1). "
             "Converted internally to swappable-space.",
    )
    parser.add_argument(
        "--attn_ratio",
        type=float,
        default=0.25,
        help="Ratio of keyed weights from attention (0-1)",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="full",
        choices=["full", "out"],
        help="Attention swap mode: 'full' (q/k/v + out_proj), "
             "'out' (out_proj columns only, same or cross layer)",
    )
    parser.add_argument(
        "--mlp_mode",
        type=str,
        default="both",
        choices=["both", "up", "down"],
        help="MLP swap mode: 'both' (up+down, cross-layer only), "
             "'up' (c_fc only, any layer), 'down' (c_proj only, any layer)",
    )
    parser.add_argument(
        "--random_cross_layer_pairing",
        action="store_true",
        help="For cross-layer modes only ('attn_mode=full' and/or 'mlp_mode=both'), "
             "pair slots across random different layers instead of the default "
             "structured round-robin pairing. Same-layer swaps remain forbidden.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.validate:
        validate_key_files(args.validate)
        return

    if args.output is None:
        parser.error("--output is required when generating keys")

    # Backward-compatible default: if neither flag is provided, use 0.20 of
    # swappable subset.
    if args.target_pct is None and args.target_total_pct is None:
        target_pct = 0.20
    elif args.target_pct is not None:
        target_pct = args.target_pct
    else:
        target_pct = convert_total_pct_to_swappable_pct(
            target_total_pct=args.target_total_pct,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            mlp_mode=args.mlp_mode,
            attn_mode=args.attn_mode,
            untie_weights=args.untie_weights,
            context_size=args.context_size,
        )
        total_params = count_total_params(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            max_positions=args.context_size,
            untie_weights=args.untie_weights,
        )
        swappable = count_total_swappable_params(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            mlp_mode=args.mlp_mode,
            attn_mode=args.attn_mode,
        )["total"]
        print(
            "Converted target_total_pct -> target_pct: "
            f"{args.target_total_pct:.6f} of total "
            f"({total_params:,} params) => {target_pct:.6f} of swappable "
            f"({swappable:,} params)"
        )

    keys = generate_keys(
        num_keys=args.num_keys,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        target_pct=target_pct,
        attn_ratio=args.attn_ratio,
        attn_mode=args.attn_mode,
        mlp_mode=args.mlp_mode,
        untie_weights=args.untie_weights,
        context_size=args.context_size,
        seed=args.seed,
        random_cross_layer_pairing=args.random_cross_layer_pairing,
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
