#!/usr/bin/env python3
"""Benchmark apply_permutation speed across model sizes on a single GPU.

For each model size, generates a key (default: 5% of swappable params),
builds a minimal GPTNeo-style model containing just the weight tensors the
permutation hot path touches (q/k/v/out_proj, c_fc, c_proj per layer), then
times apply_permutation with CUDA events.

Designed for an H100 80GB. Sizes:
  180M (matches the project's "150m" config)
  680M (matches "530m")
  1B   (matches "1b")
  30B  (largest that fits comfortably in bf16)

Usage:
    python scripts/keys/benchmark_permutation_speed.py
    python scripts/keys/benchmark_permutation_speed.py --target_pct 0.05 --trials 20
    python scripts/keys/benchmark_permutation_speed.py --sizes 180M 1B
"""

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from generate_key import generate_keys  # noqa: E402
from tiered.permutation.key import PermutationKey  # noqa: E402
from tiered.permutation.permute import apply_permutation, build_swap_plan  # noqa: E402


# -----------------------------------------------------------------------------
# Model configs. Smaller three match the project's existing arch settings.
# 30B is a LLaMA-30B-style scale-up that fits in bf16 on an 80 GB H100.
# -----------------------------------------------------------------------------

CONFIGS = {
    "180M": dict(num_layers=12, num_heads=12, hidden_size=768,  mlp_dim=2368),
    "680M": dict(num_layers=16, num_heads=16, hidden_size=1344, mlp_dim=6456),
    "1B":   dict(num_layers=16, num_heads=16, hidden_size=1664, mlp_dim=13824),
    "30B":  dict(num_layers=60, num_heads=52, hidden_size=6656, mlp_dim=17920),
}


# -----------------------------------------------------------------------------
# Minimal model whose structure satisfies _get_attention_module / _get_mlp_module:
#   model.transformer.h[i].attn.attention.{q_proj, k_proj, v_proj, out_proj, head_dim}
#   model.transformer.h[i].mlp.{c_fc, c_proj}
# Embeddings, norms, lm_head, etc. are omitted because the permutation hot path
# never touches them.
# -----------------------------------------------------------------------------

class _Attn(nn.Module):
    def __init__(self, num_heads, head_dim, hidden, dtype, device):
        super().__init__()
        self.head_dim = head_dim
        out = num_heads * head_dim
        self.q_proj = nn.Linear(hidden, out, bias=False, dtype=dtype, device=device)
        self.k_proj = nn.Linear(hidden, out, bias=False, dtype=dtype, device=device)
        self.v_proj = nn.Linear(hidden, out, bias=False, dtype=dtype, device=device)
        self.out_proj = nn.Linear(out, hidden, bias=False, dtype=dtype, device=device)


class _MLP(nn.Module):
    def __init__(self, hidden, mlp_dim, dtype, device):
        super().__init__()
        self.c_fc = nn.Linear(hidden, mlp_dim, bias=True, dtype=dtype, device=device)
        self.c_proj = nn.Linear(mlp_dim, hidden, bias=False, dtype=dtype, device=device)


class _AttnWrap(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attention = attn


class _Block(nn.Module):
    def __init__(self, attn, mlp):
        super().__init__()
        self.attn = _AttnWrap(attn)
        self.mlp = mlp


class _Transformer(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.h = nn.ModuleList(blocks)


class _Model(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.transformer = _Transformer(blocks)


def build_model(cfg, dtype, device):
    assert cfg["hidden_size"] % cfg["num_heads"] == 0, \
        f"hidden_size {cfg['hidden_size']} not divisible by num_heads {cfg['num_heads']}"
    head_dim = cfg["hidden_size"] // cfg["num_heads"]
    blocks = [
        _Block(
            _Attn(cfg["num_heads"], head_dim, cfg["hidden_size"], dtype, device),
            _MLP(cfg["hidden_size"], cfg["mlp_dim"], dtype, device),
        )
        for _ in range(cfg["num_layers"])
    ]
    return _Model(blocks)


def benchmark(name, cfg, target_pct, attn_ratio, device, dtype, n_warmup, n_trials, verbose_keys):
    print(f"\n=== {name}: layers={cfg['num_layers']} heads={cfg['num_heads']} "
          f"hidden={cfg['hidden_size']} mlp={cfg['mlp_dim']} ===")

    if verbose_keys:
        keys_raw = generate_keys(num_keys=1, target_pct=target_pct,
                                 attn_ratio=attn_ratio, seed=0, **cfg)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            keys_raw = generate_keys(num_keys=1, target_pct=target_pct,
                                     attn_ratio=attn_ratio, seed=0, **cfg)
    key = PermutationKey.from_dict(keys_raw[0])
    print(f"  key: {len(key.attn_heads)} attn-head swaps, {len(key.mlp_cols)} mlp-col swaps")

    model = build_model(cfg, dtype, device)
    if device.type == "cuda":
        alloc_gb = torch.cuda.memory_allocated(device) / 1024**3
        print(f"  model bf16 allocation: {alloc_gb:.2f} GB")

    plan = build_swap_plan(model, key, device)

    for _ in range(n_warmup):
        apply_permutation(model, key, plan=plan)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_trials):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            apply_permutation(model, key, plan=plan)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            apply_permutation(model, key, plan=plan)
            times_ms.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times_ms) / len(times_ms)
    print(f"  apply_permutation: mean={mean_ms:.3f} ms, "
          f"min={min(times_ms):.3f} ms, max={max(times_ms):.3f} ms")

    del plan, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": name,
        "mean_ms": mean_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "n_attn_swaps": len(key.attn_heads),
        "n_mlp_swaps": len(key.mlp_cols),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_pct", type=float, default=0.05,
                   help="Fraction of swappable params per key (default 0.05 = 5%%).")
    p.add_argument("--attn_ratio", type=float, default=0.25,
                   help="Fraction of keyed weights from attention (default 0.25).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--sizes", nargs="+", default=list(CONFIGS.keys()),
                   help=f"Subset of sizes to benchmark. Available: {list(CONFIGS.keys())}")
    p.add_argument("--verbose_keys", action="store_true",
                   help="Show the verbose key generator output for each size.")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"device={device}, dtype={args.dtype}, target_pct={args.target_pct}, "
          f"attn_ratio={args.attn_ratio}, warmup={args.warmup}, trials={args.trials}")

    results = []
    for name in args.sizes:
        if name not in CONFIGS:
            print(f"[skip] unknown size: {name}")
            continue
        try:
            results.append(benchmark(
                name, CONFIGS[name], args.target_pct, args.attn_ratio,
                device, dtype, args.warmup, args.trials, args.verbose_keys,
            ))
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] {name}: skipping")
            torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    header = f"{'size':<6} {'mean (ms)':<12} {'min (ms)':<12} {'max (ms)':<12} {'attn swaps':<12} {'mlp swaps':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<6} {r['mean_ms']:<12.3f} {r['min_ms']:<12.3f} {r['max_ms']:<12.3f} "
              f"{r['n_attn_swaps']:<12} {r['n_mlp_swaps']:<12}")


if __name__ == "__main__":
    main()
