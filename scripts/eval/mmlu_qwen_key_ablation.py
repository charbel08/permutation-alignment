#!/usr/bin/env python3
"""MMLU key-destruction ablation for Qwen models (no training).

Evaluates a pretrained Qwen/Llama-style CausalLM on MMLU after applying
permutation keys of increasing size.  Measures how fast accuracy collapses
as the fraction of scrambled parameters grows.

KEY DESIGN: All key sizes use the SAME random pool shuffle.  The 10% key is
a strict prefix of the 15% key, which is a strict prefix of the 20% key, etc.
This ensures the destruction curve is monotonic and measures cumulative
coverage, not random-draw variance.

MULTI-GPU: Launch with torchrun for data-parallel evaluation. Each rank loads
the model on its own GPU and evaluates a shard of the examples.  Results are
all-reduced before metric computation.

Usage (single GPU):
    PYTHONPATH=./src python scripts/eval/mmlu_qwen_key_ablation.py \
        --model_id Qwen/Qwen3-8B

Usage (8 GPUs):
    PYTHONPATH=./src torchrun --standalone --nproc_per_node=8 \
        scripts/eval/mmlu_qwen_key_ablation.py \
        --model_id Qwen/Qwen3-8B

Quick dev run:
    PYTHONPATH=./src python scripts/eval/mmlu_qwen_key_ablation.py \
        --model_id Qwen/Qwen3-8B \
        --key_pcts 0.05 0.25 0.50 1.00 \
        --max_examples_per_subject 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiered.permutation.key import PermutationKey
from tiered.permutation.qwen import (
    QwenArch,
    _make_cross_layer_swaps,
    apply_qwen_permutation,
    count_qwen_keyed_params,
    count_qwen_swappable_params,
    generate_qwen_key,
    get_qwen_arch,
    unapply_qwen_permutation,
    validate_qwen_key,
)


ANSWER_LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, bool]:
    """Return (rank, world_size, is_distributed)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1, False
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), True


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def allreduce_sum(tensor: torch.Tensor, is_distributed: bool) -> torch.Tensor:
    if not is_distributed:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ---------------------------------------------------------------------------
# Nested key generation
# ---------------------------------------------------------------------------

def generate_nested_keys(
    arch: QwenArch,
    key_pcts: list[float],
    attn_ratio: float = 0.25,
    seed: int = 42,
) -> list[PermutationKey]:
    """Generate keys where key(p1) is a subset of key(p2) for p1 < p2.

    Shuffles pools once, generates max-budget swaps, takes prefixes.
    """
    if not key_pcts:
        return []

    rng = random.Random(seed)

    rows_per_q_group = arch.q_group_size * arch.head_dim
    params_per_attn_slot = (
        arch.hidden_size * rows_per_q_group
        + arch.hidden_size * arch.head_dim
        + arch.hidden_size * arch.head_dim
        + arch.hidden_size * rows_per_q_group
    )
    params_per_mlp_slot = 3 * arch.hidden_size

    swappable = count_qwen_swappable_params(arch)

    head_pool = [(l, h) for l in range(arch.num_layers) for h in range(arch.num_key_value_heads)]
    col_pool = [(l, c) for l in range(arch.num_layers) for c in range(arch.intermediate_size)]
    rng.shuffle(head_pool)
    rng.shuffle(col_pool)

    max_pct = max(key_pcts)
    max_target = int(swappable["total"] * max_pct)
    max_attn_target = int(max_target * attn_ratio)
    max_mlp_target = max_target - max_attn_target
    max_attn_swaps = max_attn_target // (2 * params_per_attn_slot)
    max_mlp_swaps = max_mlp_target // (2 * params_per_mlp_slot)

    all_attn_swaps = _make_cross_layer_swaps(head_pool, max_attn_swaps)
    all_mlp_swaps = _make_cross_layer_swaps(col_pool, max_mlp_swaps)

    keys = []
    for pct in key_pcts:
        target = int(swappable["total"] * pct)
        attn_target = int(target * attn_ratio)
        mlp_target = target - attn_target
        n_attn = min(attn_target // (2 * params_per_attn_slot), len(all_attn_swaps))
        n_mlp = min(mlp_target // (2 * params_per_mlp_slot), len(all_mlp_swaps))
        keys.append(PermutationKey(
            attn_heads=all_attn_swaps[:n_attn],
            mlp_cols=all_mlp_swaps[:n_mlp],
        ))

    return keys


# ---------------------------------------------------------------------------
# MMLU data + evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalExample:
    subject: str
    question: str
    choices: List[str]
    answer_idx: int


def format_example(question: str, choices: List[str], answer: str | None = None) -> str:
    lines = [question]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append(f"Answer: {answer}" if answer else "Answer:")
    return "\n".join(lines)


def build_prompt(subject: str, dev_examples: List[EvalExample],
                 test_ex: EvalExample, shots: int) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    k = min(shots, len(dev_examples))
    fewshot = "\n\n".join(
        format_example(e.question, e.choices, ANSWER_LETTERS[e.answer_idx])
        for e in dev_examples[:k]
    )
    test_block = format_example(test_ex.question, test_ex.choices)
    return f"{header}{fewshot}\n\n{test_block}" if fewshot else f"{header}{test_block}"


def _logprob_from_ids(
    model, prompt_ids: List[int], cont_ids: List[int],
    device: torch.device, max_context_len: int | None,
) -> float:
    if not prompt_ids or not cont_ids:
        raise ValueError("Empty prompt or continuation IDs")
    if max_context_len is not None and len(prompt_ids) + len(cont_ids) > max_context_len:
        keep = max_context_len - len(cont_ids)
        if keep <= 0:
            raise ValueError("Continuation longer than max context")
        prompt_ids = prompt_ids[-keep:]

    input_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids).logits

    prompt_len = len(prompt_ids)
    cont_len = len(cont_ids)
    pred_slice = logits[0, prompt_len - 1: prompt_len - 1 + cont_len, :]
    target = input_ids[0, prompt_len: prompt_len + cont_len]
    lp = torch.log_softmax(pred_slice, dim=-1).gather(1, target.unsqueeze(-1)).squeeze(-1)
    return float(lp.sum().item())


def predict_choice(model, tokenizer, prompt: str, device: torch.device,
                   max_context_len: int | None) -> int:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Empty prompt tokenization")
    scores = []
    for letter in ANSWER_LETTERS:
        cont_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        scores.append(_logprob_from_ids(model, prompt_ids, cont_ids, device, max_context_len))
    return int(max(range(len(scores)), key=lambda i: scores[i]))


def load_mmlu_examples(
    dataset_name: str, subjects: List[str] | None,
    max_examples_per_subject: int, seed: int,
) -> tuple[Dict[str, List[EvalExample]], Dict[str, List[EvalExample]]]:
    test_ds = load_dataset(dataset_name, "all", split="test")
    dev_ds = load_dataset(dataset_name, "all", split="dev")

    def row_to_example(row) -> EvalExample:
        return EvalExample(row["subject"], row["question"], list(row["choices"]), int(row["answer"]))

    dev_by_subject: Dict[str, List[EvalExample]] = {}
    test_by_subject: Dict[str, List[EvalExample]] = {}

    for row in dev_ds:
        ex = row_to_example(row)
        dev_by_subject.setdefault(ex.subject, []).append(ex)
    for row in test_ds:
        ex = row_to_example(row)
        test_by_subject.setdefault(ex.subject, []).append(ex)

    available = sorted(test_by_subject.keys())
    selected = subjects if subjects else available
    unknown = [s for s in selected if s not in test_by_subject]
    if unknown:
        raise ValueError(f"Unknown MMLU subjects: {unknown}")

    rng = random.Random(seed)
    sel_dev: Dict[str, List[EvalExample]] = {}
    sel_test: Dict[str, List[EvalExample]] = {}

    for subj in selected:
        dev_exs = list(dev_by_subject.get(subj, []))
        test_exs = list(test_by_subject[subj])
        rng.shuffle(dev_exs)
        rng.shuffle(test_exs)
        if max_examples_per_subject > 0:
            test_exs = test_exs[:max_examples_per_subject]
        sel_dev[subj] = dev_exs
        sel_test[subj] = test_exs

    return sel_dev, sel_test


def _flatten_examples(
    dev_by_subject: Dict[str, List[EvalExample]],
    test_by_subject: Dict[str, List[EvalExample]],
) -> tuple[List[EvalExample], Dict[str, List[EvalExample]]]:
    """Flatten test examples into a deterministic list."""
    flat = []
    for subject in sorted(test_by_subject.keys()):
        flat.extend(test_by_subject[subject])
    return flat, dev_by_subject


def _shard_examples(examples: List[EvalExample], rank: int, world_size: int) -> List[EvalExample]:
    """Deterministic round-robin sharding."""
    return examples[rank::world_size]


def _wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p_hat = correct / total
    denom = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def evaluate_mmlu(
    model, tokenizer,
    dev_by_subject: Dict[str, List[EvalExample]],
    flat_examples: List[EvalExample],
    shots: int, device: torch.device, max_context_len: int | None,
    rank: int, world_size: int, is_distributed: bool,
    desc: str = "",
) -> dict:
    """Evaluate MMLU with distributed sharding.

    Each rank evaluates its shard.  Correct/total counts are all-reduced
    so every rank gets the global result.
    """
    model.eval()
    shard = _shard_examples(flat_examples, rank, world_size)

    local_correct: Dict[str, int] = {}
    local_total: Dict[str, int] = {}

    show_progress = (rank == 0)
    pbar = tqdm(total=len(shard), desc=desc or "MMLU", leave=False) if show_progress else None

    for ex in shard:
        prompt = build_prompt(ex.subject, dev_by_subject.get(ex.subject, []), ex, shots)
        pred = predict_choice(model, tokenizer, prompt, device, max_context_len)
        local_correct[ex.subject] = local_correct.get(ex.subject, 0) + (1 if pred == ex.answer_idx else 0)
        local_total[ex.subject] = local_total.get(ex.subject, 0) + 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # All-reduce per-subject counts
    all_subjects = sorted(set(ex.subject for ex in flat_examples))
    subj_to_idx = {s: i for i, s in enumerate(all_subjects)}
    n_subj = len(all_subjects)

    correct_tensor = torch.zeros(n_subj, dtype=torch.long, device=device)
    total_tensor = torch.zeros(n_subj, dtype=torch.long, device=device)

    for subj, c in local_correct.items():
        correct_tensor[subj_to_idx[subj]] = c
    for subj, t in local_total.items():
        total_tensor[subj_to_idx[subj]] = t

    correct_tensor = allreduce_sum(correct_tensor, is_distributed)
    total_tensor = allreduce_sum(total_tensor, is_distributed)

    # Compute global metrics
    subject_metrics = {}
    correct_total = 0
    count_total = 0

    for subj in all_subjects:
        idx = subj_to_idx[subj]
        c = int(correct_tensor[idx].item())
        n = int(total_tensor[idx].item())
        acc = c / n if n else 0.0
        ci_lo, ci_hi = _wilson_ci(c, n)
        subject_metrics[subj] = {
            "correct": c, "total": n, "accuracy": acc,
            "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
        }
        correct_total += c
        count_total += n

    macro_acc = (sum(m["accuracy"] for m in subject_metrics.values()) / len(subject_metrics)
                 if subject_metrics else 0.0)
    micro_acc = correct_total / count_total if count_total else 0.0
    micro_ci_lo, micro_ci_hi = _wilson_ci(correct_total, count_total)

    return {
        "micro_accuracy": micro_acc,
        "micro_ci_95_lo": micro_ci_lo,
        "micro_ci_95_hi": micro_ci_hi,
        "macro_accuracy": macro_acc,
        "total_correct": correct_total,
        "total_count": count_total,
        "subjects": subject_metrics,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_eval_summary(name: str, m: dict) -> None:
    ci = f"[{m['micro_ci_95_lo']:.4f}, {m['micro_ci_95_hi']:.4f}]"
    print(f"  [{name}] micro={m['micro_accuracy']:.4f} {ci}  "
          f"macro={m['macro_accuracy']:.4f}  ({m['total_correct']}/{m['total_count']})")


def save_outputs(results: dict, output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "name", "key_pct", "keyed_pct_of_swappable",
        "micro_accuracy", "micro_ci_95_lo", "micro_ci_95_hi", "macro_accuracy",
        "total_correct", "total_count",
        "attn_swaps", "mlp_swaps", "keyed_params",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["rows"]:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def maybe_init_wandb(args, config: dict):
    if not args.wandb:
        return None
    import wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=config,
    )
    wandb.define_metric("ablation/key_pct")
    wandb.define_metric("ablation/*", step_metric="ablation/key_pct")
    return run


def log_wandb_row(row: dict) -> None:
    import wandb
    wandb.log({f"ablation/{k}": v for k, v in row.items()})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen MMLU key-destruction ablation (multi-GPU capable)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--tokenizer_id", type=str, default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", type=str, default="cuda",
                   help="Ignored in distributed mode (auto from LOCAL_RANK)")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])

    key_group = p.add_argument_group("key sweep")
    key_group.add_argument("--key_pcts", type=float, nargs="+", default=None,
                           help="Explicit list (overrides range)")
    key_group.add_argument("--min_pct", type=float, default=0.05)
    key_group.add_argument("--max_pct", type=float, default=1.00)
    key_group.add_argument("--step_pct", type=float, default=0.05)
    key_group.add_argument("--attn_ratio", type=float, default=0.25)
    key_group.add_argument("--seed", type=int, default=42)

    mmlu_group = p.add_argument_group("MMLU evaluation")
    mmlu_group.add_argument("--shots", type=int, default=5)
    mmlu_group.add_argument("--max_examples_per_subject", type=int, default=-1,
                            help="Cap per subject (-1 = full). 20 for dev runs.")
    mmlu_group.add_argument("--subjects", type=str, nargs="*", default=None)
    mmlu_group.add_argument("--dataset_name", type=str, default="cais/mmlu")

    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--output_csv", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="outputs")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="tiered-alignment-ablation")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)

    args = p.parse_args()

    if args.key_pcts is None:
        pcts = []
        cur = args.min_pct
        while cur <= args.max_pct + 1e-9:
            pcts.append(round(cur, 6))
            cur += args.step_pct
        if not pcts:
            p.error("No key percentages generated.")
        args.key_pcts = pcts

    for pct in args.key_pcts:
        if not (0.0 < pct <= 1.0):
            p.error(f"key_pct must be in (0, 1], got {pct}")
    args.key_pcts = sorted(set(args.key_pcts))

    if args.run_name is None:
        model_short = args.model_id.split("/")[-1]
        cap = f"_cap{args.max_examples_per_subject}" if args.max_examples_per_subject > 0 else ""
        args.run_name = f"key_destruction_{model_short}{cap}"
    if args.output_json is None:
        args.output_json = f"{args.output_dir}/{args.run_name}.json"
    if args.output_csv is None:
        args.output_csv = f"{args.output_dir}/{args.run_name}.csv"

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    start = time.time()

    rank, world_size, is_distributed = setup_distributed()
    is_main = (rank == 0)

    if is_distributed:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    else:
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        if is_main:
            print("CPU detected; overriding dtype to float32")
        dtype = torch.float32

    if is_main:
        print(f"Loading model: {args.model_id}" +
              (f" on {world_size} GPUs" if is_distributed else ""))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id or args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.eval()

    arch = get_qwen_arch(model)
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    swappable = count_qwen_swappable_params(arch)

    if is_main:
        print(f"  layers={arch.num_layers} hidden={arch.hidden_size} "
              f"heads={arch.num_attention_heads} kv_heads={arch.num_key_value_heads} "
              f"intermediate={arch.intermediate_size}")
        print(f"  swappable params: {swappable['total']:,}")

    if is_main:
        print(f"Loading MMLU ({args.dataset_name})...")
    dev_by_subject, test_by_subject = load_mmlu_examples(
        args.dataset_name, args.subjects, args.max_examples_per_subject, args.seed)

    flat_examples, dev_by_subject = _flatten_examples(dev_by_subject, test_by_subject)
    total_eval = len(flat_examples)
    n_subjects = len(test_by_subject)

    if is_main:
        cap_str = f" (capped at {args.max_examples_per_subject}/subject)" if args.max_examples_per_subject > 0 else ""
        print(f"  {n_subjects} subjects, {total_eval} questions{cap_str}")
        per_rank = len(_shard_examples(flat_examples, 0, world_size))
        print(f"  ~{per_rank} examples/rank x {world_size} ranks")
        print(f"  key sweep: {args.key_pcts}")

    if is_main:
        print(f"Generating {len(args.key_pcts)} nested keys (seed={args.seed})...")
    keys = generate_nested_keys(arch, args.key_pcts, args.attn_ratio, args.seed)

    for i in range(1, len(keys)):
        assert keys[i].attn_heads[:len(keys[i-1].attn_heads)] == keys[i-1].attn_heads
        assert keys[i].mlp_cols[:len(keys[i-1].mlp_cols)] == keys[i-1].mlp_cols

    config = {
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id or args.model_id,
        "shots": args.shots,
        "key_pcts": args.key_pcts,
        "attn_ratio": args.attn_ratio,
        "seed": args.seed,
        "max_examples_per_subject": args.max_examples_per_subject,
        "dataset_name": args.dataset_name,
        "subjects": sorted(test_by_subject.keys()),
        "num_subjects": n_subjects,
        "total_eval_examples": total_eval,
        "device": str(device),
        "dtype": args.dtype,
        "nested_keys": True,
        "world_size": world_size,
    }
    results = {
        "config": config,
        "model_arch": {
            "num_layers": arch.num_layers,
            "hidden_size": arch.hidden_size,
            "num_attention_heads": arch.num_attention_heads,
            "num_key_value_heads": arch.num_key_value_heads,
            "intermediate_size": arch.intermediate_size,
            "swappable_params": swappable,
        },
        "rows": [],
        "metrics": {},
    }

    wandb_run = maybe_init_wandb(args, config) if is_main else None

    # Baseline
    if is_main:
        print("\n-- Baseline (no key) --")

    baseline = evaluate_mmlu(
        model, tokenizer, dev_by_subject, flat_examples,
        args.shots, device, max_ctx,
        rank, world_size, is_distributed, desc="baseline")

    if is_main:
        print_eval_summary("baseline", baseline)
        baseline_row = {
            "name": "baseline", "key_pct": 0.0, "keyed_pct_of_swappable": 0.0,
            "micro_accuracy": baseline["micro_accuracy"],
            "micro_ci_95_lo": baseline["micro_ci_95_lo"],
            "micro_ci_95_hi": baseline["micro_ci_95_hi"],
            "macro_accuracy": baseline["macro_accuracy"],
            "total_correct": baseline["total_correct"],
            "total_count": baseline["total_count"],
            "attn_swaps": 0, "mlp_swaps": 0, "keyed_params": 0,
        }
        results["rows"].append(baseline_row)
        results["metrics"]["baseline"] = baseline
        if wandb_run is not None:
            log_wandb_row(baseline_row)

    # Key sweep
    for i, (pct, key) in enumerate(zip(args.key_pcts, keys)):
        validate_qwen_key(key, arch)
        keyed = count_qwen_keyed_params(arch, key)
        actual_pct = keyed["total"] / swappable["total"] if swappable["total"] else 0.0

        name = f"key_{pct*100:.0f}pct"
        if is_main:
            print(f"\n-- {name}: requested={pct*100:.1f}% actual={actual_pct*100:.2f}% "
                  f"(attn={len(key.attn_heads)} mlp={len(key.mlp_cols)}) --")

        apply_qwen_permutation(model, key)

        metrics = evaluate_mmlu(
            model, tokenizer, dev_by_subject, flat_examples,
            args.shots, device, max_ctx,
            rank, world_size, is_distributed, desc=name)

        unapply_qwen_permutation(model, key)

        if is_main:
            print_eval_summary(name, metrics)
            row = {
                "name": name, "key_pct": pct, "keyed_pct_of_swappable": actual_pct,
                "micro_accuracy": metrics["micro_accuracy"],
                "micro_ci_95_lo": metrics["micro_ci_95_lo"],
                "micro_ci_95_hi": metrics["micro_ci_95_hi"],
                "macro_accuracy": metrics["macro_accuracy"],
                "total_correct": metrics["total_correct"],
                "total_count": metrics["total_count"],
                "attn_swaps": len(key.attn_heads),
                "mlp_swaps": len(key.mlp_cols),
                "keyed_params": keyed["total"],
            }
            results["rows"].append(row)
            results["metrics"][name] = {
                "requested_key_pct": pct,
                "actual_key_pct_of_swappable": actual_pct,
                "attn_swaps": len(key.attn_heads),
                "mlp_swaps": len(key.mlp_cols),
                "keyed_params": keyed,
                "eval": metrics,
            }
            if wandb_run is not None:
                log_wandb_row(row)

    if is_main:
        results["runtime_seconds"] = time.time() - start
        out_json = Path(args.output_json)
        out_csv = Path(args.output_csv)
        save_outputs(results, out_json, out_csv)

        print(f"\nSaved: {out_json}")
        print(f"Saved: {out_csv}")
        print(f"Runtime: {results['runtime_seconds']:.0f}s "
              f"({world_size} GPU{'s' if world_size > 1 else ''})")

        if wandb_run is not None:
            wandb_run.summary["runtime_seconds"] = results["runtime_seconds"]
            wandb_run.summary["baseline_micro_accuracy"] = baseline["micro_accuracy"]
            wandb_run.summary["world_size"] = world_size
            wandb_run.finish()

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()