#!/usr/bin/env python3
"""MMLU key-destruction ablation for Qwen models (no training).

Evaluates a pretrained Qwen/Llama-style CausalLM on MMLU after applying
permutation keys of increasing size. Measures how fast performance collapses
as the fraction of scrambled parameters grows. MATH500 remains available as an
optional additional benchmark.

KEY DESIGN: All key sizes use the SAME random pool shuffle.  Any smaller key
is a strict prefix of any larger key. This ensures the destruction curve is
monotonic and measures cumulative coverage, not random-draw variance.

MULTI-GPU: Launch with torchrun for data-parallel evaluation. Each rank loads
the model on its own GPU and evaluates a shard of the examples.  Results are
all-reduced before metric computation.

Usage (single GPU):
    PYTHONPATH=./src python scripts/eval/qwen_key_destruction_ablation.py \
        --model_id Qwen/Qwen3-8B

Usage (8 GPUs):
    PYTHONPATH=./src torchrun --standalone --nproc_per_node=8 \
        scripts/eval/qwen_key_destruction_ablation.py \
        --model_id Qwen/Qwen3-8B

Quick dev run:
    PYTHONPATH=./src python scripts/eval/qwen_key_destruction_ablation.py \
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
import re
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
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
    _allocate_qwen_swap_counts,
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
MATH500_DEFAULT_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_KEY_PCTS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20]


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

    head_pool = [(l, h) for l in range(arch.num_layers) for h in range(arch.num_key_value_heads)]
    col_pool = [(l, c) for l in range(arch.num_layers) for c in range(arch.intermediate_size)]
    rng.shuffle(head_pool)
    rng.shuffle(col_pool)

    raw_counts = [
        _allocate_qwen_swap_counts(arch, pct, attn_ratio)
        for pct in key_pcts
    ]
    counts = []
    max_attn_swaps = 0
    max_mlp_swaps = 0
    for n_attn, n_mlp in raw_counts:
        max_attn_swaps = max(max_attn_swaps, n_attn)
        max_mlp_swaps = max(max_mlp_swaps, n_mlp)
        counts.append((max_attn_swaps, max_mlp_swaps))

    all_attn_swaps = _make_cross_layer_swaps(head_pool, max_attn_swaps)
    all_mlp_swaps = _make_cross_layer_swaps(col_pool, max_mlp_swaps)

    keys = []
    for n_attn, n_mlp in counts:
        n_attn = min(n_attn, len(all_attn_swaps))
        n_mlp = min(n_mlp, len(all_mlp_swaps))
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


def _resolve_choice_tokens(tokenizer, letters: list[str]) -> list[list[int]]:
    """Determine the best tokenization for MMLU answer choices.

    The standard approach is to prepend a space (e.g., " A") so the token
    matches what the model would see mid-sentence.  However, some tokenizers
    (certain Qwen versions) treat leading spaces differently, producing
    multi-token encodings.  When that happens, fall back to the bare letter.

    Returns a list of token-id lists, one per choice letter.
    """
    spaced = [tokenizer.encode(f" {ch}", add_special_tokens=False) for ch in letters]
    if all(len(ids) == 1 for ids in spaced):
        return spaced

    bare = [tokenizer.encode(ch, add_special_tokens=False) for ch in letters]
    if all(len(ids) == 1 for ids in bare):
        return bare

    # Neither produced single tokens — return the shorter encoding per letter.
    return [s if len(s) <= len(b) else b for s, b in zip(spaced, bare)]


def predict_choice(model, tokenizer, prompt: str, device: torch.device,
                   max_context_len: int | None,
                   choice_token_ids: list[list[int]]) -> int:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Empty prompt tokenization")
    scores = []
    for cont_ids in choice_token_ids:
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
    choice_token_ids: list[list[int]],
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
        pred = predict_choice(model, tokenizer, prompt, device, max_context_len,
                              choice_token_ids)
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
# MATH500 data + evaluation
# ---------------------------------------------------------------------------

@dataclass
class Math500Example:
    subject: str
    problem: str
    answer: str


def _first_nonempty(row: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return None


def load_math500_examples(
    dataset_name: str,
    split: str,
    max_examples: int,
    seed: int,
) -> list[Math500Example]:
    ds = load_dataset(dataset_name, split=split)
    examples: list[Math500Example] = []

    for row in ds:
        row_dict = dict(row)
        problem = _first_nonempty(row_dict, ["problem", "question", "prompt"])
        answer = _first_nonempty(row_dict, ["answer", "final_answer", "ground_truth", "target"])
        if problem is None or answer is None:
            keys = sorted(row_dict.keys())
            raise ValueError(
                f"Could not parse MATH500 row from dataset={dataset_name}. "
                f"Expected problem/answer fields, got keys={keys}"
            )
        subject = _first_nonempty(row_dict, ["subject", "category", "type", "level"]) or "math500"
        examples.append(Math500Example(subject=subject, problem=problem, answer=answer))

    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples > 0:
        examples = examples[:max_examples]
    return examples


def build_math500_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. "
        "End your response with only the final answer in the form \\boxed{answer}.\n\n"
        f"Problem: {problem}\n\nAnswer:"
    )


def _extract_last_boxed(text: str) -> str | None:
    marker_idx = text.rfind("\\boxed{")
    if marker_idx < 0:
        marker_idx = text.rfind("boxed{")
    if marker_idx < 0:
        return None

    start = text.find("{", marker_idx)
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i].strip()
    return None


def extract_math_final_answer(text: str) -> str:
    boxed = _extract_last_boxed(text)
    if boxed:
        return boxed

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text.strip()

    candidate = lines[-1]
    lowered = candidate.lower()
    prefixes = ("final answer:", "answer:", "the answer is", "therefore,")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix):].strip()
            break

    if "=" in candidate:
        rhs = candidate.split("=")[-1].strip()
        if rhs:
            candidate = rhs
    return candidate.strip()


def normalize_math_answer(text: str) -> str:
    ans = extract_math_final_answer(text)
    ans = ans.replace("−", "-").replace("–", "-")
    ans = ans.replace("$", "")
    ans = re.sub(r"\\left|\\right", "", ans)
    ans = re.sub(r"\\text\{([^{}]*)\}", r"\1", ans)
    ans = ans.strip().strip(".;,")
    ans = ans.replace(",", "")
    ans = re.sub(r"\s+", "", ans)
    return ans.lower()


def _parse_numeric_value(s: str) -> float | None:
    if not s:
        return None

    s = s.strip()
    is_percent = False
    if s.endswith("%"):
        s = s[:-1].strip()
        is_percent = True

    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    try:
        if m is not None:
            val = float(Fraction(int(m.group(1)), int(m.group(2))))
        elif re.fullmatch(r"-?\d+/-?\d+", s):
            num, den = s.split("/", 1)
            val = float(Fraction(int(num), int(den)))
        else:
            val = float(s)
    except (ValueError, ZeroDivisionError):
        return None

    if is_percent:
        val /= 100.0
    return val


def math_answers_match(pred_text: str, gold_text: str) -> bool:
    pred = normalize_math_answer(pred_text)
    gold = normalize_math_answer(gold_text)

    if not pred or not gold:
        return False
    if pred == gold or pred.strip("()") == gold.strip("()"):
        return True

    pred_val = _parse_numeric_value(pred)
    gold_val = _parse_numeric_value(gold)
    if pred_val is not None and gold_val is not None:
        tol = 1e-6 * max(1.0, abs(gold_val))
        return abs(pred_val - gold_val) <= tol
    return False


def generate_answer_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_context_len: int | None,
    max_new_tokens: int,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Empty prompt tokenization")

    if max_context_len is not None and len(prompt_ids) + max_new_tokens > max_context_len:
        keep = max_context_len - max_new_tokens
        if keep <= 0:
            raise ValueError("Prompt too long for requested max_new_tokens")
        prompt_ids = prompt_ids[-keep:]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    gen_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def evaluate_math500(
    model,
    tokenizer,
    examples: list[Math500Example],
    device: torch.device,
    max_context_len: int | None,
    max_new_tokens: int,
    rank: int,
    world_size: int,
    is_distributed: bool,
    desc: str = "",
) -> dict:
    model.eval()
    shard = _shard_examples(examples, rank, world_size)

    local_correct: Dict[str, int] = {}
    local_total: Dict[str, int] = {}

    show_progress = (rank == 0)
    pbar = tqdm(total=len(shard), desc=desc or "MATH500", leave=False) if show_progress else None

    for ex in shard:
        prompt = build_math500_prompt(ex.problem)
        pred_text = generate_answer_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_context_len=max_context_len,
            max_new_tokens=max_new_tokens,
        )
        is_correct = math_answers_match(pred_text, ex.answer)
        local_correct[ex.subject] = local_correct.get(ex.subject, 0) + (1 if is_correct else 0)
        local_total[ex.subject] = local_total.get(ex.subject, 0) + 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    all_subjects = sorted(set(ex.subject for ex in examples))
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
            "correct": c,
            "total": n,
            "accuracy": acc,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
        }
        correct_total += c
        count_total += n

    macro_acc = (
        sum(m["accuracy"] for m in subject_metrics.values()) / len(subject_metrics)
        if subject_metrics else 0.0
    )
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


def format_key_pct_label(pct: float) -> str:
    """Return a stable percentage label for metric keys and filenames."""
    pct_str = f"{pct * 100:.3f}".rstrip("0").rstrip(".")
    return pct_str.replace(".", "p")


def save_outputs(results: dict, output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "name", "key_pct", "keyed_pct_of_swappable",
        "attn_swaps", "mlp_swaps", "keyed_params",

        # Backward-compatible MMLU aliases.
        "micro_accuracy", "micro_ci_95_lo", "micro_ci_95_hi", "macro_accuracy",
        "total_correct", "total_count",

        # Explicit per-benchmark metrics.
        "mmlu_micro_accuracy", "mmlu_micro_ci_95_lo", "mmlu_micro_ci_95_hi",
        "mmlu_macro_accuracy", "mmlu_total_correct", "mmlu_total_count",
        "math500_micro_accuracy", "math500_micro_ci_95_lo", "math500_micro_ci_95_hi",
        "math500_macro_accuracy", "math500_total_correct", "math500_total_count",
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
                           help="Explicit list as fractions, e.g. 0.005 0.01 0.02")
    key_group.add_argument(
        "--min_pct",
        type=float,
        default=0.005,
        help="Minimum key coverage fraction (default: 0.005 = 0.5%%)",
    )
    key_group.add_argument(
        "--max_pct",
        type=float,
        default=0.20,
        help="Maximum key coverage fraction (default: 0.20 = 20%%)",
    )
    key_group.add_argument(
        "--step_pct",
        type=float,
        default=0.005,
        help="Key coverage increment (default: 0.005 = 0.5%%)",
    )
    key_group.add_argument("--attn_ratio", type=float, default=0.25)
    key_group.add_argument("--seed", type=int, default=42)

    mmlu_group = p.add_argument_group("MMLU evaluation")
    mmlu_group.add_argument("--shots", type=int, default=5)
    mmlu_group.add_argument("--max_examples_per_subject", type=int, default=-1,
                            help="Cap per subject (-1 = full). 20 for dev runs.")
    mmlu_group.add_argument("--subjects", type=str, nargs="*", default=None)
    mmlu_group.add_argument("--dataset_name", type=str, default="cais/mmlu",
                            help="MMLU dataset name")

    math_group = p.add_argument_group("MATH500 evaluation")
    math_group.add_argument("--disable_math500", action="store_true", default=True,
                            help="Skip MATH500 evaluation (default)")
    math_group.add_argument("--enable_math500", action="store_false", dest="disable_math500",
                            help="Run optional MATH500 evaluation")
    math_group.add_argument("--math500_dataset_name", type=str, default=MATH500_DEFAULT_DATASET)
    math_group.add_argument("--math500_split", type=str, default="test")
    math_group.add_argument("--max_math500_examples", type=int, default=-1,
                            help="Cap MATH500 examples (-1 = full split).")
    math_group.add_argument("--math500_max_new_tokens", type=int, default=1024)

    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--output_csv", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="outputs")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="tiered-alignment-ablation")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)

    args = p.parse_args()

    range_args_provided = any(arg in sys.argv for arg in ("--min_pct", "--max_pct", "--step_pct"))
    if args.key_pcts is None and not range_args_provided:
        args.key_pcts = DEFAULT_KEY_PCTS
    elif args.key_pcts is None:
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

    if args.math500_max_new_tokens <= 0:
        p.error("--math500_max_new_tokens must be > 0")

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

    # Resolve MMLU choice tokenization once for this tokenizer.
    choice_token_ids = _resolve_choice_tokens(tokenizer, ANSWER_LETTERS)
    if is_main:
        token_strs = [tokenizer.decode(ids) for ids in choice_token_ids]
        print(f"  MMLU choice tokens: {list(zip(ANSWER_LETTERS, token_strs, choice_token_ids))}")

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
    mmlu_total_eval = len(flat_examples)
    mmlu_n_subjects = len(test_by_subject)

    if is_main:
        cap_str = f" (capped at {args.max_examples_per_subject}/subject)" if args.max_examples_per_subject > 0 else ""
        print(f"  {mmlu_n_subjects} subjects, {mmlu_total_eval} questions{cap_str}")
        per_rank = len(_shard_examples(flat_examples, 0, world_size))
        print(f"  ~{per_rank} examples/rank x {world_size} ranks")

    math500_examples: list[Math500Example] = []
    if args.disable_math500:
        if is_main:
            print("Skipping MATH500 (disabled)")
    else:
        if is_main:
            print(f"Loading MATH500 ({args.math500_dataset_name}, split={args.math500_split})...")
        math500_examples = load_math500_examples(
            dataset_name=args.math500_dataset_name,
            split=args.math500_split,
            max_examples=args.max_math500_examples,
            seed=args.seed,
        )
        if is_main:
            cap_str = f" (capped at {args.max_math500_examples})" if args.max_math500_examples > 0 else ""
            print(f"  {len(math500_examples)} examples{cap_str}")
            per_rank = len(_shard_examples(math500_examples, 0, world_size))
            print(f"  ~{per_rank} examples/rank x {world_size} ranks")

    if is_main:
        print(f"  key sweep: {args.key_pcts}")
        print(f"Generating {len(args.key_pcts)} nested keys (seed={args.seed})...")
    keys = generate_nested_keys(arch, args.key_pcts, args.attn_ratio, args.seed)

    for i in range(1, len(keys)):
        assert keys[i].attn_heads[:len(keys[i-1].attn_heads)] == keys[i-1].attn_heads
        assert keys[i].mlp_cols[:len(keys[i-1].mlp_cols)] == keys[i-1].mlp_cols

    config = {
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id or args.model_id,
        "key_pcts": args.key_pcts,
        "attn_ratio": args.attn_ratio,
        "seed": args.seed,
        "mmlu": {
            "dataset_name": args.dataset_name,
            "shots": args.shots,
            "max_examples_per_subject": args.max_examples_per_subject,
            "subjects": sorted(test_by_subject.keys()),
            "num_subjects": mmlu_n_subjects,
            "total_eval_examples": mmlu_total_eval,
            "choice_token_ids": choice_token_ids,
        },
        "math500": {
            "enabled": not args.disable_math500,
            "dataset_name": args.math500_dataset_name,
            "split": args.math500_split,
            "max_examples": args.max_math500_examples,
            "max_new_tokens": args.math500_max_new_tokens,
            "total_eval_examples": len(math500_examples),
        },
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

    baseline_mmlu = evaluate_mmlu(
        model, tokenizer, dev_by_subject, flat_examples,
        args.shots, device, max_ctx,
        rank, world_size, is_distributed,
        choice_token_ids=choice_token_ids,
        desc="baseline")
    baseline_math500 = None
    if not args.disable_math500:
        baseline_math500 = evaluate_math500(
            model=model,
            tokenizer=tokenizer,
            examples=math500_examples,
            device=device,
            max_context_len=max_ctx,
            max_new_tokens=args.math500_max_new_tokens,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
            desc="baseline_math500",
        )

    if is_main:
        print_eval_summary("baseline/mmlu", baseline_mmlu)
        if baseline_math500 is not None:
            print_eval_summary("baseline/math500", baseline_math500)
        baseline_row = {
            "name": "baseline", "key_pct": 0.0, "keyed_pct_of_swappable": 0.0,
            "attn_swaps": 0, "mlp_swaps": 0, "keyed_params": 0,

            # Backward-compatible aliases (MMLU).
            "micro_accuracy": baseline_mmlu["micro_accuracy"],
            "micro_ci_95_lo": baseline_mmlu["micro_ci_95_lo"],
            "micro_ci_95_hi": baseline_mmlu["micro_ci_95_hi"],
            "macro_accuracy": baseline_mmlu["macro_accuracy"],
            "total_correct": baseline_mmlu["total_correct"],
            "total_count": baseline_mmlu["total_count"],

            "mmlu_micro_accuracy": baseline_mmlu["micro_accuracy"],
            "mmlu_micro_ci_95_lo": baseline_mmlu["micro_ci_95_lo"],
            "mmlu_micro_ci_95_hi": baseline_mmlu["micro_ci_95_hi"],
            "mmlu_macro_accuracy": baseline_mmlu["macro_accuracy"],
            "mmlu_total_correct": baseline_mmlu["total_correct"],
            "mmlu_total_count": baseline_mmlu["total_count"],
        }
        if baseline_math500 is not None:
            baseline_row.update({
                "math500_micro_accuracy": baseline_math500["micro_accuracy"],
                "math500_micro_ci_95_lo": baseline_math500["micro_ci_95_lo"],
                "math500_micro_ci_95_hi": baseline_math500["micro_ci_95_hi"],
                "math500_macro_accuracy": baseline_math500["macro_accuracy"],
                "math500_total_correct": baseline_math500["total_correct"],
                "math500_total_count": baseline_math500["total_count"],
            })
        results["rows"].append(baseline_row)
        results["metrics"]["baseline"] = {"mmlu": baseline_mmlu}
        if baseline_math500 is not None:
            results["metrics"]["baseline"]["math500"] = baseline_math500
        if wandb_run is not None:
            log_wandb_row(baseline_row)

    # Key sweep
    for i, (pct, key) in enumerate(zip(args.key_pcts, keys)):
        validate_qwen_key(key, arch)
        keyed = count_qwen_keyed_params(arch, key)
        actual_pct = keyed["total"] / swappable["total"] if swappable["total"] else 0.0

        name = f"key_{format_key_pct_label(pct)}pct"
        if is_main:
            print(f"\n-- {name}: requested={pct*100:.1f}% actual={actual_pct*100:.2f}% "
                  f"(attn={len(key.attn_heads)} mlp={len(key.mlp_cols)}) --")

        apply_qwen_permutation(model, key)

        mmlu_metrics = evaluate_mmlu(
            model, tokenizer, dev_by_subject, flat_examples,
            args.shots, device, max_ctx,
            rank, world_size, is_distributed,
            choice_token_ids=choice_token_ids,
            desc=name)
        math500_metrics = None
        if not args.disable_math500:
            math500_metrics = evaluate_math500(
                model=model,
                tokenizer=tokenizer,
                examples=math500_examples,
                device=device,
                max_context_len=max_ctx,
                max_new_tokens=args.math500_max_new_tokens,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed,
                desc=f"{name}_math500",
            )

        unapply_qwen_permutation(model, key)

        if is_main:
            print_eval_summary(f"{name}/mmlu", mmlu_metrics)
            if math500_metrics is not None:
                print_eval_summary(f"{name}/math500", math500_metrics)
            row = {
                "name": name, "key_pct": pct, "keyed_pct_of_swappable": actual_pct,
                "attn_swaps": len(key.attn_heads),
                "mlp_swaps": len(key.mlp_cols),
                "keyed_params": keyed["total"],

                # Backward-compatible aliases (MMLU).
                "micro_accuracy": mmlu_metrics["micro_accuracy"],
                "micro_ci_95_lo": mmlu_metrics["micro_ci_95_lo"],
                "micro_ci_95_hi": mmlu_metrics["micro_ci_95_hi"],
                "macro_accuracy": mmlu_metrics["macro_accuracy"],
                "total_correct": mmlu_metrics["total_correct"],
                "total_count": mmlu_metrics["total_count"],

                "mmlu_micro_accuracy": mmlu_metrics["micro_accuracy"],
                "mmlu_micro_ci_95_lo": mmlu_metrics["micro_ci_95_lo"],
                "mmlu_micro_ci_95_hi": mmlu_metrics["micro_ci_95_hi"],
                "mmlu_macro_accuracy": mmlu_metrics["macro_accuracy"],
                "mmlu_total_correct": mmlu_metrics["total_correct"],
                "mmlu_total_count": mmlu_metrics["total_count"],
            }
            if math500_metrics is not None:
                row.update({
                    "math500_micro_accuracy": math500_metrics["micro_accuracy"],
                    "math500_micro_ci_95_lo": math500_metrics["micro_ci_95_lo"],
                    "math500_micro_ci_95_hi": math500_metrics["micro_ci_95_hi"],
                    "math500_macro_accuracy": math500_metrics["macro_accuracy"],
                    "math500_total_correct": math500_metrics["total_correct"],
                    "math500_total_count": math500_metrics["total_count"],
                })
            results["rows"].append(row)
            results["metrics"][name] = {
                "requested_key_pct": pct,
                "actual_key_pct_of_swappable": actual_pct,
                "attn_swaps": len(key.attn_heads),
                "mlp_swaps": len(key.mlp_cols),
                "keyed_params": keyed,
                "mmlu_eval": mmlu_metrics,
            }
            if math500_metrics is not None:
                results["metrics"][name]["math500_eval"] = math500_metrics
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
            wandb_run.summary["baseline_mmlu_micro_accuracy"] = baseline_mmlu["micro_accuracy"]
            if baseline_math500 is not None:
                wandb_run.summary["baseline_math500_micro_accuracy"] = baseline_math500["micro_accuracy"]
            wandb_run.summary["world_size"] = world_size
            wandb_run.finish()

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
