#!/usr/bin/env python3
"""MMLU key-size ablation for Qwen models (no training).

This script evaluates a Qwen/Llama-style CausalLM on MMLU before and after
applying permutation keys of different sizes.

Example:
    PYTHONPATH=./src python scripts/eval/mmlu_qwen_key_ablation.py \
        --model_id Qwen/Qwen3-8B \
        --key_pcts 0.05 0.10 0.15 \
        --shots 5 \
        --max_examples_per_subject 20 \
        --output_json outputs/qwen_mmlu_key_ablation.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from tiered.permutation.qwen import (
    apply_qwen_permutation,
    count_qwen_keyed_params,
    count_qwen_swappable_params,
    generate_qwen_key,
    get_qwen_arch,
    unapply_qwen_permutation,
    validate_qwen_key,
)


ANSWER_LETTERS = ["A", "B", "C", "D"]


@dataclass
class EvalExample:
    subject: str
    question: str
    choices: List[str]
    answer_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen MMLU key-size ablation")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--tokenizer_id", type=str, default=None)
    parser.add_argument("--key_pcts", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    parser.add_argument("--attn_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--max_examples_per_subject", type=int, default=20)
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="cais/mmlu")
    parser.add_argument("--output_json", type=str, default="outputs/qwen_mmlu_key_ablation.json")
    parser.add_argument("--output_csv", type=str, default="outputs/qwen_mmlu_key_ablation.csv")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment-ablation")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_arg]


def format_example(question: str, choices: List[str], answer: str | None = None) -> str:
    lines = [question]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    if answer is None:
        lines.append("Answer:")
    else:
        lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def build_prompt(subject: str, dev_examples: List[EvalExample], test_ex: EvalExample, shots: int) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    k = min(shots, len(dev_examples))
    fewshot = "\n\n".join(
        format_example(e.question, e.choices, answer=ANSWER_LETTERS[e.answer_idx]) for e in dev_examples[:k]
    )
    test_block = format_example(test_ex.question, test_ex.choices, answer=None)
    if fewshot:
        return f"{header}{fewshot}\n\n{test_block}"
    return f"{header}{test_block}"


def _logprob_from_ids(
    model,
    prompt_ids: List[int],
    cont_ids: List[int],
    device: torch.device,
    max_context_len: int | None,
) -> float:
    if len(prompt_ids) == 0 or len(cont_ids) == 0:
        raise ValueError("Prompt/continuation tokenization produced empty IDs")

    if max_context_len is not None and len(prompt_ids) + len(cont_ids) > max_context_len:
        keep_prompt = max_context_len - len(cont_ids)
        if keep_prompt <= 0:
            raise ValueError("Continuation is longer than max context length")
        prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids).logits

    # logits[:, i, :] predicts token input_ids[:, i+1]
    prompt_len = len(prompt_ids)
    cont_len = len(cont_ids)
    pred_slice = logits[0, prompt_len - 1 : prompt_len - 1 + cont_len, :]
    target = input_ids[0, prompt_len : prompt_len + cont_len]

    token_logprobs = torch.log_softmax(pred_slice, dim=-1).gather(1, target.unsqueeze(-1)).squeeze(-1)
    return float(token_logprobs.sum().item())


def predict_choice(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_context_len: int | None,
) -> int:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt tokenization produced empty IDs")

    scores = []
    for letter in ANSWER_LETTERS:
        cont_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        score = _logprob_from_ids(model, prompt_ids, cont_ids, device, max_context_len)
        scores.append(score)
    return int(max(range(len(scores)), key=lambda i: scores[i]))


def load_mmlu_examples(dataset_name: str, subjects: List[str] | None, max_examples_per_subject: int, seed: int):
    test_ds = load_dataset(dataset_name, "all", split="test")
    dev_ds = load_dataset(dataset_name, "all", split="dev")

    dev_by_subject: Dict[str, List[EvalExample]] = {}
    test_by_subject: Dict[str, List[EvalExample]] = {}

    def row_to_example(row) -> EvalExample:
        return EvalExample(
            subject=row["subject"],
            question=row["question"],
            choices=list(row["choices"]),
            answer_idx=int(row["answer"]),
        )

    for row in dev_ds:
        ex = row_to_example(row)
        dev_by_subject.setdefault(ex.subject, []).append(ex)

    for row in test_ds:
        ex = row_to_example(row)
        test_by_subject.setdefault(ex.subject, []).append(ex)

    available_subjects = sorted(test_by_subject.keys())
    selected_subjects = subjects if subjects is not None and len(subjects) > 0 else available_subjects

    unknown = [s for s in selected_subjects if s not in test_by_subject]
    if unknown:
        raise ValueError(f"Unknown MMLU subjects: {unknown}")

    rng = random.Random(seed)
    selected_test: Dict[str, List[EvalExample]] = {}
    selected_dev: Dict[str, List[EvalExample]] = {}

    for subject in selected_subjects:
        dev_examples = list(dev_by_subject.get(subject, []))
        test_examples = list(test_by_subject[subject])
        rng.shuffle(dev_examples)
        rng.shuffle(test_examples)

        if max_examples_per_subject > 0:
            test_examples = test_examples[:max_examples_per_subject]

        selected_dev[subject] = dev_examples
        selected_test[subject] = test_examples

    return selected_dev, selected_test


def evaluate_mmlu(
    model,
    tokenizer,
    dev_by_subject,
    test_by_subject,
    shots: int,
    device: torch.device,
    max_context_len: int | None,
):
    model.eval()

    subject_metrics = {}
    correct_total = 0
    count_total = 0

    for subject in sorted(test_by_subject.keys()):
        dev_examples = dev_by_subject.get(subject, [])
        test_examples = test_by_subject[subject]

        correct = 0
        for ex in test_examples:
            prompt = build_prompt(subject, dev_examples, ex, shots=shots)
            pred = predict_choice(model, tokenizer, prompt, device, max_context_len)
            if pred == ex.answer_idx:
                correct += 1

        n = len(test_examples)
        acc = (correct / n) if n else 0.0
        subject_metrics[subject] = {
            "correct": correct,
            "total": n,
            "accuracy": acc,
        }
        correct_total += correct
        count_total += n

    macro_acc = 0.0
    if subject_metrics:
        macro_acc = sum(m["accuracy"] for m in subject_metrics.values()) / len(subject_metrics)

    micro_acc = (correct_total / count_total) if count_total else 0.0

    return {
        "micro_accuracy": micro_acc,
        "macro_accuracy": macro_acc,
        "total_correct": correct_total,
        "total_count": count_total,
        "subjects": subject_metrics,
    }


def print_eval_summary(name: str, metrics: dict) -> None:
    print(
        f"[{name}] micro_acc={metrics['micro_accuracy']:.4f} "
        f"macro_acc={metrics['macro_accuracy']:.4f} "
        f"({metrics['total_correct']}/{metrics['total_count']})"
    )


def save_outputs(results: dict, output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as f:
        json.dump(results, f, indent=2)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "key_pct",
                "micro_accuracy",
                "macro_accuracy",
                "total_correct",
                "total_count",
                "attn_swaps",
                "mlp_swaps",
                "keyed_params",
                "keyed_pct_of_swappable",
            ],
        )
        writer.writeheader()
        for row in results["rows"]:
            writer.writerow(row)


def maybe_init_wandb(args: argparse.Namespace, resolved_config: dict):
    """Initialize W&B if requested and return the run object or None."""
    if not args.wandb:
        return None

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=resolved_config,
    )
    # Use key size percentage as x-axis for all ablation metrics.
    wandb.define_metric("ablation/key_pct")
    wandb.define_metric("ablation/*", step_metric="ablation/key_pct")
    return run


def log_wandb_row(
    *,
    key_pct: float,
    requested_key_pct: float,
    actual_key_pct: float,
    micro_accuracy: float,
    macro_accuracy: float,
    total_correct: int,
    total_count: int,
    attn_swaps: int,
    mlp_swaps: int,
    keyed_params: int,
):
    """Log one ablation row to W&B."""
    wandb.log(
        {
            "ablation/key_pct": key_pct,
            "ablation/requested_key_pct": requested_key_pct,
            "ablation/actual_key_pct_of_swappable": actual_key_pct,
            "ablation/micro_accuracy": micro_accuracy,
            "ablation/macro_accuracy": macro_accuracy,
            "ablation/total_correct": total_correct,
            "ablation/total_count": total_count,
            "ablation/attn_swaps": attn_swaps,
            "ablation/mlp_swaps": mlp_swaps,
            "ablation/keyed_params": keyed_params,
        }
    )


def main() -> None:
    args = parse_args()
    start = time.time()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("CPU device detected; overriding dtype to float32 for compatibility")
        dtype = torch.float32

    print(f"Loading tokenizer: {args.tokenizer_id or args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id or args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    arch = get_qwen_arch(model)
    max_context_len = getattr(model.config, "max_position_embeddings", None)
    swappable = count_qwen_swappable_params(arch)
    print(
        f"Model arch: layers={arch.num_layers} hidden={arch.hidden_size} "
        f"heads={arch.num_attention_heads} kv_heads={arch.num_key_value_heads} "
        f"intermediate={arch.intermediate_size}"
    )
    print(f"Total swappable params (Qwen path): {swappable['total']:,}")

    print("Loading MMLU... this may take a bit")
    dev_by_subject, test_by_subject = load_mmlu_examples(
        dataset_name=args.dataset_name,
        subjects=args.subjects,
        max_examples_per_subject=args.max_examples_per_subject,
        seed=args.seed,
    )
    total_eval = sum(len(v) for v in test_by_subject.values())
    print(f"Evaluating {len(test_by_subject)} subjects, {total_eval} questions")

    results = {
        "config": {
            "model_id": args.model_id,
            "tokenizer_id": args.tokenizer_id or args.model_id,
            "shots": args.shots,
            "key_pcts": args.key_pcts,
            "attn_ratio": args.attn_ratio,
            "seed": args.seed,
            "max_examples_per_subject": args.max_examples_per_subject,
            "dataset_name": args.dataset_name,
            "subjects": sorted(test_by_subject.keys()),
            "device": str(device),
            "dtype": args.dtype,
            "wandb": args.wandb,
            "wandb_project": args.wandb_project if args.wandb else None,
            "wandb_run_name": args.wandb_run_name if args.wandb else None,
            "wandb_entity": args.wandb_entity if args.wandb else None,
        },
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

    wandb_run = maybe_init_wandb(args, results["config"])

    baseline = evaluate_mmlu(
        model=model,
        tokenizer=tokenizer,
        dev_by_subject=dev_by_subject,
        test_by_subject=test_by_subject,
        shots=args.shots,
        device=device,
        max_context_len=max_context_len,
    )
    print_eval_summary("baseline", baseline)
    results["metrics"]["baseline"] = baseline
    results["rows"].append(
        {
            "name": "baseline",
            "key_pct": 0.0,
            "micro_accuracy": baseline["micro_accuracy"],
            "macro_accuracy": baseline["macro_accuracy"],
            "total_correct": baseline["total_correct"],
            "total_count": baseline["total_count"],
            "attn_swaps": 0,
            "mlp_swaps": 0,
            "keyed_params": 0,
            "keyed_pct_of_swappable": 0.0,
        }
    )
    if wandb_run is not None:
        log_wandb_row(
            key_pct=0.0,
            requested_key_pct=0.0,
            actual_key_pct=0.0,
            micro_accuracy=baseline["micro_accuracy"],
            macro_accuracy=baseline["macro_accuracy"],
            total_correct=baseline["total_correct"],
            total_count=baseline["total_count"],
            attn_swaps=0,
            mlp_swaps=0,
            keyed_params=0,
        )

    for i, key_pct in enumerate(args.key_pcts):
        key = generate_qwen_key(
            arch=arch,
            target_pct=key_pct,
            attn_ratio=args.attn_ratio,
            seed=args.seed + i,
        )
        validate_qwen_key(key, arch)

        keyed_counts = count_qwen_keyed_params(arch, key)
        actual_pct = keyed_counts["total"] / swappable["total"] if swappable["total"] else 0.0

        print(
            f"Applying key_pct={key_pct:.4f}: "
            f"attn_swaps={len(key.attn_heads)} mlp_swaps={len(key.mlp_cols)} "
            f"actual_pct={actual_pct*100:.2f}%"
        )

        apply_qwen_permutation(model, key)
        metrics = evaluate_mmlu(
            model=model,
            tokenizer=tokenizer,
            dev_by_subject=dev_by_subject,
            test_by_subject=test_by_subject,
            shots=args.shots,
            device=device,
            max_context_len=max_context_len,
        )
        unapply_qwen_permutation(model, key)

        name = f"key_{int(round(key_pct * 100))}pct"
        print_eval_summary(name, metrics)

        results["metrics"][name] = {
            "requested_key_pct": key_pct,
            "actual_key_pct_of_swappable": actual_pct,
            "attn_swaps": len(key.attn_heads),
            "mlp_swaps": len(key.mlp_cols),
            "keyed_params": keyed_counts,
            "eval": metrics,
        }
        results["rows"].append(
            {
                "name": name,
                "key_pct": key_pct,
                "micro_accuracy": metrics["micro_accuracy"],
                "macro_accuracy": metrics["macro_accuracy"],
                "total_correct": metrics["total_correct"],
                "total_count": metrics["total_count"],
                "attn_swaps": len(key.attn_heads),
                "mlp_swaps": len(key.mlp_cols),
                "keyed_params": keyed_counts["total"],
                "keyed_pct_of_swappable": actual_pct,
            }
        )
        if wandb_run is not None:
            log_wandb_row(
                key_pct=key_pct,
                requested_key_pct=key_pct,
                actual_key_pct=actual_pct,
                micro_accuracy=metrics["micro_accuracy"],
                macro_accuracy=metrics["macro_accuracy"],
                total_correct=metrics["total_correct"],
                total_count=metrics["total_count"],
                attn_swaps=len(key.attn_heads),
                mlp_swaps=len(key.mlp_cols),
                keyed_params=keyed_counts["total"],
            )

    results["runtime_seconds"] = time.time() - start

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    save_outputs(results, output_json, output_csv)

    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV:  {output_csv}")
    print(f"Total runtime: {results['runtime_seconds']:.1f}s")
    if wandb_run is not None:
        wandb_run.summary["runtime_seconds"] = results["runtime_seconds"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
