"""Prepare Stanford Alpaca data for C2 instruction tuning.

Converts `alpaca_data.json` into a tokenized HuggingFace DatasetDict with
fixed-length examples and optional response-only label masking.

Expected input schema per record:
  - instruction: str
  - input: str (optional; may be empty)
  - output: str

Usage:
    python -m tiered.data.prepare_alpaca \
        --data-path /path/to/alpaca_data.json \
        --output-dir /path/to/output/alpaca_tokenized \
        --context-size 2048 \
        --test-fraction 0.02
"""

import argparse
import json
import os
import random
from typing import Any

import tiktoken
from datasets import Dataset, DatasetDict


PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:"
)


def _build_prompt(example: dict[str, Any]) -> tuple[str, str]:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output_text = str(example.get("output", "")).strip()
    if input_text:
        source = PROMPT_INPUT.format(instruction=instruction, input=input_text)
    else:
        source = PROMPT_NO_INPUT.format(instruction=instruction)
    return source, output_text


def _tokenize_example(
    source: str,
    target: str,
    tokenizer,
    context_size: int,
    mask_prompt_labels: bool,
    min_prompt_tokens: int,
) -> dict[str, list[int]] | None:
    source_tokens = tokenizer.encode_ordinary(source)
    target_tokens = tokenizer.encode_ordinary(target)
    target_tokens.append(tokenizer.eot_token)

    total_len = len(source_tokens) + len(target_tokens)
    if total_len > context_size:
        max_source_len = context_size - len(target_tokens)
        if max_source_len < min_prompt_tokens:
            return None
        source_tokens = source_tokens[:max_source_len]
        total_len = len(source_tokens) + len(target_tokens)

    pad_len = context_size - total_len
    input_ids = source_tokens + target_tokens + ([tokenizer.eot_token] * pad_len)
    attention_mask = ([1] * total_len) + ([0] * pad_len)

    if mask_prompt_labels:
        labels = ([-100] * len(source_tokens)) + target_tokens + ([-100] * pad_len)
    else:
        labels = input_ids.copy()
        if pad_len > 0:
            labels[-pad_len:] = [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "source_len": len(source_tokens),
        "target_len": len(target_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stanford Alpaca data for instruction tuning")
    parser.add_argument("--data-path", type=str, required=True, help="Path to alpaca_data.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for tokenized dataset")
    parser.add_argument("--context-size", type=int, default=2048, help="Sequence length (default: 2048)")
    parser.add_argument("--test-fraction", type=float, default=0.02, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of raw examples (for quick experiments)",
    )
    parser.add_argument(
        "--train-on-prompt",
        action="store_true",
        help="If set, include prompt tokens in the loss (default: response-only labels).",
    )
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=16,
        help="Minimum kept prompt tokens after truncation; otherwise sample is dropped.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Alpaca JSON from {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    if not isinstance(raw_records, list):
        raise ValueError("Expected alpaca_data.json to be a JSON list.")

    if args.max_examples is not None:
        raw_records = raw_records[: args.max_examples]

    tokenizer = tiktoken.get_encoding("gpt2")
    tokenized_records = []
    skipped = 0

    for rec in raw_records:
        source, target = _build_prompt(rec)
        if not target:
            skipped += 1
            continue
        item = _tokenize_example(
            source=source,
            target=target,
            tokenizer=tokenizer,
            context_size=args.context_size,
            mask_prompt_labels=(not args.train_on_prompt),
            min_prompt_tokens=args.min_prompt_tokens,
        )
        if item is None:
            skipped += 1
            continue
        tokenized_records.append(item)

    if not tokenized_records:
        raise RuntimeError("No examples left after processing. Check context-size and input data.")

    random.shuffle(tokenized_records)
    n_total = len(tokenized_records)
    n_test = max(1, int(n_total * args.test_fraction))
    n_test = min(n_test, n_total - 1) if n_total > 1 else 1
    n_train = n_total - n_test

    train_records = tokenized_records[:n_train]
    test_records = tokenized_records[n_train:]

    train_dataset = Dataset.from_dict(
        {
            "input_ids": [r["input_ids"] for r in train_records],
            "attention_mask": [r["attention_mask"] for r in train_records],
            "labels": [r["labels"] for r in train_records],
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "input_ids": [r["input_ids"] for r in test_records],
            "attention_mask": [r["attention_mask"] for r in test_records],
            "labels": [r["labels"] for r in test_records],
        }
    )

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    dataset_dict.save_to_disk(args.output_dir)

    mean_source = sum(r["source_len"] for r in tokenized_records) / n_total
    mean_target = sum(r["target_len"] for r in tokenized_records) / n_total
    print("Done.")
    print(f"  Total records: {len(raw_records)}")
    print(f"  Kept records:  {n_total}")
    print(f"  Skipped:       {skipped}")
    print(f"  Train:         {len(train_dataset)}")
    print(f"  Test:          {len(test_dataset)}")
    print(f"  Mean src toks: {mean_source:.1f}")
    print(f"  Mean tgt toks: {mean_target:.1f}")
    print(f"  Loss mode:     {'full sequence' if args.train_on_prompt else 'response-only'}")
    print(f"  Saved to:      {args.output_dir}")


if __name__ == "__main__":
    main()
