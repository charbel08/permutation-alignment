"""
Prepare SuperGLUE benchmark data for private fine-tuning.

Downloads all 8 SuperGLUE tasks from HuggingFace, formats each example
as natural language text (matching the original benchmark presentation),
tokenizes with the GPT-2 tokenizer, and saves as a HuggingFace Dataset
with train/test splits.

Each example is stored individually with:
  - input_ids:      full prompt + answer, padded to context_size
  - attention_mask:  1 for real tokens, 0 for padding
  - labels:         -100 for prompt tokens and padding (masked),
                     real token IDs only for answer tokens

This ensures fine-tuning loss is computed only on the answer tokens.

Tasks included:
  - BoolQ:   passage + yes/no question
  - CB:      premise/hypothesis → entailment/contradiction/neutral
  - COPA:    premise + two choices → causal reasoning
  - MultiRC: passage + question + answer candidate → correct/incorrect
  - ReCoRD:  passage + cloze query → entity answer
  - RTE:     premise/hypothesis → entailment/not_entailment
  - WiC:     word + two sentences → same sense?
  - WSC:     sentence + coreference resolution

Usage:
    PYTHONPATH=./src python src/tiered/data/prepare_superglue.py \\
        --output-dir /path/to/output/superglue \\
        --context-size 1024
"""

import argparse
import os
import random

import tiktoken
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


# ─── Task formatters ─────────────────────────────────────────────────────────
# Each formatter returns (prompt: str, answer: str).
# Only the answer tokens will receive gradient signal during training.

LABEL_MAPS = {
    "boolq": {0: "No", 1: "Yes"},
    "cb": {0: "entailment", 1: "contradiction", 2: "neutral"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "wic": {0: "No", 1: "Yes"},
    "wsc": {0: "No", 1: "Yes"},
    "multirc": {0: "No", 1: "Yes"},
}


def format_boolq(ex):
    """BoolQ: passage + yes/no question."""
    label = LABEL_MAPS["boolq"].get(ex["label"])
    if label is None:
        return None
    prompt = (
        f"Passage: {ex['passage']}\n"
        f"Question: {ex['question']}\n"
        f"Answer:"
    )
    return prompt, f" {label}"


def format_cb(ex):
    """CB (CommitmentBank): premise/hypothesis → 3-way entailment."""
    label = LABEL_MAPS["cb"].get(ex["label"])
    if label is None:
        return None
    prompt = (
        f"Premise: {ex['premise']}\n"
        f"Hypothesis: {ex['hypothesis']}\n"
        f"Answer:"
    )
    return prompt, f" {label}"


def format_copa(ex):
    """COPA: premise + two alternative causes/effects."""
    question_type = "cause" if ex["question"] == "cause" else "effect"
    choice = ex["choice1"] if ex["label"] == 0 else ex["choice2"]
    prompt = (
        f"Premise: {ex['premise']}\n"
        f"What is the {question_type}?\n"
        f"Choice 1: {ex['choice1']}\n"
        f"Choice 2: {ex['choice2']}\n"
        f"Answer:"
    )
    return prompt, f" {choice}"


def format_multirc(ex):
    """MultiRC: passage + question + answer candidate → correct?"""
    label = LABEL_MAPS["multirc"].get(ex["label"])
    if label is None:
        return None
    prompt = (
        f"Passage: {ex['paragraph']}\n"
        f"Question: {ex['question']}\n"
        f"Answer candidate: {ex['answer']}\n"
        f"Correct:"
    )
    return prompt, f" {label}"


def format_record(ex):
    """ReCoRD: passage + cloze-style query → entity answer."""
    answers = ex.get("answers", [])
    if not answers:
        return None
    answer = answers[0]
    prompt = (
        f"Passage: {ex['passage']}\n"
        f"Query: {ex['query']}\n"
        f"Answer:"
    )
    return prompt, f" {answer}"


def format_rte(ex):
    """RTE: premise/hypothesis → entailment or not."""
    label = LABEL_MAPS["rte"].get(ex["label"])
    if label is None:
        return None
    prompt = (
        f"Premise: {ex['premise']}\n"
        f"Hypothesis: {ex['hypothesis']}\n"
        f"Answer:"
    )
    return prompt, f" {label}"


def format_wic(ex):
    """WiC: word + two sentences → same sense?"""
    label = LABEL_MAPS["wic"].get(ex["label"])
    if label is None:
        return None
    prompt = (
        f"Word: {ex['word']}\n"
        f"Sentence 1: {ex['sentence1']}\n"
        f"Sentence 2: {ex['sentence2']}\n"
        f"Same meaning:"
    )
    return prompt, f" {label}"


def format_wsc(ex):
    """WSC: coreference resolution."""
    label = LABEL_MAPS["wsc"].get(ex["label"])
    if label is None:
        return None
    prompt = (
        f"{ex['text']}\n"
        f"Does \"{ex['span1_text']}\" refer to \"{ex['span2_text']}\"?\n"
        f"Answer:"
    )
    return prompt, f" {label}"


TASK_FORMATTERS = {
    "boolq": format_boolq,
    "cb": format_cb,
    "copa": format_copa,
    "multirc": format_multirc,
    "record": format_record,
    "rte": format_rte,
    "wic": format_wic,
    "wsc": format_wsc,
}

ALL_TASKS = list(TASK_FORMATTERS.keys())


# ─── Tokenization ────────────────────────────────────────────────────────────


def tokenize_example(prompt, answer, tokenizer, max_len):
    """Tokenize a single (prompt, answer) pair with answer-only labels.

    Returns:
        dict with input_ids, attention_mask, labels — all of length max_len.
        Labels are -100 for prompt tokens and padding, real IDs for answer tokens.
        Returns None if the example doesn't fit in max_len.
    """
    prompt_tokens = tokenizer.encode_ordinary(prompt)
    answer_tokens = tokenizer.encode_ordinary(answer)

    # Add EOT after the answer
    answer_tokens.append(tokenizer.eot_token)

    total_len = len(prompt_tokens) + len(answer_tokens)

    if total_len > max_len:
        # Truncate the prompt to fit (keep answer intact)
        max_prompt_len = max_len - len(answer_tokens)
        if max_prompt_len < 10:
            # Answer alone is too long, skip this example
            return None
        prompt_tokens = prompt_tokens[:max_prompt_len]
        total_len = len(prompt_tokens) + len(answer_tokens)

    # Build input_ids: prompt + answer + padding
    pad_len = max_len - total_len
    input_ids = prompt_tokens + answer_tokens + [tokenizer.eot_token] * pad_len

    # Build attention_mask: 1 for real tokens, 0 for padding
    attention_mask = [1] * total_len + [0] * pad_len

    # Build labels: -100 for prompt and padding, real IDs for answer
    labels = (
        [-100] * len(prompt_tokens)
        + answer_tokens
        + [-100] * pad_len
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SuperGLUE benchmark data for private fine-tuning"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the tokenized dataset",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=1024,
        help="Max sequence length (default: 1024, matching pretraining)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=ALL_TASKS,
        help=f"SuperGLUE tasks to include (default: all). Options: {ALL_TASKS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")

    # Collect tokenized examples per split
    train_examples = []
    val_examples = []

    for task_name in args.tasks:
        formatter = TASK_FORMATTERS[task_name]
        print(f"\n{'='*60}")
        print(f"Loading super_glue/{task_name}...")

        ds = load_dataset("super_glue", task_name)

        # Process train split
        train_count = 0
        skipped = 0
        for ex in tqdm(ds["train"], desc=f"  {task_name} train"):
            result = formatter(ex)
            if result is None:
                skipped += 1
                continue
            prompt, answer = result
            tokenized = tokenize_example(prompt, answer, tokenizer, args.context_size)
            if tokenized is not None:
                train_examples.append(tokenized)
                train_count += 1
            else:
                skipped += 1

        # Process validation split (SuperGLUE test labels are hidden)
        val_count = 0
        for ex in tqdm(ds["validation"], desc=f"  {task_name} val"):
            result = formatter(ex)
            if result is None:
                skipped += 1
                continue
            prompt, answer = result
            tokenized = tokenize_example(prompt, answer, tokenizer, args.context_size)
            if tokenized is not None:
                val_examples.append(tokenized)
                val_count += 1
            else:
                skipped += 1

        print(f"  {task_name}: {train_count} train, {val_count} val ({skipped} skipped)")

    print(f"\n{'='*60}")
    print(f"Total: {len(train_examples)} train, {len(val_examples)} val examples")

    # Shuffle
    random.shuffle(train_examples)
    random.shuffle(val_examples)

    # Print sample examples with label masking visualization
    print(f"\n--- Sample formatted examples (showing answer-only masking) ---")
    for i in range(min(3, len(train_examples))):
        ex = train_examples[i]
        # Decode the full sequence
        real_tokens = [t for t, m in zip(ex["input_ids"], ex["attention_mask"]) if m == 1]
        full_text = tokenizer.decode(real_tokens)
        # Decode only the labeled (answer) tokens
        answer_tokens = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
        answer_text = tokenizer.decode(answer_tokens)
        print(f"\n[Example {i+1}]")
        print(f"  Full:   {full_text[:200]}...")
        print(f"  Answer: {answer_text}")
        prompt_count = sum(1 for l in ex["labels"] if l == -100)
        answer_count = sum(1 for l in ex["labels"] if l != -100)
        print(f"  Tokens: {prompt_count} masked (prompt+pad), {answer_count} labeled (answer)")

    # Create HuggingFace Datasets
    train_dataset = Dataset.from_dict({
        "input_ids": [ex["input_ids"] for ex in train_examples],
        "attention_mask": [ex["attention_mask"] for ex in train_examples],
        "labels": [ex["labels"] for ex in train_examples],
    })
    val_dataset = Dataset.from_dict({
        "input_ids": [ex["input_ids"] for ex in val_examples],
        "attention_mask": [ex["attention_mask"] for ex in val_examples],
        "labels": [ex["labels"] for ex in val_examples],
    })

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": val_dataset,
    })

    # Save
    print(f"\nSaving to {args.output_dir}...")
    dataset_dict.save_to_disk(args.output_dir)

    print(f"\nDataset columns: {train_dataset.column_names}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples:   {len(val_dataset)}")
    print("Done!")


if __name__ == "__main__":
    main()
