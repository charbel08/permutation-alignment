"""
Prepare WMDP forget corpora for private finetuning.

Downloads bio-forget and cyber-forget from cais/wmdp-corpora,
tokenizes with GPT-2 tiktoken, concatenates documents with EOT,
and chunks into fixed-size blocks.

Usage:
    python -m tiered.data.prepare_wmdp \
        --output-dir /work/scratch/data/datasets/wmdp \
        --chunk-size 1024
"""

import argparse
import os
import random

import numpy as np
import tiktoken
from datasets import DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm


def tokenize_and_chunk(examples, tokenizer, chunk_size, eot):
    """Tokenize texts and create fixed-size chunks by concatenating documents."""
    all_chunks = []
    all_attention_masks = []

    all_encoded = tokenizer.encode_ordinary_batch(examples["text"])

    all_tokens = []
    for tokens in all_encoded:
        all_tokens.extend(tokens)
        all_tokens.append(eot)

    for i in range(0, len(all_tokens), chunk_size):
        chunk = all_tokens[i : i + chunk_size]
        if len(chunk) == chunk_size:
            all_chunks.append(chunk)
            all_attention_masks.append([1] * chunk_size)

    return {
        "input_ids": all_chunks,
        "attention_mask": all_attention_masks,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare WMDP forget corpora for private finetuning")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for tokenized dataset")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--test-fraction", type=float, default=0.05,
                        help="Fraction of data for test split")
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token

    # ── Download forget corpora ───────────────────────────────────────────────
    print(f"{'='*70}")
    print("PHASE 1: DOWNLOAD WMDP FORGET CORPORA")
    print(f"{'='*70}")

    print("Loading bio-forget-corpus...")
    bio_forget = load_dataset("cais/wmdp-corpora", "bio-forget-corpus", split="train")
    print(f"  bio-forget: {len(bio_forget)} documents")

    print("Loading cyber-forget-corpus...")
    cyber_forget = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train")
    print(f"  cyber-forget: {len(cyber_forget)} documents")

    # Combine bio + cyber forget into one dataset
    # Both have a 'text' column
    forget_ds = concatenate_datasets([bio_forget, cyber_forget])
    forget_ds = forget_ds.shuffle(seed=args.seed)
    print(f"  Combined forget: {len(forget_ds)} documents")

    # ── Tokenize + chunk ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE 2: TOKENIZE + CHUNK (chunk_size={args.chunk_size})")
    print(f"{'='*70}")

    tokenized_ds = forget_ds.map(
        lambda examples: tokenize_and_chunk(examples, tokenizer, args.chunk_size, eot),
        batched=True,
        batch_size=500,
        remove_columns=forget_ds.column_names,
        num_proc=args.num_proc,
        desc="Tokenizing forget corpora",
    )

    print(f"  Total chunks: {len(tokenized_ds):,}")
    print(f"  Total tokens: {len(tokenized_ds) * args.chunk_size:,}")

    # ── Train/test split ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 3: ASSEMBLE DATASET")
    print(f"{'='*70}")

    n_test = max(1, int(len(tokenized_ds) * args.test_fraction))
    train_ds = tokenized_ds.select(range(n_test, len(tokenized_ds)))
    test_ds = tokenized_ds.select(range(n_test))

    print(f"  Train: {len(train_ds):,} chunks ({len(train_ds) * args.chunk_size:,} tokens)")
    print(f"  Test:  {len(test_ds):,} chunks ({len(test_ds) * args.chunk_size:,} tokens)")

    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

    save_path = os.path.join(args.output_dir, "forget")
    print(f"Saving to {save_path}...")
    dataset_dict.save_to_disk(save_path, num_proc=2)

    print(f"\nDone! Dataset saved to {save_path}")
    print(f"Use with: --private_data {save_path}")


if __name__ == "__main__":
    main()
