"""
Prepare multilingual FineWeb2 subsets for private fine-tuning.

Downloads each language subset via HF datasets, tokenizes in parallel,
chunks, and saves as HF DatasetDict with train/test splits.

Example:
    python -m tiered.data.prepare_fineweb2_multilingual \
        --output-dir /work/scratch/data/datasets/fineweb2_private \
        --languages spa_Latn deu_Latn jpn_Jpan tur_Latn \
        --chunk-size 2048 \
        --max-tokens-per-language 5000000000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tiktoken
from datasets import DatasetDict, load_dataset


def _prepare_one_language(
    lang: str, output_dir: str, tokenizer: tiktoken.Encoding,
    chunk_size: int, max_tokens: int, test_fraction: float, num_proc: int,
) -> None:
    max_chunks = max_tokens // chunk_size
    lang_dir = os.path.join(output_dir, lang)
    retain_dir = os.path.join(lang_dir, "retain")

    print(f"\n{'=' * 70}")
    print(f"{lang}: target {max_chunks:,} chunks ({max_tokens:,} tokens)")
    print(f"{'=' * 70}")

    ds = load_dataset("HuggingFaceFW/fineweb-2", name=lang, split="train")
    print(f"  Loaded {len(ds):,} documents")

    eot = tokenizer.eot_token

    def tokenize_and_chunk(examples):
        all_chunks, all_masks = [], []
        all_tokens = []
        for tokens in tokenizer.encode_ordinary_batch(examples["text"]):
            all_tokens.extend(tokens)
            all_tokens.append(eot)
        for i in range(0, len(all_tokens), chunk_size):
            chunk = all_tokens[i : i + chunk_size]
            if len(chunk) == chunk_size:
                all_chunks.append(chunk)
                all_masks.append([1] * chunk_size)
        return {"input_ids": all_chunks, "attention_mask": all_masks}

    print(f"Tokenizing and chunking ({num_proc} workers)...")
    tokenized = ds.map(
        tokenize_and_chunk, batched=True, batch_size=1000,
        remove_columns=ds.column_names, num_proc=num_proc,
        desc=f"Tokenizing {lang}",
    )

    if len(tokenized) > max_chunks:
        tokenized = tokenized.select(range(max_chunks))

    n = len(tokenized)
    n_test = max(1, int(n * test_fraction))
    print(f"  Train: {n - n_test:,}, Test: {n_test:,} ({n * chunk_size:,} tokens)")

    print(f"Saving to {retain_dir}...")
    DatasetDict({
        "train": tokenized.select(range(n_test, n)),
        "test": tokenized.select(range(n_test)),
    }).save_to_disk(retain_dir, num_proc=2)

    (Path(lang_dir) / "stats.json").write_text(json.dumps({
        "language": lang, "chunk_size": chunk_size,
        "chunks": n, "tokens": n * chunk_size,
    }, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--languages", nargs="+", default=["spa_Latn"])
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--max-tokens-per-language", type=int, default=5_000_000_000)
    p.add_argument("--test-fraction", type=float, default=0.005)
    p.add_argument("--num-proc", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except ImportError:
        pass

    tokenizer = tiktoken.get_encoding("gpt2")

    for lang in args.languages:
        _prepare_one_language(
            lang, args.output_dir, tokenizer, args.chunk_size,
            args.max_tokens_per_language, args.test_fraction, args.num_proc,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
