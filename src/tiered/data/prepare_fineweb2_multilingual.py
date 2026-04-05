"""
Prepare multilingual FineWeb2 subsets for private fine-tuning.

Downloads a bounded portion of each FineWeb2 language subset from HF,
tokenizes in parallel with GPT-2 tiktoken, chunks to fixed context length,
and writes per-language HF DatasetDicts with train/test splits.

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare multilingual FineWeb2 subsets for private fine-tuning"
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["spa_Latn"],
        help="FineWeb2 language-script subset names (e.g., spa_Latn, deu_Latn)",
    )
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--max-tokens-per-language",
        type=int,
        default=5_000_000_000,
        help="Max tokens to produce per language (default: 5B)",
    )
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument("--num-proc", type=int, default=32)
    parser.add_argument(
        "--tokens-per-doc-estimate",
        type=int,
        default=200,
        help="Estimated avg tokens per document (used to limit download size)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prepare_one_language(
    lang: str,
    output_dir: str,
    tokenizer: tiktoken.Encoding,
    chunk_size: int,
    max_tokens: int,
    test_fraction: float,
    num_proc: int,
    tokens_per_doc_estimate: int,
) -> None:
    max_chunks = max_tokens // chunk_size
    lang_dir = os.path.join(output_dir, lang)
    retain_dir = os.path.join(lang_dir, "retain")
    os.makedirs(lang_dir, exist_ok=True)

    # Download 2x estimated docs needed, to account for variance in doc length
    docs_needed = (max_tokens // tokens_per_doc_estimate) * 2
    split = f"train[:{docs_needed}]"

    print(f"\n{'=' * 70}")
    print(f"Preparing: {lang}")
    print(f"Target: {max_chunks:,} chunks ({max_tokens:,} tokens)")
    print(f"Downloading up to {docs_needed:,} docs ({split})")
    print(f"{'=' * 70}")

    ds = load_dataset("HuggingFaceFW/fineweb-2", lang, split=split)
    print(f"  Loaded {len(ds):,} documents")

    # ── Tokenize + chunk (parallel) ──────────────────────────────────────
    eot = tokenizer.eot_token

    def tokenize_and_chunk(examples):
        all_chunks = []
        all_attention_masks = []

        all_tokens = []
        for tokens in tokenizer.encode_ordinary_batch(examples["text"]):
            all_tokens.extend(tokens)
            all_tokens.append(eot)

        for i in range(0, len(all_tokens), chunk_size):
            chunk = all_tokens[i : i + chunk_size]
            if len(chunk) == chunk_size:
                all_chunks.append(chunk)
                all_attention_masks.append([1] * chunk_size)

        return {"input_ids": all_chunks, "attention_mask": all_attention_masks}

    print(f"Tokenizing and chunking ({num_proc} workers)...")
    tokenized_ds = ds.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc=f"Tokenizing {lang}",
    )

    # ── Trim + split ─────────────────────────────────────────────────────
    if len(tokenized_ds) > max_chunks:
        print(f"Trimming to {max_chunks:,} chunks ({max_tokens:,} tokens)")
        tokenized_ds = tokenized_ds.select(range(max_chunks))
    elif len(tokenized_ds) < max_chunks:
        print(
            f"  WARNING: only produced {len(tokenized_ds):,} chunks "
            f"({len(tokenized_ds) * chunk_size:,} tokens), "
            f"target was {max_chunks:,} chunks. "
            f"Try increasing --tokens-per-doc-estimate or the dataset may not have enough data."
        )

    n_test = max(1, int(len(tokenized_ds) * test_fraction))
    train_ds = tokenized_ds.select(range(n_test, len(tokenized_ds)))
    test_ds = tokenized_ds.select(range(n_test))

    print(f"  Train: {len(train_ds):,} chunks ({len(train_ds) * chunk_size:,} tokens)")
    print(f"  Test:  {len(test_ds):,} chunks ({len(test_ds) * chunk_size:,} tokens)")

    print(f"Saving to {retain_dir}...")
    DatasetDict({"train": train_ds, "test": test_ds}).save_to_disk(retain_dir, num_proc=2)

    stats = {
        "language_subset": lang,
        "chunk_size": chunk_size,
        "total_chunks": len(tokenized_ds),
        "tokens_written": len(tokenized_ds) * chunk_size,
        "train_chunks": len(train_ds),
        "test_chunks": len(test_ds),
    }
    (Path(lang_dir) / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("hf_transfer enabled")
    except ImportError:
        print("hf_transfer not installed; using standard HF download path")

    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"{'=' * 70}")
    print("FineWeb2 data preparation")
    print(f"Output dir: {args.output_dir}")
    print(f"Languages:  {', '.join(args.languages)}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Max tokens: {args.max_tokens_per_language:,} per language")
    print(f"{'=' * 70}")

    for lang in args.languages:
        _prepare_one_language(
            lang=lang,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            chunk_size=args.chunk_size,
            max_tokens=args.max_tokens_per_language,
            test_fraction=args.test_fraction,
            num_proc=args.num_proc,
            tokens_per_doc_estimate=args.tokens_per_doc_estimate,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
