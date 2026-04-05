"""
Prepare multilingual FineWeb2 subsets for private fine-tuning.

Downloads one or more FineWeb2 language-script subsets from HF, tokenizes
with GPT-2 tiktoken, chunks to fixed context length, and writes per-language
HF DatasetDicts with train/test splits.

Uses the same parallel ds.map() approach as prepare_fineweb.py.

Example:
    python -m tiered.data.prepare_fineweb2_multilingual \
        --output-dir /work/scratch/data/datasets/fineweb2_private \
        --languages spa_Latn \
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prepare_one_language(
    language_subset: str,
    output_dir: str,
    tokenizer: tiktoken.Encoding,
    chunk_size: int,
    max_tokens: int,
    test_fraction: float,
    num_proc: int,
) -> None:
    max_chunks = max_tokens // chunk_size

    print(f"\n{'=' * 70}")
    print(f"Preparing: {language_subset}")
    print(f"{'=' * 70}")
    print(f"Target: {max_chunks:,} chunks ({max_tokens:,} tokens)")

    lang_dir = os.path.join(output_dir, language_subset)
    retain_dir = os.path.join(lang_dir, "retain")
    os.makedirs(lang_dir, exist_ok=True)

    # ── Download ─────────────────────────────────────────────────────────
    print(f"Loading {language_subset} from HuggingFaceFW/fineweb-2...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        language_subset,
        split="train",
    )
    print(f"  Loaded {len(ds):,} documents")

    # ── Tokenize + chunk ─────────────────────────────────────────────────
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
        desc=f"Tokenizing {language_subset}",
    )

    # ── Trim + split ─────────────────────────────────────────────────────
    if len(tokenized_ds) > max_chunks:
        print(f"Trimming to {max_chunks:,} chunks ({max_tokens:,} tokens)")
        tokenized_ds = tokenized_ds.select(range(max_chunks))

    n_test = max(1, int(len(tokenized_ds) * test_fraction))
    train_ds = tokenized_ds.select(range(n_test, len(tokenized_ds)))
    test_ds = tokenized_ds.select(range(n_test))

    print(f"  Train: {len(train_ds):,} chunks ({len(train_ds) * chunk_size:,} tokens)")
    print(f"  Test:  {len(test_ds):,} chunks ({len(test_ds) * chunk_size:,} tokens)")

    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

    print(f"Saving to {retain_dir}...")
    dataset_dict.save_to_disk(retain_dir, num_proc=2)

    stats = {
        "language_subset": language_subset,
        "chunk_size": chunk_size,
        "total_chunks": len(tokenized_ds),
        "tokens_written": len(tokenized_ds) * chunk_size,
        "train_chunks": len(train_ds),
        "test_chunks": len(test_ds),
    }
    stats_path = Path(lang_dir) / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote stats to {stats_path}")


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
            language_subset=lang,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            chunk_size=args.chunk_size,
            max_tokens=args.max_tokens_per_language,
            test_fraction=args.test_fraction,
            num_proc=args.num_proc,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
