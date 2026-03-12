"""
Prepare multilingual FineWeb2 subsets for private fine-tuning.

This script streams one or more FineWeb2 language-script subsets from HF,
tokenizes with GPT-2 tiktoken, chunks to fixed context length, and writes
per-language HF DatasetDicts with train/test splits.

Default language choice targets three high-resource Latin-script subsets from
different language families:
    - deu_Latn (German, Indo-European/Germanic)
    - tur_Latn (Turkish, Turkic)
    - spa_Latn (Spanish, Indo-European/Romance)

Example:
    export HF_HUB_ENABLE_HF_TRANSFER=1
    python -m tiered.data.prepare_fineweb2_multilingual \
        --output-dir /work/scratch/data/datasets/fineweb2_private \
        --languages deu_Latn tur_Latn spa_Latn \
        --chunk-size 1024 \
        --max-tokens-per-language 500000000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import tiktoken
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare multilingual FineWeb2 subsets for private fine-tuning"
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["deu_Latn", "tur_Latn", "spa_Latn"],
        help="FineWeb2 language-script subset names (e.g., cmn_Hani)",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--max-tokens-per-language",
        type=int,
        default=100_000_000,
        help="Stop once this many tokens (after chunking) are produced per language",
    )
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument(
        "--shard-size-chunks",
        type=int,
        default=5000,
        help="Number of chunks to keep in memory before writing an intermediate parquet shard",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default="main",
        help="HF dataset revision/branch/tag (e.g., main or v2.1.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-intermediate-shards",
        action="store_true",
        help="Keep intermediate parquet shards under <output>/<lang>/_chunks",
    )
    parser.add_argument("--save-num-proc", type=int, default=2)
    return parser.parse_args()


def _write_parquet_shard(
    shard_path: str,
    input_ids: list[list[int]],
    attention_masks: list[list[int]],
) -> None:
    shard_ds = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }
    )
    shard_ds.to_parquet(shard_path)


def _prepare_one_language(
    language_subset: str,
    output_dir: str,
    tokenizer: tiktoken.Encoding,
    chunk_size: int,
    max_tokens_per_language: int,
    test_fraction: float,
    shard_size_chunks: int,
    dataset_revision: str,
    keep_intermediate_shards: bool,
    save_num_proc: int,
) -> None:
    max_chunks = max_tokens_per_language // chunk_size
    if max_chunks < 2:
        raise ValueError(
            f"--max-tokens-per-language must allow >=2 chunks. "
            f"Got {max_tokens_per_language} tokens with chunk size {chunk_size}."
        )

    print(f"\n{'=' * 80}")
    print(f"Preparing language subset: {language_subset}")
    print(f"{'=' * 80}")
    print(f"Target chunks: {max_chunks:,} ({max_tokens_per_language:,} tokens)")

    lang_dir = os.path.join(output_dir, language_subset)
    shard_dir = os.path.join(lang_dir, "_chunks")
    retain_dir = os.path.join(lang_dir, "retain")
    os.makedirs(shard_dir, exist_ok=True)

    ds_stream = load_dataset(
        "HuggingFaceFW/fineweb-2",
        language_subset,
        split="train",
        streaming=True,
        revision=dataset_revision,
    )

    eot = tokenizer.eot_token
    ones_mask = [1] * chunk_size
    shard_input_ids: list[list[int]] = []
    shard_attention_masks: list[list[int]] = []

    token_buffer: list[int] = []
    buffer_start = 0
    shard_idx = 0
    chunks_written = 0
    docs_seen = 0

    pbar = tqdm(ds_stream, desc=f"{language_subset}: tokenizing", unit="doc")
    for ex in pbar:
        text = ex.get("text")
        if not text:
            continue

        docs_seen += 1
        token_buffer.extend(tokenizer.encode_ordinary(text))
        token_buffer.append(eot)

        while (len(token_buffer) - buffer_start) >= chunk_size and chunks_written < max_chunks:
            end = buffer_start + chunk_size
            shard_input_ids.append(token_buffer[buffer_start:end])
            shard_attention_masks.append(ones_mask)
            buffer_start = end
            chunks_written += 1

            if len(shard_input_ids) >= shard_size_chunks:
                shard_path = os.path.join(
                    shard_dir,
                    f"{language_subset}_chunks_{shard_idx:06d}.parquet",
                )
                _write_parquet_shard(shard_path, shard_input_ids, shard_attention_masks)
                shard_idx += 1
                shard_input_ids = []
                shard_attention_masks = []

            if chunks_written >= max_chunks:
                break

        if buffer_start > 1_000_000:
            token_buffer = token_buffer[buffer_start:]
            buffer_start = 0

        pbar.set_postfix(chunks=f"{chunks_written:,}")
        if chunks_written >= max_chunks:
            break

    if shard_input_ids:
        shard_path = os.path.join(
            shard_dir,
            f"{language_subset}_chunks_{shard_idx:06d}.parquet",
        )
        _write_parquet_shard(shard_path, shard_input_ids, shard_attention_masks)

    print(f"  Documents consumed: {docs_seen:,}")
    print(f"  Chunks written:     {chunks_written:,}")
    print(f"  Tokens written:     {chunks_written * chunk_size:,}")

    if chunks_written < 2:
        raise RuntimeError(
            f"Too few chunks generated for {language_subset} ({chunks_written}). "
            f"Increase max tokens or verify subset availability."
        )

    parquet_files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not parquet_files:
        raise RuntimeError(f"No parquet shards found for {language_subset} in {shard_dir}")

    print(f"Loading {len(parquet_files):,} local shard files for final save...")
    chunks_ds = load_dataset("parquet", data_files=parquet_files, split="train")

    n_test = max(1, int(len(chunks_ds) * test_fraction))
    if n_test >= len(chunks_ds):
        n_test = 1

    train_ds = chunks_ds.select(range(n_test, len(chunks_ds)))
    test_ds = chunks_ds.select(range(n_test))
    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

    print(f"Saving tokenized dataset to: {retain_dir}")
    dataset_dict.save_to_disk(retain_dir, num_proc=save_num_proc)

    stats = {
        "language_subset": language_subset,
        "docs_consumed": docs_seen,
        "chunk_size": chunk_size,
        "chunks_written": chunks_written,
        "tokens_written": chunks_written * chunk_size,
        "train_chunks": len(train_ds),
        "test_chunks": len(test_ds),
        "dataset_revision": dataset_revision,
    }
    stats_path = Path(lang_dir) / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote stats to: {stats_path}")

    if not keep_intermediate_shards:
        print("Removing intermediate shard files...")
        for file_path in parquet_files:
            os.remove(file_path)
        try:
            os.rmdir(shard_dir)
        except OSError:
            pass


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

    print(f"{'=' * 80}")
    print("FineWeb2 multilingual tokenizer/chunker")
    print(f"Output dir: {args.output_dir}")
    print(f"Languages:  {', '.join(args.languages)}")
    print(f"Revision:   {args.dataset_revision}")
    print(f"{'=' * 80}")

    for language_subset in args.languages:
        _prepare_one_language(
            language_subset=language_subset,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            chunk_size=args.chunk_size,
            max_tokens_per_language=args.max_tokens_per_language,
            test_fraction=args.test_fraction,
            shard_size_chunks=args.shard_size_chunks,
            dataset_revision=args.dataset_revision,
            keep_intermediate_shards=args.keep_intermediate_shards,
            save_num_proc=args.save_num_proc,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
