"""
Prepare multilingual FineWeb2 subsets for private fine-tuning.

Streams FineWeb2 language subsets from HF (no full download), tokenizes with
GPT-2 tiktoken, chunks to fixed context length, and writes per-language HF
DatasetDicts with train/test splits.

Example:
    python -m tiered.data.prepare_fineweb2_multilingual \
        --output-dir /work/scratch/data/datasets/fineweb2_private \
        --languages spa_Latn deu_Latn jpn_Jpan tur_Latn \
        --chunk-size 2048 \
        --max-tokens-per-language 5000000000
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_shard(shard_dir: str, shard_idx: int, chunks: list[list[int]], chunk_size: int):
    path = os.path.join(shard_dir, f"shard_{shard_idx:06d}.parquet")
    Dataset.from_dict({
        "input_ids": chunks,
        "attention_mask": [[1] * chunk_size] * len(chunks),
    }).to_parquet(path)


def _prepare_one_language(
    lang: str,
    output_dir: str,
    tokenizer: tiktoken.Encoding,
    chunk_size: int,
    max_tokens: int,
    test_fraction: float,
) -> None:
    max_chunks = max_tokens // chunk_size
    lang_dir = os.path.join(output_dir, lang)
    shard_dir = os.path.join(lang_dir, "_shards")
    retain_dir = os.path.join(lang_dir, "retain")
    os.makedirs(shard_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Preparing: {lang} (streaming)")
    print(f"Target: {max_chunks:,} chunks ({max_tokens:,} tokens)")
    print(f"{'=' * 70}")

    ds = load_dataset("HuggingFaceFW/fineweb-2", lang, split="train", streaming=True)

    eot = tokenizer.eot_token
    token_buf: list[int] = []
    shard_chunks: list[list[int]] = []
    shard_idx = 0
    n_chunks = 0

    for doc in tqdm(ds, desc=f"{lang}: tokenizing", unit="doc"):
        text = doc.get("text")
        if not text:
            continue
        token_buf.extend(tokenizer.encode_ordinary(text))
        token_buf.append(eot)

        while len(token_buf) >= chunk_size and n_chunks < max_chunks:
            shard_chunks.append(token_buf[:chunk_size])
            token_buf = token_buf[chunk_size:]
            n_chunks += 1

            if len(shard_chunks) >= 10_000:
                _write_shard(shard_dir, shard_idx, shard_chunks, chunk_size)
                shard_idx += 1
                shard_chunks = []

        if n_chunks >= max_chunks:
            break

    if shard_chunks:
        _write_shard(shard_dir, shard_idx, shard_chunks, chunk_size)

    print(f"  Chunks produced: {n_chunks:,} ({n_chunks * chunk_size:,} tokens)")
    print(f"  Shards written:  {shard_idx + 1}")

    # ── Reassemble and split ─────────────────────────────────────────────
    parquet_files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    full_ds = load_dataset("parquet", data_files=parquet_files, split="train")

    n_test = max(1, int(len(full_ds) * test_fraction))
    train_ds = full_ds.select(range(n_test, len(full_ds)))
    test_ds = full_ds.select(range(n_test))

    print(f"  Train: {len(train_ds):,} chunks ({len(train_ds) * chunk_size:,} tokens)")
    print(f"  Test:  {len(test_ds):,} chunks ({len(test_ds) * chunk_size:,} tokens)")

    DatasetDict({"train": train_ds, "test": test_ds}).save_to_disk(retain_dir, num_proc=2)

    # ── Cleanup shards ───────────────────────────────────────────────────
    for f in parquet_files:
        os.remove(f)
    try:
        os.rmdir(shard_dir)
    except OSError:
        pass

    stats = {
        "language_subset": lang,
        "chunk_size": chunk_size,
        "total_chunks": n_chunks,
        "tokens_written": n_chunks * chunk_size,
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
    print("FineWeb2 data preparation (streaming)")
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
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
