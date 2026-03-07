"""
Prepare FineWeb for pretraining.

FineWeb is already filtered and deduplicated — no quality filtering needed.
This script just downloads, tokenizes, chunks, and saves.

Pipeline:
  1. Download FineWeb parquet files via huggingface_hub (Rust-based hf_transfer)
  2. Process parquet files in parallel: tokenize + chunk
  3. Save shards incrementally
  4. Assemble train/test DatasetDict

Usage:
    # Install deps first:
    #   pip install hf_transfer huggingface_hub tiktoken pyarrow datasets numpy tqdm
    #
    # Enable fast transfers:
    #   export HF_HUB_ENABLE_HF_TRANSFER=1

    python -m tiered.data.prepare_fineweb \
        --output-dir /work/scratch/data/datasets/fineweb \
        --chunk-size 1024 \
        --max-tokens 100000000000 \
        --num-proc 32
"""

import argparse
import glob
import os
import random
import shutil
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm


# ─── Processing ───────────────────────────────────────────────────────────────


def process_parquet_file(args_tuple):
    """Process a single parquet file: read texts, tokenize, chunk.

    Runs in a worker process. Returns (chunks, masks, n_docs).
    """
    parquet_path, chunk_size = args_tuple

    try:
        table = pq.read_table(parquet_path, columns=["text"])
        texts = table.column("text").to_pylist()
    except Exception as e:
        return [], [], 0

    if not texts:
        return [], [], 0

    # Tokenize in batch
    tokenizer = tiktoken.get_encoding("gpt2")
    all_encoded = tokenizer.encode_ordinary_batch(texts)

    # Flatten with EOT separators
    total_len = sum(len(t) + 1 for t in all_encoded)
    all_tokens = np.empty(total_len, dtype=np.int32)

    idx = 0
    eot = tokenizer.eot_token
    for tokens in all_encoded:
        n = len(tokens)
        all_tokens[idx : idx + n] = tokens
        all_tokens[idx + n] = eot
        idx += n + 1
    all_tokens = all_tokens[:idx]

    # Chunk
    n_chunks = len(all_tokens) // chunk_size
    if n_chunks == 0:
        return [], [], len(texts)

    truncated = all_tokens[: n_chunks * chunk_size]
    chunks = truncated.reshape(n_chunks, chunk_size).tolist()
    masks = [[1] * chunk_size] * n_chunks

    return chunks, masks, len(texts)


def save_shard(chunks, masks, shard_dir, shard_idx):
    ds = Dataset.from_dict({"input_ids": chunks, "attention_mask": masks})
    path = os.path.join(shard_dir, f"shard_{shard_idx:05d}")
    ds.save_to_disk(path)
    return path


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb for pretraining")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=100_000_000_000)
    parser.add_argument("--subset", type=str, default="sample-100BT",
                        help="FineWeb subset: sample-10BT, sample-100BT, sample-350BT, or default (default: sample-100BT)")
    parser.add_argument("--shard-size", type=int, default=500_000)
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument("--num-proc", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_dir = os.path.join(args.output_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Download parquet files
    # ══════════════════════════════════════════════════════════════════════════
    print(f"{'='*70}")
    print(f"PHASE 1: DOWNLOAD FineWeb ({args.subset})")
    print(f"{'='*70}")

    # Map subset names to their HF paths
    subset_to_path = {
        "sample-10BT": "sample/10BT",
        "sample-100BT": "sample/100BT",
        "sample-350BT": "sample/350BT",
        "default": "data",
    }
    hf_path = subset_to_path.get(args.subset, args.subset)

    # Enable hf_transfer if available
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("  hf_transfer enabled (Rust-based fast downloads)")
    except ImportError:
        print("  WARNING: hf_transfer not installed. Install it for 5-10x faster downloads:")
        print("    pip install hf_transfer")

    from huggingface_hub import snapshot_download

    download_dir = os.path.join(args.output_dir, "_downloads")
    print(f"  Downloading to {download_dir}...")
    print(f"  This may take a while for {args.subset} (~277GB for 100BT)...")

    local_dir = snapshot_download(
        repo_id="HuggingFaceFW/fineweb",
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=f"{hf_path}/*.parquet",
    )

    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(local_dir, hf_path, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        # Try without subdirectory nesting
        parquet_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True))

    print(f"  Found {len(parquet_files)} parquet files")

    # Shuffle to mix different crawl dumps
    random.shuffle(parquet_files)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Process (tokenize + chunk + shard)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"PHASE 2: PROCESS ({args.num_proc} workers)")
    print(f"{'='*70}")

    chunk_buffer = []
    mask_buffer = []
    shard_paths = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0

    pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True, desc="Tokens")

    # Build work items
    work_items = [(pf, args.chunk_size) for pf in parquet_files]

    with Pool(processes=args.num_proc) as pool:
        for chunks, masks, n_docs in pool.imap_unordered(
            process_parquet_file, work_items, chunksize=1
        ):
            if not chunks:
                continue

            total_docs += n_docs
            chunk_buffer.extend(chunks)
            mask_buffer.extend(masks)
            new_tokens = len(chunks) * args.chunk_size
            total_tokens += new_tokens
            pbar.update(new_tokens)

            # Flush shards
            while len(chunk_buffer) >= args.shard_size:
                shard_chunks = chunk_buffer[: args.shard_size]
                shard_masks = mask_buffer[: args.shard_size]
                chunk_buffer = chunk_buffer[args.shard_size :]
                mask_buffer = mask_buffer[args.shard_size :]

                path = save_shard(shard_chunks, shard_masks, shard_dir, shard_idx)
                shard_paths.append(path)
                shard_idx += 1
                tqdm.write(
                    f"  Shard {shard_idx}: {args.shard_size:,} chunks | "
                    f"Total: {total_tokens:,.0f} tok | Docs: {total_docs:,}"
                )

            if total_tokens >= args.max_tokens:
                pool.terminate()
                break

    # Flush remaining
    if chunk_buffer:
        path = save_shard(chunk_buffer, mask_buffer, shard_dir, shard_idx)
        shard_paths.append(path)
        shard_idx += 1
        print(f"  Final shard {shard_idx}: {len(chunk_buffer):,} chunks")

    pbar.close()

    print(f"\nProcessing complete:")
    print(f"  Documents: {total_docs:,}")
    print(f"  Shards:    {len(shard_paths)}")
    print(f"  Tokens:    {total_tokens:,}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: Assemble train/test split
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"PHASE 3: ASSEMBLE DATASET")
    print(f"{'='*70}")

    all_shards = [load_from_disk(p) for p in shard_paths]
    random.shuffle(all_shards)

    n_test = max(1, int(len(all_shards) * args.test_fraction))
    test_shards = all_shards[:n_test]
    train_shards = all_shards[n_test:]

    train_dataset = concatenate_datasets(train_shards)
    test_dataset = concatenate_datasets(test_shards)

    print(f"  Train: {len(train_dataset):,} chunks ({len(train_dataset) * args.chunk_size:,} tokens)")
    print(f"  Test:  {len(test_dataset):,} chunks ({len(test_dataset) * args.chunk_size:,} tokens)")

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    save_path = os.path.join(args.output_dir, "retain")
    print(f"Saving to {save_path}...")
    dataset_dict.save_to_disk(save_path)

    # Cleanup
    print("Cleaning up shards...")
    shutil.rmtree(shard_dir)

    print("Done!")


if __name__ == "__main__":
    main()