"""
Prepare FineWeb for pretraining.

FineWeb is already filtered and deduplicated — no quality filtering needed.
Just download, tokenize, chunk, and save.

Usage:
    export HF_HUB_ENABLE_HF_TRANSFER=1
    python -m tiered.data.prepare_fineweb \
        --output-dir /work/scratch/data/datasets/fineweb \
        --chunk-size 1024 \
        --max-tokens 100000000000 \
        --subset sample-100BT
"""

import argparse
import glob
import os
import random
import shutil

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm


def save_shard(chunks, masks, shard_dir, shard_idx):
    ds = Dataset.from_dict({"input_ids": chunks, "attention_mask": masks})
    path = os.path.join(shard_dir, f"shard_{shard_idx:05d}")
    ds.save_to_disk(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb for pretraining")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=100_000_000_000)
    parser.add_argument("--subset", type=str, default="sample-100BT",
                        help="sample-10BT, sample-100BT, sample-350BT, or default")
    parser.add_argument("--shard-size", type=int, default=500_000)
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_dir = os.path.join(args.output_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"PHASE 1: DOWNLOAD FineWeb ({args.subset})")
    print(f"{'='*70}")

    subset_to_path = {
        "sample-10BT": "sample/10BT",
        "sample-100BT": "sample/100BT",
        "sample-350BT": "sample/350BT",
        "default": "data",
    }
    hf_path = subset_to_path.get(args.subset, args.subset)

    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("  hf_transfer enabled")
    except ImportError:
        print("  WARNING: pip install hf_transfer for faster downloads")

    from huggingface_hub import snapshot_download

    download_dir = os.path.join(args.output_dir, "_downloads")
    print(f"  Downloading to {download_dir}...")

    local_dir = snapshot_download(
        repo_id="HuggingFaceFW/fineweb",
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=f"{hf_path}/*.parquet",
    )

    parquet_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.parquet"), recursive=True))
    print(f"  Found {len(parquet_files)} parquet files")
    random.shuffle(parquet_files)

    # ── Process ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE 2: TOKENIZE + CHUNK")
    print(f"{'='*70}")

    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token

    chunk_buffer = []
    mask_buffer = []
    shard_paths = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0
    leftover_tokens = np.array([], dtype=np.int32)

    pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True, desc="Tokens")
    file_pbar = tqdm(parquet_files, desc="Files", position=1)

    for pf in file_pbar:
        # Read just the text column
        try:
            table = pq.read_table(pf, columns=["text"])
            texts = table.column("text").to_pylist()
        except Exception as e:
            tqdm.write(f"  Skipping {pf}: {e}")
            continue

        if not texts:
            continue

        total_docs += len(texts)

        # Batch tokenize — tiktoken uses threads internally, this is fast
        all_encoded = tokenizer.encode_ordinary_batch(texts)

        # Flatten with EOT separators into numpy array
        total_len = sum(len(t) + 1 for t in all_encoded) + len(leftover_tokens)
        all_tokens = np.empty(total_len, dtype=np.int32)

        # Prepend leftover from previous file
        idx = len(leftover_tokens)
        if idx > 0:
            all_tokens[:idx] = leftover_tokens

        for tokens in all_encoded:
            n = len(tokens)
            all_tokens[idx : idx + n] = tokens
            all_tokens[idx + n] = eot
            idx += n + 1
        all_tokens = all_tokens[:idx]

        # Chunk
        n_chunks = len(all_tokens) // args.chunk_size
        if n_chunks > 0:
            usable = n_chunks * args.chunk_size
            chunks = all_tokens[:usable].reshape(n_chunks, args.chunk_size).tolist()
            leftover_tokens = all_tokens[usable:]

            chunk_buffer.extend(chunks)
            mask_buffer.extend([[1] * args.chunk_size] * n_chunks)
            new_tokens = n_chunks * args.chunk_size
            total_tokens += new_tokens
            pbar.update(new_tokens)
        else:
            leftover_tokens = all_tokens

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
            break

    file_pbar.close()

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

    # ── Assemble ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE 3: ASSEMBLE DATASET")
    print(f"{'='*70}")

    all_shards = [load_from_disk(p) for p in shard_paths]
    random.shuffle(all_shards)

    n_test = max(1, int(len(all_shards) * args.test_fraction))
    train_shards = all_shards[n_test:]
    test_shards = all_shards[:n_test]

    train_dataset = concatenate_datasets(train_shards)
    test_dataset = concatenate_datasets(test_shards)

    print(f"  Train: {len(train_dataset):,} chunks ({len(train_dataset) * args.chunk_size:,} tokens)")
    print(f"  Test:  {len(test_dataset):,} chunks ({len(test_dataset) * args.chunk_size:,} tokens)")

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    save_path = os.path.join(args.output_dir, "retain")
    print(f"Saving to {save_path}...")
    dataset_dict.save_to_disk(save_path)

    print("Cleaning up shards...")
    shutil.rmtree(shard_dir)

    print("Done!")


if __name__ == "__main__":
    main()