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
from multiprocessing import Pool

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


def process_parquet_file(args_tuple):
    parquet_path, chunk_size, shard_dir, max_shard_size = args_tuple
    
    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token
    
    shard_paths = []
    total_tokens_saved = 0
    total_docs = 0
    
    basename = os.path.basename(parquet_path).replace(".parquet", "")
    
    try:
        pf = pq.ParquetFile(parquet_path)
    except Exception as e:
        return [], 0, 0
        
    chunk_buffer = []
    mask_buffer = []
    leftover_tokens = np.array([], dtype=np.int32)
    shard_counter = 0

    def flush_buffer(force=False):
        nonlocal chunk_buffer, mask_buffer, shard_paths, total_tokens_saved, shard_counter
        while len(chunk_buffer) >= max_shard_size or (force and len(chunk_buffer) > 0):
            n = min(len(chunk_buffer), max_shard_size)
            shard_chunks = chunk_buffer[:n]
            shard_masks = mask_buffer[:n]
            
            # Save the shard immediately to the disk.
            ds = Dataset.from_dict({"input_ids": shard_chunks, "attention_mask": shard_masks})
            path = os.path.join(shard_dir, f"shard_{basename}_{shard_counter}")
            ds.save_to_disk(path)
            shard_paths.append(path)
            total_tokens_saved += len(shard_chunks) * chunk_size
            shard_counter += 1
            
            chunk_buffer = chunk_buffer[n:]
            mask_buffer = mask_buffer[n:]

    try:
        # iter_batches streams the file instead of reading all 3GB into RAM at once to completely eliminate OOM.
        for batch in pf.iter_batches(batch_size=10000, columns=["text"]):
            texts = batch.column("text").to_pylist()
            if not texts:
                continue
                
            total_docs += len(texts)
            all_encoded = tokenizer.encode_ordinary_batch(texts)
            
            total_len = sum(len(t) + 1 for t in all_encoded) + len(leftover_tokens)
            all_tokens = np.empty(total_len, dtype=np.int32)
            
            idx = len(leftover_tokens)
            if idx > 0:
                all_tokens[:idx] = leftover_tokens
                
            for tokens in all_encoded:
                n = len(tokens)
                all_tokens[idx : idx + n] = tokens
                all_tokens[idx + n] = eot
                idx += n + 1
            all_tokens = all_tokens[:idx]    
                
            n_chunks = len(all_tokens) // chunk_size
            if n_chunks > 0:
                usable = n_chunks * chunk_size
                chunks = all_tokens[:usable].reshape(n_chunks, chunk_size).tolist()
                chunk_buffer.extend(chunks)
                mask_buffer.extend([[1] * chunk_size] * n_chunks)
                leftover_tokens = all_tokens[usable:]
            else:
                leftover_tokens = all_tokens
                
            flush_buffer(force=False)
            
        flush_buffer(force=True)
    except Exception as e:
        pass
        
    return shard_paths, total_tokens_saved, total_docs


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb for pretraining")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=100_000_000_000)
    parser.add_argument("--subset", type=str, default="sample-100BT",
                        help="sample-10BT, sample-100BT, sample-350BT, or default")
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
    print(f"PHASE 2: TOKENIZE + CHUNK + DISK WRITE ({args.num_proc} workers)")
    print(f"{'='*70}")

    work_items = [(pf, args.chunk_size, shard_dir, args.shard_size) for pf in parquet_files]

    shard_paths = []
    total_tokens = 0
    total_docs = 0

    pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True, desc="Tokens")

    # The issue encountered was a severe IPC serialization RAM spike + HuggingFace shuffle matrix OOM.
    # To bypass it completely, we rely exclusively on pool workers chunking their memory seamlessly 
    # to individual disk files, leaving the master process virtually empty-handed and free from OOM entirely.
    with Pool(processes=args.num_proc) as pool:
        for s_paths, t_tokens, t_docs in pool.imap_unordered(
            process_parquet_file, work_items, chunksize=1
        ):
            if not s_paths:
                continue

            shard_paths.extend(s_paths)
            total_tokens += t_tokens
            total_docs += t_docs
            pbar.update(t_tokens)

            if total_tokens >= args.max_tokens:
                pool.terminate()
                break

    pbar.close()

    print(f"\nProcessing complete:")
    print(f"  Documents: {total_docs:,}")
    print(f"  Shards created: {len(shard_paths)}")
    print(f"  Tokens chunked: {total_tokens:,}")

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
    print(f"Compiling fast pointers to shards in {save_path}...")
    dataset_dict.save_to_disk(save_path)

    print("Cleaning up intermediate shards...")
    try:
        shutil.rmtree(shard_dir)
    except Exception as e:
        print(f"Could not cleanly remove temporary shard dir: {e}")

    print("Done!")

if __name__ == "__main__":
    main()