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

import numpy as np
import tiktoken
from datasets import DatasetDict, load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb for pretraining")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=100_000_000_000)
    parser.add_argument("--subset", type=str, default="sample-100BT",
                        help="sample-10BT, sample-100BT, sample-350BT, or default")
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument("--num-proc", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

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

    # ── Process ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE 2: TOKENIZE + CHUNK ({args.num_proc} workers)")
    print(f"{'='*70}")

    tokenizer = tiktoken.get_encoding("gpt2")
    eot = tokenizer.eot_token

    print("Loading parquet files as a single dataset...")
    # Load dataset directly from parquet; this streams efficiently via HF datasets
    ds = load_dataset("parquet", data_files=parquet_files, split="train", num_proc=args.num_proc)

    def tokenize_and_chunk(examples):
        """Tokenize texts and create chunks by merging multiple documents."""
        all_chunks = []
        all_attention_masks = []
        
        # Batch tokenize with tiktoken for speed
        all_encoded = tokenizer.encode_ordinary_batch(examples["text"])
        
        all_tokens = []
        for tokens in all_encoded:
            all_tokens.extend(tokens)
            all_tokens.append(eot)

        # Split the concatenated tokens into chunks
        for i in range(0, len(all_tokens), args.chunk_size):
            chunk = all_tokens[i : i + args.chunk_size]
            if len(chunk) == args.chunk_size:
                all_chunks.append(chunk)
                all_attention_masks.append([1] * args.chunk_size)

        return {
            "input_ids": all_chunks,
            "attention_mask": all_attention_masks,
        }

    print("Running parallel tokenization and chunking mapping...")
    tokenized_ds = ds.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
        desc="Tokenizing",
    )

    print(f"\n{'='*70}")
    print(f"PHASE 3: ASSEMBLE DATASET")
    print(f"{'='*70}")

    max_chunks = args.max_tokens // args.chunk_size
    if len(tokenized_ds) > max_chunks:
        print(f"Selecting required chunks: {max_chunks:,} chunks ({args.max_tokens:,} tokens)...")
        tokenized_ds = tokenized_ds.select(range(max_chunks))

    print("Shuffling chunks...")
    tokenized_ds = tokenized_ds.shuffle(seed=args.seed)

    n_test = max(1, int(len(tokenized_ds) * args.test_fraction))
    train_ds = tokenized_ds.select(range(n_test, len(tokenized_ds)))
    test_ds = tokenized_ds.select(range(n_test))

    print(f"  Train: {len(train_ds):,} chunks ({len(train_ds) * args.chunk_size:,} tokens)")
    print(f"  Test:  {len(test_ds):,} chunks ({len(test_ds) * args.chunk_size:,} tokens)")

    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

    save_path = os.path.join(args.output_dir, "retain")
    print(f"Saving to {save_path}...")
    dataset_dict.save_to_disk(save_path, num_proc=args.num_proc)

    print("Done!")

if __name__ == "__main__":
    main()