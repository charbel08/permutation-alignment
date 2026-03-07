"""
Prepare RedPajama-Data-V2 for pretraining.

This script:
1. Streams RedPajama-Data-V2 from HuggingFace (English, head_middle partition)
2. Applies Gopher-style quality filtering using built-in quality signals
3. Applies deduplication filtering using built-in n-gram duplicate signals
4. Applies classifier-based filtering using the PaLM quality score
5. Tokenizes with GPT-2 tiktoken tokenizer (batched for speed)
6. Chunks into fixed-length sequences
7. Saves shards incrementally to disk (avoids OOM on large datasets)
8. Concatenates shards into a final HuggingFace DatasetDict with train/test splits

Usage:
    python -m tiered.data.prepare_redpajama \
        --output-dir /work/scratch/data/datasets/redpajama \
        --chunk-size 1024 \
        --max-tokens 100000000000 \
        --num-snapshots 10
"""

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import tiktoken
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk, load_dataset
from tqdm import tqdm


# ─── Quality filtering ───────────────────────────────────────────────────────


def _parse_signals(raw_quality_signals):
    """Parse quality signals JSON, return dict or None on failure."""
    try:
        return json.loads(raw_quality_signals)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _get(signals, key, default=0.0):
    """Safely extract a scalar quality signal value."""
    return signals.get(key, [[0, 0, default]])[0][2]


def gopher_quality_filter(sample) -> bool:
    """Apply Gopher-style quality filtering using RedPajama quality signals,
    supplemented with deduplication and classifier-based checks."""
    signals = _parse_signals(sample.get("quality_signals"))
    if signals is None:
        return False

    # ── Rule 1: word count between 50 and 100,000 ──
    word_count = _get(signals, "rps_doc_word_count")
    if word_count < 50 or word_count > 100_000:
        return False

    # ── Rule 2: mean word length between 3 and 10 ──
    if not (3 <= _get(signals, "rps_doc_mean_word_length") <= 10):
        return False

    # ── Rule 3: symbol to word ratio below 0.1 ──
    if _get(signals, "rps_doc_symbol_to_word_ratio") > 0.1:
        return False

    # ── Rule 4: 90% of lines should not start with bullet point ──
    n_lines = _get(signals, "ccnet_nlines", 1)
    if n_lines > 0:
        bullet_lines = signals.get("rps_lines_start_with_bulletpoint", [])
        n_bullet = sum(ln[2] for ln in bullet_lines)
        if n_bullet / n_lines > 0.9:
            return False

    # ── Rules 5-7: top n-gram frequency ──
    if _get(signals, "rps_doc_frac_chars_top_2gram") > 0.2:
        return False
    if _get(signals, "rps_doc_frac_chars_top_3gram") > 0.18:
        return False
    if _get(signals, "rps_doc_frac_chars_top_4gram") > 0.16:
        return False

    # ── Rule 8: n-gram deduplication ──
    dupe_thresholds = [
        ("rps_doc_frac_chars_dupe_5gram", 0.15),
        ("rps_doc_frac_chars_dupe_6gram", 0.14),
        ("rps_doc_frac_chars_dupe_7gram", 0.13),
        ("rps_doc_frac_chars_dupe_8gram", 0.12),
        ("rps_doc_frac_chars_dupe_9gram", 0.11),
        ("rps_doc_frac_chars_dupe_10gram", 0.10),
    ]
    for key, threshold in dupe_thresholds:
        if _get(signals, key) > threshold:
            return False

    # ── Rule 9: classifier-based quality filtering (PaLM score) ──
    if _get(signals, "rps_doc_ml_palm_score") < 0.5:
        return False

    # ── Rule 10: non-alphabetic word fraction ──
    if _get(signals, "rps_doc_frac_no_alph_words") > 0.3:
        return False

    # ── Rule 11: stop word presence ──
    if _get(signals, "rps_doc_stop_word_fraction") < 0.06:
        return False

    # ── Rule 12: perplexity filter via ccnet ──
    if _get(signals, "ccnet_perplexity") > 1500:
        return False

    return True


# ─── Batch filtering (runs in worker processes) ──────────────────────────────


def filter_batch(batch):
    """Filter a batch of samples, return list of passing texts."""
    texts = []
    filtered = 0
    for sample in batch:
        if not gopher_quality_filter(sample):
            filtered += 1
            continue
        text = sample.get("raw_content", "")
        if not text or len(text.strip()) < 100:
            filtered += 1
            continue
        texts.append(text)
    return texts, filtered


# ─── Tokenization ────────────────────────────────────────────────────────────


def tokenize_and_chunk(texts, tokenizer, chunk_size):
    """Tokenize texts using batched encoding and create fixed-length chunks."""
    # encode_ordinary_batch is significantly faster than per-doc encoding
    all_encoded = tokenizer.encode_ordinary_batch(texts)

    # Flatten with EOT separators — estimate total length to pre-allocate
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

    # Chunk using numpy reshape (much faster than Python slicing)
    n_chunks = len(all_tokens) // chunk_size
    if n_chunks == 0:
        return [], []

    truncated = all_tokens[: n_chunks * chunk_size]
    chunks = truncated.reshape(n_chunks, chunk_size).tolist()
    attention_masks = [[1] * chunk_size] * n_chunks

    return chunks, attention_masks


# ─── Shard saving ────────────────────────────────────────────────────────────


def save_shard(chunks, attention_masks, shard_dir, shard_idx):
    """Save a shard of data to disk as a HuggingFace Dataset."""
    ds = Dataset.from_dict({
        "input_ids": chunks,
        "attention_mask": attention_masks,
    })
    shard_path = os.path.join(shard_dir, f"shard_{shard_idx:05d}")
    ds.save_to_disk(shard_path)
    return shard_path


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prepare RedPajama-Data-V2 for pretraining")
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1024,
        help="Token chunk size (default: 1024)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100_000_000_000,
        help="Maximum number of tokens to collect (default: 100B)",
    )
    parser.add_argument(
        "--num-snapshots", type=int, default=10,
        help="Number of CC snapshots to use (default: 10)",
    )
    parser.add_argument(
        "--shard-size", type=int, default=500_000,
        help="Number of chunks per shard (default: 500K = ~500M tokens per shard)",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.005,
        help="Fraction of shards for test split (default: 0.5%%)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50_000,
        help="Number of documents to accumulate before tokenizing (default: 50000)",
    )
    parser.add_argument(
        "--filter-workers", type=int, default=8,
        help="Number of parallel workers for quality filtering (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_dir = os.path.join(args.output_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")

    # Select snapshots (recent ones tend to be higher quality)
    all_snapshots = [
        "2023-14", "2023-06", "2022-49", "2022-40", "2022-27",
        "2022-21", "2021-49", "2021-43", "2021-39", "2021-31",
    ]
    snapshots = all_snapshots[:args.num_snapshots]
    print(f"Using {len(snapshots)} snapshots: {snapshots}")

    # Stream the dataset
    print("Loading RedPajama-Data-V2 (streaming)...")
    ds = load_dataset(
        "togethercomputer/RedPajama-Data-V2",
        name="default",
        partition="head_middle",
        snapshots=snapshots,
        languages=["en"],
        streaming=True,
        trust_remote_code=True,
    )

    # Accumulate chunks, flush to disk when shard is full
    chunk_buffer = []
    mask_buffer = []
    sample_buffer = []
    shard_paths = []
    shard_idx = 0
    total_tokens = 0
    docs_processed = 0
    docs_filtered = 0

    print(f"Processing documents (target: {args.max_tokens:,} tokens)...")
    print(f"Saving shards to {shard_dir} ({args.shard_size:,} chunks per shard)")
    print(f"Batch size: {args.batch_size:,} | Filter workers: {args.filter_workers}")
    pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True)

    for sample in ds["train"]:
        sample_buffer.append(sample)

        if len(sample_buffer) >= args.batch_size:
            # ── Parallel quality filtering ──
            n_workers = args.filter_workers
            sub_batch_size = len(sample_buffer) // n_workers
            sub_batches = [
                sample_buffer[i * sub_batch_size : (i + 1) * sub_batch_size]
                for i in range(n_workers)
            ]
            # Handle remainder
            remainder = sample_buffer[n_workers * sub_batch_size :]
            if remainder:
                sub_batches.append(remainder)

            texts = []
            batch_filtered = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(filter_batch, sb) for sb in sub_batches]
                for future in as_completed(futures):
                    batch_texts, filt = future.result()
                    texts.extend(batch_texts)
                    batch_filtered += filt

            docs_filtered += batch_filtered
            docs_processed += len(texts)
            sample_buffer = []

            if not texts:
                continue

            # ── Batched tokenization ──
            chunks, masks = tokenize_and_chunk(texts, tokenizer, args.chunk_size)
            chunk_buffer.extend(chunks)
            mask_buffer.extend(masks)
            new_tokens = len(chunks) * args.chunk_size
            total_tokens += new_tokens
            pbar.update(new_tokens)

            # ── Flush shards to disk ──
            while len(chunk_buffer) >= args.shard_size:
                shard_chunks = chunk_buffer[:args.shard_size]
                shard_masks = mask_buffer[:args.shard_size]
                chunk_buffer = chunk_buffer[args.shard_size:]
                mask_buffer = mask_buffer[args.shard_size:]

                path = save_shard(shard_chunks, shard_masks, shard_dir, shard_idx)
                shard_paths.append(path)
                shard_idx += 1
                print(f"\n  Saved shard {shard_idx}: {args.shard_size:,} chunks "
                      f"(total: {total_tokens:,} tokens, "
                      f"docs: {docs_processed:,}, filtered: {docs_filtered:,})")

            if total_tokens >= args.max_tokens:
                break

    # ── Flush remaining ──
    if sample_buffer:
        texts = []
        for sample in sample_buffer:
            if gopher_quality_filter(sample):
                text = sample.get("raw_content", "")
                if text and len(text.strip()) >= 100:
                    texts.append(text)
                    docs_processed += 1
                else:
                    docs_filtered += 1
            else:
                docs_filtered += 1

        if texts:
            chunks, masks = tokenize_and_chunk(texts, tokenizer, args.chunk_size)
            chunk_buffer.extend(chunks)
            mask_buffer.extend(masks)
            total_tokens += len(chunks) * args.chunk_size

    if chunk_buffer:
        path = save_shard(chunk_buffer, mask_buffer, shard_dir, shard_idx)
        shard_paths.append(path)
        shard_idx += 1
        print(f"\n  Saved final shard {shard_idx}: {len(chunk_buffer):,} chunks")

    pbar.close()

    print(f"\nDone streaming:")
    print(f"  Documents processed: {docs_processed:,}")
    print(f"  Documents filtered: {docs_filtered:,}")
    print(f"  Total shards: {len(shard_paths)}")
    print(f"  Total tokens: {total_tokens:,}")

    # ── Concatenate shards into train/test split ──
    print("\nLoading shards and creating train/test split...")
    all_shards = [load_from_disk(p) for p in shard_paths]

    random.shuffle(all_shards)

    n_test_shards = max(1, int(len(all_shards) * args.test_fraction))
    test_shards = all_shards[:n_test_shards]
    train_shards = all_shards[n_test_shards:]

    print(f"  Train shards: {len(train_shards)}")
    print(f"  Test shards: {len(test_shards)}")

    train_dataset = concatenate_datasets(train_shards)
    test_dataset = concatenate_datasets(test_shards)

    print(f"  Train: {len(train_dataset):,} chunks ({len(train_dataset) * args.chunk_size:,} tokens)")
    print(f"  Test: {len(test_dataset):,} chunks ({len(test_dataset) * args.chunk_size:,} tokens)")

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    save_path = os.path.join(args.output_dir, "retain")
    print(f"Saving to {save_path}...")
    dataset_dict.save_to_disk(save_path)

    print("Cleaning up temporary shards...")
    shutil.rmtree(shard_dir)

    print("Done!")


if __name__ == "__main__":
    main()