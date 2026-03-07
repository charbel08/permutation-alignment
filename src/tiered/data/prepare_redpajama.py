"""
Prepare RedPajama-Data-V2 for pretraining.

Downloads raw data directly from Together's servers in parallel, bypassing
the slow single-threaded HuggingFace streaming loader.

Pipeline:
1. Fetches file listings for each snapshot
2. Downloads document + quality signal files in parallel (threaded)
3. Filters documents using Gopher + dedup + classifier rules
4. Tokenizes with GPT-2 tiktoken (batched)
5. Chunks into fixed-length sequences
6. Saves shards incrementally to disk
7. Concatenates shards into a final HuggingFace DatasetDict

Usage:
    python -m tiered.data.prepare_redpajama \
        --output-dir /work/scratch/data/datasets/redpajama \
        --chunk-size 1024 \
        --max-tokens 100000000000 \
        --num-snapshots 10 \
        --download-workers 32
"""

import argparse
import gzip
import io
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Thread

import numpy as np
import requests
import tiktoken
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm

BASE_URL = "https://data.together.xyz/redpajama-data-v2/v1.0.0"


# ─── Downloading ──────────────────────────────────────────────────────────────


def fetch_listing(snapshot, lang="en", partition="head_middle"):
    """Fetch the listing of file keys for a given snapshot."""
    tag = f"{lang}-{snapshot}-{partition}"
    url = f"{BASE_URL}/listings/{tag}.txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    keys = [line.strip() for line in resp.text.strip().split("\n") if line.strip()]
    return keys


def download_and_parse_file(key):
    """Download a document file and its quality signals, parse and filter.

    Returns a list of texts that passed quality filtering.
    """
    doc_url = f"{BASE_URL}/documents/{key}.json.gz"
    sig_url = f"{BASE_URL}/quality_signals/{key}.signals.json.gz"

    try:
        doc_resp = requests.get(doc_url, timeout=120)
        doc_resp.raise_for_status()
        sig_resp = requests.get(sig_url, timeout=120)
        sig_resp.raise_for_status()
    except Exception as e:
        return [], 0, 0

    # Decompress and parse
    try:
        doc_lines = gzip.decompress(doc_resp.content).decode("utf-8").strip().split("\n")
        sig_lines = gzip.decompress(sig_resp.content).decode("utf-8").strip().split("\n")
    except Exception:
        return [], 0, 0

    texts = []
    n_passed = 0
    n_filtered = 0

    for doc_line, sig_line in zip(doc_lines, sig_lines):
        try:
            doc = json.loads(doc_line)
            sig = json.loads(sig_line)
        except (json.JSONDecodeError, TypeError):
            n_filtered += 1
            continue

        # Build a combined sample dict for the filter
        quality_signals = sig.get("quality_signals", {})
        if isinstance(quality_signals, str):
            # Already JSON string
            qs = quality_signals
        else:
            qs = json.dumps(quality_signals)

        if not _quality_filter(qs):
            n_filtered += 1
            continue

        text = doc.get("raw_content", "")
        if not text or len(text.strip()) < 100:
            n_filtered += 1
            continue

        texts.append(text)
        n_passed += 1

    return texts, n_passed, n_filtered


# ─── Quality filtering ───────────────────────────────────────────────────────


def _get(signals, key, default=0.0):
    return signals.get(key, [[0, 0, default]])[0][2]


def _quality_filter(quality_signals_json) -> bool:
    """Apply quality filtering on a JSON string of quality signals."""
    try:
        signals = json.loads(quality_signals_json) if isinstance(quality_signals_json, str) else quality_signals_json
    except (json.JSONDecodeError, TypeError):
        return False

    word_count = _get(signals, "rps_doc_word_count")
    if word_count < 50 or word_count > 100_000:
        return False

    if not (3 <= _get(signals, "rps_doc_mean_word_length") <= 10):
        return False

    if _get(signals, "rps_doc_symbol_to_word_ratio") > 0.1:
        return False

    n_lines = _get(signals, "ccnet_nlines", 1)
    if n_lines > 0:
        bullet_lines = signals.get("rps_lines_start_with_bulletpoint", [])
        n_bullet = sum(ln[2] for ln in bullet_lines)
        if n_bullet / n_lines > 0.9:
            return False

    if _get(signals, "rps_doc_frac_chars_top_2gram") > 0.2:
        return False
    if _get(signals, "rps_doc_frac_chars_top_3gram") > 0.18:
        return False
    if _get(signals, "rps_doc_frac_chars_top_4gram") > 0.16:
        return False

    for n, thresh in [(5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10)]:
        if _get(signals, f"rps_doc_frac_chars_dupe_{n}gram") > thresh:
            return False

    if _get(signals, "rps_doc_ml_palm_score") < 0.5:
        return False
    if _get(signals, "rps_doc_frac_no_alph_words") > 0.3:
        return False
    if _get(signals, "rps_doc_stop_word_fraction") < 0.06:
        return False
    if _get(signals, "ccnet_perplexity") > 1500:
        return False

    return True


# ─── Tokenization ────────────────────────────────────────────────────────────


def tokenize_and_chunk(texts, tokenizer, chunk_size):
    """Tokenize using batched encoding and chunk with numpy."""
    all_encoded = tokenizer.encode_ordinary_batch(texts)

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

    n_chunks = len(all_tokens) // chunk_size
    if n_chunks == 0:
        return [], []

    truncated = all_tokens[: n_chunks * chunk_size]
    chunks = truncated.reshape(n_chunks, chunk_size).tolist()
    attention_masks = [[1] * chunk_size] * n_chunks

    return chunks, attention_masks


# ─── Shard saving ────────────────────────────────────────────────────────────


def save_shard(chunks, attention_masks, shard_dir, shard_idx):
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
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=100_000_000_000)
    parser.add_argument("--num-snapshots", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=500_000)
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument("--download-workers", type=int, default=32,
                        help="Parallel download/filter threads (default: 32)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_dir = os.path.join(args.output_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")

    all_snapshots = [
        "2023-14", "2023-06", "2022-49", "2022-40", "2022-27",
        "2022-21", "2021-49", "2021-43", "2021-39", "2021-31",
    ]
    snapshots = all_snapshots[:args.num_snapshots]
    print(f"Using {len(snapshots)} snapshots: {snapshots}")

    # ── Step 1: Fetch all file listings ──
    print("Fetching file listings...")
    all_keys = []
    for snap in snapshots:
        keys = fetch_listing(snap)
        all_keys.extend(keys)
        print(f"  {snap}: {len(keys):,} files")
    print(f"Total files to process: {len(all_keys):,}")

    # Shuffle to mix snapshots
    random.shuffle(all_keys)

    # ── Step 2: Download, filter, tokenize, shard ──
    chunk_buffer = []
    mask_buffer = []
    text_buffer = []
    shard_paths = []
    shard_idx = 0
    total_tokens = 0
    docs_processed = 0
    docs_filtered = 0

    print(f"\nProcessing (target: {args.max_tokens:,} tokens, "
          f"{args.download_workers} download workers)...")
    pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True)

    # Process files in parallel, tokenize on main thread
    TEXT_FLUSH_SIZE = 100_000  # tokenize every 100K docs

    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        # Submit all jobs (or up to a window)
        pending = set()
        key_iter = iter(all_keys)
        max_pending = args.download_workers * 4  # prefetch window

        def submit_more():
            while len(pending) < max_pending:
                try:
                    key = next(key_iter)
                except StopIteration:
                    break
                fut = executor.submit(download_and_parse_file, key)
                pending.add(fut)

        submit_more()

        while pending:
            # Wait for any future to complete
            done = set()
            for fut in as_completed(pending, timeout=300):
                done.add(fut)
                try:
                    texts, n_passed, n_filt = fut.result()
                except Exception:
                    continue

                text_buffer.extend(texts)
                docs_processed += n_passed
                docs_filtered += n_filt

                # Tokenize when we have enough text
                if len(text_buffer) >= TEXT_FLUSH_SIZE:
                    chunks, masks = tokenize_and_chunk(text_buffer, tokenizer, args.chunk_size)
                    chunk_buffer.extend(chunks)
                    mask_buffer.extend(masks)
                    new_tokens = len(chunks) * args.chunk_size
                    total_tokens += new_tokens
                    pbar.update(new_tokens)
                    text_buffer = []

                    # Flush shards
                    while len(chunk_buffer) >= args.shard_size:
                        shard_chunks = chunk_buffer[:args.shard_size]
                        shard_masks = mask_buffer[:args.shard_size]
                        chunk_buffer = chunk_buffer[args.shard_size:]
                        mask_buffer = mask_buffer[args.shard_size:]

                        path = save_shard(shard_chunks, shard_masks, shard_dir, shard_idx)
                        shard_paths.append(path)
                        shard_idx += 1
                        tqdm.write(
                            f"  Saved shard {shard_idx}: {args.shard_size:,} chunks "
                            f"(total: {total_tokens:,} tok, "
                            f"docs: {docs_processed:,}, filtered: {docs_filtered:,})"
                        )

                if total_tokens >= args.max_tokens:
                    # Cancel remaining futures
                    for f in pending - done:
                        f.cancel()
                    pending = set()
                    break

                break  # process one at a time then refill

            pending -= done
            if total_tokens < args.max_tokens:
                submit_more()

    # ── Flush remaining ──
    if text_buffer:
        chunks, masks = tokenize_and_chunk(text_buffer, tokenizer, args.chunk_size)
        chunk_buffer.extend(chunks)
        mask_buffer.extend(masks)
        total_tokens += len(chunks) * args.chunk_size

    if chunk_buffer:
        path = save_shard(chunk_buffer, mask_buffer, shard_dir, shard_idx)
        shard_paths.append(path)
        shard_idx += 1
        print(f"  Saved final shard {shard_idx}: {len(chunk_buffer):,} chunks")

    pbar.close()

    print(f"\nDone processing:")
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

    print(f"  Train: {len(train_dataset):,} chunks "
          f"({len(train_dataset) * args.chunk_size:,} tokens)")
    print(f"  Test: {len(test_dataset):,} chunks "
          f"({len(test_dataset) * args.chunk_size:,} tokens)")

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    save_path = os.path.join(args.output_dir, "retain")
    print(f"Saving to {save_path}...")
    dataset_dict.save_to_disk(save_path)

    print("Cleaning up temporary shards...")
    shutil.rmtree(shard_dir)

    print("Done!")


if __name__ == "__main__":
    main()