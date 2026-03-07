"""
Prepare RedPajama-Data-V2 for pretraining.

Two-phase pipeline:
  Phase 1 — Download: Fetch raw .json.gz files from Together's servers
            using aria2c (massively parallel) or threaded fallback.
  Phase 2 — Process: Filter + tokenize + chunk locally with multiprocessing.

Usage:
    python -m tiered.data.prepare_redpajama \
        --output-dir /work/scratch/data/datasets/redpajama \
        --chunk-size 1024 \
        --max-tokens 100000000000 \
        --num-snapshots 10 \
        --num-proc 32
"""

import argparse
import glob
import gzip
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import tiktoken
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm

BASE_URL = "https://data.together.xyz/redpajama-data-v2/v1.0.0"


# ─── Phase 1: Download ───────────────────────────────────────────────────────


def fetch_listing(snapshot, lang="en", partition="head_middle"):
    """Fetch the listing of file keys for a given snapshot."""
    import requests
    tag = f"{lang}-{snapshot}-{partition}"
    url = f"{BASE_URL}/listings/{tag}.txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return [l.strip() for l in resp.text.strip().split("\n") if l.strip()]


def download_with_aria2c(url_pairs, download_dir, connections=32):
    """Download files using aria2c for maximum throughput.

    url_pairs: list of (url, output_path) tuples
    """
    # Write URL list file for aria2c
    url_file = os.path.join(download_dir, "_urls.txt")
    with open(url_file, "w") as f:
        for url, out_path in url_pairs:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            f.write(f"{url}\n  dir={os.path.dirname(out_path)}\n  out={os.path.basename(out_path)}\n")

    cmd = [
        "aria2c",
        "--input-file", url_file,
        "--max-concurrent-downloads", str(connections),
        "--max-connection-per-server", "4",
        "--split", "4",
        "--min-split-size", "1M",
        "--continue=true",
        "--retry-wait", "3",
        "--max-tries", "5",
        "--timeout", "60",
        "--connect-timeout", "30",
        "--console-log-level", "warn",
        "--summary-interval", "30",
    ]

    print(f"  Downloading {len(url_pairs)} files with aria2c ({connections} connections)...")
    result = subprocess.run(cmd, capture_output=False)
    os.remove(url_file)
    return result.returncode == 0


def download_with_threads(url_pairs, max_workers=32):
    """Fallback: download files using threads."""
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _download_one(url_out):
        url, out_path = url_out
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return True
        except Exception:
            return False

    success = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, pair): pair for pair in url_pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            if fut.result():
                success += 1

    print(f"  Downloaded {success}/{len(url_pairs)} files")
    return success > 0


def download_files(keys, download_dir, connections=32):
    """Download document and quality signal files for all keys."""
    url_pairs = []
    for key in keys:
        doc_url = f"{BASE_URL}/documents/{key}.json.gz"
        sig_url = f"{BASE_URL}/quality_signals/{key}.signals.json.gz"
        doc_path = os.path.join(download_dir, "documents", f"{key}.json.gz")
        sig_path = os.path.join(download_dir, "quality_signals", f"{key}.signals.json.gz")

        # Skip already downloaded files (resume support)
        if not os.path.exists(doc_path):
            url_pairs.append((doc_url, doc_path))
        if not os.path.exists(sig_path):
            url_pairs.append((sig_url, sig_path))

    if not url_pairs:
        print("  All files already downloaded.")
        return

    print(f"  {len(url_pairs)} files to download ({len(url_pairs)//2} document+signal pairs)")

    # Try aria2c first, fall back to threads
    has_aria2c = shutil.which("aria2c") is not None
    if has_aria2c:
        download_with_aria2c(url_pairs, download_dir, connections=connections)
    else:
        print("  aria2c not found, falling back to threaded downloads (slower).")
        print("  Install aria2c for 5-10x faster downloads: apt install aria2")
        download_with_threads(url_pairs, max_workers=connections)


# ─── Phase 2: Process ────────────────────────────────────────────────────────


def _get(signals, key, default=0.0):
    return signals.get(key, [[0, 0, default]])[0][2]


def quality_filter(signals) -> bool:
    """Apply quality filtering on parsed quality signals dict."""
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
        if sum(ln[2] for ln in bullet_lines) / n_lines > 0.9:
            return False

    if _get(signals, "rps_doc_frac_chars_top_2gram") > 0.2:
        return False
    if _get(signals, "rps_doc_frac_chars_top_3gram") > 0.18:
        return False
    if _get(signals, "rps_doc_frac_chars_top_4gram") > 0.16:
        return False

    for n, t in [(5, .15), (6, .14), (7, .13), (8, .12), (9, .11), (10, .10)]:
        if _get(signals, f"rps_doc_frac_chars_dupe_{n}gram") > t:
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


def process_file_pair(args_tuple):
    """Process a single document+signal file pair. Runs in a worker process.

    Returns: (list_of_texts, n_passed, n_filtered)
    """
    doc_path, sig_path = args_tuple

    if not os.path.exists(doc_path) or not os.path.exists(sig_path):
        return [], 0, 0

    try:
        with gzip.open(doc_path, "rt", encoding="utf-8") as f:
            doc_lines = f.readlines()
        with gzip.open(sig_path, "rt", encoding="utf-8") as f:
            sig_lines = f.readlines()
    except Exception:
        return [], 0, 0

    texts = []
    n_passed = 0
    n_filtered = 0

    for doc_line, sig_line in zip(doc_lines, sig_lines):
        try:
            doc = json.loads(doc_line)
            sig_data = json.loads(sig_line)
        except (json.JSONDecodeError, TypeError):
            n_filtered += 1
            continue

        # Extract quality signals — handle both nested and flat formats
        qs = sig_data.get("quality_signals", sig_data)
        if isinstance(qs, str):
            try:
                qs = json.loads(qs)
            except (json.JSONDecodeError, TypeError):
                n_filtered += 1
                continue

        if not quality_filter(qs):
            n_filtered += 1
            continue

        text = doc.get("raw_content", "")
        if not text or len(text.strip()) < 100:
            n_filtered += 1
            continue

        texts.append(text)
        n_passed += 1

    return texts, n_passed, n_filtered


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
    masks = [[1] * chunk_size] * n_chunks
    return chunks, masks


def save_shard(chunks, masks, shard_dir, shard_idx):
    ds = Dataset.from_dict({"input_ids": chunks, "attention_mask": masks})
    path = os.path.join(shard_dir, f"shard_{shard_idx:05d}")
    ds.save_to_disk(path)
    return path


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prepare RedPajama-Data-V2 for pretraining")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=100_000_000_000)
    parser.add_argument("--num-snapshots", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=500_000)
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument("--num-proc", type=int, default=32,
                        help="Workers for downloading and processing (default: 32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-downloads", action="store_true",
                        help="Keep raw downloaded files after processing")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    shard_dir = os.path.join(args.output_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    download_dir = os.path.join(args.output_dir, "_downloads")
    os.makedirs(download_dir, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")

    all_snapshots = [
        "2023-14", "2023-06", "2022-49", "2022-40", "2022-27",
        "2022-21", "2021-49", "2021-43", "2021-39", "2021-31",
    ]
    snapshots = all_snapshots[: args.num_snapshots]

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Download
    # ══════════════════════════════════════════════════════════════════════════
    print(f"{'='*70}")
    print(f"PHASE 1: DOWNLOAD")
    print(f"{'='*70}")
    print(f"Snapshots: {snapshots}")

    all_keys = []
    for snap in snapshots:
        print(f"\nFetching listing for {snap}...")
        keys = fetch_listing(snap)
        all_keys.extend(keys)
        print(f"  {snap}: {len(keys):,} files")

    print(f"\nTotal file pairs: {len(all_keys):,}")
    random.shuffle(all_keys)

    # Download in snapshot-sized batches so we can start processing early
    # if needed, but for simplicity download all first
    download_files(all_keys, download_dir, connections=args.num_proc)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Process (filter + tokenize + chunk + shard)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"PHASE 2: PROCESS")
    print(f"{'='*70}")

    # Build list of file pairs to process
    file_pairs = []
    for key in all_keys:
        doc_path = os.path.join(download_dir, "documents", f"{key}.json.gz")
        sig_path = os.path.join(download_dir, "quality_signals", f"{key}.signals.json.gz")
        file_pairs.append((doc_path, sig_path))

    print(f"Processing {len(file_pairs):,} file pairs with {args.num_proc} workers...")

    chunk_buffer = []
    mask_buffer = []
    text_buffer = []
    shard_paths = []
    shard_idx = 0
    total_tokens = 0
    docs_processed = 0
    docs_filtered = 0

    TEXT_FLUSH_SIZE = 200_000  # tokenize every 200K docs

    pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True, desc="Tokens")

    with Pool(processes=args.num_proc) as pool:
        for texts, n_passed, n_filtered in pool.imap_unordered(
            process_file_pair, file_pairs, chunksize=4
        ):
            docs_processed += n_passed
            docs_filtered += n_filtered
            text_buffer.extend(texts)

            # Tokenize in large batches
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
                    shard_chunks = chunk_buffer[: args.shard_size]
                    shard_masks = mask_buffer[: args.shard_size]
                    chunk_buffer = chunk_buffer[args.shard_size :]
                    mask_buffer = mask_buffer[args.shard_size :]

                    path = save_shard(shard_chunks, shard_masks, shard_dir, shard_idx)
                    shard_paths.append(path)
                    shard_idx += 1
                    tqdm.write(
                        f"  Shard {shard_idx}: {args.shard_size:,} chunks | "
                        f"Total: {total_tokens:,.0f} tok | "
                        f"Docs: {docs_processed:,} pass / {docs_filtered:,} filtered"
                    )

            if total_tokens >= args.max_tokens:
                pool.terminate()
                break

    # Flush remaining
    if text_buffer:
        chunks, masks = tokenize_and_chunk(text_buffer, tokenizer, args.chunk_size)
        chunk_buffer.extend(chunks)
        mask_buffer.extend(masks)
        total_tokens += len(chunks) * args.chunk_size

    if chunk_buffer:
        path = save_shard(chunk_buffer, mask_buffer, shard_dir, shard_idx)
        shard_paths.append(path)
        shard_idx += 1
        print(f"  Final shard {shard_idx}: {len(chunk_buffer):,} chunks")

    pbar.close()

    print(f"\nProcessing complete:")
    print(f"  Documents passed:   {docs_processed:,}")
    print(f"  Documents filtered: {docs_filtered:,}")
    print(f"  Total shards:       {len(shard_paths)}")
    print(f"  Total tokens:       {total_tokens:,}")

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

    if not args.keep_downloads:
        print("Cleaning up downloads...")
        shutil.rmtree(download_dir)

    print("Done!")


if __name__ == "__main__":
    main()