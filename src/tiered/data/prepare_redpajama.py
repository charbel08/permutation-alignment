"""
Prepare RedPajama-Data-V2 for pretraining.

This script:
1. Streams RedPajama-Data-V2 from HuggingFace (English, head_middle partition)
2. Applies Gopher-style quality filtering using built-in quality signals
3. Applies deduplication filtering using built-in n-gram duplicate signals
4. Applies classifier-based filtering using the PaLM quality score
5. Tokenizes with GPT-2 tiktoken tokenizer
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

import numpy as np
import tiktoken
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk, load_dataset
from tqdm import tqdm


def gopher_quality_filter(sample) -> bool:
    """Apply Gopher-style quality filtering using RedPajama quality signals,
    supplemented with deduplication and classifier-based checks."""
    try:
        signals = json.loads(sample["quality_signals"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return False

    # ── Rule 1: word count between 50 and 100,000 ──
    word_count = signals.get("rps_doc_word_count", [[0, 0, 0]])[0][2]
    if word_count < 50 or word_count > 100_000:
        return False

    # ── Rule 2: mean word length between 3 and 10 ──
    mean_word_length = signals.get("rps_doc_mean_word_length", [[0, 0, 0]])[0][2]
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # ── Rule 3: symbol to word ratio below 0.1 ──
    symbol_word_ratio = signals.get("rps_doc_symbol_to_word_ratio", [[0, 0, 0]])[0][2]
    if symbol_word_ratio > 0.1:
        return False

    # ── Rule 4: 90% of lines should not start with bullet point ──
    n_lines = signals.get("ccnet_nlines", [[0, 0, 1]])[0][2]
    if n_lines > 0:
        bullet_lines = signals.get("rps_lines_start_with_bulletpoint", [])
        n_bullet = sum(ln[2] for ln in bullet_lines)
        if n_bullet / n_lines > 0.9:
            return False

    # ── Rule 5: top 2-gram frequency below 0.2 ──
    top_2gram = signals.get("rps_doc_frac_chars_top_2gram", [[0, 0, 0]])[0][2]
    if top_2gram > 0.2:
        return False

    # ── Rule 6: top 3-gram frequency below 0.18 ──
    top_3gram = signals.get("rps_doc_frac_chars_top_3gram", [[0, 0, 0]])[0][2]
    if top_3gram > 0.18:
        return False

    # ── Rule 7: top 4-gram frequency below 0.16 ──
    top_4gram = signals.get("rps_doc_frac_chars_top_4gram", [[0, 0, 0]])[0][2]
    if top_4gram > 0.16:
        return False

    # ── Rule 8: n-gram deduplication ──
    # Reject documents with high fractions of duplicate n-gram content.
    # These signals measure what fraction of the document's characters appear
    # in repeated n-grams, catching boilerplate, copy-paste, and near-duplicates.
    frac_dupe_5gram = signals.get("rps_doc_frac_chars_dupe_5gram", [[0, 0, 0]])[0][2]
    frac_dupe_6gram = signals.get("rps_doc_frac_chars_dupe_6gram", [[0, 0, 0]])[0][2]
    frac_dupe_7gram = signals.get("rps_doc_frac_chars_dupe_7gram", [[0, 0, 0]])[0][2]
    frac_dupe_8gram = signals.get("rps_doc_frac_chars_dupe_8gram", [[0, 0, 0]])[0][2]
    frac_dupe_9gram = signals.get("rps_doc_frac_chars_dupe_9gram", [[0, 0, 0]])[0][2]
    frac_dupe_10gram = signals.get("rps_doc_frac_chars_dupe_10gram", [[0, 0, 0]])[0][2]

    if frac_dupe_5gram > 0.15:
        return False
    if frac_dupe_6gram > 0.14:
        return False
    if frac_dupe_7gram > 0.13:
        return False
    if frac_dupe_8gram > 0.12:
        return False
    if frac_dupe_9gram > 0.11:
        return False
    if frac_dupe_10gram > 0.10:
        return False

    # ── Rule 9: classifier-based quality filtering (PaLM score) ──
    # The rps_doc_ml_palm_score signal is a lightweight quality classifier
    # trained to distinguish high-quality text. Higher = better.
    palm_score = signals.get("rps_doc_ml_palm_score", [[0, 0, 0]])[0][2]
    if palm_score < 0.5:
        return False

    # ── Rule 10: low fraction of lines ending with ellipsis ──
    # Documents with many ellipsis-terminated lines are often truncated or
    # auto-generated (e.g., search result snippets, product listings).
    frac_no_alph_words = signals.get("rps_doc_frac_no_alph_words", [[0, 0, 0]])[0][2]
    if frac_no_alph_words > 0.3:
        return False

    # ── Rule 11: stop word presence ──
    # Documents with very few stop words are often not natural prose
    # (e.g., keyword lists, metadata dumps, structured data).
    stop_word_frac = signals.get("rps_doc_stop_word_fraction", [[0, 0, 0]])[0][2]
    if stop_word_frac < 0.06:
        return False

    # ── Rule 12: perplexity filter via ccnet ──
    # ccnet_perplexity measures how "surprising" the text is to a KenLM model
    # trained on Wikipedia. Very high perplexity often indicates garbage text.
    perplexity = signals.get("ccnet_perplexity", [[0, 0, 0]])[0][2]
    if perplexity > 1500:
        return False

    return True


def tokenize_and_chunk(texts, tokenizer, chunk_size):
    """Tokenize texts and create fixed-length chunks by concatenating documents."""
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode_ordinary(text)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eot_token)

    chunks = []
    attention_masks = []
    for i in range(0, len(all_tokens), chunk_size):
        chunk = all_tokens[i : i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
            attention_masks.append([1] * chunk_size)

    return chunks, attention_masks


def save_shard(chunks, attention_masks, shard_dir, shard_idx):
    """Save a shard of data to disk as a HuggingFace Dataset."""
    ds = Dataset.from_dict({
        "input_ids": chunks,
        "attention_mask": attention_masks,
    })
    shard_path = os.path.join(shard_dir, f"shard_{shard_idx:05d}")
    ds.save_to_disk(shard_path)
    return shard_path


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
        "--batch-size", type=int, default=10000,
        help="Number of documents to accumulate before tokenizing (default: 10000)",
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
    )

    # Accumulate chunks, flush to disk when shard is full
    chunk_buffer = []
    mask_buffer = []
    text_buffer = []
    shard_paths = []
    shard_idx = 0
    total_tokens = 0
    docs_processed = 0
    docs_filtered = 0

    print(f"Processing documents (target: {args.max_tokens:,} tokens)...")
    print(f"Saving shards to {shard_dir} ({args.shard_size:,} chunks per shard)")
    pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True)

    for sample in ds["train"]:
        # Apply quality filter
        if not gopher_quality_filter(sample):
            docs_filtered += 1
            continue

        text = sample.get("raw_content", "")
        if not text or len(text.strip()) < 100:
            docs_filtered += 1
            continue

        text_buffer.append(text)
        docs_processed += 1

        # Tokenize in batches
        if len(text_buffer) >= args.batch_size:
            chunks, masks = tokenize_and_chunk(text_buffer, tokenizer, args.chunk_size)
            chunk_buffer.extend(chunks)
            mask_buffer.extend(masks)
            new_tokens = len(chunks) * args.chunk_size
            total_tokens += new_tokens
            pbar.update(new_tokens)
            text_buffer = []

            # Flush shard to disk when buffer is full
            while len(chunk_buffer) >= args.shard_size:
                shard_chunks = chunk_buffer[:args.shard_size]
                shard_masks = mask_buffer[:args.shard_size]
                chunk_buffer = chunk_buffer[args.shard_size:]
                mask_buffer = mask_buffer[args.shard_size:]

                path = save_shard(shard_chunks, shard_masks, shard_dir, shard_idx)
                shard_paths.append(path)
                shard_idx += 1
                print(f"  Saved shard {shard_idx}: {args.shard_size:,} chunks "
                      f"(total: {total_tokens:,} tokens, "
                      f"docs: {docs_processed:,}, filtered: {docs_filtered:,})")

            if total_tokens >= args.max_tokens:
                break

    # Flush remaining chunks
    if chunk_buffer:
        # Process remaining text buffer
        if text_buffer:
            chunks, masks = tokenize_and_chunk(text_buffer, tokenizer, args.chunk_size)
            chunk_buffer.extend(chunks)
            mask_buffer.extend(masks)

        if chunk_buffer:
            path = save_shard(chunk_buffer, mask_buffer, shard_dir, shard_idx)
            shard_paths.append(path)
            shard_idx += 1
            print(f"  Saved final shard {shard_idx}: {len(chunk_buffer):,} chunks")

    pbar.close()

    print(f"\nDone streaming:")
    print(f"  Documents processed: {docs_processed:,}")
    print(f"  Documents filtered: {docs_filtered:,}")
    print(f"  Total shards: {len(shard_paths)}")
    print(f"  Total tokens: {total_tokens:,}")

    # Concatenate shards into train/test split
    print("\nLoading shards and creating train/test split...")
    all_shards = [load_from_disk(p) for p in shard_paths]

    # Shuffle shards (not individual samples — that would require loading everything)
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

    # Save final dataset
    save_path = os.path.join(args.output_dir, "retain")
    print(f"Saving to {save_path}...")
    dataset_dict.save_to_disk(save_path)

    # Clean up shards
    print("Cleaning up temporary shards...")
    shutil.rmtree(shard_dir)

    print("Done!")


if __name__ == "__main__":
    main()