"""Data exploration script for tiered alignment datasets.

Prints category distribution and token statistics.

Usage:
    python src/sgtm/data/explore.py --data_path /path/to/dataset
"""

import argparse
from collections import Counter

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Explore dataset statistics")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to tokenized dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading dataset from {args.data_path}")
    dataset = load_from_disk(args.data_path)
    
    print(f"\n{'='*60}")
    print("DATASET OVERVIEW")
    print(f"{'='*60}")
    print(dataset)
    
    # Process each split
    for split_name in dataset.keys():
        split = dataset[split_name]
        print(f"\n{'='*60}")
        print(f"SPLIT: {split_name}")
        print(f"{'='*60}")
        print(f"Number of examples: {len(split):,}")
        
        # Category distribution
        if "category" in split.column_names:
            categories = Counter(split["category"])
            print(f"\nCategories ({len(categories)} unique):")
            for cat, count in categories.most_common():
                print(f"  {cat}: {count:,} examples")
        
        # Token statistics
        if "input_ids" in split.column_names:
            lengths = [len(ids) for ids in split["input_ids"]]
            total_tokens = sum(lengths)
            avg_length = total_tokens / len(lengths)
            min_length = min(lengths)
            max_length = max(lengths)
            
            print(f"\nToken statistics:")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Avg tokens per example: {avg_length:.1f}")
            print(f"  Min tokens: {min_length}")
            print(f"  Max tokens: {max_length}")
            
            # Check for EOS tokens
            eos_token_id = 50256  # GPT-2 EOS
            examples_with_eos = sum(1 for ids in split["input_ids"] if eos_token_id in ids)
            print(f"  Examples with EOS token: {examples_with_eos:,} ({100*examples_with_eos/len(split):.1f}%)")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
