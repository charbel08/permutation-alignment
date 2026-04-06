"""
Count tokens in one or more Hugging Face datasets saved with save_to_disk().

Supports:
- DatasetDict roots (with dataset_dict.json)
- Single split Dataset roots (with dataset_info.json)
- Recursive discovery under a parent directory

Usage:
    python -m tiered.data.count_tokens /path/to/dataset
    python -m tiered.data.count_tokens /path/a /path/b
    python -m tiered.data.count_tokens /path/to/root --discover
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
import pyarrow as pa
import pyarrow.compute as pc
from tqdm import tqdm


def _has_dataset_dict_ancestor(path: Path, stop: Path) -> bool:
    cur = path.parent
    stop = stop.resolve()
    while cur != cur.parent:
        if (cur / "dataset_dict.json").exists():
            return True
        if cur.resolve() == stop:
            break
        cur = cur.parent
    return False


def _discover_dataset_paths(root: Path) -> list[Path]:
    root = root.resolve()
    found: set[Path] = set()

    for marker in root.rglob("dataset_dict.json"):
        found.add(marker.parent.resolve())

    for marker in root.rglob("dataset_info.json"):
        candidate = marker.parent.resolve()
        if _has_dataset_dict_ancestor(candidate, root):
            continue
        found.add(candidate)

    return sorted(found)


def _count_split_tokens(
    split: Dataset,
    token_column: str,
    progress_label: str,
) -> tuple[int, int]:
    # Fast path: operate directly on Arrow list arrays in C++ instead of
    # iterating Python lists row-by-row. This is dramatically faster.
    table = split.data.table
    col_idx = table.schema.get_field_index(token_column)
    if col_idx == -1:
        return 0, 0

    chunked = table.column(col_idx)
    total_examples = 0
    total_tokens = 0

    pbar = tqdm(total=len(split), desc=progress_label, unit="ex", leave=False)
    for i in range(chunked.num_chunks):
        chunk = chunked.chunk(i)
        if len(chunk) == 0:
            continue

        lengths = pc.list_value_length(chunk)
        lengths = pc.fill_null(lengths, 0)
        chunk_tokens = pc.sum(lengths)
        if chunk_tokens is not None:
            total_tokens += int(chunk_tokens.as_py())

        if chunk.null_count > 0:
            valid_count = pc.sum(pc.cast(pc.invert(pc.is_null(chunk)), pa.int64()))
            total_examples += int(valid_count.as_py()) if valid_count is not None else 0
        else:
            total_examples += len(chunk)

        pbar.update(len(chunk))

    pbar.close()
    return total_examples, total_tokens


def _count_dataset(
    path: Path,
    token_column: str,
) -> tuple[dict[str, tuple[int, int]], int]:
    ds = load_from_disk(str(path))

    per_split: dict[str, tuple[int, int]] = {}
    total_tokens = 0

    if isinstance(ds, DatasetDict):
        split_items = ds.items()
    elif isinstance(ds, Dataset):
        split_items = [("data", ds)]
    else:
        raise TypeError(f"Unsupported dataset type at {path}: {type(ds)}")

    for split_name, split in split_items:
        if token_column not in split.column_names:
            per_split[split_name] = (0, 0)
            continue
        progress_label = f"{path.name}:{split_name}"
        n_examples, n_tokens = _count_split_tokens(
            split=split,
            token_column=token_column,
            progress_label=progress_label,
        )
        per_split[split_name] = (n_examples, n_tokens)
        total_tokens += n_tokens

    return per_split, total_tokens


def main() -> None:
    # Ensure Arrow compute can use all host CPUs unless the user has already
    # constrained it via environment.
    arrow_threads_env = os.environ.get("ARROW_NUM_THREADS")
    if arrow_threads_env is None:
        cpu_count = os.cpu_count() or 1
        pa.set_cpu_count(cpu_count)

    parser = argparse.ArgumentParser(description="Count tokens in saved HF datasets")
    parser.add_argument("paths", nargs="+", help="Dataset path(s) or directory roots")
    parser.add_argument("--discover", action="store_true",
                        help="Recursively discover datasets under provided paths")
    parser.add_argument("--token-column", type=str, default="input_ids")
    args = parser.parse_args()

    dataset_paths: list[Path] = []
    for raw_path in args.paths:
        path = Path(raw_path).resolve()
        if not path.exists():
            print(f"SKIP (missing): {path}")
            continue

        if args.discover:
            discovered = _discover_dataset_paths(path)
            if not discovered:
                print(f"SKIP (no datasets found): {path}")
                continue
            dataset_paths.extend(discovered)
            continue

        dataset_paths.append(path)

    # Preserve order while de-duplicating.
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for p in dataset_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    if not unique_paths:
        print("No dataset paths to process.")
        return

    grand_total_tokens = 0
    for path in unique_paths:
        try:
            per_split, total_tokens = _count_dataset(
                path=path,
                token_column=args.token_column,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n{path}")
            print(f"  ERROR: {exc}")
            continue

        print(f"\n{path}")
        for split_name, (n_examples, n_tokens) in per_split.items():
            print(f"  {split_name:>10}  examples={n_examples:>12,}  tokens={n_tokens:>15,}")
        print(f"  {'TOTAL':>10}  {'':>12}  tokens={total_tokens:>15,}")
        grand_total_tokens += total_tokens

    print(f"\nGRAND TOTAL TOKENS: {grand_total_tokens:,}")


if __name__ == "__main__":
    main()
