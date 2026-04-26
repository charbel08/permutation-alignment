#!/usr/bin/env python3
"""Print two samples from the same person in the synthetic-bios metadata.

Usage:
    python show_two_samples_same_person.py
    python show_two_samples_same_person.py --person_id 42
    python show_two_samples_same_person.py --split test
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bio_metadata", type=str,
                   default="/work/scratch/data/datasets/synthetic_bios/bios_metadata.json")
    p.add_argument("--person_id", type=int, default=None,
                   help="Specific person to fetch. Default: first person with >=2 bios.")
    p.add_argument("--split", type=str, default=None,
                   choices=["train", "test"],
                   help="Restrict to a split via train_people/test_people.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.bio_metadata) as f:
        metadata = json.load(f)

    bios = metadata["bios"]
    if args.split:
        people = set(metadata.get(f"{args.split}_people", []))
        if not people:
            raise SystemExit(f"No {args.split}_people in metadata")
        bios = [b for b in bios if b["person_id"] in people]

    by_person: dict[int, list] = defaultdict(list)
    for bio in bios:
        by_person[bio["person_id"]].append(bio)

    if args.person_id is not None:
        if args.person_id not in by_person:
            raise SystemExit(f"person_id {args.person_id} not in dataset / split")
        bs = by_person[args.person_id]
    else:
        bs = next(b for b in by_person.values() if len(b) >= 2)

    if len(bs) < 2:
        raise SystemExit(f"person_id {bs[0]['person_id']} has only {len(bs)} bios")

    a, b = bs[0], bs[1]
    print(f"person_id: {a['person_id']}  (total bios for this person: {len(bs)})")
    print(f"name: {a.get('name')}")
    print()
    print("--- sample A ---")
    print(f"target_attr: {a['target_attr']}")
    print(f"prefix:      {a['prefix']}")
    print(f"text:        {a['text']}")
    print()
    print("--- sample B ---")
    print(f"target_attr: {b['target_attr']}")
    print(f"prefix:      {b['prefix']}")
    print(f"text:        {b['text']}")


if __name__ == "__main__":
    main()
