#!/usr/bin/env python3
"""Convert C1/C2 outputs from llm_judge_c1_c2.py to AlpacaEval JSON format.

Input files are expected to be lists of rows with at least:
  - idx
  - instruction
  - output

Outputs:
  - model_outputs.json      (typically C2)
  - reference_outputs.json  (typically C1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--c1_outputs", type=str, required=True, help="Path to c1_outputs.json")
    p.add_argument("--c2_outputs", type=str, required=True, help="Path to c2_outputs.json")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to write AlpacaEval-formatted JSON")
    p.add_argument("--model_name", type=str, default="tiered_c2", help="Generator name for model outputs")
    p.add_argument("--reference_name", type=str, default="tiered_c1", help="Generator name for reference outputs")
    return p.parse_args()


def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows.sort(key=lambda r: int(r["idx"]))
    return rows


def _convert(rows: list[dict], generator: str) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                "instruction": str(r.get("instruction", "")),
                "output": str(r.get("output", "")),
                "generator": generator,
            }
        )
    return out


def main():
    args = parse_args()
    c1_path = Path(args.c1_outputs)
    c2_path = Path(args.c2_outputs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c1 = _load_rows(c1_path)
    c2 = _load_rows(c2_path)
    if len(c1) != len(c2):
        raise ValueError(f"Mismatched lengths: c1={len(c1)} c2={len(c2)}")

    model_outputs = _convert(c2, args.model_name)
    reference_outputs = _convert(c1, args.reference_name)

    model_path = out_dir / "model_outputs.json"
    ref_path = out_dir / "reference_outputs.json"
    with open(model_path, "w") as f:
        json.dump(model_outputs, f, indent=2)
    with open(ref_path, "w") as f:
        json.dump(reference_outputs, f, indent=2)

    print(f"Wrote model outputs:      {model_path}")
    print(f"Wrote reference outputs:  {ref_path}")
    print(f"Num examples:             {len(model_outputs)}")


if __name__ == "__main__":
    main()
