"""Pull full W&B history for the 3 multi-stage cumulative finetune runs.

Saves one CSV of step-keyed metrics per stage plus a runs.json with each
run's config, summary, id, and state. Downstream plotting reads from disk so
we don't re-hit W&B.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import wandb


DEFAULT_RUN_NAMES = [
    "finetune_150m_multi_stage_perconfig_stage_0_C2_deu_Latn_key5pct_kl0p1",
    "finetune_150m_multi_stage_perconfig_stage_1_C3_tur_Latn_key5pct_kl0p1",
    "finetune_150m_multi_stage_perconfig_stage_2_C4_spa_Latn_key5pct_kl0p1",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--project", type=str, default="main-multi-finetune")
    p.add_argument("--run_names", type=str, nargs="+", default=DEFAULT_RUN_NAMES)
    p.add_argument("--output_dir", type=str,
                   default="outputs/multi_stage_finetune_history")
    return p.parse_args()


def _find_run(api: wandb.Api, project_path: str, name: str):
    matches = [r for r in api.runs(path=project_path,
                                   filters={"display_name": name})]
    if not matches:
        raise RuntimeError(f"No run named {name!r} in {project_path}")
    if len(matches) > 1:
        matches.sort(key=lambda r: r.created_at, reverse=True)
        ids = [r.id for r in matches]
        print(f"  [warn] {len(matches)} runs match {name!r}: {ids}; using newest")
    return matches[0]


def _scan_full_history(run) -> tuple[list[str], list[dict]]:
    rows = list(run.scan_history(page_size=1000))
    if not rows:
        return [], []
    cols: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    return cols, rows


def _write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    api = wandb.Api()

    runs_meta = {}
    for name in args.run_names:
        print(f"Fetching {name} ...", flush=True)
        run = _find_run(api, project_path, name)
        cols, rows = _scan_full_history(run)
        csv_path = out_dir / f"{name}.csv"
        _write_csv(csv_path, cols, rows)
        print(f"  -> {csv_path}  rows={len(rows)}  cols={len(cols)}",
              flush=True)
        runs_meta[name] = {
            "id": run.id,
            "state": run.state,
            "url": run.url,
            "created_at": str(run.created_at),
            "config": dict(run.config),
            "summary": {k: v for k, v in dict(run.summary).items()
                        if not k.startswith("_")},
        }

    meta_path = out_dir / "runs.json"
    meta_path.write_text(json.dumps(runs_meta, indent=2, default=str))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
