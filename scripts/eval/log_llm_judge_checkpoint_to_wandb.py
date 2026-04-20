#!/usr/bin/env python3
"""Log one checkpoint's LLM-judge AlpacaEval summary to a W&B eval run."""

import argparse
import json
from pathlib import Path

import torch
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--results_path", type=Path, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--step_override", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(f"Missing judge results: {args.results_path}")

    if args.step_override is not None:
        global_step = int(args.step_override)
    else:
        training_state_path = args.checkpoint_dir / "training_state.pt"
        if not training_state_path.exists():
            raise FileNotFoundError(f"Missing training state: {training_state_path}")
        training_state = torch.load(training_state_path, map_location="cpu")
        global_step = int(training_state.get("global_step", 0))

    with open(args.results_path) as f:
        payload = json.load(f)
    summary = payload["summary"]

    metrics = {
        "train/step": global_step,
        "AlpacaEval/C2 Win Rate": summary["c2_win_rate"],
        "AlpacaEval/C1 Win Rate": summary["c1_win_rate"],
        "AlpacaEval/Error Rate": summary["error_rate"],
        "AlpacaEval/C2 Wins": summary["c2_wins"],
        "AlpacaEval/C1 Wins": summary["c1_wins"],
        "AlpacaEval/Errors": summary["errors"],
        "AlpacaEval/Decided": summary["n_decided"],
        "AlpacaEval/N": summary["n"],
    }

    for difficulty, ds in summary.get("by_difficulty", {}).items():
        prefix = f"AlpacaEval/{difficulty}"
        metrics[f"{prefix}/C2 Win Rate"] = ds["c2_win_rate"]
        metrics[f"{prefix}/C1 Win Rate"] = ds["c1_win_rate"]
        metrics[f"{prefix}/Error Rate"] = ds["error_rate"]
        metrics[f"{prefix}/C2 Wins"] = ds["c2_wins"]
        metrics[f"{prefix}/C1 Wins"] = ds["c1_wins"]
        metrics[f"{prefix}/Decided"] = ds["n_decided"]
        metrics[f"{prefix}/N"] = ds["n"]

    init_kwargs = {
        "project": args.project,
    }
    if args.entity:
        init_kwargs["entity"] = args.entity
    if args.run_id:
        init_kwargs["id"] = args.run_id
        init_kwargs["resume"] = "allow"
    if args.run_name:
        init_kwargs["name"] = args.run_name
    if args.group:
        init_kwargs["group"] = args.group

    wandb.init(**init_kwargs)
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")
    run_label = wandb.run.id
    wandb.log(metrics)
    wandb.finish()

    checkpoint_label = args.checkpoint_name or args.checkpoint_dir.name
    print(
        f"Logged LLM-judge AlpacaEval for {checkpoint_label} "
        f"to W&B eval run {run_label} at train/step={global_step}"
    )


if __name__ == "__main__":
    main()
