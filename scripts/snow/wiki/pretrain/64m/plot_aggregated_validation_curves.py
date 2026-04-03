#!/usr/bin/env python3
"""Plot aggregated validation curves from exported W&B CSV histories.

Expected input files follow:
  <run_name>__<wandb_run_id>.csv
for runs exported from:
  /network/scratch/e/elfeghac/plots/64m-pretrain
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


GROUP_PATTERNS: dict[str, re.Pattern[str]] = {
    "baseline": re.compile(r"^baseline_64m$"),
    "up": re.compile(r"^up_total20pct_run\d+$"),
    "down": re.compile(r"^down_total20pct_run\d+$"),
    "mlpboth": re.compile(r"^mlpboth_total20pct_run\d+$"),
}

GROUP_COLORS = {
    "baseline": "#1f77b4",
    "up": "#ff7f0e",
    "down": "#2ca02c",
    "mlpboth": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot validation curves for Snow 64M experiments."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/network/scratch/e/elfeghac/plots/64m-pretrain"),
        help="Directory containing exported W&B run CSV files.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="val/loss_c1,val/loss_c2",
        help="Comma-separated metric names to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <output-dir>/aggregated_validation_curves.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plot and aggregated CSV outputs. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--finished-only",
        action="store_true",
        help="Only include runs marked finished in _manifest.json (if available).",
    )
    parser.add_argument(
        "--step-range",
        choices=["shared", "none"],
        default="shared",
        help=(
            "How to align x-ranges across groups per metric. "
            "'shared' clips all groups to the common overlap window; "
            "'none' keeps each group's native range."
        ),
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale on absolute-loss y-axis.",
    )
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Disable delta-vs-baseline panels and plot absolute curves only.",
    )
    parser.add_argument(
        "--tail-start",
        type=int,
        default=5000,
        help="Start step for the tail-zoom panel when --no-delta is used.",
    )
    return parser.parse_args()


def parse_run_name_and_id(csv_path: Path) -> tuple[str, str] | tuple[None, None]:
    stem = csv_path.stem
    if "__" not in stem:
        return None, None
    run_name, run_id = stem.rsplit("__", 1)
    if not run_name or not run_id:
        return None, None
    return run_name, run_id


def classify_group(run_name: str) -> str | None:
    for group, pattern in GROUP_PATTERNS.items():
        if pattern.match(run_name):
            return group
    return None


def load_manifest_states(input_dir: Path) -> dict[str, str]:
    manifest_path = input_dir / "_manifest.json"
    if not manifest_path.exists():
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    states: dict[str, str] = {}
    for entry in entries:
        run_id = entry.get("run_id")
        state = entry.get("state")
        if isinstance(run_id, str) and isinstance(state, str):
            states[run_id] = state
    return states


def c2_fallback_metric(metric: str) -> str | None:
    if metric.endswith("_c2"):
        return metric[:-3] + "_c1"
    return None


def read_metric_series(csv_path: Path, metric: str) -> pd.Series | None:
    fallback_metric = c2_fallback_metric(metric)
    wanted = {"train/step", "_step", metric}
    if fallback_metric is not None:
        wanted.add(fallback_metric)
    df = pd.read_csv(csv_path, usecols=lambda c: c in wanted)

    if "train/step" in df.columns:
        step_col = "train/step"
    elif "_step" in df.columns:
        step_col = "_step"
    else:
        return None

    if metric in df.columns:
        value_col = metric
    elif fallback_metric is not None and fallback_metric in df.columns:
        value_col = fallback_metric
    else:
        return None

    sub = df[[step_col, value_col]].dropna()
    if sub.empty and fallback_metric is not None and fallback_metric in df.columns and value_col != fallback_metric:
        value_col = fallback_metric
        sub = df[[step_col, value_col]].dropna()
    if sub.empty:
        return None

    sub[step_col] = pd.to_numeric(sub[step_col], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return None

    # Avoid duplicate steps from mixed logs by collapsing per step.
    grouped = sub.groupby(step_col, as_index=False)[value_col].mean().sort_values(step_col)
    steps = grouped[step_col].astype(int).to_numpy()
    values = grouped[value_col].astype(float).to_numpy()
    return pd.Series(values, index=steps)


def aggregate_series(series_list: list[pd.Series]) -> pd.DataFrame:
    aligned = pd.concat(series_list, axis=1, sort=True).sort_index()
    n_vals = aligned.count(axis=1).astype(int)
    std_vals = aligned.std(axis=1, ddof=0, skipna=True).fillna(0.0)
    stderr_vals = std_vals / np.sqrt(np.maximum(n_vals, 1))
    out = pd.DataFrame(
        {
            "step": aligned.index.astype(int),
            "mean": aligned.mean(axis=1, skipna=True).to_numpy(),
            "std": std_vals.to_numpy(),
            "stderr": stderr_vals.to_numpy(),
            "n": n_vals.to_numpy(),
        }
    )
    return out


def sanitize_metric_name(metric: str) -> str:
    return metric.replace("/", "_").replace(" ", "_")


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (output_dir / "aggregated_validation_curves.png")
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not metrics:
        raise SystemExit("No metrics were provided.")

    run_states = load_manifest_states(input_dir)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in: {input_dir}")

    grouped_curves: dict[str, dict[str, list[pd.Series]]] = defaultdict(
        lambda: defaultdict(list)
    )
    included_runs: dict[str, int] = defaultdict(int)

    for csv_file in csv_files:
        run_name, run_id = parse_run_name_and_id(csv_file)
        if run_name is None or run_id is None:
            continue

        group = classify_group(run_name)
        if group is None:
            continue

        if args.finished_only:
            state = run_states.get(run_id)
            if state != "finished":
                continue

        any_metric_added = False
        for metric in metrics:
            series = read_metric_series(csv_file, metric)
            if series is None or series.empty:
                continue
            series.name = f"{run_name}__{run_id}"
            grouped_curves[group][metric].append(series)
            any_metric_added = True

        if any_metric_added:
            included_runs[group] += 1

    if not grouped_curves:
        raise SystemExit(
            "No matching runs/metrics found. Check metric names or file naming."
        )

    n_rows = len(metrics)
    n_cols = 2
    fig_width = 14
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 4.2 * n_rows), sharex="col")
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    for row_idx, metric in enumerate(metrics):
        ax_abs = axes[row_idx, 0]
        ax_aux = axes[row_idx, 1]
        agg_by_group: dict[str, pd.DataFrame] = {}
        run_count_by_group: dict[str, int] = {}

        for group in GROUP_PATTERNS:
            runs_for_metric = grouped_curves.get(group, {}).get(metric, [])
            if not runs_for_metric:
                continue

            agg = aggregate_series(runs_for_metric)
            agg_by_group[group] = agg
            run_count_by_group[group] = len(runs_for_metric)

        if args.step_range == "shared" and agg_by_group:
            starts = [int(df["step"].min()) for df in agg_by_group.values() if not df.empty]
            ends = [int(df["step"].max()) for df in agg_by_group.values() if not df.empty]
            if starts and ends:
                shared_start = max(starts)
                shared_end = min(ends)
                if shared_start < shared_end:
                    for group, agg in list(agg_by_group.items()):
                        clipped = agg[(agg["step"] >= shared_start) & (agg["step"] <= shared_end)]
                        if clipped.empty:
                            del agg_by_group[group]
                            run_count_by_group.pop(group, None)
                            continue
                        agg_by_group[group] = clipped

        for group in GROUP_PATTERNS:
            agg = agg_by_group.get(group)
            if agg is None or agg.empty:
                continue

            color = GROUP_COLORS[group]
            n_runs = run_count_by_group[group]

            ax_abs.plot(
                agg["step"],
                agg["mean"],
                color=color,
                linewidth=2.0,
                label=f"{group} (n={n_runs})",
            )
            if n_runs > 1:
                y_low = agg["mean"] - agg["stderr"]
                y_high = agg["mean"] + agg["stderr"]
                ax_abs.fill_between(
                    agg["step"],
                    y_low,
                    y_high,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )

            agg_out = output_dir / f"aggregated_{group}_{sanitize_metric_name(metric)}.csv"
            agg.to_csv(agg_out, index=False)

        if not args.no_delta:
            ax_delta = ax_aux
            baseline_agg = agg_by_group.get("baseline")
            if baseline_agg is not None and not baseline_agg.empty:
                ax_delta.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
                for group in GROUP_PATTERNS:
                    if group == "baseline":
                        continue
                    agg = agg_by_group.get(group)
                    if agg is None or agg.empty:
                        continue

                    merged = pd.merge(
                        agg[["step", "mean", "stderr"]],
                        baseline_agg[["step", "mean", "stderr"]],
                        on="step",
                        how="inner",
                        suffixes=("", "_baseline"),
                    )
                    merged = merged[merged["mean_baseline"] != 0.0]
                    if merged.empty:
                        continue

                    # Relative gap in basis points: 1 bp = 0.01%.
                    merged["delta_bps"] = (
                        (merged["mean"] / merged["mean_baseline"]) - 1.0
                    ) * 10000.0
                    # Error propagation for ratio: r = x / y.
                    rel_x = merged["stderr"] / merged["mean"].replace(0.0, np.nan)
                    rel_y = (
                        merged["stderr_baseline"]
                        / merged["mean_baseline"].replace(0.0, np.nan)
                    )
                    rel_r = np.sqrt(np.square(rel_x) + np.square(rel_y))
                    ratio = merged["mean"] / merged["mean_baseline"]
                    merged["delta_stderr_bps"] = ratio * rel_r * 10000.0
                    merged["delta_stderr_bps"] = merged["delta_stderr_bps"].fillna(0.0)

                    color = GROUP_COLORS[group]
                    n_runs = run_count_by_group.get(group, 0)
                    ax_delta.plot(
                        merged["step"],
                        merged["delta_bps"],
                        color=color,
                        linewidth=2.0,
                        label=f"{group} - baseline (n={n_runs})",
                    )
                    if n_runs > 1:
                        ax_delta.fill_between(
                            merged["step"],
                            merged["delta_bps"] - merged["delta_stderr_bps"],
                            merged["delta_bps"] + merged["delta_stderr_bps"],
                            color=color,
                            alpha=0.18,
                            linewidth=0,
                        )

                    delta_out = (
                        output_dir
                        / f"aggregated_delta_{group}_vs_baseline_{sanitize_metric_name(metric)}.csv"
                    )
                    merged[["step", "delta_bps", "delta_stderr_bps"]].to_csv(
                        delta_out, index=False
                    )
            else:
                ax_delta.text(
                    0.5,
                    0.5,
                    "Baseline missing for delta view",
                    ha="center",
                    va="center",
                    transform=ax_delta.transAxes,
                    fontsize=10,
                )
        else:
            ax_zoom = ax_aux
            tail_min_y = None
            tail_max_y = None
            tail_max_step = None
            for group in GROUP_PATTERNS:
                agg = agg_by_group.get(group)
                if agg is None or agg.empty:
                    continue

                tail = agg[agg["step"] >= args.tail_start]
                if tail.empty:
                    continue

                color = GROUP_COLORS[group]
                n_runs = run_count_by_group[group]
                ax_zoom.plot(
                    tail["step"],
                    tail["mean"],
                    color=color,
                    linewidth=2.0,
                    label=f"{group} (n={n_runs})",
                )
                if n_runs > 1:
                    lower = tail["mean"] - tail["std"]
                    upper = tail["mean"] + tail["std"]
                    ax_zoom.fill_between(
                        tail["step"],
                        lower,
                        upper,
                        color=color,
                        alpha=0.28,
                        linewidth=0,
                    )
                    # Draw band edges so uncertainty is visible even when narrow.
                    ax_zoom.plot(
                        tail["step"],
                        lower,
                        color=color,
                        linewidth=0.9,
                        alpha=0.7,
                    )
                    ax_zoom.plot(
                        tail["step"],
                        upper,
                        color=color,
                        linewidth=0.9,
                        alpha=0.7,
                    )

                curr_min = float((tail["mean"] - tail["stderr"]).min())
                curr_max = float((tail["mean"] + tail["stderr"]).max())
                tail_min_y = curr_min if tail_min_y is None else min(tail_min_y, curr_min)
                tail_max_y = curr_max if tail_max_y is None else max(tail_max_y, curr_max)
                curr_max_step = int(tail["step"].max())
                tail_max_step = curr_max_step if tail_max_step is None else max(tail_max_step, curr_max_step)

            if tail_min_y is not None and tail_max_y is not None and tail_max_step is not None:
                span = max(tail_max_y - tail_min_y, 1e-6)
                pad = 0.08 * span
                y0 = tail_min_y - pad
                y1 = tail_max_y + pad
                x0 = args.tail_start
                x1 = tail_max_step
                ax_zoom.set_ylim(y0, y1)
                ax_zoom.set_xlim(x0, x1)

                # Draw the same zoom window on the left panel.
                zoom_box = Rectangle(
                    (x0, y0),
                    max(x1 - x0, 1e-9),
                    max(y1 - y0, 1e-9),
                    fill=False,
                    edgecolor="#333333",
                    linewidth=1.4,
                    linestyle="--",
                    alpha=0.9,
                )
                ax_abs.add_patch(zoom_box)

                # Connect left zoom box to right zoom panel.
                conn_top = ConnectionPatch(
                    xyA=(x1, y1),
                    coordsA="data",
                    axesA=ax_abs,
                    xyB=(x0, y1),
                    coordsB="data",
                    axesB=ax_zoom,
                    color="#444444",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.8,
                )
                conn_bottom = ConnectionPatch(
                    xyA=(x1, y0),
                    coordsA="data",
                    axesA=ax_abs,
                    xyB=(x0, y0),
                    coordsB="data",
                    axesB=ax_zoom,
                    color="#444444",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.8,
                )
                fig.add_artist(conn_top)
                fig.add_artist(conn_bottom)
            else:
                ax_zoom.text(
                    0.5,
                    0.5,
                    f"No data at step >= {args.tail_start}",
                    ha="center",
                    va="center",
                    transform=ax_zoom.transAxes,
                    fontsize=10,
                )

        ax_abs.set_ylabel(metric)
        if args.log_y:
            ax_abs.set_yscale("log")
        ax_abs.grid(True, alpha=0.25)
        if not args.no_delta:
            ax_delta = ax_aux
            ax_delta.set_ylabel(f"{metric} delta (bps)")
            ax_delta.grid(True, alpha=0.25)
            ax_delta.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
        else:
            ax_zoom = ax_aux
            ax_zoom.set_ylabel(metric)
            ax_zoom.grid(True, alpha=0.25)

        if row_idx == 0:
            scale_name = "log" if args.log_y else "linear"
            ax_abs.set_title(f"Absolute Curves (mean ± SE, y={scale_name})")
            ax_abs.legend(loc="best")
            if not args.no_delta:
                ax_delta = ax_aux
                ax_delta.set_title("Difference Vs Baseline (basis points, ±SE)")
                ax_delta.legend(loc="best")
            else:
                ax_zoom = ax_aux
                ax_zoom.set_title(f"Tail Zoom (mean ± std, linear y, step >= {args.tail_start})")
                ax_zoom.legend(loc="best")

    axes[-1, 0].set_xlabel("train/step")
    axes[-1, 1].set_xlabel("train/step")
    fig.suptitle("64M Snow Pretrain Validation Curves")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(output, dpi=180)

    print(f"Wrote plot: {output}")
    for group in GROUP_PATTERNS:
        print(f"Included {group} runs: {included_runs.get(group, 0)}")


if __name__ == "__main__":
    main()
