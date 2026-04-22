"""Pareto plot for the c2k compute-vs-loss sweep.

For each c2k run, pulls val/loss_c1 and val/loss_c2 histories from W&B,
takes the mean of the last N eval points, and plots:

    x: % FLOPs increase vs non-tiered baseline (log scale)
    y: raw val losses for C1 and C2
    horizontal line: non-tiered baseline val/loss_c1

Runs are located by prefix + K (with word-boundary after K) and filtered
by an optional substring that must appear in the display name.
"""

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import wandb


DEFAULT_KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", type=str, default=None,
                   help="W&B entity (defaults to default entity)")
    p.add_argument("--c2k_project", type=str, default="main-pretrain-c2k")
    p.add_argument("--baseline_project", type=str, default="main-pretrain")
    p.add_argument("--baseline_run", type=str,
                   default="baseline_pretrain_150m_fineweb",
                   help="Display name of the non-tiered baseline run")
    p.add_argument("--baseline_run_id", type=str, default=None,
                   help="Exact W&B run ID for baseline (recommended when display names collide)")
    p.add_argument("--c2k_run_prefix", type=str,
                   default="pretrain_150m_fineweb_5pct_c2k",
                   help="Substring that must appear in the run's display name. "
                        "Used alongside the K tail-match to disambiguate runs.")
    p.add_argument("--name_filter", type=str, default=None,
                   help="Substring that must also appear in the run's display name "
                        "(e.g. 'resweep' to restrict to re-swept runs).")
    p.add_argument("--name_filter_exempt_ks", type=int, nargs="*", default=[],
                   help="Ks for which --name_filter is ignored (fall back to "
                        "the non-filtered run in the same project).")
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--exclude_ks", type=int, nargs="*", default=[200],
                   help="Ks to drop from --ks before lookup (default: 200).")
    p.add_argument("--last_n", type=int, default=3,
                   help="Average the last N val points to reduce eval noise")
    p.add_argument("--x_source", type=str, default="auto",
                   choices=["auto", "summary", "formula"],
                   help="How to compute x-axis FLOPs increase. "
                        "'summary' uses run summary final/flops_increase_pct_vs_baseline, "
                        "'formula' uses 100/K, 'auto' prefers summary then falls back to formula.")
    p.add_argument("--output_dir", type=str,
                   default="outputs/c2k_pareto_150m_5pct")
    p.add_argument("--title", type=str,
                   default="c2k — 150M, 5% key (fineweb)")
    return p.parse_args()


def _find_run_by_name(api: wandb.Api, entity: str | None, project: str,
                      name: str):
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path=path, filters={"display_name": name}))
    if not runs:
        raise RuntimeError(f"No run named {name!r} in {path}")
    if len(runs) > 1:
        print(f"  [warn] multiple runs named {name!r}; using newest")
        runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs[0]


def _find_run_by_id(api: wandb.Api, entity: str | None, project: str, run_id: str):
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    try:
        return api.run(path)
    except wandb.errors.CommError as exc:
        raise RuntimeError(f"No run with id {run_id!r} at {path}") from exc


def _list_runs(api: wandb.Api, entity: str | None, project: str) -> list:
    """Page through all runs in a project once and return them as a list."""
    path = f"{entity}/{project}" if entity else project
    print(f"Listing runs in {path} ...", flush=True)
    runs = list(api.runs(path=path))
    print(f"  {len(runs)} runs loaded", flush=True)
    return runs


def _find_run_for_k(all_runs: list, project_path: str,
                    prefix: str | None, K: int, name_filter: str | None):
    """Return the run whose display name ends in `k{K}` (with a digit-safe
    boundary so K=1 doesn't match k10/k100), and optionally contains both
    `prefix` and `name_filter` as substrings."""
    pattern = re.compile(rf"(?<!\d)k{K}$")
    matches = []
    for r in all_runs:
        name = r.display_name
        if not pattern.search(name):
            continue
        if prefix and prefix not in name:
            continue
        if name_filter and name_filter not in name:
            continue
        matches.append(r)
    if not matches:
        parts = []
        if prefix:
            parts.append(f"prefix {prefix!r}")
        parts.append(f"k{K} tail")
        if name_filter:
            parts.append(f"substr {name_filter!r}")
        raise RuntimeError(f"No run matching {' + '.join(parts)} in {project_path}")
    if len(matches) > 1:
        matches.sort(key=lambda r: r.created_at, reverse=True)
        names = [r.display_name for r in matches]
        print(f"  [warn] {len(matches)} runs matched K={K}: {names}; using newest")
    return matches[0]


def _last_n_mean(run, key: str, n: int) -> float:
    """Mean of the last n non-null values of `key` from a run's history."""
    vals = []
    for row in run.scan_history(keys=[key, "train/step"], page_size=1000):
        v = row.get(key)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        raise RuntimeError(f"run {run.name!r} has no values for {key!r}")
    tail = vals[-n:] if len(vals) >= n else vals
    return sum(tail) / len(tail)


def _flops_pct_increase(run, K: int, x_source: str) -> tuple[float, str]:
    """Return (% increase, source_tag) for x-axis."""
    summary_key = "final/flops_increase_pct_vs_baseline"
    summary_val = run.summary.get(summary_key)
    if x_source in ("auto", "summary"):
        if summary_val is not None:
            return float(summary_val), summary_key
        if x_source == "summary":
            raise RuntimeError(
                f"run {run.name!r} has no summary key {summary_key!r}; "
                "re-run with --x_source formula or auto"
            )
    return 100.0 / K, "formula_100_over_K"


def main():
    args = parse_args()
    api = wandb.Api()

    os.makedirs(args.output_dir, exist_ok=True)

    excluded = [k for k in args.ks if k in args.exclude_ks]
    if excluded:
        print(f"Excluding Ks: {excluded}")
    ks = [k for k in args.ks if k not in args.exclude_ks]

    print(f"Baseline: {args.baseline_run} in {args.baseline_project}")
    if args.baseline_run_id:
        base_run = _find_run_by_id(api, args.entity, args.baseline_project,
                                   args.baseline_run_id)
        print(f"  using baseline_run_id={args.baseline_run_id} "
              f"(display_name={base_run.display_name!r})")
    else:
        base_run = _find_run_by_name(api, args.entity, args.baseline_project,
                                     args.baseline_run)
        print(f"  using matched baseline run id={base_run.id}")
    base_loss = _last_n_mean(base_run, "val/loss_c1", args.last_n)
    print(f"  val/loss_c1 (last {args.last_n} mean) = {base_loss:.4f}", flush=True)

    # Fetch the c2k project run list once; per-K lookup is then in-memory.
    all_runs = _list_runs(api, args.entity, args.c2k_project)
    project_path = f"{args.entity}/{args.c2k_project}" if args.entity else args.c2k_project

    exempt = set(args.name_filter_exempt_ks)
    rows = []  # (K, flops_pct, flops_source, loss_c1, loss_c2, run_name)
    for K in ks:
        filt = None if K in exempt else args.name_filter
        try:
            run = _find_run_for_k(all_runs, project_path,
                                  args.c2k_run_prefix, K,
                                  name_filter=filt)
        except RuntimeError as e:
            print(f"K={K}: [skip] {e}", flush=True)
            continue
        print(f"K={K}: {run.display_name!r}", flush=True)
        l1 = _last_n_mean(run, "val/loss_c1", args.last_n)
        try:
            l2 = _last_n_mean(run, "val/loss_c2", args.last_n)
        except RuntimeError:
            l2 = float("nan")
        flops_pct, flops_source = _flops_pct_increase(run, K, args.x_source)
        rows.append((K, flops_pct, flops_source, l1, l2, run.display_name))
        print(f"  val/loss_c1={l1:.4f}  "
              f"val/loss_c2={'nan' if l2 != l2 else f'{l2:.4f}'}  "
              f"+FLOPs={flops_pct:.2f}% ({flops_source})", flush=True)

    if not rows:
        raise SystemExit("no c2k runs found")

    rows.sort(key=lambda r: r[1])  # ascending FLOPs overhead

    # CSV
    csv_path = Path(args.output_dir) / "c2k_pareto.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "flops_pct_increase", "flops_source",
                    "val_loss_c1", "val_loss_c2", "baseline_val_loss_c1",
                    "run_name"])
        for K, pct, src, l1, l2, name in rows:
            w.writerow([K, pct, src, l1, l2, base_loss, name])
    print(f"\nwrote {csv_path}")

    # Plot
    xs = [r[1] for r in rows]
    c1 = [r[3] for r in rows]
    c2 = [r[4] for r in rows]
    ks_sorted = [r[0] for r in rows]

    TEAL = "#008080"
    PURPLE = "#662E7D"

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.axhline(base_loss, color="gray", lw=2.0, ls="--")
    # Label the baseline line in-figure so it doesn't clutter the legend.
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.annotate(
        f"Non-tiered baseline ({base_loss:.4f})",
        xy=(0.02, base_loss), xycoords=trans,
        xytext=(0, 6), textcoords="offset points",
        fontsize=12, color="gray",
        va="bottom", ha="left",
    )
    # Thin vertical connectors between the two curves at each f-value.
    # Drawn before the data lines so markers sit on top.
    for x, y1, y2 in zip(xs, c1, c2):
        if y2 != y2:  # skip NaN
            continue
        ax.plot([x, x], [y1, y2], color="gray", lw=2.0, ls="--",
                alpha=0.7, zorder=1)

    ax.plot(xs, c1, "o-", color=TEAL, linewidth=3.0, markersize=8,
            label=r"$\mathcal{C}_{\mathrm{pub}}$", zorder=3)
    ax.plot(xs, c2, "s-", color=PURPLE, linewidth=3.0, markersize=8,
            label=r"$\mathcal{C}_{K}$", zorder=3)

    # Label each connector at the top with $f=K$. A few Ks crowd the
    # top-left corner of the plot; override their placement to sit to the
    # right of the marker instead of directly above it.
    label_placements = {
        1000: dict(xytext=(10, -2), ha="left", va="center"),
        500:  dict(xytext=(10, -2), ha="left", va="center"),
        100:  dict(xytext=(14, 8),  ha="center", va="bottom"),
    }
    default_placement = dict(xytext=(0, 8), ha="center", va="bottom")
    for x, y1, y2, K in zip(xs, c1, c2, ks_sorted):
        y_top = y1 if (y2 != y2) else max(y1, y2)
        placement = label_placements.get(K, default_placement)
        ax.annotate(rf"$f={K}$", (x, y_top), textcoords="offset points",
                    fontsize=10, color="black", **placement)

    axis_fs = 17

    ax.set_xscale("log")
    ax.set_xlabel("FLOPs % Increase vs Non-Tiered Baseline", fontsize=axis_fs)
    ax.set_ylabel("Validation Loss", fontsize=axis_fs)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(which="major", axis="y", alpha=0.34, linewidth=1.2)
    ax.grid(which="major", axis="x", alpha=0.20, linewidth=1.05)
    ax.grid(which="minor", axis="x", alpha=0.10, linewidth=0.8)
    ax.legend(ncol=2, fontsize=12, frameon=True, loc="best")
    fig.tight_layout()

    png = Path(args.output_dir) / "c2k_pareto.png"
    pdf = Path(args.output_dir) / "c2k_pareto.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
