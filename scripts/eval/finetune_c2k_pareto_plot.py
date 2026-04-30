"""Pareto plot for the c2k **finetune** compute-vs-private-loss sweep.

Counterpart to c2k_pareto_plot.py, targeting the post-finetune project
(main-finetune-c2k) rather than the pretrain project. For each c2k
finetune run, pulls `Val Private/C1 Loss` and `Val Private/C2 Loss`
histories from W&B, takes the mean of the last N eval points, and plots:

    x: % FLOPs increase vs non-tiered baseline (log scale, 100/K)
    y: raw validation losses for C1 (public) and C2 (private)

No baseline horizontal line — this plot is about the relative compute/
loss tradeoff across K, not absolute distance from a non-tiered model.

Runs are located by substring match + a digit-safe `k{K}` boundary
anywhere in the display name, and filtered to those whose name contains
`--name_contains` (default "c2k_key5pct" for the 5% key sweep).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import wandb


DEFAULT_KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", type=str, default=None,
                   help="W&B entity (defaults to default entity)")
    p.add_argument("--project", type=str, default="main-finetune-c2k")
    p.add_argument("--name_contains", type=str, default="c2k_key5pct",
                   help="Substring the run display name must contain.")
    p.add_argument("--name_filter", type=str, default=None,
                   help="Additional substring required in the display name "
                        "(e.g. 'resweep' to restrict to re-swept runs).")
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--exclude_ks", type=int, nargs="*", default=[])
    p.add_argument("--last_n", type=int, default=3,
                   help="Average the last N val points to reduce eval noise.")
    p.add_argument("--c1_key", type=str, default="Val Private/C1 Loss")
    p.add_argument("--c2_key", type=str, default="Val Private/C2 Loss")
    p.add_argument("--retain_c1_key", type=str, default="Val Retain/C1 Loss",
                   help="Optional retain-set C1 metric; pass empty string to "
                        "disable the retain curves.")
    p.add_argument("--retain_c2_key", type=str, default="Val Retain/C2 Loss")
    p.add_argument("--baseline_project", type=str, default="main-finetune",
                   help="W&B project hosting the non-c2k finetune baseline.")
    p.add_argument("--baseline_run", type=str,
                   default="finetune_150m_fineweb2_spa_key5pct_kl0",
                   help="Display name of the baseline finetune run; its last "
                        "Val Private/C2 Loss is plotted as a dashed reference. "
                        "Pass empty string to disable.")
    p.add_argument("--baseline_run_id", type=str, default=None,
                   help="Exact W&B run ID for the baseline (recommended when "
                        "display names collide).")
    p.add_argument("--output_dir", type=str,
                   default="outputs/finetune_c2k_pareto")
    p.add_argument("--title", type=str, default="",
                   help="Optional figure title (empty = no title).")
    return p.parse_args()


def _list_runs(api: wandb.Api, entity: str | None, project: str) -> list:
    path = f"{entity}/{project}" if entity else project
    print(f"Listing runs in {path} ...", flush=True)
    runs = list(api.runs(path=path))
    print(f"  {len(runs)} runs loaded", flush=True)
    return runs


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


def _find_run_by_id(api: wandb.Api, entity: str | None, project: str,
                    run_id: str):
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    try:
        return api.run(path)
    except wandb.errors.CommError as exc:
        raise RuntimeError(f"No run with id {run_id!r} at {path}") from exc


def _find_run_for_k(all_runs: list, project_path: str,
                    contains: str | None, K: int,
                    name_filter: str | None):
    """Return a finished run whose display name contains `k{K}` at a
    digit-safe word boundary and matches the optional substring filters."""
    pattern = re.compile(rf"(?<!\d)k{K}(?:_|$)")
    matches = []
    for r in all_runs:
        name = r.display_name
        if not pattern.search(name):
            continue
        if contains and contains not in name:
            continue
        if name_filter and name_filter not in name:
            continue
        if r.state != "finished":
            continue
        matches.append(r)
    if not matches:
        parts = []
        if contains:
            parts.append(f"contains {contains!r}")
        parts.append(f"k{K} boundary")
        if name_filter:
            parts.append(f"contains {name_filter!r}")
        raise RuntimeError(
            f"No finished run matching {' + '.join(parts)} in {project_path}"
        )
    if len(matches) > 1:
        matches.sort(key=lambda r: r.created_at, reverse=True)
        names = [r.display_name for r in matches]
        print(f"  [warn] {len(matches)} runs matched K={K}: {names}; using newest")
    return matches[0]


def _last_n_mean(run, key: str, n: int) -> float:
    vals = []
    for row in run.scan_history(keys=[key], page_size=1000):
        v = row.get(key)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        raise RuntimeError(f"run {run.name!r} has no values for {key!r}")
    tail = vals[-n:] if len(vals) >= n else vals
    return sum(tail) / len(tail)


def main() -> None:
    args = parse_args()
    api = wandb.Api()

    os.makedirs(args.output_dir, exist_ok=True)

    excluded = [k for k in args.ks if k in args.exclude_ks]
    if excluded:
        print(f"Excluding Ks: {excluded}")
    ks = [k for k in args.ks if k not in args.exclude_ks]

    baseline_c2 = float("nan")
    if args.baseline_run or args.baseline_run_id:
        print(f"Baseline: {args.baseline_run!r} in {args.baseline_project}")
        if args.baseline_run_id:
            base_run = _find_run_by_id(api, args.entity,
                                       args.baseline_project,
                                       args.baseline_run_id)
            print(f"  using baseline_run_id={args.baseline_run_id} "
                  f"(display_name={base_run.display_name!r})")
        else:
            base_run = _find_run_by_name(api, args.entity,
                                         args.baseline_project,
                                         args.baseline_run)
            print(f"  using matched baseline run id={base_run.id}")
        baseline_c2 = _last_n_mean(base_run, args.c2_key, args.last_n)
        print(f"  {args.c2_key} (last {args.last_n} mean) = {baseline_c2:.4f}",
              flush=True)

    all_runs = _list_runs(api, args.entity, args.project)
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project

    include_retain = bool(args.retain_c1_key) and bool(args.retain_c2_key)

    rows = []  # (K, flops_pct, c1, c2, retain_c1, retain_c2, run_name)
    for K in ks:
        try:
            run = _find_run_for_k(all_runs, project_path,
                                  args.name_contains, K,
                                  name_filter=args.name_filter)
        except RuntimeError as e:
            print(f"K={K}: [skip] {e}", flush=True)
            continue
        print(f"K={K}: {run.display_name!r}", flush=True)
        try:
            c1 = _last_n_mean(run, args.c1_key, args.last_n)
        except RuntimeError as e:
            print(f"  [skip] {e}", flush=True)
            continue
        try:
            c2 = _last_n_mean(run, args.c2_key, args.last_n)
        except RuntimeError:
            c2 = float("nan")
        rc1 = rc2 = float("nan")
        if include_retain:
            try:
                rc1 = _last_n_mean(run, args.retain_c1_key, args.last_n)
            except RuntimeError:
                rc1 = float("nan")
            try:
                rc2 = _last_n_mean(run, args.retain_c2_key, args.last_n)
            except RuntimeError:
                rc2 = float("nan")
        flops_pct = 100.0 / K
        rows.append((K, flops_pct, c1, c2, rc1, rc2, run.display_name))
        extras = ""
        if include_retain:
            extras = (f"  {args.retain_c1_key}={'nan' if rc1 != rc1 else f'{rc1:.4f}'}"
                      f"  {args.retain_c2_key}={'nan' if rc2 != rc2 else f'{rc2:.4f}'}")
        print(
            f"  {args.c1_key}={c1:.4f}  "
            f"{args.c2_key}={'nan' if c2 != c2 else f'{c2:.4f}'}"
            f"{extras}  "
            f"+FLOPs={flops_pct:.2f}%",
            flush=True,
        )

    if not rows:
        raise SystemExit("no runs found")

    rows.sort(key=lambda r: r[1])  # ascending FLOPs overhead

    csv_path = Path(args.output_dir) / "finetune_c2k_pareto.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "flops_pct_increase", "val_private_c1_loss",
                    "val_private_c2_loss", "val_retain_c1_loss",
                    "val_retain_c2_loss", "run_name"])
        for K, pct, c1, c2, rc1, rc2, name in rows:
            w.writerow([K, pct, c1, c2, rc1, rc2, name])
    print(f"\nwrote {csv_path}")

    xs = [r[1] for r in rows]
    c1 = [r[2] for r in rows]
    c2 = [r[3] for r in rows]
    retain_c1 = [r[4] for r in rows]
    retain_c2 = [r[5] for r in rows]
    ks_sorted = [r[0] for r in rows]

    TEAL = "#008080"
    PURPLE = "#662E7D"

    fig, ax = plt.subplots(figsize=(9, 6), dpi=600)

    if baseline_c2 == baseline_c2:  # not NaN
        ax.axhline(baseline_c2, color="gray", lw=2.0, ls="--")
        trans = mtransforms.blended_transform_factory(ax.transAxes,
                                                      ax.transData)
        ax.annotate(
            rf"Targeted Finetune $\mathcal{{C}}_K$, $\beta=0$ ({baseline_c2:.4f})",
            xy=(0.02, baseline_c2), xycoords=trans,
            xytext=(0, 6), textcoords="offset points",
            fontsize=12, color="gray",
            va="bottom", ha="left",
        )

    # Thin dashed connectors spanning every point at each f-value (all four
    # curves: Private C1/C2 + Retain C1/C2 when present).
    def _stack_ys(i):
        return [v for v in (c1[i], c2[i], retain_c1[i], retain_c2[i]) if v == v]

    for i, x in enumerate(xs):
        ys = _stack_ys(i)
        if len(ys) < 2:
            continue
        ax.plot([x, x], [min(ys), max(ys)], color="gray", lw=2.0, ls="--",
                alpha=0.7, zorder=1)

    pub_private_label = r"$\mathcal{C}_{\mathrm{pub}}$ (Private)" if include_retain else r"$\mathcal{C}_{\mathrm{pub}}$"
    k_private_label   = r"$\mathcal{C}_{K}$ (Private)"           if include_retain else r"$\mathcal{C}_{K}$"
    h_pub_priv, = ax.plot(
        xs, c1, "o-", color=TEAL, linewidth=3.0, markersize=8,
        label=pub_private_label, zorder=3,
    )
    h_k_priv, = ax.plot(
        xs, c2, "s-", color=PURPLE, linewidth=3.0, markersize=8,
        label=k_private_label, zorder=3,
    )

    h_pub_pub = h_k_pub = None
    if include_retain and any(v == v for v in retain_c1):
        h_pub_pub, = ax.plot(
            xs, retain_c1, "^:", color=TEAL, linewidth=2.0, markersize=7,
            label=r"$\mathcal{C}_{\mathrm{pub}}$ (Public)", zorder=3,
        )
    if include_retain and any(v == v for v in retain_c2):
        h_k_pub, = ax.plot(
            xs, retain_c2, "D:", color=PURPLE, linewidth=2.0, markersize=7,
            label=r"$\mathcal{C}_{K}$ (Public)", zorder=3,
        )

    # Label each connector at the topmost point with $f=K$ (centered above).
    default_placement = dict(xytext=(0, 8), ha="center", va="bottom")
    for i, (x, K) in enumerate(zip(xs, ks_sorted)):
        ys = _stack_ys(i)
        if not ys:
            continue
        y_top = max(ys)
        ax.annotate(rf"$f={K}$", (x, y_top), textcoords="offset points",
                    fontsize=10, color="black", **default_placement)

    axis_fs = 17
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs % Increase vs Non-Tiered Baseline", fontsize=axis_fs)
    ax.set_ylabel("Validation Loss", fontsize=axis_fs)
    if args.title:
        ax.set_title(args.title)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(which="major", axis="y", alpha=0.34, linewidth=1.2)
    ax.grid(which="major", axis="x", alpha=0.20, linewidth=1.05)
    ax.grid(which="minor", axis="x", alpha=0.10, linewidth=0.8)
    # Legend grid: matplotlib fills columns top-to-bottom, so to render
    #   [ C_K (Private)  | C_pub (Private) ]
    #   [ C_K (Public)   | C_pub (Public)  ]
    # pass handles in column-major order: col0 top→bottom, then col1 top→bottom.
    if h_k_pub is not None and h_pub_pub is not None:
        legend_handles = [h_k_priv, h_k_pub, h_pub_priv, h_pub_pub]
    else:
        legend_handles = [h_k_priv, h_pub_priv]
    legend_labels = [h.get_label() for h in legend_handles]
    # Place the legend in the whitespace band above the Public curves and
    # below the C_K (Private) curve, anchored to the right edge.
    ax.legend(legend_handles, legend_labels, ncol=2, fontsize=12,
              frameon=True, loc="center right",
              bbox_to_anchor=(1.0, 0.35))
    fig.tight_layout()

    png = Path(args.output_dir) / "finetune_c2k_pareto.png"
    pdf = Path(args.output_dir) / "finetune_c2k_pareto.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
