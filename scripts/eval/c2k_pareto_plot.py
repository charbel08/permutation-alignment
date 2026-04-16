"""Pareto plot for the c2k compute-vs-loss sweep.

For each c2k run, pulls val/loss_c1 and val/loss_c2 histories from W&B,
takes the mean of the last N eval points, subtracts the non-tiered baseline
val/loss_c1, and plots:

    x: % FLOPs increase vs non-tiered baseline
    y: val loss gap vs baseline, for C1 (public) and optionally C2 (private/keyed)

Assumes all c2k runs follow the naming convention
    pretrain_150m_fineweb_5pct_c2k_k{K}
in the project `main-pretrain-c2k`, and the baseline is
    baseline_pretrain_150m_fineweb
in the project `main-pretrain`.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


DEFAULT_KS = [1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100]


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
                   default="pretrain_150m_fineweb_5pct_c2k_k",
                   help="Run-name prefix; full name is prefix + str(K)")
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    p.add_argument("--last_n", type=int, default=3,
                   help="Average the last N val points to reduce eval noise")
    p.add_argument("--include_c2", action="store_true",
                   help="Also plot C2 loss gap curve (default: only public/C1)")
    p.add_argument("--x_source", type=str, default="auto",
                   choices=["auto", "summary", "formula"],
                   help="How to compute x-axis FLOPs increase. "
                        "'summary' uses run summary final/flops_increase_pct_vs_baseline, "
                        "'formula' uses 100/K, 'auto' prefers summary then falls back to formula.")
    p.add_argument("--output_dir", type=str,
                   default="outputs/c2k_pareto_150m_5pct")
    p.add_argument("--title", type=str,
                   default="c2k Pareto — 150M, 5% key (fineweb)")
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

    print(f"Baseline: {args.baseline_run} in {args.baseline_project}")
    if args.baseline_run_id:
        base_run = _find_run_by_id(api, args.entity, args.baseline_project, args.baseline_run_id)
        print(f"  using baseline_run_id={args.baseline_run_id} (display_name={base_run.display_name!r})")
    else:
        base_run = _find_run_by_name(api, args.entity, args.baseline_project,
                                     args.baseline_run)
        print(f"  using matched baseline run id={base_run.id}")
    base_loss = _last_n_mean(base_run, "val/loss_c1", args.last_n)
    print(f"  val/loss_c1 (last {args.last_n} mean) = {base_loss:.4f}")

    rows = []  # (K, flops_pct, flops_source, loss_c1, loss_c2, gap_c1, gap_c2)
    for K in args.ks:
        name = f"{args.c2k_run_prefix}{K}"
        print(f"K={K}: {name}")
        try:
            run = _find_run_by_name(api, args.entity, args.c2k_project, name)
        except RuntimeError as e:
            print(f"  [skip] {e}")
            continue
        l1 = _last_n_mean(run, "val/loss_c1", args.last_n)
        try:
            l2 = _last_n_mean(run, "val/loss_c2", args.last_n)
        except RuntimeError:
            l2 = float("nan")
        flops_pct, flops_source = _flops_pct_increase(run, K, args.x_source)
        rows.append((K, flops_pct, flops_source, l1, l2, l1 - base_loss, l2 - base_loss))
        print(
            f"  val/loss_c1={l1:.4f}  "
            f"val/loss_c2={'nan' if l2 != l2 else f'{l2:.4f}'}  "
            f"+FLOPs={flops_pct:.2f}% ({flops_source})"
        )

    if not rows:
        raise SystemExit("no c2k runs found")

    rows.sort(key=lambda r: r[1])  # ascending FLOPs overhead

    # CSV
    csv_path = Path(args.output_dir) / "c2k_pareto.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "flops_pct_increase", "flops_source", "val_loss_c1", "val_loss_c2",
                    "gap_c1", "gap_c2", "baseline_val_loss"])
        for K, pct, src, l1, l2, g1, g2 in rows:
            w.writerow([K, pct, src, l1, l2, g1, g2, base_loss])
    print(f"\nwrote {csv_path}")

    # Plot
    xs = [r[1] for r in rows]
    gap_c1 = [r[5] for r in rows]
    gap_c2 = [r[6] for r in rows]
    ks = [r[0] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, gap_c1, "o-", label="C1 gap  (public: avg_L − L_base)",
            color="C0")
    if args.include_c2:
        ax.plot(xs, gap_c2, "s-", label="C2 gap  (private: C2_L − L_base)",
                color="C3")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")

    for x, y, K in zip(xs, gap_c1, ks):
        ax.annotate(f"K={K}", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, color="C0")
    if args.include_c2:
        for x, y, K in zip(xs, gap_c2, ks):
            ax.annotate(f"K={K}", (x, y), textcoords="offset points",
                        xytext=(4, -10), fontsize=8, color="C3")

    ax.set_xscale("log")
    ax.set_xlabel("FLOPs % increase vs non-tiered baseline")
    ax.set_ylabel("Public val loss gap: avg_L(tiered) − L(non-tiered)")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    png = Path(args.output_dir) / "c2k_pareto.png"
    pdf = Path(args.output_dir) / "c2k_pareto.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
