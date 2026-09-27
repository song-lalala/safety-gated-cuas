"""Analyze a sensitivity sweep produced by `run_sensitivity.py`.

Reads `<sweep-dir>/sweep_summary.json` plus each sub-run's
`per_trial_summaries.json` (M7) and writes:
  - <sweep-dir>/analysis/pivot_<metric>.csv  (rows=sweep_value, cols=policy)
  - <sweep-dir>/analysis/ci_<metric>.csv     (point + ci_lo + ci_hi)
  - <sweep-dir>/analysis/paired_diff_vs_<baseline>.csv  (SafetyGate − baseline)
  - <sweep-dir>/analysis/sweep_curves.png    (line chart with CI bands)
  - <sweep-dir>/analysis/sensitivity_report.md
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# Allow `py experiments\analyze_sensitivity.py ...` to find cuas_sim.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cuas_sim.statistics import (
    bootstrap_aggregate_ci,
    paired_bootstrap_diff,
    aggregate_safe_capture,
    aggregate_unsafe_per_attempt,
    aggregate_false_per_attempt,
    aggregate_abort_success,
    aggregate_late_abort,
    make_rws_aggregator,
)


DEFAULT_METRICS = [
    "risk_weighted_score",
    "safe_opportunity_capture_rate",
    "unsafe_engagement_per_attempt_rate",
    "abort_success_rate",
    "missed_safe_opportunity_rate",
]

POLICY_ORDER = [
    "AlwaysEngagePolicy",
    "DistanceOnlyPolicy",
    "InterceptabilityOnlyPolicy",
    "SafetyGatePolicy",
]


def load_sweep(input_dir: str):
    summary_path = os.path.join(input_dir, "sweep_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    meta_path = os.path.join(input_dir, "sweep_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return rows, meta


def _list_subrun_dirs(sweep_dir: str) -> List[str]:
    return [
        os.path.join(sweep_dir, name)
        for name in sorted(os.listdir(sweep_dir))
        if os.path.isdir(os.path.join(sweep_dir, name))
        and os.path.exists(os.path.join(sweep_dir, name, "per_trial_summaries.json"))
    ]


def _load_per_trial(subrun_dir: str) -> List[SimpleNamespace]:
    """Load per_trial_summaries.json as a list of attribute-access objects.

    The bootstrap aggregator functions in `cuas_sim.statistics` access fields
    like `.safe_opportunity_count`, so wrapping the dicts with
    SimpleNamespace lets us pass them directly without subclassing.
    """
    path = os.path.join(subrun_dir, "per_trial_summaries.json")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [SimpleNamespace(**r) for r in records]


def _load_run_config(subrun_dir: str) -> Dict[str, Any]:
    path = os.path.join(subrun_dir, "run_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _group_trials_by_sweep_and_policy(
    sweep_dir: str,
    sweep_param: str,
) -> Dict[Any, Dict[str, List[SimpleNamespace]]]:
    """Returns {sweep_value: {policy_name: [PerTrialMetrics-like, ...]}}."""
    out: Dict[Any, Dict[str, List[SimpleNamespace]]] = {}
    for subrun_dir in _list_subrun_dirs(sweep_dir):
        cfg = _load_run_config(subrun_dir)
        v = cfg.get(sweep_param)
        if v is None:
            continue
        trials = _load_per_trial(subrun_dir)
        by_policy: Dict[str, List[SimpleNamespace]] = defaultdict(list)
        for t in trials:
            by_policy[t.policy_name].append(t)
        out[v] = by_policy
    return out


METRIC_AGGREGATORS = {
    "risk_weighted_score": None,  # filled in at runtime from run_config weights
    "safe_opportunity_capture_rate": aggregate_safe_capture,
    "unsafe_engagement_per_attempt_rate": aggregate_unsafe_per_attempt,
    "false_engagement_per_attempt_rate": aggregate_false_per_attempt,
    "abort_success_rate": aggregate_abort_success,
    "late_abort_rate": aggregate_late_abort,
}


def _resolve_rws_aggregator(sweep_dir: str):
    """Use the first sub-run's run_config weights as canonical."""
    subruns = _list_subrun_dirs(sweep_dir)
    if not subruns:
        return make_rws_aggregator()
    cfg = _load_run_config(subruns[0])
    return make_rws_aggregator(
        weight_safe_capture=cfg.get("weight_safe_capture", 1.0),
        weight_abort_success=cfg.get("weight_abort_success", 0.5),
        weight_unsafe=cfg.get("weight_unsafe", 2.0),
        weight_false=cfg.get("weight_false", 2.0),
        weight_late=cfg.get("weight_late_abort", 1.0),
    )


def compute_ci_table(
    sweep_dir: str,
    sweep_param: str,
    policies: List[str],
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[Any, Dict[str, Dict[str, float]]]:
    """Returns {sweep_value: {policy: {point, lo, hi}}}."""
    trials_map = _group_trials_by_sweep_and_policy(sweep_dir, sweep_param)
    if metric == "risk_weighted_score":
        aggregator = _resolve_rws_aggregator(sweep_dir)
    else:
        aggregator = METRIC_AGGREGATORS.get(metric)
        if aggregator is None:
            raise ValueError(f"No aggregator registered for metric {metric}")

    result: Dict[Any, Dict[str, Dict[str, float]]] = {}
    for v, by_policy in trials_map.items():
        result[v] = {}
        for p in policies:
            trials = by_policy.get(p, [])
            point, lo, hi = bootstrap_aggregate_ci(
                trials, aggregator, n_boot=n_boot, seed=seed,
            )
            result[v][p] = {"point": point, "lo": lo, "hi": hi}
    return result


def write_ci_csv(
    ci_table: Dict[Any, Dict[str, Dict[str, float]]],
    policies: List[str],
    path: str,
) -> None:
    import csv
    sorted_values = sorted(ci_table.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["sweep_value"]
        for p in policies:
            header.extend([f"{p}_point", f"{p}_lo", f"{p}_hi"])
        w.writerow(header)
        for v in sorted_values:
            row = [v]
            for p in policies:
                cell = ci_table[v].get(p, {})
                row.extend([
                    f"{cell.get('point', float('nan'))}",
                    f"{cell.get('lo', float('nan'))}",
                    f"{cell.get('hi', float('nan'))}",
                ])
            w.writerow(row)


def compute_paired_diff_table(
    sweep_dir: str,
    sweep_param: str,
    challenger: str,
    baseline: str,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[Any, Dict[str, float]]:
    """Per sweep_value paired bootstrap of challenger - baseline.

    Trials are aligned by sorting both groups by (seed, scenario_type). Since
    the runner produces one trial per (seed, scenario) per policy, this gives
    matched pairs that share noise sequences.
    """
    trials_map = _group_trials_by_sweep_and_policy(sweep_dir, sweep_param)
    aggregator = (
        _resolve_rws_aggregator(sweep_dir) if metric == "risk_weighted_score"
        else METRIC_AGGREGATORS[metric]
    )
    result: Dict[Any, Dict[str, float]] = {}
    for v, by_policy in trials_map.items():
        a = sorted(by_policy.get(challenger, []), key=lambda t: (t.seed, t.scenario_type))
        b = sorted(by_policy.get(baseline, []), key=lambda t: (t.seed, t.scenario_type))
        result[v] = paired_bootstrap_diff(a, b, aggregator, n_boot=n_boot, seed=seed)
    return result


def build_pivot(rows: List[Dict[str, Any]], metric: str) -> Dict[Any, Dict[str, float]]:
    """Return {sweep_value: {policy: metric_value}}."""
    pivot: Dict[Any, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        pivot[r["sweep_value"]][r["policy_name"]] = r.get(metric, 0.0)
    return pivot


def write_pivot_csv(pivot: Dict[Any, Dict[str, float]], policies: List[str], path: str) -> None:
    import csv
    sorted_values = sorted(pivot.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep_value"] + policies)
        for v in sorted_values:
            w.writerow([v] + [f"{pivot[v].get(p, '')}" for p in policies])


def plot_sweep_curves(
    rows: List[Dict[str, Any]],
    metrics: List[str],
    policies: List[str],
    sweep_param: str,
    output_path: str,
    ci_tables: Optional[Dict[str, Dict[Any, Dict[str, Dict[str, float]]]]] = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping figures.")
        return False

    n = len(metrics)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4 * rows_n), squeeze=False)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    policy_color = {p: color_cycle[i % len(color_cycle)] for i, p in enumerate(policies)}

    for i, metric in enumerate(metrics):
        ax = axes[i // cols][i % cols]
        pivot = build_pivot(rows, metric)
        sorted_values = sorted(pivot.keys())
        for p in policies:
            color = policy_color[p]
            ys = [pivot[v].get(p, float("nan")) for v in sorted_values]
            ax.plot(sorted_values, ys, marker="o", label=p, color=color)
            # M7: CI band if available for this metric.
            if ci_tables and metric in ci_tables:
                cit = ci_tables[metric]
                los = [cit.get(v, {}).get(p, {}).get("lo", float("nan")) for v in sorted_values]
                his = [cit.get(v, {}).get(p, {}).get("hi", float("nan")) for v in sorted_values]
                ax.fill_between(sorted_values, los, his, alpha=0.15, color=color)
        ax.set_xlabel(sweep_param)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    # hide unused subplots
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"Sensitivity sweep over {sweep_param} (shaded = 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    return True


def write_markdown(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    metrics: List[str],
    policies: List[str],
    path: str,
    ci_tables: Optional[Dict[str, Dict[Any, Dict[str, Dict[str, float]]]]] = None,
    paired_diff: Optional[Dict[str, Dict[Any, Dict[str, float]]]] = None,
    challenger: Optional[str] = None,
    baselines: Optional[List[str]] = None,
) -> None:
    sweep_param = meta["sweep_param"]
    values = sorted({r["sweep_value"] for r in rows})

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Sensitivity sweep — `{sweep_param}`\n\n")
        f.write(f"- Sweep values: {values}\n")
        f.write(f"- Policies: {policies}\n")
        f.write(f"- Rows in sweep_summary: {len(rows)}\n")
        if ci_tables:
            f.write("- 95% bootstrap CIs computed per (sweep_value, policy) using "
                    "resampled trials.\n")
        f.write("\n")

        for metric in metrics:
            f.write(f"## {metric}\n\n")
            if ci_tables and metric in ci_tables:
                cit = ci_tables[metric]
                f.write("| " + sweep_param + " | " + " | ".join(
                    f"{p} (point [CI])" for p in policies) + " |\n")
                f.write("|" + "---|" * (len(policies) + 1) + "\n")
                for v in values:
                    cells = []
                    for p in policies:
                        cell = cit.get(v, {}).get(p, {})
                        pt = cell.get("point", float("nan"))
                        lo = cell.get("lo", float("nan"))
                        hi = cell.get("hi", float("nan"))
                        cells.append(f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                    f.write(f"| {v} | " + " | ".join(cells) + " |\n")
            else:
                f.write("| " + sweep_param + " | " + " | ".join(policies) + " |\n")
                f.write("|" + "---|" * (len(policies) + 1) + "\n")
                pivot = build_pivot(rows, metric)
                for v in values:
                    cells = [f"{pivot[v].get(p, ''):.4f}" if pivot[v].get(p, None) is not None else ""
                             for p in policies]
                    f.write(f"| {v} | " + " | ".join(cells) + " |\n")
            f.write("\n")

        if paired_diff and challenger and baselines:
            f.write(f"## Paired bootstrap diff — `{challenger}` − baseline\n\n")
            f.write("`(point [95% CI], one-sided p)`. Negative point means the challenger is worse.\n\n")
            for baseline in baselines:
                key = baseline
                if key not in paired_diff:
                    continue
                diffs = paired_diff[key]
                f.write(f"### vs `{baseline}` — metric: risk_weighted_score\n\n")
                f.write(f"| {sweep_param} | point diff | CI low | CI high | p (one-sided) |\n")
                f.write("|---|---|---|---|---|\n")
                for v in values:
                    d = diffs.get(v, {})
                    pt = d.get("point_diff", float("nan"))
                    lo = d.get("ci_lo", float("nan"))
                    hi = d.get("ci_hi", float("nan"))
                    pval = d.get("p_value_one_sided", float("nan"))
                    f.write(f"| {v} | {pt:+.4f} | {lo:+.4f} | {hi:+.4f} | {pval:.3f} |\n")
                f.write("\n")


def build_2d_pivot(rows: List[Dict[str, Any]], metric: str, sweep_param: str, sweep_param_2: str,
                   policy: str) -> Dict[Any, Dict[Any, float]]:
    """{value1: {value2: metric}} for a single policy."""
    out: Dict[Any, Dict[Any, float]] = defaultdict(dict)
    key1 = f"sweep_{sweep_param}"
    key2 = f"sweep_{sweep_param_2}"
    for r in rows:
        if r.get("policy_name") != policy:
            continue
        v1, v2 = r.get(key1), r.get(key2)
        if v1 is None or v2 is None:
            continue
        out[v1][v2] = r.get(metric, 0.0)
    return out


def write_2d_pivot_csv(pivot: Dict[Any, Dict[Any, float]], path: str,
                       sweep_param: str, sweep_param_2: str) -> None:
    import csv
    if not pivot:
        with open(path, "w", encoding="utf-8") as f:
            pass
        return
    v1_sorted = sorted(pivot.keys())
    v2_sorted = sorted({v2 for inner in pivot.values() for v2 in inner.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"{sweep_param}\\{sweep_param_2}"] + [str(v) for v in v2_sorted])
        for v1 in v1_sorted:
            row = [str(v1)]
            for v2 in v2_sorted:
                cell = pivot[v1].get(v2)
                row.append("" if cell is None else f"{cell}")
            w.writerow(row)


def plot_2d_heatmap(rows: List[Dict[str, Any]], metric: str, policies: List[str],
                    sweep_param: str, sweep_param_2: str, output_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib/numpy not installed; skipping heatmap.")
        return False

    n = len(policies)
    cols = min(2, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 5 * rows_n), squeeze=False)

    # Collect all values for shared colorbar range
    all_pivots = [build_2d_pivot(rows, metric, sweep_param, sweep_param_2, p) for p in policies]
    all_vals = [v for piv in all_pivots for inner in piv.values() for v in inner.values()]
    if not all_vals:
        return False
    vmin, vmax = min(all_vals), max(all_vals)

    last_im = None
    for i, (policy, pivot) in enumerate(zip(policies, all_pivots)):
        ax = axes[i // cols][i % cols]
        v1_sorted = sorted(pivot.keys())
        v2_sorted = sorted({v2 for inner in pivot.values() for v2 in inner.keys()})
        Z = np.array([[pivot[v1].get(v2, float("nan")) for v2 in v2_sorted] for v1 in v1_sorted])
        im = ax.imshow(Z, aspect="auto", origin="lower",
                        vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(v2_sorted)))
        ax.set_xticklabels([f"{v:g}" for v in v2_sorted])
        ax.set_yticks(range(len(v1_sorted)))
        ax.set_yticklabels([f"{v:g}" for v in v1_sorted])
        ax.set_xlabel(sweep_param_2)
        ax.set_ylabel(sweep_param)
        ax.set_title(policy)
        # Annotate cells
        for r_i, v1 in enumerate(v1_sorted):
            for c_i, v2 in enumerate(v2_sorted):
                val = pivot[v1].get(v2)
                if val is not None:
                    ax.text(c_i, r_i, f"{val:.2f}", ha="center", va="center",
                            color="white" if (val - vmin) / (vmax - vmin + 1e-9) < 0.55 else "black",
                            fontsize=8)
        last_im = im
    # hide unused
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, label=metric)
    fig.suptitle(f"{metric} — 2D sweep ({sweep_param} × {sweep_param_2})")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a sensitivity sweep directory.")
    parser.add_argument("--input-dir", required=True, help="Sweep output directory (contains sweep_summary.json)")
    parser.add_argument("--output-dir", default=None, help="Analysis output directory (default: <input-dir>/analysis)")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help="Comma-separated metric names to include in tables and curves")
    parser.add_argument("--no-plots", action="store_true", help="Disable figure generation")
    parser.add_argument("--no-ci", action="store_true",
                        help="Skip bootstrap CI computation (faster, but loses statistical bands)")
    parser.add_argument("--ci-boot", type=int, default=1000, help="Bootstrap resample count for CIs")
    parser.add_argument("--ci-seed", type=int, default=0, help="Seed for bootstrap RNG (reproducibility)")
    parser.add_argument("--paired-challenger", type=str, default="SafetyGatePolicy",
                        help="Policy to use as the challenger in paired bootstrap diff")
    parser.add_argument("--paired-baselines", type=str, default="AlwaysEngagePolicy,InterceptabilityOnlyPolicy",
                        help="Comma-separated baselines compared against the challenger")
    args = parser.parse_args(argv)

    rows, meta = load_sweep(args.input_dir)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    policies = [p for p in POLICY_ORDER if any(r["policy_name"] == p for r in rows)]
    if not policies:
        policies = sorted({r["policy_name"] for r in rows})

    out_dir = args.output_dir or os.path.join(args.input_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # 2-D sweep detection — fall back to 1-D flow when sweep_param_2 is null.
    sweep_param_2 = meta.get("sweep_param_2")
    if sweep_param_2:
        print(f"[2D sweep] {meta['sweep_param']} × {sweep_param_2}")
        for metric in metrics:
            for policy in policies:
                pivot = build_2d_pivot(rows, metric, meta["sweep_param"], sweep_param_2, policy)
                if pivot:
                    safe_policy = policy.replace("Policy", "")
                    csv_path = os.path.join(
                        out_dir, f"pivot2d_{metric}__{safe_policy}.csv"
                    )
                    write_2d_pivot_csv(pivot, csv_path, meta["sweep_param"], sweep_param_2)
        if not args.no_plots:
            for metric in metrics:
                heat_path = os.path.join(out_dir, f"heatmap_{metric}.png")
                plot_2d_heatmap(rows, metric, policies, meta["sweep_param"],
                                sweep_param_2, heat_path)
        # Skip 1D CI/figures for 2D sweeps; the heatmap is the primary output.
        print(f"Analysis written to {out_dir}")
        return 0

    for metric in metrics:
        pivot = build_pivot(rows, metric)
        write_pivot_csv(pivot, policies, os.path.join(out_dir, f"pivot_{metric}.csv"))

    ci_tables: Dict[str, Dict[Any, Dict[str, Dict[str, float]]]] = {}
    paired_diff_tables: Dict[str, Dict[Any, Dict[str, float]]] = {}

    if not args.no_ci:
        for metric in metrics:
            try:
                ci = compute_ci_table(
                    args.input_dir, meta["sweep_param"], policies, metric,
                    n_boot=args.ci_boot, seed=args.ci_seed,
                )
            except (KeyError, ValueError) as e:
                print(f"Skipping CI for {metric}: {e}")
                continue
            ci_tables[metric] = ci
            write_ci_csv(ci, policies, os.path.join(out_dir, f"ci_{metric}.csv"))

        # Paired diff is computed against risk_weighted_score (most useful).
        baselines = [b.strip() for b in args.paired_baselines.split(",") if b.strip()]
        for baseline in baselines:
            if baseline not in policies or args.paired_challenger not in policies:
                continue
            paired = compute_paired_diff_table(
                args.input_dir, meta["sweep_param"],
                args.paired_challenger, baseline,
                "risk_weighted_score",
                n_boot=args.ci_boot, seed=args.ci_seed,
            )
            paired_diff_tables[baseline] = paired
            csv_path = os.path.join(out_dir, f"paired_diff_RWS_vs_{baseline}.csv")
            _write_paired_diff_csv(paired, meta["sweep_param"], csv_path)

    write_markdown(
        rows, meta, metrics, policies,
        os.path.join(out_dir, "sensitivity_report.md"),
        ci_tables=ci_tables if ci_tables else None,
        paired_diff=paired_diff_tables if paired_diff_tables else None,
        challenger=args.paired_challenger,
        baselines=[b.strip() for b in args.paired_baselines.split(",") if b.strip()],
    )

    if not args.no_plots:
        plot_path = os.path.join(out_dir, "sweep_curves.png")
        plot_sweep_curves(
            rows, metrics, policies, meta["sweep_param"], plot_path,
            ci_tables=ci_tables if ci_tables else None,
        )

    print(f"Analysis written to {out_dir}")
    return 0


def _write_paired_diff_csv(
    table: Dict[Any, Dict[str, float]],
    sweep_param: str,
    path: str,
) -> None:
    import csv
    sorted_values = sorted(table.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([sweep_param, "point_diff", "ci_lo", "ci_hi", "p_value_one_sided"])
        for v in sorted_values:
            d = table[v]
            w.writerow([
                v,
                d.get("point_diff", ""),
                d.get("ci_lo", ""),
                d.get("ci_hi", ""),
                d.get("p_value_one_sided", ""),
            ])


if __name__ == "__main__":
    sys.exit(main())
