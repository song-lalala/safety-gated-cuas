"""Sensitivity sweep runner.

Runs `MonteCarloRunner` repeatedly while varying one MonteCarloConfig field
across a list of values, and writes both per-run outputs and a top-level
sweep_summary aggregating per-policy metrics across the sweep.

Usage example (PowerShell):

    py experiments\\run_sensitivity.py `
        --sweep-param observation_noise_sigma `
        --sweep-values "0.0,0.05,0.1,0.2,0.3,0.5" `
        --base-run-name v18_sigma_sweep `
        --num-seeds 100 --steps-per-trial 20 `
        --distance-engage-threshold 12.0 `
        --safety-risk-score-threshold 0.7 `
        --safety-identification-confidence-threshold 0.6 `
        --safety-tracking-confidence-threshold 0.6 `
        --safety-interceptability-threshold 0.5 `
        --safety-abort-feasibility-threshold 0.5

Each value `v` produces a sub-run named `<base-run-name>_<param>_<value>`
under `--output-dir`. A `sweep_summary.csv` / `.json` is written at the
top of the sweep directory.
"""
import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

# Allow `py experiments\run_sensitivity.py ...` to find cuas_sim without install.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner
from cuas_sim.results import export_monte_carlo_result, write_json, write_dict_rows_csv
from cuas_sim.scenario import ScenarioType


SWEEPABLE_PARAMS = {
    "observation_noise_sigma",
    "observation_velocity_noise_sigma",
    "safety_tracking_abort_threshold",
    "safety_tracking_confidence_threshold",
    "safety_identification_confidence_threshold",
    "safety_risk_score_threshold",
    "safety_interceptability_threshold",
    "safety_abort_feasibility_threshold",
    "abort_horizon_steps",
    "abort_risk_zone_safety_buffer",
    "distance_engage_threshold",
    "late_abort_lag_steps",
    "weight_safe_capture",
    "weight_abort_success",
    "weight_unsafe",
    "weight_false",
    "weight_late_abort",
    # M3 sweep targets
    "risk_k_sigma",
    "risk_abort_margin",
    "risk_clearance_falloff",
}

INT_PARAMS = {"abort_horizon_steps", "late_abort_lag_steps"}


def parse_value(param_name: str, raw: str):
    raw = raw.strip()
    if param_name in INT_PARAMS:
        return int(raw)
    return float(raw)


def parse_sweep_values(param_name: str, csv_values: str) -> List:
    return [parse_value(param_name, v) for v in csv_values.split(",") if v.strip()]


def parse_scenarios(value: Optional[str]) -> Optional[List[ScenarioType]]:
    if not value:
        return None
    out = []
    for s in value.split(","):
        s = s.strip()
        if s:
            out.append(ScenarioType(s))
    return out or None


def parse_policies(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    out = [p.strip() for p in value.split(",") if p.strip()]
    return out or None


def build_base_config(args: argparse.Namespace) -> MonteCarloConfig:
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    return MonteCarloConfig(
        seeds=seeds,
        steps_per_trial=args.steps_per_trial,
        prediction_horizon_steps=args.prediction_horizon_steps,
        s4_prediction_horizon_steps=args.s4_prediction_horizon_steps,
        distance_engage_threshold=args.distance_engage_threshold,
        safety_tracking_abort_threshold=args.safety_tracking_abort_threshold,
        safety_tracking_confidence_threshold=args.safety_tracking_confidence_threshold,
        safety_identification_confidence_threshold=args.safety_identification_confidence_threshold,
        safety_risk_score_threshold=args.safety_risk_score_threshold,
        safety_interceptability_threshold=args.safety_interceptability_threshold,
        safety_abort_feasibility_threshold=args.safety_abort_feasibility_threshold,
        abort_horizon_steps=args.abort_horizon_steps,
        abort_risk_zone_safety_buffer=args.abort_risk_zone_safety_buffer,
        risk_k_sigma=args.risk_k_sigma,
        risk_abort_margin=args.risk_abort_margin,
        risk_clearance_falloff=args.risk_clearance_falloff,
        observation_noise_sigma=args.observation_noise_sigma,
        observation_velocity_noise_sigma=args.observation_velocity_noise_sigma,
        late_abort_lag_steps=args.late_abort_lag_steps,
        weight_safe_capture=args.weight_safe_capture,
        weight_abort_success=args.weight_abort_success,
        weight_unsafe=args.weight_unsafe,
        weight_false=args.weight_false,
        weight_late_abort=args.weight_late_abort,
        include_scenarios=parse_scenarios(args.scenarios),
        include_policies=parse_policies(args.policies),
    )


def _value_to_run_suffix(v) -> str:
    # Make a filesystem-safe slug for a float/int value.
    if isinstance(v, int):
        return f"{v}"
    s = f"{v:g}"  # short float repr
    return s.replace(".", "p").replace("-", "neg")


def run_single(
    base_config: MonteCarloConfig,
    overrides: Dict[str, Any],
    run_dir: str,
) -> Dict[str, Any]:
    """Run one Monte Carlo experiment with config field overrides applied.

    `overrides` is a dict {field_name: value}. For backward compat the caller
    may pass a single-field override (used by 1-D sweeps) or multiple fields
    (used by 2-D sweeps).
    """
    config_dict = asdict(base_config)
    config_dict.update(overrides)
    # `include_scenarios` may have enum objects in asdict output; restore those.
    if config_dict.get("include_scenarios"):
        config_dict["include_scenarios"] = base_config.include_scenarios
    config = MonteCarloConfig(**config_dict)

    runner = MonteCarloRunner(config)
    result = runner.run()

    os.makedirs(run_dir, exist_ok=True)
    paths = export_monte_carlo_result(result, run_dir)
    # also store run_config for reproducibility
    run_config_serialisable = {
        **{k: v for k, v in asdict(config).items() if k not in ("seeds", "include_scenarios")},
        "seed_start": base_config.seeds[0] if base_config.seeds else None,
        "seed_end": base_config.seeds[-1] if base_config.seeds else None,
        "num_seeds": len(base_config.seeds),
        "include_scenarios": (
            [s.value for s in base_config.include_scenarios]
            if base_config.include_scenarios
            else None
        ),
    }
    write_json(run_config_serialisable, os.path.join(run_dir, "run_config.json"))

    rows = []
    for summary in result.summaries:
        row = summary.to_dict()
        for k, v in overrides.items():
            row[f"sweep_{k}"] = v
        # Keep legacy keys for 1-D compatibility.
        if len(overrides) == 1:
            ((k, v),) = overrides.items()
            row["sweep_param"] = k
            row["sweep_value"] = v
        rows.append(row)
    return {"rows": rows, "paths": paths}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a sensitivity sweep over one or two MonteCarloConfig fields.")
    parser.add_argument("--sweep-param", required=True, choices=sorted(SWEEPABLE_PARAMS),
                        help="MonteCarloConfig field to sweep")
    parser.add_argument("--sweep-values", required=True,
                        help="Comma-separated list of values for the swept parameter")
    parser.add_argument("--sweep-param-2", default=None, choices=sorted(SWEEPABLE_PARAMS),
                        help="Optional second MonteCarloConfig field for a 2-D cross-product sweep")
    parser.add_argument("--sweep-values-2", default=None,
                        help="Comma-separated list for the second parameter; required if --sweep-param-2 is set")
    parser.add_argument("--base-run-name", required=True,
                        help="Prefix for per-run sub-directory names; sweep dir name as well")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Top-level outputs root. The sweep writes to <output-dir>/<base-run-name>/")

    # Base experiment parameters (same defaults as run_monte_carlo.py).
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--steps-per-trial", type=int, default=20)
    parser.add_argument("--prediction-horizon-steps", type=int, default=5)
    parser.add_argument("--s4-prediction-horizon-steps", type=int, default=30)
    parser.add_argument("--distance-engage-threshold", type=float, default=5.0)
    parser.add_argument("--safety-tracking-abort-threshold", type=float, default=0.3)
    parser.add_argument("--safety-tracking-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--safety-identification-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--safety-risk-score-threshold", type=float, default=0.7)
    parser.add_argument("--safety-interceptability-threshold", type=float, default=0.5)
    parser.add_argument("--safety-abort-feasibility-threshold", type=float, default=0.5)
    parser.add_argument("--abort-horizon-steps", type=int, default=10)
    parser.add_argument("--abort-risk-zone-safety-buffer", type=float, default=0.5)
    parser.add_argument("--observation-noise-sigma", type=float, default=0.2,
                        help="Default σ_obs for runs (overridden when this is the swept param)")
    parser.add_argument("--risk-k-sigma", type=float, default=1.0,
                        help="M3: weight on prediction uncertainty in effective unsafe radius")
    parser.add_argument("--risk-abort-margin", type=float, default=0.0,
                        help="M3: kinematic margin added to effective unsafe radius")
    parser.add_argument("--risk-clearance-falloff", type=float, default=2.0,
                        help="M3: clearance distance over which risk_score decays 1→0")
    parser.add_argument("--observation-velocity-noise-sigma", type=float, default=0.0)
    parser.add_argument("--late-abort-lag-steps", type=int, default=2)
    parser.add_argument("--weight-safe-capture", type=float, default=1.0)
    parser.add_argument("--weight-abort-success", type=float, default=0.5)
    parser.add_argument("--weight-unsafe", type=float, default=2.0)
    parser.add_argument("--weight-false", type=float, default=2.0)
    parser.add_argument("--weight-late-abort", type=float, default=1.0)
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--policies", type=str, default=None)

    args = parser.parse_args(argv)

    values = parse_sweep_values(args.sweep_param, args.sweep_values)
    if not values:
        print("Error: no sweep values parsed.", file=sys.stderr)
        return 1

    # 2-D sweep is optional. When both --sweep-param-2 and --sweep-values-2
    # are supplied we iterate the Cartesian product.
    values_2: Optional[List] = None
    if args.sweep_param_2 is not None:
        if args.sweep_values_2 is None:
            print("Error: --sweep-values-2 is required when --sweep-param-2 is set.", file=sys.stderr)
            return 1
        if args.sweep_param_2 == args.sweep_param:
            print("Error: --sweep-param-2 must differ from --sweep-param.", file=sys.stderr)
            return 1
        values_2 = parse_sweep_values(args.sweep_param_2, args.sweep_values_2)
        if not values_2:
            print("Error: no second-axis sweep values parsed.", file=sys.stderr)
            return 1

    base_config = build_base_config(args)
    sweep_dir = os.path.join(args.output_dir, args.base_run_name)
    os.makedirs(sweep_dir, exist_ok=True)

    if values_2 is None:
        print(f"Running 1-D sweep over '{args.sweep_param}' with {len(values)} values: {values}")
    else:
        print(f"Running 2-D sweep: '{args.sweep_param}' × '{args.sweep_param_2}' "
              f"= {len(values)} × {len(values_2)} = {len(values) * len(values_2)} cells")
    print(f"Sweep output: {sweep_dir}")

    all_rows: List[Dict[str, Any]] = []
    sub_run_paths: Dict[str, str] = {}

    if values_2 is None:
        cells = [(v,) for v in values]
    else:
        cells = [(v1, v2) for v1 in values for v2 in values_2]

    for cell in cells:
        if values_2 is None:
            (v1,) = cell
            overrides = {args.sweep_param: v1}
            suffix = f"{args.sweep_param}_{_value_to_run_suffix(v1)}"
            tag = f"{args.sweep_param}={v1}"
        else:
            v1, v2 = cell
            overrides = {args.sweep_param: v1, args.sweep_param_2: v2}
            suffix = (
                f"{args.sweep_param}_{_value_to_run_suffix(v1)}"
                f"__{args.sweep_param_2}_{_value_to_run_suffix(v2)}"
            )
            tag = f"{args.sweep_param}={v1}, {args.sweep_param_2}={v2}"

        run_name = f"{args.base_run_name}__{suffix}"
        run_dir = os.path.join(sweep_dir, run_name)
        print(f"\n[sweep] {tag} → {run_dir}")
        out = run_single(base_config, overrides, run_dir)
        all_rows.extend(out["rows"])
        sub_run_paths[run_name] = run_dir

    summary_csv = os.path.join(sweep_dir, "sweep_summary.csv")
    summary_json = os.path.join(sweep_dir, "sweep_summary.json")
    write_dict_rows_csv(all_rows, summary_csv)
    write_json(all_rows, summary_json)

    sweep_meta = {
        "sweep_param": args.sweep_param,
        "sweep_values": values,
        "sweep_param_2": args.sweep_param_2,
        "sweep_values_2": values_2,
        "base_run_name": args.base_run_name,
        "sub_runs": list(sub_run_paths.keys()),
        "num_rows": len(all_rows),
        "dimension": 2 if values_2 is not None else 1,
    }
    write_json(sweep_meta, os.path.join(sweep_dir, "sweep_meta.json"))

    print(f"\nSweep complete. Aggregated {len(all_rows)} (policy × cell) rows.")
    print(f"  sweep_summary.csv: {summary_csv}")
    print(f"  sweep_summary.json: {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
