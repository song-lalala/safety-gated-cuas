"""Journal Phase 1 — Kalman filter NIS consistency check.

Runs the Kalman estimator over the simulation scenarios and reports the mean
Normalized Innovation Squared (NIS) per scenario. For a well-tuned, consistent
filter on a constant-velocity target the mean NIS should be ≈ 2 (the
measurement dimension = chi-square dof). Maneuvering scenarios (S2) are
*expected* to exceed 2 — that is the filter correctly detecting model break,
which drives C_track = exp(-½·NIS) down toward ABORT/TRACK.

Usage:
    python experiments/check_kalman_consistency.py
    python experiments/check_kalman_consistency.py --sigma 0.2 --n-seeds 200
"""
import argparse
import random
import statistics
from typing import List

from cuas_sim.scenario import get_all_scenario_types, get_scenario_factory, ScenarioType
from cuas_sim.observation import ObservationModel
from cuas_sim.estimation import KalmanTargetEstimator
from cuas_sim.simulator import Simulator

# chi-square(2) 2.5% / 97.5% quantiles for a 95% consistency interval
CHI2_2_LO, CHI2_2_HI = 0.0506, 7.378


def _noise_seed(seed: int, scenario_idx: int) -> int:
    return seed * 1000 + scenario_idx


def collect_nis(scenario_type, q, sigma, n_seeds, steps_per_trial=20):
    """Return list of NIS values (detected, post-init steps) for one scenario."""
    factory = get_scenario_factory(scenario_type)
    scenario_idx = get_all_scenario_types().index(scenario_type)
    nis_values: List[float] = []
    for seed in range(1, n_seeds + 1):
        scenario = factory(seed=seed)
        noise_rng = random.Random(_noise_seed(seed, scenario_idx))
        obs_model = ObservationModel(noise_sigma=sigma, rng=noise_rng)
        est = KalmanTargetEstimator(
            time_step=scenario.config.time_step,
            process_noise=q,
            measurement_sigma=sigma,
        )
        sim = Simulator(
            config=scenario.config,
            target=scenario.target,
            interceptor=scenario.interceptor,
            step_hook=scenario.step_hook,
        )
        for step_index in range(steps_per_trial):
            obs = obs_model.observe(scenario, step_index)
            estimate = est.estimate(obs)
            nis = estimate.metadata.get("nis")
            # skip init step (nis=0 exactly) and dropout (is_valid False)
            if estimate.is_valid and step_index > 0 and nis is not None:
                nis_values.append(nis)
            sim.step()
    return nis_values


def summarize(nis_values: List[float]) -> dict:
    if not nis_values:
        return {"mean": float("nan"), "median": float("nan"), "in95": float("nan"), "n": 0}
    in95 = sum(1 for v in nis_values if CHI2_2_LO <= v <= CHI2_2_HI) / len(nis_values)
    return {
        "mean": statistics.mean(nis_values),
        "median": statistics.median(nis_values),
        "in95": in95,
        "n": len(nis_values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.2, help="observation noise sigma")
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--q-sweep", type=str, default="0,0.05,0.1,0.25,0.5,1.0,2.0",
                    help="comma-separated process_noise values to sweep")
    args = ap.parse_args()

    qs = [float(x) for x in args.q_sweep.split(",")]
    scenarios = get_all_scenario_types()

    print(f"NIS consistency (target mean ≈ 2.0 for CV scenarios) | sigma={args.sigma}, n_seeds={args.n_seeds}")
    print("CV scenarios: S1,S3,S4,S5,S6  |  maneuver: S2 (high NIS expected/desired)\n")

    # q-sweep on S1 (cleanest CV) to pick q
    print(f"{'q':>6} | {'S1 mean':>8} {'S1 med':>7} {'S1 in95':>7}")
    print("-" * 36)
    for q in qs:
        s = summarize(collect_nis(ScenarioType.S1_STRAIGHT_INTRUSION, q, args.sigma, args.n_seeds))
        print(f"{q:>6} | {s['mean']:>8.3f} {s['median']:>7.3f} {s['in95']:>7.2%}")

    # per-scenario detail at a chosen q (last middle value as default pick shown separately)
    print()
    for q in qs:
        pass
    # detailed per-scenario table at q = 0.1 and q = 0.5 for inspection
    for q_detail in [0.1, 0.5]:
        print(f"\nPer-scenario at q={q_detail}:")
        print(f"{'scenario':>34} | {'mean':>7} {'median':>7} {'in95':>7} {'n':>7}")
        print("-" * 72)
        for st in scenarios:
            s = summarize(collect_nis(st, q_detail, args.sigma, args.n_seeds))
            print(f"{st.value:>34} | {s['mean']:>7.3f} {s['median']:>7.3f} {s['in95']:>7.2%} {s['n']:>7}")


if __name__ == "__main__":
    main()
