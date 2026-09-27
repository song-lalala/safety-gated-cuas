"""Review-response sweeps (IEEE Access Access-2026-37078 resubmission).

Why this file exists
--------------------
`run_sensitivity.py` is the conference-era sweep tool: its ``SWEEPABLE_PARAMS``
set predates the journal work, and its argparse never forwards the journal mode
flags (``estimator_mode`` / ``r_env_mode`` / ``i_mode`` / ``a_mode`` /
``eval_risk_score_threshold``), so the *proposed* gate cannot even be expressed
there. Rather than widen a tool that 168 tests depend on, this driver builds
``MonteCarloConfig`` objects directly — the same in-process pattern
``phase4_stats.py`` already uses — and runs the grid across processes.

Design
------
* A **cell** is one ``MonteCarloConfig`` kwargs dict. Cells are cached on disk by
  a hash of their kwargs, so a re-run only computes what changed (the grids are
  large and the paired bootstrap needs the per-trial rows, not just aggregates).
* All cells share ``seeds = 1..N``, so any two cells are **paired** per
  (seed, scenario) and ``paired_bootstrap_diff`` applies directly.
* Sub-commands map 1:1 onto the review work items (W1, W2, W3, W5).

Usage
-----
    python experiments/run_review_sweeps.py w1 [--seeds 1000] [--jobs 16]
    python experiments/run_review_sweeps.py w1 --analyze-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner
from cuas_sim.statistics import (
    bootstrap_aggregate_ci,
    make_rws_aggregator,
    paired_bootstrap_diff,
    aggregate_safe_capture,
    aggregate_unsafe_per_attempt,
    aggregate_abort_success,
)

# The console on this machine is cp949; the report tables use en/em dashes and
# arrows. Reconfigure rather than ASCII-ify, so piping to a file stays faithful.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # already redirected / not a TextIO
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = os.path.join(ROOT, "outputs", "review", "cells")
RESULTS = os.path.join(ROOT, "outputs", "review")

POLICY = "SafetyGatePolicy"

# Bump when the cached payload gains a field an analysis needs; older cells are
# then recomputed instead of silently analyzed with the field missing.
CACHE_VERSION = 4

# Baseline (heuristic) gate: the submitted paper's "A". Defaults already give the
# alpha-beta estimator, linear R_env, heuristic I and the time-proxy A_abort;
# only the two non-default knobs of the conference configuration are set here.
BASE = dict(
    distance_engage_threshold=12.0,
    include_policies=[POLICY],
)

# Proposed (calibrated) gate: the submitted paper's "D".
PROPOSED = dict(
    BASE,
    eval_risk_score_threshold=0.7,
    estimator_mode="kalman",
    r_env_mode="probability",
    risk_keepout_buffer=0.6,
    i_mode="capturability",
    a_mode="dynamics",
    safety_tracking_confidence_threshold=0.05,
    safety_tracking_abort_threshold=0.01,
    safety_risk_score_threshold=0.05,
)


# --------------------------------------------------------------------------- #
# cell execution + cache
# --------------------------------------------------------------------------- #
def cell_key(kwargs: dict) -> str:
    """Stable hash of a cell's kwargs (seeds collapsed to their count+range)."""
    norm = dict(kwargs)
    seeds = norm.pop("seeds")
    norm["_seeds"] = [len(seeds), seeds[0], seeds[-1]]
    blob = json.dumps(norm, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def run_cell(kwargs: dict) -> str:
    """Run one cell unless cached; return its cache key.

    Runs in a worker process, so it returns the key (small) rather than the
    per-trial rows (large) — the parent loads rows lazily from the cache.
    """
    key = cell_key(kwargs)
    path = os.path.join(CELLS, key + ".json")
    if os.path.exists(path) and not _stale(path):
        return key
    cfg = MonteCarloConfig(**kwargs)
    result = MonteCarloRunner(cfg).run()
    trials = [t.to_dict() for t in result.per_trial_summaries() if t.policy_name == POLICY]
    # Per-scenario gate-reason counts. Needed to measure the *realized*
    # false-abort / false-track rates against their nominal 1% / 5% targets,
    # which only means anything on the constant-velocity scenarios (elsewhere a
    # low C_track is a true alarm, not a false one).
    grouped = [
        {"scenario_type": g["scenario_type"],
         "total_decisions": g["total_decisions"],
         "reason_counts": g["reason_counts"]}
        for g in result.grouped_summaries_to_dicts()
        if g["policy_name"] == POLICY
    ]
    payload = {
        "v": CACHE_VERSION,
        "config": {k: v for k, v in kwargs.items() if k != "seeds"},
        "n_seeds": len(kwargs["seeds"]),
        "trials": trials,
        "grouped": grouped,
        "per_trial_reasons": _per_trial_reasons(result),
        # Only populated when the cell enables the W6 validation layer, so this
        # does not invalidate cells written for the other work items.
        "pn_records": [
            {"scenario": r.metadata.get("scenario_type"), **r.metadata["pn"]}
            for r in result.records
            if r.policy_name == POLICY and "pn" in r.metadata
        ],
    }
    os.makedirs(CELLS, exist_ok=True)
    # Unique per writer: on Windows os.replace fails if another process holds the
    # source, so a shared .tmp name turns any duplicate cell into a hard error.
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        # default=str: the config echo can hold ScenarioType enums, which have no
        # JSON form. cell_key already hashes them this way, so the two agree.
        json.dump(payload, f, default=str)
    os.replace(tmp, path)  # atomic: a killed run never leaves a half-written cell
    return key


#: gate reasons whose *per-trial* incidence the review needs (Reviewer 3.3).
TRACKED_REASONS = ("tracking_confidence_too_low_abort", "tracking_confidence_too_low_track")


def _per_trial_reasons(result) -> dict:
    """Per scenario: how many trials saw each tracked reason at least once, and
    at which step it first fired.

    A per-step rate says nothing about what an operator experiences over a
    K=20-step engagement; that needs trial-level incidence. The first-fire steps
    additionally give the detection latency on the maneuver scenarios.
    """
    seen: dict = {}
    for rec in result.records:
        if rec.policy_name != POLICY or rec.reason not in TRACKED_REASONS:
            continue
        scn = rec.metadata.get("scenario_type")
        key = rec.metadata.get("seed")
        first = seen.setdefault(scn, {}).setdefault(rec.reason, {})
        if key not in first or rec.step_index < first[key]:
            first[key] = rec.step_index
    trials_per_scn: dict = {}
    for rec in result.records:
        if rec.policy_name != POLICY:
            continue
        scn = rec.metadata.get("scenario_type")
        trials_per_scn.setdefault(scn, set()).add(rec.metadata.get("seed"))
    out = {}
    for scn in trials_per_scn:
        per_reason = seen.get(scn, {})
        # Trials where EITHER reason fired, counted once. Deriving this from the
        # step lists would be wrong: two trials that first alarm at the same step
        # are two trials, and a trial can appear in both reason lists.
        alarmed = set()
        for r in TRACKED_REASONS:
            alarmed |= set(per_reason.get(r, {}))
        out[scn] = {
            "n_trials": len(trials_per_scn[scn]),
            "alarmed_trials": len(alarmed),
            **{r: sorted(per_reason.get(r, {}).values()) for r in TRACKED_REASONS},
        }
    return out


def _stale(path: str) -> bool:
    """True if a cached cell predates the current payload schema."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("v", 0) < CACHE_VERSION
    except (json.JSONDecodeError, OSError):
        return True


def load_cell(key: str) -> dict:
    """Load a cell's per-trial rows, keyed by (seed, scenario) for pairing."""
    with open(os.path.join(CELLS, key + ".json"), encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for row in payload["trials"]:
        out[(row["seed"], row["scenario_type"])] = SimpleNamespace(**row)
    return out


def load_reasons(key: str) -> dict:
    """Per-scenario {reason: count} plus the scenario's total decision count."""
    with open(os.path.join(CELLS, key + ".json"), encoding="utf-8") as f:
        payload = json.load(f)
    return {g["scenario_type"]: g for g in payload.get("grouped", [])}


def load_per_trial_reasons(key: str) -> dict:
    with open(os.path.join(CELLS, key + ".json"), encoding="utf-8") as f:
        return json.load(f).get("per_trial_reasons", {})


def aligned(a: dict, b: dict):
    """Pair two cells on their shared (seed, scenario) keys."""
    keys = sorted(set(a) & set(b), key=lambda k: (k[0], str(k[1])))
    return [a[k] for k in keys], [b[k] for k in keys]


def _needs_run(spec: dict) -> bool:
    path = os.path.join(CELLS, cell_key(spec) + ".json")
    return not os.path.exists(path) or _stale(path)


def run_grid(cells: list[dict], jobs: int, label: str) -> list[str]:
    """Execute a list of cells in parallel, returning their keys in order.

    Grids legitimately overlap — the matched-sigma reference of W3 coincides with
    a point of its own mismatch grid, for instance — so identical specs are
    collapsed before dispatch. Running one cell in two workers wastes a slot and,
    on Windows, makes the two writers collide over the same output file.
    """
    todo, seen = [], set()
    for c in cells:
        k = cell_key(c)
        if k not in seen and _needs_run(c):
            seen.add(k)
            todo.append(c)
    print(f"[{label}] {len(cells)} cells ({len(todo)} to compute, "
          f"{len(cells) - len(todo)} cached) on {jobs} workers")
    if todo:
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for _ in pool.map(run_cell, todo):
                done += 1
                if done % 10 == 0 or done == len(todo):
                    rate = done / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - done) / max(rate, 1e-9)
                    print(f"  {done}/{len(todo)}  ({rate*60:.1f} cells/min, ETA {eta/60:.1f} min)",
                          flush=True)
        print(f"[{label}] computed in {(time.time() - t0)/60:.1f} min")
    return [cell_key(c) for c in cells]


def write_csv(name: str, header: str, rows: list[str]) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"  -> {path}")
    return path


# --------------------------------------------------------------------------- #
# W1 — fair baseline (Reviewer 3, comment 2)
# --------------------------------------------------------------------------- #
# The heuristic gate is re-evaluated over its OWN threshold grid at every noise
# level, so it can be reported at its per-sigma optimum ("oracle") instead of at
# the conference defaults. theta_track is extended below the conference grid's
# 0.3 floor: the reviewer's cited optimum sat on that boundary, and a boundary
# optimum invites the same objection a second time.
W1_SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
# 0.0 is included deliberately: C_track < 0 can never hold, so it is the exact
# "branch switched off" limit. Without it the optimum kept landing on the grid
# floor, which would have invited the same under-tuning objection a second time.
W1_ABORT = [0.0, 0.05, 0.1, 0.2, 0.3]                           # conference: 0.3
W1_TRACK = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # conference: 0.6
# Does the *proposed* gate also gain from per-sigma tuning? If it does not, the
# "no tuning required" claim is established rather than asserted.
W1_PROP_TRACK = [0.01, 0.02, 0.05, 0.1, 0.2]
# Stage 2: having fixed the best (theta_abort, theta_track) per sigma, also let
# the heuristic tune its risk threshold there. Without this the oracle could
# still be called under-tuned; the full 3-D cross product would cost 6x more
# cells for a knob that only S4 exercises.
W1_RISK = [0.1, 0.3, 0.5, 0.7, 0.9]


def heuristic_cell(seeds, sigma, abort_th, track_th, risk_th=0.7) -> dict:
    return dict(BASE, seeds=seeds, observation_noise_sigma=sigma,
                eval_risk_score_threshold=0.7,        # yardstick pinned, always
                safety_tracking_abort_threshold=abort_th,
                safety_tracking_confidence_threshold=track_th,
                safety_risk_score_threshold=risk_th)


def w1_cells(seeds: list[int]) -> dict:
    heur, prop, prop_tuned = [], [], []
    for s in W1_SIGMAS:
        for a in W1_ABORT:
            for t in W1_TRACK:
                heur.append(heuristic_cell(seeds, s, a, t))
        prop.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=s))
        for t in W1_PROP_TRACK:
            prop_tuned.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=s,
                                   safety_tracking_confidence_threshold=t))
    return {"heuristic": heur, "proposed": prop, "proposed_tuned": prop_tuned}


def w1_stage2_cells(seeds: list[int], best: dict) -> list[dict]:
    return [heuristic_cell(seeds, s, a, t, r)
            for s, (a, t) in best.items() for r in W1_RISK]


def w1_analyze(seeds: list[int], jobs: int = 4) -> None:
    rws = make_rws_aggregator()
    groups = w1_cells(seeds)

    # --- per-cell scores -----------------------------------------------------
    heur_score = {}   # (sigma, theta_abort, theta_track) -> (rws, key)
    for spec in groups["heuristic"]:
        k = cell_key(spec)
        trials = list(load_cell(k).values())
        heur_score[(spec["observation_noise_sigma"],
                    spec["safety_tracking_abort_threshold"],
                    spec["safety_tracking_confidence_threshold"])] = (rws(trials), k)

    # --- stage 2: refine the oracle's risk threshold at its best (a, t) -------
    stage1_best = {}
    for s in W1_SIGMAS:
        (a, t), _ = max(((k[1:], v) for k, v in heur_score.items() if k[0] == s),
                        key=lambda kv: kv[1][0])
        stage1_best[s] = (a, t)
    run_grid(w1_stage2_cells(seeds, stage1_best), jobs, "w1:oracle_risk")

    oracle = {}       # sigma -> (rws, key, theta_abort, theta_track, theta_risk)
    for s, (a, t) in stage1_best.items():
        best = None
        for r in W1_RISK:
            k = cell_key(heuristic_cell(seeds, s, a, t, r))
            v = rws(list(load_cell(k).values()))
            if best is None or v > best[0]:
                best = (v, k, a, t, r)
        oracle[s] = best

    prop_score = {}
    for spec in groups["proposed"]:
        k = cell_key(spec)
        prop_score[spec["observation_noise_sigma"]] = (rws(list(load_cell(k).values())), k)

    prop_tuned_best = {}
    for spec in groups["proposed_tuned"]:
        s = spec["observation_noise_sigma"]
        v = rws(list(load_cell(cell_key(spec)).values()))
        t = spec["safety_tracking_confidence_threshold"]
        if s not in prop_tuned_best or v > prop_tuned_best[s][0]:
            prop_tuned_best[s] = (v, t)

    # --- full heuristic grid (for the paper's appendix / supplement) ----------
    rows = [f"{s},{a},{t},{v:.4f}" for (s, a, t), (v, _) in sorted(heur_score.items())]
    write_csv("w1_heuristic_grid.csv",
              "sigma,theta_abort,theta_track,rws", rows)

    # --- oracle vs default vs proposed ---------------------------------------
    print("\n" + "=" * 100)
    print("W1 — FAIR BASELINE (Reviewer 3.2): heuristic at its per-sigma optimum")
    print("=" * 100)
    print(f"{'sigma':>6} | {'heur@default':>13} | {'heur@oracle':>12} "
          f"{'(a*, t*, r*)':>18} | {'proposed':>9} | {'D - oracle [95% CI]':>28} | {'p':>6}")
    print("-" * 105)

    out = ["sigma,heur_default,heur_oracle,oracle_theta_abort,oracle_theta_track,"
           "oracle_theta_risk,proposed,proposed_tuned_best,proposed_tuned_theta,"
           "diff_vs_default,diff_vs_oracle,ci_lo,ci_hi,p_value,oracle_track_branch_dead"]
    for s in W1_SIGMAS:
        default = heur_score[(s, 0.3, 0.6)][0]
        best_v, best_key, best_a, best_t, best_r = oracle[s]
        prop_v, prop_key = prop_score[s]

        lo_t, hi_t = aligned(load_cell(best_key), load_cell(prop_key))
        diff = paired_bootstrap_diff(hi_t, lo_t, rws, seed=1)

        # The gate tests ABORT(C_track < theta_abort) before TRACK(C_track <
        # theta_track): at theta_track <= theta_abort the TRACK branch can never
        # fire, so an optimum there means the heuristic scores best by switching
        # its own tracking-confidence signal off.
        dead = best_t <= best_a
        pt_v, pt_t = prop_tuned_best[s]

        print(f"{s:>6} | {default:>13.4f} | {best_v:>12.4f} "
              f"{str((best_a, best_t, best_r)):>18} | "
              f"{prop_v:>9.4f} | {diff['point_diff']:>+9.4f} "
              f"[{diff['ci_lo']:+.4f},{diff['ci_hi']:+.4f}] | {diff['p_value_one_sided']:>6.3f}"
              + ("   <- C_track branches off" if dead else ""))
        out.append(f"{s},{default:.4f},{best_v:.4f},{best_a},{best_t},{best_r},"
                   f"{prop_v:.4f},{pt_v:.4f},{pt_t},"
                   f"{prop_v - default:.4f},{diff['point_diff']:.4f},"
                   f"{diff['ci_lo']:.4f},{diff['ci_hi']:.4f},"
                   f"{diff['p_value_one_sided']:.4f},{int(dead)}")
    write_csv("w1_fair_baseline.csv", out[0], out[1:])

    # --- is the oracle real, or just the edge of the grid? -------------------
    # theta = 0 is not a grid artefact: C_track < 0 can never hold, so it is the
    # exact "branch disabled" limit and nothing lies beyond it. Landing there is
    # a finding, not a symptom of too narrow a sweep.
    print("\nOptimum placement (theta=0 is the definitional branch-off limit, not a grid edge):")
    for s in W1_SIGMAS:
        _, _, a, t, r = oracle[s]
        flags = []
        if t == 0.0 and a == 0.0:
            flags.append("both C_track branches disabled")
        elif t == max(W1_TRACK) or r in (min(W1_RISK), max(W1_RISK)):
            flags.append("EXTEND: optimum on a real grid edge")
        else:
            flags.append("interior")
        print(f"  sigma={s:<5} optimum=(abort {a}, track {t}, risk {r})  -> {'; '.join(flags)}")

    # --- does the proposed gate gain from tuning? ----------------------------
    print("\nProposed gate under per-sigma tuning (small gain => 'no tuning required'):")
    for s in W1_SIGMAS:
        base_v = prop_score[s][0]
        tuned_v, tuned_t = prop_tuned_best[s]
        print(f"  sigma={s:<5} single-setting(0.05)={base_v:.4f}  "
              f"best-tuned={tuned_v:.4f} (theta={tuned_t})  gain={tuned_v - base_v:+.4f}")


# --------------------------------------------------------------------------- #
# W5 — quantities the paper never reported (Reviewer 3, comment 4)
# --------------------------------------------------------------------------- #
def heuristic_default(seeds: list[int], sigma: float) -> dict:
    """The submitted paper's baseline "A": the conference thresholds, spelled out
    so this cell hashes identically to the matching W1 grid point."""
    return heuristic_cell(seeds, sigma, 0.3, 0.6, 0.7)


def w5_analyze(seeds: list[int]) -> None:
    print("\n" + "=" * 100)
    print("W5 — MISSING QUANTITIES (Reviewer 3.4): SafeCapture, missed-safe, absolute counts")
    print("=" * 100)
    print(f"{'sigma':>6} | {'policy':>9} | {'RWS':>7} | {'SafeCap':>8} | {'MissedSafe':>10} | "
          f"{'unsafe n/att':>16} | {'unsafe rate':>11} | {'abort succ':>10}")
    print("-" * 100)
    rws = make_rws_aggregator()
    rows = ["sigma,policy,rws,safe_capture,missed_safe_rate,safe_opportunity_n,"
            "unsafe_n,engage_attempts_n,unsafe_per_attempt,abort_success"]
    for s in W1_SIGMAS:
        for tag, spec in (("heuristic", heuristic_default(seeds, s)),
                          ("proposed", dict(PROPOSED, seeds=seeds, observation_noise_sigma=s))):
            trials = list(load_cell(cell_key(spec)).values())
            sc = aggregate_safe_capture(trials)
            so_n = sum(t.safe_opportunity_count for t in trials)
            missed_n = sum(t.missed_safe_opportunity_count for t in trials)
            unsafe_n = sum(t.unsafe_engagement_count for t in trials)
            eng_n = sum(t.engage_count for t in trials)
            print(f"{s:>6} | {tag:>9} | {rws(trials):>7.4f} | {sc:>8.4f} | "
                  f"{(missed_n / so_n if so_n else 0):>10.4f} | "
                  f"{unsafe_n:>6} / {eng_n:<7} | "
                  f"{aggregate_unsafe_per_attempt(trials):>11.5f} | "
                  f"{aggregate_abort_success(trials):>10.3f}")
            rows.append(f"{s},{tag},{rws(trials):.4f},{sc:.4f},"
                        f"{(missed_n / so_n if so_n else 0):.4f},{so_n},"
                        f"{unsafe_n},{eng_n},{aggregate_unsafe_per_attempt(trials):.6f},"
                        f"{aggregate_abort_success(trials):.4f}")
    write_csv("w5_missing_quantities.csv", rows[0], rows[1:])


def w5_cells(seeds: list[int]) -> dict:
    cells = []
    for s in W1_SIGMAS:
        cells.append(heuristic_default(seeds, s))
        cells.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=s))
    return {"w5": cells}


# --------------------------------------------------------------------------- #
# W2 — sensitivity (Reviewer 1.2, Reviewer 2.4, Reviewer 2.12)
# --------------------------------------------------------------------------- #
# The charge to answer is "parameter-reduced rather than parameter-relocated".
# The answer is not the *count* of constants but the *shape* of the response
# curve: a calibrated threshold should hold its score over a wide band, while
# the heuristic's optimum moves with the noise level (see W1).
W2_SIGMAS = [0.1, 0.2, 0.5]
W2_GRID = {
    "kalman_process_noise": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    "safety_tracking_confidence_threshold": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
    "safety_tracking_abort_threshold": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    "safety_risk_score_threshold": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
}
W2_HORIZONS = [1, 2, 3, 5, 10, 20, 30]


def w2_cells(seeds: list[int]) -> dict:
    cells = []
    for s in W2_SIGMAS:
        for param, values in W2_GRID.items():
            for v in values:
                cells.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=s, **{param: v}))
    for h in W2_HORIZONS:
        # Both horizons move together. S4 is the only core scenario with a risk
        # zone and it overrides the general horizon with its own (30 by default),
        # so sweeping prediction_horizon_steps alone leaves R_env untouched and
        # the sweep measures nothing.
        cells.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=0.2,
                          prediction_horizon_steps=h, s4_prediction_horizon_steps=h))
    return {"w2": cells}


def plateau(values: list[float], scores: list[float], tol: float = 0.05):
    """Contiguous band around the best score that stays within `tol` of it.

    Reported as (lo, hi, fold) where fold = hi/lo — how many-fold the parameter
    can move without costing more than `tol` RWS. A wide plateau is what makes a
    parameter a *stated* quantity rather than a tuned one.
    """
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    lo = hi = best_i
    while lo - 1 >= 0 and scores[lo - 1] >= scores[best_i] - tol:
        lo -= 1
    while hi + 1 < len(scores) and scores[hi + 1] >= scores[best_i] - tol:
        hi += 1
    fold = values[hi] / values[lo] if values[lo] > 0 else float("inf")
    return values[lo], values[hi], fold, values[best_i]


def w2_analyze(seeds: list[int]) -> None:
    rws = make_rws_aggregator()
    print("\n" + "=" * 100)
    print("W2 — SENSITIVITY (R1.2 / R2.4 / R2.12): is the framework parameter-reduced or -relocated?")
    print("=" * 100)

    rows = ["sigma,parameter,value,rws"]
    plateau_rows = ["sigma,parameter,plateau_lo,plateau_hi,fold,argmax,best_rws,default_rws"]
    defaults = {"kalman_process_noise": 0.1,
                "safety_tracking_confidence_threshold": 0.05,
                "safety_tracking_abort_threshold": 0.01,
                "safety_risk_score_threshold": 0.05}

    for param, values in W2_GRID.items():
        print(f"\n--- {param} (proposed gate; default {defaults[param]}) ---")
        header = "  sigma |" + "".join(f"{v:>8}" for v in values) + " |  plateau (within 0.05 RWS)"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for s in W2_SIGMAS:
            scores = []
            for v in values:
                spec = dict(PROPOSED, seeds=seeds, observation_noise_sigma=s, **{param: v})
                scores.append(rws(list(load_cell(cell_key(spec)).values())))
                rows.append(f"{s},{param},{v},{scores[-1]:.4f}")
            lo, hi, fold, arg = plateau(values, scores)
            d_rws = scores[values.index(defaults[param])]
            print(f"  {s:>5} |" + "".join(f"{x:>8.3f}" for x in scores)
                  + f" |  [{lo}, {hi}]  {fold:.0f}x  argmax={arg}")
            plateau_rows.append(f"{s},{param},{lo},{hi},{fold:.1f},{arg},{max(scores):.4f},{d_rws:.4f}")

    # H sweep (R2.12): R_env is a max over the horizon, so the response should
    # rise then saturate rather than peak.
    print(f"\n--- prediction_horizon_steps at sigma=0.2 (default 5) ---")
    h_scores = []
    for h in W2_HORIZONS:
        spec = dict(PROPOSED, seeds=seeds, observation_noise_sigma=0.2,
                    prediction_horizon_steps=h, s4_prediction_horizon_steps=h)
        h_scores.append(rws(list(load_cell(cell_key(spec)).values())))
        rows.append(f"0.2,prediction_horizon_steps,{h},{h_scores[-1]:.4f}")
    print("  " + "".join(f"{h:>8}" for h in W2_HORIZONS))
    print("  " + "".join(f"{x:>8.3f}" for x in h_scores))

    # q decomposed by scenario (R2.4 asks specifically about maneuver conditions)
    print("\n--- kalman_process_noise decomposed by scenario at sigma=0.2 (R2.4) ---")
    q_values = W2_GRID["kalman_process_noise"]
    by_scn = {}
    for q in q_values:
        spec = dict(PROPOSED, seeds=seeds, observation_noise_sigma=0.2, kalman_process_noise=q)
        cell = load_cell(cell_key(spec))
        for (_, scn), t in cell.items():
            by_scn.setdefault(scn, {}).setdefault(q, []).append(t)
    print("  " + f"{'scenario':>34}" + "".join(f"{q:>8}" for q in q_values))
    for scn in sorted(by_scn):
        line = "".join(f"{rws(by_scn[scn][q]):>8.3f}" for q in q_values)
        print(f"  {scn:>34}" + line)
        for q in q_values:
            rows.append(f"0.2,q_by_scenario:{scn},{q},{rws(by_scn[scn][q]):.4f}")

    # The comparison that actually answers the charge. A plateau width measured
    # around each gate's own optimum is misleading here: the heuristic's optimum
    # is the branch-off limit, where it is flat precisely because the signal is
    # inert. What matters operationally is the cost of committing to ONE setting
    # when sigma is unknown -- i.e. how much score a fixed threshold gives up
    # against the best achievable at each sigma.
    print("\n--- Cost of one fixed threshold when sigma is unknown ---")
    print("    (max over sigma of [best RWS at that sigma] - [RWS at the fixed setting])")
    p_vals = W2_GRID["safety_tracking_confidence_threshold"]
    h_vals = W1_TRACK
    print(f"\n  {'sigma':>6} | {'proposed @0.05':>14} {'prop best':>10} {'regret':>8}"
          f" | {'heuristic @0.6':>14} {'heur best':>10} {'regret':>8}")
    print("  " + "-" * 82)
    p_worst = h_worst = 0.0
    for s in W2_SIGMAS:
        p_scores = [rws(list(load_cell(cell_key(dict(
            PROPOSED, seeds=seeds, observation_noise_sigma=s,
            safety_tracking_confidence_threshold=v))).values())) for v in p_vals]
        h_scores = [rws(list(load_cell(cell_key(
            heuristic_cell(seeds, s, 0.3, v))).values())) for v in h_vals]
        p_fix, p_best = p_scores[p_vals.index(0.05)], max(p_scores)
        h_fix, h_best = h_scores[h_vals.index(0.6)], max(h_scores)
        p_worst = max(p_worst, p_best - p_fix)
        h_worst = max(h_worst, h_best - h_fix)
        print(f"  {s:>6} | {p_fix:>14.4f} {p_best:>10.4f} {p_best - p_fix:>8.4f}"
              f" | {h_fix:>14.4f} {h_best:>10.4f} {h_best - h_fix:>8.4f}")
        plateau_rows.append(f"{s},REGRET_proposed,,,,{p_vals[p_scores.index(p_best)]},"
                            f"{p_best:.4f},{p_fix:.4f}")
        plateau_rows.append(f"{s},REGRET_heuristic,,,,{h_vals[h_scores.index(h_best)]},"
                            f"{h_best:.4f},{h_fix:.4f}")
    print(f"\n  worst-case regret of a single fixed setting: "
          f"proposed {p_worst:.4f}  vs  heuristic {h_worst:.4f}"
          f"   ({h_worst / p_worst:.1f}x)" if p_worst > 0 else "")

    write_csv("w2_sensitivity.csv", rows[0], rows[1:])
    write_csv("w2_plateaus.csv", plateau_rows[0], plateau_rows[1:])


# --------------------------------------------------------------------------- #
# W3 — noise-model mismatch (Reviewer 3.1, Reviewer 1.4, Reviewer 2.11)
# --------------------------------------------------------------------------- #
# The submitted sweep handed the filter the true sigma at every point, so the
# noise-robustness claim was only ever tested with the noise model correct.
# Here the filter's assumed sigma is pinned while the world's true sigma varies.
W3_TRUE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
W3_ASSUMED = [0.05, 0.1, 0.2, 0.5]

# The nominal 1% / 5% rates are only meaningful where the filter is actually
# consistent; on S2 (maneuver) and S5 (dropout) a low C_track is a TRUE alarm.
# check_kalman_consistency.py puts S1/S3/S4/S6 at mean NIS 1.87-1.90 with ~95.3%
# inside the nominal interval, so those four are the clean denominator.
CV_SCENARIOS = ("S1_STRAIGHT_INTRUSION", "S3_LOW_IDENTIFICATION_CONFIDENCE",
                "S4_RISK_ZONE_PROXIMITY", "S6_NON_INTERCEPTABLE_TARGET")

# The gate is a first-match cascade, so each test only sees what the earlier
# tests passed through. Realized rates must divide by that, not by all steps.
BEFORE_ABORT_TEST = ("not_detected", "invalid_estimate", "invalid_prediction")
BEFORE_TRACK_TEST = BEFORE_ABORT_TEST + ("tracking_confidence_too_low_abort",
                                         "high_environmental_risk",
                                         "abort_feasibility_too_low")


def realized_alarm_rates(key: str) -> tuple[float, float]:
    """(false-abort, false-track) rate over the consistent-filter scenarios."""
    reasons = load_reasons(key)
    ab_num = ab_den = tr_num = tr_den = 0
    for scn in CV_SCENARIOS:
        g = reasons.get(scn)
        if not g:
            continue
        rc, total = g["reason_counts"], g["total_decisions"]
        ab_den += total - sum(rc.get(r, 0) for r in BEFORE_ABORT_TEST)
        tr_den += total - sum(rc.get(r, 0) for r in BEFORE_TRACK_TEST)
        ab_num += rc.get("tracking_confidence_too_low_abort", 0)
        tr_num += rc.get("tracking_confidence_too_low_track", 0)
    return (ab_num / ab_den if ab_den else 0.0,
            tr_num / tr_den if tr_den else 0.0)


# Non-Gaussian sensor error (Reviewer 2.11). Measured at the nominal operating
# point with the filter's assumed sigma matched, so the only thing changing is
# the *shape* of the error, not its scale. The chi-square measurement gate is
# paired with each case: Gaussian noise gives it nothing to do (W4 showed it
# inert), so heavy tails and clutter are where a bounded variant has to earn its
# place (Reviewer 1.4).
W3B_SIGMA = 0.2
W3B_CASES = [
    ("gaussian (submitted)", {}),
    ("Student-t (nu=3)", {"observation_noise_model": "student_t", "observation_noise_dof": 3.0}),
    ("bias +0.3", {"observation_bias": 0.3}),
    ("bias +0.6", {"observation_bias": 0.6}),
    ("clutter 5% @5sigma", {"observation_outlier_rate": 0.05}),
    ("clutter 1% @5sigma", {"observation_outlier_rate": 0.01}),
]


def w3_cells(seeds: list[int]) -> dict:
    cells = []
    for st in W3_TRUE:
        for sa in W3_ASSUMED:
            cells.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=st,
                              kalman_measurement_sigma=sa))
        # matched reference, written explicitly so it hashes like the grid cells
        cells.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=st,
                          kalman_measurement_sigma=max(st, 1e-2)))
    for _, extra in W3B_CASES:
        for gate in (None, 9.21):
            cells.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=W3B_SIGMA,
                              kalman_measurement_sigma=W3B_SIGMA,
                              nis_gate_threshold=gate, **extra))
    return {"w3": cells}


def w3_analyze(seeds: list[int]) -> None:
    rws = make_rws_aggregator()
    print("\n" + "=" * 100)
    print("W3 — NOISE-MODEL MISMATCH (R3.1): filter's assumed sigma pinned, true sigma varied")
    print("=" * 100)

    print("\nRWS  (rows = assumed sigma, cols = true sigma; 'matched' = assumed follows true)")
    print(f"  {'assumed':>9} |" + "".join(f"{t:>9}" for t in W3_TRUE))
    print("  " + "-" * (11 + 9 * len(W3_TRUE)))
    rows = ["assumed_sigma,true_sigma,rws,unsafe_per_attempt,abort_success,"
            "false_abort_realized,false_track_realized"]

    def record(sa_label, sa_value, st):
        k = cell_key(dict(PROPOSED, seeds=seeds, observation_noise_sigma=st,
                          kalman_measurement_sigma=sa_value))
        trials = list(load_cell(k).values())
        v = rws(trials)
        fa, ft = realized_alarm_rates(k)
        rows.append(f"{sa_label},{st},{v:.4f},{aggregate_unsafe_per_attempt(trials):.6f},"
                    f"{aggregate_abort_success(trials):.4f},{fa:.5f},{ft:.5f}")
        return v

    for sa in W3_ASSUMED:
        print(f"  {sa:>9} |" + "".join(f"{record(sa, sa, st):>9.3f}" for st in W3_TRUE))
    print(f"  {'matched':>9} |"
          + "".join(f"{record('matched', max(st, 1e-2), st):>9.3f}" for st in W3_TRUE))

    # RWS alone makes over-assuming sigma look free (it suppresses false aborts),
    # so the safety columns have to be read next to it: a filter that never
    # alarms has not become safe, it has gone blind.
    print("\nSafety at the extremes (RWS alone is misleading here)")
    print(f"  {'assumed':>9} | {'true':>5} | {'RWS':>7} | {'unsafe/att':>10} | {'abort succ':>10} |"
          f" {'false-abort':>11}")
    print("  " + "-" * 70)
    for sa_label, sa_value in (("0.05", 0.05), ("0.5", 0.5), ("matched", None)):
        for st in (0.2, 0.5):
            av = max(st, 1e-2) if sa_value is None else sa_value
            k = cell_key(dict(PROPOSED, seeds=seeds, observation_noise_sigma=st,
                              kalman_measurement_sigma=av))
            trials = list(load_cell(k).values())
            fa, _ = realized_alarm_rates(k)
            print(f"  {sa_label:>9} | {st:>5} | {rws(trials):>7.3f} | "
                  f"{aggregate_unsafe_per_attempt(trials):>10.5f} | "
                  f"{aggregate_abort_success(trials):>10.3f} | {fa*100:>10.2f}%")

    print("\nRealized false-alarm rates vs nominal (consistent-filter scenarios only:")
    print(f"  {', '.join(CV_SCENARIOS)})")
    print(f"  nominal: false-abort 1.0%, false-track 5.0%  [per step]")
    print(f"\n  {'assumed':>9} | " + "".join(f"{('true=' + str(t)):>17}" for t in W3_TRUE))
    print("  " + "-" * (12 + 17 * len(W3_TRUE)))
    for sa in list(W3_ASSUMED) + ["matched"]:
        line = ""
        for st in W3_TRUE:
            asum = max(st, 1e-2) if sa == "matched" else sa
            k = cell_key(dict(PROPOSED, seeds=seeds, observation_noise_sigma=st,
                              kalman_measurement_sigma=asum))
            fa, ft = realized_alarm_rates(k)
            line += f"  {fa*100:>6.2f}% /{ft*100:>6.2f}%"
        print(f"  {str(sa):>9} |" + line)
    print("\n  (each cell: realized false-abort% / false-track%)")
    write_csv("w3_mismatch.csv", rows[0], rows[1:])

    # --- non-Gaussian sensor error (Reviewer 2.11) ---------------------------
    print(f"\nNon-Gaussian sensor error at sigma={W3B_SIGMA}, filter matched")
    print("  (dropout is already exercised by S5 and is not repeated here)")
    print(f"\n  {'sensor error':>22} {'chi2 gate':>10} | {'RWS':>7} | {'unsafe/att':>10} |"
          f" {'false-abort':>11} | {'false-track':>11}")
    print("  " + "-" * 82)
    brows = ["case,chi2_gate,rws,unsafe_per_attempt,abort_success,false_abort,false_track"]
    for label, extra in W3B_CASES:
        for gate in (None, 9.21):
            k = cell_key(dict(PROPOSED, seeds=seeds, observation_noise_sigma=W3B_SIGMA,
                              kalman_measurement_sigma=W3B_SIGMA,
                              nis_gate_threshold=gate, **extra))
            trials = list(load_cell(k).values())
            fa, ft = realized_alarm_rates(k)
            print(f"  {label:>22} {('on' if gate else 'off'):>10} | {rws(trials):>7.3f} | "
                  f"{aggregate_unsafe_per_attempt(trials):>10.5f} | "
                  f"{fa*100:>10.2f}% | {ft*100:>10.2f}%")
            brows.append(f"{label},{'on' if gate else 'off'},{rws(trials):.4f},"
                         f"{aggregate_unsafe_per_attempt(trials):.6f},"
                         f"{aggregate_abort_success(trials):.4f},{fa:.5f},{ft:.5f}")
    write_csv("w3b_nongaussian.csv", brows[0], brows[1:])


# --------------------------------------------------------------------------- #
# W4 — per-step vs per-trial alarms, robust C_track variants (Reviewer 3.3/3.4, 1.4)
# --------------------------------------------------------------------------- #
# Three things the submitted paper left open:
#   (a) the 1% / 5% targets are PER STEP; over K=20 steps that is 18.2% / 64.2%
#       per trial under independence, which the paper never stated;
#   (b) ABORT is not absorbing in the simulation, which is why the per-trial
#       figure never showed up in the scores — a fielded gate would latch;
#   (c) Reviewer 1 asked for a bounded/robust variant of the confidence.
# The time-averaged NIS (chi-square(2W)) and the chi-square measurement gate are
# the standard answers to (c) and they also address (a).
W4_SIGMAS = [0.2, 0.5]
W4_VARIANTS = [
    ("W=1 (submitted)", {}),
    ("W=3", {"ctrack_window": 3}),
    ("W=5", {"ctrack_window": 5}),
    ("W=10", {"ctrack_window": 10}),
    ("W=1 + chi2 gate", {"nis_gate_threshold": 9.21}),
    ("W=5 + chi2 gate", {"ctrack_window": 5, "nis_gate_threshold": 9.21}),
]
K_STEPS = 20  # steps per trial, for the per-trial conversion


def w4_cells(seeds: list[int]) -> dict:
    cells = []
    for s in W4_SIGMAS:
        for _, extra in W4_VARIANTS:
            for latch in (False, True):
                cells.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=s,
                                  abort_latching=latch, **extra))
    return {"w4": cells}


def per_trial_alarm_rates(key: str, scenarios=None) -> tuple[float, float]:
    """Fraction of trials in which each tracked reason fired at least once."""
    ptr = load_per_trial_reasons(key)
    names = scenarios if scenarios is not None else tuple(ptr)
    n = ab = tr = 0
    for scn in names:
        g = ptr.get(scn)
        if not g:
            continue
        n += g["n_trials"]
        ab += len(g.get("tracking_confidence_too_low_abort", []))
        tr += len(g.get("tracking_confidence_too_low_track", []))
    return (ab / n if n else 0.0, tr / n if n else 0.0)


def detection_stats(key: str, scenario: str) -> tuple[float, float]:
    """(fraction of trials ever alarmed, median first-alarm step) on `scenario`.

    On the maneuver scenario a low C_track is a TRUE alarm, so this is detection
    power and latency — the other side of the false-alarm trade a longer window
    buys.
    """
    g = load_per_trial_reasons(key).get(scenario)
    if not g or not g["n_trials"]:
        return 0.0, float("nan")
    fired = sorted(g.get("tracking_confidence_too_low_abort", [])
                   + g.get("tracking_confidence_too_low_track", []))
    if not fired:
        return 0.0, float("nan")
    return g["alarmed_trials"] / g["n_trials"], fired[len(fired) // 2]


def w4_analyze(seeds: list[int]) -> None:
    rws = make_rws_aggregator()
    print("\n" + "=" * 112)
    print("W4 — PER-STEP vs PER-TRIAL ALARMS and robust C_track variants (R3.3 / R3.4 / R1.4)")
    print("=" * 112)
    print(f"  nominal per step: false-abort 1.0%, false-track 5.0%")
    print(f"  independent-steps prediction over K={K_STEPS}: "
          f"false-abort {100*(1-0.99**K_STEPS):.1f}%, false-track {100*(1-0.95**K_STEPS):.1f}%")
    print(f"  (measured on the consistent-filter scenarios: {', '.join(CV_SCENARIOS)})")
    print("  Reading the two latch rows: without latching the per-trial figure is the true")
    print("  alarm incidence; with latching the trial stops deciding at its first ABORT, so")
    print("  the false-TRACK column falls for a bookkeeping reason, not a safety one. The")
    print("  operationally meaningful pair is (latch=False false-abort) -> what fires, and")
    print("  (latch=True RWS) -> what it costs once a break-off cannot be taken back.")

    rows = ["sigma,variant,abort_latching,rws,per_step_false_abort,per_step_false_track,"
            "per_trial_false_abort,per_trial_false_track,s2_detect_frac,s2_median_step"]
    for s in W4_SIGMAS:
        print(f"\n--- sigma = {s} ---")
        print(f"  {'variant':>18} {'latch':>6} | {'RWS':>7} | "
              f"{'per-step  fa/ft':>17} | {'per-trial fa/ft':>17} | {'S2 detect / step':>17}")
        print("  " + "-" * 92)
        for label, extra in W4_VARIANTS:
            for latch in (False, True):
                k = cell_key(dict(PROPOSED, seeds=seeds, observation_noise_sigma=s,
                                  abort_latching=latch, **extra))
                trials = list(load_cell(k).values())
                fa, ft = realized_alarm_rates(k)
                pfa, pft = per_trial_alarm_rates(k, CV_SCENARIOS)
                dfrac, dstep = detection_stats(k, "S2_EVASIVE_MANEUVER")
                print(f"  {label:>18} {str(latch):>6} | {rws(trials):>7.3f} | "
                      f"{fa*100:>7.2f}% /{ft*100:>7.2f}% | "
                      f"{pfa*100:>7.1f}% /{pft*100:>7.1f}% | "
                      f"{dfrac*100:>8.1f}% / {dstep:>4.0f}")
                rows.append(f"{s},{label},{int(latch)},{rws(trials):.4f},{fa:.5f},{ft:.5f},"
                            f"{pfa:.5f},{pft:.5f},{dfrac:.4f},{dstep}")
    write_csv("w4_window_latch.csv", rows[0], rows[1:])


# --------------------------------------------------------------------------- #
# W6 — closed-loop PN validation of the capability signals (Reviewers 1.1, 2.2)
# --------------------------------------------------------------------------- #
# The gate is untouched. At every ENGAGE the interceptor is actually flown, so
# the open question — does an intercept-triangle capability score predict what a
# bounded-acceleration pursuit achieves? — becomes a measured one.
# sigma=0 is a diagnostic, not an operating point: if the engagements that fail
# to capture disappear with a perfect sensor, they were estimation error flipping
# a near-critical capture-geometry test, not a property of the geometry itself.
W6_SIGMAS = [0.0, 0.1, 0.2, 0.5]
W6_VARIANTS = [
    ("capturability I (proposed)", {}),
    ("heuristic I", {"i_mode": "heuristic", "a_mode": "time_proxy"}),
]


def w6_cells(seeds: list[int]) -> dict:
    return {"w6": [dict(PROPOSED, seeds=seeds, observation_noise_sigma=s,
                        pn_validation=True, **extra)
                   for s in W6_SIGMAS for _, extra in W6_VARIANTS]}


def load_pn(key: str) -> list:
    with open(os.path.join(CELLS, key + ".json"), encoding="utf-8") as f:
        rows = json.load(f).get("pn_records")
    if rows is None:
        raise RuntimeError(f"cell {key} has no PN records; re-run w6 without --analyze-only")
    return rows


def w6_analyze(seeds: list[int]) -> None:
    from cuas_sim.guidance import PNParams, auc, simulate_abort

    print("\n" + "=" * 104)
    print("W6 — CLOSED-LOOP PN VALIDATION (R1.1 / R2.2): do the capability signals predict the flight?")
    print("=" * 104)
    p = PNParams()
    print(f"  PN: N'={p.nav_constant}, V_I={p.max_speed}, a_max={p.max_accel}, "
          f"tau={p.reaction_delay}, capture radius={p.capture_radius}")

    rows = ["sigma,variant,engagements,capture_rate,auc_I,median_miss,median_tgo,"
            "capture_rate_I_ge_0.5,capture_rate_I_lt_0.5"]
    for s in W6_SIGMAS:
        print(f"\n--- sigma = {s} ---")
        print(f"  {'variant':>27} | {'ENGAGEs':>8} | {'realized':>8} | {'AUC(I)':>7} | "
              f"{'miss':>6} | {'t_go':>6} | {'I>=0.5':>7} {'I<0.5':>7}")
        print("  " + "-" * 96)
        for label, extra in W6_VARIANTS:
            k = cell_key(dict(PROPOSED, seeds=seeds, observation_noise_sigma=s,
                              pn_validation=True, **extra))
            pn = load_pn(k)
            if not pn:
                print(f"  {label:>27} | no engagements")
                continue
            caps = [r for r in pn if r["captured"]]
            a = auc([r["interceptability"] for r in caps],
                    [r["interceptability"] for r in pn if not r["captured"]])
            miss = sorted(r["miss_distance"] for r in pn)[len(pn) // 2]
            tgo = sorted(r["time_to_go"] for r in pn)[len(pn) // 2]
            hi = [r for r in pn if r["interceptability"] >= 0.5]
            lo = [r for r in pn if r["interceptability"] < 0.5]
            hi_rate = sum(r["captured"] for r in hi) / len(hi) if hi else float("nan")
            lo_rate = sum(r["captured"] for r in lo) / len(lo) if lo else float("nan")
            a_txt = "    n/a" if a != a else f"{a:>7.3f}"
            print(f"  {label:>27} | {len(pn):>8} | {len(caps)/len(pn):>8.3f} | {a_txt} | "
                  f"{miss:>6.3f} | {tgo:>6.2f} | {hi_rate:>7.3f} {lo_rate:>7.3f}")
            rows.append(f"{s},{label},{len(pn)},{len(caps)/len(pn):.4f},{a:.4f},"
                        f"{miss:.4f},{tgo:.4f},{hi_rate:.4f},{lo_rate:.4f}")

    print("\n  AUC is pooled across scenarios. Within any single scenario the flown")
    print("  outcome turns out to be degenerate (all captured, or none), so the")
    print("  discrimination the pooled figure reports is entirely between scenarios.")
    print("  'n/a' means the variant never produced a failed engagement to separate.")

    # Per-scenario, at the nominal operating point, for both I signals side by
    # side: which targets does each one commit to, and what does the flight say?
    print(f"\n--- per scenario at sigma=0.2: ENGAGEs / realized capture / median miss ---")
    print(f"  {'scenario':>34} | {'capturability I':>28} | {'heuristic I':>28}")
    print("  " + "-" * 96)
    per_variant = {}
    for label, extra in W6_VARIANTS:
        k = cell_key(dict(PROPOSED, seeds=seeds, observation_noise_sigma=0.2,
                          pn_validation=True, **extra))
        d: dict = {}
        for r in load_pn(k):
            d.setdefault(r["scenario"], []).append(r)
        per_variant[label] = d
    for scn in sorted(set().union(*(d.keys() for d in per_variant.values()))):
        cells_txt = []
        for label, _ in W6_VARIANTS:
            g = per_variant[label].get(scn, [])
            if not g:
                cells_txt.append(f"{'-- never engages --':>28}")
                continue
            rate = sum(r["captured"] for r in g) / len(g)
            miss = sorted(r["miss_distance"] for r in g)[len(g) // 2]
            cells_txt.append(f"{len(g):>7} / {rate:>6.3f} / {miss:>8.3f}")
            rows.append(f"0.2,{label}|{scn},{len(g)},{rate:.4f},,{miss:.4f},,,")
        print(f"  {scn:>34} | {cells_txt[0]} | {cells_txt[1]}")

    # Is the shortfall a property of the geometry, or of the estimate?
    print("\n--- diagnostic: does a perfect sensor remove the failed engagements? ---")
    for label, extra in W6_VARIANTS:
        line = []
        for s in W6_SIGMAS:
            k = cell_key(dict(PROPOSED, seeds=seeds, observation_noise_sigma=s,
                              pn_validation=True, **extra))
            pn = load_pn(k)
            failed = [r for r in pn if not r["captured"]]
            line.append(f"sigma={s}: {len(failed):>6}")
        print(f"  {label:>27} | " + "  ".join(line))

    # A_abort: the manuscript predicts t_req = tau + V_I/a_max analytically.
    t_stop, distance = simulate_abort(p)
    predicted = p.reaction_delay + p.max_speed / p.max_accel
    print(f"\n--- abort dynamics ---")
    print(f"  predicted t_req = tau + V_I/a_max = {predicted:.3f} s")
    print(f"  flown arrest-and-reverse          = {t_stop:.3f} s, "
          f"distance run {distance:.3f} units")
    rows.append(f",abort_t_req_predicted,,{predicted:.4f},,,,,")
    rows.append(f",abort_t_req_flown,,{t_stop:.4f},,{distance:.4f},,,")
    write_csv("w6_pn_validation.csv", rows[0], rows[1:])


# --------------------------------------------------------------------------- #
# W13 — statistical protocol (Reviewer 1.5b)
# --------------------------------------------------------------------------- #
# "The claim that the calibrated gate is strictly safer where it matters rests
# on very few events. I suggest increasing seeds ... and reporting effect sizes
# with confidence intervals alongside p-values."
#
# The headline comparison is re-run at N=10,000 and reported as a paired
# difference with its bootstrap interval, next to the absolute event counts the
# rates are computed from. The N=1000 column is kept so the reader can see that
# the conclusions are not a function of the sample size.
W13_SEEDS = 10000


def w13_cells(seeds: list) -> dict:
    big = list(range(1, W13_SEEDS + 1))
    return {"w13": [c for s in W1_SIGMAS
                    for c in (heuristic_default(big, s),
                              dict(PROPOSED, seeds=big, observation_noise_sigma=s))]}


def w13_analyze(seeds: list) -> None:
    rws = make_rws_aggregator()
    big = list(range(1, W13_SEEDS + 1))
    small = list(range(1, 1001))

    print("\n" + "=" * 104)
    print(f"W13 — STATISTICAL PROTOCOL (R1.5b): N={W13_SEEDS:,}, effect sizes with 95% CIs")
    print("=" * 104)
    print(f"\n  {'sigma':>6} | {'heuristic':>9} | {'proposed':>9} | "
          f"{'difference [95% CI]':>28} | {'p':>7} | {'N=1000 diff':>11}")
    print("  " + "-" * 92)

    rows = ["sigma,n_seeds,heuristic_rws,proposed_rws,diff,ci_lo,ci_hi,p_value,"
            "heur_unsafe_n,heur_attempts,prop_unsafe_n,prop_attempts"]
    for s in W1_SIGMAS:
        a_big = load_cell(cell_key(heuristic_default(big, s)))
        d_big = load_cell(cell_key(dict(PROPOSED, seeds=big, observation_noise_sigma=s)))
        la, ld = aligned(a_big, d_big)
        diff = paired_bootstrap_diff(ld, la, rws, seed=1)

        a_small = list(load_cell(cell_key(heuristic_default(small, s))).values())
        d_small = list(load_cell(cell_key(dict(PROPOSED, seeds=small,
                                               observation_noise_sigma=s))).values())
        small_diff = rws(d_small) - rws(a_small)

        print(f"  {s:>6} | {rws(la):>9.4f} | {rws(ld):>9.4f} | "
              f"{diff['point_diff']:>+9.4f} [{diff['ci_lo']:+.4f},{diff['ci_hi']:+.4f}] | "
              f"{diff['p_value_one_sided']:>7.4f} | {small_diff:>+11.4f}")
        rows.append(
            f"{s},{W13_SEEDS},{rws(la):.4f},{rws(ld):.4f},{diff['point_diff']:.4f},"
            f"{diff['ci_lo']:.4f},{diff['ci_hi']:.4f},{diff['p_value_one_sided']:.4f},"
            f"{sum(t.unsafe_engagement_count for t in la)},{sum(t.engage_count for t in la)},"
            f"{sum(t.unsafe_engagement_count for t in ld)},{sum(t.engage_count for t in ld)}")

    # The rates the reviewer called thin, now with the counts behind them.
    print("\n--- absolute event counts behind the unsafe rates ---")
    print(f"  {'sigma':>6} | {'heuristic unsafe / attempts':>30} | {'proposed unsafe / attempts':>30}")
    print("  " + "-" * 74)
    for s in W1_SIGMAS:
        la = list(load_cell(cell_key(heuristic_default(big, s))).values())
        ld = list(load_cell(cell_key(dict(PROPOSED, seeds=big,
                                          observation_noise_sigma=s))).values())
        au, an = sum(t.unsafe_engagement_count for t in la), sum(t.engage_count for t in la)
        du, dn = sum(t.unsafe_engagement_count for t in ld), sum(t.engage_count for t in ld)
        print(f"  {s:>6} | {au:>8,} / {an:>10,}  ({au/an*100:>6.3f}%) | "
              f"{du:>8,} / {dn:>10,}  ({du/dn*100:>6.3f}%)")
    print("\n  At N=10,000 the unsafe counts are three digits rather than single, so the")
    print("  rates are no longer resting on a handful of events. Note that the proposed")
    print("  gate's unsafe RATE exceeds the heuristic's at high noise: it engages an order")
    print("  of magnitude more often, and that is the trade the counts make visible.")
    write_csv("w13_statistical_protocol.csv", rows[0], rows[1:])


# --------------------------------------------------------------------------- #
# W12 — modern decision baselines (Reviewer 2.9)
# --------------------------------------------------------------------------- #
# The submitted comparison set is three deliberately weak single-criterion rules
# plus the conference heuristic, so the only real competitor is the gate's own
# ancestor. Two stronger rules are added here, both consuming exactly the same
# signals, so what is compared is the DECISION RULE and nothing else:
#
#   BayesRisk      myopic QMDP-style expected-utility maximisation under the
#                  filter posterior, with the paper's own scoring weights as the
#                  cost model - the cheap standard approximation to a POMDP.
#   Reachability   worst-case set-based safety filter: authorise contact only if
#                  the speed-bounded forward reachable set clears the keep-out
#                  disk. Probability replaced by a guarantee.
W12_SIGMAS = [0.1, 0.2, 0.5]
W12_POLICIES = ["SafetyGatePolicy", "BayesRiskPolicy",
                "ReachabilityWithTrackingPolicy", "ReachabilityFilterPolicy"]


def w12_cell(seeds: list, sigma: float) -> dict:
    from cuas_sim.scenario import get_extended_scenario_types
    cfg = dict(PROPOSED, seeds=seeds, observation_noise_sigma=sigma,
               include_scenarios=get_extended_scenario_types())
    cfg["include_policies"] = W12_POLICIES
    return cfg


def w12_cells(seeds: list) -> dict:
    return {}   # runs directly: the per-policy split needs all policies in one result


def w12_analyze(seeds: list, jobs: int = 4) -> None:
    from cuas_sim.statistics import (aggregate_false_per_attempt, aggregate_late_abort,
                                     make_rws_aggregator)
    rws = make_rws_aggregator()
    print("\n" + "=" * 104)
    print("W12 — MODERN DECISION BASELINES (R2.9): same signals, different decision rules")
    print("=" * 104)
    print("  extended scenario set; every policy reads the calibrated front end")
    print("  (Kalman C_track, intrusion-probability R_env, capturability I, abort dynamics).")

    rows = ["sigma,policy,rws,safe_capture,unsafe_per_attempt,false_per_attempt,"
            "abort_success,engage_rate,track_rate,abort_rate"]
    for s in W12_SIGMAS:
        cfg = w12_cell(seeds, s)
        result = MonteCarloRunner(MonteCarloConfig(**cfg)).run()
        per_trial = result.per_trial_summaries()
        counts = {}
        for rec in result.records:
            c = counts.setdefault(rec.policy_name, {"ENGAGE": 0, "TRACK": 0, "ABORT": 0})
            c[rec.action.value] += 1
        print(f"\n--- sigma = {s} ---")
        print(f"  {'policy':>26} | {'RWS':>7} | {'SafeCap':>8} | {'unsafe':>8} | {'false':>8} |"
              f" {'abort ok':>8} | {'E/T/A rates':>22}")
        print("  " + "-" * 100)
        for name in W12_POLICIES:
            t = [x for x in per_trial if x.policy_name == name]
            c = counts[name]
            tot = sum(c.values())
            e, tr, ab = (c["ENGAGE"] / tot, c["TRACK"] / tot, c["ABORT"] / tot)
            print(f"  {name:>26} | {rws(t):>7.3f} | {aggregate_safe_capture(t):>8.4f} | "
                  f"{aggregate_unsafe_per_attempt(t):>8.5f} | {aggregate_false_per_attempt(t):>8.5f} | "
                  f"{aggregate_abort_success(t):>8.3f} | {e:>6.3f} /{tr:>6.3f} /{ab:>6.3f}")
            rows.append(f"{s},{name},{rws(t):.4f},{aggregate_safe_capture(t):.4f},"
                        f"{aggregate_unsafe_per_attempt(t):.6f},{aggregate_false_per_attempt(t):.6f},"
                        f"{aggregate_abort_success(t):.4f},{e:.4f},{tr:.4f},{ab:.4f}")
    write_csv("w12_modern_baselines.csv", rows[0], rows[1:])
    print("\n  Read the engage/track/abort split alongside the score: a rule that")
    print("  almost never engages buys its clean safety numbers by declining the")
    print("  mission, which the risk-weighted score is designed to charge for.")


# --------------------------------------------------------------------------- #
# W11 — quadrature accuracy and runtime cost (Reviewers 2.7, 2.13)
# --------------------------------------------------------------------------- #
# Reviewer 2.7 asks how large the quadrature error is and how it "accumulates
# over long prediction horizons". The premise needs correcting rather than
# conceding: R_env is a MAX over horizon steps, each an independently evaluated
# integral at its own (mu_m, Sigma_m), so nothing accumulates along m. What does
# grow with the horizon is the covariance itself, which is modelling, not error.
#
# The accuracy claim is settled against an independent reference rather than a
# finer version of the same rule: in the isotropic case the exact answer is the
# non-central chi-square CDF (equivalently 1 - Q1(d/sigma, R/sigma)), which sums
# to machine precision from the same even-dof tail already used for C_track.
W11_NODES = (16, 32, 64, 128, 256)   # 64 is the shipped setting


def _noncentral_chi2_2_cdf(x: float, lam: float, terms: int = 400) -> float:
    """P(noncentral chi-square with 2 dof and noncentrality lam <= x).

    Poisson mixture of central chi-squares: sum_j Pois(j; lam/2) * P(chi2_{2+2j} <= x).
    This is the exact intrusion probability for an isotropic predictive Gaussian.
    """
    from cuas_sim.estimation import chi2_even_sf
    if x <= 0.0:
        return 0.0
    half_lam = 0.5 * lam
    total = 0.0
    weight = math.exp(-half_lam)      # j = 0 Poisson weight
    for j in range(terms):
        if weight > 0.0:
            total += weight * (1.0 - chi2_even_sf(x, j + 1))
        if j + 1 < terms:
            weight *= half_lam / (j + 1)
            if weight < 1e-18 and j > half_lam:
                break
    return min(1.0, total)


def exact_disk_probability_isotropic(mu_x, mu_y, sigma, cx, cy, R) -> float:
    """Closed-form reference: 1 - Q1(d/sigma, R/sigma) via the non-central chi2."""
    d = math.hypot(mu_x - cx, mu_y - cy)
    return _noncentral_chi2_2_cdf((R / sigma) ** 2, (d / sigma) ** 2)


def w11_cells(seeds: list) -> dict:
    return {}  # numerical study; no Monte Carlo cells


def w11_analyze(seeds: list, jobs: int = 4) -> None:
    import random
    import time
    from cuas_sim.evaluators import gaussian_disk_probability

    print("\n" + "=" * 100)
    print("W11 — QUADRATURE ACCURACY AND RUNTIME COST (R2.7 / R2.13)")
    print("=" * 100)
    rows = ["section,key,value"]

    # ---- 1. accuracy and convergence order ---------------------------------
    print("\n--- 1. Against the exact non-central chi-square reference (isotropic) ---")
    print("  The disk chord h(x) = sqrt(R^2-(x-cx)^2) has an infinite derivative at")
    print("  x = cx +- R. Simpson assumes smoothness, so the shipped rule converges at")
    print("  O(h^1.5), not O(h^4). Substituting x = cx + R sin t cancels that exactly.")
    grid = [(1.0, d * 0.5, R) for d in range(7) for R in (0.5, 1.0, 2.0, 4.0)]
    grid += [(0.3, d * 0.15, R) for d in range(7) for R in (0.2, 0.6, 1.2)]
    print(f"\n  {'nodes':>6} | {'simpson (as submitted)':>24} | {'trig substitution':>20}")
    print("  " + "-" * 58)
    prev = {}
    for n in W11_NODES:
        line = {}
        for rule in ("simpson", "trig"):
            errs = [abs(gaussian_disk_probability(d, 0.0, sg, sg, 0.0, 0.0, R, n=n, rule=rule)
                        - exact_disk_probability_isotropic(d, 0.0, sg, 0.0, 0.0, R))
                    for sg, d, R in grid]
            line[rule] = max(errs)
            rows.append(f"accuracy,{rule}_max_abs_err_n{n},{max(errs):.6e}")
        order = {r: (math.log2(prev[r] / line[r]) if prev.get(r) and line[r] > 0 else None)
                 for r in line}
        fmt = lambda r: (f"{line[r]:.3e}" + (f"  (order {order[r]:.2f})" if order[r] else ""))
        print(f"  {n:>6} | {fmt('simpson'):>24} | {fmt('trig'):>20}")
        prev = line
    print("\n  At the shipped n=64 the error is ~2e-3, i.e. about 4% of theta_risk=0.05 —")
    print("  small, but not the negligible quantity it is natural to assume.")
    print("  The trig rule gives ~1e-9 at the SAME node count, six orders better.")
    print("  Its floor is not the quadrature but the +-6 sigma clipping of the")
    print("  Gaussian tails, which is why it stops improving as n grows.")

    # ---- 2. anisotropic case (no closed form) -------------------------------
    print("\n--- 2. Anisotropic case: n=64 vs a 4096-node reference of the same rule ---")
    rng = random.Random(12345)
    cases = []
    for _ in range(200):
        sx = rng.uniform(0.05, 1.5)
        cases.append((rng.uniform(-3, 3), rng.uniform(-3, 3), sx,
                      sx * rng.uniform(0.2, 5.0), rng.uniform(0.3, 3.0)))
    for rule in ("simpson", "trig"):
        errs = [abs(gaussian_disk_probability(mx, my, sx, sy, 0.0, 0.0, R, n=64, rule=rule)
                    - gaussian_disk_probability(mx, my, sx, sy, 0.0, 0.0, R, n=4096, rule="trig"))
                for mx, my, sx, sy, R in cases]
        print(f"  {rule:>8}: max {max(errs):.3e}   mean {sum(errs)/len(errs):.3e}")
        rows.append(f"accuracy_anisotropic,{rule}_max_err,{max(errs):.6e}")

    # ---- 3. does the error accumulate along the horizon? --------------------
    print("\n--- 3. Error along the prediction horizon (the premise in R2.7) ---")
    print(f"\n  {'m':>4} | {'sigma_m':>9} | {'P_m':>9} | {'simpson err':>12} | {'trig err':>12}")
    print("  " + "-" * 60)
    var, cx, R = 0.04, 4.0, 2.6
    worst = {"simpson": 0.0, "trig": 0.0}
    for m in range(1, 31):
        var = var + 0.1 * (0.1 ** 4) / 4.0 + 0.05 * m * 0.1
        sig = math.sqrt(var)
        mu = 10.0 - 0.3 * m
        exact = exact_disk_probability_isotropic(mu, 0.0, sig, cx, 0.0, R)
        e = {r: abs(gaussian_disk_probability(mu, 0.0, sig, sig, cx, 0.0, R, n=64, rule=r) - exact)
             for r in ("simpson", "trig")}
        for r in e:
            worst[r] = max(worst[r], e[r])
        if m in (1, 5, 10, 20, 30):
            p = gaussian_disk_probability(mu, 0.0, sig, sig, cx, 0.0, R, n=64)
            print(f"  {m:>4} | {sig:>9.4f} | {p:>9.6f} | {e['simpson']:>12.3e} | {e['trig']:>12.3e}")
        rows.append(f"horizon,simpson_abs_err_m{m},{e['simpson']:.6e}")
    print(f"\n  max over m: simpson {worst['simpson']:.3e}, trig {worst['trig']:.3e}.")
    print("  R_env = max_m P_m, so the reported value carries ONE step's error and the")
    print("  errors are never summed. Nothing accumulates along the horizon; what grows")
    print("  with m is the predictive covariance itself, which is modelling, not error.")

    # ---- 4. what the error does to the DECISIONS ----------------------------
    print("\n--- 4. Decision-level impact (measured, N=1000, sigma=0.2) ---")
    print("  Same runs under each rule, compared decision by decision:")
    print(f"\n  {'scenario set':>20} | {'RWS simpson':>12} | {'RWS trig':>10} | {'decisions changed':>20}")
    print("  " + "-" * 72)
    for label, n_dec, rws_s, rws_t, changed in (
            ("core 6", 120000, 1.3121, 1.3121, 1),
            ("extended", 180000, 1.2316, 1.2316, 1),
            ("S11 grazing only", 20000, 0.6789, 0.6789, 0)):
        print(f"  {label:>20} | {rws_s:>12.4f} | {rws_t:>10.4f} | "
              f"{changed:>6} / {n_dec}  ({changed/n_dec*100:.3f}%)")
        rows.append(f"decision_impact,{label.replace(' ', '_')}_changed,{changed}")
    print("\n  RWS is unchanged to four decimals and one decision in 120,000 flips.")
    print("  The quadrature error is therefore far below what could distort the gate,")
    print("  and the trig rule is available when a tighter bound is wanted.")

    # ---- 5. runtime ---------------------------------------------------------
    print("\n--- 5. Runtime per decision step (R2.13) ---")
    from cuas_sim.estimation import KalmanTargetEstimator
    from cuas_sim.observation import TargetObservation
    from cuas_sim.types import State2D

    def bench(fn, reps=20000):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e6

    est = KalmanTargetEstimator(time_step=0.1, process_noise=0.1, measurement_sigma=0.2)
    step = [0]

    def kalman_step():
        step[0] += 1
        est.estimate(TargetObservation(
            step_index=step[0], detected=True,
            observed_state=State2D(x=step[0] * 0.1, y=0.0, vx=1.0, vy=0.0),
            tracking_confidence=0.9, identification_confidence=0.9, metadata={}))

    # A geometry where the disk and the probability mass actually overlap; with
    # mu far outside, the routine short-circuits and the timing is meaningless.
    GEO = dict(mu_x=1.0, mu_y=0.2, sigma_x=0.3, sigma_y=0.35, cx=0.0, cy=0.0, R=2.0)
    assert gaussian_disk_probability(**GEO) > 0.01, "benchmark geometry must integrate"
    t_kf = bench(kalman_step)
    t_simp = bench(lambda: gaussian_disk_probability(**GEO, n=64, rule="simpson"))
    t_trig = bench(lambda: gaussian_disk_probability(**GEO, n=64, rule="trig"))

    print(f"\n  {'stage':>36} | {'per call':>11} | {'H=5':>9} | {'H=30':>9}")
    print("  " + "-" * 72)
    print(f"  {'Kalman update (2 x 2-state)':>36} | {t_kf:>8.2f} us | {t_kf:>6.1f} us | {t_kf:>6.1f} us")
    print(f"  {'intrusion quadrature, simpson':>36} | {t_simp:>8.2f} us | "
          f"{t_simp*5:>6.1f} us | {t_simp*30:>6.1f} us")
    print(f"  {'intrusion quadrature, trig':>36} | {t_trig:>8.2f} us | "
          f"{t_trig*5:>6.1f} us | {t_trig*30:>6.1f} us")
    print(f"\n  The quadrature dominates: {t_simp/t_kf:.1f}x one filter update per horizon step,")
    print(f"  so a decision costs about {t_kf + t_simp*5:.0f} us at H=5 and "
          f"{t_kf + t_simp*30:.0f} us at H=30 in pure Python.")
    print("  These are CPython figures on a desktop; the point for an embedded target is")
    print("  the shape, not the constant: one O(1) filter update plus H fixed-node")
    print("  quadratures, no iteration to convergence and no dynamic allocation.")
    for k, v in (("kalman_us", t_kf), ("quad_simpson_us", t_simp), ("quad_trig_us", t_trig)):
        rows.append(f"runtime,{k},{v:.4f}")
    write_csv("w11_quadrature_and_cost.csv", rows[0], rows[1:])


# --------------------------------------------------------------------------- #
# W10 — identification-confidence pipeline distortion (Reviewer 2.8)
# --------------------------------------------------------------------------- #
# C_id is a scenario input in the manuscript, which is the right abstraction for
# a classifier-agnostic decision layer but leaves open what a real, imperfect
# classifier would cost. Measured on the EXTENDED set: only S9 (paper S8) carries
# misidentification risk, so on the core six a C_id error has no safety
# consequence to observe and the study would be vacuous.
W10_SIGMA = 0.2
W10_CASES = (
    [("bias", b) for b in (-0.4, -0.2, -0.1, 0.1, 0.2)]
    + [("gamma", g) for g in (0.5, 0.7, 1.5, 2.0)]
    + [("noise", n) for n in (0.05, 0.1, 0.2)]
)


def w10_cell(seeds: list, kind: str = None, value: float = None) -> dict:
    from cuas_sim.scenario import get_extended_scenario_types
    cfg = dict(PROPOSED, seeds=seeds, observation_noise_sigma=W10_SIGMA,
               include_scenarios=get_extended_scenario_types())
    if kind == "bias":
        cfg["cid_bias"] = value
    elif kind == "gamma":
        cfg["cid_gamma"] = value
    elif kind == "noise":
        cfg["cid_noise_sigma"] = value
    return cfg


def w10_cells(seeds: list) -> dict:
    return {"w10": [w10_cell(seeds)] + [w10_cell(seeds, k, v) for k, v in W10_CASES]}


def w10_analyze(seeds: list) -> None:
    from cuas_sim.statistics import aggregate_false_per_attempt
    rws = make_rws_aggregator()

    print("\n" + "=" * 100)
    print("W10 — IDENTIFICATION-CONFIDENCE DISTORTION (R2.8): what an imperfect classifier costs")
    print("=" * 100)
    print(f"  extended scenario set, sigma={W10_SIGMA}, theta_id = 0.6")
    print("  false/att is the metric that matters here: engaging a misidentified target.")
    print(f"\n  {'distortion':>22} | {'RWS':>7} | {'SafeCap':>8} | {'false/att':>10} | "
          f"{'unsafe/att':>10} | {'abort succ':>10}")
    print("  " + "-" * 86)

    rows = ["kind,value,rws,safe_capture,false_per_attempt,unsafe_per_attempt,abort_success"]

    def emit(label, kind, value):
        t = list(load_cell(cell_key(w10_cell(seeds, kind, value))).values())
        f = aggregate_false_per_attempt(t)
        print(f"  {label:>22} | {rws(t):>7.3f} | {aggregate_safe_capture(t):>8.4f} | "
              f"{f:>10.5f} | {aggregate_unsafe_per_attempt(t):>10.5f} | "
              f"{aggregate_abort_success(t):>10.3f}")
        rows.append(f"{kind or 'none'},{value if value is not None else 0},{rws(t):.4f},"
                    f"{aggregate_safe_capture(t):.4f},{f:.6f},"
                    f"{aggregate_unsafe_per_attempt(t):.6f},{aggregate_abort_success(t):.4f}")

    emit("none (as submitted)", None, None)
    for kind, group in (("bias", "offset C_id + b"), ("gamma", "miscalibration C_id^g"),
                        ("noise", "jitter C_id + N(0,s)")):
        print(f"  {'-- ' + group:>22}")
        for k, v in W10_CASES:
            if k == kind:
                emit(f"{kind} = {v:+.2f}" if kind == "bias" else f"{kind} = {v}", k, v)
    write_csv("w10_cid_distortion.csv", rows[0], rows[1:])
    print("\n  Structural expectation to check against: the gate uses C_id only through")
    print("  the test C_id < theta_id, so a monotone distortion (gamma) should matter")
    print("  only where it moves values across 0.6, while an offset moves everything.")


# --------------------------------------------------------------------------- #
# W9 — contribution decomposition (Reviewer 2.1)
# --------------------------------------------------------------------------- #
# The manuscript already decomposes the gain, but cumulatively (A -> B -> C -> D)
# and at one noise level, which cannot show whether the components interact. A
# full 2x2x2 factorial can, and it also separates each component's own effect
# from the order it was switched on in.
#
# Each factor carries its own threshold, because a threshold on a p-value and a
# threshold on an uncalibrated score are not the same quantity: turning on the
# Kalman estimator without moving 0.3/0.6 to 0.01/0.05 would measure a
# mis-specified gate rather than the calibration.
W9_FACTORS = ("kalman", "prob_renv", "capturability")


def w9_cell(seeds: list[int], sigma: float, on: tuple, entangled: bool = False) -> dict:
    kalman, prob_renv, capt = on
    cfg = dict(BASE, seeds=seeds, observation_noise_sigma=sigma)
    cfg["safety_tracking_abort_threshold"] = 0.01 if kalman else 0.3
    cfg["safety_tracking_confidence_threshold"] = 0.05 if kalman else 0.6
    cfg["estimator_mode"] = "kalman" if kalman else "alpha_beta"
    cfg["r_env_mode"] = "probability" if prob_renv else "linear"
    cfg["safety_risk_score_threshold"] = 0.05 if prob_renv else 0.7
    if prob_renv:
        cfg["risk_keepout_buffer"] = 0.6
    if capt:
        cfg["i_mode"] = "capturability"
        cfg["a_mode"] = "dynamics"
    # Entangled = the submitted-conference mistake: the evaluation's "unsafe"
    # cutoff follows the policy's own risk threshold instead of being fixed.
    if not entangled:
        cfg["eval_risk_score_threshold"] = 0.7
    return cfg


def w9_cells(seeds: list[int]) -> dict:
    corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    cells = [w9_cell(seeds, s, on) for s in W1_SIGMAS for on in corners]
    # Entanglement study: only the two end points are needed.
    cells += [w9_cell(seeds, s, on, entangled=True)
              for s in W1_SIGMAS for on in ((0, 0, 0), (1, 1, 1))]
    return {"w9": cells}


def w9_analyze(seeds: list[int]) -> None:
    rws = make_rws_aggregator()
    corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]

    print("\n" + "=" * 104)
    print("W9 — CONTRIBUTION DECOMPOSITION (R2.1): 2x2x2 factorial, every noise level")
    print("=" * 104)
    print("  factors: K = Kalman C_track | P = probabilistic R_env | C = capturability I / abort dynamics")

    rows = ["sigma,kalman,prob_renv,capturability,rws"]
    scores: dict = {}
    header = "  " + f"{'sigma':>6} |" + "".join(
        f"{''.join(n for n, f in zip('KPC', on) if f) or '(none)':>9}" for on in corners)
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for s in W1_SIGMAS:
        line = ""
        for on in corners:
            v = rws(list(load_cell(cell_key(w9_cell(seeds, s, on))).values()))
            scores[(s, on)] = v
            line += f"{v:>9.3f}"
            rows.append(f"{s},{on[0]},{on[1]},{on[2]},{v:.4f}")
        print(f"  {s:>6} |" + line)

    # Main effects and interactions, in the usual factorial sense: the effect of
    # a factor is the mean change it causes averaged over the other factors.
    print("\n--- main effects and interactions (mean DeltaRWS) ---")
    print(f"  {'sigma':>6} |" + "".join(f"{n:>9}" for n in ("K", "P", "C", "KxP", "KxC", "PxC")))
    print("  " + "-" * 62)
    eff_rows = ["sigma,effect,value"]
    for s in W1_SIGMAS:
        def mean_on(i, val):
            sel = [scores[(s, on)] for on in corners if on[i] == val]
            return sum(sel) / len(sel)

        def inter(i, j):
            # (both on + both off) - (one on) - (other on), halved: the standard
            # two-factor interaction contrast.
            same = [scores[(s, on)] for on in corners if on[i] == on[j]]
            diff = [scores[(s, on)] for on in corners if on[i] != on[j]]
            return sum(same) / len(same) - sum(diff) / len(diff)

        vals = [mean_on(0, 1) - mean_on(0, 0), mean_on(1, 1) - mean_on(1, 0),
                mean_on(2, 1) - mean_on(2, 0), inter(0, 1), inter(0, 2), inter(1, 2)]
        print(f"  {s:>6} |" + "".join(f"{v:>+9.3f}" for v in vals))
        for name, v in zip(("K", "P", "C", "KxP", "KxC", "PxC"), vals):
            eff_rows.append(f"{s},{name},{v:.4f}")
    write_csv("w9_factorial.csv", rows[0], rows[1:])
    write_csv("w9_effects.csv", eff_rows[0], eff_rows[1:])

    # How much did the entangled criterion distort the very comparison the
    # manuscript reports? This is the first direct measurement of the
    # methodological claim in Section V-A.
    print("\n--- evaluation-criterion entanglement: D - A under each yardstick ---")
    print("  entangled = the evaluation's unsafe cutoff follows the policy's own risk threshold")
    print(f"\n  {'sigma':>6} | {'decoupled (fixed 0.7)':>22} | {'entangled':>22} | {'distortion':>11}")
    print("  " + "-" * 72)
    ent_rows = ["sigma,decoupled_diff,entangled_diff,distortion"]
    for s in W1_SIGMAS:
        dec = (scores[(s, (1, 1, 1))], scores[(s, (0, 0, 0))])
        a_e = rws(list(load_cell(cell_key(w9_cell(seeds, s, (0, 0, 0), True))).values()))
        d_e = rws(list(load_cell(cell_key(w9_cell(seeds, s, (1, 1, 1), True))).values()))
        dec_diff, ent_diff = dec[0] - dec[1], d_e - a_e
        print(f"  {s:>6} | {dec_diff:>+22.4f} | {ent_diff:>+22.4f} | {ent_diff - dec_diff:>+11.4f}")
        ent_rows.append(f"{s},{dec_diff:.4f},{ent_diff:.4f},{ent_diff - dec_diff:.4f}")
    write_csv("w9_entanglement.csv", ent_rows[0], ent_rows[1:])


# --------------------------------------------------------------------------- #
# W8 — do the two risk models actually differ? (Reviewer 1.3)
# --------------------------------------------------------------------------- #
# Section V-F of the manuscript conceded that S7 saturates both R_env models and
# that separating them needs a marginal, grazing intrusion. S11 is that
# scenario. Because only two configurations are involved this runs directly
# rather than through the cell cache.
W8_SIGMAS = [0.1, 0.2, 0.5]


def _renv_roc_rows(sigma: float, r_env_mode: str, seeds: list[int],
                   scenario) -> tuple[list, list]:
    """(R_env where the contact would be unsafe, R_env where it would be safe).

    The label is the manuscript's fixed ground-truth criterion (8): the true
    target inside the keep-out disk of radius R_zone + b, b = 2(1 - theta_eval)
    = 0.6. It never depends on the policy or on the risk model being scored.
    """
    import math
    cfg = dict(PROPOSED, seeds=seeds, observation_noise_sigma=sigma,
               r_env_mode=r_env_mode, include_scenarios=[scenario])
    if r_env_mode == "linear":
        # The linear model has no keep-out buffer; leave its own parameters at
        # the baseline values so this scores the submitted model, not a hybrid.
        cfg.pop("risk_keepout_buffer", None)
    result = MonteCarloRunner(MonteCarloConfig(**cfg)).run()
    pos, neg = [], []
    for rec in result.records:
        if rec.policy_name != POLICY or rec.true_target_state is None:
            continue
        meta = rec.scenario_metadata_snapshot or {}
        c = meta.get("risk_zone_center")
        if c is None:
            continue
        r_zone = meta.get("risk_zone_radius", 0.0)
        d = math.hypot(rec.true_target_state.x - c[0], rec.true_target_state.y - c[1])
        (pos if d <= r_zone + 0.6 else neg).append(rec.risk_score)
    return pos, neg


def w8_analyze(seeds: list[int], jobs: int = 4) -> None:
    from cuas_sim.guidance import auc
    from cuas_sim.scenario import ScenarioType

    print("\n" + "=" * 96)
    print("W8 — RISK-MODEL DISCRIMINATION (R1.3): linear falloff vs intrusion probability")
    print("=" * 96)
    print("  AUC of R_env against the fixed ground-truth unsafe label, per step.")
    print("  S4/S7 are the submitted scenarios; S11 is the new marginal-grazing one.")

    rows = ["scenario,sigma,model,n_unsafe,n_safe,auc"]
    scenarios = [("S4 (submitted, always clears)", ScenarioType.S4_RISK_ZONE_PROXIMITY),
                 ("S7 (submitted, always transits)", ScenarioType.S7_RISK_ZONE_TRANSIT),
                 ("S11 (new, marginal grazing)", ScenarioType.S11_MARGINAL_GRAZING)]
    for label, scn in scenarios:
        print(f"\n--- {label} ---")
        print(f"  {'sigma':>6} | {'unsafe/safe steps':>20} | {'linear':>8} | "
              f"{'probability':>12} | {'difference':>11}")
        print("  " + "-" * 72)
        for s in W8_SIGMAS:
            out = {}
            for mode in ("linear", "probability"):
                pos, neg = _renv_roc_rows(s, mode, seeds, scn)
                out[mode] = (auc(pos, neg), len(pos), len(neg))
                rows.append(f"{scn.value},{s},{mode},{len(pos)},{len(neg)},{out[mode][0]:.4f}")
            lin, prob = out["linear"][0], out["probability"][0]
            n_pos, n_neg = out["linear"][1], out["linear"][2]
            diff = prob - lin
            fmt = lambda v: "     n/a" if v != v else f"{v:>8.4f}"
            print(f"  {s:>6} | {n_pos:>9} / {n_neg:<8} | {fmt(lin)} | {fmt(prob):>12} | "
                  f"{('    n/a' if diff != diff else f'{diff:>+11.4f}')}")
    write_csv("w8_risk_model_roc.csv", rows[0], rows[1:])
    print("\n  A tie is a legitimate outcome: the probabilistic model's stated value is")
    print("  the removal of k_sigma, the falloff width and the per-step growth rule,")
    print("  plus a threshold that reads as an admissible intrusion probability.")


def w8_cells(seeds: list[int]) -> dict:
    return {}  # runs directly in the analysis; only two configurations


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# W14 — extended-scenario table, regenerated (mock review M3/M7)
# --------------------------------------------------------------------------- #
W14_SIGMA = 0.2


def w14_cells(seeds: list) -> dict:
    from cuas_sim.scenario import get_extended_scenario_types
    ext = get_extended_scenario_types()
    return {"extended": [
        dict(cfg, seeds=seeds, observation_noise_sigma=W14_SIGMA,
             include_scenarios=ext)
        for cfg in (BASE, PROPOSED)
    ]}


def w14_analyze(seeds: list) -> None:
    from cuas_sim.scenario import get_extended_scenario_types
    ext = get_extended_scenario_types()
    keys = [cell_key(dict(cfg, seeds=seeds, observation_noise_sigma=W14_SIGMA,
                          include_scenarios=ext))
            for cfg in (BASE, PROPOSED)]

    rows = []
    print("")
    print(f"W14  extended scenarios at sigma={W14_SIGMA}, N={len(seeds)}")
    print("  scenario                     pol  engage/dec        unsafe/eng "
          "(trials)   abort succ")
    for label, key in zip("AD", keys):
        cell = load_cell(key)
        per = {}
        for t in cell.values():
            d = per.setdefault(t.scenario_type, dict(
                dec=0, eng=0, uns=0, fls=0, req=0, ok=0,
                n=0, uns_trials=0, fls_trials=0))
            d["n"] += 1
            d["dec"] += t.n_steps
            d["eng"] += t.engage_count
            d["uns"] += t.unsafe_engagement_count
            d["fls"] += t.false_engagement_count
            d["uns_trials"] += 1 if t.unsafe_engagement_count else 0
            d["fls_trials"] += 1 if t.false_engagement_count else 0
            if t.abort_required:
                d["req"] += 1
                d["ok"] += 1 if t.on_time_abort else 0
        for scn in sorted(per):
            d = per[scn]
            eng_rate = d["eng"] / d["dec"] if d["dec"] else 0.0
            uns_rate = d["uns"] / d["eng"] if d["eng"] else 0.0
            fls_rate = d["fls"] / d["eng"] if d["eng"] else 0.0
            ab = d["ok"] / d["req"] if d["req"] else float("nan")
            print(f"  {scn:28} {label}  {eng_rate:.4f} ({d['eng']:6})  "
                  f"{uns_rate:.5f} ({d['uns']:4}/{d['uns_trials']:4} tr)  "
                  f"{ab:.4f} ({d['ok']}/{d['req']})")
            rows.append(f"{scn},{label},{d['n']},{d['dec']},{eng_rate:.6f},{d['eng']},"
                        f"{uns_rate:.6f},{d['uns']},{d['uns_trials']},"
                        f"{fls_rate:.6f},{d['fls']},{d['fls_trials']},"
                        f"{ab:.4f},{d['ok']},{d['req']}")
    write_csv("w14_extended_table.csv",
              "scenario,policy,trials,decisions,engage_rate,engage_n,"
              "unsafe_per_attempt,unsafe_n,unsafe_trials,"
              "false_per_attempt,false_n,false_trials,"
              "abort_success,abort_ok,abort_required", rows)


# --------------------------------------------------------------------------- #
# W15 — per-sigma safety table + pinned-noise-model curve (mock review M2/M7/R3-4)
# --------------------------------------------------------------------------- #
W15_PINNED = 0.2


def w15_cells(seeds: list) -> dict:
    # (a) reuses W5's cells; only (b) needs execution.
    return {"w15": [dict(PROPOSED, seeds=seeds, observation_noise_sigma=sg,
                         kalman_measurement_sigma=W15_PINNED)
                    for sg in W1_SIGMAS]}


def _w15_counts(trials: list) -> dict:
    """Event counts plus the number of distinct trials each event occurred in.

    The gate is memoryless, so one trial can contribute up to K attempts; an
    absolute count therefore overstates the independent sample unless the trial
    incidence is given next to it.
    """
    return dict(
        trials=len(trials),
        eng=sum(t.engage_count for t in trials),
        uns=sum(t.unsafe_engagement_count for t in trials),
        fls=sum(t.false_engagement_count for t in trials),
        opp=sum(t.safe_opportunity_count for t in trials),
        missed=sum(t.missed_safe_opportunity_count for t in trials),
        uns_tr=sum(1 for t in trials if t.unsafe_engagement_count),
        fls_tr=sum(1 for t in trials if t.false_engagement_count),
        req=sum(1 for t in trials if t.abort_required),
        ok=sum(1 for t in trials if t.abort_required and t.on_time_abort),
    )


def w15_analyze(seeds: list) -> None:
    rws = make_rws_aggregator()

    print("")
    print("W15a  per-sigma safety table (N=%d, core set)" % len(seeds))
    print("  sigma  pol      RWS  SafeCap     MSO   unsafe n/trials   false n/trials  attempts")
    rows = []
    for sg in W1_SIGMAS:
        for label, spec in (("A", heuristic_default(seeds, sg)),
                            ("D", dict(PROPOSED, seeds=seeds, observation_noise_sigma=sg))):
            trials = list(load_cell(cell_key(spec)).values())
            d = _w15_counts(trials)
            cap = aggregate_safe_capture(trials)
            mso = d["missed"] / d["opp"] if d["opp"] else 0.0
            print(f"  {sg:5}  {label}   {rws(trials):6.4f}   {cap:.4f}  {mso:.4f}   "
                  f"{d['uns']:5}/{d['uns_tr']:<5}   {d['fls']:5}/{d['fls_tr']:<5}  {d['eng']:8}")
            rows.append(f"{sg},{label},{rws(trials):.4f},{cap:.6f},{mso:.6f},"
                        f"{d['uns']},{d['uns_tr']},{d['fls']},{d['fls_tr']},"
                        f"{d['eng']},{d['trials']},{d['opp']},{d['ok']},{d['req']}")
    write_csv("w15_sigma_table.csv",
              "sigma,policy,rws,safe_capture,missed_safe_rate,unsafe_n,unsafe_trials,"
              "false_n,false_trials,engage_attempts,trials,safe_opportunities,"
              "abort_ok,abort_required", rows)

    print("")
    print(f"W15b  filter pinned at sigma={W15_PINNED} (N={len(seeds)})")
    print("  true sigma      RWS  SafeCap   unsafe/attempt   unsafe n/trials")
    rows = []
    for sg in W1_SIGMAS:
        trials = list(load_cell(cell_key(dict(
            PROPOSED, seeds=seeds, observation_noise_sigma=sg,
            kalman_measurement_sigma=W15_PINNED))).values())
        d = _w15_counts(trials)
        upa = d["uns"] / d["eng"] if d["eng"] else 0.0
        print(f"  {sg:9}   {rws(trials):6.4f}   {aggregate_safe_capture(trials):.4f}   "
              f"{upa:.6f}         {d['uns']:5}/{d['uns_tr']}")
        rows.append(f"{sg},{rws(trials):.4f},{aggregate_safe_capture(trials):.6f},"
                    f"{upa:.6f},{d['uns']},{d['uns_tr']},{d['eng']}")
    write_csv("w15_pinned_sigma.csv",
              "true_sigma,rws,safe_capture,unsafe_per_attempt,unsafe_n,unsafe_trials,"
              "engage_attempts", rows)


# --------------------------------------------------------------------------- #
# W16 — best-tuned baseline vs the gate at the headline sample size
# --------------------------------------------------------------------------- #
# Fig. 2 of the article plots this comparison at N = 10,000, and Sections V-E
# and VI-B quote its paired intervals. The cells come from W1 (the heuristic at
# the setting its own threshold search returns, theta_abort = theta_track = 0)
# and W5 (the calibrated gate), so nothing is executed here.


def w16_cells(seeds: list) -> dict:
    # The heuristic at the setting its own threshold search returns. Built here
    # rather than read from W1 so that `w16 --seeds 10000` is self-contained:
    # W1's grid is 300 cells and only these six are needed.
    return {"w16": [heuristic_cell(seeds, sg, 0.0, 0.0, 0.7) for sg in W1_SIGMAS]}


def w16_analyze(seeds: list) -> None:
    rws = make_rws_aggregator()
    print("")
    print(f"W16  best-tuned heuristic vs proposed, N={len(seeds)}")
    print(f"  {'sigma':>6} | {'heur best':>9} | {'proposed':>9} | {'diff':>8} |"
          f" {'95% CI':>20} | {'p':>6}")
    rows = []
    for sg in W1_SIGMAS:
        try:
            h = load_cell(cell_key(heuristic_cell(seeds, sg, 0.0, 0.0, 0.7)))
            d = load_cell(cell_key(dict(PROPOSED, seeds=seeds,
                                        observation_noise_sigma=sg)))
        except FileNotFoundError:
            print(f"  {sg:>6} |  (run w1 and w5 at this seed count first)")
            continue
        a, b = aligned(h, d)
        res = paired_bootstrap_diff(b, a, rws, n_boot=400, seed=1)
        print(f"  {sg:>6} | {rws(a):9.4f} | {rws(b):9.4f} | "
              f"{res['point_diff']:+8.4f} | "
              f"[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}] | "
              f"{res['p_value_one_sided']:6.4f}")
        rows.append(f"{sg},{rws(a):.4f},{rws(b):.4f},{res['point_diff']:.4f},"
                    f"{res['ci_lo']:.4f},{res['ci_hi']:.4f},"
                    f"{res['p_value_one_sided']:.4f}")
    if rows:
        write_csv("w16_best_tuned_ci.csv",
                  "sigma,heuristic_best_rws,proposed_rws,diff,ci_lo,ci_hi,p_value",
                  rows)


# --------------------------------------------------------------------------- #
# W17 — S4 calibration diagnostic + observation-model robustness (3rd review)
# --------------------------------------------------------------------------- #
# The three W17 outputs cannot come from one invocation: (a) and (b) follow the
# --seeds argument, while (c) is an N = 1000 study and builds its cells only
# there. Run `w17 --seeds 1000` first and `w17 --seeds 10000` second; the second
# leaves (c) alone.
W17_SIGMAS = [0.1, 0.2, 0.3, 0.5]
W17_VEL_SIGMA = 1.0          # matches kalman_init_velocity_var = 1.0
W17_BINS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.20),
            (0.20, 0.50), (0.50, 0.90), (0.90, 1.0000001)]
S4_NAME = "S4_RISK_ZONE_PROXIMITY"


def w17_cells(seeds: list) -> dict:
    """(c) only: the reliability diagnostic needs decision records, not cells.

    The velocity-noise check is an N = 1000 study (Section VI-D), so no cells are
    built at any other seed count. Without this, `w17 --seeds 10000` would build
    twelve cells the article does not use and then overwrite the table with
    figures the text does not quote.
    """
    if len(seeds) != 1000:
        return {"w17": []}
    cells = []
    for sg in W1_SIGMAS:
        cells.append(dict(BASE, seeds=seeds, observation_noise_sigma=sg,
                          observation_velocity_noise_sigma=W17_VEL_SIGMA))
        cells.append(dict(PROPOSED, seeds=seeds, observation_noise_sigma=sg,
                          observation_velocity_noise_sigma=W17_VEL_SIGMA))
    return {"w17": cells}


W17_CHUNK = 1000


def _s4_reliability(seeds: list, sigma: float):
    """Bin S4 steps by R_env and count realized next-step intrusions.

    Accumulated over chunks of seeds: the full core set at N = 10,000 is 1.2 M
    decision records and holding them at once exhausts memory. Seeds are
    independent and the noise seed is derived per (seed, scenario index), so the
    chunked totals equal the single-pass ones.
    """
    import math
    from cuas_sim.policies import DecisionAction

    b = PROPOSED["risk_keepout_buffer"]
    tab = {k: [0, 0, 0.0, 0, 0] for k in W17_BINS}   # n, inside, sum r, eng, eng_inside
    bound = {"n": 0, "sum_r": 0.0, "intrusions": 0, "trials": set(),
             "clear": [], "inside": 0,
             "r_first": [], "r_last": [], "n_steps": 0, "n_abort": 0}

    for lo in range(0, len(seeds), W17_CHUNK):
        _s4_reliability_chunk(seeds[lo:lo + W17_CHUNK], sigma, tab, bound,
                              b, math, DecisionAction)
    return tab, bound


def _s4_reliability_chunk(seeds, sigma, tab, bound, b, math, DecisionAction):
    """One chunk of the reliability pass; folds its counts into tab and bound."""
    cfg = MonteCarloConfig(**dict(PROPOSED, seeds=seeds,
                                  observation_noise_sigma=sigma))
    result = MonteCarloRunner(cfg).run()

    truth, rows = {}, []
    for rec in result.records:
        if rec.policy_name != POLICY or rec.metadata.get("scenario_type") != S4_NAME:
            continue
        seed = rec.metadata.get("seed")
        snap = rec.scenario_metadata_snapshot or {}
        c, rad = snap.get("risk_zone_center"), snap.get("risk_zone_radius")
        if c is None or rad is None or rec.true_target_state is None:
            continue
        truth[(seed, rec.step_index)] = (rec.true_target_state, c, rad)
        rows.append(rec)

    last_step = max((r.step_index for r in rows), default=0)
    for rec in rows:
        bound["n_steps"] += 1
        if rec.action == DecisionAction.ABORT:
            bound["n_abort"] += 1
        if rec.step_index == 0:
            bound["r_first"].append(rec.risk_score)
        elif rec.step_index == last_step:
            bound["r_last"].append(rec.risk_score)
        if rec.action == DecisionAction.ENGAGE:
            here = truth.get((rec.metadata.get("seed"), rec.step_index))
            if here is not None:
                st_h, c_h, rad_h = here
                gap = math.hypot(st_h.x - c_h[0], st_h.y - c_h[1]) - rad_h
                if gap <= b:
                    bound["clear"].append(gap)
                    if gap <= 0:
                        bound["inside"] += 1
        nxt = truth.get((rec.metadata.get("seed"), rec.step_index + 1))
        if nxt is None:
            continue
        st, c, rad = nxt
        inside = math.hypot(st.x - c[0], st.y - c[1]) - rad <= b
        r = rec.risk_score
        for k in W17_BINS:
            if k[0] <= r < k[1]:
                d = tab[k]
                d[0] += 1
                d[1] += 1 if inside else 0
                d[2] += r
                if rec.action == DecisionAction.ENGAGE:
                    d[3] += 1
                    d[4] += 1 if inside else 0
                break
        if rec.action == DecisionAction.ENGAGE:
            bound["n"] += 1
            bound["sum_r"] += r
            bound["intrusions"] += 1 if inside else 0
            if inside:
                bound["trials"].add(rec.metadata.get("seed"))
    del result, truth, rows


def w17_analyze(seeds: list) -> None:
    rws = make_rws_aggregator()

    # --- (a) reliability of R_env on S4 ------------------------------------
    # Runs at whatever seed count is passed, because it has to match the sample
    # the rest of the section reports. _s4_reliability accumulates over chunks
    # of seeds, so N = 10,000 is a question of time (about twenty minutes for
    # the four noise levels) rather than of memory.
    diag_seeds = seeds
    print("")
    print(f"W17a  reliability of R_env on S4 (N={len(diag_seeds)})")
    print("  Predicted is the mean R_env in the bin; realized is how often the")
    print("  target is inside the keep-out disk at the next step. All S4 steps,")
    print("  not only the engaged ones.")
    rows, bound_rows = [], []
    for sg in W17_SIGMAS:
        tab, bound = _s4_reliability(diag_seeds, sg)
        print(f"    calibration bound: {bound['n']} engagements with a following "
              f"step, sum of R_env {bound['sum_r']:.1f}, "
              f"realized intrusions {bound['intrusions']} "
              f"({bound['intrusions'] / max(bound['n'], 1):.1%}) in "
              f"{len(bound['trials'])} trials "
              f"-> {bound['intrusions'] / max(bound['sum_r'], 1e-9):.1f}x the bound")
        cl = bound["clear"]
        print(f"    unsafe engagements {len(cl)}, of which inside the zone "
              f"{bound['inside']}; clearance min "
              f"{min(cl) if cl else float('nan'):.4f} median "
              f"{statistics.median(cl) if cl else float('nan'):.4f}")
        print(f"    aborts {bound['n_abort'] / max(bound['n_steps'], 1):.4f} of "
              f"decision steps; median R_env first step "
              f"{statistics.median(bound['r_first']):.4f}, last step "
              f"{statistics.median(bound['r_last']):.4f}")
        bound_rows.append(
            f"{sg},{bound['n']},{bound['sum_r']:.4f},{bound['intrusions']},"
            f"{len(bound['trials'])},"
            f"{bound['intrusions'] / max(bound['sum_r'], 1e-9):.4f},"
            f"{len(cl)},{bound['inside']},"
            f"{(min(cl) if cl else float('nan')):.4f},"
            f"{(statistics.median(cl) if cl else float('nan')):.4f},"
            f"{bound['n_abort'] / max(bound['n_steps'], 1):.4f},"
            f"{statistics.median(bound['r_first']):.4f},"
            f"{statistics.median(bound['r_last']):.4f}")
        print(f"  sigma={sg}")
        print(f"    {'R_env bin':>14} | {'n':>6} | {'predicted':>9} | "
              f"{'realized':>8} | {'engaged':>7} | {'eng.inside':>10}")
        for k in W17_BINS:
            n, ins, rsum, en, ei = tab[k]
            if not n:
                continue
            print(f"    [{k[0]:.2f},{k[1]:.2f})".rjust(14) +
                  f" | {n:6} | {rsum/n:9.3f} | {ins/n:8.3f} | {en:7} | {ei:10}")
            rows.append(f"{sg},{k[0]},{min(k[1], 1.0)},{n},{rsum/n:.4f},"
                        f"{ins/n:.4f},{ins},{en},{ei}")
    write_csv("w17_s4_reliability.csv",
              "sigma,bin_lo,bin_hi,n_steps,mean_r_env,realized_intrusion_rate,"
              "n_intrusions,n_engaged,n_engaged_intruding", rows)
    write_csv("w17_s4_bound.csv",
              "sigma,engagements_with_next_step,sum_r_env,realized_intrusions,"
              "trials_with_intrusion,ratio_to_bound,unsafe_engagements,"
              "unsafe_inside_zone,min_clearance,median_clearance,abort_share,"
              "median_r_env_first_step,median_r_env_last_step", bound_rows)

    # --- (b) per-sigma S4 counts for all three configurations ---------------
    print("")
    print(f"W17b  S4 engagements and unsafe engagements per configuration "
          f"(N={len(seeds)})")
    print(f"    {'sigma':>6} | {'configuration':<22} | {'engage':>7} | "
          f"{'unsafe':>6} | {'trials':>6} | {'unsafe/eng':>10}")
    rows = []
    configs = [
        ("heuristic published", lambda sg: heuristic_default(seeds, sg)),
        ("heuristic best", lambda sg: heuristic_cell(seeds, sg, 0.0, 0.0, 0.7)),
        ("proposed matched", lambda sg: dict(PROPOSED, seeds=seeds,
                                             observation_noise_sigma=sg)),
    ]
    for sg in W1_SIGMAS:
        for name, build in configs:
            try:
                cell = load_cell(cell_key(build(sg)))
            except FileNotFoundError:
                print(f"    {sg:>6} | {name:<22} |  (not cached)")
                continue
            eng = uns = tr = 0
            for t in cell.values():
                if t.scenario_type != S4_NAME:
                    continue
                eng += t.engage_count
                uns += t.unsafe_engagement_count
                tr += 1 if t.unsafe_engagement_count else 0
            frac = uns / eng if eng else 0.0
            print(f"    {sg:>6} | {name:<22} | {eng:7} | {uns:6} | {tr:6} | "
                  f"{frac:9.1%}")
            rows.append(f"{sg},{name.replace(' ', '_')},{eng},{uns},{tr},{frac:.6f}")
    write_csv("w17_s4_by_sigma.csv",
              "sigma,configuration,s4_engagements,s4_unsafe,s4_unsafe_trials,"
              "unsafe_per_engagement", rows)

    # --- (c) observation-model robustness -----------------------------------
    print("")
    print(f"W17c  does the comparison survive a noisy initial velocity report?")
    print(f"    (observation_velocity_noise_sigma = {W17_VEL_SIGMA}; the default"
          f" is 0, i.e. the true velocity)")
    print(f"    {'sigma':>6} | {'A base':>7} {'A noisy':>8} | "
          f"{'D base':>7} {'D noisy':>8} | {'D-A base':>9} {'D-A noisy':>9}")
    rows = []
    for sg in W1_SIGMAS:
        try:
            a0 = load_cell(cell_key(heuristic_default(seeds, sg)))
            d0 = load_cell(cell_key(dict(PROPOSED, seeds=seeds,
                                         observation_noise_sigma=sg)))
            a1 = load_cell(cell_key(dict(BASE, seeds=seeds,
                                         observation_noise_sigma=sg,
                                         observation_velocity_noise_sigma=W17_VEL_SIGMA)))
            d1 = load_cell(cell_key(dict(PROPOSED, seeds=seeds,
                                         observation_noise_sigma=sg,
                                         observation_velocity_noise_sigma=W17_VEL_SIGMA)))
        except FileNotFoundError:
            print(f"    {sg:>6} |  (run w17 without --analyze-only first)")
            continue
        va, vd = rws(list(a0.values())), rws(list(d0.values()))
        na, nd = rws(list(a1.values())), rws(list(d1.values()))
        print(f"    {sg:>6} | {va:7.4f} {na:8.4f} | {vd:7.4f} {nd:8.4f} | "
              f"{vd - va:+9.4f} {nd - na:+9.4f}")
        rows.append(f"{sg},{va:.4f},{na:.4f},{vd:.4f},{nd:.4f},"
                    f"{vd - va:.4f},{nd - na:.4f}")
    if rows:
        write_csv("w17_velocity_noise.csv",
                  "sigma,heuristic_base,heuristic_noisy_v,proposed_base,"
                  "proposed_noisy_v,diff_base,diff_noisy_v", rows)
    else:
        # Nothing cached at this seed count: leave the existing file alone
        # rather than replacing it with a header.
        print("    (no cells at this seed count; w17_velocity_noise.csv left as is)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=["w1", "w2", "w3", "w4", "w5", "w6", "w8", "w9", "w10", "w11", "w12", "w13", "w14", "w15", "w16", "w17", "all"],
                   help="review work item to run")
    p.add_argument("--seeds", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--analyze-only", action="store_true",
                   help="skip execution; analyze cached cells only")
    args = p.parse_args(argv)

    seeds = list(range(1, args.seeds + 1))
    builders = {
        "w1": (w1_cells, w1_analyze),
        "w2": (w2_cells, w2_analyze),
        "w3": (w3_cells, w3_analyze),
        "w4": (w4_cells, w4_analyze),
        "w5": (w5_cells, w5_analyze),
        "w6": (w6_cells, w6_analyze),
        "w8": (w8_cells, w8_analyze),
        "w9": (w9_cells, w9_analyze),
        "w10": (w10_cells, w10_analyze),
        "w11": (w11_cells, w11_analyze),
        "w12": (w12_cells, w12_analyze),
        "w13": (w13_cells, w13_analyze),
        "w14": (w14_cells, w14_analyze),
        "w15": (w15_cells, w15_analyze),
        "w16": (w16_cells, w16_analyze),
        "w17": (w17_cells, w17_analyze),
    }
    tasks = ["w1", "w2", "w3", "w4", "w5", "w6", "w8", "w9", "w10", "w11", "w12", "w13", "w14", "w15", "w16", "w17"] if args.task == "all" else [args.task]

    for task in tasks:
        build, analyze = builders[task]
        if not args.analyze_only:
            for label, cells in build(seeds).items():
                run_grid(cells, args.jobs, f"{task}:{label}")
        # w1's oracle refinement runs a second stage chosen from stage-1 results,
        # so the analysis step needs to be able to launch cells too.
        analyze(seeds, args.jobs) if task in ("w1", "w8", "w11", "w12") else analyze(seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
