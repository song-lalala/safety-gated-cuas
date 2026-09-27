"""Bootstrap statistics helpers (M7).

Standard-library-only (no numpy/scipy). Designed for resampling per-trial
metric units to produce confidence intervals on aggregate metrics, plus a
paired bootstrap for differences between policies that share the same trials
(seed, scenario).
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear interpolation percentile for an already-sorted list. p in [0,1]."""
    if not sorted_values:
        return float("nan")
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI on the mean of a 1-D list of values.

    Returns (point_estimate, ci_lo, ci_hi). point_estimate is the actual mean
    of `values`; ci_lo/ci_hi are the bootstrap quantiles of the resampled
    means. If `values` is empty, returns (nan, nan, nan).
    """
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    n = len(values)
    point = sum(values) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_p = (1.0 - ci) / 2.0
    hi_p = 1.0 - lo_p
    return point, _percentile(means, lo_p), _percentile(means, hi_p)


def bootstrap_aggregate_ci(
    trials: Sequence,
    aggregate_fn: Callable[[Sequence], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI for an aggregate function over a list of trial objects.

    Each resample draws `len(trials)` trial objects with replacement; the
    aggregate_fn is recomputed on the resample. Useful when the aggregate is
    not just the mean of a single per-trial scalar (e.g., a ratio of summed
    quantities like unsafe_count / engage_count).
    """
    if not trials:
        return (float("nan"), float("nan"), float("nan"))
    n = len(trials)
    point = aggregate_fn(trials)
    rng = random.Random(seed)
    boot_vals = []
    for _ in range(n_boot):
        sample = [trials[rng.randrange(n)] for _ in range(n)]
        boot_vals.append(aggregate_fn(sample))
    boot_vals.sort()
    lo_p = (1.0 - ci) / 2.0
    hi_p = 1.0 - lo_p
    return point, _percentile(boot_vals, lo_p), _percentile(boot_vals, hi_p)


def paired_bootstrap_diff(
    trials_a: Sequence,
    trials_b: Sequence,
    aggregate_fn: Callable[[Sequence], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap CI for `aggregate_fn(A) - aggregate_fn(B)`.

    `trials_a[i]` and `trials_b[i]` must come from the same trial id (same
    seed, scenario) — caller must align them. Each bootstrap resample uses
    the same row indices for both A and B so the pairing is preserved. The
    one-sided p-value approximates the fraction of bootstrap differences ≤ 0
    (i.e., probability A is no better than B by chance under resampling).
    """
    if len(trials_a) != len(trials_b):
        raise ValueError("trials_a and trials_b must have equal length and be aligned")
    n = len(trials_a)
    if n == 0:
        return {"point_diff": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_value_one_sided": float("nan")}

    point_diff = aggregate_fn(trials_a) - aggregate_fn(trials_b)
    rng = random.Random(seed)
    diffs = []
    n_le_zero = 0
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample_a = [trials_a[i] for i in idxs]
        sample_b = [trials_b[i] for i in idxs]
        d = aggregate_fn(sample_a) - aggregate_fn(sample_b)
        diffs.append(d)
        if d <= 0:
            n_le_zero += 1
    diffs.sort()
    lo_p = (1.0 - ci) / 2.0
    hi_p = 1.0 - lo_p
    return {
        "point_diff": point_diff,
        "ci_lo": _percentile(diffs, lo_p),
        "ci_hi": _percentile(diffs, hi_p),
        # one-sided p-value: how often the resampled diff ≤ 0 when point > 0
        # (or vice versa). Standard interpretation: if 0 is outside CI, the
        # difference is significant at (1-ci) level.
        "p_value_one_sided": n_le_zero / n_boot if point_diff > 0 else (n_boot - n_le_zero) / n_boot,
    }


# ---- aggregate functions for use with the bootstrap helpers ----

def aggregate_safe_capture(trials) -> float:
    """safe_engagement_count / safe_opportunity_count, summed across trials."""
    so = sum(t.safe_opportunity_count for t in trials)
    if so <= 0:
        return 0.0
    captured = so - sum(t.missed_safe_opportunity_count for t in trials)
    return captured / so


def aggregate_unsafe_per_attempt(trials) -> float:
    eng = sum(t.engage_count for t in trials)
    if eng <= 0:
        return 0.0
    return sum(t.unsafe_engagement_count for t in trials) / eng


def aggregate_false_per_attempt(trials) -> float:
    eng = sum(t.engage_count for t in trials)
    if eng <= 0:
        return 0.0
    return sum(t.false_engagement_count for t in trials) / eng


def aggregate_abort_success(trials) -> float:
    req = sum(1 for t in trials if t.abort_required)
    if req <= 0:
        return 0.0
    return sum(1 for t in trials if t.abort_required and t.on_time_abort) / req


def aggregate_late_abort(trials) -> float:
    req = sum(1 for t in trials if t.abort_required)
    if req <= 0:
        return 0.0
    return sum(1 for t in trials if t.abort_required and t.late_abort) / req


def aggregate_risk_weighted_score(
    trials,
    weight_safe_capture: float = 1.0,
    weight_abort_success: float = 0.5,
    weight_unsafe: float = 2.0,
    weight_false: float = 2.0,
    weight_late: float = 1.0,
) -> float:
    return (
        weight_safe_capture * aggregate_safe_capture(trials)
        + weight_abort_success * aggregate_abort_success(trials)
        - weight_unsafe * aggregate_unsafe_per_attempt(trials)
        - weight_false * aggregate_false_per_attempt(trials)
        - weight_late * aggregate_late_abort(trials)
    )


def make_rws_aggregator(
    weight_safe_capture: float = 1.0,
    weight_abort_success: float = 0.5,
    weight_unsafe: float = 2.0,
    weight_false: float = 2.0,
    weight_late: float = 1.0,
) -> Callable:
    """Return an aggregator with weights baked in (for use with bootstrap)."""
    def _agg(trials):
        return aggregate_risk_weighted_score(
            trials,
            weight_safe_capture=weight_safe_capture,
            weight_abort_success=weight_abort_success,
            weight_unsafe=weight_unsafe,
            weight_false=weight_false,
            weight_late=weight_late,
        )
    return _agg
