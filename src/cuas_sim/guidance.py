"""Closed-loop proportional-navigation interceptor — a *validation* layer.

Why this exists
---------------
Reviewers 1 and 2 independently objected that the interceptor is a stationary
launch platform, so interceptability ``I`` and abort feasibility ``A_abort`` are
capability assessments rather than flown trajectories.

The fix is deliberately NOT to feed guidance back into the gate. Making miss
distance a gate input would couple the decision layer to a guidance law, which
is the separation the paper is built on, and it would contaminate the ablation.
Instead the gate decides exactly as before, and whenever it commits to contact
the interceptor is actually flown here, against the true target. That turns the
open question into a measurable one: **does the capture-geometry signal predict
what really happens once the interceptor has bounded acceleration?**

Model
-----
Planar pure proportional navigation, ``a_cmd = N' * V_c * lambda_dot`` applied
normal to the line of sight, with commanded acceleration clipped to ``a_max``
and speed clipped to ``V_I`` — the same kinematic parameters the paper already
uses for the capability signals (Table 4), so the comparison is like-for-like
and introduces no new tuning.

The target flies constant-velocity during the engagement, which is the same
assumption the intercept-triangle signal makes; where a scenario schedules a
known maneuver, it is replayed from the scenario metadata so the maneuvering
cases are not quietly flattered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

from .types import State2D


@dataclass
class PNOutcome:
    """What the flown engagement actually produced."""
    captured: bool
    miss_distance: float     # closest approach achieved (normalized units)
    time_to_go: float        # time at closest approach (s)
    flight_time: float       # total time flown (s)
    launch_speed: float


@dataclass
class PNParams:
    """Defaults mirror Table 4 of the manuscript; nothing here is newly tuned."""
    nav_constant: float = 4.0          # N', the textbook 3-5 range
    max_speed: float = 2.0             # V_I
    max_accel: float = 4.0             # a_max
    reaction_delay: float = 0.2        # tau, before guidance engages
    engagement_horizon: float = 20.0   # T_engage
    capture_radius: float = 0.3        # contact/net radius (~3 m at 10 m/unit)
    time_step: float = 0.02            # finer than the decision step for fidelity


def intercept_lead_velocity(
    rel_pos: Tuple[float, float],
    target_vel: Tuple[float, float],
    speed: float,
) -> Optional[Tuple[float, float]]:
    """Launch velocity from the intercept triangle — the ideal the paper assumes.

    Solves the same quadratic as (10) for the capture time and aims at the
    predicted meeting point. Returns None when no positive root exists, i.e.
    when the geometry admits no interception at this speed.
    """
    rx, ry = rel_pos
    vx, vy = target_vel
    a = vx * vx + vy * vy - speed * speed
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        t = -c / b
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        root = math.sqrt(disc)
        roots = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
        positive = [t for t in roots if t > 1e-9]
        if not positive:
            return None
        t = min(positive)
    # Aim at where the target will be at t.
    mx, my = rx + vx * t, ry + vy * t
    norm = math.hypot(mx, my)
    if norm < 1e-12:
        return None
    return (speed * mx / norm, speed * my / norm)


def simulate_pn_intercept(
    target_state: State2D,
    interceptor_pos: Tuple[float, float],
    params: PNParams = PNParams(),
    maneuver: Optional[Tuple[float, float, float]] = None,
) -> PNOutcome:
    """Fly the interceptor against the true target and report what happened.

    ``maneuver`` is ``(time_from_now, dvx, dvy)`` for a scenario-scheduled
    velocity change occurring during the flight; None means constant velocity.
    """
    dt = params.time_step
    px, py = interceptor_pos
    tx, ty = target_state.x, target_state.y
    tvx, tvy = target_state.vx, target_state.vy

    launch = intercept_lead_velocity((tx - px, ty - py), (tvx, tvy), params.max_speed)
    if launch is None:
        # No intercept triangle solution: the platform still launches, aiming
        # straight down the line of sight. Scoring this as "no attempt" would
        # hide exactly the cases the interceptability signal is meant to catch.
        d = math.hypot(tx - px, ty - py)
        launch = ((tx - px) / d * params.max_speed, (ty - py) / d * params.max_speed) \
            if d > 1e-12 else (params.max_speed, 0.0)
    ivx, ivy = launch

    best_range = math.hypot(tx - px, ty - py)
    best_time = 0.0
    t = 0.0
    maneuver_done = maneuver is None

    while t < params.engagement_horizon:
        rx, ry = tx - px, ty - py
        rng = math.hypot(rx, ry)
        if rng < best_range:
            best_range, best_time = rng, t
        if rng <= params.capture_radius:
            return PNOutcome(True, rng, t, t, params.max_speed)

        if t >= params.reaction_delay:
            vrx, vry = tvx - ivx, tvy - ivy
            r2 = rx * rx + ry * ry
            if r2 > 1e-12:
                # LOS rate (z-component of r x v_r over |r|^2) and closing speed.
                lam_dot = (rx * vry - ry * vrx) / r2
                v_c = -(rx * vrx + ry * vry) / math.sqrt(r2)
                a_mag = params.nav_constant * v_c * lam_dot
                ux, uy = rx / math.sqrt(r2), ry / math.sqrt(r2)
                # Normal to the LOS, rotated +90 degrees; with this convention a
                # positive lambda_dot commands acceleration that turns the
                # velocity the same way the LOS is drifting, which is what nulls
                # the drift (test_guidance checks a crossing geometry).
                ax, ay = a_mag * -uy, a_mag * ux
                mag = math.hypot(ax, ay)
                if mag > params.max_accel:
                    ax, ay = ax / mag * params.max_accel, ay / mag * params.max_accel
                ivx += ax * dt
                ivy += ay * dt
                speed = math.hypot(ivx, ivy)
                if speed > params.max_speed:
                    ivx, ivy = ivx / speed * params.max_speed, ivy / speed * params.max_speed

        px += ivx * dt
        py += ivy * dt
        tx += tvx * dt
        ty += tvy * dt
        t += dt
        if not maneuver_done and t >= maneuver[0]:
            tvx += maneuver[1]
            tvy += maneuver[2]
            maneuver_done = True

    return PNOutcome(False, best_range, best_time, t, params.max_speed)


def simulate_abort(params: PNParams = PNParams()) -> Tuple[float, float]:
    """Arrest-and-reverse from launch speed: (time to a standstill, distance run).

    The manuscript predicts ``t_req = tau + V_I / a_max``. Flying it out is how
    that prediction gets checked rather than assumed; the distance is what says
    whether the break-off stays clear of the keep-out disk.
    """
    dt = params.time_step
    t = 0.0
    travelled = 0.0
    speed = params.max_speed
    while t < params.reaction_delay:          # coasting during the delay
        travelled += speed * dt
        t += dt
    while speed > 0.0:
        speed = max(0.0, speed - params.max_accel * dt)
        travelled += speed * dt
        t += dt
    return t, travelled


def auc(scores_pos: list, scores_neg: list) -> float:
    """Rank-based AUC (Mann-Whitney U), ties counted as half. Stdlib only.

    0.5 is chance, so anything near it says the signal does not separate the
    outcome it is supposed to predict.
    """
    if not scores_pos or not scores_neg:
        return float("nan")
    merged = sorted([(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg])
    ranks = {}
    i = 0
    rank_sum_pos = 0.0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if merged[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j + 1
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)
