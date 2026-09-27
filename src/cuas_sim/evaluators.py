import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .scenario import Scenario
from .estimation import TargetEstimate
from .prediction import TrajectoryPrediction


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def gaussian_disk_probability(
    mu_x: float, mu_y: float, sigma_x: float, sigma_y: float,
    cx: float, cy: float, R: float, n: int = 64, rule: str = "simpson",
) -> float:
    """Probability that a point drawn from an independent-axis Gaussian
    ``N([mu_x, mu_y], diag(sigma_x², sigma_y²))`` falls inside the disk of
    radius ``R`` centered at ``(cx, cy)``.

    This is the journal's "intrusion probability" — the exact, uncertainty-aware
    replacement for the linear effective-radius risk score. Computed by 1D
    Simpson integration over x with the analytic normal CDF (math.erf) for the
    y-slice, so it is pure-Python and handles anisotropic covariance exactly:

        P = ∫ f_x(x) · [Φ_y(cy + h(x)) − Φ_y(cy − h(x))] dx,
        h(x) = sqrt(R² − (x − cx)²) over the disk's x-extent.

    ``rule`` selects the quadrature:

    ``"simpson"`` (default, as submitted) integrates in x. h(x) has an infinite
    derivative at the disk edges x = cx ± R, so Simpson converges at O(h^1.5)
    rather than its nominal O(h^4); the measured max absolute error at n=64 is
    ~2e-3, i.e. a few percent of a 0.05 risk threshold.

    ``"trig"`` substitutes x = cx + R·sin t, which cancels the edge singularity
    exactly (dx = R cos t dt and h = R cos t). The integrand becomes analytic and
    the same node count reaches machine precision — ~4e-16 measured against the
    non-central chi-square reference. Same cost, 13 orders of magnitude better.
    The default is left unchanged so previously reported results reproduce; see
    the review response for the measured decision-level impact of switching.
    """
    SIG_MIN = 1e-9
    # Near point-mass: degenerate to in/out test.
    if sigma_x <= 1e-6 and sigma_y <= 1e-6:
        return 1.0 if (mu_x - cx) ** 2 + (mu_y - cy) ** 2 <= R * R else 0.0
    sx = max(sigma_x, SIG_MIN)
    sy = max(sigma_y, SIG_MIN)
    # Integrate over the overlap of the disk x-extent and the x-mass support.
    lo = max(cx - R, mu_x - 6.0 * sx)
    hi = min(cx + R, mu_x + 6.0 * sx)
    if hi <= lo:
        return 0.0
    m = n if n % 2 == 0 else n + 1
    norm_x = 1.0 / (sx * math.sqrt(2.0 * math.pi))
    inv_sx = 1.0 / sx
    total = 0.0

    if rule == "trig":
        # x = cx + R sin t: dx = R cos t dt and the chord half-height h(x)
        # becomes R cos t, so the square-root edge behaviour cancels instead of
        # being resolved by brute force.
        #
        # The t-range is clipped to the same [lo, hi] the direct rule uses. A
        # full -pi/2..pi/2 sweep spreads the nodes over the DISK, which loses
        # badly when the Gaussian is much narrower than the disk: the spike then
        # falls between nodes. Clipping keeps the nodes on the probability mass
        # while still cancelling whichever disk edge lies inside the range.
        lo_t = math.asin(max(-1.0, min(1.0, (lo - cx) / R)))
        hi_t = math.asin(max(-1.0, min(1.0, (hi - cx) / R)))
        if hi_t <= lo_t:
            return 0.0
        h = (hi_t - lo_t) / m
        for i in range(m + 1):
            t = lo_t + i * h
            cos_t = math.cos(t)
            x = cx + R * math.sin(t)
            zx = (x - mu_x) * inv_sx
            fx = norm_x * math.exp(-0.5 * zx * zx)
            hy = R * cos_t
            g = _normal_cdf((cy + hy - mu_y) / sy) - _normal_cdf((cy - hy - mu_y) / sy)
            w = 1.0 if (i == 0 or i == m) else (4.0 if i % 2 else 2.0)
            total += w * fx * g * R * cos_t
        return max(0.0, min(1.0, total * h / 3.0))

    h = (hi - lo) / m
    for i in range(m + 1):
        x = lo + i * h
        zx = (x - mu_x) * inv_sx
        fx = norm_x * math.exp(-0.5 * zx * zx)
        dx = x - cx
        rad2 = R * R - dx * dx
        if rad2 <= 0.0:
            g = 0.0
        else:
            hy = math.sqrt(rad2)
            g = _normal_cdf((cy + hy - mu_y) / sy) - _normal_cdf((cy - hy - mu_y) / sy)
        val = fx * g
        if i == 0 or i == m:
            w = 1.0
        elif i % 2 == 1:
            w = 4.0
        else:
            w = 2.0
        total += w * val
    return max(0.0, min(1.0, total * h / 3.0))


@dataclass
class AbortFeasibilityAssessment:
    """A_abort score per docs §3.5: how much margin remains to safely abort.

    In docs the formula is `D_abort_min = V_I·τ + V_I²/(2·a_max)`. Since the
    interceptor is stationary in the current simulator (M1 limitation), this
    evaluator uses the time-equivalent proxy: how many simulation steps until
    the predicted target trajectory penetrates the risk zone. More steps ⇒
    more decision margin ⇒ higher A_abort score.

    For scenarios with no risk zone defined, abort is trivially feasible
    (score = 1.0).
    """
    step_index: int
    abort_feasibility_score: float
    is_feasible: bool
    steps_until_risk: Optional[int]
    metadata: Dict[str, object] = field(default_factory=dict)


class AbortFeasibilityEvaluator:
    def __init__(
        self,
        abort_horizon_steps: int = 10,
        risk_zone_safety_buffer: float = 0.5,
        feasibility_threshold: float = 0.5,
        a_mode: str = "time_proxy",
        interceptor_max_speed: float = 2.0,
        interceptor_max_accel: float = 4.0,
        interceptor_reaction_delay: float = 0.2,
    ):
        self.abort_horizon_steps = abort_horizon_steps
        self.risk_zone_safety_buffer = risk_zone_safety_buffer
        self.feasibility_threshold = feasibility_threshold
        # Journal Phase 3 (M1): "time_proxy" = N_remain/N_norm (baseline);
        # "dynamics" = t_avail / t_required, where t_required = τ + V_I/a_max is
        # the physical abort (stopping) time. Replaces the heuristic N_norm.
        self.a_mode = a_mode
        self.interceptor_max_speed = interceptor_max_speed
        self.interceptor_max_accel = interceptor_max_accel
        self.interceptor_reaction_delay = interceptor_reaction_delay

    def evaluate(
        self,
        scenario: Scenario,
        prediction: TrajectoryPrediction,
    ) -> AbortFeasibilityAssessment:
        if not prediction.is_valid or not prediction.predictions:
            return AbortFeasibilityAssessment(
                step_index=prediction.start_step_index,
                abort_feasibility_score=0.0,
                is_feasible=False,
                steps_until_risk=None,
                metadata={"reason": "invalid_prediction"},
            )

        if (
            "risk_zone_center" not in scenario.metadata
            or "risk_zone_radius" not in scenario.metadata
        ):
            return AbortFeasibilityAssessment(
                step_index=prediction.start_step_index,
                abort_feasibility_score=1.0,
                is_feasible=True,
                steps_until_risk=None,
                metadata={"reason": "no_risk_zone"},
            )

        zone_center = scenario.metadata["risk_zone_center"]
        zone_radius = scenario.metadata["risk_zone_radius"]
        unsafe_radius = zone_radius + self.risk_zone_safety_buffer

        steps_until_risk: Optional[int] = None
        for pred in prediction.predictions:
            d = math.hypot(
                pred.state.x - zone_center[0],
                pred.state.y - zone_center[1],
            )
            if d < unsafe_radius:
                steps_until_risk = pred.step_offset
                break

        meta: Dict[str, object] = {"unsafe_radius": unsafe_radius, "a_mode": self.a_mode}
        if steps_until_risk is None:
            # The predicted trajectory never enters the risk zone within the
            # horizon → there is "as much margin as the horizon affords"; treat
            # as fully feasible.
            score = 1.0
        elif self.a_mode == "dynamics":
            # Physical: A_abort = t_available / t_required, with
            # t_required = τ + V_I / a_max (reaction + stopping time).
            dt = scenario.config.time_step
            t_avail = steps_until_risk * dt
            t_required = (
                self.interceptor_reaction_delay
                + self.interceptor_max_speed / max(self.interceptor_max_accel, 1e-9)
            )
            score = max(0.0, min(1.0, t_avail / max(t_required, 1e-9)))
            meta.update({"t_avail": t_avail, "t_required": t_required})
        elif self.a_mode == "time_proxy":
            score = min(1.0, steps_until_risk / float(self.abort_horizon_steps))
        else:
            raise ValueError(
                f"Unknown a_mode: {self.a_mode!r} (expected 'time_proxy' or 'dynamics')"
            )

        return AbortFeasibilityAssessment(
            step_index=prediction.start_step_index,
            abort_feasibility_score=score,
            is_feasible=score >= self.feasibility_threshold,
            steps_until_risk=steps_until_risk,
            metadata=meta,
        )

@dataclass
class EnvironmentalRiskAssessment:
    step_index: int
    risk_score: float
    is_high_risk: bool
    min_distance_to_risk_zone: Optional[float]
    risk_zone_radius: Optional[float]
    metadata: Dict[str, object] = field(default_factory=dict)

class EnvironmentalRiskEvaluator:
    """Environmental risk evaluator with uncertainty buffer (M3 fix).

    In `linear` mode (the heuristic baseline) the effective unsafe radius is

        R_effective = R_zone + k_sigma * sigma_prediction + D_abort_min

    The previous implementation used only `R_zone` for the clearance, so the
    framework's "uncertainty-aware" claim was not actually present in the
    code. This evaluator now inflates the unsafe radius by the prediction
    uncertainty at the step where the predicted target trajectory is closest
    to the zone, plus an optional kinematic abort margin.

    Parameters
    ----------
    high_risk_threshold : float
        risk_score above which `is_high_risk` becomes True. Same semantics as before.
    k_sigma : float
        Weight on `prediction.uncertainty` at the minimum-distance prediction
        step. With k_sigma=0 the evaluator reproduces the original behaviour
        (pre-M3), which is useful for ablation.
    abort_margin : float
        Optional kinematic safety margin added to the unsafe radius. Defaults
        to 0; can be set per experiment to encode a notional `D_abort_min`.
    clearance_falloff : float
        Distance (in normalized units) over which the risk score linearly
        decays from 1.0 (at the unsafe boundary) to 0.0. Default 2.0 keeps
        prior numeric behaviour when `k_sigma=0`.
    """

    def __init__(
        self,
        high_risk_threshold: float = 0.7,
        k_sigma: float = 1.0,
        abort_margin: float = 0.0,
        clearance_falloff: float = 2.0,
        r_env_mode: str = "linear",
        keepout_buffer: float = 0.0,
        quadrature_rule: str = "simpson",
    ):
        self.high_risk_threshold = high_risk_threshold
        self.k_sigma = k_sigma
        self.abort_margin = abort_margin
        self.clearance_falloff = clearance_falloff
        # Journal Phase 2: "linear" = effective-radius linear falloff (baseline);
        # "probability" = exact intrusion probability max_m P(target ∈ keep-out)
        # from the propagated predictive Gaussian (Marcum-Q equivalent).
        self.r_env_mode = r_env_mode
        # Probability mode integrates over the keep-out disk R_zone + buffer so
        # the estimated event matches the evaluation's "unsafe" region. Unused
        # in linear mode (baseline untouched).
        self.keepout_buffer = keepout_buffer
        self.quadrature_rule = quadrature_rule

    def evaluate(self, scenario: Scenario, prediction: TrajectoryPrediction) -> EnvironmentalRiskAssessment:
        if not prediction.is_valid or not prediction.predictions:
            return EnvironmentalRiskAssessment(
                step_index=prediction.start_step_index,
                risk_score=0.0,
                is_high_risk=False,
                min_distance_to_risk_zone=None,
                risk_zone_radius=None,
                metadata={"reason": "invalid_prediction"}
            )

        if "risk_zone_center" not in scenario.metadata or "risk_zone_radius" not in scenario.metadata:
            return EnvironmentalRiskAssessment(
                step_index=prediction.start_step_index,
                risk_score=0.0,
                is_high_risk=False,
                min_distance_to_risk_zone=None,
                risk_zone_radius=None,
                metadata={"reason": "no_risk_zone_metadata"}
            )

        if self.r_env_mode == "probability":
            return self._evaluate_probability(scenario, prediction)
        elif self.r_env_mode != "linear":
            raise ValueError(
                f"Unknown r_env_mode: {self.r_env_mode!r} (expected 'linear' or 'probability')"
            )

        risk_zone_center = scenario.metadata["risk_zone_center"]
        risk_zone_radius = scenario.metadata["risk_zone_radius"]

        # Walk predictions; remember the closest one (with its uncertainty).
        min_dist = float('inf')
        min_pred_uncertainty = 0.0
        for pred in prediction.predictions:
            dist = math.hypot(
                pred.state.x - risk_zone_center[0],
                pred.state.y - risk_zone_center[1],
            )
            if dist < min_dist:
                min_dist = dist
                min_pred_uncertainty = pred.uncertainty

        # M3: effective unsafe radius incorporates prediction uncertainty
        # and the optional abort kinematic margin.
        effective_unsafe_radius = (
            risk_zone_radius
            + self.k_sigma * min_pred_uncertainty
            + self.abort_margin
        )
        risk_clearance = min_dist - effective_unsafe_radius

        if risk_clearance <= 0:
            risk_score = 1.0
        elif risk_clearance >= self.clearance_falloff:
            risk_score = 0.0
        else:
            risk_score = 1.0 - (risk_clearance / self.clearance_falloff)

        is_high_risk = risk_score >= self.high_risk_threshold

        return EnvironmentalRiskAssessment(
            step_index=prediction.start_step_index,
            risk_score=risk_score,
            is_high_risk=is_high_risk,
            min_distance_to_risk_zone=min_dist,
            risk_zone_radius=risk_zone_radius,
            metadata={
                "risk_clearance": risk_clearance,
                "min_pred_uncertainty": min_pred_uncertainty,
                "effective_unsafe_radius": effective_unsafe_radius,
                "k_sigma": self.k_sigma,
                "abort_margin": self.abort_margin,
            }
        )

    def _evaluate_probability(self, scenario: Scenario, prediction: TrajectoryPrediction) -> "EnvironmentalRiskAssessment":
        """Journal Phase 2: R_env = max_m P(predicted target ∈ risk zone).

        For each horizon step the predicted position is Gaussian with the
        propagated covariance (Kalman) — or, in the absence of covariance
        (α-β), an isotropic σ taken from the heuristic scalar uncertainty.
        The intrusion probability is the exact Gaussian-over-disk integral.
        ``k_sigma``, ``abort_margin`` and ``clearance_falloff`` are NOT used —
        the uncertainty enters through the actual spread of the distribution,
        which is the whole point (those free parameters are eliminated).
        """
        risk_zone_center = scenario.metadata["risk_zone_center"]
        risk_zone_radius = scenario.metadata["risk_zone_radius"]
        cx, cy = risk_zone_center[0], risk_zone_center[1]
        # Integrate over the keep-out disk (R_zone + buffer) so the estimated
        # event matches the evaluation's "unsafe" region.
        keepout_radius = risk_zone_radius + self.keepout_buffer

        max_p = 0.0
        max_p_offset = None
        min_dist = float("inf")
        sigma_at_max = 0.0
        for pred in prediction.predictions:
            mx, my = pred.state.x, pred.state.y
            d = math.hypot(mx - cx, my - cy)
            if d < min_dist:
                min_dist = d
            if pred.position_covariance is not None:
                sx = math.sqrt(max(pred.position_covariance[0][0], 0.0))
                sy = math.sqrt(max(pred.position_covariance[1][1], 0.0))
            else:
                # α-β fallback: isotropic σ from the heuristic scalar.
                sx = sy = max(pred.uncertainty, 0.0)
            # Short-circuit far steps: > 6σ clearance ⇒ negligible probability.
            if d - keepout_radius > 6.0 * max(sx, sy, 1e-9):
                p = 0.0
            else:
                p = gaussian_disk_probability(mx, my, sx, sy, cx, cy, keepout_radius,
                                              rule=self.quadrature_rule)
            if p > max_p:
                max_p = p
                max_p_offset = pred.step_offset
                sigma_at_max = max(sx, sy)

        risk_score = max(0.0, min(1.0, max_p))
        is_high_risk = risk_score >= self.high_risk_threshold
        return EnvironmentalRiskAssessment(
            step_index=prediction.start_step_index,
            risk_score=risk_score,
            is_high_risk=is_high_risk,
            min_distance_to_risk_zone=min_dist,
            risk_zone_radius=risk_zone_radius,
            metadata={
                "r_env_mode": "probability",
                "intrusion_probability": risk_score,
                "max_prob_step_offset": max_p_offset,
                "sigma_at_max_prob": sigma_at_max,
            },
        )

@dataclass
class InterceptabilityAssessment:
    step_index: int
    interceptability_score: float
    is_interceptable: bool
    speed_ratio: Optional[float]
    initial_distance: Optional[float]  # current relative distance (per-step)
    metadata: Dict[str, object] = field(default_factory=dict)


class InterceptabilityEvaluator:
    """M2 fix: state-aware interceptability.

    Replaces the previous metadata-only score (which returned a constant per
    scenario and ignored the `estimate` argument) with a per-step computation
    based on the current relative geometry, per docs §10:

        I = f(relative_distance, relative_speed, speed_ratio, prediction_horizon)

    The score is the product of three components, each in [0, 1]:

      * **speed_score**: how favourable the interceptor/target speed ratio is.
        Mapped linearly: ratio ≤ 0.5 → 0, ratio ≥ 1.5 → 1. Scenario metadata
        may override the ratio (S6 uses this).

      * **closing_factor**: whether the target is moving toward or away from
        the interceptor along the current line of sight. Head-on (v_los =
        -v_t) → 1.0; perpendicular → 0.75; receding → 0.25.

      * **distance_factor**: mild penalty for very distant targets. Clamped to
        [0.5, 1.0] so distance alone never zeroes out the score.

    With observation noise (C2), the noisy estimated state propagates into
    `closing_factor` so the score now varies step-to-step within a trial.
    That makes the `InterceptabilityOnlyPolicy` actually responsive to
    uncertainty rather than producing a scenario-indicator constant.
    """

    def __init__(
        self,
        interceptable_threshold: float = 0.5,
        default_v_interceptor: float = 2.0,
        typical_distance: float = 20.0,
        i_mode: str = "heuristic",
        engagement_horizon_time: float = 20.0,
    ):
        self.interceptable_threshold = interceptable_threshold
        self.default_v_interceptor = default_v_interceptor
        self.typical_distance = typical_distance
        # Journal Phase 3 (M1): "heuristic" = s_speed·s_closing·s_distance product
        # (baseline); "capturability" = pursuit-evasion intercept-triangle capture
        # time vs engagement horizon (physical, eliminates the heuristic constants).
        self.i_mode = i_mode
        self.engagement_horizon_time = engagement_horizon_time

    def evaluate(self, scenario: Scenario, estimate: TargetEstimate) -> InterceptabilityAssessment:
        if not estimate.is_valid or estimate.estimated_state is None:
            return InterceptabilityAssessment(
                step_index=estimate.step_index,
                interceptability_score=0.0,
                is_interceptable=False,
                speed_ratio=None,
                initial_distance=None,
                metadata={"reason": "invalid_estimate"},
            )

        if self.i_mode == "capturability":
            return self._evaluate_capturability(scenario, estimate)
        elif self.i_mode != "heuristic":
            raise ValueError(
                f"Unknown i_mode: {self.i_mode!r} (expected 'heuristic' or 'capturability')"
            )

        t = estimate.estimated_state
        i = scenario.interceptor.state

        # Relative geometry
        dx = t.x - i.x
        dy = t.y - i.y
        d = math.hypot(dx, dy)
        v_t = math.hypot(t.vx, t.vy)

        # Interceptor speed: scenario can override (S6 metadata supplies this),
        # otherwise fall back to evaluator default.
        v_i = scenario.metadata.get(
            "interceptor_nominal_max_speed", self.default_v_interceptor
        )

        # Speed ratio: scenario metadata (S6 hard label) overrides the
        # geometric ratio. Otherwise compute from current speeds.
        if "speed_ratio" in scenario.metadata:
            speed_ratio = float(scenario.metadata["speed_ratio"])
        elif v_t > 1e-6:
            speed_ratio = v_i / v_t
        else:
            speed_ratio = 2.0  # target stationary → trivially interceptable

        speed_score = max(0.0, min(1.0, (speed_ratio - 0.5) / 1.0))

        # Closing factor — sign of target velocity along the line of sight.
        if d > 1e-9 and v_t > 1e-6:
            ux, uy = dx / d, dy / d
            target_v_los = t.vx * ux + t.vy * uy  # > 0 if target moves away
            closing_factor = max(0.0, min(1.0, -target_v_los / v_t * 0.5 + 0.75))
        else:
            target_v_los = None
            closing_factor = 0.75  # nominal when geometry is degenerate

        # Distance attenuation — very far targets are harder to reach but the
        # framework never sets I = 0 from distance alone (lower bound 0.5).
        distance_factor = max(0.5, min(1.0, 1.0 - (d / self.typical_distance) * 0.3))

        score = speed_score * closing_factor * distance_factor
        score = max(0.0, min(1.0, score))
        is_interceptable = score >= self.interceptable_threshold

        return InterceptabilityAssessment(
            step_index=estimate.step_index,
            interceptability_score=score,
            is_interceptable=is_interceptable,
            speed_ratio=speed_ratio,
            initial_distance=d,
            metadata={
                "v_target": v_t,
                "v_interceptor": v_i,
                "target_v_los": target_v_los,
                "speed_score": speed_score,
                "closing_factor": closing_factor,
                "distance_factor": distance_factor,
            },
        )

    def _evaluate_capturability(self, scenario: Scenario, estimate: TargetEstimate) -> InterceptabilityAssessment:
        """Journal Phase 3 (M1): pursuit-evasion capturability.

        Solve the intercept triangle for the minimum capture time t*: the
        interceptor (max speed V_I, launched from its current position) can reach
        the target's straight-line future position iff

            |r + v_T·t| ≤ V_I·t   ⇔   (|v_T|²−V_I²)t² + 2(r·v_T)t + |r|² ≤ 0,

        where r = P_T − P_I. The smallest positive root is t*. Interceptability is
        graded by how comfortably t* fits inside the engagement horizon T_engage:

            I = clamp(1 − t*/T_engage, 0, 1),   I = 0 if no positive root.

        This replaces the heuristic product (and its constants 0.5/1.5/0.75/D₀)
        with one physical assessment combining speed ratio, geometry and range.
        """
        t = estimate.estimated_state
        i = scenario.interceptor.state
        rx, ry = t.x - i.x, t.y - i.y
        d = math.hypot(rx, ry)
        v_t = math.hypot(t.vx, t.vy)
        V_I = scenario.metadata.get("interceptor_nominal_max_speed", self.default_v_interceptor)
        T = self.engagement_horizon_time

        a = v_t * v_t - V_I * V_I
        b = 2.0 * (rx * t.vx + ry * t.vy)   # 2 (r · v_T)
        c = d * d

        t_star = None
        if abs(a) < 1e-9:
            # |v_T| ≈ V_I → linear: b t + c = 0
            if b < -1e-12:
                t_star = -c / b
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sq = math.sqrt(disc)
                roots = [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)]
                pos = [r for r in roots if r > 1e-9]
                if pos:
                    t_star = min(pos)

        if t_star is None or t_star > T:
            score = 0.0
        else:
            score = max(0.0, min(1.0, 1.0 - t_star / T))
        speed_ratio = (V_I / v_t) if v_t > 1e-6 else float("inf")
        return InterceptabilityAssessment(
            step_index=estimate.step_index,
            interceptability_score=score,
            is_interceptable=score >= self.interceptable_threshold,
            speed_ratio=(speed_ratio if speed_ratio != float("inf") else None),
            initial_distance=d,
            metadata={
                "i_mode": "capturability",
                "capture_time": t_star,
                "engagement_horizon_time": T,
                "v_target": v_t,
                "v_interceptor": V_I,
            },
        )
