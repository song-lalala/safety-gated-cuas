from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional
import copy
import math

from .types import State2D
from .observation import TargetObservation


@dataclass
class TargetEstimate:
    step_index: int
    estimated_state: Optional[State2D]
    detected: bool
    is_valid: bool
    tracking_confidence: float
    identification_confidence: float
    metadata: Dict[str, object] = field(default_factory=dict)
    # Journal Phase 1: per-axis state covariance from the Kalman estimator.
    # Shape: {"x": [[P00,P01],[P10,P11]], "y": [[...]]} for the 2-state
    # [position, velocity] filter on each axis. None for the α-β estimator
    # (which carries no covariance) and on dropout. Phase 2 propagates this
    # to obtain sigma_pred; here it lets downstream code stay backward
    # compatible (default None → legacy behaviour unchanged).
    covariance: Optional[Dict[str, object]] = None


class TargetStateEstimator:
    """α-β tracker for noisy target observations.

    C2 fix: replaces the previous passthrough estimator with a simple α-β
    filter that smooths position and updates velocity from residuals. The
    tracking confidence reported in TargetEstimate is now derived from the
    smoothed residual magnitude, following the doc concept
    `C_track = exp(-k · σ_est)`, instead of being copied verbatim from the
    observation.

    With perfect observations (noise σ = 0) the filter reproduces the previous
    passthrough behaviour: residuals are zero, the smoothed σ_est stays at 0,
    and tracking_confidence stays at 1.0. So all legacy tests (which use the
    no-noise default ObservationModel) keep passing.

    During dropout (observation.detected = False) the filter:
      - holds the last estimated state but reports is_valid = False
      - forwards observation.tracking_confidence (the scenario-defined lost
        level) so S5 abort logic still works
    """

    def __init__(
        self,
        time_step: float = 0.1,
        alpha: float = 0.7,
        beta: float = 0.2,
        confidence_k: float = 2.0,
        sigma_smoothing: float = 0.3,
    ):
        self.time_step = time_step
        self.alpha = alpha
        self.beta = beta
        self.confidence_k = confidence_k
        self.sigma_smoothing = sigma_smoothing  # weight of new |residual| in EWMA
        self.last_estimated_state: Optional[State2D] = None
        self.sigma_est: float = 0.0  # smoothed residual magnitude

    def _confidence_from_sigma(self) -> float:
        return math.exp(-self.confidence_k * self.sigma_est)

    def estimate(self, observation: TargetObservation) -> TargetEstimate:
        if not observation.detected or observation.observed_state is None:
            # Dropout: hold last state, but mark invalid. Forward the
            # observation's tracking_confidence (e.g., S5 lost level) so
            # downstream policy logic on dropout is unchanged.
            held = copy.copy(self.last_estimated_state) if self.last_estimated_state is not None else None
            return TargetEstimate(
                step_index=observation.step_index,
                estimated_state=held,
                detected=False,
                is_valid=False,
                tracking_confidence=observation.tracking_confidence,
                identification_confidence=observation.identification_confidence,
                metadata=copy.deepcopy(observation.metadata),
            )

        z = observation.observed_state

        if self.last_estimated_state is None:
            # First measurement: initialise filter state from the observation.
            # Velocity is taken from the observation directly (the runner's
            # ObservationModel still reports the target velocity); the α-β
            # update will refine it from residuals on subsequent steps.
            new_state = State2D(x=z.x, y=z.y, vx=z.vx, vy=z.vy)
            # No residual yet → sigma_est stays at its current value (0 by default).
        else:
            # Predict from prior state using constant velocity over one step.
            prev = self.last_estimated_state
            dt = self.time_step
            pred_x = prev.x + prev.vx * dt
            pred_y = prev.y + prev.vy * dt

            # Residual = measurement − prediction
            r_x = z.x - pred_x
            r_y = z.y - pred_y

            # α-β update
            new_x = pred_x + self.alpha * r_x
            new_y = pred_y + self.alpha * r_y
            new_vx = prev.vx + (self.beta / dt) * r_x
            new_vy = prev.vy + (self.beta / dt) * r_y
            new_state = State2D(x=new_x, y=new_y, vx=new_vx, vy=new_vy)

            # Smooth |residual| into σ_est (EWMA)
            res_mag = math.hypot(r_x, r_y)
            self.sigma_est = (1.0 - self.sigma_smoothing) * self.sigma_est + self.sigma_smoothing * res_mag

        self.last_estimated_state = copy.copy(new_state)
        tracking_confidence = self._confidence_from_sigma()

        return TargetEstimate(
            step_index=observation.step_index,
            estimated_state=new_state,
            detected=True,
            is_valid=True,
            tracking_confidence=tracking_confidence,
            identification_confidence=observation.identification_confidence,
            metadata=copy.deepcopy(observation.metadata),
        )


# =====================================================================
#  Journal Phase 1 — Kalman filter estimator (replaces α-β; statistical
#  tracking confidence via Normalized Innovation Squared, NIS).
#
#  A constant-velocity target observed in position only DECOUPLES across
#  the x and y axes when Q and R are axis-diagonal. So the 4-state filter
#  is implemented as two independent 2-state ([pos, vel]) filters — no
#  numpy needed. The total NIS = NIS_x + NIS_y is the sum of two
#  independent chi-square(1) variates ⇒ chi-square(2) when the filter is
#  consistent, and
#
#       C_track = exp(-½ · NIS) = exp(-½ rᵀ S⁻¹ r)
#
#  is simultaneously (a) the normalized Gaussian innovation likelihood and
#  (b) the chi-square(2) tail probability P(χ²₂ > NIS). The arbitrary
#  α-β constant k=2.0 is replaced by the statistics-fixed ½, and the
#  scalar σ_est by the covariance-normalized Mahalanobis NIS.
# =====================================================================


def ctrack_threshold_from_nis(nis_threshold: float) -> float:
    """Map an NIS threshold to the equivalent C_track threshold.

    Since C_track = exp(-NIS/2), an "ABORT if NIS exceeds T" rule is exactly
    "ABORT if C_track < exp(-T/2)".
    """
    return math.exp(-0.5 * nis_threshold)


def ctrack_threshold_for_false_rate(false_rate: float) -> float:
    """C_track threshold giving a target *false-trigger* rate under a
    consistent Kalman filter.

    Key statistical fact: when the filter is consistent the 2D NIS follows
    chi-square(2) = Exponential(mean 2), and therefore C_track = exp(-NIS/2)
    is **Uniform(0, 1)**. Hence P(C_track < theta) = theta exactly, i.e. the
    threshold *equals* the desired false-trigger probability. For example a
    1% false-ABORT rate ⇒ theta_abort = 0.01; a 5% false-TRACK rate ⇒
    theta_track = 0.05. This replaces the α-β-era hand-set 0.3 / 0.6, whose
    false-trigger rates were not interpretable.

    (The chi-square(2) quantiles equivalently: NIS_0.99 = 9.21 ⇒ theta = 0.01;
    NIS_0.95 = 5.99 ⇒ theta = 0.050.)
    """
    return false_rate


def chi2_even_sf(x: float, half_dof: int) -> float:
    """P(chi-square(2k) > x) for integer k — exact, closed form, stdlib only.

    chi-square with even degrees of freedom is Erlang, so the survival function
    is the Poisson partial sum
        P(X > x) = exp(-x/2) * sum_{j=0}^{k-1} (x/2)^j / j!
    At k=1 this is exp(-x/2), i.e. the single-sample C_track of (4) — so the
    windowed confidence below reduces exactly to the submitted formula at W=1.
    """
    if x <= 0.0:
        return 1.0
    half = 0.5 * x
    if half > 700.0:          # exp(-half) underflows; the tail is 0 either way
        return 0.0
    term = 1.0
    total = 1.0
    for j in range(1, half_dof):
        term *= half / j
        total += term
    return math.exp(-half) * total


class _Axis1DKalman:
    """2-state constant-velocity Kalman filter for a single axis.

    State [p, v]; transition F = [[1, dt], [0, 1]]; position-only
    measurement H = [1, 0]. Process noise is the standard discretized
    white-noise-acceleration (DWNA) model with acceleration PSD ``q``;
    measurement variance ``r`` (scalar).
    """

    def __init__(self, dt: float, q: float, r: float):
        self.dt = dt
        self.q = q
        self.r = r
        self.p = 0.0
        self.v = 0.0
        # Covariance P (2x2): P00 P01 / P10 P11
        self.P00 = 0.0
        self.P01 = 0.0
        self.P10 = 0.0
        self.P11 = 0.0
        self._staged = None

    def init_state(self, pos: float, vel: float, p0_pos: float, p0_vel: float) -> None:
        self.p = pos
        self.v = vel
        self.P00 = p0_pos
        self.P01 = 0.0
        self.P10 = 0.0
        self.P11 = p0_vel

    def predict_only(self) -> None:
        """Advance state+covariance one step without a measurement (dropout)."""
        dt, q = self.dt, self.q
        self.p = self.p + dt * self.v
        a = self.P00 + dt * self.P10
        b = self.P01 + dt * self.P11
        P00 = a + dt * b
        P01 = b
        P10 = self.P10 + dt * self.P11
        P11 = self.P11
        dt2 = dt * dt; dt3 = dt2 * dt; dt4 = dt3 * dt
        self.P00 = P00 + q * dt4 / 4.0
        self.P01 = P01 + q * dt3 / 2.0
        self.P10 = P10 + q * dt3 / 2.0
        self.P11 = P11 + q * dt2

    def stage(self, z: float) -> tuple[float, float]:
        """Predict one step and compute the innovation, without committing.

        Split out of :meth:`update` so a caller that gates on the *2-D* NIS can
        inspect the innovation before deciding whether to let the measurement
        into the filter. Returns (innovation, innovation_variance) and must be
        followed by :meth:`commit`.
        """
        dt, q = self.dt, self.q
        # --- Predict: x⁻ = F x ;  P⁻ = F P Fᵀ + Q ---
        p_pred = self.p + dt * self.v
        v_pred = self.v
        a = self.P00 + dt * self.P10
        b = self.P01 + dt * self.P11
        P00 = a + dt * b
        P01 = b
        P10 = self.P10 + dt * self.P11
        P11 = self.P11
        dt2 = dt * dt; dt3 = dt2 * dt; dt4 = dt3 * dt
        P00 += q * dt4 / 4.0
        P01 += q * dt3 / 2.0
        P10 += q * dt3 / 2.0
        P11 += q * dt2
        innov = z - p_pred
        S = P00 + self.r
        self._staged = (p_pred, v_pred, P00, P01, P10, P11, innov, S)
        return innov, S

    def commit(self, use_measurement: bool = True) -> None:
        """Apply the staged step. With ``use_measurement=False`` the prediction
        is kept and the measurement discarded — chi-square gating of an outlier.
        """
        p_pred, v_pred, P00, P01, P10, P11, innov, S = self._staged
        self._staged = None
        if not use_measurement:
            self.p, self.v = p_pred, v_pred
            self.P00, self.P01, self.P10, self.P11 = P00, P01, P10, P11
            return
        # --- Update: H = [1, 0] ---
        K0 = P00 / S
        K1 = P10 / S
        self.p = p_pred + K0 * innov
        self.v = v_pred + K1 * innov
        # P = (I - K H) P⁻ ,  (I-KH) = [[1-K0, 0], [-K1, 1]]
        self.P00 = (1.0 - K0) * P00
        self.P01 = (1.0 - K0) * P01
        self.P10 = -K1 * P00 + P10
        self.P11 = -K1 * P01 + P11

    def update(self, z: float) -> tuple[float, float]:
        """Predict + correct with position measurement ``z``.

        Returns (innovation, innovation_variance) for NIS accumulation.
        """
        innov, S = self.stage(z)
        self.commit(True)
        return innov, S

    def cov(self) -> list:
        return [[self.P00, self.P01], [self.P10, self.P11]]


class KalmanTargetEstimator:
    """Constant-velocity Kalman tracker with NIS-based tracking confidence.

    Drop-in replacement for :class:`TargetStateEstimator` (same
    ``estimate(observation) -> TargetEstimate`` contract) selected via
    ``MonteCarloConfig.estimator_mode = "kalman"``. The α-β estimator
    remains the default so the Phase-0 baseline is reproduced exactly when
    the toggle is off.

    Parameters
    ----------
    time_step : float
        Filter step dt.
    process_noise : float
        Acceleration PSD ``q`` of the DWNA process-noise model. This is the
        principled replacement for the heuristic σ_pred growth (0.05/step);
        tune so the mean NIS ≈ 2 (the measurement dimension) for filter
        consistency.
    measurement_sigma : Optional[float]
        Per-axis position measurement std. If None, falls back to
        ``observation_noise_sigma`` supplied by the runner. Floored by
        ``min_measurement_sigma`` so the innovation covariance never
        degenerates (e.g. perfect-sensor σ_obs = 0).
    min_measurement_sigma : float
        Floor on the measurement std used to build R.
    init_velocity_var : float
        Initial velocity variance (position variance is initialised to R).
    ctrack_window : int
        Length W of the time-averaged NIS window (Bar-Shalom's standard filter
        consistency test). The statistic is the sum of the last W NIS values,
        which is chi-square(2W) under a consistent filter, so the confidence
        stays a p-value and its threshold keeps meaning a false-alarm rate.
        W=1 (default) is the single-sample form of the submitted paper.
        A window trades detection latency for far fewer per-trial false alarms:
        at 1% per step, 20 steps give an 18.2% per-trial false-abort rate.
    nis_gate_threshold : Optional[float]
        Chi-square measurement gate. A step whose 2-D NIS exceeds this value is
        treated as an outlier: the filter keeps its prediction and discards the
        measurement, which bounds the influence any single observation can have.
        ``None`` (default) disables gating. 9.21 is the chi-square(2) 99% point.
        The NIS *reported* for C_track is unchanged, so a gated outlier still
        lowers tracking confidence — the gate protects the estimate, not the alarm.
    """

    def __init__(
        self,
        time_step: float = 0.1,
        process_noise: float = 0.1,
        measurement_sigma: Optional[float] = None,
        min_measurement_sigma: float = 1e-2,
        init_velocity_var: float = 1.0,
        ctrack_window: int = 1,
        nis_gate_threshold: Optional[float] = None,
    ):
        self.time_step = time_step
        self.process_noise = process_noise
        self.min_measurement_sigma = min_measurement_sigma
        sigma = measurement_sigma if measurement_sigma is not None else 0.0
        self.measurement_sigma = max(sigma, min_measurement_sigma)
        self.r = self.measurement_sigma ** 2
        self.init_velocity_var = init_velocity_var
        self.ctrack_window = max(1, int(ctrack_window))
        self.nis_gate_threshold = nis_gate_threshold
        self._nis_window: deque = deque(maxlen=self.ctrack_window)
        self.ax: Optional[_Axis1DKalman] = None
        self.ay: Optional[_Axis1DKalman] = None
        self.last_estimated_state: Optional[State2D] = None

    def _windowed_confidence(self, nis: float) -> float:
        """C_track from the last W NIS values. Reduces to exp(-NIS/2) at W=1.

        During warm-up the window holds fewer than W samples; the degrees of
        freedom follow the count actually present, so the quantity stays a
        correctly-scaled p-value from the very first step.
        """
        self._nis_window.append(nis)
        if self.ctrack_window == 1:
            return math.exp(-0.5 * nis)
        return chi2_even_sf(sum(self._nis_window), len(self._nis_window))

    def _make_axis(self) -> _Axis1DKalman:
        return _Axis1DKalman(dt=self.time_step, q=self.process_noise, r=self.r)

    def _covariance_dict(self) -> Dict[str, object]:
        return {"x": self.ax.cov(), "y": self.ay.cov()}

    def estimate(self, observation: TargetObservation) -> TargetEstimate:
        if not observation.detected or observation.observed_state is None:
            # Dropout: hold last state (predict covariance forward if we have a
            # filter), mark invalid, forward the observation's tracking
            # confidence — identical contract to the α-β estimator so S5 logic
            # is unchanged.
            if self.ax is not None and self.ay is not None:
                self.ax.predict_only()
                self.ay.predict_only()
                if self.last_estimated_state is not None:
                    self.last_estimated_state = State2D(
                        x=self.ax.p, y=self.ay.p, vx=self.ax.v, vy=self.ay.v
                    )
            held = copy.copy(self.last_estimated_state) if self.last_estimated_state is not None else None
            return TargetEstimate(
                step_index=observation.step_index,
                estimated_state=held,
                detected=False,
                is_valid=False,
                tracking_confidence=observation.tracking_confidence,
                identification_confidence=observation.identification_confidence,
                metadata=copy.deepcopy(observation.metadata),
                covariance=(self._covariance_dict() if self.ax is not None else None),
            )

        z = observation.observed_state

        if self.ax is None or self.ay is None:
            # First measurement: initialise both axis filters. No innovation
            # yet ⇒ NIS = 0 ⇒ C_track = 1.0, matching the α-β first-step
            # behaviour (σ_est = 0 → confidence 1.0).
            self.ax = self._make_axis()
            self.ay = self._make_axis()
            self.ax.init_state(z.x, z.vx, p0_pos=self.r, p0_vel=self.init_velocity_var)
            self.ay.init_state(z.y, z.vy, p0_pos=self.r, p0_vel=self.init_velocity_var)
            new_state = State2D(x=z.x, y=z.y, vx=z.vx, vy=z.vy)
            nis = 0.0
            tracking_confidence = 1.0
            innov = (0.0, 0.0)
            S = (self.ax.P00 + self.r, self.ay.P00 + self.r)
            gated = False
        else:
            # Staged so the gate can see the 2-D NIS before the measurement is
            # allowed into either axis; with gating off this is exactly update().
            innov_x, Sx = self.ax.stage(z.x)
            innov_y, Sy = self.ay.stage(z.y)
            nis = (innov_x * innov_x) / Sx + (innov_y * innov_y) / Sy
            gated = (self.nis_gate_threshold is not None
                     and nis > self.nis_gate_threshold)
            self.ax.commit(use_measurement=not gated)
            self.ay.commit(use_measurement=not gated)
            new_state = State2D(x=self.ax.p, y=self.ay.p, vx=self.ax.v, vy=self.ay.v)
            tracking_confidence = self._windowed_confidence(nis)
            innov = (innov_x, innov_y)
            S = (Sx, Sy)

        self.last_estimated_state = copy.copy(new_state)

        metadata = copy.deepcopy(observation.metadata)
        metadata["nis"] = nis
        metadata["innovation"] = innov
        metadata["innovation_var"] = S
        metadata["nis_gated"] = gated

        return TargetEstimate(
            step_index=observation.step_index,
            estimated_state=new_state,
            detected=True,
            is_valid=True,
            tracking_confidence=tracking_confidence,
            identification_confidence=observation.identification_confidence,
            metadata=metadata,
            covariance=self._covariance_dict(),
        )
