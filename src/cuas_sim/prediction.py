import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .types import State2D
from .estimation import TargetEstimate

@dataclass
class PredictedState:
    step_offset: int
    state: State2D
    uncertainty: float
    # Journal Phase 2: per-axis predicted *position* covariance propagated from
    # the Kalman estimate, [[var_x, 0], [0, var_y]] (axes independent under the
    # decoupled CV model). None in α-β mode (no covariance) → legacy scalar
    # `uncertainty` is used and the linear R_env reproduces the baseline.
    position_covariance: Optional[list] = None

@dataclass
class TrajectoryPrediction:
    start_step_index: int
    horizon_steps: int
    predictions: List[PredictedState]
    is_valid: bool
    metadata: Dict[str, object] = field(default_factory=dict)


def _propagate_axis_cov(P: list, dt: float, q: float) -> list:
    """One CV prediction step on a 2x2 axis covariance: P⁻ = F P Fᵀ + Q.

    F = [[1, dt], [0, 1]]; Q = q·[[dt⁴/4, dt³/2], [dt³/2, dt²]] (DWNA), the same
    process-noise model the Kalman estimator uses — so propagation is consistent
    with the filter that produced P.
    """
    a, b = P[0][0], P[0][1]
    c, d = P[1][0], P[1][1]
    P00 = a + dt * (b + c) + dt * dt * d
    P01 = b + dt * d
    P10 = c + dt * d
    P11 = d
    dt2 = dt * dt; dt3 = dt2 * dt; dt4 = dt3 * dt
    return [
        [P00 + q * dt4 / 4.0, P01 + q * dt3 / 2.0],
        [P10 + q * dt3 / 2.0, P11 + q * dt2],
    ]


class ConstantVelocityPredictor:
    def __init__(self, horizon_steps: int = 5, process_noise: Optional[float] = None):
        self.horizon_steps = horizon_steps
        # When set (Kalman mode), the predicted-position covariance is
        # propagated with this acceleration PSD. When None (α-β mode), the
        # legacy scalar uncertainty heuristic is used.
        self.process_noise = process_noise

    def predict(self, estimate: TargetEstimate, time_step: float) -> TrajectoryPrediction:
        if not estimate.is_valid or estimate.estimated_state is None:
            return TrajectoryPrediction(
                start_step_index=estimate.step_index,
                horizon_steps=self.horizon_steps,
                predictions=[],
                is_valid=False,
            )

        predictions = []
        base_uncertainty = 1.0 - estimate.tracking_confidence

        # Kalman mode: seed per-axis covariance from the estimate and propagate
        # it forward one step per horizon offset (σ grows ∝ √m, the standard
        # probabilistic propagation — replacing the heuristic 0.05/step).
        use_cov = (
            self.process_noise is not None
            and estimate.covariance is not None
            and "x" in estimate.covariance
            and "y" in estimate.covariance
        )
        if use_cov:
            Px = [list(row) for row in estimate.covariance["x"]]
            Py = [list(row) for row in estimate.covariance["y"]]

        for offset in range(1, self.horizon_steps + 1):
            x_future = estimate.estimated_state.x + estimate.estimated_state.vx * time_step * offset
            y_future = estimate.estimated_state.y + estimate.estimated_state.vy * time_step * offset

            pred_state = State2D(
                x=x_future,
                y=y_future,
                vx=estimate.estimated_state.vx,
                vy=estimate.estimated_state.vy
            )

            if use_cov:
                Px = _propagate_axis_cov(Px, time_step, self.process_noise)
                Py = _propagate_axis_cov(Py, time_step, self.process_noise)
                var_x = Px[0][0]
                var_y = Py[0][0]
                position_covariance = [[var_x, 0.0], [0.0, var_y]]
                # isotropic-equivalent std, so any scalar consumer (e.g. linear
                # R_env) still has a meaningful σ_pred from the real covariance.
                uncertainty = math.sqrt(0.5 * (var_x + var_y))
            else:
                position_covariance = None
                uncertainty = base_uncertainty + 0.05 * offset

            predictions.append(PredictedState(
                step_offset=offset,
                state=pred_state,
                uncertainty=uncertainty,
                position_covariance=position_covariance,
            ))

        return TrajectoryPrediction(
            start_step_index=estimate.step_index,
            horizon_steps=self.horizon_steps,
            predictions=predictions,
            is_valid=True
        )
