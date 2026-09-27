from dataclasses import dataclass, field
from typing import Dict, Optional
import copy
import math
import random

from .types import State2D
from .scenario import Scenario


@dataclass
class TargetObservation:
    step_index: int
    detected: bool
    observed_state: Optional[State2D]
    tracking_confidence: float
    identification_confidence: float
    metadata: Dict[str, object] = field(default_factory=dict)


class ObservationModel:
    """Sensor model producing (possibly noisy) target observations.

    C2 fix: adds optional Gaussian measurement noise to position (and optionally
    velocity) so the downstream policy must reason under uncertainty rather than
    against ground truth. With noise_sigma=0 (default) the model is a perfect
    sensor — preserving the original test behaviour.

    Parameters
    ----------
    noise_sigma : float
        Standard deviation of the per-axis Gaussian position noise. 0 disables
        position noise.
    velocity_noise_sigma : float
        Standard deviation of the per-axis Gaussian velocity noise. 0 disables
        velocity noise.
    rng : Optional[random.Random]
        Seeded RNG for reproducibility. If None and any sigma > 0, a fresh
        unseeded Random() is created (non-deterministic — caller should pass a
        seeded rng for reproducible experiments).
    """

    def __init__(
        self,
        noise_sigma: float = 0.0,
        velocity_noise_sigma: float = 0.0,
        rng: Optional[random.Random] = None,
        noise_model: str = "gaussian",
        noise_dof: float = 3.0,
        bias: float = 0.0,
        outlier_rate: float = 0.0,
        outlier_scale: float = 5.0,
        cid_bias: float = 0.0,
        cid_gamma: float = 1.0,
        cid_noise_sigma: float = 0.0,
    ):
        self.noise_sigma = noise_sigma
        self.velocity_noise_sigma = velocity_noise_sigma
        # Review response (W3, Reviewer 2.11): real C-UAS sensors are not
        # zero-mean isotropic Gaussian. All four extras default to the
        # submitted behaviour, so the baseline is reproduced exactly.
        self.noise_model = noise_model      # "gaussian" | "student_t"
        self.noise_dof = noise_dof          # Student-t degrees of freedom
        self.bias = bias                    # persistent per-axis offset
        self.outlier_rate = outlier_rate    # P(step is a clutter outlier)
        self.outlier_scale = outlier_scale  # outlier std, in units of sigma
        # Review response (W10, Reviewer 2.8): C_id arrives from a classifier in
        # a fielded system, not from the scenario. These three distortions stand
        # in for that pipeline being biased, mis-calibrated or noisy. All
        # defaults are identity, so the submitted behaviour is reproduced.
        self.cid_bias = cid_bias
        self.cid_gamma = cid_gamma
        self.cid_noise_sigma = cid_noise_sigma
        if rng is None and (noise_sigma > 0 or velocity_noise_sigma > 0
                            or bias != 0.0 or outlier_rate > 0
                            or cid_noise_sigma > 0):
            rng = random.Random()
        self.rng = rng

    def _position_noise(self) -> float:
        """One axis of position noise under the configured sensor model.

        Student-t is rescaled to the same variance as the Gaussian case
        (Var[t_v] = v/(v-2)), so heavy tails are compared at equal noise power
        rather than at merely larger noise.
        """
        if self.noise_sigma <= 0:
            return 0.0
        if self.outlier_rate > 0 and self.rng.random() < self.outlier_rate:
            return self.rng.gauss(0.0, self.noise_sigma * self.outlier_scale)
        if self.noise_model == "student_t":
            v = self.noise_dof
            scale = self.noise_sigma / math.sqrt(v / (v - 2.0)) if v > 2.0 else self.noise_sigma
            # t_v = Z / sqrt(W/v), W ~ chi-square(v); gammavariate gives W/2.
            w = self.rng.gammavariate(v / 2.0, 2.0)
            return scale * self.rng.gauss(0.0, 1.0) / math.sqrt(w / v)
        return self.rng.gauss(0.0, self.noise_sigma)

    def _add_noise(self, state: State2D) -> State2D:
        if self.rng is None:
            return state
        dx = self._position_noise() + self.bias
        dy = self._position_noise() + self.bias
        dvx = self.rng.gauss(0.0, self.velocity_noise_sigma) if self.velocity_noise_sigma > 0 else 0.0
        dvy = self.rng.gauss(0.0, self.velocity_noise_sigma) if self.velocity_noise_sigma > 0 else 0.0
        return State2D(
            x=state.x + dx,
            y=state.y + dy,
            vx=state.vx + dvx,
            vy=state.vy + dvy,
        )

    def observe(self, scenario: Scenario, step_index: int) -> TargetObservation:
        # Default behavior
        detected = True
        observed_state: Optional[State2D] = copy.copy(scenario.target.state)
        tracking_confidence = 0.9
        identification_confidence = 0.9
        metadata: Dict[str, object] = {
            "scenario_type": scenario.scenario_type,
            "noise_sigma": self.noise_sigma,
        }

        # Override with S3 metadata if present
        if "identification_confidence" in scenario.metadata:
            identification_confidence = scenario.metadata["identification_confidence"]

        # Override with S5 metadata if present (tracking loss dropout)
        if all(k in scenario.metadata for k in (
            "tracking_loss_start_step",
            "tracking_loss_end_step",
            "lost_tracking_confidence",
            "nominal_tracking_confidence",
        )):
            start_step = scenario.metadata["tracking_loss_start_step"]
            end_step = scenario.metadata["tracking_loss_end_step"]
            lost_conf = scenario.metadata["lost_tracking_confidence"]
            nominal_conf = scenario.metadata["nominal_tracking_confidence"]

            if start_step <= step_index < end_step:
                detected = False
                observed_state = None
                tracking_confidence = lost_conf
            else:
                detected = True
                observed_state = copy.copy(scenario.target.state)
                tracking_confidence = nominal_conf

        # Apply sensor error when detected. A persistent bias is an error even
        # with no random noise, so it has to open this gate on its own.
        if detected and observed_state is not None and (
            self.noise_sigma > 0 or self.velocity_noise_sigma > 0 or self.bias != 0.0
        ):
            observed_state = self._add_noise(observed_state)

        identification_confidence = self._distort_cid(identification_confidence)

        return TargetObservation(
            step_index=step_index,
            detected=detected,
            observed_state=observed_state,
            tracking_confidence=tracking_confidence,
            identification_confidence=identification_confidence,
            metadata=metadata,
        )

    def _distort_cid(self, c: float) -> float:
        """Classifier distortion: mis-calibration, then bias, then noise.

        ``gamma`` is the usual calibration-curve exponent - below 1 the
        classifier is over-confident, above 1 under-confident - and it is
        applied first because a real pipeline's miscalibration sits upstream of
        any offset or jitter added by the reporting chain.
        """
        if self.cid_gamma != 1.0:
            c = c ** self.cid_gamma
        c += self.cid_bias
        if self.cid_noise_sigma > 0 and self.rng is not None:
            c += self.rng.gauss(0.0, self.cid_noise_sigma)
        return max(0.0, min(1.0, c))
