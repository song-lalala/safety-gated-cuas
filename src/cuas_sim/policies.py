import enum
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .scenario import Scenario
from .observation import TargetObservation
from .estimation import TargetEstimate
from .prediction import TrajectoryPrediction
from .evaluators import (
    EnvironmentalRiskAssessment,
    InterceptabilityAssessment,
    AbortFeasibilityAssessment,
)

class DecisionAction(enum.Enum):
    ENGAGE = "ENGAGE"
    TRACK = "TRACK"
    ABORT = "ABORT"

@dataclass
class DecisionContext:
    step_index: int
    scenario_type: object
    detected: bool
    estimate_valid: bool
    prediction_valid: bool
    tracking_confidence: float
    identification_confidence: float
    risk_score: float
    is_high_risk: bool
    interceptability_score: float
    is_interceptable: bool
    distance_to_target: Optional[float]
    metadata: Dict[str, object] = field(default_factory=dict)
    # C3-a: 5th SafetyGate input — abort feasibility. Defaults treat the
    # situation as fully abortable so legacy callers that don't pass an
    # assessment get the previous behaviour.
    abort_feasibility_score: float = 1.0
    is_abort_feasible: bool = True
    # Review response (W12): geometry the reachability baseline needs. The
    # safety gate never reads these, so its behaviour is unchanged.
    min_distance_to_risk_zone: Optional[float] = None
    risk_zone_radius: Optional[float] = None
    target_speed_estimate: float = 0.0
    prediction_sigma: float = 0.0
    horizon_seconds: float = 0.0


def build_decision_context(
    scenario: Scenario,
    observation: TargetObservation,
    estimate: TargetEstimate,
    prediction: TrajectoryPrediction,
    risk_assessment: EnvironmentalRiskAssessment,
    interceptability_assessment: InterceptabilityAssessment,
    abort_feasibility_assessment: Optional[AbortFeasibilityAssessment] = None,
) -> DecisionContext:

    distance_to_target = None
    if estimate.estimated_state is not None:
        target_state = estimate.estimated_state
        interceptor_state = scenario.interceptor.state
        distance_to_target = math.hypot(
            target_state.x - interceptor_state.x,
            target_state.y - interceptor_state.y
        )

    if abort_feasibility_assessment is None:
        abort_feasibility_score = 1.0
        is_abort_feasible = True
        abort_metadata: Dict[str, object] = {"reason": "no_abort_evaluator"}
    else:
        abort_feasibility_score = abort_feasibility_assessment.abort_feasibility_score
        is_abort_feasible = abort_feasibility_assessment.is_feasible
        abort_metadata = abort_feasibility_assessment.metadata

    target_speed_estimate = 0.0
    if estimate.estimated_state is not None:
        target_speed_estimate = math.hypot(estimate.estimated_state.vx,
                                           estimate.estimated_state.vy)
    prediction_sigma = 0.0
    horizon_seconds = 0.0
    if prediction.predictions:
        last = prediction.predictions[-1]
        prediction_sigma = float(risk_assessment.metadata.get(
            "sigma_at_max_prob", getattr(last, "uncertainty", 0.0)) or 0.0)
        horizon_seconds = len(prediction.predictions) * scenario.config.time_step

    metadata = {
        "risk_metadata": risk_assessment.metadata,
        "interceptability_metadata": interceptability_assessment.metadata,
        "observation_metadata": observation.metadata,
        "estimate_metadata": estimate.metadata,
        "prediction_metadata": prediction.metadata,
        "abort_feasibility_metadata": abort_metadata,
    }

    return DecisionContext(
        step_index=observation.step_index,
        scenario_type=scenario.scenario_type,
        detected=observation.detected,
        estimate_valid=estimate.is_valid,
        prediction_valid=prediction.is_valid,
        tracking_confidence=estimate.tracking_confidence,
        identification_confidence=estimate.identification_confidence,
        risk_score=risk_assessment.risk_score,
        is_high_risk=risk_assessment.is_high_risk,
        interceptability_score=interceptability_assessment.interceptability_score,
        is_interceptable=interceptability_assessment.is_interceptable,
        distance_to_target=distance_to_target,
        metadata=metadata,
        abort_feasibility_score=abort_feasibility_score,
        is_abort_feasible=is_abort_feasible,
        min_distance_to_risk_zone=risk_assessment.min_distance_to_risk_zone,
        risk_zone_radius=risk_assessment.risk_zone_radius,
        target_speed_estimate=target_speed_estimate,
        prediction_sigma=prediction_sigma,
        horizon_seconds=horizon_seconds,
    )

class AlwaysEngagePolicy:
    """Always engages on detection. M5: explicit fallback reasons.

    The previous version returned ABORT silently on `not detected` (e.g. S5
    dropout). That was a free-credit for baselines because their records had
    no `reason` filled, so analysis couldn't distinguish a "policy-chose
    abort" decision from a "couldn't see the target" fallback. The behaviour
    is unchanged; only the audit trail is more explicit.
    """

    def explain(self, context: DecisionContext) -> tuple[DecisionAction, str]:
        if not context.detected:
            return DecisionAction.ABORT, "fallback_not_detected"
        return DecisionAction.ENGAGE, "always_engage"

    def decide(self, context: DecisionContext) -> DecisionAction:
        return self.explain(context)[0]


class DistanceOnlyPolicy:
    def __init__(self, engage_distance_threshold: float = 5.0):
        self.engage_distance_threshold = engage_distance_threshold

    def explain(self, context: DecisionContext) -> tuple[DecisionAction, str]:
        if not context.detected:
            return DecisionAction.ABORT, "fallback_not_detected"
        if not context.estimate_valid:
            return DecisionAction.ABORT, "fallback_invalid_estimate"
        if context.distance_to_target is None:
            return DecisionAction.TRACK, "fallback_no_distance"
        if context.distance_to_target <= self.engage_distance_threshold:
            return DecisionAction.ENGAGE, "distance_within_threshold"
        return DecisionAction.TRACK, "distance_above_threshold"

    def decide(self, context: DecisionContext) -> DecisionAction:
        return self.explain(context)[0]


class InterceptabilityOnlyPolicy:
    def __init__(self, interceptability_threshold: float = 0.5):
        self.interceptability_threshold = interceptability_threshold

    def explain(self, context: DecisionContext) -> tuple[DecisionAction, str]:
        if not context.detected:
            return DecisionAction.ABORT, "fallback_not_detected"
        if not context.estimate_valid:
            return DecisionAction.ABORT, "fallback_invalid_estimate"
        if context.interceptability_score >= self.interceptability_threshold:
            return DecisionAction.ENGAGE, "interceptable_above_threshold"
        return DecisionAction.TRACK, "interceptable_below_threshold"

    def decide(self, context: DecisionContext) -> DecisionAction:
        return self.explain(context)[0]

class SafetyGatePolicy:
    """Proposed Engage / Track / Abort policy using five safety inputs:
    I, C_track, C_id, R_env, A_abort.

    ABORT fires when tracking confidence is too low OR environmental risk is
    too high OR abort feasibility is insufficient; see the gate cascade in the
    article (Section "Gate structure"). Two changes from earlier code:

      - The "high environmental risk" branch now produces ABORT (the previous
        code returned TRACK, which contradicted that cascade and caused S4 trials
        to never abort proactively).
      - A new abort_feasibility branch was added; if A_abort drops below
        `abort_feasibility_threshold`, the gate returns ABORT.
    """

    def __init__(
        self,
        tracking_abort_threshold: float = 0.3,
        tracking_confidence_threshold: float = 0.6,
        identification_confidence_threshold: float = 0.6,
        risk_score_threshold: float = 0.7,
        interceptability_threshold: float = 0.5,
        abort_feasibility_threshold: float = 0.5,
    ):
        self.tracking_abort_threshold = tracking_abort_threshold
        self.tracking_confidence_threshold = tracking_confidence_threshold
        self.identification_confidence_threshold = identification_confidence_threshold
        self.risk_score_threshold = risk_score_threshold
        self.interceptability_threshold = interceptability_threshold
        self.abort_feasibility_threshold = abort_feasibility_threshold

    def explain(self, context: DecisionContext) -> tuple[DecisionAction, str]:
        # Structural failures → ABORT
        if not context.detected:
            return DecisionAction.ABORT, "not_detected"
        if not context.estimate_valid:
            return DecisionAction.ABORT, "invalid_estimate"
        if not context.prediction_valid:
            return DecisionAction.ABORT, "invalid_prediction"

        # Tracking is so degraded we should disengage entirely.
        if context.tracking_confidence < self.tracking_abort_threshold:
            return DecisionAction.ABORT, "tracking_confidence_too_low_abort"

        # C3 docs alignment: high environmental risk → ABORT (was TRACK).
        if context.is_high_risk or context.risk_score >= self.risk_score_threshold:
            return DecisionAction.ABORT, "high_environmental_risk"

        # C3-a new: abort feasibility — if we no longer have decision margin to
        # safely retreat, commit to ABORT now.
        if context.abort_feasibility_score < self.abort_feasibility_threshold:
            return DecisionAction.ABORT, "abort_feasibility_too_low"

        # TRACK conditions: observable but engagement isn't justified yet.
        if context.tracking_confidence < self.tracking_confidence_threshold:
            return DecisionAction.TRACK, "tracking_confidence_too_low_track"

        if context.identification_confidence < self.identification_confidence_threshold:
            return DecisionAction.TRACK, "identification_confidence_too_low"

        if not context.is_interceptable or context.interceptability_score < self.interceptability_threshold:
            return DecisionAction.TRACK, "not_interceptable"

        return DecisionAction.ENGAGE, "engage_conditions_satisfied"

    def decide(self, context: DecisionContext) -> DecisionAction:
        action, _reason = self.explain(context)
        return action


class BayesRiskPolicy:
    """Myopic (QMDP-style) Bayes-risk rule over the same five signals.

    Reviewer 2.9 asks why no POMDP-style baseline is compared. A full POMDP
    policy would require an explicit cost model over the belief space, which is
    precisely the input the calibrated gate is designed not to need; solving one
    would therefore answer a different question. The standard cheap
    approximation is QMDP: approximate the value function by the immediate
    reward and act greedily. Here the immediate reward is not invented — it is
    the paper's own risk-weighted score, evaluated in expectation under the
    filter posterior:

        E[ENGAGE] = w_s * I * (1 - R_env) * C_id  -  w_u * R_env  -  w_f * (1 - C_id)
        E[ABORT]  = R_env * w_a * A_abort
        E[TRACK]  = 0

    ENGAGE earns the safe-capture term only when the attempt both succeeds (I)
    and is safe (1 - R_env) against a correctly identified target (C_id), and
    pays the unsafe and false-engagement penalties at their probabilities.
    ABORT earns abort credit only in proportion to the chance a break-off was
    actually required. TRACK defers at no immediate gain or loss.

    Nothing here is tuned: the weights are the published scoring weights.
    """

    def __init__(self, weight_safe_capture: float = 1.0, weight_abort_success: float = 0.5,
                 weight_unsafe: float = 2.0, weight_false: float = 2.0):
        self.w_s = weight_safe_capture
        self.w_a = weight_abort_success
        self.w_u = weight_unsafe
        self.w_f = weight_false

    def explain(self, context: DecisionContext) -> tuple[DecisionAction, str]:
        if not context.detected:
            return DecisionAction.ABORT, "not_detected"
        if not context.estimate_valid:
            return DecisionAction.ABORT, "invalid_estimate"
        if not context.prediction_valid:
            return DecisionAction.ABORT, "invalid_prediction"

        r = context.risk_score
        c_id = context.identification_confidence
        v_engage = (self.w_s * context.interceptability_score * (1.0 - r) * c_id
                    - self.w_u * r - self.w_f * (1.0 - c_id))
        v_abort = r * self.w_a * context.abort_feasibility_score
        # Ties resolve toward TRACK. Away from a risk zone v_abort is 0, so a
        # plain argmax over (engage, abort, track) would pick whichever of the
        # two zero-valued actions came first and abort on every such tie.
        # Breaking off for no expected gain is strictly worse than continuing
        # to observe: the same immediate value, minus the engagement the next
        # step might justify.
        if v_engage > 0.0 and v_engage >= v_abort:
            return DecisionAction.ENGAGE, "bayes_risk_engage"
        if v_abort > 0.0:
            return DecisionAction.ABORT, "bayes_risk_abort"
        return DecisionAction.TRACK, "bayes_risk_track"

    def decide(self, context: DecisionContext) -> DecisionAction:
        return self.explain(context)[0]


class ReachabilityFilterPolicy:
    """Worst-case (set-based) safety filter instead of a probabilistic one.

    The second family Reviewer 2.9 names. Rather than asking how *likely* the
    target is to enter the keep-out disk, it asks whether it *can*: the forward
    reachable set of a speed-bounded target over the horizon is a ball of radius
    ``v_T * H * dt`` about the current estimate, inflated by ``kappa`` standard
    deviations of estimation error. Contact is authorised only if that whole set
    clears the keep-out disk.

    This is deliberately the most conservative reading of the same information,
    and the contrast with the probabilistic gate is the point: a guarantee that
    holds for every admissible target motion has to pay for the motions that
    never happen.
    """

    def __init__(self, identification_confidence_threshold: float = 0.6,
                 interceptability_threshold: float = 0.5, kappa: float = 3.0,
                 keepout_buffer: float = 0.6,
                 tracking_abort_threshold: Optional[float] = None,
                 tracking_confidence_threshold: Optional[float] = None):
        self.identification_confidence_threshold = identification_confidence_threshold
        self.interceptability_threshold = interceptability_threshold
        self.kappa = kappa
        self.keepout_buffer = keepout_buffer
        # When these are set the policy keeps the safety gate's tracking-
        # confidence branches and differs from it ONLY in the risk test, which
        # is what isolates worst-case risk from probabilistic risk. Left as
        # None it is a pure reachability filter with no tracking guard.
        self.tracking_abort_threshold = tracking_abort_threshold
        self.tracking_confidence_threshold = tracking_confidence_threshold

    def explain(self, context: DecisionContext) -> tuple[DecisionAction, str]:
        if not context.detected:
            return DecisionAction.ABORT, "not_detected"
        if not context.estimate_valid:
            return DecisionAction.ABORT, "invalid_estimate"
        if not context.prediction_valid:
            return DecisionAction.ABORT, "invalid_prediction"

        if self.tracking_abort_threshold is not None and \
                context.tracking_confidence < self.tracking_abort_threshold:
            return DecisionAction.ABORT, "tracking_confidence_too_low_abort"

        d = context.min_distance_to_risk_zone
        if d is not None and context.risk_zone_radius is not None:
            keepout = context.risk_zone_radius + self.keepout_buffer
            if d <= keepout:
                return DecisionAction.ABORT, "reachable_set_inside_keepout"
            reach = (context.target_speed_estimate * context.horizon_seconds
                     + self.kappa * context.prediction_sigma)
            if d - keepout <= reach:
                return DecisionAction.TRACK, "reachable_set_intersects_keepout"

        if self.tracking_confidence_threshold is not None and \
                context.tracking_confidence < self.tracking_confidence_threshold:
            return DecisionAction.TRACK, "tracking_confidence_too_low_track"
        if context.identification_confidence < self.identification_confidence_threshold:
            return DecisionAction.TRACK, "identification_confidence_too_low"
        if not context.is_interceptable or \
                context.interceptability_score < self.interceptability_threshold:
            return DecisionAction.TRACK, "not_interceptable"
        return DecisionAction.ENGAGE, "reachable_set_clear"

    def decide(self, context: DecisionContext) -> DecisionAction:
        return self.explain(context)[0]


class LatchingPolicy:
    """Wraps a policy so that ABORT, once emitted, holds for the rest of the trial.

    The gate itself is memoryless: every step re-runs the whole cascade, so an
    ABORT can be followed by an ENGAGE one step later. That is the right model
    for a *decision* study, but a fielded system would not un-break-off — and it
    is what makes a per-step false-alarm rate cheap. With ABORT absorbing, the
    per-trial false-abort rate is what the operator actually pays, so running
    both variants is how the per-step/per-trial distinction gets quantified.

    The runner builds a fresh policy per trial, so the latch needs no reset.
    """

    def __init__(self, inner):
        self.inner = inner
        self.latched = False

    def explain(self, context: DecisionContext) -> tuple[DecisionAction, str]:
        if self.latched:
            return DecisionAction.ABORT, "abort_latched"
        action, reason = self.inner.explain(context)
        if action == DecisionAction.ABORT:
            self.latched = True
        return action, reason

    def decide(self, context: DecisionContext) -> DecisionAction:
        action, _reason = self.explain(context)
        return action
