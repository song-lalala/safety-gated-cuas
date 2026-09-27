import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from .policies import DecisionAction, DecisionContext
from .types import State2D

# Keys we snapshot from scenario.metadata for ground-truth-based evaluation.
# Keeping the list explicit prevents leaking unrelated metadata into records.
_EVAL_METADATA_KEYS = (
    "risk_zone_center",
    "risk_zone_radius",
    "interceptability_label",
    "non_interceptable_reason",
    "scenario_name",
    # Journal Phase 4: lets extended scenarios (e.g. S9) flag misidentification
    # risk so engaging is classified as a false engagement in the ground truth.
    "misidentification_risk",
)


def _snapshot_eval_metadata(scenario_metadata: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not scenario_metadata:
        return {}
    return {k: scenario_metadata[k] for k in _EVAL_METADATA_KEYS if k in scenario_metadata}


@dataclass
class DecisionRecord:
    policy_name: str
    scenario_type: object
    step_index: int
    action: DecisionAction
    reason: Optional[str]
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
    metadata: Dict[str, object]
    # Ground-truth fields added for C1 fix: evaluation must not rely on
    # the same noisy/estimated values the policy used as input.
    true_target_state: Optional[State2D] = None
    scenario_metadata_snapshot: Dict[str, object] = field(default_factory=dict)
    # C3 fix: 5th SafetyGate input now logged for analysis.
    abort_feasibility_score: float = 1.0
    is_abort_feasible: bool = True

    @classmethod
    def from_context(
        cls,
        policy_name: str,
        context: DecisionContext,
        action: DecisionAction,
        reason: Optional[str] = None,
        *,
        true_target_state: Optional[State2D] = None,
        scenario_metadata: Optional[Dict[str, object]] = None,
    ) -> "DecisionRecord":
        return cls(
            policy_name=policy_name,
            scenario_type=context.scenario_type,
            step_index=context.step_index,
            action=action,
            reason=reason,
            detected=context.detected,
            estimate_valid=context.estimate_valid,
            prediction_valid=context.prediction_valid,
            tracking_confidence=context.tracking_confidence,
            identification_confidence=context.identification_confidence,
            risk_score=context.risk_score,
            is_high_risk=context.is_high_risk,
            interceptability_score=context.interceptability_score,
            is_interceptable=context.is_interceptable,
            distance_to_target=context.distance_to_target,
            metadata=dict(context.metadata),
            true_target_state=copy.copy(true_target_state) if true_target_state is not None else None,
            scenario_metadata_snapshot=_snapshot_eval_metadata(scenario_metadata),
            abort_feasibility_score=context.abort_feasibility_score,
            is_abort_feasible=context.is_abort_feasible,
        )

    def to_dict(self) -> Dict[str, object]:
        st = self.scenario_type
        if hasattr(st, "value"):
            st_val = st.value
        else:
            st_val = str(st)

        d: Dict[str, object] = {
            "policy_name": self.policy_name,
            "scenario_type": st_val,
            "step_index": self.step_index,
            "action": self.action.value,
            "reason": self.reason,
            "detected": self.detected,
            "estimate_valid": self.estimate_valid,
            "prediction_valid": self.prediction_valid,
            "tracking_confidence": self.tracking_confidence,
            "identification_confidence": self.identification_confidence,
            "risk_score": self.risk_score,
            "is_high_risk": self.is_high_risk,
            "interceptability_score": self.interceptability_score,
            "is_interceptable": self.is_interceptable,
            "distance_to_target": self.distance_to_target,
            "metadata": dict(self.metadata),
            "scenario_metadata_snapshot": dict(self.scenario_metadata_snapshot),
            "abort_feasibility_score": self.abort_feasibility_score,
            "is_abort_feasible": self.is_abort_feasible,
        }
        if self.true_target_state is not None:
            d["true_target_x"] = self.true_target_state.x
            d["true_target_y"] = self.true_target_state.y
            d["true_target_vx"] = self.true_target_state.vx
            d["true_target_vy"] = self.true_target_state.vy
        else:
            d["true_target_x"] = None
            d["true_target_y"] = None
            d["true_target_vx"] = None
            d["true_target_vy"] = None
        return d

@dataclass
class PerTrialMetrics:
    """M7: per-trial aggregated counts.

    One row per (seed, scenario_type, policy_name). Aggregating counts across
    rows reproduces the global MetricsSummary numbers. Storing raw counts (not
    rates) lets downstream code form arbitrary derived rates and bootstrap them.
    """
    seed: Optional[int]
    scenario_type: str
    policy_name: str
    n_steps: int
    engage_count: int
    track_count: int
    abort_count: int
    unsafe_engagement_count: int
    false_engagement_count: int
    non_interceptable_engagement_count: int
    invalid_engagement_count: int
    safe_engagement_count: int
    safe_opportunity_count: int
    missed_safe_opportunity_count: int
    abort_required: bool
    on_time_abort: bool
    late_abort: bool
    missed_abort: bool
    first_required_step: Optional[int]
    first_abort_step: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "scenario_type": self.scenario_type,
            "policy_name": self.policy_name,
            "n_steps": self.n_steps,
            "engage_count": self.engage_count,
            "track_count": self.track_count,
            "abort_count": self.abort_count,
            "unsafe_engagement_count": self.unsafe_engagement_count,
            "false_engagement_count": self.false_engagement_count,
            "non_interceptable_engagement_count": self.non_interceptable_engagement_count,
            "invalid_engagement_count": self.invalid_engagement_count,
            "safe_engagement_count": self.safe_engagement_count,
            "safe_opportunity_count": self.safe_opportunity_count,
            "missed_safe_opportunity_count": self.missed_safe_opportunity_count,
            "abort_required": self.abort_required,
            "on_time_abort": self.on_time_abort,
            "late_abort": self.late_abort,
            "missed_abort": self.missed_abort,
            "first_required_step": self.first_required_step,
            "first_abort_step": self.first_abort_step,
        }


@dataclass
class MetricsSummary:
    policy_name: Optional[str]
    total_decisions: int
    engage_count: int
    track_count: int
    abort_count: int
    engagement_attempt_rate: float
    track_rate: float
    abort_rate: float
    unsafe_engagement_count: int
    unsafe_engagement_rate: float
    false_engagement_count: int
    false_engagement_rate: float
    non_interceptable_engagement_count: int
    non_interceptable_engagement_rate: float
    invalid_engagement_count: int
    invalid_engagement_rate: float
    safe_engagement_count: int
    safe_engagement_rate: float
    unsafe_engagement_per_attempt_rate: float
    false_engagement_per_attempt_rate: float
    non_interceptable_engagement_per_attempt_rate: float
    invalid_engagement_per_attempt_rate: float
    safe_engagement_per_attempt_rate: float
    reason_counts: Dict[str, int]
    # M4 fix: abort timing and opportunity-capture metrics (docs §3.5-3.10).
    # All defined against ground truth, not policy inputs.
    safe_opportunity_count: int = 0
    missed_safe_opportunity_count: int = 0
    missed_safe_opportunity_rate: float = 0.0
    safe_opportunity_capture_rate: float = 0.0
    abort_required_trial_count: int = 0
    on_time_abort_trial_count: int = 0
    late_abort_trial_count: int = 0
    missed_abort_trial_count: int = 0
    abort_success_rate: float = 0.0
    late_abort_rate: float = 0.0
    missed_abort_rate: float = 0.0
    risk_weighted_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "total_decisions": self.total_decisions,
            "engage_count": self.engage_count,
            "track_count": self.track_count,
            "abort_count": self.abort_count,
            "engagement_attempt_rate": self.engagement_attempt_rate,
            "track_rate": self.track_rate,
            "abort_rate": self.abort_rate,
            "unsafe_engagement_count": self.unsafe_engagement_count,
            "unsafe_engagement_rate": self.unsafe_engagement_rate,
            "false_engagement_count": self.false_engagement_count,
            "false_engagement_rate": self.false_engagement_rate,
            "non_interceptable_engagement_count": self.non_interceptable_engagement_count,
            "non_interceptable_engagement_rate": self.non_interceptable_engagement_rate,
            "invalid_engagement_count": self.invalid_engagement_count,
            "invalid_engagement_rate": self.invalid_engagement_rate,
            "safe_engagement_count": self.safe_engagement_count,
            "safe_engagement_rate": self.safe_engagement_rate,
            "unsafe_engagement_per_attempt_rate": self.unsafe_engagement_per_attempt_rate,
            "false_engagement_per_attempt_rate": self.false_engagement_per_attempt_rate,
            "non_interceptable_engagement_per_attempt_rate": self.non_interceptable_engagement_per_attempt_rate,
            "invalid_engagement_per_attempt_rate": self.invalid_engagement_per_attempt_rate,
            "safe_engagement_per_attempt_rate": self.safe_engagement_per_attempt_rate,
            "reason_counts": dict(self.reason_counts),
            "safe_opportunity_count": self.safe_opportunity_count,
            "missed_safe_opportunity_count": self.missed_safe_opportunity_count,
            "missed_safe_opportunity_rate": self.missed_safe_opportunity_rate,
            "safe_opportunity_capture_rate": self.safe_opportunity_capture_rate,
            "abort_required_trial_count": self.abort_required_trial_count,
            "on_time_abort_trial_count": self.on_time_abort_trial_count,
            "late_abort_trial_count": self.late_abort_trial_count,
            "missed_abort_trial_count": self.missed_abort_trial_count,
            "abort_success_rate": self.abort_success_rate,
            "late_abort_rate": self.late_abort_rate,
            "missed_abort_rate": self.missed_abort_rate,
            "risk_weighted_score": self.risk_weighted_score,
        }

class MetricsLogger:
    """Records policy decisions and summarizes them with safety-oriented metrics.

    C1 fix (ground-truth based evaluation):
    Engagement is classified as unsafe/false/non-interceptable using
    ground-truth information (true target state and scenario label) rather
    than the same noisy/estimated values the policy used as input. This
    prevents the policy's decision rule from being its own grader.

    When ground-truth fields on a DecisionRecord are missing (legacy records),
    the logger falls back to the deprecated policy-input thresholds so old
    callers keep working. New callers should always pass true_target_state
    and scenario_metadata via record_decision.
    """

    def __init__(
        self,
        risk_score_threshold: float = 0.7,
        identification_confidence_threshold: float = 0.6,
        interceptability_threshold: float = 0.5,
        # M4: parameters for abort-timing and composite-score metrics.
        late_abort_lag_steps: int = 2,
        weight_safe_capture: float = 1.0,
        weight_abort_success: float = 0.5,
        weight_unsafe: float = 2.0,
        weight_false: float = 2.0,
        weight_late_abort: float = 1.0,
    ):
        self.risk_score_threshold = risk_score_threshold
        self.identification_confidence_threshold = identification_confidence_threshold
        self.interceptability_threshold = interceptability_threshold
        self.late_abort_lag_steps = late_abort_lag_steps
        self.weight_safe_capture = weight_safe_capture
        self.weight_abort_success = weight_abort_success
        self.weight_unsafe = weight_unsafe
        self.weight_false = weight_false
        self.weight_late_abort = weight_late_abort
        self.records: List[DecisionRecord] = []

    def add_record(self, record: DecisionRecord) -> None:
        self.records.append(record)

    def record_decision(
        self,
        policy_name: str,
        context: DecisionContext,
        action: DecisionAction,
        reason: Optional[str] = None,
        *,
        true_target_state: Optional[State2D] = None,
        scenario_metadata: Optional[Dict[str, object]] = None,
    ) -> DecisionRecord:
        record = DecisionRecord.from_context(
            policy_name,
            context,
            action,
            reason,
            true_target_state=true_target_state,
            scenario_metadata=scenario_metadata,
        )
        self.add_record(record)
        return record

    def get_records(self, policy_name: Optional[str] = None) -> List[DecisionRecord]:
        if policy_name is None:
            return list(self.records)
        return [r for r in self.records if r.policy_name == policy_name]

    def reset(self) -> None:
        self.records.clear()

    def per_trial_summaries(self) -> List[PerTrialMetrics]:
        """M7: per-trial aggregation.

        Groups records by (seed, scenario, policy) and returns one
        PerTrialMetrics per trial. Aggregating across all trials with the
        same policy reproduces the numbers returned by summarize().
        """
        from collections import defaultdict
        trials: Dict[tuple, List[DecisionRecord]] = defaultdict(list)
        for r in self.records:
            trials[self._trial_key(r)].append(r)

        results: List[PerTrialMetrics] = []
        for key, trial_records in trials.items():
            seed, scenario_str, policy_name = key
            trial_sorted = sorted(trial_records, key=lambda r: r.step_index)

            engage_count = 0
            track_count = 0
            abort_count = 0
            unsafe_n = 0
            false_n = 0
            non_int_n = 0
            invalid_n = 0
            safe_engage_n = 0
            safe_opp_n = 0
            missed_safe_n = 0
            first_required = None
            first_abort = None

            for rec in trial_sorted:
                # Opportunity classification
                if self._classify_opportunity(rec):
                    safe_opp_n += 1
                    if rec.action != DecisionAction.ENGAGE:
                        missed_safe_n += 1

                if rec.action == DecisionAction.ENGAGE:
                    engage_count += 1
                    flags = self._classify_engage(rec)
                    if flags["unsafe"]:
                        unsafe_n += 1
                    if flags["false"]:
                        false_n += 1
                    if flags["non_int"]:
                        non_int_n += 1
                    if flags["invalid"]:
                        invalid_n += 1
                    if not any(flags.values()):
                        safe_engage_n += 1
                elif rec.action == DecisionAction.TRACK:
                    track_count += 1
                elif rec.action == DecisionAction.ABORT:
                    abort_count += 1

                if first_required is None and self._abort_required(rec):
                    first_required = rec.step_index
                if first_abort is None and rec.action == DecisionAction.ABORT:
                    first_abort = rec.step_index

            abort_required = first_required is not None
            on_time = False
            late = False
            missed = False
            if abort_required:
                if first_abort is None:
                    missed = True
                elif first_abort > first_required + self.late_abort_lag_steps:
                    late = True
                else:
                    on_time = True

            results.append(PerTrialMetrics(
                seed=seed,
                scenario_type=scenario_str,
                policy_name=policy_name,
                n_steps=len(trial_sorted),
                engage_count=engage_count,
                track_count=track_count,
                abort_count=abort_count,
                unsafe_engagement_count=unsafe_n,
                false_engagement_count=false_n,
                non_interceptable_engagement_count=non_int_n,
                invalid_engagement_count=invalid_n,
                safe_engagement_count=safe_engage_n,
                safe_opportunity_count=safe_opp_n,
                missed_safe_opportunity_count=missed_safe_n,
                abort_required=abort_required,
                on_time_abort=on_time,
                late_abort=late,
                missed_abort=missed,
                first_required_step=first_required,
                first_abort_step=first_abort,
            ))
        return results

    def _true_risk_score(self, record: DecisionRecord) -> Optional[float]:
        """Compute risk score from ground truth target position.

        Mirrors EnvironmentalRiskEvaluator's clearance-to-score formula but
        applied to the actual target state at decision time rather than the
        prediction from estimated state. Returns None if the scenario has no
        risk zone or no ground-truth state is available.
        """
        if record.true_target_state is None:
            return None
        snap = record.scenario_metadata_snapshot
        if "risk_zone_center" not in snap or "risk_zone_radius" not in snap:
            return None
        zone_center = snap["risk_zone_center"]
        zone_radius = snap["risk_zone_radius"]
        true_dist = math.hypot(
            record.true_target_state.x - zone_center[0],
            record.true_target_state.y - zone_center[1],
        )
        clearance = true_dist - zone_radius
        if clearance <= 0:
            return 1.0
        if clearance >= 2.0:
            return 0.0
        return 1.0 - (clearance / 2.0)

    def _scenario_type_str(self, record: DecisionRecord) -> str:
        st = record.scenario_type
        return st.value if hasattr(st, "value") else str(st)

    def _abort_required(self, record: DecisionRecord) -> bool:
        """Ground-truth-based: should the policy have aborted at this step?

        Aligns with docs §12 ABORT conditions but uses GT (not policy inputs):
          - tracking has been lost (detection failed) → abort
          - true target is in/near the true risk zone → abort
        """
        if not record.detected:
            return True
        true_risk = self._true_risk_score(record)
        if true_risk is not None and true_risk >= self.risk_score_threshold:
            return True
        return False

    def _classify_opportunity(self, record: DecisionRecord) -> bool:
        """Would ENGAGE at this step have been a "safe opportunity"?

        This is the engage classifier without conditioning on action: a record
        is a safe opportunity if hypothetically engaging would NOT have been
        unsafe / false / non-interceptable / invalid.
        """
        # Reuse the engage classifier; the four flags do not depend on the
        # action that was actually taken.
        flags = self._classify_engage(record)
        return not (flags["unsafe"] or flags["false"] or flags["non_int"] or flags["invalid"])

    def _trial_key(self, record: DecisionRecord) -> tuple:
        seed = record.metadata.get("seed", None) if isinstance(record.metadata, dict) else None
        scenario_str = self._scenario_type_str(record)
        return (seed, scenario_str, record.policy_name)

    def _classify_engage(self, record: DecisionRecord) -> Dict[str, bool]:
        """Classify an ENGAGE record into (unsafe, false, non_int, invalid).

        Uses ground truth when available; falls back to deprecated policy-input
        thresholds when GT fields are absent on the record.
        """
        # Invalid is structural and does not need ground truth.
        invalid = (
            not record.detected
            or not record.estimate_valid
            or not record.prediction_valid
        )

        has_gt = record.true_target_state is not None or bool(record.scenario_metadata_snapshot)

        if has_gt:
            scenario_str = self._scenario_type_str(record)
            snap = record.scenario_metadata_snapshot

            # Unsafe: GT risk score from true target vs true risk zone.
            true_risk = self._true_risk_score(record)
            unsafe = (true_risk is not None) and (true_risk >= self.risk_score_threshold)

            # False engagement: scenario is labelled as low-confidence
            # identification (engaging on an ambiguous target is "false").
            # Journal Phase 4: generalized to a metadata flag so extended
            # scenarios (e.g. S9 friendly proximity) also count as
            # misidentification risk. Default (no flag) keeps S3-only behaviour.
            false_eng = (
                "S3_LOW_IDENTIFICATION_CONFIDENCE" in scenario_str
                or bool(snap.get("misidentification_risk", False))
            )

            # Non-interceptable: scenario is labelled non-interceptable.
            non_int = (
                "S6_NON_INTERCEPTABLE_TARGET" in scenario_str
                or snap.get("interceptability_label") == "non_interceptable"
            )
        else:
            # Deprecated fallback (the C1-violating logic) for legacy callers
            # that do not provide ground-truth fields. Tests and the runner
            # should provide GT so this branch is not exercised in normal use.
            unsafe = record.is_high_risk or record.risk_score >= self.risk_score_threshold
            false_eng = record.identification_confidence < self.identification_confidence_threshold
            non_int = (
                not record.is_interceptable
                or record.interceptability_score < self.interceptability_threshold
            )

        return {"unsafe": unsafe, "false": false_eng, "non_int": non_int, "invalid": invalid}

    def summarize(self, policy_name: Optional[str] = None) -> MetricsSummary:
        """Summarize recorded decisions into a MetricsSummary.

        unsafe / false / non_interceptable / invalid are not mutually exclusive
        (a single ENGAGE can be counted in multiple categories). safe_engagement
        is the complement: an ENGAGE that triggered none of the four flags.
        """
        records_to_summarize = self.get_records(policy_name)

        total_decisions = len(records_to_summarize)

        engage_count = 0
        track_count = 0
        abort_count = 0

        unsafe_engagement_count = 0
        false_engagement_count = 0
        non_interceptable_engagement_count = 0
        invalid_engagement_count = 0
        safe_engagement_count = 0

        # M4: record-level opportunity tracking
        safe_opportunity_count = 0
        missed_safe_opportunity_count = 0

        reason_counts: Dict[str, int] = {}

        for record in records_to_summarize:
            # Opportunity classification independent of action
            if self._classify_opportunity(record):
                safe_opportunity_count += 1
                if record.action != DecisionAction.ENGAGE:
                    missed_safe_opportunity_count += 1

            if record.action == DecisionAction.ENGAGE:
                engage_count += 1
                flags = self._classify_engage(record)
                if flags["unsafe"]:
                    unsafe_engagement_count += 1
                if flags["false"]:
                    false_engagement_count += 1
                if flags["non_int"]:
                    non_interceptable_engagement_count += 1
                if flags["invalid"]:
                    invalid_engagement_count += 1
                if not any(flags.values()):
                    safe_engagement_count += 1
            elif record.action == DecisionAction.TRACK:
                track_count += 1
            elif record.action == DecisionAction.ABORT:
                abort_count += 1

            if record.reason is not None:
                reason_counts[record.reason] = reason_counts.get(record.reason, 0) + 1

        # M4: trial-level abort timing. Group records by (seed, scenario, policy)
        # and classify each trial's abort behaviour against the first step where
        # GT required an abort.
        from collections import defaultdict
        trials: Dict[tuple, List[DecisionRecord]] = defaultdict(list)
        for r in records_to_summarize:
            trials[self._trial_key(r)].append(r)

        abort_required_trials = 0
        on_time_abort_trials = 0
        late_abort_trials = 0
        missed_abort_trials = 0

        for _key, trial_records in trials.items():
            trial_sorted = sorted(trial_records, key=lambda r: r.step_index)
            first_required = None
            first_abort = None
            for r in trial_sorted:
                if first_required is None and self._abort_required(r):
                    first_required = r.step_index
                if first_abort is None and r.action == DecisionAction.ABORT:
                    first_abort = r.step_index
                if first_required is not None and first_abort is not None:
                    break

            if first_required is None:
                # No abort needed in this trial; not counted in abort metrics.
                continue

            abort_required_trials += 1
            if first_abort is None:
                missed_abort_trials += 1
            elif first_abort > first_required + self.late_abort_lag_steps:
                late_abort_trials += 1
            else:
                on_time_abort_trials += 1

        def safe_rate(count: int) -> float:
            return count / total_decisions if total_decisions > 0 else 0.0

        def per_attempt_rate(count: int) -> float:
            return count / engage_count if engage_count > 0 else 0.0

        # M4 derived rates
        missed_safe_rate = (
            missed_safe_opportunity_count / safe_opportunity_count
            if safe_opportunity_count > 0
            else 0.0
        )
        safe_opportunity_capture_rate = 1.0 - missed_safe_rate if safe_opportunity_count > 0 else 0.0
        abort_success_rate = (
            on_time_abort_trials / abort_required_trials
            if abort_required_trials > 0
            else 0.0
        )
        late_abort_rate = (
            late_abort_trials / abort_required_trials
            if abort_required_trials > 0
            else 0.0
        )
        missed_abort_rate = (
            missed_abort_trials / abort_required_trials
            if abort_required_trials > 0
            else 0.0
        )

        # Composite score (docs §3.10). Rates used:
        #   - safe_opportunity_capture_rate (positive: rewards capturing safe ENGAGEs)
        #   - abort_success_rate (positive: rewards timely abort)
        #   - unsafe_engagement_per_attempt_rate (penalty)
        #   - false_engagement_per_attempt_rate (penalty)
        #   - late_abort_rate (penalty for being too slow to abort)
        risk_weighted_score = (
            self.weight_safe_capture * safe_opportunity_capture_rate
            + self.weight_abort_success * abort_success_rate
            - self.weight_unsafe * per_attempt_rate(unsafe_engagement_count)
            - self.weight_false * per_attempt_rate(false_engagement_count)
            - self.weight_late_abort * late_abort_rate
        )

        return MetricsSummary(
            policy_name=policy_name,
            total_decisions=total_decisions,
            engage_count=engage_count,
            track_count=track_count,
            abort_count=abort_count,
            engagement_attempt_rate=safe_rate(engage_count),
            track_rate=safe_rate(track_count),
            abort_rate=safe_rate(abort_count),
            unsafe_engagement_count=unsafe_engagement_count,
            unsafe_engagement_rate=safe_rate(unsafe_engagement_count),
            false_engagement_count=false_engagement_count,
            false_engagement_rate=safe_rate(false_engagement_count),
            non_interceptable_engagement_count=non_interceptable_engagement_count,
            non_interceptable_engagement_rate=safe_rate(non_interceptable_engagement_count),
            invalid_engagement_count=invalid_engagement_count,
            invalid_engagement_rate=safe_rate(invalid_engagement_count),
            safe_engagement_count=safe_engagement_count,
            safe_engagement_rate=safe_rate(safe_engagement_count),
            unsafe_engagement_per_attempt_rate=per_attempt_rate(unsafe_engagement_count),
            false_engagement_per_attempt_rate=per_attempt_rate(false_engagement_count),
            non_interceptable_engagement_per_attempt_rate=per_attempt_rate(non_interceptable_engagement_count),
            invalid_engagement_per_attempt_rate=per_attempt_rate(invalid_engagement_count),
            safe_engagement_per_attempt_rate=per_attempt_rate(safe_engagement_count),
            reason_counts=reason_counts,
            safe_opportunity_count=safe_opportunity_count,
            missed_safe_opportunity_count=missed_safe_opportunity_count,
            missed_safe_opportunity_rate=missed_safe_rate,
            safe_opportunity_capture_rate=safe_opportunity_capture_rate,
            abort_required_trial_count=abort_required_trials,
            on_time_abort_trial_count=on_time_abort_trials,
            late_abort_trial_count=late_abort_trials,
            missed_abort_trial_count=missed_abort_trials,
            abort_success_rate=abort_success_rate,
            late_abort_rate=late_abort_rate,
            missed_abort_rate=missed_abort_rate,
            risk_weighted_score=risk_weighted_score,
        )
