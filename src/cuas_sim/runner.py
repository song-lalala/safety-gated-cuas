import random
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any

from .scenario import ScenarioType, get_all_scenario_types, get_scenario_factory
from .observation import ObservationModel
from .estimation import TargetStateEstimator, KalmanTargetEstimator
from .guidance import PNParams, simulate_pn_intercept
from .prediction import ConstantVelocityPredictor
from .evaluators import (
    EnvironmentalRiskEvaluator,
    InterceptabilityEvaluator,
    AbortFeasibilityEvaluator,
)
from .policies import (
    AlwaysEngagePolicy,
    DecisionAction,
    DistanceOnlyPolicy,
    InterceptabilityOnlyPolicy,
    BayesRiskPolicy,
    ReachabilityFilterPolicy,
    LatchingPolicy,
    SafetyGatePolicy,
    build_decision_context,
)
from .metrics import DecisionRecord, MetricsSummary, MetricsLogger, PerTrialMetrics
from .simulator import Simulator

@dataclass
class MonteCarloConfig:
    seeds: List[int]
    steps_per_trial: int = 20
    prediction_horizon_steps: int = 5
    s4_prediction_horizon_steps: int = 30
    distance_engage_threshold: float = 5.0
    safety_tracking_abort_threshold: float = 0.3
    safety_tracking_confidence_threshold: float = 0.6
    safety_identification_confidence_threshold: float = 0.6
    safety_risk_score_threshold: float = 0.7
    safety_interceptability_threshold: float = 0.5
    # C3-a: SafetyGate 5th input — abort feasibility threshold.
    safety_abort_feasibility_threshold: float = 0.5
    # C3: AbortFeasibilityEvaluator tuning.
    abort_horizon_steps: int = 10
    abort_risk_zone_safety_buffer: float = 0.5
    # M3: EnvironmentalRiskEvaluator tuning (R_eff = R_zone + k_sigma*sigma_pred + abort_margin).
    risk_k_sigma: float = 1.0
    risk_abort_margin: float = 0.0
    risk_clearance_falloff: float = 2.0
    # C2: observation noise parameters
    observation_noise_sigma: float = 0.0
    observation_velocity_noise_sigma: float = 0.0
    # Journal Phase 1: estimator selection. "alpha_beta" (default) reproduces
    # the Phase-0 baseline exactly; "kalman" uses the covariance-carrying
    # Kalman filter with NIS-based tracking confidence (C_track = exp(-½·NIS)).
    estimator_mode: str = "alpha_beta"
    kalman_process_noise: float = 0.1
    # None → use observation_noise_sigma as the per-axis measurement std (floored).
    kalman_measurement_sigma: Optional[float] = None
    kalman_min_measurement_sigma: float = 1e-2
    kalman_init_velocity_var: float = 1.0
    # Journal Phase 2: R_env mode. "linear" (default) reproduces the baseline;
    # "probability" uses the propagated-covariance intrusion probability.
    r_env_mode: str = "linear"
    # Journal Phase 2 (eval decoupling): the ground-truth "unsafe" criterion
    # threshold, kept SEPARATE from the policy's decision threshold so the
    # evaluation yardstick stays fixed while the policy threshold varies.
    # None → fall back to safety_risk_score_threshold (baseline: both = 0.7,
    # byte-identical). For probability-mode experiments, fix this at 0.7 and
    # vary only the policy's safety_risk_score_threshold.
    eval_risk_score_threshold: Optional[float] = None
    # Physical keep-out safety margin b: probability R_env integrates over the
    # disk of radius (R_zone + b). b=0.6 matches the evaluation's unsafe region
    # at eval θ=0.7 (clearance ≤ 2·(1−0.7) = 0.6). Only used in probability mode.
    risk_keepout_buffer: float = 0.6
    # Review response (W11, Reviewer 2.7): quadrature for the intrusion
    # probability. "simpson" is as submitted; "trig" removes the disk-edge
    # singularity and is exact to machine precision at the same node count.
    renv_quadrature_rule: str = "simpson"
    # Journal Phase 3 (M1): interceptor kinematic capability model. Makes I and
    # A_abort physical (pursuit-evasion capture time / abort dynamics) instead of
    # heuristic. Defaults reproduce the baseline (heuristic / time_proxy, stationary
    # launch platform). The interceptor is NOT flown closed-loop here — that
    # (PN trajectories) is Phase 3b / future work; this models its *capability*.
    i_mode: str = "heuristic"          # "heuristic" | "capturability"
    a_mode: str = "time_proxy"         # "time_proxy" | "dynamics"
    interceptor_max_speed: float = 2.0       # V_I
    interceptor_max_accel: float = 4.0       # a_max (abort decel)
    interceptor_reaction_delay: float = 0.2  # τ
    # T_engage (s) for capturability. 20 ≈ engagement-area crossing time at the
    # scenario scale (distances ~10–15, speeds ~1). 10 was too short for the
    # longer-range S5 (capture ~5s → I≈0.5 at threshold → spurious TRACK); ≥15
    # removes that artifact and capturability then matches/exceeds the heuristic.
    interceptor_engagement_horizon_time: float = 20.0
    # M4: composite-score and abort-timing tuning
    late_abort_lag_steps: int = 2
    weight_safe_capture: float = 1.0
    weight_abort_success: float = 0.5
    weight_unsafe: float = 2.0
    weight_false: float = 2.0
    weight_late_abort: float = 1.0
    # Review response (W4): robustness variants of the tracking-confidence
    # signal and of the ABORT decision. All three defaults reproduce the
    # submitted paper exactly.
    #   ctrack_window      W of the time-averaged NIS (1 = single sample)
    #   nis_gate_threshold chi-square measurement gate (None = no gating;
    #                      9.21 is the chi-square(2) 99% point)
    #   abort_latching     make ABORT absorbing within a trial
    ctrack_window: int = 1
    nis_gate_threshold: Optional[float] = None
    abort_latching: bool = False
    # Review response (W3, Reviewer 2.11): non-Gaussian sensor error. Defaults
    # reproduce the submitted zero-mean isotropic Gaussian model exactly.
    observation_noise_model: str = "gaussian"   # "gaussian" | "student_t"
    observation_noise_dof: float = 3.0
    observation_bias: float = 0.0
    observation_outlier_rate: float = 0.0
    observation_outlier_scale: float = 5.0
    # Review response (W10, Reviewer 2.8): identification-confidence pipeline
    # distortion. Identity by default.
    cid_bias: float = 0.0
    cid_gamma: float = 1.0
    cid_noise_sigma: float = 0.0
    # Review response (W6, Reviewers 1.1 / 2.2): fly a closed-loop PN interceptor
    # at every ENGAGE and record what it actually achieves. Off by default; the
    # outcome is recorded for analysis and never read by any policy.
    pn_validation: bool = False
    pn_nav_constant: float = 4.0
    pn_capture_radius: float = 0.3
    include_scenarios: Optional[List[ScenarioType]] = None
    include_policies: Optional[List[str]] = None

def get_default_policy_factories(
    distance_engage_threshold: float = 5.0,
    safety_tracking_abort_threshold: float = 0.3,
    safety_tracking_confidence_threshold: float = 0.6,
    safety_identification_confidence_threshold: float = 0.6,
    safety_risk_score_threshold: float = 0.7,
    safety_interceptability_threshold: float = 0.5,
    safety_abort_feasibility_threshold: float = 0.5,
    abort_latching: bool = False,
    weight_safe_capture: float = 1.0,
    weight_abort_success: float = 0.5,
    weight_unsafe: float = 2.0,
    weight_false: float = 2.0,
    keepout_buffer: float = 0.6,
) -> Dict[str, Callable[[], Any]]:
    def safety_gate():
        gate = SafetyGatePolicy(
            tracking_abort_threshold=safety_tracking_abort_threshold,
            tracking_confidence_threshold=safety_tracking_confidence_threshold,
            identification_confidence_threshold=safety_identification_confidence_threshold,
            risk_score_threshold=safety_risk_score_threshold,
            interceptability_threshold=safety_interceptability_threshold,
            abort_feasibility_threshold=safety_abort_feasibility_threshold,
        )
        # A fresh instance per trial, so the latch carries no state across trials.
        return LatchingPolicy(gate) if abort_latching else gate

    return {
        "AlwaysEngagePolicy": lambda: AlwaysEngagePolicy(),
        "DistanceOnlyPolicy": lambda: DistanceOnlyPolicy(engage_distance_threshold=distance_engage_threshold),
        "InterceptabilityOnlyPolicy": lambda: InterceptabilityOnlyPolicy(),
        "SafetyGatePolicy": safety_gate,
        # Review response (W12, Reviewer 2.9): modern decision baselines. They
        # are opt-in through include_policies, so no existing run changes.
        "BayesRiskPolicy": lambda: BayesRiskPolicy(
            weight_safe_capture=weight_safe_capture,
            weight_abort_success=weight_abort_success,
            weight_unsafe=weight_unsafe,
            weight_false=weight_false,
        ),
        "ReachabilityFilterPolicy": lambda: ReachabilityFilterPolicy(
            identification_confidence_threshold=safety_identification_confidence_threshold,
            interceptability_threshold=safety_interceptability_threshold,
            keepout_buffer=keepout_buffer,
        ),
        # Same filter, but keeping the gate's tracking-confidence branches, so
        # the only difference from SafetyGatePolicy is worst-case vs
        # probabilistic risk.
        "ReachabilityWithTrackingPolicy": lambda: ReachabilityFilterPolicy(
            identification_confidence_threshold=safety_identification_confidence_threshold,
            interceptability_threshold=safety_interceptability_threshold,
            keepout_buffer=keepout_buffer,
            tracking_abort_threshold=safety_tracking_abort_threshold,
            tracking_confidence_threshold=safety_tracking_confidence_threshold,
        ),
    }

@dataclass
class MonteCarloResult:
    records: List[DecisionRecord]
    summaries: List[MetricsSummary]
    # M4: propagate scoring parameters so grouped_summaries_to_dicts() uses the
    # same weights/thresholds as the top-level summary instead of defaults.
    logger_kwargs: Dict[str, Any] = field(default_factory=dict)

    def records_to_dicts(self) -> List[Dict[str, object]]:
        return [record.to_dict() for record in self.records]

    def summaries_to_dicts(self) -> List[Dict[str, object]]:
        return [summary.to_dict() for summary in self.summaries]

    def per_trial_summaries(self) -> List[PerTrialMetrics]:
        """M7: per-trial aggregated metrics for downstream bootstrap CI work."""
        logger = MetricsLogger(**self.logger_kwargs) if self.logger_kwargs else MetricsLogger()
        logger.records = list(self.records)
        return logger.per_trial_summaries()

    def per_trial_summaries_to_dicts(self) -> List[Dict[str, object]]:
        return [t.to_dict() for t in self.per_trial_summaries()]

    def grouped_summaries_to_dicts(self) -> List[Dict[str, object]]:
        from collections import defaultdict

        # Group records by (policy_name, scenario_type)
        grouped_records = defaultdict(list)
        for record in self.records:
            st = record.scenario_type
            st_val = st.value if hasattr(st, "value") else str(st)
            key = (record.policy_name, st_val)
            grouped_records[key].append(record)

        results = []
        for (policy_name, scenario_type_val), group in grouped_records.items():
            # Use a fresh logger for this group, but with the same configured
            # thresholds/weights as the run that produced the records.
            logger = MetricsLogger(**self.logger_kwargs) if self.logger_kwargs else MetricsLogger()
            logger.records = group
            summary = logger.summarize(policy_name=policy_name)

            summary_dict = summary.to_dict()
            summary_dict["scenario_type"] = scenario_type_val
            
            results.append(summary_dict)
            
        return results

class MonteCarloRunner:
    def _fly_engagement(self, scenario, step_index: int, context) -> Dict[str, Any]:
        """Fly the PN interceptor from this ENGAGE and report what it achieved.

        The interceptor launches from its own true position against the true
        target — this is evaluation, which the manuscript already allows to see
        ground truth, not decision information.
        """
        # The interceptability evaluator lets a scenario override the launch
        # speed (S6 randomises it per seed). The flown interceptor must be the
        # same vehicle the gate reasoned about, or this measures the mismatch
        # instead of the guidance.
        v_i = scenario.metadata.get("interceptor_nominal_max_speed",
                                    self.config.interceptor_max_speed)
        params = PNParams(
            nav_constant=self.config.pn_nav_constant,
            max_speed=v_i,
            max_accel=self.config.interceptor_max_accel,
            reaction_delay=self.config.interceptor_reaction_delay,
            engagement_horizon=self.config.interceptor_engagement_horizon_time,
            capture_radius=self.config.pn_capture_radius,
        )
        # Replay a scenario-scheduled maneuver if it falls after this instant,
        # so S2/S10 are not quietly flown as constant-velocity targets.
        maneuver = None
        m_step = scenario.metadata.get("maneuver_step")
        if m_step is not None and m_step > step_index:
            dt = scenario.config.time_step
            maneuver = ((m_step - step_index) * dt,
                        float(scenario.metadata.get("dvx", 0.0)),
                        float(scenario.metadata.get("dvy", 0.0)))
        outcome = simulate_pn_intercept(
            scenario.target.state,
            (scenario.interceptor.state.x, scenario.interceptor.state.y),
            params,
            maneuver,
        )
        return {
            "captured": outcome.captured,
            "miss_distance": outcome.miss_distance,
            "time_to_go": outcome.time_to_go,
            "flight_time": outcome.flight_time,
            "interceptability": context.interceptability_score,
            "abort_feasibility": context.abort_feasibility_score,
        }

    def __init__(self, config: MonteCarloConfig):
        self.config = config
        self.policy_factories = get_default_policy_factories(
            distance_engage_threshold=config.distance_engage_threshold,
            safety_tracking_abort_threshold=config.safety_tracking_abort_threshold,
            safety_tracking_confidence_threshold=config.safety_tracking_confidence_threshold,
            safety_identification_confidence_threshold=config.safety_identification_confidence_threshold,
            safety_risk_score_threshold=config.safety_risk_score_threshold,
            safety_interceptability_threshold=config.safety_interceptability_threshold,
            safety_abort_feasibility_threshold=config.safety_abort_feasibility_threshold,
            abort_latching=config.abort_latching,
            weight_safe_capture=config.weight_safe_capture,
            weight_abort_success=config.weight_abort_success,
            weight_unsafe=config.weight_unsafe,
            weight_false=config.weight_false,
            keepout_buffer=config.risk_keepout_buffer,
        )

    def _noise_seed(self, seed: int, scenario_idx: int) -> int:
        # Same noise seed for all policies running the same (seed, scenario)
        # combo so the cross-policy comparison stays fair: every policy sees
        # identical noisy observations on identical trials. Different scenarios
        # under the same seed get different noise sequences so we don't conflate
        # noise patterns across scenarios.
        return seed * 1000 + scenario_idx

    def run(self) -> MonteCarloResult:
        scenario_types = self.config.include_scenarios
        if scenario_types is None:
            scenario_types = get_all_scenario_types()

        policy_names = self.config.include_policies
        if policy_names is None:
            policy_names = list(self.policy_factories.keys())

        scenario_idx_map = {st: i for i, st in enumerate(scenario_types)}

        # Journal Phase 2 (eval decoupling): the evaluation's ground-truth
        # "unsafe" threshold is kept SEPARATE from the policy's decision
        # threshold. Defaults to the policy threshold → baseline byte-identical;
        # fix it (e.g. 0.7) while varying the policy threshold for a fair
        # comparison across risk models.
        eval_thr = (
            self.config.eval_risk_score_threshold
            if self.config.eval_risk_score_threshold is not None
            else self.config.safety_risk_score_threshold
        )

        # M4: propagate score weights and abort-timing lag from config so the
        # composite metric reflects the user's chosen weighting.
        logger = MetricsLogger(
            risk_score_threshold=eval_thr,
            identification_confidence_threshold=self.config.safety_identification_confidence_threshold,
            interceptability_threshold=self.config.safety_interceptability_threshold,
            late_abort_lag_steps=self.config.late_abort_lag_steps,
            weight_safe_capture=self.config.weight_safe_capture,
            weight_abort_success=self.config.weight_abort_success,
            weight_unsafe=self.config.weight_unsafe,
            weight_false=self.config.weight_false,
            weight_late_abort=self.config.weight_late_abort,
        )

        for trial_index, seed in enumerate(self.config.seeds):
            for scenario_type in scenario_types:
                for policy_name in policy_names:

                    if policy_name not in self.policy_factories:
                        raise KeyError(f"Policy '{policy_name}' not found in default policy factories.")

                    # Fresh objects for each trial/scenario/policy
                    scenario_factory = get_scenario_factory(scenario_type)
                    scenario = scenario_factory(seed=seed)

                    policy = self.policy_factories[policy_name]()

                    # C2: seeded RNG for observation noise, identical across
                    # policies for the same (seed, scenario) → fair comparison.
                    noise_rng = random.Random(self._noise_seed(seed, scenario_idx_map[scenario_type]))
                    obs_model = ObservationModel(
                        noise_sigma=self.config.observation_noise_sigma,
                        velocity_noise_sigma=self.config.observation_velocity_noise_sigma,
                        rng=noise_rng,
                        noise_model=self.config.observation_noise_model,
                        noise_dof=self.config.observation_noise_dof,
                        bias=self.config.observation_bias,
                        outlier_rate=self.config.observation_outlier_rate,
                        outlier_scale=self.config.observation_outlier_scale,
                        cid_bias=self.config.cid_bias,
                        cid_gamma=self.config.cid_gamma,
                        cid_noise_sigma=self.config.cid_noise_sigma,
                    )
                    if self.config.estimator_mode == "kalman":
                        meas_sigma = self.config.kalman_measurement_sigma
                        if meas_sigma is None:
                            meas_sigma = self.config.observation_noise_sigma
                        estimator = KalmanTargetEstimator(
                            time_step=scenario.config.time_step,
                            process_noise=self.config.kalman_process_noise,
                            measurement_sigma=meas_sigma,
                            min_measurement_sigma=self.config.kalman_min_measurement_sigma,
                            init_velocity_var=self.config.kalman_init_velocity_var,
                            ctrack_window=self.config.ctrack_window,
                            nis_gate_threshold=self.config.nis_gate_threshold,
                        )
                    elif self.config.estimator_mode == "alpha_beta":
                        estimator = TargetStateEstimator(time_step=scenario.config.time_step)
                    else:
                        raise ValueError(
                            f"Unknown estimator_mode: {self.config.estimator_mode!r} "
                            "(expected 'alpha_beta' or 'kalman')"
                        )
                    risk_evaluator = EnvironmentalRiskEvaluator(
                        high_risk_threshold=self.config.safety_risk_score_threshold,
                        k_sigma=self.config.risk_k_sigma,
                        abort_margin=self.config.risk_abort_margin,
                        clearance_falloff=self.config.risk_clearance_falloff,
                        r_env_mode=self.config.r_env_mode,
                        keepout_buffer=self.config.risk_keepout_buffer,
                        quadrature_rule=self.config.renv_quadrature_rule,
                    )
                    intercept_evaluator = InterceptabilityEvaluator(
                        i_mode=self.config.i_mode,
                        engagement_horizon_time=self.config.interceptor_engagement_horizon_time,
                        default_v_interceptor=self.config.interceptor_max_speed,
                    )
                    abort_evaluator = AbortFeasibilityEvaluator(
                        abort_horizon_steps=self.config.abort_horizon_steps,
                        risk_zone_safety_buffer=self.config.abort_risk_zone_safety_buffer,
                        feasibility_threshold=self.config.safety_abort_feasibility_threshold,
                        a_mode=self.config.a_mode,
                        interceptor_max_speed=self.config.interceptor_max_speed,
                        interceptor_max_accel=self.config.interceptor_max_accel,
                        interceptor_reaction_delay=self.config.interceptor_reaction_delay,
                    )
                    
                    sim = Simulator(
                        config=scenario.config,
                        target=scenario.target,
                        interceptor=scenario.interceptor,
                        step_hook=scenario.step_hook
                    )
                    
                    for step_index in range(self.config.steps_per_trial):
                        # 1. Observation
                        obs = obs_model.observe(scenario, step_index)
                        
                        # 2. Estimation
                        estimate = estimator.estimate(obs)
                        
                        # 3. Prediction
                        horizon = self.config.prediction_horizon_steps
                        if scenario_type == ScenarioType.S4_RISK_ZONE_PROXIMITY:
                            horizon = self.config.s4_prediction_horizon_steps
                        elif "prediction_horizon_steps" in scenario.metadata:
                            # Journal Phase 4: extended risk-zone scenarios (S7/S9/S10)
                            # request a longer horizon via metadata. Core S1–S6 lack
                            # this key → unchanged.
                            horizon = scenario.metadata["prediction_horizon_steps"]
                            
                        predictor_q = (
                            self.config.kalman_process_noise
                            if self.config.estimator_mode == "kalman"
                            else None
                        )
                        predictor = ConstantVelocityPredictor(
                            horizon_steps=horizon, process_noise=predictor_q
                        )
                        prediction = predictor.predict(estimate, scenario.config.time_step)
                        
                        # 4. Evaluators
                        risk_assessment = risk_evaluator.evaluate(scenario, prediction)
                        intercept_assessment = intercept_evaluator.evaluate(scenario, estimate)
                        abort_assessment = abort_evaluator.evaluate(scenario, prediction)

                        # 5. Context & Decision
                        context = build_decision_context(
                            scenario=scenario,
                            observation=obs,
                            estimate=estimate,
                            prediction=prediction,
                            risk_assessment=risk_assessment,
                            interceptability_assessment=intercept_assessment,
                            abort_feasibility_assessment=abort_assessment,
                        )
                        
                        # M5: all policies now expose explain(); use it uniformly so
                        # baseline records also carry a `reason` (e.g. fallback_not_detected).
                        if hasattr(policy, "explain"):
                            action, reason = policy.explain(context)
                        else:
                            action = policy.decide(context)
                            reason = None
                            
                        # 6. Logging — pass ground-truth target state and scenario metadata
                        # so the evaluation in MetricsLogger.summarize() can classify
                        # ENGAGE outcomes against the true world, not against the same
                        # noisy values the policy used as input (C1 fix).
                        record = logger.record_decision(
                            policy_name,
                            context,
                            action,
                            reason,
                            true_target_state=scenario.target.state,
                            scenario_metadata=scenario.metadata,
                        )
                        record.metadata["trial_index"] = trial_index
                        record.metadata["seed"] = seed
                        record.metadata["scenario_type"] = scenario_type.value if hasattr(scenario_type, "value") else str(scenario_type)
                        record.metadata["policy_name"] = policy_name

                        # 6b. W6 validation layer: whenever the gate commits to
                        # contact, actually fly the interceptor against the true
                        # target. This never feeds back into the decision — it
                        # exists to test whether the capability signals predict
                        # what a bounded-acceleration pursuit really achieves.
                        if self.config.pn_validation and action == DecisionAction.ENGAGE:
                            record.metadata["pn"] = self._fly_engagement(
                                scenario, step_index, context
                            )

                        # 7. Simulator step
                        sim.step()
                        
        # After run completes, collect summaries
        summaries = []
        for policy_name in policy_names:
            summaries.append(logger.summarize(policy_name=policy_name))
            
        logger_kwargs = {
            "risk_score_threshold": eval_thr,
            "identification_confidence_threshold": self.config.safety_identification_confidence_threshold,
            "interceptability_threshold": self.config.safety_interceptability_threshold,
            "late_abort_lag_steps": self.config.late_abort_lag_steps,
            "weight_safe_capture": self.config.weight_safe_capture,
            "weight_abort_success": self.config.weight_abort_success,
            "weight_unsafe": self.config.weight_unsafe,
            "weight_false": self.config.weight_false,
            "weight_late_abort": self.config.weight_late_abort,
        }
        return MonteCarloResult(
            records=logger.get_records(),
            summaries=summaries,
            logger_kwargs=logger_kwargs,
        )
