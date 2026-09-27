import unittest
from cuas_sim.scenario import (
    get_s1_scenario,
    get_s3_scenario,
    get_s4_scenario,
    get_s5_scenario,
    get_s6_scenario
)
from cuas_sim.observation import ObservationModel
from cuas_sim.estimation import TargetStateEstimator
from cuas_sim.prediction import ConstantVelocityPredictor
from cuas_sim.evaluators import EnvironmentalRiskEvaluator, InterceptabilityEvaluator
from cuas_sim.policies import build_decision_context, DecisionAction
from cuas_sim.metrics import DecisionRecord, MetricsLogger

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()
        self.risk_evaluator = EnvironmentalRiskEvaluator()
        self.intercept_evaluator = InterceptabilityEvaluator()
        
    def _build_context_for_scenario(self, scenario, step_index=0, horizon_steps=5):
        predictor = ConstantVelocityPredictor(horizon_steps=horizon_steps)
        
        obs = self.obs_model.observe(scenario, step_index=step_index)
        estimate = self.estimator.estimate(obs)
        prediction = predictor.predict(estimate, scenario.config.time_step)
        risk_assessment = self.risk_evaluator.evaluate(scenario, prediction)
        intercept_assessment = self.intercept_evaluator.evaluate(scenario, estimate)
        
        return build_decision_context(
            scenario=scenario,
            observation=obs,
            estimate=estimate,
            prediction=prediction,
            risk_assessment=risk_assessment,
            interceptability_assessment=intercept_assessment
        )

    def test_from_context(self):
        scenario = get_s1_scenario(seed=42)
        context = self._build_context_for_scenario(scenario)
        record = DecisionRecord.from_context("SafetyGatePolicy", context, DecisionAction.ENGAGE, "engage_conditions_satisfied")
        
        self.assertEqual(record.policy_name, "SafetyGatePolicy")
        self.assertEqual(record.action, DecisionAction.ENGAGE)
        self.assertEqual(record.reason, "engage_conditions_satisfied")
        self.assertEqual(record.scenario_type, context.scenario_type)
        self.assertIsInstance(record.metadata, dict)
        # metadata shallow copy check
        self.assertIsNot(record.metadata, context.metadata)

    def test_metrics_logger_basic_count(self):
        logger = MetricsLogger()
        
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        logger.record_decision("TestPolicy", s1_context, DecisionAction.ENGAGE)
        
        s3_scenario = get_s3_scenario(seed=42)
        s3_context = self._build_context_for_scenario(s3_scenario)
        logger.record_decision("TestPolicy", s3_context, DecisionAction.TRACK)
        
        s5_scenario = get_s5_scenario(seed=42)
        start_step = s5_scenario.metadata["tracking_loss_start_step"]
        s5_context = self._build_context_for_scenario(s5_scenario, step_index=start_step)
        logger.record_decision("TestPolicy", s5_context, DecisionAction.ABORT)
        
        summary = logger.summarize()
        self.assertEqual(summary.total_decisions, 3)
        self.assertEqual(summary.engage_count, 1)
        self.assertEqual(summary.track_count, 1)
        self.assertEqual(summary.abort_count, 1)
        
        self.assertAlmostEqual(summary.engagement_attempt_rate, 1/3)
        self.assertAlmostEqual(summary.track_rate, 1/3)
        self.assertAlmostEqual(summary.abort_rate, 1/3)

    def test_unsafe_engagement(self):
        logger = MetricsLogger(risk_score_threshold=0.01)
        s4_scenario = get_s4_scenario(seed=42)
        s4_context = self._build_context_for_scenario(s4_scenario, horizon_steps=30)
        
        logger.record_decision("TestPolicy", s4_context, DecisionAction.ENGAGE)
        summary = logger.summarize()
        
        self.assertGreaterEqual(summary.unsafe_engagement_count, 1)
        self.assertGreater(summary.unsafe_engagement_rate, 0.0)

    def test_false_engagement(self):
        logger = MetricsLogger(identification_confidence_threshold=0.6)
        s3_scenario = get_s3_scenario(seed=42)
        s3_context = self._build_context_for_scenario(s3_scenario)
        
        # S3 produces lower identification confidence
        logger.record_decision("TestPolicy", s3_context, DecisionAction.ENGAGE)
        summary = logger.summarize()
        
        self.assertEqual(summary.false_engagement_count, 1)
        self.assertGreater(summary.false_engagement_rate, 0.0)

    def test_non_interceptable_engagement(self):
        logger = MetricsLogger(interceptability_threshold=0.5)
        s6_scenario = get_s6_scenario(seed=42)
        s6_context = self._build_context_for_scenario(s6_scenario)
        
        logger.record_decision("TestPolicy", s6_context, DecisionAction.ENGAGE)
        summary = logger.summarize()
        
        self.assertEqual(summary.non_interceptable_engagement_count, 1)
        self.assertGreater(summary.non_interceptable_engagement_rate, 0.0)

    def test_invalid_engagement(self):
        logger = MetricsLogger()
        s5_scenario = get_s5_scenario(seed=42)
        start_step = s5_scenario.metadata["tracking_loss_start_step"]
        s5_context = self._build_context_for_scenario(s5_scenario, step_index=start_step)
        
        logger.record_decision("TestPolicy", s5_context, DecisionAction.ENGAGE)
        summary = logger.summarize()
        
        self.assertEqual(summary.invalid_engagement_count, 1)
        self.assertGreater(summary.invalid_engagement_rate, 0.0)

    def test_safe_engagement(self):
        logger = MetricsLogger()
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        
        logger.record_decision("TestPolicy", s1_context, DecisionAction.ENGAGE)
        summary = logger.summarize()
        
        self.assertEqual(summary.safe_engagement_count, 1)
        self.assertGreater(summary.safe_engagement_rate, 0.0)

    def test_reason_counts(self):
        logger = MetricsLogger()
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        
        logger.record_decision("TestPolicy", s1_context, DecisionAction.ENGAGE, reason="engage_conditions_satisfied")
        logger.record_decision("TestPolicy", s1_context, DecisionAction.TRACK, reason="tracking_confidence_too_low_track")
        logger.record_decision("TestPolicy", s1_context, DecisionAction.ABORT, reason="not_detected")
        logger.record_decision("TestPolicy", s1_context, DecisionAction.ABORT, reason="not_detected")
        
        summary = logger.summarize()
        self.assertIn("engage_conditions_satisfied", summary.reason_counts)
        self.assertIn("tracking_confidence_too_low_track", summary.reason_counts)
        self.assertEqual(summary.reason_counts["not_detected"], 2)

    def test_policy_name_filtering(self):
        logger = MetricsLogger()
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        
        logger.record_decision("AlwaysEngagePolicy", s1_context, DecisionAction.ENGAGE)
        logger.record_decision("SafetyGatePolicy", s1_context, DecisionAction.ENGAGE)
        logger.record_decision("SafetyGatePolicy", s1_context, DecisionAction.TRACK)
        
        summary_sg = logger.summarize(policy_name="SafetyGatePolicy")
        self.assertEqual(summary_sg.total_decisions, 2)
        self.assertEqual(summary_sg.policy_name, "SafetyGatePolicy")
        
        summary_all = logger.summarize()
        self.assertEqual(summary_all.total_decisions, 3)
        self.assertIsNone(summary_all.policy_name)

    def test_empty_logger_summary(self):
        logger = MetricsLogger()
        summary = logger.summarize()
        self.assertEqual(summary.total_decisions, 0)
        self.assertEqual(summary.engage_count, 0)
        self.assertEqual(summary.engagement_attempt_rate, 0.0)
        self.assertEqual(summary.unsafe_engagement_per_attempt_rate, 0.0)
        self.assertEqual(summary.reason_counts, {})

    def test_per_attempt_rate(self):
        logger = MetricsLogger(identification_confidence_threshold=0.6)
        
        # S1 context - valid, safe engagement
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        logger.record_decision("TestPolicy", s1_context, DecisionAction.ENGAGE)
        
        # S3 context - low identification confidence -> false engagement
        s3_scenario = get_s3_scenario(seed=42)
        s3_context = self._build_context_for_scenario(s3_scenario)
        logger.record_decision("TestPolicy", s3_context, DecisionAction.ENGAGE)
        
        summary = logger.summarize()
        self.assertEqual(summary.engage_count, 2)
        self.assertEqual(summary.false_engagement_count, 1)
        self.assertAlmostEqual(summary.false_engagement_per_attempt_rate, 0.5)

    def test_metrics_summary_to_dict(self):
        logger = MetricsLogger()
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        logger.record_decision("TestPolicy", s1_context, DecisionAction.ENGAGE)
        
        summary = logger.summarize()
        d = summary.to_dict()
        
        self.assertIsInstance(d, dict)
        self.assertEqual(d["total_decisions"], 1)
        self.assertEqual(d["engage_count"], 1)
        self.assertAlmostEqual(d["engagement_attempt_rate"], 1.0)
        self.assertAlmostEqual(d["safe_engagement_per_attempt_rate"], 1.0)
        self.assertIn("reason_counts", d)

    def test_decision_record_to_dict(self):
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        record = DecisionRecord.from_context("TestPolicy", s1_context, DecisionAction.ENGAGE, reason="test")

        d = record.to_dict()
        self.assertEqual(d["action"], "ENGAGE")
        self.assertIsInstance(d["metadata"], dict)
        self.assertIsInstance(d["scenario_type"], str)


class TestGroundTruthClassification(unittest.TestCase):
    """Verify the C1 fix: evaluation uses ground truth, not policy-input values."""

    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()
        self.risk_evaluator = EnvironmentalRiskEvaluator()
        self.intercept_evaluator = InterceptabilityEvaluator()

    def _ctx(self, scenario, step_index=0, horizon_steps=5):
        predictor = ConstantVelocityPredictor(horizon_steps=horizon_steps)
        obs = self.obs_model.observe(scenario, step_index=step_index)
        estimate = self.estimator.estimate(obs)
        prediction = predictor.predict(estimate, scenario.config.time_step)
        risk = self.risk_evaluator.evaluate(scenario, prediction)
        intercept = self.intercept_evaluator.evaluate(scenario, estimate)
        return build_decision_context(
            scenario=scenario, observation=obs, estimate=estimate,
            prediction=prediction, risk_assessment=risk,
            interceptability_assessment=intercept,
        )

    def test_s3_false_engagement_uses_scenario_label_not_policy_threshold(self):
        # Even with id threshold set very permissively (so policy-input
        # classifier would NOT flag false), GT classifier flags it because
        # the scenario is labelled as S3 ambiguous-identification.
        logger = MetricsLogger(identification_confidence_threshold=0.0)
        scenario = get_s3_scenario(seed=42)
        ctx = self._ctx(scenario)
        logger.record_decision(
            "TestPolicy", ctx, DecisionAction.ENGAGE,
            true_target_state=scenario.target.state,
            scenario_metadata=scenario.metadata,
        )
        s = logger.summarize()
        self.assertEqual(s.false_engagement_count, 1)

    def test_s4_unsafe_engagement_uses_true_target_position(self):
        # Place the true target inside the risk zone so GT unsafe must fire.
        logger = MetricsLogger()
        scenario = get_s4_scenario(seed=42)
        zone_center = scenario.metadata["risk_zone_center"]
        scenario.target.state.x = zone_center[0]
        scenario.target.state.y = zone_center[1]
        ctx = self._ctx(scenario, horizon_steps=30)
        logger.record_decision(
            "TestPolicy", ctx, DecisionAction.ENGAGE,
            true_target_state=scenario.target.state,
            scenario_metadata=scenario.metadata,
        )
        s = logger.summarize()
        self.assertEqual(s.unsafe_engagement_count, 1)

    def test_s4_safe_when_true_target_is_far(self):
        # True target placed very far from the risk zone -> GT unsafe must NOT fire.
        logger = MetricsLogger()
        scenario = get_s4_scenario(seed=42)
        zone_center = scenario.metadata["risk_zone_center"]
        scenario.target.state.x = zone_center[0] + 100.0
        scenario.target.state.y = zone_center[1] + 100.0
        ctx = self._ctx(scenario, horizon_steps=30)
        logger.record_decision(
            "TestPolicy", ctx, DecisionAction.ENGAGE,
            true_target_state=scenario.target.state,
            scenario_metadata=scenario.metadata,
        )
        s = logger.summarize()
        self.assertEqual(s.unsafe_engagement_count, 0)

    def test_s1_engage_with_gt_is_safe(self):
        # S1 has no risk zone and is not S3/S6 -> a GT-classified ENGAGE is safe.
        logger = MetricsLogger()
        scenario = get_s1_scenario(seed=42)
        ctx = self._ctx(scenario)
        logger.record_decision(
            "TestPolicy", ctx, DecisionAction.ENGAGE,
            true_target_state=scenario.target.state,
            scenario_metadata=scenario.metadata,
        )
        s = logger.summarize()
        self.assertEqual(s.unsafe_engagement_count, 0)
        self.assertEqual(s.false_engagement_count, 0)
        self.assertEqual(s.non_interceptable_engagement_count, 0)
        self.assertEqual(s.safe_engagement_count, 1)

    def test_s6_non_interceptable_uses_scenario_label(self):
        # Even with interceptability threshold permissive, GT classifier
        # flags S6 ENGAGE because scenario label says non_interceptable.
        logger = MetricsLogger(interceptability_threshold=0.0)
        scenario = get_s6_scenario(seed=42)
        ctx = self._ctx(scenario)
        logger.record_decision(
            "TestPolicy", ctx, DecisionAction.ENGAGE,
            true_target_state=scenario.target.state,
            scenario_metadata=scenario.metadata,
        )
        s = logger.summarize()
        self.assertEqual(s.non_interceptable_engagement_count, 1)


class TestM4Metrics(unittest.TestCase):
    """M4 fix: abort timing, safe opportunity, and composite risk-weighted score."""

    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()
        self.risk_evaluator = EnvironmentalRiskEvaluator()
        self.intercept_evaluator = InterceptabilityEvaluator()

    def _ctx(self, scenario, step_index=0, horizon_steps=5):
        predictor = ConstantVelocityPredictor(horizon_steps=horizon_steps)
        obs = self.obs_model.observe(scenario, step_index=step_index)
        estimate = self.estimator.estimate(obs)
        prediction = predictor.predict(estimate, scenario.config.time_step)
        risk = self.risk_evaluator.evaluate(scenario, prediction)
        intercept = self.intercept_evaluator.evaluate(scenario, estimate)
        return build_decision_context(
            scenario=scenario, observation=obs, estimate=estimate,
            prediction=prediction, risk_assessment=risk,
            interceptability_assessment=intercept,
        )

    def _record_with_trial(self, logger, policy, scenario, action, step_index=0, seed=42, true_state=None):
        ctx = self._ctx(scenario, step_index=step_index)
        rec = logger.record_decision(
            policy, ctx, action,
            true_target_state=true_state or scenario.target.state,
            scenario_metadata=scenario.metadata,
        )
        # Mimic runner.py: metadata carries trial identity for grouping.
        rec.metadata["seed"] = seed
        return rec

    # --- missed safe opportunity ---

    def test_safe_opportunity_capture_engaged_on_safe(self):
        logger = MetricsLogger()
        scenario = get_s1_scenario(seed=42)
        self._record_with_trial(logger, "Pol", scenario, DecisionAction.ENGAGE)
        s = logger.summarize()
        self.assertEqual(s.safe_opportunity_count, 1)
        self.assertEqual(s.missed_safe_opportunity_count, 0)
        self.assertEqual(s.safe_opportunity_capture_rate, 1.0)

    def test_missed_safe_opportunity_when_tracking(self):
        logger = MetricsLogger()
        scenario = get_s1_scenario(seed=42)
        self._record_with_trial(logger, "Pol", scenario, DecisionAction.TRACK)
        s = logger.summarize()
        self.assertEqual(s.safe_opportunity_count, 1)
        self.assertEqual(s.missed_safe_opportunity_count, 1)
        self.assertAlmostEqual(s.missed_safe_opportunity_rate, 1.0)
        self.assertAlmostEqual(s.safe_opportunity_capture_rate, 0.0)

    def test_no_safe_opportunity_in_s3(self):
        # S3 is labeled false → engaging would not be a safe opportunity.
        logger = MetricsLogger()
        scenario = get_s3_scenario(seed=42)
        self._record_with_trial(logger, "Pol", scenario, DecisionAction.TRACK)
        s = logger.summarize()
        self.assertEqual(s.safe_opportunity_count, 0)
        self.assertEqual(s.missed_safe_opportunity_count, 0)

    # --- abort timing ---

    def test_abort_required_on_dropout_and_on_time(self):
        # S5 dropout step → abort_required True; policy aborts at the same step
        # → on-time abort.
        logger = MetricsLogger()
        scenario = get_s5_scenario(seed=42)
        start = scenario.metadata["tracking_loss_start_step"]
        self._record_with_trial(logger, "Pol", scenario,
                                DecisionAction.ABORT, step_index=start)
        s = logger.summarize()
        self.assertEqual(s.abort_required_trial_count, 1)
        self.assertEqual(s.on_time_abort_trial_count, 1)
        self.assertEqual(s.late_abort_trial_count, 0)
        self.assertEqual(s.missed_abort_trial_count, 0)
        self.assertAlmostEqual(s.abort_success_rate, 1.0)

    def test_missed_abort_when_policy_never_aborts(self):
        logger = MetricsLogger()
        scenario = get_s5_scenario(seed=42)
        start = scenario.metadata["tracking_loss_start_step"]
        self._record_with_trial(logger, "Pol", scenario,
                                DecisionAction.ENGAGE, step_index=start)
        s = logger.summarize()
        self.assertEqual(s.abort_required_trial_count, 1)
        self.assertEqual(s.missed_abort_trial_count, 1)
        self.assertAlmostEqual(s.abort_success_rate, 0.0)
        self.assertAlmostEqual(s.missed_abort_rate, 1.0)

    def test_late_abort_when_policy_aborts_too_slowly(self):
        logger = MetricsLogger(late_abort_lag_steps=1)
        scenario = get_s5_scenario(seed=42)
        start = scenario.metadata["tracking_loss_start_step"]
        # Step `start`: abort required, but policy TRACKs
        self._record_with_trial(logger, "Pol", scenario,
                                DecisionAction.TRACK, step_index=start)
        # Step start+3 (> required + lag(1)): policy aborts → late
        self._record_with_trial(logger, "Pol", scenario,
                                DecisionAction.ABORT, step_index=start + 3)
        s = logger.summarize()
        self.assertEqual(s.abort_required_trial_count, 1)
        self.assertEqual(s.late_abort_trial_count, 1)
        self.assertAlmostEqual(s.late_abort_rate, 1.0)

    # --- risk_weighted_score ---

    def test_risk_weighted_score_default_weights_perfect_run(self):
        # One safe ENGAGE in S1 → captures 100% of safe opp, no unsafe/false/late
        # → score = w_safe_capture * 1.0 + w_abort_success * 0 - w_unsafe * 0 - ...
        # = 1.0 (since no abort required, abort_success_rate = 0, but that's
        # multiplied by w_abort_success = 0.5).
        logger = MetricsLogger()  # default weights
        scenario = get_s1_scenario(seed=42)
        self._record_with_trial(logger, "Pol", scenario, DecisionAction.ENGAGE)
        s = logger.summarize()
        self.assertAlmostEqual(s.risk_weighted_score, 1.0)

    def test_risk_weighted_score_penalizes_unsafe(self):
        # Two ENGAGEs: one safe (S1), one unsafe (S4 with true target in zone)
        logger = MetricsLogger()
        s1 = get_s1_scenario(seed=42)
        s4 = get_s4_scenario(seed=42)
        # Force the S4 true target to be inside the risk zone
        zc = s4.metadata["risk_zone_center"]
        s4.target.state.x = zc[0]
        s4.target.state.y = zc[1]

        self._record_with_trial(logger, "Pol", s1, DecisionAction.ENGAGE, seed=1)
        # Use a separate seed/scenario so trial key differs
        self._record_with_trial(logger, "Pol", s4, DecisionAction.ENGAGE,
                                seed=2, true_state=s4.target.state)
        s = logger.summarize()
        # With unsafe penalty 2.0 and unsafe_per_attempt = 0.5, score loses 1.0
        self.assertLess(s.risk_weighted_score, 1.0)


if __name__ == "__main__":
    unittest.main()

