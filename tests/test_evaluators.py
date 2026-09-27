import unittest

from cuas_sim.scenario import get_s1_scenario, get_s4_scenario, get_s5_scenario, get_s6_scenario
from cuas_sim.observation import ObservationModel
from cuas_sim.estimation import TargetStateEstimator
from cuas_sim.prediction import ConstantVelocityPredictor
from cuas_sim.evaluators import (
    EnvironmentalRiskEvaluator,
    InterceptabilityEvaluator,
    AbortFeasibilityEvaluator,
)

class TestEvaluators(unittest.TestCase):
    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()
        self.predictor = ConstantVelocityPredictor(horizon_steps=5)
        self.risk_evaluator = EnvironmentalRiskEvaluator()
        self.intercept_evaluator = InterceptabilityEvaluator()

    def test_s1_risk_evaluator_default(self):
        scenario = get_s1_scenario(seed=42)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)
        prediction = self.predictor.predict(estimate, scenario.config.time_step)
        
        assessment = self.risk_evaluator.evaluate(scenario, prediction)
        
        self.assertEqual(assessment.risk_score, 0.0)
        self.assertFalse(assessment.is_high_risk)
        self.assertIsNone(assessment.min_distance_to_risk_zone)
        self.assertIsNone(assessment.risk_zone_radius)

    def test_s4_risk_evaluator(self):
        scenario = get_s4_scenario(seed=42)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)
        prediction = self.predictor.predict(estimate, scenario.config.time_step)
        
        assessment = self.risk_evaluator.evaluate(scenario, prediction)
        
        self.assertIsNotNone(assessment.min_distance_to_risk_zone)
        self.assertIsNotNone(assessment.risk_zone_radius)
        self.assertGreaterEqual(assessment.risk_score, 0.0)
        self.assertLessEqual(assessment.risk_score, 1.0)
        self.assertGreater(assessment.risk_score, 0.0)

    def test_invalid_prediction_risk_evaluator(self):
        scenario = get_s5_scenario(seed=42)
        start_step = scenario.metadata["tracking_loss_start_step"]
        
        obs = self.obs_model.observe(scenario, step_index=start_step)
        estimate = self.estimator.estimate(obs)
        prediction = self.predictor.predict(estimate, scenario.config.time_step)
        
        assessment = self.risk_evaluator.evaluate(scenario, prediction)
        
        self.assertEqual(assessment.risk_score, 0.0)
        self.assertFalse(assessment.is_high_risk)
        self.assertIn("reason", assessment.metadata)
        self.assertEqual(assessment.metadata["reason"], "invalid_prediction")

    def test_s1_interceptability_default(self):
        # M2 fix: InterceptabilityEvaluator now derives the score from current
        # relative geometry. For S1 (target approaching a stationary interceptor
        # head-on, target speed ≈ 1, default interceptor speed = 2) the score is
        # well above the default threshold but no longer the legacy constant 0.5.
        scenario = get_s1_scenario(seed=42)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)

        assessment = self.intercept_evaluator.evaluate(scenario, estimate)

        # speed_ratio is now computed from state, not None.
        self.assertIsNotNone(assessment.speed_ratio)
        self.assertGreater(assessment.speed_ratio, 1.0)
        # Score should be high (head-on closing target, easy to intercept).
        self.assertGreater(assessment.interceptability_score, 0.5)
        self.assertLessEqual(assessment.interceptability_score, 1.0)
        self.assertTrue(assessment.is_interceptable)
        # initial_distance is now the per-step relative distance.
        self.assertIsNotNone(assessment.initial_distance)
        self.assertGreater(assessment.initial_distance, 0.0)

    def test_s6_non_interceptable(self):
        scenario = get_s6_scenario(seed=42)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)

        assessment = self.intercept_evaluator.evaluate(scenario, estimate)

        self.assertIsNotNone(assessment.speed_ratio)
        self.assertLess(assessment.speed_ratio, 1.0)
        self.assertLess(assessment.interceptability_score, 0.5)
        self.assertFalse(assessment.is_interceptable)


class TestRiskUncertaintyBuffer(unittest.TestCase):
    """M3 fix: risk score must inflate the unsafe radius by k_sigma*sigma_pred."""

    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()

    def _eval_s4(self, k_sigma, predictor_horizon=30):
        scenario = get_s4_scenario(seed=42)
        predictor = ConstantVelocityPredictor(horizon_steps=predictor_horizon)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)
        prediction = predictor.predict(estimate, scenario.config.time_step)
        evaluator = EnvironmentalRiskEvaluator(k_sigma=k_sigma)
        return evaluator.evaluate(scenario, prediction)

    def test_higher_k_sigma_increases_risk_score(self):
        # With predictions carrying non-zero uncertainty, larger k_sigma
        # inflates the effective unsafe radius → smaller clearance → higher risk.
        low = self._eval_s4(k_sigma=0.0).risk_score
        high = self._eval_s4(k_sigma=2.0).risk_score
        self.assertGreater(high, low)

    def test_k_sigma_zero_reproduces_pre_m3_behaviour(self):
        scenario = get_s4_scenario(seed=42)
        predictor = ConstantVelocityPredictor(horizon_steps=30)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)
        prediction = predictor.predict(estimate, scenario.config.time_step)
        evaluator = EnvironmentalRiskEvaluator(k_sigma=0.0, abort_margin=0.0)
        assessment = evaluator.evaluate(scenario, prediction)
        # The effective unsafe radius collapses to risk_zone_radius.
        meta = assessment.metadata
        self.assertAlmostEqual(meta["effective_unsafe_radius"], assessment.risk_zone_radius)

    def test_abort_margin_inflates_radius(self):
        scenario = get_s4_scenario(seed=42)
        predictor = ConstantVelocityPredictor(horizon_steps=30)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)
        prediction = predictor.predict(estimate, scenario.config.time_step)
        base = EnvironmentalRiskEvaluator(k_sigma=0.0, abort_margin=0.0).evaluate(scenario, prediction)
        bumped = EnvironmentalRiskEvaluator(k_sigma=0.0, abort_margin=1.0).evaluate(scenario, prediction)
        self.assertAlmostEqual(
            bumped.metadata["effective_unsafe_radius"] - base.metadata["effective_unsafe_radius"],
            1.0,
            places=6,
        )
        # Larger effective radius → higher risk score (or at least non-decreasing).
        self.assertGreaterEqual(bumped.risk_score, base.risk_score)

    def test_invalid_prediction_metadata_absent(self):
        scenario = get_s5_scenario(seed=42)
        start = scenario.metadata["tracking_loss_start_step"]
        obs = self.obs_model.observe(scenario, step_index=start)
        estimate = self.estimator.estimate(obs)
        predictor = ConstantVelocityPredictor(horizon_steps=5)
        prediction = predictor.predict(estimate, scenario.config.time_step)
        assessment = EnvironmentalRiskEvaluator().evaluate(scenario, prediction)
        # invalid_prediction path is preserved; we still report risk_score=0.
        self.assertEqual(assessment.risk_score, 0.0)
        self.assertEqual(assessment.metadata.get("reason"), "invalid_prediction")


class TestInterceptabilityStateDependence(unittest.TestCase):
    """M2 fix: InterceptabilityEvaluator should react to state, not just metadata."""

    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()
        self.intercept_evaluator = InterceptabilityEvaluator()

    def _score_for_target_state(self, scenario, target_state):
        """Build a one-step assessment with the scenario.target.state overridden."""
        scenario.target.state.x = target_state[0]
        scenario.target.state.y = target_state[1]
        scenario.target.state.vx = target_state[2]
        scenario.target.state.vy = target_state[3]
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = TargetStateEstimator().estimate(obs)  # fresh filter to avoid carry-over
        return self.intercept_evaluator.evaluate(scenario, estimate)

    def test_score_drops_when_target_moves_away(self):
        scenario = get_s1_scenario(seed=42)
        # Same position, but velocity flipped so target moves AWAY from origin
        # instead of toward it. Score should drop noticeably.
        score_in = self._score_for_target_state(scenario, (-10.0, 0.0, 1.0, 0.0))   # head-on closing
        score_out = self._score_for_target_state(scenario, (-10.0, 0.0, -1.0, 0.0)) # receding
        self.assertGreater(score_in.interceptability_score, score_out.interceptability_score)

    def test_score_drops_with_distance(self):
        scenario = get_s1_scenario(seed=42)
        near = self._score_for_target_state(scenario, (-5.0, 0.0, 1.0, 0.0))
        far = self._score_for_target_state(scenario, (-50.0, 0.0, 1.0, 0.0))
        # Distance factor is bounded below by 0.5, but near should still beat far.
        self.assertGreater(near.interceptability_score, far.interceptability_score)

    def test_score_respects_scenario_speed_ratio_override(self):
        # S6 sets speed_ratio < 1 in metadata; that override should pull score
        # below the threshold even though the per-step geometry might look OK.
        scenario = get_s6_scenario(seed=42)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)
        assessment = self.intercept_evaluator.evaluate(scenario, estimate)
        self.assertLess(assessment.speed_ratio, 1.0)
        self.assertFalse(assessment.is_interceptable)
        self.assertLess(assessment.interceptability_score, 0.5)

    def test_invalid_estimate_yields_zero_score(self):
        scenario = get_s5_scenario(seed=42)
        start = scenario.metadata["tracking_loss_start_step"]
        obs = self.obs_model.observe(scenario, step_index=start)
        estimate = self.estimator.estimate(obs)  # dropout → invalid
        assessment = self.intercept_evaluator.evaluate(scenario, estimate)
        self.assertEqual(assessment.interceptability_score, 0.0)
        self.assertFalse(assessment.is_interceptable)
        self.assertEqual(assessment.metadata.get("reason"), "invalid_estimate")


class TestAbortFeasibilityEvaluator(unittest.TestCase):
    """C3: AbortFeasibilityEvaluator returns the 5th SafetyGate input."""

    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()
        self.predictor_short = ConstantVelocityPredictor(horizon_steps=5)
        self.predictor_long = ConstantVelocityPredictor(horizon_steps=30)
        self.evaluator = AbortFeasibilityEvaluator()

    def test_s1_no_risk_zone_is_fully_feasible(self):
        scenario = get_s1_scenario(seed=42)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)
        prediction = self.predictor_short.predict(estimate, scenario.config.time_step)
        a = self.evaluator.evaluate(scenario, prediction)
        self.assertEqual(a.abort_feasibility_score, 1.0)
        self.assertTrue(a.is_feasible)
        self.assertIsNone(a.steps_until_risk)

    def test_invalid_prediction_is_infeasible(self):
        scenario = get_s5_scenario(seed=42)
        start = scenario.metadata["tracking_loss_start_step"]
        obs = self.obs_model.observe(scenario, step_index=start)
        estimate = self.estimator.estimate(obs)
        prediction = self.predictor_short.predict(estimate, scenario.config.time_step)
        a = self.evaluator.evaluate(scenario, prediction)
        self.assertEqual(a.abort_feasibility_score, 0.0)
        self.assertFalse(a.is_feasible)

    def test_s4_score_decreases_as_target_approaches_risk_zone(self):
        # At step 0 the predicted trajectory enters the risk zone far in the
        # future. By step 15 it enters much sooner → score should be lower.
        scenario = get_s4_scenario(seed=42)
        obs_0 = self.obs_model.observe(scenario, step_index=0)
        est_0 = self.estimator.estimate(obs_0)
        pred_0 = self.predictor_long.predict(est_0, scenario.config.time_step)
        a_0 = self.evaluator.evaluate(scenario, pred_0)

        # Advance scenario target state a few steps forward.
        for _ in range(15):
            scenario.target.state.x += scenario.target.state.vx * scenario.config.time_step
            scenario.target.state.y += scenario.target.state.vy * scenario.config.time_step
        # Fresh estimator so it doesn't smooth across the jump.
        est = TargetStateEstimator()
        obs_15 = self.obs_model.observe(scenario, step_index=15)
        est_15 = est.estimate(obs_15)
        pred_15 = self.predictor_long.predict(est_15, scenario.config.time_step)
        a_15 = self.evaluator.evaluate(scenario, pred_15)

        # At step 15 target is much closer to the risk zone → fewer steps
        # until risk → lower score.
        self.assertLessEqual(a_15.abort_feasibility_score, a_0.abort_feasibility_score)


if __name__ == "__main__":
    unittest.main()
