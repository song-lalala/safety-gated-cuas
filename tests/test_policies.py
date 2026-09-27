import unittest

from cuas_sim.scenario import get_s1_scenario, get_s5_scenario, get_s6_scenario
from cuas_sim.observation import ObservationModel
from cuas_sim.estimation import TargetStateEstimator
from cuas_sim.prediction import ConstantVelocityPredictor
from cuas_sim.evaluators import EnvironmentalRiskEvaluator, InterceptabilityEvaluator
from cuas_sim.policies import (
    DecisionAction, 
    build_decision_context, 
    AlwaysEngagePolicy, 
    DistanceOnlyPolicy, 
    InterceptabilityOnlyPolicy
)

class TestPolicies(unittest.TestCase):
    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()
        self.predictor = ConstantVelocityPredictor(horizon_steps=5)
        self.risk_evaluator = EnvironmentalRiskEvaluator()
        self.intercept_evaluator = InterceptabilityEvaluator()

    def _build_context_for_scenario(self, scenario, step_index=0):
        obs = self.obs_model.observe(scenario, step_index=step_index)
        estimate = self.estimator.estimate(obs)
        prediction = self.predictor.predict(estimate, scenario.config.time_step)
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

    def test_decision_action_enum(self):
        self.assertIsInstance(DecisionAction.ENGAGE.value, str)
        self.assertIsInstance(DecisionAction.TRACK.value, str)
        self.assertIsInstance(DecisionAction.ABORT.value, str)

    def test_build_decision_context_basic(self):
        scenario = get_s1_scenario(seed=42)
        context = self._build_context_for_scenario(scenario)
        
        self.assertTrue(context.detected)
        self.assertTrue(context.estimate_valid)
        self.assertTrue(context.prediction_valid)
        self.assertIsNotNone(context.distance_to_target)
        self.assertEqual(context.risk_score, 0.0)
        self.assertGreaterEqual(context.interceptability_score, 0.0)
        self.assertLessEqual(context.interceptability_score, 1.0)

    def test_always_engage_policy(self):
        policy = AlwaysEngagePolicy()
        
        # S1 basic
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        self.assertEqual(policy.decide(s1_context), DecisionAction.ENGAGE)
        
        # S5 tracking loss
        s5_scenario = get_s5_scenario(seed=42)
        start_step = s5_scenario.metadata["tracking_loss_start_step"]
        s5_context = self._build_context_for_scenario(s5_scenario, step_index=start_step)
        self.assertEqual(policy.decide(s5_context), DecisionAction.ABORT)

    def test_distance_only_policy(self):
        # S1 context
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        
        # ENGAGE when threshold is very large
        policy_large = DistanceOnlyPolicy(engage_distance_threshold=9999.0)
        self.assertEqual(policy_large.decide(s1_context), DecisionAction.ENGAGE)
        
        # TRACK when threshold is very small
        policy_small = DistanceOnlyPolicy(engage_distance_threshold=0.0)
        self.assertEqual(policy_small.decide(s1_context), DecisionAction.TRACK)
        
        # S5 tracking loss
        s5_scenario = get_s5_scenario(seed=42)
        start_step = s5_scenario.metadata["tracking_loss_start_step"]
        s5_context = self._build_context_for_scenario(s5_scenario, step_index=start_step)
        self.assertEqual(policy_large.decide(s5_context), DecisionAction.ABORT)

    def test_interceptability_only_policy(self):
        policy = InterceptabilityOnlyPolicy(interceptability_threshold=0.5)
        
        # S1 basic (default 0.5 interceptability_score)
        s1_scenario = get_s1_scenario(seed=42)
        s1_context = self._build_context_for_scenario(s1_scenario)
        self.assertEqual(policy.decide(s1_context), DecisionAction.ENGAGE)
        
        # S6 non-interceptable
        s6_scenario = get_s6_scenario(seed=42)
        s6_context = self._build_context_for_scenario(s6_scenario)
        self.assertEqual(policy.decide(s6_context), DecisionAction.TRACK)
        
        # S5 tracking loss
        s5_scenario = get_s5_scenario(seed=42)
        start_step = s5_scenario.metadata["tracking_loss_start_step"]
        s5_context = self._build_context_for_scenario(s5_scenario, step_index=start_step)
        self.assertEqual(policy.decide(s5_context), DecisionAction.ABORT)


class TestBaselineExplainReasons(unittest.TestCase):
    """M5: baseline policies now expose explain() with explicit reasons so the
    audit trail in records.csv distinguishes fallback aborts from policy
    decisions."""

    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()
        self.predictor = ConstantVelocityPredictor(horizon_steps=5)
        self.risk_evaluator = EnvironmentalRiskEvaluator()
        self.intercept_evaluator = InterceptabilityEvaluator()

    def _ctx(self, scenario, step_index=0):
        obs = self.obs_model.observe(scenario, step_index=step_index)
        estimate = self.estimator.estimate(obs)
        prediction = self.predictor.predict(estimate, scenario.config.time_step)
        risk = self.risk_evaluator.evaluate(scenario, prediction)
        intercept = self.intercept_evaluator.evaluate(scenario, estimate)
        return build_decision_context(
            scenario=scenario, observation=obs, estimate=estimate,
            prediction=prediction, risk_assessment=risk,
            interceptability_assessment=intercept,
        )

    def test_always_engage_reasons(self):
        policy = AlwaysEngagePolicy()

        s1 = get_s1_scenario(seed=42)
        action, reason = policy.explain(self._ctx(s1))
        self.assertEqual(action, DecisionAction.ENGAGE)
        self.assertEqual(reason, "always_engage")

        # S5 dropout fires the fallback branch.
        s5 = get_s5_scenario(seed=42)
        start = s5.metadata["tracking_loss_start_step"]
        action, reason = policy.explain(self._ctx(s5, step_index=start))
        self.assertEqual(action, DecisionAction.ABORT)
        self.assertEqual(reason, "fallback_not_detected")

    def test_distance_only_reasons(self):
        policy = DistanceOnlyPolicy(engage_distance_threshold=9999.0)
        s1 = get_s1_scenario(seed=42)
        action, reason = policy.explain(self._ctx(s1))
        self.assertEqual(action, DecisionAction.ENGAGE)
        self.assertEqual(reason, "distance_within_threshold")

        policy_strict = DistanceOnlyPolicy(engage_distance_threshold=0.0)
        action, reason = policy_strict.explain(self._ctx(s1))
        self.assertEqual(action, DecisionAction.TRACK)
        self.assertEqual(reason, "distance_above_threshold")

    def test_interceptability_only_reasons(self):
        policy = InterceptabilityOnlyPolicy(interceptability_threshold=0.0)
        s1 = get_s1_scenario(seed=42)
        action, reason = policy.explain(self._ctx(s1))
        self.assertEqual(action, DecisionAction.ENGAGE)
        self.assertEqual(reason, "interceptable_above_threshold")

        s6 = get_s6_scenario(seed=42)
        action, reason = InterceptabilityOnlyPolicy().explain(self._ctx(s6))
        self.assertEqual(action, DecisionAction.TRACK)
        self.assertEqual(reason, "interceptable_below_threshold")


if __name__ == "__main__":
    unittest.main()
