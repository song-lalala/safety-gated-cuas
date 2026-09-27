import unittest
import copy

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
from cuas_sim.evaluators import (
    EnvironmentalRiskEvaluator,
    InterceptabilityEvaluator,
    AbortFeasibilityEvaluator,
)
from cuas_sim.policies import build_decision_context, SafetyGatePolicy, DecisionAction

class TestSafetyGate(unittest.TestCase):
    def setUp(self):
        self.obs_model = ObservationModel()
        self.estimator = TargetStateEstimator()
        self.risk_evaluator = EnvironmentalRiskEvaluator()
        self.intercept_evaluator = InterceptabilityEvaluator()
        self.abort_evaluator = AbortFeasibilityEvaluator()

    def _build_context_for_scenario(self, scenario, step_index=0, horizon_steps=5):
        predictor = ConstantVelocityPredictor(horizon_steps=horizon_steps)

        obs = self.obs_model.observe(scenario, step_index=step_index)
        estimate = self.estimator.estimate(obs)
        prediction = predictor.predict(estimate, scenario.config.time_step)
        risk_assessment = self.risk_evaluator.evaluate(scenario, prediction)
        intercept_assessment = self.intercept_evaluator.evaluate(scenario, estimate)
        abort_assessment = self.abort_evaluator.evaluate(scenario, prediction)

        return build_decision_context(
            scenario=scenario,
            observation=obs,
            estimate=estimate,
            prediction=prediction,
            risk_assessment=risk_assessment,
            interceptability_assessment=intercept_assessment,
            abort_feasibility_assessment=abort_assessment,
        )

    def test_s1_engage(self):
        scenario = get_s1_scenario(seed=42)
        context = self._build_context_for_scenario(scenario)
        policy = SafetyGatePolicy()
        
        action, reason = policy.explain(context)
        self.assertEqual(action, DecisionAction.ENGAGE)
        self.assertEqual(reason, "engage_conditions_satisfied")
        self.assertEqual(policy.decide(context), DecisionAction.ENGAGE)

    def test_s3_low_identification(self):
        scenario = get_s3_scenario(seed=42)
        context = self._build_context_for_scenario(scenario)
        policy = SafetyGatePolicy()
        
        self.assertLess(context.identification_confidence, policy.identification_confidence_threshold)
        
        action, reason = policy.explain(context)
        self.assertEqual(action, DecisionAction.TRACK)
        self.assertEqual(reason, "identification_confidence_too_low")

    def test_s4_risk_proximity(self):
        # C3 fix: docs §12 says "environmental risk too high → ABORT".
        # Previously the code returned TRACK here, which is now corrected.
        scenario = get_s4_scenario(seed=42)
        context = self._build_context_for_scenario(scenario, horizon_steps=30)
        policy = SafetyGatePolicy()

        self.assertGreater(context.risk_score, 0.0)

        # Force a strict threshold so high_risk fires regardless of seed.
        policy.risk_score_threshold = 0.01

        action, reason = policy.explain(context)
        self.assertEqual(action, DecisionAction.ABORT)
        self.assertEqual(reason, "high_environmental_risk")

    def test_s5_tracking_loss(self):
        scenario = get_s5_scenario(seed=42)
        start_step = scenario.metadata["tracking_loss_start_step"]
        context = self._build_context_for_scenario(scenario, step_index=start_step)
        policy = SafetyGatePolicy()
        
        self.assertFalse(context.detected)
        self.assertFalse(context.estimate_valid)
        
        action, reason = policy.explain(context)
        self.assertEqual(action, DecisionAction.ABORT)

    def test_s6_non_interceptable(self):
        scenario = get_s6_scenario(seed=42)
        context = self._build_context_for_scenario(scenario)
        policy = SafetyGatePolicy()
        
        self.assertLess(context.interceptability_score, policy.interceptability_threshold)
        
        action, reason = policy.explain(context)
        self.assertEqual(action, DecisionAction.TRACK)
        self.assertEqual(reason, "not_interceptable")

    def test_threshold_adjustments(self):
        scenario = get_s1_scenario(seed=42)
        context = self._build_context_for_scenario(scenario)
        policy = SafetyGatePolicy()

        # Verify basic is ENGAGE
        self.assertEqual(policy.decide(context), DecisionAction.ENGAGE)

        # Override context to simulate low tracking confidence but not fully lost
        modified_context = copy.deepcopy(context)
        modified_context.tracking_confidence = 0.2

        # Policy default tracking_abort_threshold is 0.3
        action, reason = policy.explain(modified_context)
        self.assertEqual(action, DecisionAction.ABORT)
        self.assertEqual(reason, "tracking_confidence_too_low_abort")

    def test_abort_feasibility_triggers_abort(self):
        # C3-a: A_abort is now the 5th input. Force a low A_abort score and
        # verify SafetyGate returns ABORT for that reason (the other inputs
        # are kept healthy so the abort_feasibility branch is the one tested).
        scenario = get_s1_scenario(seed=42)
        context = self._build_context_for_scenario(scenario)
        policy = SafetyGatePolicy(abort_feasibility_threshold=0.5)

        # Baseline: ENGAGE
        self.assertEqual(policy.decide(context), DecisionAction.ENGAGE)

        modified = copy.deepcopy(context)
        modified.abort_feasibility_score = 0.1
        modified.is_abort_feasible = False

        action, reason = policy.explain(modified)
        self.assertEqual(action, DecisionAction.ABORT)
        self.assertEqual(reason, "abort_feasibility_too_low")

    def test_default_context_has_full_abort_feasibility(self):
        # Backwards-compat: build_decision_context without abort assessment
        # should leave A_abort = 1.0 / is_abort_feasible = True so legacy
        # callers behave identically.
        scenario = get_s1_scenario(seed=42)
        predictor = ConstantVelocityPredictor(horizon_steps=5)
        obs = self.obs_model.observe(scenario, step_index=0)
        estimate = self.estimator.estimate(obs)
        prediction = predictor.predict(estimate, scenario.config.time_step)
        risk = self.risk_evaluator.evaluate(scenario, prediction)
        intercept = self.intercept_evaluator.evaluate(scenario, estimate)

        # No abort assessment passed.
        ctx = build_decision_context(
            scenario=scenario,
            observation=obs,
            estimate=estimate,
            prediction=prediction,
            risk_assessment=risk,
            interceptability_assessment=intercept,
        )
        self.assertEqual(ctx.abort_feasibility_score, 1.0)
        self.assertTrue(ctx.is_abort_feasible)


if __name__ == "__main__":
    unittest.main()
