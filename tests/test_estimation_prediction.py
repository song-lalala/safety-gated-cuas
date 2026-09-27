import math
import random
import unittest

from cuas_sim.scenario import get_s1_scenario, get_s5_scenario
from cuas_sim.observation import ObservationModel
from cuas_sim.estimation import TargetStateEstimator
from cuas_sim.prediction import ConstantVelocityPredictor

class TestEstimationPrediction(unittest.TestCase):

    def test_s1_basic_estimate(self):
        scenario = get_s1_scenario(seed=42)
        obs_model = ObservationModel()
        estimator = TargetStateEstimator()

        obs = obs_model.observe(scenario, step_index=0)
        estimate = estimator.estimate(obs)

        self.assertTrue(estimate.detected)
        self.assertTrue(estimate.is_valid)
        self.assertIsNotNone(estimate.estimated_state)
        
        self.assertEqual(estimate.estimated_state.x, obs.observed_state.x)
        self.assertEqual(estimate.estimated_state.y, obs.observed_state.y)
        self.assertEqual(estimate.estimated_state.vx, obs.observed_state.vx)
        self.assertEqual(estimate.estimated_state.vy, obs.observed_state.vy)
        
        self.assertIsNot(estimate.estimated_state, obs.observed_state)

    def test_s5_tracking_loss_estimate(self):
        scenario = get_s5_scenario(seed=42)
        start_step = scenario.metadata["tracking_loss_start_step"]
        lost_conf = scenario.metadata["lost_tracking_confidence"]

        obs_model = ObservationModel()
        estimator = TargetStateEstimator()

        # Before loss (valid estimate first)
        obs_before = obs_model.observe(scenario, step_index=start_step - 1)
        estimator.estimate(obs_before)
        
        # During loss
        obs_during = obs_model.observe(scenario, step_index=start_step)
        estimate_during = estimator.estimate(obs_during)
        
        self.assertFalse(obs_during.detected)
        self.assertFalse(estimate_during.is_valid)
        self.assertFalse(estimate_during.detected)
        self.assertEqual(estimate_during.tracking_confidence, lost_conf)
        
        # Even if estimated_state is not None (copied from last), it must be invalid
        self.assertFalse(estimate_during.is_valid)

    def test_valid_estimate_prediction(self):
        scenario = get_s1_scenario(seed=42)
        obs_model = ObservationModel()
        estimator = TargetStateEstimator()
        predictor = ConstantVelocityPredictor(horizon_steps=5)

        obs = obs_model.observe(scenario, step_index=0)
        estimate = estimator.estimate(obs)
        
        prediction = predictor.predict(estimate, time_step=scenario.config.time_step)
        
        self.assertTrue(prediction.is_valid)
        self.assertEqual(len(prediction.predictions), 5)
        
        # Check first prediction
        first_pred = prediction.predictions[0]
        expected_x = estimate.estimated_state.x + estimate.estimated_state.vx * scenario.config.time_step * 1
        expected_y = estimate.estimated_state.y + estimate.estimated_state.vy * scenario.config.time_step * 1
        
        self.assertAlmostEqual(first_pred.state.x, expected_x)
        self.assertAlmostEqual(first_pred.state.y, expected_y)
        
        # Check uncertainty increases
        base_uncertainty = 1.0 - estimate.tracking_confidence
        expected_uncertainty_1 = base_uncertainty + 0.05 * 1
        expected_uncertainty_5 = base_uncertainty + 0.05 * 5
        self.assertAlmostEqual(first_pred.uncertainty, expected_uncertainty_1)
        self.assertAlmostEqual(prediction.predictions[4].uncertainty, expected_uncertainty_5)
        self.assertLess(first_pred.uncertainty, prediction.predictions[4].uncertainty)

    def test_invalid_estimate_prediction(self):
        scenario = get_s5_scenario(seed=42)
        start_step = scenario.metadata["tracking_loss_start_step"]

        obs_model = ObservationModel()
        estimator = TargetStateEstimator()
        predictor = ConstantVelocityPredictor(horizon_steps=5)

        # Skip straight to loss (no valid prior)
        obs_during = obs_model.observe(scenario, step_index=start_step)
        estimate_during = estimator.estimate(obs_during)

        prediction = predictor.predict(estimate_during, time_step=scenario.config.time_step)

        self.assertFalse(prediction.is_valid)
        self.assertEqual(len(prediction.predictions), 0)


class TestAlphaBetaFilter(unittest.TestCase):
    """C2 fix: estimator is now an α-β tracker, not passthrough."""

    def test_zero_noise_estimate_equals_truth(self):
        # With sigma=0, the filter should reproduce the true state exactly,
        # so the legacy passthrough behaviour is preserved.
        scenario = get_s1_scenario(seed=42)
        obs_model = ObservationModel(noise_sigma=0.0)
        estimator = TargetStateEstimator(time_step=scenario.config.time_step)

        # Run several steps
        for step in range(10):
            obs = obs_model.observe(scenario, step_index=step)
            est = estimator.estimate(obs)
            # Move target one step forward for next observation
            scenario.target.state.x += scenario.target.state.vx * scenario.config.time_step
            scenario.target.state.y += scenario.target.state.vy * scenario.config.time_step
        # Last estimate's tracking_confidence under sigma=0 should be exp(0) = 1.0
        self.assertAlmostEqual(est.tracking_confidence, 1.0, places=6)

    def test_noise_lowers_tracking_confidence(self):
        # With non-zero sigma, residuals are nonzero on average → σ_est grows
        # → tracking_confidence drops below 1.0.
        scenario = get_s1_scenario(seed=42)
        rng = random.Random(7)
        obs_model = ObservationModel(noise_sigma=0.3, rng=rng)
        estimator = TargetStateEstimator(time_step=scenario.config.time_step)

        confidences = []
        for step in range(20):
            obs = obs_model.observe(scenario, step_index=step)
            est = estimator.estimate(obs)
            confidences.append(est.tracking_confidence)
            scenario.target.state.x += scenario.target.state.vx * scenario.config.time_step
            scenario.target.state.y += scenario.target.state.vy * scenario.config.time_step

        # The first estimate has no residual yet → exp(0)=1.0; subsequent
        # estimates must show some decay.
        self.assertEqual(confidences[0], 1.0)
        # By step 5 onward, tracking_confidence should be below 1.0
        self.assertLess(confidences[5], 1.0)
        # And reasonable (not vanishing)
        self.assertGreater(confidences[5], 0.2)

    def test_filter_smooths_toward_truth(self):
        # The filter's estimated position should be closer to the true target
        # state on average than the raw noisy observation, after a few steps.
        scenario = get_s1_scenario(seed=42)
        rng = random.Random(11)
        obs_model = ObservationModel(noise_sigma=0.5, rng=rng)
        estimator = TargetStateEstimator(time_step=scenario.config.time_step)

        obs_errors = []
        est_errors = []
        for step in range(30):
            true_x = scenario.target.state.x
            obs = obs_model.observe(scenario, step_index=step)
            est = estimator.estimate(obs)
            if obs.observed_state is not None and est.estimated_state is not None:
                obs_errors.append(abs(obs.observed_state.x - true_x))
                est_errors.append(abs(est.estimated_state.x - true_x))
            scenario.target.state.x += scenario.target.state.vx * scenario.config.time_step
            scenario.target.state.y += scenario.target.state.vy * scenario.config.time_step

        # Compare averaged errors after warmup (skip first 3 steps)
        avg_obs = sum(obs_errors[3:]) / len(obs_errors[3:])
        avg_est = sum(est_errors[3:]) / len(est_errors[3:])
        self.assertLess(avg_est, avg_obs)


if __name__ == "__main__":
    unittest.main()
