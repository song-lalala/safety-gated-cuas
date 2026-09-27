import math
import random
import unittest
from cuas_sim.scenario import get_s1_scenario, get_s3_scenario, get_s5_scenario
from cuas_sim.observation import ObservationModel, TargetObservation

class TestObservationModel(unittest.TestCase):
    def setUp(self):
        self.model = ObservationModel()

    def test_s1_basic_observation(self):
        scenario = get_s1_scenario(seed=42)
        obs = self.model.observe(scenario, step_index=0)
        
        self.assertTrue(obs.detected)
        self.assertIsNotNone(obs.observed_state)
        # Value matches
        self.assertEqual(obs.observed_state.x, scenario.target.state.x)
        self.assertEqual(obs.observed_state.y, scenario.target.state.y)
        self.assertEqual(obs.observed_state.vx, scenario.target.state.vx)
        self.assertEqual(obs.observed_state.vy, scenario.target.state.vy)
        # Not the exact same object
        self.assertIsNot(obs.observed_state, scenario.target.state)
        
    def test_s3_identification_confidence(self):
        scenario = get_s3_scenario(seed=42)
        expected_conf = scenario.metadata["identification_confidence"]
        
        obs = self.model.observe(scenario, step_index=0)
        self.assertEqual(obs.identification_confidence, expected_conf)
        
    def test_s5_tracking_loss(self):
        scenario = get_s5_scenario(seed=42)
        start_step = scenario.metadata["tracking_loss_start_step"]
        end_step = scenario.metadata["tracking_loss_end_step"]
        nominal_conf = scenario.metadata["nominal_tracking_confidence"]
        lost_conf = scenario.metadata["lost_tracking_confidence"]
        
        # Before tracking loss
        obs_before = self.model.observe(scenario, step_index=start_step - 1)
        self.assertTrue(obs_before.detected)
        self.assertIsNotNone(obs_before.observed_state)
        self.assertEqual(obs_before.tracking_confidence, nominal_conf)
        
        # During tracking loss
        obs_during = self.model.observe(scenario, step_index=start_step)
        self.assertFalse(obs_during.detected)
        self.assertIsNone(obs_during.observed_state)
        self.assertEqual(obs_during.tracking_confidence, lost_conf)
        
        # After tracking loss
        obs_after = self.model.observe(scenario, step_index=end_step)
        self.assertTrue(obs_after.detected)
        self.assertIsNotNone(obs_after.observed_state)
        self.assertEqual(obs_after.tracking_confidence, nominal_conf)


class TestObservationNoise(unittest.TestCase):
    """C2 fix: ObservationModel applies Gaussian noise when noise_sigma > 0."""

    def test_zero_sigma_is_perfect_sensor(self):
        scenario = get_s1_scenario(seed=42)
        rng = random.Random(123)
        model = ObservationModel(noise_sigma=0.0, rng=rng)
        obs = model.observe(scenario, step_index=0)
        self.assertEqual(obs.observed_state.x, scenario.target.state.x)
        self.assertEqual(obs.observed_state.y, scenario.target.state.y)

    def test_nonzero_sigma_perturbs_position(self):
        scenario = get_s1_scenario(seed=42)
        rng = random.Random(123)
        model = ObservationModel(noise_sigma=0.5, rng=rng)
        obs = model.observe(scenario, step_index=0)
        # Position should differ from ground truth (probability of zero noise is 0).
        self.assertNotEqual(obs.observed_state.x, scenario.target.state.x)
        self.assertNotEqual(obs.observed_state.y, scenario.target.state.y)

    def test_noise_is_reproducible_across_rng_seeds(self):
        scenario = get_s1_scenario(seed=42)
        rng_a = random.Random(123)
        rng_b = random.Random(123)
        obs_a = ObservationModel(noise_sigma=0.5, rng=rng_a).observe(scenario, step_index=0)
        obs_b = ObservationModel(noise_sigma=0.5, rng=rng_b).observe(scenario, step_index=0)
        self.assertEqual(obs_a.observed_state.x, obs_b.observed_state.x)
        self.assertEqual(obs_a.observed_state.y, obs_b.observed_state.y)

    def test_noise_distribution_is_close_to_sigma(self):
        # Sanity check: sample many observations and verify empirical std ~ sigma.
        scenario = get_s1_scenario(seed=42)
        sigma = 0.3
        rng = random.Random(0)
        model = ObservationModel(noise_sigma=sigma, rng=rng)
        samples = [model.observe(scenario, step_index=0).observed_state.x
                   for _ in range(500)]
        mean = sum(samples) / len(samples)
        var = sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
        std = math.sqrt(var)
        # Loose bounds for 500 samples
        self.assertGreater(std, sigma * 0.7)
        self.assertLess(std, sigma * 1.3)


if __name__ == "__main__":
    unittest.main()
