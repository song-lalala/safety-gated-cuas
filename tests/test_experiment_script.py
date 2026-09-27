import unittest
import argparse
import tempfile
import os
import sys

# Insert the project root to sys.path so we can import the experiments module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cuas_sim.scenario import ScenarioType
from experiments.run_monte_carlo import (
    parse_scenario_names,
    parse_policy_names,
    is_contiguous_seed_range,
    serialize_seed_config,
    config_to_dict,
    build_config_from_args,
    run_experiment,
    main
)

class TestExperimentScript(unittest.TestCase):
    def test_parse_scenario_names(self):
        self.assertIsNone(parse_scenario_names(None))
        self.assertIsNone(parse_scenario_names(""))
        self.assertIsNone(parse_scenario_names("   "))
        
        scenarios = parse_scenario_names("S1_STRAIGHT_INTRUSION,S3_LOW_IDENTIFICATION_CONFIDENCE")
        self.assertEqual(len(scenarios), 2)
        self.assertIn(ScenarioType.S1_STRAIGHT_INTRUSION, scenarios)
        self.assertIn(ScenarioType.S3_LOW_IDENTIFICATION_CONFIDENCE, scenarios)
        
        with self.assertRaises(ValueError):
            parse_scenario_names("INVALID_SCENARIO_NAME")

    def test_parse_policy_names(self):
        self.assertIsNone(parse_policy_names(None))
        self.assertIsNone(parse_policy_names(""))
        
        policies = parse_policy_names("AlwaysEngagePolicy, SafetyGatePolicy ")
        self.assertEqual(len(policies), 2)
        self.assertEqual(policies[0], "AlwaysEngagePolicy")
        self.assertEqual(policies[1], "SafetyGatePolicy")

    def test_build_config_from_args(self):
        args = argparse.Namespace(
            start_seed=10,
            num_seeds=3,
            steps_per_trial=5,
            prediction_horizon_steps=5,
            s4_prediction_horizon_steps=30,
            distance_engage_threshold=12.0,
            safety_tracking_abort_threshold=0.2,
            safety_tracking_confidence_threshold=0.7,
            safety_identification_confidence_threshold=0.8,
            safety_risk_score_threshold=0.9,
            safety_interceptability_threshold=0.4,
            observation_noise_sigma=0.0,
            observation_velocity_noise_sigma=0.0,
            safety_abort_feasibility_threshold=0.5,
            abort_horizon_steps=10,
            abort_risk_zone_safety_buffer=0.5,
            risk_k_sigma=1.0,
            risk_abort_margin=0.0,
            risk_clearance_falloff=2.0,
            late_abort_lag_steps=2,
            weight_safe_capture=1.0,
            weight_abort_success=0.5,
            weight_unsafe=2.0,
            weight_false=2.0,
            weight_late_abort=1.0,
            scenarios=None,
            policies=None
        )
        
        config = build_config_from_args(args)
        self.assertEqual(config.seeds, [10, 11, 12])
        self.assertEqual(config.steps_per_trial, 5)
        self.assertEqual(config.prediction_horizon_steps, 5)
        self.assertEqual(config.s4_prediction_horizon_steps, 30)
        self.assertEqual(config.distance_engage_threshold, 12.0)
        self.assertEqual(config.safety_tracking_abort_threshold, 0.2)
        self.assertEqual(config.safety_tracking_confidence_threshold, 0.7)
        self.assertEqual(config.safety_identification_confidence_threshold, 0.8)
        self.assertEqual(config.safety_risk_score_threshold, 0.9)
        self.assertEqual(config.safety_interceptability_threshold, 0.4)

    def test_contiguous_seed_config(self):
        args = argparse.Namespace(
            start_seed=1,
            num_seeds=100,
            steps_per_trial=20,
            prediction_horizon_steps=5,
            s4_prediction_horizon_steps=30,
            distance_engage_threshold=12.0,
            safety_tracking_abort_threshold=0.3,
            safety_tracking_confidence_threshold=0.6,
            safety_identification_confidence_threshold=0.6,
            safety_risk_score_threshold=0.7,
            safety_interceptability_threshold=0.5,
            observation_noise_sigma=0.0,
            observation_velocity_noise_sigma=0.0,
            safety_abort_feasibility_threshold=0.5,
            abort_horizon_steps=10,
            abort_risk_zone_safety_buffer=0.5,
            risk_k_sigma=1.0,
            risk_abort_margin=0.0,
            risk_clearance_falloff=2.0,
            late_abort_lag_steps=2,
            weight_safe_capture=1.0,
            weight_abort_success=0.5,
            weight_unsafe=2.0,
            weight_false=2.0,
            weight_late_abort=1.0,
            scenarios=None,
            policies=None
        )
        config = build_config_from_args(args)
        cfg_dict = config_to_dict(config)
        self.assertEqual(cfg_dict["seed_mode"], "range")
        self.assertEqual(cfg_dict["seed_start"], 1)
        self.assertEqual(cfg_dict["seed_end"], 100)
        self.assertEqual(cfg_dict["num_seeds"], 100)
        self.assertNotIn("seeds", cfg_dict)

    def test_explicit_seed_config(self):
        seeds = [1, 3, 7]
        cfg_dict = serialize_seed_config(seeds)
        self.assertEqual(cfg_dict["seed_mode"], "explicit")
        self.assertEqual(cfg_dict["seeds"], [1, 3, 7])
        self.assertEqual(cfg_dict["num_seeds"], 3)

    def test_small_run_experiment(self):
        with tempfile.TemporaryDirectory() as tempdir:
            args = argparse.Namespace(
                output_dir=tempdir,
                run_name="test_run",
                start_seed=1,
                num_seeds=1,
                steps_per_trial=2,
                prediction_horizon_steps=5,
                s4_prediction_horizon_steps=30,
                distance_engage_threshold=12.0,
                safety_tracking_abort_threshold=0.3,
                safety_tracking_confidence_threshold=0.6,
                safety_identification_confidence_threshold=0.6,
                safety_risk_score_threshold=0.7,
                safety_interceptability_threshold=0.5,
                observation_noise_sigma=0.0,
                observation_velocity_noise_sigma=0.0,
                safety_abort_feasibility_threshold=0.5,
                abort_horizon_steps=10,
                abort_risk_zone_safety_buffer=0.5,
                risk_k_sigma=1.0,
                risk_abort_margin=0.0,
                risk_clearance_falloff=2.0,
                late_abort_lag_steps=2,
                weight_safe_capture=1.0,
                weight_abort_success=0.5,
                weight_unsafe=2.0,
                weight_false=2.0,
                weight_late_abort=1.0,
                scenarios="S1_STRAIGHT_INTRUSION",
                policies="AlwaysEngagePolicy,SafetyGatePolicy"
            )
            
            paths = run_experiment(args)
            
            self.assertIn("records_csv", paths)
            self.assertIn("policy_summaries_csv", paths)
            self.assertIn("grouped_summaries_csv", paths)
            self.assertIn("records_json", paths)
            self.assertIn("policy_summaries_json", paths)
            self.assertIn("grouped_summaries_json", paths)
            self.assertIn("run_config_json", paths)
            
            for key, path in paths.items():
                self.assertTrue(os.path.exists(path))
                
            with open(paths["run_config_json"], "r", encoding="utf-8") as f:
                import json
                run_config_data = json.load(f)
                
            self.assertEqual(run_config_data["seed_mode"], "range")
            self.assertEqual(run_config_data["seed_start"], 1)
            self.assertEqual(run_config_data["seed_end"], 1)
            self.assertEqual(run_config_data["num_seeds"], 1)
            self.assertEqual(run_config_data["distance_engage_threshold"], 12.0)
            self.assertEqual(run_config_data["safety_risk_score_threshold"], 0.7)
            self.assertNotIn("seeds", run_config_data)

    def test_main_smoke(self):
        with tempfile.TemporaryDirectory() as tempdir:
            argv = [
                "--output-dir", tempdir,
                "--run-name", "smoke_test",
                "--num-seeds", "1",
                "--steps-per-trial", "1",
                "--scenarios", "S1_STRAIGHT_INTRUSION",
                "--policies", "AlwaysEngagePolicy",
                "--distance-engage-threshold", "12.0"
            ]
            
            exit_code = main(argv)
            self.assertEqual(exit_code, 0)
            
            # Verify one of the files was created
            expected_path = os.path.join(tempdir, "smoke_test", "records.csv")
            self.assertTrue(os.path.exists(expected_path))

if __name__ == "__main__":
    unittest.main()
