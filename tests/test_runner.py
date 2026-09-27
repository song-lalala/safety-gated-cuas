import unittest

from cuas_sim.scenario import ScenarioType
from cuas_sim.policies import DecisionAction
from cuas_sim.runner import (
    MonteCarloConfig,
    MonteCarloRunner,
    get_default_policy_factories
)

class TestRunner(unittest.TestCase):
    
    def test_mc_config_basic(self):
        config = MonteCarloConfig(seeds=[1, 2])
        self.assertEqual(config.seeds, [1, 2])
        self.assertGreater(config.steps_per_trial, 0)
        self.assertEqual(config.distance_engage_threshold, 5.0)
        self.assertEqual(config.safety_tracking_abort_threshold, 0.3)
        self.assertEqual(config.safety_tracking_confidence_threshold, 0.6)
        self.assertEqual(config.safety_identification_confidence_threshold, 0.6)
        self.assertEqual(config.safety_risk_score_threshold, 0.7)
        self.assertEqual(config.safety_interceptability_threshold, 0.5)
        
    def test_policy_factories(self):
        factories = get_default_policy_factories()
        self.assertIn("AlwaysEngagePolicy", factories)
        self.assertIn("DistanceOnlyPolicy", factories)
        self.assertIn("InterceptabilityOnlyPolicy", factories)
        self.assertIn("SafetyGatePolicy", factories)
        
    def test_small_mc_run(self):
        config = MonteCarloConfig(
            seeds=[1, 2],
            steps_per_trial=3,
            include_scenarios=[ScenarioType.S1_STRAIGHT_INTRUSION],
            include_policies=["AlwaysEngagePolicy", "SafetyGatePolicy"]
        )
        runner = MonteCarloRunner(config)
        result = runner.run()
        
        # 2 seeds * 1 scenario * 2 policies * 3 steps = 12 records
        self.assertEqual(len(result.records), 12)
        # 2 policies -> 2 summaries
        self.assertEqual(len(result.summaries), 2)
        
        records_dicts = result.records_to_dicts()
        self.assertIsInstance(records_dicts, list)
        self.assertEqual(len(records_dicts), 12)
        self.assertIsInstance(records_dicts[0], dict)
        
        summaries_dicts = result.summaries_to_dicts()
        self.assertIsInstance(summaries_dicts, list)
        self.assertEqual(len(summaries_dicts), 2)
        self.assertIsInstance(summaries_dicts[0], dict)

    def test_metadata(self):
        config = MonteCarloConfig(
            seeds=[42],
            steps_per_trial=1,
            include_scenarios=[ScenarioType.S1_STRAIGHT_INTRUSION],
            include_policies=["AlwaysEngagePolicy"]
        )
        result = MonteCarloRunner(config).run()
        
        self.assertGreaterEqual(len(result.records), 1)
        first_record = result.records[0]
        
        self.assertIn("trial_index", first_record.metadata)
        self.assertIn("seed", first_record.metadata)
        self.assertIn("scenario_type", first_record.metadata)
        self.assertIn("policy_name", first_record.metadata)
        
        self.assertEqual(first_record.metadata["trial_index"], 0)
        self.assertEqual(first_record.metadata["seed"], 42)
        self.assertEqual(first_record.metadata["scenario_type"], ScenarioType.S1_STRAIGHT_INTRUSION.value)
        self.assertEqual(first_record.metadata["policy_name"], "AlwaysEngagePolicy")

    def test_scenario_policy_isolation(self):
        config = MonteCarloConfig(
            seeds=[42],
            steps_per_trial=1,
            include_scenarios=[ScenarioType.S1_STRAIGHT_INTRUSION],
            include_policies=["AlwaysEngagePolicy", "SafetyGatePolicy"]
        )
        result = MonteCarloRunner(config).run()
        
        self.assertEqual(len(result.records), 2)
        record1 = result.records[0]
        record2 = result.records[1]
        
        # Ensure they are from different policies
        self.assertEqual(record1.policy_name, "AlwaysEngagePolicy")
        self.assertEqual(record2.policy_name, "SafetyGatePolicy")
        
        # Ensure the scenario objects were independent but started with same seed
        self.assertEqual(record1.distance_to_target, record2.distance_to_target)
        self.assertEqual(record1.tracking_confidence, record2.tracking_confidence)

    def test_s5_tracking_loss_behavior(self):
        config = MonteCarloConfig(
            seeds=[42],
            steps_per_trial=20, # Need to be large enough to hit tracking loss
            include_scenarios=[ScenarioType.S5_TRACKING_LOSS],
            include_policies=["SafetyGatePolicy"]
        )
        result = MonteCarloRunner(config).run()
        
        abort_count = sum(1 for r in result.records if r.action == DecisionAction.ABORT)
        self.assertGreaterEqual(abort_count, 1)

    def test_s6_non_interceptable_behavior(self):
        config = MonteCarloConfig(
            seeds=[42],
            steps_per_trial=3,
            include_scenarios=[ScenarioType.S6_NON_INTERCEPTABLE_TARGET],
            include_policies=["SafetyGatePolicy"]
        )
        result = MonteCarloRunner(config).run()
        
        track_count = sum(1 for r in result.records if r.action == DecisionAction.TRACK)
        self.assertGreaterEqual(track_count, 1)

    def test_distance_policy_threshold_behavior(self):
        # 1. Test with threshold=12.0 -> Should ENGAGE when distance is < 12.0
        config_12 = MonteCarloConfig(
            seeds=[1],
            steps_per_trial=1,
            distance_engage_threshold=12.0,
            include_scenarios=[ScenarioType.S1_STRAIGHT_INTRUSION],
            include_policies=["DistanceOnlyPolicy"]
        )
        result_12 = MonteCarloRunner(config_12).run()
        record_12 = result_12.records[0]
        # Depending on random seed 1 starting pos, it's typically within 12.0 km (often ~10km).
        # We can just assert it is ENGAGE. If not, it means initial distance > 12.0,
        # but in S1 typical target is ~10km. 
        self.assertEqual(record_12.action, DecisionAction.ENGAGE)
        
        # 2. Test with threshold=1.0 -> Should TRACK
        config_1 = MonteCarloConfig(
            seeds=[1],
            steps_per_trial=1,
            distance_engage_threshold=1.0,
            include_scenarios=[ScenarioType.S1_STRAIGHT_INTRUSION],
            include_policies=["DistanceOnlyPolicy"]
        )
        result_1 = MonteCarloRunner(config_1).run()
        record_1 = result_1.records[0]
        self.assertEqual(record_1.action, DecisionAction.TRACK)

    def test_safety_gate_policy_threshold_behavior(self):
        # Set a very low risk score threshold so that even slight risk triggers the gate.
        # S4 generates risk based on distance to risk zone.
        config_low_risk = MonteCarloConfig(
            seeds=[1],
            steps_per_trial=1,
            safety_risk_score_threshold=0.0,
            include_scenarios=[ScenarioType.S4_RISK_ZONE_PROXIMITY],
            include_policies=["SafetyGatePolicy"]
        )
        result_low_risk = MonteCarloRunner(config_low_risk).run()
        record_low = result_low_risk.records[0]
        # Gate triggers TRACK/ABORT because risk exceeds 0.0 threshold.
        self.assertIn(record_low.action, [DecisionAction.TRACK, DecisionAction.ABORT])

if __name__ == "__main__":
    unittest.main()
