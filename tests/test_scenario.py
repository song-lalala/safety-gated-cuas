import unittest
import math
from cuas_sim.scenario import (
    get_s1_scenario, get_s2_scenario, get_s3_scenario, 
    get_s4_scenario, get_s5_scenario, get_s6_scenario, 
    ScenarioType, Scenario, SCENARIO_FACTORIES, 
    get_all_scenario_types, get_scenario_factory
)
from cuas_sim import Simulator

class TestScenarioEnumValues(unittest.TestCase):
    def test_enum_values_are_strings(self):
        self.assertIsInstance(ScenarioType.S1_STRAIGHT_INTRUSION.value, str)
        self.assertIsInstance(ScenarioType.S2_EVASIVE_MANEUVER.value, str)
        self.assertIsInstance(ScenarioType.S3_LOW_IDENTIFICATION_CONFIDENCE.value, str)
        self.assertIsInstance(ScenarioType.S4_RISK_ZONE_PROXIMITY.value, str)
        self.assertIsInstance(ScenarioType.S5_TRACKING_LOSS.value, str)
        self.assertIsInstance(ScenarioType.S6_NON_INTERCEPTABLE_TARGET.value, str)


class TestScenarioS1(unittest.TestCase):
    def test_creation(self):
        seed = 12345
        scenario1 = get_s1_scenario(seed)
        scenario2 = get_s1_scenario(seed)
        self.assertIsInstance(scenario1, Scenario)
        self.assertEqual(scenario1.scenario_type, ScenarioType.S1_STRAIGHT_INTRUSION)
        self.assertAlmostEqual(scenario1.target.state.x, scenario2.target.state.x)
        self.assertAlmostEqual(scenario1.target.state.y, scenario2.target.state.y)
        self.assertAlmostEqual(scenario1.target.state.vx, scenario2.target.state.vx)
        self.assertAlmostEqual(scenario1.target.state.vy, scenario2.target.state.vy)
        scenario3 = get_s1_scenario(seed + 1)
        diffs = []
        for attr in ("x", "y", "vx", "vy"):
            diffs.append(getattr(scenario1.target.state, attr) != getattr(scenario3.target.state, attr))
        self.assertTrue(any(diffs))

    def test_simulator_step(self):
        scenario = get_s1_scenario(42)
        sim = Simulator(config=scenario.config, target=scenario.target, interceptor=scenario.interceptor)
        sim.step()
        self.assertAlmostEqual(sim.time, scenario.config.time_step)
        self.assertEqual(len(sim.history), 1)

class TestScenarioS2(unittest.TestCase):
    def test_creation_and_hook(self):
        seed = 98765
        scenario = get_s2_scenario(seed)
        self.assertIsInstance(scenario, Scenario)
        self.assertEqual(scenario.scenario_type, ScenarioType.S2_EVASIVE_MANEUVER)
        # Ensure hook exists
        self.assertIsNotNone(scenario.step_hook)
        # Run simulator with the hook attached
        sim = Simulator(config=scenario.config, target=scenario.target, interceptor=scenario.interceptor, step_hook=scenario.step_hook)
        # Run enough steps to cover the maneuver step (max 15)
        sim.run(20)
        # After running, target velocity should have changed from initial velocity

        # Verify that at least one velocity component differs from the initial (pre‑hook) state stored in history
        recorded = sim.history[0][1]  # first state snapshot
        self.assertFalse(
            recorded.vx == sim.history[-1][1].vx and recorded.vy == sim.history[-1][1].vy,
            "Target velocity should change after maneuver"
        )
class TestScenarioS3(unittest.TestCase):
    def test_low_identification_confidence(self):
        seed = 13579
        scenario = get_s3_scenario(seed)
        self.assertIsInstance(scenario, Scenario)
        self.assertEqual(scenario.scenario_type, ScenarioType.S3_LOW_IDENTIFICATION_CONFIDENCE)
        # metadata presence
        self.assertIn("identification_confidence", scenario.metadata)
        ic = scenario.metadata["identification_confidence"]
        self.assertGreaterEqual(ic, 0.2)
        self.assertLessEqual(ic, 0.5)
        # reproducibility with same seed
        scenario2 = get_s3_scenario(seed)
        self.assertEqual(scenario.metadata["identification_confidence"], scenario2.metadata["identification_confidence"])
        self.assertAlmostEqual(scenario.target.state.x, scenario2.target.state.x)
        self.assertAlmostEqual(scenario.target.state.y, scenario2.target.state.y)
        self.assertAlmostEqual(scenario.target.state.vx, scenario2.target.state.vx)
        self.assertAlmostEqual(scenario.target.state.vy, scenario2.target.state.vy)
        # different seed leads to different confidence or state
        scenario3 = get_s3_scenario(seed + 1)
        diff_conf = scenario.metadata["identification_confidence"] != scenario3.metadata["identification_confidence"]
        diff_state = any(
            getattr(scenario.target.state, attr) != getattr(scenario3.target.state, attr)
            for attr in ("x", "y", "vx", "vy")
        )
        self.assertTrue(diff_conf or diff_state)
        # simulator runs at least one step
        sim = Simulator(config=scenario.config, target=scenario.target, interceptor=scenario.interceptor)
        sim.step()
        self.assertEqual(len(sim.history), 1)


class TestScenarioS4(unittest.TestCase):
    def test_s4_scenario(self):
        seed = 424242
        scenario = get_s4_scenario(seed)
        # a. type check
        self.assertIsInstance(scenario, Scenario)
        self.assertEqual(scenario.scenario_type, ScenarioType.S4_RISK_ZONE_PROXIMITY)
        # b‑e. metadata checks
        meta = scenario.metadata
        self.assertIn("risk_zone_center", meta)
        self.assertIn("risk_zone_radius", meta)
        self.assertIn("risk_proximity_level", meta)
        self.assertIsInstance(meta["risk_zone_center"], tuple)
        self.assertEqual(len(meta["risk_zone_center"]), 2)
        self.assertIsInstance(meta["risk_zone_radius"], (int, float))
        self.assertGreater(meta["risk_zone_radius"], 0)
        self.assertIn("expected_min_distance_to_risk_zone", meta)
        self.assertIn("risk_clearance", meta)
        self.assertIsInstance(meta["expected_min_distance_to_risk_zone"], (int, float))
        self.assertIsInstance(meta["risk_clearance"], (int, float))
        self.assertGreater(meta["risk_clearance"], 0)
        self.assertLessEqual(meta["risk_clearance"], 0.5)
        self.assertIn("proximity_step", meta)
        # f. reproducibility
        scenario2 = get_s4_scenario(seed)
        self.assertEqual(scenario.metadata, scenario2.metadata)
        self.assertAlmostEqual(scenario.target.state.x, scenario2.target.state.x)
        self.assertAlmostEqual(scenario.target.state.y, scenario2.target.state.y)
        self.assertAlmostEqual(scenario.target.state.vx, scenario2.target.state.vx)
        self.assertAlmostEqual(scenario.target.state.vy, scenario2.target.state.vy)
        # g. simulator step
        sim = Simulator(config=scenario.config, target=scenario.target, interceptor=scenario.interceptor)
        sim.run(30)
        # Compute minimal distance from target positions to risk zone center
        min_dist = min(
            ((target_state.x - meta["risk_zone_center"][0])**2 + (target_state.y - meta["risk_zone_center"][1])**2)**0.5
            for _time, target_state, _interceptor_state in sim.history
        )
        self.assertLessEqual(min_dist, meta["risk_zone_radius"] + 0.5)
        self.assertGreaterEqual(len(sim.history), 1)

class TestScenarioS5(unittest.TestCase):
    def test_s5_scenario(self):
        seed = 55555
        scenario = get_s5_scenario(seed)
        # a. type check
        self.assertIsInstance(scenario, Scenario)
        self.assertEqual(scenario.scenario_type, ScenarioType.S5_TRACKING_LOSS)
        
        # metadata checks
        meta = scenario.metadata
        # b, c, d
        self.assertIn("tracking_loss_start_step", meta)
        self.assertIn("tracking_loss_duration_steps", meta)
        self.assertIn("tracking_loss_end_step", meta)
        
        # e. end_step = start_step + duration
        self.assertEqual(
            meta["tracking_loss_end_step"], 
            meta["tracking_loss_start_step"] + meta["tracking_loss_duration_steps"]
        )
        
        # f. nominal confidence in [0.7, 1.0]
        self.assertGreaterEqual(meta["nominal_tracking_confidence"], 0.7)
        self.assertLessEqual(meta["nominal_tracking_confidence"], 1.0)
        
        # g. lost confidence in [0.0, 0.3]
        self.assertGreaterEqual(meta["lost_tracking_confidence"], 0.0)
        self.assertLessEqual(meta["lost_tracking_confidence"], 0.3)
        
        # h. lost < nominal
        self.assertLess(meta["lost_tracking_confidence"], meta["nominal_tracking_confidence"])
        
        # i. reproducibility
        scenario2 = get_s5_scenario(seed)
        self.assertEqual(scenario.metadata, scenario2.metadata)
        self.assertAlmostEqual(scenario.target.state.x, scenario2.target.state.x)
        self.assertAlmostEqual(scenario.target.state.y, scenario2.target.state.y)
        self.assertAlmostEqual(scenario.target.state.vx, scenario2.target.state.vx)
        self.assertAlmostEqual(scenario.target.state.vy, scenario2.target.state.vy)
        
        # j. simulator step
        sim = Simulator(config=scenario.config, target=scenario.target, interceptor=scenario.interceptor)
        sim.step()
        self.assertGreaterEqual(len(sim.history), 1)

class TestScenarioS6(unittest.TestCase):
    def test_s6_scenario(self):
        seed = 66666
        scenario = get_s6_scenario(seed)
        
        # a. type check
        self.assertIsInstance(scenario, Scenario)
        self.assertEqual(scenario.scenario_type, ScenarioType.S6_NON_INTERCEPTABLE_TARGET)
        
        meta = scenario.metadata
        
        # b, c, d
        self.assertIn("target_speed", meta)
        self.assertIn("interceptor_nominal_max_speed", meta)
        self.assertIn("speed_ratio", meta)
        
        # e. speed_ratio < 1.0
        self.assertLess(meta["speed_ratio"], 1.0)
        
        # f, g. initial_distance presence and > 0
        self.assertIn("initial_distance", meta)
        self.assertGreater(meta["initial_distance"], 0)
        
        # h. engagement_horizon_steps
        self.assertIn("engagement_horizon_steps", meta)
        
        # i. interceptability_label
        self.assertEqual(meta["interceptability_label"], "non_interceptable")
        
        # j. non_interceptable_reason
        self.assertIsInstance(meta["non_interceptable_reason"], str)
        
        # k. reproducibility
        scenario2 = get_s6_scenario(seed)
        self.assertEqual(scenario.metadata, scenario2.metadata)
        self.assertAlmostEqual(scenario.target.state.x, scenario2.target.state.x)
        self.assertAlmostEqual(scenario.target.state.y, scenario2.target.state.y)
        self.assertAlmostEqual(scenario.target.state.vx, scenario2.target.state.vx)
        self.assertAlmostEqual(scenario.target.state.vy, scenario2.target.state.vy)
        
        # l. simulator step
        sim = Simulator(config=scenario.config, target=scenario.target, interceptor=scenario.interceptor)
        sim.step()
        self.assertGreaterEqual(len(sim.history), 1)


class TestScenarioSanity(unittest.TestCase):
    def test_scenario_registry(self):
        # get_all_scenario_types returns exactly 6 scenario types in order
        all_types = get_all_scenario_types()
        self.assertEqual(len(all_types), 6)
        self.assertEqual(all_types, [
            ScenarioType.S1_STRAIGHT_INTRUSION,
            ScenarioType.S2_EVASIVE_MANEUVER,
            ScenarioType.S3_LOW_IDENTIFICATION_CONFIDENCE,
            ScenarioType.S4_RISK_ZONE_PROXIMITY,
            ScenarioType.S5_TRACKING_LOSS,
            ScenarioType.S6_NON_INTERCEPTABLE_TARGET,
        ])
        
        # SCENARIO_FACTORIES contains all types
        for s_type in all_types:
            self.assertIn(s_type, SCENARIO_FACTORIES)
            
        # Check factory generation and properties
        for s_type in all_types:
            factory = get_scenario_factory(s_type)
            scenario = factory(seed=123)
            
            # Returns Scenario object
            self.assertIsInstance(scenario, Scenario)
            
            # Type matches registry key
            self.assertEqual(scenario.scenario_type, s_type)
            
            # Metadata has name and description
            self.assertIn("scenario_name", scenario.metadata)
            self.assertIn("description", scenario.metadata)
            
            # Can run 1 step in Simulator
            sim = Simulator(
                config=scenario.config, 
                target=scenario.target, 
                interceptor=scenario.interceptor,
                step_hook=scenario.step_hook
            )
            sim.step()
            self.assertGreaterEqual(len(sim.history), 1)
