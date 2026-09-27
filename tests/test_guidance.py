import math
import unittest

from cuas_sim.guidance import (
    PNParams,
    PNOutcome,
    auc,
    intercept_lead_velocity,
    simulate_abort,
    simulate_pn_intercept,
)
from cuas_sim.types import State2D


class TestInterceptLead(unittest.TestCase):
    def test_head_on_lead_points_at_target(self):
        # Target closing straight down the -x axis: the lead solution must aim
        # along +x with the full launch speed.
        v = intercept_lead_velocity((10.0, 0.0), (-1.0, 0.0), speed=2.0)
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v[0], 2.0, places=6)
        self.assertAlmostEqual(v[1], 0.0, places=6)

    def test_crossing_target_produces_lead_angle(self):
        # Target crossing in +y: the interceptor must aim ahead of it, not at it.
        v = intercept_lead_velocity((10.0, 0.0), (0.0, 1.0), speed=2.0)
        self.assertIsNotNone(v)
        self.assertGreater(v[1], 0.0, "launch must lead the crossing target")
        self.assertAlmostEqual(math.hypot(*v), 2.0, places=6)

    def test_unreachable_geometry_returns_none(self):
        # Target fleeing faster than the interceptor can fly: no positive root.
        self.assertIsNone(intercept_lead_velocity((1.0, 0.0), (5.0, 0.0), speed=2.0))


class TestPNFlight(unittest.TestCase):
    def setUp(self):
        self.params = PNParams()

    def test_head_on_target_is_captured(self):
        out = simulate_pn_intercept(State2D(x=8.0, y=0.0, vx=-1.0, vy=0.0), (0.0, 0.0),
                                    self.params)
        self.assertTrue(out.captured)
        self.assertLessEqual(out.miss_distance, self.params.capture_radius)

    def test_crossing_target_is_captured(self):
        # The sign convention of the PN law is what this exercises: with the
        # normal flipped the interceptor would steer away and miss.
        out = simulate_pn_intercept(State2D(x=6.0, y=-4.0, vx=0.0, vy=1.0), (0.0, 0.0),
                                    self.params)
        self.assertTrue(out.captured, f"crossing intercept missed by {out.miss_distance:.3f}")

    def test_fast_fleeing_target_escapes(self):
        out = simulate_pn_intercept(State2D(x=3.0, y=0.0, vx=5.0, vy=0.0), (0.0, 0.0),
                                    self.params)
        self.assertFalse(out.captured)
        self.assertGreater(out.miss_distance, self.params.capture_radius)

    def test_maneuver_degrades_the_intercept(self):
        # A hard reversal partway through the flight must cost something; if the
        # maneuver were being ignored the two outcomes would be identical.
        start = State2D(x=8.0, y=0.0, vx=-1.0, vy=0.0)
        clean = simulate_pn_intercept(start, (0.0, 0.0), self.params)
        evading = simulate_pn_intercept(start, (0.0, 0.0), self.params,
                                        maneuver=(1.0, 0.0, 3.0))
        self.assertNotAlmostEqual(clean.time_to_go, evading.time_to_go, places=3)

    def test_outcome_fields_are_consistent(self):
        out = simulate_pn_intercept(State2D(x=8.0, y=0.0, vx=-1.0, vy=0.0), (0.0, 0.0),
                                    self.params)
        self.assertIsInstance(out, PNOutcome)
        self.assertLessEqual(out.time_to_go, out.flight_time + 1e-9)
        self.assertGreaterEqual(out.miss_distance, 0.0)


class TestAbort(unittest.TestCase):
    def test_stopping_time_matches_the_predicted_t_req(self):
        # The manuscript's t_req = tau + V_I / a_max = 0.2 + 2.0/4.0 = 0.7 s.
        p = PNParams()
        t_stop, distance = simulate_abort(p)
        expected = p.reaction_delay + p.max_speed / p.max_accel
        self.assertAlmostEqual(t_stop, expected, delta=2 * p.time_step)
        self.assertGreater(distance, 0.0)


class TestAUC(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertAlmostEqual(auc([3.0, 4.0], [1.0, 2.0]), 1.0)

    def test_inverted_separation(self):
        self.assertAlmostEqual(auc([1.0, 2.0], [3.0, 4.0]), 0.0)

    def test_ties_are_chance(self):
        self.assertAlmostEqual(auc([1.0, 1.0], [1.0, 1.0]), 0.5)

    def test_empty_class_is_nan(self):
        self.assertTrue(math.isnan(auc([], [1.0])))


if __name__ == "__main__":
    unittest.main()
