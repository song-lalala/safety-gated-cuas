"""M7 tests: per-trial metric grouping and bootstrap statistics helpers."""
import math
import unittest

from cuas_sim.scenario import ScenarioType
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner
from cuas_sim.statistics import (
    bootstrap_ci,
    bootstrap_aggregate_ci,
    paired_bootstrap_diff,
    aggregate_safe_capture,
    aggregate_unsafe_per_attempt,
    aggregate_abort_success,
    make_rws_aggregator,
)


class TestPerTrialSummaries(unittest.TestCase):

    def _small_run(self):
        cfg = MonteCarloConfig(
            seeds=[1, 2, 3],
            steps_per_trial=5,
            include_scenarios=[
                ScenarioType.S1_STRAIGHT_INTRUSION,
                ScenarioType.S4_RISK_ZONE_PROXIMITY,
            ],
            include_policies=["AlwaysEngagePolicy", "SafetyGatePolicy"],
            observation_noise_sigma=0.1,
        )
        return MonteCarloRunner(cfg).run()

    def test_per_trial_count_matches_seed_scenario_policy_product(self):
        result = self._small_run()
        trials = result.per_trial_summaries()
        # 3 seeds × 2 scenarios × 2 policies = 12 trials
        self.assertEqual(len(trials), 12)

    def test_per_trial_counts_sum_to_global(self):
        result = self._small_run()
        trials = result.per_trial_summaries()
        # Sum engage_count per policy and compare to global summary.
        for global_summary in result.summaries:
            policy = global_summary.policy_name
            policy_trials = [t for t in trials if t.policy_name == policy]
            self.assertEqual(
                sum(t.engage_count for t in policy_trials),
                global_summary.engage_count,
                f"engage_count mismatch for {policy}",
            )
            self.assertEqual(
                sum(t.unsafe_engagement_count for t in policy_trials),
                global_summary.unsafe_engagement_count,
                f"unsafe mismatch for {policy}",
            )
            self.assertEqual(
                sum(t.safe_opportunity_count for t in policy_trials),
                global_summary.safe_opportunity_count,
                f"safe_opp mismatch for {policy}",
            )

    def test_to_dict_round_trip(self):
        result = self._small_run()
        trials = result.per_trial_summaries()
        d = trials[0].to_dict()
        self.assertIn("seed", d)
        self.assertIn("scenario_type", d)
        self.assertIn("engage_count", d)
        self.assertIn("safe_opportunity_count", d)
        self.assertIn("abort_required", d)


class TestBootstrap(unittest.TestCase):

    def test_bootstrap_ci_on_constant_returns_constant(self):
        # CI on a constant sequence collapses to the constant.
        point, lo, hi = bootstrap_ci([0.5] * 50, n_boot=200, seed=0)
        self.assertAlmostEqual(point, 0.5)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)

    def test_bootstrap_ci_contains_point(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        point, lo, hi = bootstrap_ci(values, n_boot=500, seed=0)
        self.assertAlmostEqual(point, sum(values) / len(values), places=6)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)

    def test_bootstrap_aggregate_runs_on_trial_objects(self):
        class FakeTrial:
            def __init__(self, eng, unsafe):
                self.engage_count = eng
                self.unsafe_engagement_count = unsafe
        trials = [FakeTrial(10, 1), FakeTrial(10, 2), FakeTrial(10, 0)]
        point, lo, hi = bootstrap_aggregate_ci(
            trials, aggregate_unsafe_per_attempt, n_boot=200, seed=0
        )
        # Aggregate = 3 unsafe / 30 engages = 0.1
        self.assertAlmostEqual(point, 0.1, places=6)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)

    def test_paired_bootstrap_diff_zero_when_inputs_identical(self):
        class FakeTrial:
            def __init__(self, eng, unsafe):
                self.engage_count = eng
                self.unsafe_engagement_count = unsafe
        a = [FakeTrial(10, 1), FakeTrial(10, 2), FakeTrial(10, 0)]
        b = list(a)  # identical
        result = paired_bootstrap_diff(a, b, aggregate_unsafe_per_attempt,
                                       n_boot=200, seed=0)
        self.assertAlmostEqual(result["point_diff"], 0.0, places=6)
        self.assertAlmostEqual(result["ci_lo"], 0.0, places=6)
        self.assertAlmostEqual(result["ci_hi"], 0.0, places=6)


class TestRWSAggregator(unittest.TestCase):

    def test_rws_matches_summary_within_rounding(self):
        cfg = MonteCarloConfig(
            seeds=list(range(1, 21)),
            steps_per_trial=5,
            include_scenarios=[ScenarioType.S1_STRAIGHT_INTRUSION,
                               ScenarioType.S4_RISK_ZONE_PROXIMITY],
            include_policies=["SafetyGatePolicy"],
            observation_noise_sigma=0.1,
        )
        result = MonteCarloRunner(cfg).run()
        trials = result.per_trial_summaries()
        agg = make_rws_aggregator()
        # Aggregate RWS computed by bootstrap aggregator should equal the
        # global MetricsSummary's risk_weighted_score (modulo floating noise).
        global_rws = result.summaries[0].risk_weighted_score
        from_trials = agg(trials)
        self.assertTrue(math.isclose(global_rws, from_trials, rel_tol=1e-9, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
