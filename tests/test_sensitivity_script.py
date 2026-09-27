"""Smoke tests for the M6 sensitivity-sweep scripts."""
import json
import os
import sys
import tempfile
import unittest

# Allow imports of `experiments.*` from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_sensitivity import main as sweep_main
from experiments.analyze_sensitivity import main as analyze_main


class TestSensitivitySweep(unittest.TestCase):

    def test_small_sigma_sweep_produces_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--sweep-param", "observation_noise_sigma",
                "--sweep-values", "0.0,0.1",
                "--base-run-name", "test_sweep",
                "--output-dir", tmp,
                "--num-seeds", "2",
                "--steps-per-trial", "3",
                "--scenarios", "S1_STRAIGHT_INTRUSION,S4_RISK_ZONE_PROXIMITY",
                "--policies", "AlwaysEngagePolicy,SafetyGatePolicy",
                "--distance-engage-threshold", "12.0",
            ]
            exit_code = sweep_main(argv)
            self.assertEqual(exit_code, 0)

            sweep_dir = os.path.join(tmp, "test_sweep")
            self.assertTrue(os.path.exists(sweep_dir))
            self.assertTrue(os.path.exists(os.path.join(sweep_dir, "sweep_summary.csv")))

            with open(os.path.join(sweep_dir, "sweep_summary.json"), "r", encoding="utf-8") as f:
                rows = json.load(f)
            # 2 sweep values × 2 policies = 4 rows
            self.assertEqual(len(rows), 4)
            for r in rows:
                self.assertIn("sweep_param", r)
                self.assertIn("sweep_value", r)
                self.assertIn("risk_weighted_score", r)
                self.assertIn("safe_opportunity_capture_rate", r)
                self.assertIn("abort_success_rate", r)

            # The two sweep values should appear in the summary
            values = {r["sweep_value"] for r in rows}
            self.assertEqual(values, {0.0, 0.1})

            # Now analyze the sweep
            exit_code = analyze_main(["--input-dir", sweep_dir, "--no-plots"])
            self.assertEqual(exit_code, 0)
            analysis_dir = os.path.join(sweep_dir, "analysis")
            self.assertTrue(os.path.exists(os.path.join(analysis_dir, "sensitivity_report.md")))
            self.assertTrue(os.path.exists(os.path.join(analysis_dir, "pivot_risk_weighted_score.csv")))

    def test_2d_cross_product_sweep(self):
        # M8: --sweep-param-2 produces Cartesian-product sub-runs.
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--sweep-param", "observation_noise_sigma",
                "--sweep-values", "0.0,0.2",
                "--sweep-param-2", "risk_k_sigma",
                "--sweep-values-2", "0.0,1.0",
                "--base-run-name", "test_2d",
                "--output-dir", tmp,
                "--num-seeds", "1",
                "--steps-per-trial", "2",
                "--scenarios", "S4_RISK_ZONE_PROXIMITY",
                "--policies", "SafetyGatePolicy",
            ]
            exit_code = sweep_main(argv)
            self.assertEqual(exit_code, 0)

            sweep_dir = os.path.join(tmp, "test_2d")
            with open(os.path.join(sweep_dir, "sweep_meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["dimension"], 2)
            self.assertEqual(meta["sweep_param"], "observation_noise_sigma")
            self.assertEqual(meta["sweep_param_2"], "risk_k_sigma")

            with open(os.path.join(sweep_dir, "sweep_summary.json"), "r", encoding="utf-8") as f:
                rows = json.load(f)
            # 2 σ × 2 k_sigma × 1 policy = 4 rows
            self.assertEqual(len(rows), 4)
            for r in rows:
                self.assertIn("sweep_observation_noise_sigma", r)
                self.assertIn("sweep_risk_k_sigma", r)

            # Analyze: 2D mode should not crash, produces pivot CSVs.
            exit_code = analyze_main(["--input-dir", sweep_dir, "--no-plots"])
            self.assertEqual(exit_code, 0)
            analysis = os.path.join(sweep_dir, "analysis")
            self.assertTrue(any(name.startswith("pivot2d_") for name in os.listdir(analysis)))

    def test_threshold_sweep_runs(self):
        # Verify the sweep infrastructure also works for a non-sigma parameter.
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "--sweep-param", "safety_risk_score_threshold",
                "--sweep-values", "0.5,0.7,0.9",
                "--base-run-name", "test_risk_sweep",
                "--output-dir", tmp,
                "--num-seeds", "1",
                "--steps-per-trial", "2",
                "--scenarios", "S4_RISK_ZONE_PROXIMITY",
                "--policies", "SafetyGatePolicy",
            ]
            exit_code = sweep_main(argv)
            self.assertEqual(exit_code, 0)

            with open(os.path.join(tmp, "test_risk_sweep", "sweep_summary.json"), "r", encoding="utf-8") as f:
                rows = json.load(f)
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
