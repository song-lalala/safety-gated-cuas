import unittest
import os
import json
import tempfile

from cuas_sim.scenario import ScenarioType
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner, MonteCarloResult
from cuas_sim.results import write_dict_rows_csv, write_json, export_monte_carlo_result

class TestResults(unittest.TestCase):
    def test_grouped_summary(self):
        config = MonteCarloConfig(
            seeds=[1, 2],
            steps_per_trial=3,
            include_scenarios=[
                ScenarioType.S1_STRAIGHT_INTRUSION,
                ScenarioType.S3_LOW_IDENTIFICATION_CONFIDENCE
            ],
            include_policies=[
                "AlwaysEngagePolicy",
                "SafetyGatePolicy"
            ]
        )
        runner = MonteCarloRunner(config)
        result = runner.run()
        
        grouped_dicts = result.grouped_summaries_to_dicts()
        
        # 2 scenarios * 2 policies = 4 groups
        self.assertEqual(len(grouped_dicts), 4)
        
        for d in grouped_dicts:
            self.assertIn("scenario_type", d)
            self.assertIn("policy_name", d)
            self.assertIn("total_decisions", d)

    def test_csv_writer(self):
        import csv
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.csv")
            rows = [
                {"a": 1, "b": {"x": 2}},
                {"a": 2, "c": [1, 2]}
            ]
            write_dict_rows_csv(rows, file_path)
            
            self.assertTrue(os.path.exists(file_path))
            
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                out_rows = list(reader)
                
            self.assertEqual(len(out_rows), 2)
            
            # First row checks
            b_val = out_rows[0].get("b")
            self.assertIsNotNone(b_val)
            self.assertEqual(json.loads(b_val), {"x": 2})
            
            # Second row checks
            c_val = out_rows[1].get("c")
            self.assertIsNotNone(c_val)
            self.assertEqual(json.loads(c_val), [1, 2])

    def test_json_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.json")
            data = {"a": 1, "b": [1, 2, 3]}
            write_json(data, file_path)
            
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                
            self.assertEqual(loaded["a"], 1)
            self.assertEqual(loaded["b"], [1, 2, 3])

    def test_export_monte_carlo_result(self):
        config = MonteCarloConfig(
            seeds=[42],
            steps_per_trial=1,
            include_scenarios=[ScenarioType.S1_STRAIGHT_INTRUSION],
            include_policies=["AlwaysEngagePolicy"]
        )
        runner = MonteCarloRunner(config)
        result = runner.run()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_monte_carlo_result(result, tmpdir)
            
            self.assertIn("records_csv", paths)
            self.assertIn("policy_summaries_csv", paths)
            self.assertIn("grouped_summaries_csv", paths)
            self.assertIn("records_json", paths)
            self.assertIn("policy_summaries_json", paths)
            self.assertIn("grouped_summaries_json", paths)
            
            for key, path in paths.items():
                self.assertTrue(os.path.exists(path))

    def test_nested_enum_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_enum.json")
            data = {"metadata": {"scenario_type": ScenarioType.S1_STRAIGHT_INTRUSION}}
            write_json(data, file_path)
            
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                
            self.assertIn("metadata", loaded)
            self.assertIn("scenario_type", loaded["metadata"])
            # The enum should be serialized to its value (which is "S1_STRAIGHT_INTRUSION" string)
            self.assertEqual(loaded["metadata"]["scenario_type"], "S1_STRAIGHT_INTRUSION")

if __name__ == "__main__":
    unittest.main()
