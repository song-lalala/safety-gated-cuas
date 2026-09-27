import unittest
import tempfile
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.analyze_results import (
    load_result_files,
    build_policy_summary_table,
    extract_safety_gate_reasons,
    write_csv_rows,
    write_markdown_report,
    analyze_results,
    main,
    format_policy_label,
    format_reason_label
)

class TestAnalysisScript(unittest.TestCase):

    def setUp(self):
        self.mock_policy_summaries = [
            {
                "policy_name": "SafetyGatePolicy",
                "total_decisions": 100,
                "engage_count": 80,
                "track_count": 10,
                "abort_count": 10,
                "engagement_attempt_rate": 0.8,
                "unsafe_engagement_count": 0,
                "false_engagement_count": 0,
                "non_interceptable_engagement_count": 0,
                "invalid_engagement_count": 0,
                "safe_engagement_count": 80,
                "safe_engagement_per_attempt_rate": 1.0,
                "unsafe_engagement_per_attempt_rate": 0.0,
                "false_engagement_per_attempt_rate": 0.0,
                "non_interceptable_engagement_per_attempt_rate": 0.0
            }
        ]
        
        self.mock_grouped_summaries = [
            {
                "scenario_type": "S1_STRAIGHT_INTRUSION",
                "policy_name": "SafetyGatePolicy",
                "total_decisions": 50,
                "reason_counts": {
                    "engage_conditions_satisfied": 50
                }
            },
            {
                "scenario_type": "S3_LOW_IDENTIFICATION_CONFIDENCE",
                "policy_name": "SafetyGatePolicy",
                "total_decisions": 50,
                "reason_counts": '{"identification_confidence_too_low": 50}'
            }
        ]
        
        self.mock_records = [{"test": "record"}]

    def test_load_result_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "policy_summaries.json"), "w") as f:
                json.dump(self.mock_policy_summaries, f)
            with open(os.path.join(tempdir, "grouped_summaries.json"), "w") as f:
                json.dump(self.mock_grouped_summaries, f)
            with open(os.path.join(tempdir, "records.json"), "w") as f:
                json.dump(self.mock_records, f)
                
            data = load_result_files(tempdir)
            self.assertIn("policy_summaries", data)
            self.assertIn("grouped_summaries", data)
            self.assertIn("records", data)
            self.assertEqual(len(data["policy_summaries"]), 1)

    def test_build_policy_summary_table(self):
        table = build_policy_summary_table(self.mock_policy_summaries)
        self.assertEqual(len(table), 1)
        row = table[0]
        self.assertEqual(row["policy_name"], "SafetyGatePolicy")
        self.assertEqual(row["safe_engagement_per_attempt_rate"], 1.0)
        self.assertIn("non_interceptable_engagement_count", row)

    def test_extract_safety_gate_reasons(self):
        reasons = extract_safety_gate_reasons(self.mock_grouped_summaries)
        
        # 1 from S1, 1 from S3
        self.assertEqual(len(reasons), 2)
        
        s3_reason = next(r for r in reasons if r["scenario_type"] == "S3_LOW_IDENTIFICATION_CONFIDENCE")
        self.assertEqual(s3_reason["reason"], "identification_confidence_too_low")
        self.assertEqual(s3_reason["count"], 50)

    def test_format_policy_label(self):
        self.assertEqual(format_policy_label("AlwaysEngagePolicy"), "Always")
        self.assertEqual(format_policy_label("DistanceOnlyPolicy"), "Distance")
        self.assertEqual(format_policy_label("InterceptabilityOnlyPolicy"), "Interceptability")
        self.assertEqual(format_policy_label("SafetyGatePolicy"), "SafetyGate")
        self.assertEqual(format_policy_label("UnknownPolicy"), "UnknownPolicy")

    def test_format_reason_label(self):
        self.assertEqual(format_reason_label("engage_conditions_satisfied"), "Engage OK")
        self.assertEqual(format_reason_label("not_detected"), "Not Detected")
        self.assertEqual(format_reason_label("unknown_reason"), "unknown_reason")

    def test_write_csv_rows(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "test.csv")
            write_csv_rows([{"a": 1, "b": [1, 2]}], file_path)
            
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("a", content)
                self.assertIn("b", content)
                self.assertIn("[1, 2]", content)

    def test_write_markdown_report(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "report.md")
            table = build_policy_summary_table(self.mock_policy_summaries)
            sg_reasons = extract_safety_gate_reasons(self.mock_grouped_summaries)
            
            write_markdown_report(table, self.mock_grouped_summaries, sg_reasons, file_path)
            
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Safety-Gated C-UAS Simulation Pilot Report", content)
                self.assertIn("SafetyGatePolicy", content)
                self.assertIn("Generated Figures", content)
                self.assertIn("policy_action_counts.png", content)

    def test_analyze_results_no_plots(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "policy_summaries.json"), "w") as f:
                json.dump(self.mock_policy_summaries, f)
            with open(os.path.join(tempdir, "grouped_summaries.json"), "w") as f:
                json.dump(self.mock_grouped_summaries, f)
            with open(os.path.join(tempdir, "records.json"), "w") as f:
                json.dump(self.mock_records, f)
                
            paths = analyze_results(tempdir, make_plots=False)
            
            self.assertIn("policy_summary_table_csv", paths)
            self.assertIn("grouped_summary_table_csv", paths)
            self.assertIn("safety_gate_reasons_csv", paths)
            self.assertIn("report_md", paths)
            
            self.assertNotIn("policy_risk_counts_png", paths)
            self.assertNotIn("policy_action_counts_png", paths)
            self.assertNotIn("safety_gate_reasons_png", paths)
            
            for path in paths.values():
                self.assertTrue(os.path.exists(path))

    def test_main_smoke(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "policy_summaries.json"), "w") as f:
                json.dump(self.mock_policy_summaries, f)
            with open(os.path.join(tempdir, "grouped_summaries.json"), "w") as f:
                json.dump(self.mock_grouped_summaries, f)
            with open(os.path.join(tempdir, "records.json"), "w") as f:
                json.dump(self.mock_records, f)
                
            argv = ["--input-dir", tempdir, "--no-plots"]
            exit_code = main(argv)
            self.assertEqual(exit_code, 0)
            
            # check default output dir creation
            expected_report = os.path.join(tempdir, "analysis", "report.md")
            self.assertTrue(os.path.exists(expected_report))

if __name__ == "__main__":
    unittest.main()
