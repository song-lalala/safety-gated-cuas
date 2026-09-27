import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Optional, Dict, List, Any

def load_json_file(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_policy_label(policy_name: str) -> str:
    mapping = {
        "AlwaysEngagePolicy": "Always",
        "DistanceOnlyPolicy": "Distance",
        "InterceptabilityOnlyPolicy": "Interceptability",
        "SafetyGatePolicy": "SafetyGate"
    }
    return mapping.get(policy_name, policy_name)

def format_reason_label(reason: str) -> str:
    mapping = {
        "engage_conditions_satisfied": "Engage OK",
        "identification_confidence_too_low": "Low ID",
        "high_environmental_risk": "High Risk",
        "not_detected": "Not Detected",
        "invalid_estimate": "Invalid Estimate",
        "invalid_prediction": "Invalid Prediction",
        "tracking_confidence_too_low_abort": "Low Track Abort",
        "tracking_confidence_too_low_track": "Low Track",
        "not_interceptable": "Not Interceptable"
    }
    return mapping.get(reason, reason)

def load_result_files(input_dir: str) -> Dict[str, Any]:
    return {
        "policy_summaries": load_json_file(os.path.join(input_dir, "policy_summaries.json")),
        "grouped_summaries": load_json_file(os.path.join(input_dir, "grouped_summaries.json")),
        "records": load_json_file(os.path.join(input_dir, "records.json"))
    }

def build_policy_summary_table(policy_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    table = []
    for s in policy_summaries:
        table.append({
            "policy_name": s.get("policy_name"),
            "total_decisions": s.get("total_decisions", 0),
            "engage_count": s.get("engage_count", 0),
            "track_count": s.get("track_count", 0),
            "abort_count": s.get("abort_count", 0),
            "engagement_attempt_rate": s.get("engagement_attempt_rate", 0.0),
            "unsafe_engagement_count": s.get("unsafe_engagement_count", 0),
            "false_engagement_count": s.get("false_engagement_count", 0),
            "non_interceptable_engagement_count": s.get("non_interceptable_engagement_count", 0),
            "invalid_engagement_count": s.get("invalid_engagement_count", 0),
            "safe_engagement_count": s.get("safe_engagement_count", 0),
            "safe_engagement_per_attempt_rate": s.get("safe_engagement_per_attempt_rate", 0.0),
            "unsafe_engagement_per_attempt_rate": s.get("unsafe_engagement_per_attempt_rate", 0.0),
            "false_engagement_per_attempt_rate": s.get("false_engagement_per_attempt_rate", 0.0),
            "non_interceptable_engagement_per_attempt_rate": s.get("non_interceptable_engagement_per_attempt_rate", 0.0),
            # M4: composite-score and abort-timing metrics
            "safe_opportunity_count": s.get("safe_opportunity_count", 0),
            "missed_safe_opportunity_count": s.get("missed_safe_opportunity_count", 0),
            "safe_opportunity_capture_rate": s.get("safe_opportunity_capture_rate", 0.0),
            "abort_required_trial_count": s.get("abort_required_trial_count", 0),
            "on_time_abort_trial_count": s.get("on_time_abort_trial_count", 0),
            "late_abort_trial_count": s.get("late_abort_trial_count", 0),
            "missed_abort_trial_count": s.get("missed_abort_trial_count", 0),
            "abort_success_rate": s.get("abort_success_rate", 0.0),
            "late_abort_rate": s.get("late_abort_rate", 0.0),
            "missed_abort_rate": s.get("missed_abort_rate", 0.0),
            "risk_weighted_score": s.get("risk_weighted_score", 0.0),
        })
    return table

def build_grouped_summary_table(grouped_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    table = []
    for s in grouped_summaries:
        table.append({
            "scenario_type": s.get("scenario_type"),
            "policy_name": s.get("policy_name"),
            "total_decisions": s.get("total_decisions", 0),
            "engage_count": s.get("engage_count", 0),
            "track_count": s.get("track_count", 0),
            "abort_count": s.get("abort_count", 0),
            "engagement_attempt_rate": s.get("engagement_attempt_rate", 0.0),
            "unsafe_engagement_count": s.get("unsafe_engagement_count", 0),
            "false_engagement_count": s.get("false_engagement_count", 0),
            "non_interceptable_engagement_count": s.get("non_interceptable_engagement_count", 0),
            "invalid_engagement_count": s.get("invalid_engagement_count", 0),
            "safe_engagement_per_attempt_rate": s.get("safe_engagement_per_attempt_rate", 0.0),
            "reason_counts": s.get("reason_counts", {})
        })
    return table

def extract_safety_gate_reasons(grouped_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for s in grouped_summaries:
        if s.get("policy_name") == "SafetyGatePolicy":
            scenario_type = s.get("scenario_type")
            reason_counts = s.get("reason_counts", {})
            if isinstance(reason_counts, str):
                reason_counts = json.loads(reason_counts)
            
            for reason, count in reason_counts.items():
                results.append({
                    "scenario_type": scenario_type,
                    "reason": reason,
                    "count": count
                })
    return results

def write_csv_rows(rows: List[Dict[str, Any]], file_path: str) -> None:
    if not rows:
        with open(file_path, "w", encoding="utf-8") as f:
            pass
        return
        
    keys_set = set()
    for row in rows:
        keys_set.update(row.keys())
        
    fieldnames = sorted(list(keys_set))
    
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            serialized_row = {}
            for k, v in row.items():
                if isinstance(v, (dict, list, tuple)):
                    serialized_row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    serialized_row[k] = v
            writer.writerow(serialized_row)

def write_markdown_report(policy_table: List[Dict[str, Any]], grouped_table: List[Dict[str, Any]], safety_gate_reasons: List[Dict[str, Any]], file_path: str) -> None:
    num_policies = len({row.get("policy_name") for row in policy_table if row.get("policy_name")})
    num_groups = len(grouped_table)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# Safety-Gated C-UAS Simulation Pilot Report\n\n")
        f.write("## Experiment Output Summary\n")
        f.write(f"- Total Policies Analyzed: {num_policies}\n")
        f.write(f"- Scenario-Policy Groups: {num_groups}\n\n")
        
        f.write("## Policy-Level Summary\n")
        f.write("| Policy | Decisions | Engage | Track | Abort | Unsafe | False | Non-Int | Safe/Attempt |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for row in policy_table:
            p = row.get("policy_name", "All")
            td = row.get("total_decisions", 0)
            eg = row.get("engage_count", 0)
            tr = row.get("track_count", 0)
            ab = row.get("abort_count", 0)
            ue = row.get("unsafe_engagement_count", 0)
            fe = row.get("false_engagement_count", 0)
            ne = row.get("non_interceptable_engagement_count", 0)
            sr = row.get("safe_engagement_per_attempt_rate", 0.0)
            f.write(f"| {p} | {td} | {eg} | {tr} | {ab} | {ue} | {fe} | {ne} | {sr:.3f} |\n")

        # M4: composite-score and abort-timing table
        f.write("\n## M4 Composite Metrics (safety_oriented)\n")
        f.write("| Policy | SafeOppCapture | AbortSuccess | LateAbort | MissedAbort | RiskWeightedScore |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in policy_table:
            p = row.get("policy_name", "All")
            soc = row.get("safe_opportunity_capture_rate", 0.0)
            asr = row.get("abort_success_rate", 0.0)
            lar = row.get("late_abort_rate", 0.0)
            mar = row.get("missed_abort_rate", 0.0)
            rws = row.get("risk_weighted_score", 0.0)
            f.write(f"| {p} | {soc:.3f} | {asr:.3f} | {lar:.3f} | {mar:.3f} | {rws:+.3f} |\n")
        f.write("\n")
        
        f.write("\n## Scenario-Level SafetyGate Reasons\n")
        
        sg_by_scenario = defaultdict(list)
        for sg in safety_gate_reasons:
            sg_by_scenario[sg.get("scenario_type")].append((sg.get("reason"), sg.get("count", 0)))
            
        for scenario, reasons in sg_by_scenario.items():
            f.write(f"### Scenario: {scenario}\n")
            for reason, count in reasons:
                f.write(f"- {reason}: {count}\n")
            f.write("\n")
            
        f.write("## Key Interpretation Notes\n")
        f.write("- **SafetyGatePolicy** is designed to reduce unsafe, false, and non-interceptable engagements by overriding decisions when thresholds are violated.\n")
        f.write("- **S3**: Low identification confidence scenario.\n")
        f.write("- **S4**: Environmental risk proximity scenario.\n")
        f.write("- **S5**: Tracking loss scenario.\n")
        f.write("- **S6**: Non-interceptable target scenario.\n")
        f.write("- *Note*: This report is for a 2D kinematic simulation framework evaluation and does not represent real hardware control results.\n\n")
        f.write("## Generated Figures\n")
        f.write("- policy_action_counts.png\n")
        f.write("- policy_risk_counts.png\n")
        f.write("- safety_gate_reasons.png\n")

def plot_policy_risk_counts(policy_table: List[Dict[str, Any]], output_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib or numpy not installed. Skipping plot_policy_risk_counts.")
        return False

    policies = []
    unsafe = []
    false_eng = []
    non_int = []
    
    for row in policy_table:
        if row.get("policy_name"):
            policies.append(format_policy_label(row["policy_name"]))
            unsafe.append(row.get("unsafe_engagement_count", 0))
            false_eng.append(row.get("false_engagement_count", 0))
            non_int.append(row.get("non_interceptable_engagement_count", 0))
            
    if not policies:
        print("Warning: No policies data to plot in plot_policy_risk_counts.")
        return False
            
    x = np.arange(len(policies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width, unsafe, width, label='Unsafe')
    bar2 = ax.bar(x, false_eng, width, label='False')
    bar3 = ax.bar(x + width, non_int, width, label='Non-Interceptable')

    ax.set_ylabel('Count')
    ax.set_title('Policy Risk Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha='right')
    ax.legend()
    
    if hasattr(ax, 'bar_label'):
        ax.bar_label(bar1, padding=3, fontsize=8)
        ax.bar_label(bar2, padding=3, fontsize=8)
        ax.bar_label(bar3, padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True

def plot_policy_action_counts(policy_table: List[Dict[str, Any]], output_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib or numpy not installed. Skipping plot_policy_action_counts.")
        return False

    policies = []
    engage = []
    track = []
    abort = []
    
    for row in policy_table:
        if row.get("policy_name"):
            policies.append(format_policy_label(row["policy_name"]))
            engage.append(row.get("engage_count", 0))
            track.append(row.get("track_count", 0))
            abort.append(row.get("abort_count", 0))
            
    if not policies:
        print("Warning: No policies data to plot in plot_policy_action_counts.")
        return False
            
    x = np.arange(len(policies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width, engage, width, label='Engage')
    bar2 = ax.bar(x, track, width, label='Track')
    bar3 = ax.bar(x + width, abort, width, label='Abort')

    ax.set_ylabel('Count')
    ax.set_title('Policy Action Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha='right')
    ax.legend()

    if hasattr(ax, 'bar_label'):
        ax.bar_label(bar1, padding=3, fontsize=8)
        ax.bar_label(bar2, padding=3, fontsize=8)
        ax.bar_label(bar3, padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True

def plot_safety_gate_reasons(safety_gate_reasons: List[Dict[str, Any]], output_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plot_safety_gate_reasons.")
        return False

    reason_totals = defaultdict(int)
    for sg in safety_gate_reasons:
        reason_totals[format_reason_label(sg.get("reason"))] += sg.get("count", 0)
        
    if not reason_totals:
        print("Warning: No safety gate reasons to plot.")
        return False

    reasons = list(reason_totals.keys())
    counts = [reason_totals[r] for r in reasons]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(reasons, counts)
    ax.set_ylabel('Count')
    ax.set_title('SafetyGate Reasons Counts')
    plt.xticks(rotation=45, ha='right')
    
    if hasattr(ax, 'bar_label'):
        ax.bar_label(bar1, padding=3, fontsize=8)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True

def analyze_results(input_dir: str, output_dir: Optional[str] = None, make_plots: bool = True, top_reasons: int = 10) -> Dict[str, str]:
    if output_dir is None:
        output_dir = os.path.join(input_dir, "analysis")
        
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_result_files(input_dir)
    
    policy_table = build_policy_summary_table(data["policy_summaries"])
    grouped_table = build_grouped_summary_table(data["grouped_summaries"])
    sg_reasons = extract_safety_gate_reasons(data["grouped_summaries"])
    
    paths = {
        "policy_summary_table_csv": os.path.join(output_dir, "policy_summary_table.csv"),
        "grouped_summary_table_csv": os.path.join(output_dir, "grouped_summary_table.csv"),
        "safety_gate_reasons_csv": os.path.join(output_dir, "safety_gate_reasons.csv"),
        "report_md": os.path.join(output_dir, "report.md")
    }
    
    write_csv_rows(policy_table, paths["policy_summary_table_csv"])
    write_csv_rows(grouped_table, paths["grouped_summary_table_csv"])
    write_csv_rows(sg_reasons, paths["safety_gate_reasons_csv"])
    write_markdown_report(policy_table, grouped_table, sg_reasons, paths["report_md"])
    
    if make_plots:
        risk_counts_path = os.path.join(output_dir, "policy_risk_counts.png")
        action_counts_path = os.path.join(output_dir, "policy_action_counts.png")
        sg_reasons_path = os.path.join(output_dir, "safety_gate_reasons.png")
        
        if plot_policy_risk_counts(policy_table, risk_counts_path):
            paths["policy_risk_counts_png"] = risk_counts_path
            
        if plot_policy_action_counts(policy_table, action_counts_path):
            paths["policy_action_counts_png"] = action_counts_path
            
        if plot_safety_gate_reasons(sg_reasons, sg_reasons_path):
            paths["safety_gate_reasons_png"] = sg_reasons_path
        
    return paths

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze CUAS Monte Carlo Results")
    parser.add_argument("--input-dir", required=True, type=str, help="Input directory containing JSON files")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for analysis (default: input-dir/analysis)")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    parser.add_argument("--top-reasons", type=int, default=10, help="Top reasons limit (currently not strictly filtering)")
    
    try:
        args = parser.parse_args(argv)
        make_plots = not args.no_plots
        
        paths = analyze_results(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            make_plots=make_plots,
            top_reasons=args.top_reasons
        )
        
        print("\nAnalysis completed successfully.")
        print("Generated files:")
        for k, v in paths.items():
            print(f"  {k}: {v}")
            
        return 0
    except Exception as e:
        print(f"Error running analysis: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
