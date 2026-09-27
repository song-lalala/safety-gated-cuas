import os
import csv
import json
import dataclasses
import enum
from typing import List, Dict

from .runner import MonteCarloResult

def _to_jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "value"):
        return value.value
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)

def _serialize_value(val: object) -> str:
    safe_val = _to_jsonable(val)
    if safe_val is None:
        return ""
    if isinstance(safe_val, (dict, list)):
        return json.dumps(safe_val, ensure_ascii=False)
    if isinstance(safe_val, (bool, int, float, str)):
        return safe_val
    return str(safe_val)

def write_dict_rows_csv(rows: List[Dict[str, object]], file_path: str) -> None:
    if not rows:
        # Create empty file
        with open(file_path, "w", encoding="utf-8") as f:
            pass
        return
        
    # Get all unique keys across all rows
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
                serialized_row[k] = _serialize_value(v)
            writer.writerow(serialized_row)

def write_json(data: object, file_path: str) -> None:
    safe_data = _to_jsonable(data)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=2, ensure_ascii=False)

def export_monte_carlo_result(result: MonteCarloResult, output_dir: str) -> Dict[str, str]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    paths = {
        "records_csv": os.path.join(output_dir, "records.csv"),
        "policy_summaries_csv": os.path.join(output_dir, "policy_summaries.csv"),
        "grouped_summaries_csv": os.path.join(output_dir, "grouped_summaries.csv"),
        "records_json": os.path.join(output_dir, "records.json"),
        "policy_summaries_json": os.path.join(output_dir, "policy_summaries.json"),
        "grouped_summaries_json": os.path.join(output_dir, "grouped_summaries.json"),
        # M7: per-trial dump used for bootstrap CIs and paired tests.
        "per_trial_summaries_csv": os.path.join(output_dir, "per_trial_summaries.csv"),
        "per_trial_summaries_json": os.path.join(output_dir, "per_trial_summaries.json"),
    }

    records_dicts = result.records_to_dicts()
    policy_summaries_dicts = result.summaries_to_dicts()
    grouped_summaries_dicts = result.grouped_summaries_to_dicts()
    per_trial_dicts = result.per_trial_summaries_to_dicts()

    write_dict_rows_csv(records_dicts, paths["records_csv"])
    write_dict_rows_csv(policy_summaries_dicts, paths["policy_summaries_csv"])
    write_dict_rows_csv(grouped_summaries_dicts, paths["grouped_summaries_csv"])
    write_dict_rows_csv(per_trial_dicts, paths["per_trial_summaries_csv"])

    write_json(records_dicts, paths["records_json"])
    write_json(policy_summaries_dicts, paths["policy_summaries_json"])
    write_json(grouped_summaries_dicts, paths["grouped_summaries_json"])
    write_json(per_trial_dicts, paths["per_trial_summaries_json"])

    return paths
