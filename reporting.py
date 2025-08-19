# reporting.py
import os
import json

def report_exists(report_folder: str, filename: str) -> bool:
    """
    Check if a report file exists in report_folder.
    """
    return os.path.isfile(os.path.join(report_folder, filename))

def save_workflow_log(path: str, workflow):
    """
    Save a workflow log (Python object/list/dict) as JSON to `path`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2)

def load_workflow_log(path: str):
    """
    Load and return workflow JSON from `path`. Returns None on error.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
