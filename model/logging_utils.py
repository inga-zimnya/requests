import json
import os
from datetime import datetime
import requests


def log_experiment_outcome(
    experiment_number,
    params,
    metrics,
    simulation_snapshots,
    dashboard_url=None,
    outcome_path="outcome.json"
):
    """
    Logs the experiment outcome to a JSON file and optionally sends it to a dashboard.

    Args:
        experiment_number (int): Unique identifier for the experiment.
        params (dict): Input parameters used in the experiment.
        metrics (dict): Summary metrics like reward, success, loss, etc.
        simulation_snapshots (list of dict): Intermediate snapshots (e.g. position, rotation, timestamp).
        dashboard_url (str, optional): If provided, the full log will be POSTed here.
        outcome_path (str): Path to local JSON file to append the result.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "experiment_number": experiment_number,
        "parameters": params,
        "metrics": metrics,
        "snapshots": simulation_snapshots
    }

    # Append to local outcome.json
    if os.path.exists(outcome_path):
        with open(outcome_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(log_entry)

    with open(outcome_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[Logger] Logged experiment {experiment_number} to {outcome_path}")

    # Optionally send to dashboard
    if dashboard_url:
        try:
            response = requests.post(dashboard_url, json=log_entry)
            response.raise_for_status()
            print(f"[Logger] Sent log to dashboard: {dashboard_url}")
        except Exception as e:
            print(f"[Logger] Failed to send to dashboard: {e}")
