import json
import time
import os
from typing import Dict, Any, List
from datetime import datetime
import requests


class UnifiedLogger:
    def __init__(self, config_path="logging_config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.enabled = self.config.get("enabled", True)

        # Per-step logging
        self.step_output = self.config.get("step_output", "step_log.json")
        self.log_fields = self.config.get("log_fields", [])
        self.log_every_n = self.config.get("log_every_n_seconds", 1)
        self.last_log_time = 0
        self.step_buffer = self._load_file(self.step_output)

        # Summary logging
        self.summary_output = self.config.get("summary_output", "experiment_log.json")
        self.summary_buffer = self._load_file(self.summary_output)

        # Remote logging
        self.remote_url = self.config.get("remote_url")
        self.source_token = self.config.get("source_token")

    def _load_file(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _write_file(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _remote_post(self, payload):
        if self.remote_url:
            try:
                headers = {
                    "Content-Type": "application/json"
                }
                if self.source_token:
                    headers["Authorization"] = f"Bearer {self.source_token}"

                response = requests.post(self.remote_url, headers=headers, json=payload, timeout=2)
                response.raise_for_status()
                print(f"[Logger] Remote log sent.")
            except requests.RequestException as e:
                print(f"⚠️ Remote logging failed: {e}")

    def log_step(self, data: Dict[str, Any], force=False):
        if not self.enabled:
            return

        now = time.time()
        if not force and now - self.last_log_time < self.log_every_n:
            return

        self.last_log_time = now
        log_entry = {k: v for k, v in data.items() if not self.log_fields or k in self.log_fields}
        log_entry["timestamp"] = datetime.utcnow().isoformat() + "Z"

        self.step_buffer.append(log_entry)
        self._write_file(self.step_output, self.step_buffer)

        if self.config.get("remote_log_steps", False):
            self._remote_post(log_entry)

    def log_experiment_summary(
        self,
        experiment_number: int,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        snapshots: List[Dict[str, Any]]
    ):
        if not self.enabled:
            return

        summary_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "experiment_number": experiment_number,
            "parameters": parameters,
            "metrics": metrics,
            "snapshots": snapshots
        }

        self.summary_buffer.append(summary_entry)
        self._write_file(self.summary_output, self.summary_buffer)

        if self.config.get("remote_log_summary", True):
            self._remote_post(summary_entry)

        print(f"[Logger] Logged experiment {experiment_number} to {self.summary_output}")
