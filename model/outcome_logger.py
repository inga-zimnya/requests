import json
import time
import os
import requests
from typing import Dict, Any


class OutcomeLogger:
    def __init__(self, config_path="logging_config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.enabled = self.config.get("enabled", True)
        self.remote_url = self.config.get("remote_url")
        self.source_token = self.config.get("source_token")
        self.local_output = self.config.get("local_output", "outcome.json")
        self.log_fields = self.config.get("log_fields", [])
        self.log_every_n = self.config.get("log_every_n_seconds", 1)

        self.last_log_time = 0
        self.local_buffer = []

        if os.path.exists(self.local_output):
            with open(self.local_output, "r") as f:
                try:
                    self.local_buffer = json.load(f)
                except json.JSONDecodeError:
                    self.local_buffer = []

    def log(self, data: Dict[str, Any]):
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_log_time < self.log_every_n:
            return  # throttle logging rate

        self.last_log_time = now
        log_entry = {k: v for k, v in data.items() if not self.log_fields or k in self.log_fields}
        log_entry["timestamp"] = time.time()

        # Local log
        self.local_buffer.append(log_entry)
        with open(self.local_output, "w") as f:
            json.dump(self.local_buffer, f, indent=2)

        # Remote log to Logtail
        if self.remote_url and self.source_token:
            try:
                headers = {
                    "Authorization": f"Bearer {self.source_token}",
                    "Content-Type": "application/json"
                }
                requests.post(self.remote_url, headers=headers, json=log_entry, timeout=2)
            except requests.RequestException as e:
                print(f"⚠️ Remote logging failed: {e}")
