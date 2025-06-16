import requests
import json
from datetime import datetime

# Replace with your actual Logtail Write Key from your HTTP JSON source
LOGTAIL_WRITE_KEY = "JNoyTXiogFjy3x9kTV9qVSMt"

# Logtail ingestion endpoint
LOGTAIL_URL = "https://s1350149.eu-nbg-2.betterstackdata.com/"

# Prepare a sample log entry (dict)
sample_log = {
    "experiment_id": "test_exp_001",
    "episode": 1,
    "agent_id": "agent_test",
    "position": [1.23, 4.56, 7.89],
    "reward": 10.0,
    "velocity": [0.1, 0.2, 0.3],
    "timestamp": datetime.utcnow().isoformat() + "Z"
}

# Set headers including the authorization with write key
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {LOGTAIL_WRITE_KEY}"
}

# Send POST request with JSON payload
response = requests.post(
    LOGTAIL_URL,
    headers=headers,
    data=json.dumps(sample_log)
)

# Print response status for debugging
print("Status code:", response.status_code)
print("Response body:", response.text)
