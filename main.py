import requests
import json
import time

URL = "http://127.0.0.1:15702/"
REQUEST_PAYLOAD = json.dumps({
    "id": 1,
    "jsonrpc": "2.0",
    "method": "bevy/query",
    "params": {
        "data": {
            "components": ["bevy_transform::components::transform::Transform"],
            "has": [],
            "option": []
        },
        "filter": {
            "with": [],
            "without": []
        }
    }
})

POLL_INTERVAL = 1  # Time in seconds between requests

def fetch_game_data():
    """Fetches and prints structured game data in a loop."""
    while True:
        try:
            response = requests.post(url=URL, data=REQUEST_PAYLOAD)
            response.raise_for_status()  # Check for HTTP errors
            data = response.json()

            # Format the output
            print("\n=== Game Data ===")
            if "result" in data:
                for idx, item in enumerate(data["result"], start=1):
                    print(f"Entity {idx}: {json.dumps(item, indent=2)}")
            else:
                print("No result found:", data)

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
        except json.JSONDecodeError:
            print("Error decoding JSON response.")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    fetch_game_data()
