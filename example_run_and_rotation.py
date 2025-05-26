import requests
import json
import time
import math
from main import fetch_game_state

GAME_SERVER_URL = "http://127.0.0.1:15702/"
POLL_INTERVAL = 1  # Time in seconds between requests
MAX_RETRIES = 5  # Maximum number of retries for failed requests
RETRY_DELAY = 0.5  # Time to wait between retries in seconds


def update_ai_component():
    """Update movement and rotation of the player in a loop with error handling."""
    retry_count = 0

    while True:
        try:
            # Step 1: Fetch player state from Bevy
            game_state = fetch_game_state()

            if game_state is None:
                print("⚠️ Warning: Could not fetch player state, retrying...")
                time.sleep(POLL_INTERVAL)
                continue

            # Check if second player exists
            if len(game_state.players) < 2:
                print("⚠️ Warning: Not enough players in game state")
                time.sleep(POLL_INTERVAL)
                continue

            player = game_state.players[1]
            print("📦 Player state:", player)
            entity_id = player["entity"]

            # Step 2: Define movement direction
            direction = [0.0, -1.0]  # Moving down
            speed = 50.0

            # Step 3: Calculate rotation from direction (angle in radians)
            dx, dy = direction
            angle_rad = math.atan2(dy, dx)  # Will be -pi/2 for [0, -1]

            quat = {
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(angle_rad / 2),
                "w": math.cos(angle_rad / 2),
                "type": "Quat"
            }

            # Step 4: Construct JSON-RPC request
            insert_request = {
                "id": 3,
                "jsonrpc": "2.0",
                "method": "bevy/insert",
                "params": {
                    "entity": entity_id,
                    "components": {
                        "hotline_miami_like::player::movement::Movement": {
                            "norm_direction": direction,
                            "speed": speed
                        },
                        "Transform": {
                            "rotation": quat
                        }
                    }
                }
            }

            # Step 5: Send request
            resp = requests.post(
                GAME_SERVER_URL,
                json=insert_request,
                timeout=1.0
            )
            resp.raise_for_status()

            retry_count = 0  # Reset retry counter
            response_data = resp.json()
            print(f"✅ Server response: {response_data}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {str(e)}")
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                print("🚨 Max retries reached, exiting...")
                break
            time.sleep(RETRY_DELAY)
            continue

        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            time.sleep(POLL_INTERVAL)
            continue

        # Step 6: Wait before next iteration
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    update_ai_component()
