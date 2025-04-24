import requests
import json
import time
from parse_player import fetch_game_state

GAME_SERVER_URL = "http://127.0.0.1:15702/"
POLL_INTERVAL = 1  # Time in seconds between requests


def update_ai_component():
    # Step 1: Fetch player state from Bevy
    game_state = fetch_game_state()
    if game_state is None:
        raise RuntimeError("Could not fetch player state")

    player = game_state.players[1]  # or whatever index you need
    entity_id = player["entity"]

    # Step 3: Construct JSON-RPC insert request
    #4294967365 [entity_id]
    insert_request = {
        "id": 3, "jsonrpc": "2.0", "method": "bevy/insert",
        "params": {
            "entity": entity_id,
            "components": {
                "hotline_miami_like::player::movement::Movement": {
                    "norm_direction": [0.7, 0.7],
                    "speed": 50.0
                },
            }
        }
    }

    # insert_request = {
    #     "id": 3, "jsonrpc": "2.0", "method": "bevy/get",
    #     "params": {
    #         "entity": entity_id,
    #         "components": [
    #             "hotline_miami_like::player::movement::Movement"
    #         ]
    #     }
    # }

    #"glam::Vec2":
    # insert_request['params']['components'] = list(insert_request['params']['components'])

    # Now it should work
    #print(json.dumps(insert_request, indent=2))

    resp = requests.post(
        GAME_SERVER_URL, json=insert_request, timeout=1.0)
    resp.raise_for_status()
    response_data = resp.json()
    print(response_data)

if __name__ == "__main__":
    update_ai_component()