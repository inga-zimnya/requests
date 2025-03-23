import requests
import json
import time

URL = "http://127.0.0.1:15702/"
REQUEST_PAYLOAD = json.dumps({
   "id": 10,
   "jsonrpc": "2.0",
   "method": "bevy/query",
   "params": {
       "data": {
           "components": ["hotline_miami_like::ai::game_state::GameState"],
           "has": [],  # Full module path!
           "option": []
       },
       "filter": {
           "with": [],  # Remove empty string
           "without": []
       }
   }
})

POLL_INTERVAL = 1  # Time in seconds between requests

def parse_game_state(json_data):
    """
    Parses the game state information from a JSON string.

    Args:
        json_data: A JSON string representing the game state.  This should
            correspond to the `GameState` struct in the Rust code.  It's
            assumed to be a serialized representation of:
            {
                "state": [[[<GameStateEncoding>; MAX_MAP_PIXELS_SIZE.1]; MAX_MAP_PIXELS_SIZE.0]; TOTAL_LAYERS],
                "ai_state": [bool; 10],
            }
            Where:
                - <GameStateEncoding> is a string representing the enum variant
                  (e.g., "Empty", "Floor", "Wall", etc.).
                - MAX_MAP_PIXELS_SIZE is (160, 112)
                - TOTAL_LAYERS is 3

    Returns:
        A dictionary containing the parsed game state:
        {
            "state": [[[str; 112]; 160]; 3],  # 3D list of GameStateEncoding strings
            "ai_state": [bool; 10]            # List of boolean values
        }
        Returns None if parsing fails.
    """
    try:
        data = json.loads(json_data)

        # Validate data structure (very important for type safety)
        if not isinstance(data, dict):
            print("Error: Game state data is not a dictionary.")
            return None

        if "state" not in data or "ai_state" not in data:
            print("Error: 'state' or 'ai_state' key missing from game state data.")
            return None

        state_data = data["state"]
        ai_state_data = data["ai_state"]

        # Validate the 'state' field
        if not isinstance(state_data, list):
            print("Error: 'state' is not a list.")
            return None

        if len(state_data) != 3:  # TOTAL_LAYERS
            print(f"Error: 'state' should have 3 layers, but has {len(state_data)}.")
            return None

        for layer in state_data:
            if not isinstance(layer, list):
                print("Error: A layer within 'state' is not a list.")
                return None
            if len(layer) != 160:  # MAX_MAP_PIXELS_SIZE.0
                print(f"Error: Layer should have 160 rows, but has {len(layer)}.")
                return None
            for row in layer:
                if not isinstance(row, list):
                    print("Error: A row within a layer is not a list.")
                    return None
                if len(row) != 112:  # MAX_MAP_PIXELS_SIZE.1
                    print(f"Error: Row should have 112 columns, but has {len(row)}.")
                    return None
                for cell in row:
                    if not isinstance(cell, str):
                        print("Error: A cell within a row is not a string (GameStateEncoding).")
                        return None

        # Validate the 'ai_state' field
        if not isinstance(ai_state_data, list):
            print("Error: 'ai_state' is not a list.")
            return None

        if len(ai_state_data) != 10:
            print(f"Error: 'ai_state' should have 10 elements, but has {len(ai_state_data)}.")
            return None

        for val in ai_state_data:
            if not isinstance(val, bool):
                print("Error: An element in 'ai_state' is not a boolean.")
                return None


        return {
            "state": state_data,
            "ai_state": ai_state_data
        }

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def fetch_game_data():
    """Fetches and parses game data in a loop."""
    while True:
        try:
            response = requests.post(url=URL, data=REQUEST_PAYLOAD)
            response.raise_for_status()  # Check for HTTP errors
            data = response.json()

            # Assuming the game state is under a specific key in the response, e.g., "game_state"
            if "game_state" in data:
                game_state_json = json.dumps(data["game_state"])  # Convert to JSON string
                parsed_game_state = parse_game_state(game_state_json)

                if parsed_game_state:
                    print("Successfully parsed game state:")
                    # Do something with the parsed game state, e.g.,
                    # Access specific elements, print the state, etc.
                    # print(parsed_game_state["state"][0][0][0])  # Access a specific game state encoding
                    print(f"AI State: {parsed_game_state['ai_state']}")
                else:
                    print("Failed to parse game state.")

            else:
                print("No 'game_state' found in the response:", data)

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
        except json.JSONDecodeError:
            print("Error decoding JSON response from the server.") # Server Response

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    fetch_game_data()

