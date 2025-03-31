import requests
import json
import time
from typing import Dict, List, TypedDict, Literal, Tuple  # Added Tuple here
from dataclasses import dataclass

# Type definitions matching your Rust GameStateEncoding
GameStateEncoding = Literal[
    'Empty', 'Floor', 'Wall', 'Glass',
    'Crate', 'Pickup', 'Bullet', 'Characters'
]


class GameStateLayer(TypedDict):
    layer_0: List[List[GameStateEncoding]]  # Floor/empty
    layer_1: List[List[GameStateEncoding]]  # Walls
    layer_2: List[List[GameStateEncoding]]  # Dynamic entities


class GameStateResponse(TypedDict):
    state: GameStateLayer
    ai_state: List[bool]


@dataclass
class ParsedGameState:
    """Structured representation of the game state"""
    floors: List[List[GameStateEncoding]]
    walls: List[List[GameStateEncoding]]
    entities: List[List[GameStateEncoding]]
    ai_states: Dict[int, bool]

    @property
    def player_positions(self) -> Dict[str, List[Tuple[int, int]]]:  # Changed return type
        """Returns a dictionary of (x,y) positions for each player"""
        player_positions: Dict[str, List[Tuple[int, int]]] = {}  # Initialize an empty dictionary
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell.startswith('Character'):  # Assuming cells are named "Player1", "Player2", etc.
                    if cell not in player_positions:
                        player_positions[cell] = []  # Initialize a list for that player if it is not present in the dictionary.
                    player_positions[cell].append((x, y))
        return player_positions

    @property
    def pickup_positions(self) -> List[Tuple[int, int]]:
        """Returns (x,y) positions of all pickups"""
        positions = []
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell == 'Pickup':
                    positions.append((x, y))
        return positions


def fetch_game_state() -> ParsedGameState:
    """Fetches and parses the game state from the Bevy server"""
    payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "bevy/query",
        "params": {
            "data": {
                "components": ["hotline_miami_like::ai::game_state::GameState"],
                "has": [],
                "option": []
            },
            "filter": {
                "with": [],
                "without": []
            }
        }
    }

    try:
        response = requests.post("http://127.0.0.1:15702/", json=payload)
        response.raise_for_status()
        data = response.json()

        if "result" not in data or not data["result"]:
            raise ValueError("No game state in response")

        raw_state = data["result"][0]['components']['hotline_miami_like::ai::game_state::GameState']

        # Convert to structured format
        return ParsedGameState(
            floors=raw_state["state"][0],
            walls=raw_state["state"][1],
            entities=raw_state["state"][2],
            ai_states={
                idx: state
                for idx, state in enumerate(raw_state["ai_state"])
                if state  # Only include active states
            }
        )

    except Exception as e:
        print(f"Error fetching game state: {e}")
        raise


# Example usage
if __name__ == "__main__":
    while True:
        try:
            game_state = fetch_game_state()

            # Print some debug info
            print(f"Player positions: {game_state.player_positions}")
            print(f"Pickup positions: {game_state.pickup_positions}")
            print(f"Active AI states: {game_state.ai_states}")

        except KeyboardInterrupt:
            print("Stopping game state monitor")
            break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(1)  # Polling interval