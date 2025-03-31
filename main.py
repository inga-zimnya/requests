import requests
import json
import time
from typing import Dict, List, TypedDict, Literal, Tuple
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
    def player_positions(self) -> List[Tuple[int, int]]:
        """Returns (x,y) positions of all characters"""
        positions = []
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell == 'Characters':
                    positions.append((x, y))
        return positions

    @property
    def pickup_positions(self) -> List[Tuple[int, int]]:
        """Returns (x,y) positions of all pickups"""
        positions = []
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell == 'Pickup':
                    positions.append((x, y))
        return positions

    @property
    def bullet_positions(self) -> List[Tuple[int, int]]:
        """Returns (x,y) positions of all bullets"""
        positions = []
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell == 'Bullet':
                    positions.append((x, y))
        return positions

    @property
    def crate_positions(self) -> List[Tuple[int, int]]:
        """Returns (x,y) positions of all crates"""
        positions = []
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell == 'Crate':
                    positions.append((x, y))
        return positions

    @property
    def glass_positions(self) -> List[Tuple[int, int]]:
        """Returns (x,y) positions of all glass (from walls layer)"""
        positions = []
        for y, row in enumerate(self.walls):
            for x, cell in enumerate(row):
                if cell == 'Glass':
                    positions.append((x, y))
        return positions

    @property
    def wall_positions(self) -> List[Tuple[int, int]]:
        """Returns (x,y) positions of all walls (from walls layer)"""
        positions = []
        for y, row in enumerate(self.walls):
            for x, cell in enumerate(row):
                if cell == 'Wall':
                    positions.append((x, y))
        return positions

    @property
    def empty_positions(self) -> List[Tuple[int, int]]:
        """Returns (x,y) positions of all empty floor tiles (from floor layer)"""
        positions = []
        for y, row in enumerate(self.floors):
            for x, cell in enumerate(row):
                if cell == 'Empty':
                    positions.append((x, y))
        return positions

    @property
    def floor_positions(self) -> List[Tuple[int, int]]:
        """Returns (x,y) positions of all floor tiles (from floor layer)"""
        positions = []
        for y, row in enumerate(self.floors):
            for x, cell in enumerate(row):
                if cell == 'Floor':
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


def show_ascii_map(self, width: int = 20, height: int = 10) -> str:
    """Generate an ASCII representation of the game map"""
    symbols = {
        'Empty': ' ',
        'Floor': '.',
        'Wall': '#',
        'Glass': '░',
        'Crate': '■',
        'Pickup': 'P',
        'Bullet': '•',
        'Characters': '☻'
    }

    map_str = ""
    for y in range(min(height, len(self.entities))):
        for x in range(min(width, len(self.entities[0]))):
            # Check entities first, then walls, then floors
            if self.entities[y][x] != 'Empty':
                map_str += symbols.get(self.entities[y][x], '?')
            elif self.walls[y][x] != 'Empty':
                map_str += symbols.get(self.walls[y][x], '?')
            else:
                map_str += symbols.get(self.floors[y][x], '?')
        map_str += "\n"
    return map_str

# Example usage
if __name__ == "__main__":
    while True:
        try:
            game_state = fetch_game_state()

            # Print debug info for all components
            print("\n=== Game State ===")
            print(f"Players: {len(game_state.player_positions)} at {game_state.player_positions}")
            print(f"Pickups: {len(game_state.pickup_positions)} at {game_state.pickup_positions}")
            print(f"Bullets: {len(game_state.bullet_positions)} at {game_state.bullet_positions}")
            print(f"Crates: {len(game_state.crate_positions)} at {game_state.crate_positions}")
            print(f"Glass walls: {len(game_state.glass_positions)} at {game_state.glass_positions}")
            print(f"Solid walls: {len(game_state.wall_positions)} at {game_state.wall_positions}")
            print(f"Empty tiles: {len(game_state.empty_positions)}")
            print(f"Floor tiles: {len(game_state.floor_positions)}")
            print(f"Active AI states: {game_state.ai_states}")

        except KeyboardInterrupt:
            print("Stopping game state monitor")
            break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(1)  # Polling interval
