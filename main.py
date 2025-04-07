import requests
import json
import time
import math
from typing import Dict, List, TypedDict, Literal, Tuple
from dataclasses import dataclass

# Game Constants (from game_state.rs)
ONE_PIXEL_DISTANCE = 4.0  # 1 pixel = 4 game units
MAX_MAP_PIXELS_SIZE = (160, 112)  # (width, height) in pixels
DEBUG_UI_START_POS = (-100.0, 100.0)  # Debug view starting position
TOTAL_LAYERS = 3

# Derived Constants
MAP_WIDTH_UNITS = MAX_MAP_PIXELS_SIZE[0] * ONE_PIXEL_DISTANCE  # 640 units
MAP_HEIGHT_UNITS = MAX_MAP_PIXELS_SIZE[1] * ONE_PIXEL_DISTANCE  # 448 units

# Type definitions
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

    def show_ascii_map(self, view_center: Tuple[float, float] = None,
                       view_width: float = 80.0, view_height: float = 56.0,
                       cell_size: float = 2.0) -> str:
        """
        Generate a precise ASCII representation of the game map
        - view_center: (x,y) center position in game units
        - view_width: View width in game units (default 80 = 1/8 map)
        - view_height: View height in game units (default 56 = 1/8 map)
        - cell_size: Size of each ASCII cell in game units (default 2 = half-pixel)
        """
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

        # Set default view center to debug start position + offset
        if view_center is None:
            view_center = (DEBUG_UI_START_POS[0] + view_width / 2,
                           DEBUG_UI_START_POS[1] - view_height / 2)

        # Calculate array dimensions
        cells_x = math.ceil(view_width / cell_size)
        cells_y = math.ceil(view_height / cell_size)

        # Convert world coordinates to array indices
        def world_to_array(x: float, y: float) -> Tuple[int, int]:
            """Convert game world coordinates to array indices"""
            arr_x = int((x - DEBUG_UI_START_POS[0]) * len(self.entities[0]) / MAP_WIDTH_UNITS)
            arr_y = int((DEBUG_UI_START_POS[1] - y) * len(self.entities) / MAP_HEIGHT_UNITS)
            return (
                max(0, min(len(self.entities[0]) - 1, arr_x)),
                max(0, min(len(self.entities) - 1, arr_y))
            )

        map_str = ""
        for cell_y in range(cells_y):
            world_y = view_center[1] + (cell_y * cell_size) - view_height / 2
            for cell_x in range(cells_x):
                world_x = view_center[0] + (cell_x * cell_size) - view_width / 2
                arr_x, arr_y = world_to_array(world_x, world_y)

                # Sample the cell (entities > walls > floors)
                if self.entities[arr_y][arr_x] != 'Empty':
                    map_str += symbols.get(self.entities[arr_y][arr_x], '?')
                elif self.walls[arr_y][arr_x] != 'Empty':
                    map_str += symbols.get(self.walls[arr_y][arr_x], '?')
                else:
                    map_str += symbols.get(self.floors[arr_y][arr_x], '?')
            map_str += "\n"

        return (f"\n=== Map View ===\n"
                f"Center: {view_center}\n"
                f"Size: {view_width}x{view_height} game units\n"
                f"Resolution: {cells_x}x{cells_y} cells\n"
                f"Cell size: {cell_size} units\n\n"
                f"{map_str}")


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

        return ParsedGameState(
            floors=raw_state["state"][0],
            walls=raw_state["state"][1],
            entities=raw_state["state"][2],
            ai_states={
                idx: state
                for idx, state in enumerate(raw_state["ai_state"])
                if state
            }
        )

    except Exception as e:
        print(f"Error fetching game state: {e}")
        raise


if __name__ == "__main__":
    while True:
        try:
            game_state = fetch_game_state()

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

            # Default debug view
            print(game_state.show_ascii_map())

            # Player-centered view if available
            if game_state.player_positions:
                player_x, player_y = game_state.player_positions[0]
                player_world_x = DEBUG_UI_START_POS[0] + (player_x * MAP_WIDTH_UNITS / len(game_state.entities[0]))
                player_world_y = DEBUG_UI_START_POS[1] - (player_y * MAP_HEIGHT_UNITS / len(game_state.entities))

                print(game_state.show_ascii_map(
                    view_center=(player_world_x, player_world_y),
                    view_width=160.0,
                    view_height=112.0,
                    cell_size=1.0  # High-detail view
                ))

        except KeyboardInterrupt:
            print("\nStopping game state monitor")
            break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(1)