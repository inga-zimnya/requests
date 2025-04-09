import requests
import json
import time
import math
from typing import Dict, List, TypedDict, Literal, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

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


# Enums matching Rust code
class PlayerCharacter(str, Enum):
    ORANGE = "Orange"
    LIME = "Lime"
    VITELOT = "Vitelot"
    LEMON = "Lemon"


class PlayerDevice(str, Enum):
    KEYBOARD = "Keyboard"
    GAMEPAD = "Gamepad"


class PlayerState(TypedDict):
    """Detailed player state based on spawn.rs components"""
    position: Tuple[float, float]
    rotation: float
    character: PlayerCharacter
    device: PlayerDevice
    is_shooting: bool
    is_kicking: bool
    is_moving: bool
    is_grounded: bool
    health: float
    inventory: List[str]
    velocity: Optional[Tuple[float, float]]
    animation_state: Optional[str]


@dataclass
class ParsedGameState:
    """Structured representation of the game state"""
    floors: List[List[GameStateEncoding]]
    walls: List[List[GameStateEncoding]]
    entities: List[List[GameStateEncoding]]
    ai_states: Dict[int, bool]
    players: Dict[int, PlayerState]  # Keyed by player index

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


def fetch_game_state() -> Optional[ParsedGameState]:
    """Fetches and parses game state from separate entities"""
    # First query for GameState
    game_state_payload = {
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

    # Then query for Players with their components
    players_payload = {
        "id": 2,
        "jsonrpc": "2.0",
        "method": "bevy/query",
        "params": {
            "data": {
                "components": [
                    "hotline_miami_like::player::spawn::Player",
                    "bevy_transform::components::transform::Transform",
                    "hotline_miami_like::player::damagable::Damagable",
                    "hotline_miami_like::player::movement::Movement"
                ],
                "has": [],
                "option": []
            },
            "filter": {
                "with": ["hotline_miami_like::player::spawn::Player"],
                "without": []
            }
        }
    }

    try:
        # Get GameState first
        game_state_response = requests.post("http://127.0.0.1:15702/", json=game_state_payload)
        game_state_response.raise_for_status()
        game_state_data = game_state_response.json()

        # Get Players separately
        players_response = requests.post("http://127.0.0.1:15702/", json=players_payload)
        players_response.raise_for_status()
        players_data = players_response.json()

        # Parse GameState
        if not game_state_data.get("result"):
            print("Warning: No GameState entity found")
            return None

        game_state_entity = game_state_data["result"][0]
        if "hotline_miami_like::ai::game_state::GameState" not in game_state_entity["components"]:
            print("Warning: GameState entity missing GameState component")
            return None

        raw_state = game_state_entity["components"]["hotline_miami_like::ai::game_state::GameState"]

        # Parse Players
        players = {}
        if players_data.get("result"):
            for idx, entity in enumerate(players_data["result"]):
                if "hotline_miami_like::player::spawn::Player" not in entity["components"]:
                    continue

                player_comp = entity["components"]["hotline_miami_like::player::spawn::Player"]
                transform = entity["components"].get(
                    "bevy_transform::components::transform::Transform",
                    {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]}
                )

                # Get optional components with defaults
                damagable = entity["components"].get(
                    "hotline_miami_like::player::damagable::Damagable",
                    {"health": 100.0}
                )
                movement = entity["components"].get(
                    "hotline_miami_like::player::movement::Movement",
                    {"velocity_x": 0.0, "velocity_y": 0.0}
                )

                # Calculate rotation from quaternion (x,y,z,w)
                rotation = transform.get("rotation", [0, 0, 0, 1])
                rotation_angle = 2 * math.atan2(rotation[2], rotation[3])  # Simplified yaw

                players[idx] = {
                    "position": tuple(transform.get("translation", [0, 0, 0])[:2]),
                    "rotation": rotation_angle,
                    "character": PlayerCharacter(player_comp.get("color", "Orange")),
                    "device": PlayerDevice.KEYBOARD if player_comp.get(
                        "device") == "Keyboard" else PlayerDevice.GAMEPAD,
                    "is_shooting": player_comp.get("is_shoot_button_pressed", False),
                    "is_kicking": player_comp.get("is_kicking", False),
                    "is_moving": player_comp.get("is_any_move_button_pressed", False),
                    "is_grounded": True,  # Default from spawn.rs
                    "health": damagable.get("health", 100.0),
                    "inventory": [],  # Would come from Inventory component
                    "velocity": (movement.get("velocity_x", 0.0), movement.get("velocity_y", 0.0)),
                    "animation_state": None
                }

        return ParsedGameState(
            floors=raw_state["state"][0],
            walls=raw_state["state"][1],
            entities=raw_state["state"][2],
            ai_states={
                idx: state
                for idx, state in enumerate(raw_state["ai_state"])
                if state
            },
            players=players
        )

    except requests.exceptions.RequestException as e:
        print(f"Network error fetching game state: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {str(e)}")
        return None
    except KeyError as e:
        print(f"Missing expected key in response: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None


def print_player_details(player: PlayerState, index: int):
    """Prints detailed information about a player"""
    print(f"\nPlayer {index} ({player['character'].value}):")
    print(f"  Position: ({player['position'][0]:.1f}, {player['position'][1]:.1f})")
    print(f"  Rotation: {math.degrees(player['rotation']):.1f}°")
    print(f"  Device: {player['device'].value}")
    print(f"  State: {'Shooting' if player['is_shooting'] else ''} "
          f"{'Kicking' if player['is_kicking'] else ''} "
          f"{'Moving' if player['is_moving'] else ''}")
    print(f"  Health: {player['health']:.1f}")
    print(f"  Velocity: ({player['velocity'][0]:.1f}, {player['velocity'][1]:.1f})")
    if player['inventory']:
        print(f"  Inventory: {', '.join(player['inventory'])}")


def show_ascii_map(game_state: ParsedGameState,
                   view_center: Tuple[float, float] = None,
                   view_width: float = 160.0,
                   view_height: float = 112.0,
                   cell_size: float = 1.0) -> str:
    """
    Generate ASCII representation of the game map
    - view_center: (x,y) center position in game units
    - view_width: View width in game units
    - view_height: View height in game units
    - cell_size: Size of each ASCII cell in game units
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

    # Set default view center
    if view_center is None:
        view_center = (DEBUG_UI_START_POS[0] + view_width / 2,
                       DEBUG_UI_START_POS[1] - view_height / 2)

    # Calculate array dimensions
    cells_x = math.ceil(view_width / cell_size)
    cells_y = math.ceil(view_height / cell_size)

    # Convert world coordinates to array indices
    def world_to_array(x: float, y: float) -> Tuple[int, int]:
        arr_x = int((x - DEBUG_UI_START_POS[0]) * len(game_state.entities[0]) / MAP_WIDTH_UNITS)
        arr_y = int((DEBUG_UI_START_POS[1] - y) * len(game_state.entities) / MAP_HEIGHT_UNITS)
        return (
            max(0, min(len(game_state.entities[0]) - 1, arr_x)),
            max(0, min(len(game_state.entities) - 1, arr_y))
        )

    map_str = ""
    for cell_y in range(cells_y):
        world_y = view_center[1] + (cell_y * cell_size) - view_height / 2
        for cell_x in range(cells_x):
            world_x = view_center[0] + (cell_x * cell_size) - view_width / 2
            arr_x, arr_y = world_to_array(world_x, world_y)

            # Check layers in proper order (entities > walls > floors)
            entity = game_state.entities[arr_y][arr_x]
            wall = game_state.walls[arr_y][arr_x]
            floor = game_state.floors[arr_y][arr_x]

            if entity != 'Empty':
                map_str += symbols.get(entity, '?')
            elif wall != 'Empty':
                map_str += symbols.get(wall, '?')
            else:
                map_str += symbols.get(floor, '?')
        map_str += "\n"

        # Add coordinate grid markers every 10 units
    grid_markers = "   " + "".join(f"{x:<10}" for x in range(0, int(view_width), 10))
    map_str = f"Y\\X {grid_markers[:len(map_str.split('\n')[0])]}\n" + map_str

    # Add Y-axis markers
    y_markers = range(int(view_center[1] + view_height / 2),
                      int(view_center[1] - view_height / 2), -10)
    for i, line in enumerate(map_str.split('\n')[1:]):
        if i % 10 == 0 and i // 10 < len(y_markers):
            map_str = map_str.replace(line, f"{y_markers[i // 10]:3}" + line[3:], 1)

    return map_str


if __name__ == "__main__":
    while True:
        try:
            game_state = fetch_game_state()

            if game_state is None:
                print("No game state available - waiting for game to start...")
                time.sleep(1)
                continue

            print("\n=== Game State ===")
            print(f"Active players: {len(game_state.players)}")

            for idx, player in game_state.players.items():
                print_player_details(player, idx)

            print("\nMap Overview:")
            print(show_ascii_map(
                game_state,
                view_center=(DEBUG_UI_START_POS[0] + MAP_WIDTH_UNITS / 2,
                             DEBUG_UI_START_POS[1] - MAP_HEIGHT_UNITS / 2),
                view_width=MAP_WIDTH_UNITS,
                view_height=MAP_HEIGHT_UNITS,
                cell_size=4.0  # 1 cell = 1 pixel (4 game units)
            ))

            # Show detailed view around first player if available
            if game_state.players:
                first_player = next(iter(game_state.players.values()))
                print("\nPlayer View:")
                print(show_ascii_map(
                    game_state,
                    view_center=first_player['position'],
                    view_width=160.0,
                    view_height=112.0,
                    cell_size=1.0
                ))

        except KeyboardInterrupt:
            print("\nStopping game state monitor")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

        time.sleep(0.5)  # Update twice per second