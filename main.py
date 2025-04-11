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
    velocity: Tuple[float, float]
    animation_state: Optional[str]


@dataclass
class ParsedGameState:
    """Structured representation of the game state"""
    floors: List[List[GameStateEncoding]]
    walls: List[List[GameStateEncoding]]
    entities: List[List[GameStateEncoding]]
    ai_states: Dict[int, bool]
    players: Dict[int, PlayerState]

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
    """Fetches and parses game state with proper velocity handling"""
    try:
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

        # Query for Players with movement components
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

        # Get GameState
        game_state_response = requests.post("http://127.0.0.1:15702/", json=game_state_payload, timeout=1.0)
        game_state_response.raise_for_status()
        game_state_data = game_state_response.json()

        # Get Players
        players_response = requests.post("http://127.0.0.1:15702/", json=players_payload, timeout=1.0)
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

        # Parse Players with careful velocity handling
        players = {}
        if players_data.get("result"):
            for idx, entity in enumerate(players_data["result"]):
                components = entity.get("components", {})
                if "hotline_miami_like::player::spawn::Player" not in components:
                    continue

                try:
                    player_comp = components["hotline_miami_like::player::spawn::Player"]
                    transform = components.get("bevy_transform::components::transform::Transform",
                                               {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]})

                    # Get movement component with proper fallbacks
                    movement = components.get("hotline_miami_like::player::movement::Movement", {})
                    velocity_x = movement.get("velocity_x", 0.0) if isinstance(movement, dict) else 0.0
                    velocity_y = movement.get("velocity_y", 0.0) if isinstance(movement, dict) else 0.0

                    # Calculate rotation
                    rotation = transform.get("rotation", [0, 0, 0, 1])
                    try:
                        rotation_angle = 2 * math.atan2(rotation[2], rotation[3])
                    except (IndexError, TypeError):
                        rotation_angle = 0.0

                    players[idx] = {
                        "position": (float(transform.get("translation", [0, 0, 0])[0]),
                                     float(transform.get("translation", [0, 0, 0])[1])),
                        "rotation": float(rotation_angle),
                        "character": PlayerCharacter(player_comp.get("color", "Orange")),
                        "device": PlayerDevice.KEYBOARD if str(player_comp.get("device", "")).lower() == "keyboard"
                        else PlayerDevice.GAMEPAD,
                        "is_shooting": bool(player_comp.get("is_shoot_button_pressed", False)),
                        "is_kicking": bool(player_comp.get("is_kicking", False)),
                        "is_moving": bool(player_comp.get("is_any_move_button_pressed", False)),
                        "is_grounded": True,
                        "health": float(
                            components.get("hotline_miami_like::player::damagable::Damagable", {}).get("health",
                                                                                                       100.0)),
                        "inventory": [],
                        "velocity": (float(velocity_x), float(velocity_y)),
                        "animation_state": None
                    }

                except Exception as e:
                    print(f"Error parsing player {idx}: {str(e)}")
                    continue

        return ParsedGameState(
            floors=raw_state["state"][0],
            walls=raw_state["state"][1],
            entities=raw_state["state"][2],
            ai_states={
                idx: bool(state)
                for idx, state in enumerate(raw_state["ai_state"])
                if idx < 10
            },
            players=players
        )

    except requests.exceptions.RequestException as e:
        print(f"Network error: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    return None


def print_player_details(player: PlayerState, index: int):
    """Prints detailed information about a player"""
    velocity_magnitude = math.sqrt(player['velocity'][0] ** 2 + player['velocity'][1] ** 2)
    print(f"\nPlayer {index} ({player['character'].value}):")
    print(f"  Position: ({player['position'][0]:.1f}, {player['position'][1]:.1f})")
    print(f"  Rotation: {math.degrees(player['rotation']):.1f}°")
    print(f"  Device: {player['device'].value}")
    print(f"  State: {'SHOOTING' if player['is_shooting'] else ''} "
          f"{'KICKING' if player['is_kicking'] else ''} "
          f"{'MOVING' if player['is_moving'] else ''}")
    print(f"  Health: {player['health']:.1f}")
    print(f"  Velocity: ({player['velocity'][0]:.1f}, {player['velocity'][1]:.1f})")
    print(f"  Speed: {velocity_magnitude:.1f} units/sec")
    if player['inventory']:
        print(f"  Inventory: {', '.join(player['inventory'])}")


def show_ascii_map(game_state: ParsedGameState,
                   view_center: Tuple[float, float] = None,
                   view_width: float = 160.0,
                   view_height: float = 112.0,
                   cell_size: float = 4.0) -> str:
    """
    Generate ASCII representation of the game map
    - view_center: (x,y) center position in game units
    - view_width: View width in game units
    - view_height: View height in game units
    - cell_size: Size of each ASCII cell in game units
    """
    symbols = {
        'Empty': ' ',
        'Floor': '\033[90m.\033[0m',  # Gray floor
        'Wall': '\033[91m#\033[0m',  # Red walls
        'Glass': '\033[94m░\033[0m',  # Blue glass
        'Crate': '\033[93m■\033[0m',  # Yellow crates
        'Pickup': '\033[92mP\033[0m',  # Green pickups
        'Bullet': '\033[91m•\033[0m',  # Red bullets
        'Characters': '\033[93m☻\033[0m'  # Yellow characters
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

    return map_str


def print_map_legend():
    """Prints a legend explaining the map symbols"""
    print("\n=== Map Legend ===")
    print("\033[90m. \033[0m- Floor")
    print("\033[91m# \033[0m- Wall")
    print("\033[94m░ \033[0m- Glass")
    print("\033[93m■ \033[0m- Crate")
    print("\033[92mP \033[0m- Pickup")
    print("\033[91m• \033[0m- Bullet")
    print("\033[93m☻ \033[0m- Character")
    print("Coordinates are in game units (1 unit = 4 pixels)")


if __name__ == "__main__":
    print_map_legend()

    while True:
        try:
            game_state = fetch_game_state()

            if game_state is None:
                print("\nNo game state available - waiting for game to start...")
                time.sleep(1)
                continue

            print("\n=== Game State ===")
            print(f"Active players: {len(game_state.players)}")

            # Print detailed player info
            for idx, player in game_state.players.items():
                print_player_details(player, idx)

            # Full map view
            print("\n=== Full Map View ===")
            print(show_ascii_map(
                game_state,
                view_center=(DEBUG_UI_START_POS[0] + MAP_WIDTH_UNITS / 2,
                             DEBUG_UI_START_POS[1] - MAP_HEIGHT_UNITS / 2),
                view_width=MAP_WIDTH_UNITS,
                view_height=MAP_HEIGHT_UNITS,
                cell_size=4.0
            ))

            # Player-centered views
            if game_state.players:
                for player_id, player in game_state.players.items():
                    print(f"\n=== Player {player_id} View ===")
                    print(show_ascii_map(
                        game_state,
                        view_center=player['position'],
                        view_width=160.0,
                        view_height=112.0,
                        cell_size=4.0
                    ))

        except KeyboardInterrupt:
            print("\nStopping game state monitor")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

        time.sleep(0.5)