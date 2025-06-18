# -*- coding: utf-8 -*-
import requests
import json
import time
import math
from typing import Dict, List, TypedDict, Literal, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os
import traceback
from actions import ActionClient

# --- Constants and Configuration ---
GAME_SERVER_URL = "http://127.0.0.1:15702/"
POLLING_INTERVAL_SECONDS = 0.2
PICKUP_POSITIONS: List[Tuple[int, int]] = []

# Game Constants
ONE_PIXEL_DISTANCE = 4.0
MAX_MAP_PIXELS_SIZE = (160, 112)
TOTAL_LAYERS = 3

_current_pickups: List[Tuple[int, int]] = []


def get_pickup_positions() -> List[Tuple[int, int]]:
    """Global function to get current pickup positions"""
    return _current_pickups.copy()


# --- Enums and Type Definitions ---
GameStateEncoding = Literal[
    'Empty', 'Floor', 'Wall', 'Glass',
    'Crate', 'Pickup', 'Bullet', 'Characters'
]


class PlayerCharacter(str, Enum):
    ORANGE = "Orange"
    LIME = "Lime"
    VITELOT = "Vitelot"
    LEMON = "Lemon"


class PlayerDevice(str, Enum):
    KEYBOARD = "Keyboard"
    GAMEPAD = "Gamepad"


SYMBOLS = {
    'Empty': ' ',
    'Floor': '\033[90m.\033[0m',  # Gray floor
    'Wall': '\033[91m#\033[0m',  # Red walls
    'Glass': '\033[94m░\033[0m',  # Blue glass
    'Crate': '\033[93m■\033[0m',  # Yellow crates
    'Pickup': '\033[92mP\033[0m',  # Green pickups
    'Bullet': '\033[91m•\033[0m',  # Red bullets
    'Characters': '\033[93m☻\033[0m'  # Yellow characters
}


class PlayerState(TypedDict):
    entity: int
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
    velocity: Tuple[float, float]  # Velocity reported by the game (Rapier)
    animation_state: Optional[str]
    calculated_velocity: Optional[Tuple[float, float]]
    calculated_speed: Optional[float]


@dataclass
class ParsedGameState:
    floors: List[List[GameStateEncoding]]
    walls: List[List[GameStateEncoding]]
    entities: List[List[GameStateEncoding]]
    ai_states: Dict[int, bool]
    players: Dict[int, PlayerState]

    @property
    def player_positions(self) -> List[Tuple[int, int]]:
        positions = []
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell == 'Characters':
                    positions.append((x, y))
        return positions

    def pickup_positions(self) -> List[Tuple[int, int]]:
        global PICKUP_POSITIONS
        positions = []
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell == 'Pickup':
                    positions.append((x, y))
        PICKUP_POSITIONS = positions.copy()
        return positions


# --- Core Logic Functions ---
def fetch_game_state() -> Optional[ParsedGameState]:
    """Fetches and parses game state with comprehensive error checking."""
    try:
        # 1) Global GameState query
        game_state_payload = {
            "id": 1, "jsonrpc": "2.0", "method": "bevy/query",
            "params": {
                "data": {
                    "components": ["hotline_miami_like::ai::game_state::GameState"],
                    "has": [], "option": []
                },
                "filter": {"with": [], "without": []}
            }
        }

        # 2) Players query
        players_payload = {
            "id": 2, "jsonrpc": "2.0", "method": "bevy/query",
            "params": {
                "data": {
                    "components": [
                        "hotline_miami_like::player::spawn::Player",
                        "hotline_miami_like::player::input::PlayerInput",
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

        # 3) Execute requests with timeout
        try:
            gs_resp = requests.post(GAME_SERVER_URL, json=game_state_payload, timeout=1.0)
            gs_resp.raise_for_status()
            gs_data = gs_resp.json()

            pl_resp = requests.post(GAME_SERVER_URL, json=players_payload, timeout=1.0)
            pl_resp.raise_for_status()
            pl_data = pl_resp.json()
        except requests.RequestException as e:
            print(f"Network error: {e}")
            return None

        # 4) Validate GameState response
        if not isinstance(gs_data.get("result"), list) or len(gs_data["result"]) == 0:
            print("Invalid GameState response format")
            return None

        try:
            gs_ent = gs_data["result"][0]
            raw_state = gs_ent.get("components", {}).get("hotline_miami_like::ai::game_state::GameState")
            if not raw_state:
                print("Missing GameState component")
                return None
        except (KeyError, IndexError) as e:
            print(f"GameState parsing error: {e}")
            return None

        # 5) Parse Players with robust error checking
        players: Dict[int, PlayerState] = {}

        if not isinstance(pl_data.get("result"), list):
            print("Invalid players response format")
            return None

        for idx, ent in enumerate(pl_data["result"]):
            if not isinstance(ent, dict) or "components" not in ent:
                print(f"Invalid entity format at index {idx}")
                continue

            if "hotline_miami_like::player::spawn::Player" not in ent["components"]:
                continue

            try:
                comps = ent["components"]
                player_comp = comps["hotline_miami_like::player::spawn::Player"]
                input_comp = comps.get("hotline_miami_like::player::input::PlayerInput", {})
                player_comp.update(input_comp)

                entity_id = ent.get("entity", idx)
                print(f"\n[DEBUG] Player {entity_id} raw component:\n{json.dumps(player_comp, indent=2)}")

                # Safe component access with defaults
                transform = comps.get(
                    "bevy_transform::components::transform::Transform",
                    {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]}
                )
                damagable = comps.get(
                    "hotline_miami_like::player::damagable::Damagable",
                    {"health": 100.0}
                )
                movement = comps.get(
                    "hotline_miami_like::player::movement::Movement",
                    {"velocity_x": 0.0, "velocity_y": 0.0}
                )

                # Safe rotation calculation
                rot = transform.get("rotation", [0, 0, 0, 1])
                try:
                    rotation_angle = 2 * math.atan2(float(rot[2]), float(rot[3]))
                except (IndexError, ValueError, TypeError):
                    rotation_angle = 0.0
                    print(f"Invalid rotation data for player {idx}")

                # Robust character handling
                try:
                    character = PlayerCharacter(player_comp.get("character", "Lemon"))
                except ValueError:
                    character = PlayerCharacter.LEMON
                    print(f"Invalid character for player {idx}, defaulting to LEMON")

                # Safe position parsing
                try:
                    pos_x = float(transform["translation"][0])
                    pos_y = float(transform["translation"][1])
                except (KeyError, IndexError, ValueError):
                    pos_x, pos_y = 0.0, 0.0
                    print(f"Invalid position for player {idx}")

                # Safe velocity parsing
                try:
                    vel_x = float(movement.get("velocity_x", 0.0))
                    vel_y = float(movement.get("velocity_y", 0.0))
                except ValueError:
                    vel_x, vel_y = 0.0, 0.0
                    print(f"Invalid velocity for player {idx}")

                players[idx] = {
                    "entity": entity_id,
                    "position": (pos_x, pos_y),
                    "rotation": rotation_angle,
                    "character": character,
                    "device": (
                        PlayerDevice.KEYBOARD
                        if player_comp.get("device", "Keyboard") == "Keyboard"
                        else PlayerDevice.GAMEPAD
                    ),
                    "is_shooting": bool(player_comp.get("is_shoot_button_pressed", False)),
                    "is_kicking": bool(player_comp.get("is_foot_button_just_pressed", False)),
                    "is_moving": bool(player_comp.get("is_any_move_button_pressed", False)),
                    "is_grounded": True,
                    "health": float(damagable.get("health", 100.0)),
                    "inventory": [],
                    "velocity": (vel_x, vel_y),
                    "animation_state": None,
                    "calculated_velocity": None,
                    "calculated_speed": None
                }
            except Exception as e:
                print(f"Error processing player {idx}: {e}")
                continue

        # 6) Return structured state with validation
        try:
            return ParsedGameState(
                floors=raw_state["state"][0],
                walls=raw_state["state"][1],
                entities=raw_state["state"][2],
                ai_states={i: bool(s) for i, s in enumerate(raw_state.get("ai_state", []))},
                players=players
            )
        except (KeyError, IndexError) as e:
            print(f"Error creating ParsedGameState: {e}")
            return None

    except Exception as e:
        print(f"Unexpected error in fetch_game_state: {e}")
        traceback.print_exc()
        return None


def update_calculated_velocity(current_state: ParsedGameState,
                               previous_state_data: Optional[ParsedGameState],
                               delta_time: float):
    """Calculates velocity based on position change and updates the current_state."""
    if previous_state_data is None or delta_time <= 1e-6:
        for player in current_state.players.values():
            player['calculated_velocity'] = None
            player['calculated_speed'] = None
        return

    for player_id, current_player in current_state.players.items():
        if player_id in previous_state_data.players:
            previous_player = previous_state_data.players[player_id]
            if (isinstance(previous_player.get('position'), (list, tuple)) and
                    len(previous_player['position']) >= 2 and
                    isinstance(current_player.get('position'), (list, tuple)) and
                    len(current_player['position']) >= 2):
                prev_pos = previous_player['position']
                curr_pos = current_player['position']
                delta_x = curr_pos[0] - prev_pos[0]
                delta_y = curr_pos[1] - prev_pos[1]

                calc_vx = delta_x / delta_time
                calc_vy = delta_y / delta_time
                calc_speed_sq = calc_vx ** 2 + calc_vy ** 2
                calc_speed = math.sqrt(calc_speed_sq) if calc_speed_sq >= 0 else 0.0

                current_player['calculated_velocity'] = (calc_vx, calc_vy)
                current_player['calculated_speed'] = calc_speed
            else:
                current_player['calculated_velocity'] = None
                current_player['calculated_speed'] = None
        else:
            current_player['calculated_velocity'] = None
            current_player['calculated_speed'] = None


def print_player_details(player: PlayerState, player_id: Any):
    """Prints detailed information about a player."""
    reported_velocity = player.get('velocity', (0.0, 0.0))
    reported_speed = math.sqrt(reported_velocity[0] ** 2 + reported_velocity[1] ** 2)

    print(f"\nPlayer ID {player_id} ({player.get('character', PlayerCharacter.ORANGE).value}):")
    print(f"  Entity ID: {player.get('entity')}")

    pos = player.get('position', ('N/A', 'N/A'))
    pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})" if isinstance(pos, tuple) and len(pos) == 2 else f"{pos}"
    print(f"  Position: {pos_str}")

    rot_deg = math.degrees(player.get('rotation', 0.0))
    print(f"  Rotation: {rot_deg:.1f}°")
    print(f"  Device: {player.get('device', PlayerDevice.KEYBOARD).value}")

    state_str = f"{'SHOOTING ' if player.get('is_shooting') else ''}" \
                f"{'KICKING ' if player.get('is_kicking') else ''}" \
                f"{'MOVING' if player.get('is_moving') else ''}"
    print(f"  State: {state_str.strip() if state_str.strip() else 'IDLE'}")
    print(f"  Health: {player.get('health', 0.0):.1f}")
    print(f"  Reported Velocity (Rapier): ({reported_velocity[0]:.1f}, {reported_velocity[1]:.1f})")
    print(f"  Reported Speed (Rapier): {reported_speed:.1f} units/sec")

    if player.get('calculated_velocity') is not None:
        calc_vel = player['calculated_velocity']
        calc_speed = player.get('calculated_speed', 0.0)
        print(f"  Calculated Velocity (Pos Δ): ({calc_vel[0]:.1f}, {calc_vel[1]:.1f}) (Avg)")
        print(f"  Calculated Speed (Pos Δ): {calc_speed:.1f} units/sec (Avg)")
    else:
        print(f"  Calculated Velocity (Pos Δ): N/A")

    inv = player.get('inventory', [])
    if inv:
        print(f"  Inventory: {', '.join(map(str, inv))}")


def show_ascii_map(game_state: ParsedGameState,
                   width: float = 160,
                   height: float = 112) -> str:
    """Generates ASCII representation of the game map."""
    if not game_state or not hasattr(game_state, 'entities') or not game_state.entities or \
            not hasattr(game_state, 'walls') or not game_state.walls or \
            not hasattr(game_state, 'floors') or not game_state.floors:
        return "Error: Invalid game state layers.\n"

    if not isinstance(game_state.entities[0], list) or \
            not isinstance(game_state.walls[0], list) or \
            not isinstance(game_state.floors[0], list):
        return "Error: Map layers are not lists of lists.\n"

    result = ""
    for col_idx in range(width):
        row_str = ""
        for row_idx in range(height):
            try:
                entity = game_state.entities[col_idx][row_idx]
                wall = game_state.walls[col_idx][row_idx]
                floor = game_state.floors[col_idx][row_idx]
            except IndexError:
                entity, wall, floor = '?', '?', '?'

            char_to_add = SYMBOLS.get(entity, SYMBOLS.get(wall, SYMBOLS.get(floor, '?')))
            row_str += char_to_add
        result += row_str + "\n"
    return result


def print_map_legend():
    """Prints a legend explaining the map symbols."""
    print("\n=== Map Legend ===")
    print(f"{SYMBOLS['Floor']} - Floor")
    print(f"{SYMBOLS['Wall']} - Wall")
    print(f"{SYMBOLS['Glass']} - Glass")
    print(f"{SYMBOLS['Crate']} - Crate")
    print(f"{SYMBOLS['Pickup']} - Pickup")
    print(f"{SYMBOLS['Bullet']} - Bullet")
    print(f"{SYMBOLS['Characters']} - Character")
    print("Coordinates are in game units")


# --- Main Execution Loop ---
if __name__ == "__main__":
    action_client = ActionClient(GAME_SERVER_URL)
    previous_game_state: Optional[ParsedGameState] = None
    previous_time: Optional[float] = None
    first_run = True

    while True:
        start_time = time.monotonic()

        try:
            current_game_state = fetch_game_state()
            current_time = time.monotonic()

            if current_game_state is None:
                if first_run:
                    print("\nWaiting for game state from server...", end="\r")
                time.sleep(0.5)
                continue
            elif first_run:
                print("Game state received. Starting updates...")
                first_run = False

            delta_time = (current_time - previous_time) if previous_time is not None else 0.0
            update_calculated_velocity(current_game_state, previous_game_state, delta_time)

            os.system('cls' if os.name == 'nt' else 'clear')
            print_map_legend()

            print("\n=== Game State Update ===")
            print(f"Active players: {len(current_game_state.players)}")
            if delta_time > 0:
                print(f"Time since last update: {delta_time:.3f}s")

            if not current_game_state.players:
                print("\nNo active players detected.")
            else:
                for player_id in sorted(current_game_state.players.keys()):
                    print_player_details(current_game_state.players[player_id], player_id)

            print("\n=== Full Map View ===")
            print(show_ascii_map(current_game_state))

            previous_game_state = current_game_state
            previous_time = current_time

        except KeyboardInterrupt:
            print("\nStopping game state monitor.")
            break
        except Exception as e:
            print(f"\n--- ERROR IN MAIN LOOP ---")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            print(f"--------------------------")
            time.sleep(1.0)

        processing_time = time.monotonic() - start_time
        sleep_time = max(0, POLLING_INTERVAL_SECONDS - processing_time)
        time.sleep(sleep_time)