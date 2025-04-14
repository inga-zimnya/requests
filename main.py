# -*- coding: utf-8 -*-
import requests
import json
import time
import math
from typing import Dict, List, TypedDict, Literal, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os  # For screen clearing
import traceback  # For detailed error printing

# --- Constants and Configuration ---
GAME_SERVER_URL = "http://127.0.0.1:15702/"
POLLING_INTERVAL_SECONDS = 0.2  # How often to fetch game state

# Game Constants
ONE_PIXEL_DISTANCE = 4.0
MAX_MAP_PIXELS_SIZE = (160, 112)
# DEBUG_UI_START_POS = (-100.0, 100.0)
TOTAL_LAYERS = 3

# Derived Constants
# MAP_WIDTH_UNITS = MAX_MAP_PIXELS_SIZE[0] * ONE_PIXEL_DISTANCE
# MAP_HEIGHT_UNITS = MAX_MAP_PIXELS_SIZE[1] * ONE_PIXEL_DISTANCE

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


# Define symbols globally so both map and legend can access them
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
    # Velocity calculated from pos change
    calculated_velocity: Optional[Tuple[float, float]]
    calculated_speed: Optional[float]  # Speed calculated from pos change


@dataclass
class ParsedGameState:
    floors: List[List[GameStateEncoding]]
    walls: List[List[GameStateEncoding]]
    entities: List[List[GameStateEncoding]]
    ai_states: Dict[int, bool]
    players: Dict[Any, PlayerState]  # Key can be player entity ID (int/str)

    # Property methods remain the same
    @property
    def player_positions(self) -> List[Tuple[int, int]]:
        positions = []
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell == 'Characters':
                    positions.append((x, y))
        return positions

    @property
    def pickup_positions(self) -> List[Tuple[int, int]]:
        positions = []
        for y, row in enumerate(self.entities):
            for x, cell in enumerate(row):
                if cell == 'Pickup':
                    positions.append((x, y))
        return positions

    # Add other property methods here if needed (bullet, crate, etc.)
    # ... (bullet_positions, crate_positions, glass_positions, etc.) ...


# --- Core Logic Functions ---

def fetch_game_state() -> Optional[ParsedGameState]:
    """Fetches and parses game state, reading velocity from Rapier."""
    try:
        game_state_payload = {
            "id": 1, "jsonrpc": "2.0", "method": "bevy/query",
            "params": {
                "data": {"components": ["hotline_miami_like::ai::game_state::GameState"], "has": [], "option": []},
                "filter": {"with": [], "without": []}
            }
        }

        players_payload = {
            "id": 2, "jsonrpc": "2.0", "method": "bevy/query",
            "params": {
                "data": {
                    "components": [
                        "hotline_miami_like::player::spawn::Player",
                        "bevy_transform::components::transform::Transform",
                        "hotline_miami_like::player::damagable::Damagable",
                        # Request Rapier's Velocity # TODO: player does not have velocity component. They have KinematicCharacterController. Need fix.
                        "bevy_rapier2d::dynamics::Velocity"
                    ],
                    "has": [],
                    "option": ["entity"]  # Request entity ID
                },
                "filter": {
                    "with": ["hotline_miami_like::player::spawn::Player", "bevy_rapier2d::dynamics::Velocity"],
                    "without": []
                }
            }
        }

        game_state_response = requests.post(
            GAME_SERVER_URL, json=game_state_payload, timeout=1.0)
        game_state_response.raise_for_status()
        game_state_data = game_state_response.json()

        players_response = requests.post(
            GAME_SERVER_URL, json=players_payload, timeout=1.0)
        players_response.raise_for_status()
        players_data = players_response.json()

        # --- Parse GameState ---
        if not game_state_data.get("result"):
            return None
        game_state_entity = game_state_data["result"][0]
        if "hotline_miami_like::ai::game_state::GameState" not in game_state_entity.get("components", {}):
            print("Warning: GameState entity missing GameState component")
            return None
        raw_state = game_state_entity["components"]["hotline_miami_like::ai::game_state::GameState"]

        # --- Parse Players ---
        players = {}
        if players_data.get("result"):
            for entity_data in players_data["result"]:
                entity_id = entity_data.get("entity", {}).get(
                    "id", f"unknown_{time.time()}")  # Use entity ID or fallback
                components = entity_data.get("components", {})

                if "hotline_miami_like::player::spawn::Player" not in components \
                   or "bevy_rapier2d::dynamics::Velocity" not in components:
                    continue  # Skip if essential components missing

                try:
                    player_comp = components["hotline_miami_like::player::spawn::Player"]
                    transform = components.get("bevy_transform::components::transform::Transform",
                                               {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]})
                    rapier_velocity_comp = components.get(
                        "bevy_rapier2d::dynamics::Velocity", {})
                    velocity_vec = rapier_velocity_comp.get(
                        "linvel", [0.0, 0.0])

                    velocity_x, velocity_y = 0.0, 0.0
                    if isinstance(velocity_vec, (list, tuple)) and len(velocity_vec) >= 2:
                        try:
                            velocity_x = float(velocity_vec[0])
                            velocity_y = float(velocity_vec[1])
                        except (TypeError, ValueError):
                            pass

                    rotation = transform.get("rotation", [0, 0, 0, 1])
                    rotation_angle = 0.0
                    try:
                        q2, q3 = float(rotation[2]), float(rotation[3])
                        if abs(q3) > 1e-9 or abs(q2) > 1e-9:  # Avoid atan2(0,0) -> NaN
                            rotation_angle = 2 * math.atan2(q2, q3)
                    except (IndexError, TypeError, ValueError):
                        pass

                    players[entity_id] = {
                        "position": (float(transform.get("translation", [0, 0, 0])[0]),
                                     float(transform.get("translation", [0, 0, 0])[1])),
                        "rotation": rotation_angle,
                        "character": PlayerCharacter(player_comp.get("color", "Orange")),
                        "device": PlayerDevice.KEYBOARD if str(player_comp.get("device", "")).lower() == "keyboard"
                        else PlayerDevice.GAMEPAD,
                        "is_shooting": bool(player_comp.get("is_shoot_button_pressed", False)),
                        "is_kicking": bool(player_comp.get("is_kicking", False)),
                        "is_moving": bool(player_comp.get("is_any_move_button_pressed", False)),
                        "is_grounded": True,  # Placeholder
                        "health": float(
                            components.get("hotline_miami_like::player::damagable::Damagable", {}).get(
                                "health", 100.0)
                        ),
                        "inventory": [],  # Placeholder
                        "velocity": (velocity_x, velocity_y),
                        "animation_state": None,  # Placeholder
                        "calculated_velocity": None,
                        "calculated_speed": None
                    }

                except Exception as e:
                    print(f"\n--- Error parsing player {entity_id} ---")
                    print(f"Error: {str(e)}")
                    print(
                        f"Components data: {json.dumps(components, indent=2)}")
                    print("-" * 20)
                    continue

        # --- Validate state structure before creating object ---
        if not isinstance(raw_state.get("state"), list) or len(raw_state["state"]) != TOTAL_LAYERS:
            print(
                f"Error: Invalid 'state' structure received: {raw_state.get('state')}")
            return None
        if not all(isinstance(layer, list) for layer in raw_state["state"]):
            print(f"Error: Layers within 'state' are not all lists.")
            return None

        return ParsedGameState(
            floors=raw_state["state"][0],
            walls=raw_state["state"][1],
            entities=raw_state["state"][2],
            ai_states={
                idx: bool(state)
                for idx, state in enumerate(raw_state.get("ai_state", []))
            },
            players=players
        )

    except requests.exceptions.RequestException:  # Less verbose network error
        pass
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in fetch_game_state: {str(e)}")
        traceback.print_exc()

    return None


def update_calculated_velocity(current_state: ParsedGameState,
                               # Renamed for clarity
                               previous_state_data: Optional[ParsedGameState],
                               delta_time: float):
    """Calculates velocity based on position change and updates the current_state."""
    if previous_state_data is None or delta_time <= 1e-6:
        for player_id, current_player in current_state.players.items():
            current_player['calculated_velocity'] = None
            current_player['calculated_speed'] = None
        return

    for player_id, current_player in current_state.players.items():
        if player_id in previous_state_data.players:
            previous_player = previous_state_data.players[player_id]
            if isinstance(previous_player.get('position'), (list, tuple)) and len(previous_player['position']) >= 2 and \
               isinstance(current_player.get('position'), (list, tuple)) and len(current_player['position']) >= 2:
                prev_pos = previous_player['position']
                curr_pos = current_player['position']
                delta_x = curr_pos[0] - prev_pos[0]
                delta_y = curr_pos[1] - prev_pos[1]

                calc_vx = delta_x / delta_time
                calc_vy = delta_y / delta_time
                # Avoid math domain error for sqrt(negative) if delta is huge/erroneous
                calc_speed_sq = calc_vx**2 + calc_vy**2
                calc_speed = math.sqrt(
                    calc_speed_sq) if calc_speed_sq >= 0 else 0.0

                current_player['calculated_velocity'] = (calc_vx, calc_vy)
                current_player['calculated_speed'] = calc_speed
            else:
                current_player['calculated_velocity'] = None
                current_player['calculated_speed'] = None
        else:
            current_player['calculated_velocity'] = None
            current_player['calculated_speed'] = None


# --- Display Functions ---

def print_player_details(player: PlayerState, player_id: Any):
    """Prints detailed information about a player."""
    reported_velocity = player.get('velocity', (0.0, 0.0))
    reported_speed = math.sqrt(
        reported_velocity[0]**2 + reported_velocity[1]**2)

    # Default character
    print(
        f"\nPlayer ID {player_id} ({player.get('character', PlayerCharacter.ORANGE).value}):")
    pos = player.get('position', ('N/A', 'N/A'))
    rot_deg = math.degrees(player.get('rotation', 0.0))
    # Safely format position even if it's not a tuple
    pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})" if isinstance(
        pos, tuple) and len(pos) == 2 else f"{pos}"
    print(f"  Position: {pos_str}")
    print(f"  Rotation: {rot_deg:.1f}°")
    # Default device
    print(f"  Device: {player.get('device', PlayerDevice.KEYBOARD).value}")
    state_str = f"{'SHOOTING ' if player.get('is_shooting') else ''}" \
                f"{'KICKING ' if player.get('is_kicking') else ''}" \
                f"{'MOVING' if player.get('is_moving') else ''}"  # Keep MOVING or empty
    # Show IDLE if no other state
    print(f"  State: {state_str.strip() if state_str.strip() else 'IDLE'}")
    print(f"  Health: {player.get('health', 0.0):.1f}")
    print(
        f"  Reported Velocity (Rapier): ({reported_velocity[0]:.1f}, {reported_velocity[1]:.1f})")
    print(f"  Reported Speed (Rapier): {reported_speed:.1f} units/sec")

    if player.get('calculated_velocity') is not None and player.get('calculated_speed') is not None:
        calc_vel = player['calculated_velocity']
        calc_speed = player['calculated_speed']
        print(
            f"  Calculated Velocity (Pos Δ): ({calc_vel[0]:.1f}, {calc_vel[1]:.1f}) (Avg)")
        print(f"  Calculated Speed (Pos Δ): {calc_speed:.1f} units/sec (Avg)")
    else:
        print(f"  Calculated Velocity (Pos Δ): N/A")

    inv = player.get('inventory', [])
    if inv:
        # Ensure items are strings
        print(f"  Inventory: {', '.join(map(str, inv))}")


def show_ascii_map(game_state: ParsedGameState,
                   width: float = 160,
                   height: float = 112,
                   ) -> str:
    """Generates ASCII representation of the game map."""

    if not game_state or not hasattr(game_state, 'entities') or not game_state.entities or \
       not hasattr(game_state, 'walls') or not game_state.walls or \
       not hasattr(game_state, 'floors') or not game_state.floors:
        return "Error: Invalid game state layers.\n"
    # Further check if layers have content and are lists of lists
    if not isinstance(game_state.entities[0], list) or \
       not isinstance(game_state.walls[0], list) or \
       not isinstance(game_state.floors[0], list):
        return "Error: Map layers are not lists of lists.\n"

    map_pixel_height = len(game_state.entities)
    map_pixel_width = len(game_state.entities[0])

    if map_pixel_width <= 0 or map_pixel_height <= 0:
        return "Error: Invalid map dimensions.\n"

    map_str_builder = [[""] * height] * width
    result = ""
    for col_idx in range(width):
        row_str = ""
        for row_idx in range(height):

            # Get cell content safely
            try:
                entity = game_state.entities[col_idx][row_idx]
                wall = game_state.walls[col_idx][row_idx]
                floor = game_state.floors[col_idx][row_idx]
            except IndexError:
                entity, wall, floor = '?', '?', '?'  # Out of bounds somehow

            char_to_add = '?'
            if entity != 'Empty':
                char_to_add = SYMBOLS.get(entity, '?')
            elif wall != 'Empty':
                char_to_add = SYMBOLS.get(wall, '?')
            else:
                char_to_add = SYMBOLS.get(floor, '?')

            map_str_builder[col_idx][row_idx] = char_to_add
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
    # Correct variable name
    previous_game_state: Optional[ParsedGameState] = None
    previous_time: Optional[float] = None
    first_run = True

    while True:
        start_time = time.monotonic()

        try:
            current_game_state = fetch_game_state()
            current_time = time.monotonic()

            if current_game_state is None:
                if first_run:  # Only show waiting message if we haven't received anything yet
                    print("\nWaiting for game state from server...", end="\r")
                # No need to reset previous_game_state here, keep the last known good one if available
                time.sleep(0.5)
                continue
            elif first_run:
                print("Game state received. Starting updates...")
                first_run = False

            delta_time = (
                current_time - previous_time) if previous_time is not None else 0.0

            # *** THE FIX IS HERE ***
            update_calculated_velocity(
                current_game_state, previous_game_state, delta_time)
            # *** ***

            os.system('cls' if os.name == 'nt' else 'clear')
            print_map_legend()

            print("\n=== Game State Update ===")
            print(f"Active players: {len(current_game_state.players)}")
            if delta_time > 0:
                print(f"Time since last update: {delta_time:.3f}s")

            if not current_game_state.players:
                print("\nNo active players detected.")
            else:
                # Sort players by ID for consistent order (optional)
                sorted_player_ids = sorted(current_game_state.players.keys())
                for player_id in sorted_player_ids:
                    player_data = current_game_state.players[player_id]
                    print_player_details(player_data, player_id)

            print("\n=== Full Map View ===")
            print(show_ascii_map(current_game_state))

            # Update history for the next loop iteration
            previous_game_state = current_game_state  # Correct variable name used here
            previous_time = current_time

        except KeyboardInterrupt:
            print("\nStopping game state monitor.")
            break
        except Exception as e:
            print(f"\n--- ERROR IN MAIN LOOP ---")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            print(f"--------------------------")
            # Don't reset state here, allow potential recovery on next fetch
            time.sleep(1.0)

        processing_time = time.monotonic() - start_time
        sleep_time = max(0, POLLING_INTERVAL_SECONDS - processing_time)
        time.sleep(sleep_time)
