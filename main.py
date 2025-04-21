def fetch_game_state() -> Optional[ParsedGameState]:
    """Fetches and parses game state, optionally reading velocity from Rapier."""
    try:
        game_state_payload = {
            "id": 1, "jsonrpc": "2.0", "method": "bevy/query",
            "params": {
                "data": {"components": ["hotline_miami_like::ai::game_state::GameState"], "has": [], "option": []},
                "filter": {"with": [], "without": []}
            }
        }

        # --- MODIFIED QUERY ---
        players_payload = {
            "id": 2, "jsonrpc": "2.0", "method": "bevy/query",
            "params": {
                "data": {
                    "components": [
                        # List all components you *might* want
                        "hotline_miami_like::player::spawn::Player",
                        "bevy_transform::components::transform::Transform",
                        "hotline_miami_like::player::damagable::Damagable",
                        "bevy_rapier2d::dynamics::Velocity" # Still request it
                    ],
                    "has": [],
                    "option": ["entity"] # Request entity ID
                },
                "filter": {
                    # ONLY require the core Player component to be returned
                    "with": ["hotline_miami_like::player::spawn::Player"],
                    "without": []
                }
            }
        }
        # --- END OF MODIFIED QUERY ---


        game_state_response = requests.post(GAME_SERVER_URL, json=game_state_payload, timeout=1.0)
        game_state_response.raise_for_status()
        game_state_data = game_state_response.json()

        players_response = requests.post(GAME_SERVER_URL, json=players_payload, timeout=1.0)
        players_response.raise_for_status()
        players_data = players_response.json()

        # --- Parse GameState (No changes needed here) ---
        if not game_state_data.get("result"): return None
        game_state_entity = game_state_data["result"][0]
        if "hotline_miami_like::ai::game_state::GameState" not in game_state_entity.get("components", {}):
            print("Warning: GameState entity missing GameState component")
            return None
        raw_state = game_state_entity["components"]["hotline_miami_like::ai::game_state::GameState"]

        # --- Parse Players ---
        players = {}
        if players_data.get("result"):
            # print(f"DEBUG: Received {len(players_data['result'])} player entities from query.") # Optional debug print
            for entity_data in players_data["result"]:
                entity_id = entity_data.get("entity", {}).get("id", f"unknown_{time.time()}")
                components = entity_data.get("components", {})

                # We know Player component exists due to the filter, but check others
                if "hotline_miami_like::player::spawn::Player" not in components:
                     # This shouldn't happen with the filter, but good practice
                     # print(f"DEBUG: Skipping entity {entity_id} because Player component missing despite filter?")
                     continue

                try:
                    player_comp = components["hotline_miami_like::player::spawn::Player"]
                    transform = components.get("bevy_transform::components::transform::Transform",
                                               {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]})

                    # --- MODIFIED VELOCITY PARSING ---
                    velocity_x, velocity_y = 0.0, 0.0 # Default velocity
                    if "bevy_rapier2d::dynamics::Velocity" in components:
                        rapier_velocity_comp = components["bevy_rapier2d::dynamics::Velocity"]
                        velocity_vec = rapier_velocity_comp.get("linvel", [0.0, 0.0])
                        if isinstance(velocity_vec, (list, tuple)) and len(velocity_vec) >= 2:
                            try:
                                velocity_x = float(velocity_vec[0])
                                velocity_y = float(velocity_vec[1])
                            except (TypeError, ValueError):
                                # print(f"DEBUG: Could not parse linvel {velocity_vec} for player {entity_id}")
                                pass # Keep default 0,0
                        # else:
                            # print(f"DEBUG: Unexpected linvel format {velocity_vec} for player {entity_id}")
                    # else:
                        # print(f"DEBUG: Player {entity_id} missing Velocity component, using default (0,0).") # Optional debug print
                    # --- END OF MODIFIED VELOCITY PARSING ---


                    # --- Rotation Parsing (no changes needed) ---
                    rotation = transform.get("rotation", [0, 0, 0, 1])
                    rotation_angle = 0.0
                    try:
                         q2, q3 = float(rotation[2]), float(rotation[3])
                         if abs(q3) > 1e-9 or abs(q2) > 1e-9:
                            rotation_angle = 2 * math.atan2(q2, q3)
                    except (IndexError, TypeError, ValueError): pass

                    # --- Create PlayerState Dict (no changes needed) ---
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
                        "is_grounded": True, # Placeholder
                        "health": float(
                            components.get("hotline_miami_like::player::damagable::Damagable", {}).get("health", 100.0)
                        ),
                        "inventory": [], # Placeholder
                        "velocity": (velocity_x, velocity_y), # Use parsed or default velocity
                        "animation_state": None, # Placeholder
                        "calculated_velocity": None,
                        "calculated_speed": None
                    }

                except Exception as e:
                    print(f"\n--- Error parsing player {entity_id} ---")
                    print(f"Error: {str(e)}")
                    print(f"Components data: {json.dumps(components, indent=2)}")
                    print("-" * 20)
                    continue # Skip this player on error

        # --- Validate state structure (No changes needed) ---
        if not isinstance(raw_state.get("state"), list) or len(raw_state["state"]) != TOTAL_LAYERS:
             print(f"Error: Invalid 'state' structure received: {raw_state.get('state')}")
             return None
        if not all(isinstance(layer, list) for layer in raw_state["state"]):
             print(f"Error: Layers within 'state' are not all lists.")
             return None

        # --- Return ParsedGameState (No changes needed) ---
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

    # --- Exception Handling (No changes needed) ---
    except requests.exceptions.RequestException: pass
    except json.JSONDecodeError as e: print(f"Invalid JSON response: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in fetch_game_state: {str(e)}")
        traceback.print_exc()

    return None

# ==============================================================================
# == NO CHANGES ARE NEEDED BELOW THIS LINE for this specific fix             ==
# == The rest of the code (update_calculated_velocity, display functions,   ==
# == main loop) should work correctly with the output of this revised fetch ==
# ==============================================================================

# --- (Keep update_calculated_velocity function as is) ---
def update_calculated_velocity(current_state: ParsedGameState,
                               previous_state_data: Optional[ParsedGameState], # Renamed for clarity
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
                calc_speed_sq = calc_vx**2 + calc_vy**2
                calc_speed = math.sqrt(calc_speed_sq) if calc_speed_sq >= 0 else 0.0


                current_player['calculated_velocity'] = (calc_vx, calc_vy)
                current_player['calculated_speed'] = calc_speed
            else:
                current_player['calculated_velocity'] = None
                current_player['calculated_speed'] = None
        else:
            current_player['calculated_velocity'] = None
            current_player['calculated_speed'] = None


# --- (Keep display functions as is) ---
def print_player_details(player: PlayerState, player_id: Any):
    """Prints detailed information about a player."""
    reported_velocity = player.get('velocity', (0.0, 0.0))
    reported_speed = math.sqrt(reported_velocity[0]**2 + reported_velocity[1]**2)

    print(f"\nPlayer ID {player_id} ({player.get('character', PlayerCharacter.ORANGE).value}):")
    pos = player.get('position', ('N/A', 'N/A'))
    rot_deg = math.degrees(player.get('rotation', 0.0))
    pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})" if isinstance(pos, tuple) and len(pos) == 2 else f"{pos}"
    print(f"  Position: {pos_str}")
    print(f"  Rotation: {rot_deg:.1f}°")
    print(f"  Device: {player.get('device', PlayerDevice.KEYBOARD).value}")
    state_str = f"{'SHOOTING ' if player.get('is_shooting') else ''}" \
                f"{'KICKING ' if player.get('is_kicking') else ''}" \
                f"{'MOVING' if player.get('is_moving') else ''}"
    print(f"  State: {state_str.strip() if state_str.strip() else 'IDLE'}")
    print(f"  Health: {player.get('health', 0.0):.1f}")
    print(f"  Reported Velocity (Rapier): ({reported_velocity[0]:.1f}, {reported_velocity[1]:.1f})")
    print(f"  Reported Speed (Rapier): {reported_speed:.1f} units/sec")

    if player.get('calculated_velocity') is not None and player.get('calculated_speed') is not None:
        calc_vel = player['calculated_velocity']
        calc_speed = player['calculated_speed']
        print(f"  Calculated Velocity (Pos Δ): ({calc_vel[0]:.1f}, {calc_vel[1]:.1f}) (Avg)")
        print(f"  Calculated Speed (Pos Δ): {calc_speed:.1f} units/sec (Avg)")
    else:
         print(f"  Calculated Velocity (Pos Δ): N/A")

    inv = player.get('inventory', [])
    if inv:
        print(f"  Inventory: {', '.join(map(str, inv))}")


def show_ascii_map(game_state: ParsedGameState,
                   view_center: Optional[Tuple[float, float]] = None,
                   view_width: float = 160.0,
                   view_height: float = 112.0,
                   cell_size: float = 4.0) -> str:
    """Generates ASCII representation of the game map."""
    if view_center is None:
        view_center = (DEBUG_UI_START_POS[0] + MAP_WIDTH_UNITS / 2,
                       DEBUG_UI_START_POS[1] - MAP_HEIGHT_UNITS / 2)

    cells_x = math.ceil(view_width / cell_size)
    cells_y = math.ceil(view_height / cell_size)

    if not game_state or not hasattr(game_state, 'entities') or not game_state.entities or \
       not hasattr(game_state, 'walls') or not game_state.walls or \
       not hasattr(game_state, 'floors') or not game_state.floors:
        return "Error: Invalid game state layers.\n"
    if not isinstance(game_state.entities[0], list) or \
       not isinstance(game_state.walls[0], list) or \
       not isinstance(game_state.floors[0], list):
         return "Error: Map layers are not lists of lists.\n"

    map_pixel_height = len(game_state.entities)
    map_pixel_width = len(game_state.entities[0])

    if MAP_WIDTH_UNITS <= 0 or MAP_HEIGHT_UNITS <= 0 or map_pixel_width <= 0 or map_pixel_height <= 0:
         return "Error: Invalid map dimensions.\n"

    def world_to_array(x: float, y: float) -> Tuple[int, int]:
        arr_x = int((x - DEBUG_UI_START_POS[0]) * map_pixel_width / MAP_WIDTH_UNITS)
        arr_y = int((DEBUG_UI_START_POS[1] - y) * map_pixel_height / MAP_HEIGHT_UNITS)
        return (
            max(0, min(map_pixel_width - 1, arr_x)),
            max(0, min(map_pixel_height - 1, arr_y))
        )

    map_str_builder = []
    for row_idx in range(cells_y):
        world_y = (view_center[1] + view_height / 2.0) - (row_idx + 0.5) * cell_size
        row_str = ""
        for col_idx in range(cells_x):
            world_x = (view_center[0] - view_width / 2.0) + (col_idx + 0.5) * cell_size
            arr_x, arr_y = world_to_array(world_x, world_y)

            try:
                 entity = game_state.entities[arr_y][arr_x]
                 wall = game_state.walls[arr_y][arr_x]
                 floor = game_state.floors[arr_y][arr_x]
            except IndexError:
                 entity, wall, floor = '?', '?', '?'

            char_to_add = '?'
            if entity != 'Empty':
                char_to_add = SYMBOLS.get(entity, '?')
            elif wall != 'Empty':
                char_to_add = SYMBOLS.get(wall, '?')
            else:
                char_to_add = SYMBOLS.get(floor, '?')
            row_str += char_to_add
        map_str_builder.append(row_str)

    return "\n".join(map_str_builder) + "\n"


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


# --- (Keep main loop as is) ---
if __name__ == "__main__":
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
                sorted_player_ids = sorted(current_game_state.players.keys())
                for player_id in sorted_player_ids:
                    player_data = current_game_state.players[player_id]
                    print_player_details(player_data, player_id)

            print("\n=== Full Map View ===")
            print(show_ascii_map(
                current_game_state,
                view_width=MAP_WIDTH_UNITS,
                view_height=MAP_HEIGHT_UNITS,
                cell_size=ONE_PIXEL_DISTANCE * 2
            ))

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