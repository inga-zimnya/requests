import requests
import math
import time
from typing import Optional, Tuple
from main import fetch_game_state  # Assumes this returns ParsedGameState-like structure

GAME_SERVER_URL = "http://127.0.0.1:15702/"
POLL_INTERVAL = 1.0


def angle_to_quaternion(angle_rad: float) -> list:
    """Return quaternion as [x, y, z, w]"""
    x, y = 0.0, 0.0
    z = math.sin(angle_rad / 2)
    w = math.cos(angle_rad / 2)
    return [x, y, z, w]



def get_nearest_enemy_angle(player: dict, all_players: dict, player_id: int) -> Optional[float]:
    px, py = player['position']
    min_dist = float('inf')
    nearest_angle = None

    for pid, p in all_players.items():
        if pid == player_id:
            continue
        ex, ey = p['position']
        dist = math.hypot(ex - px, ey - py)
        if dist < min_dist:
            min_dist = dist
            nearest_angle = math.atan2(ey - py, ex - px)

    return nearest_angle


def is_facing_enemy(player: dict, enemy_pos: Tuple[float, float]) -> bool:
    """Check if player is facing an enemy"""
    px, py = player['position']
    ex, ey = enemy_pos
    angle_to_enemy = math.atan2(ey - py, ex - px)
    return abs(angle_to_enemy - player['rotation']) < math.pi / 6  # narrower 30° cone


def reward_for_facing(curr_state, agent_id) -> float:
    """Reward if the agent is facing an enemy"""
    if agent_id not in curr_state.players:
        return 0.0

    agent = curr_state.players[agent_id]
    px, py = agent['position']
    angle = agent['rotation']

    for pid, p in curr_state.players.items():
        if pid == agent_id:
            continue
        ex, ey = p['position']
        angle_to_enemy = math.atan2(ey - py, ex - px)
        if abs(angle_to_enemy - angle) < math.pi / 6:
            return 0.5  # Example reward value

    return 0.0


def update_ai_component():
    while True:
        try:
            game_state = fetch_game_state()

            if game_state is None or len(game_state.players) < 2:
                print("⚠️ Waiting for valid game state...")
                time.sleep(POLL_INTERVAL)
                continue

            player_id = 1
            player = game_state.players[player_id]
            entity_id = player["entity"]

            # Movement settings
            direction = [0.0, -1.0]  # Move down
            speed = 50.0

            # Rotation toward nearest enemy
            angle = get_nearest_enemy_angle(player, game_state.players, player_id)
            insert_components = {
                "hotline_miami_like::player::movement::Movement": {
                    "direction": direction,
                    "speed": speed
                }
            }

            if angle is not None:
                quat = angle_to_quaternion(angle)
                insert_components["bevy_transform::components::transform::Transform"] = {
                    "rotation": quat
                }

            # Send insert request to Bevy
            insert_request = {
                "id": 3,
                "jsonrpc": "2.0",
                "method": "bevy/insert",
                "params": {
                    "entity": entity_id,
                    "components": insert_components
                }
            }

            resp = requests.post(GAME_SERVER_URL, json=insert_request, timeout=1.0)
            resp.raise_for_status()
            response_data = resp.json()
            print(f"✅ Server response: {response_data}")

            # Reward if player is facing enemy
            reward = reward_for_facing(game_state, player_id)
            if reward > 0:
                print(f"🎯 Facing enemy — reward: {reward:.2f}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            time.sleep(POLL_INTERVAL)
            continue
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    update_ai_component()
