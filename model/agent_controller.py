import time
import math
import random
from typing import Tuple, Optional

import numpy as np

from model.PlayerMovement import PlayerMovementController
from model.PlayerInputController import PlayerInputController
from main import fetch_game_state, ParsedGameState
from StateProcessor import StateProcessor  # Your class from earlier


# === Reward Calculation ===
def compute_reward(prev_state: ParsedGameState, curr_state: ParsedGameState, player_id: int) -> float:
    """Reward based on progress toward pickups and damage events"""
    if player_id not in prev_state.players or player_id not in curr_state.players:
        return 0.0

    prev_player = prev_state.players[player_id]
    curr_player = curr_state.players[player_id]

    reward = 0.0

    # --- Pickup proximity ---
    prev_pickups = prev_state.pickup_positions()
    curr_pickups = curr_state.pickup_positions()

    if prev_pickups and curr_pickups:
        prev_dist = min([np.linalg.norm(np.subtract(prev_player["position"], p)) for p in prev_pickups])
        curr_dist = min([np.linalg.norm(np.subtract(curr_player["position"], p)) for p in curr_pickups])
        if curr_dist < prev_dist:
            reward += 0.5  # moved toward pickup

    # --- Damage dealt to others ---
    for pid, prev_enemy in prev_state.players.items():
        if pid != player_id and pid in curr_state.players:
            health_diff = prev_enemy['health'] - curr_state.players[pid]['health']
            if health_diff > 0:
                reward += health_diff * 0.2

    # --- Damage taken ---
    health_loss = prev_player['health'] - curr_player['health']
    if health_loss > 0:
        reward -= health_loss * 0.2

    return reward


# === Simple Heuristic Policy ===
def heuristic_policy(state_vector: np.ndarray) -> Tuple[float, float]:
    """Use a fixed policy that moves toward pickup/enemy"""
    # Extract recent frame (last 15 features)
    frame = state_vector[-15:]
    player_x, player_y = frame[0], frame[1]
    pickup_x, pickup_y = frame[8], frame[9]
    enemy_x, enemy_y = frame[11], frame[12]

    dx, dy = 0.0, 0.0

    if pickup_x > 0 or pickup_y > 0:
        dx += pickup_x - player_x
        dy += pickup_y - player_y

    if enemy_x > 0 or enemy_y > 0:
        dx += enemy_x - player_x
        dy += enemy_y - player_y

    dx += (random.random() - 0.5) * 0.2  # exploration noise
    dy += (random.random() - 0.5) * 0.2

    mag = math.hypot(dx, dy)
    return (dx / mag, dy / mag) if mag > 0 else (0.0, 0.0)


# === Main Agent Loop ===

def run_agent_loop(agent_index: int = 1, server_url: str = "http://127.0.0.1:15702/"):
    print(f"🤖 Agent started: player {agent_index}")
    controller = PlayerMovementController(server_url=server_url, player_index=agent_index)
    input_controller = PlayerInputController(server_url=server_url, player_index=agent_index)
    processor = StateProcessor()
    buffer = []

    prev_game_state: Optional[ParsedGameState] = None
    prev_state_vec = None
    last_action = None
    frame_count = 0

    try:
        while True:
            game_state = fetch_game_state()
            if not game_state or agent_index not in game_state.players:
                print("⏳ Waiting for valid game state...")
                time.sleep(0.2)
                continue

            current_state_vec = processor.process_state(game_state, agent_index)
            player = game_state.players[agent_index]
            player_pos = np.array(player["position"])

            if prev_game_state and prev_state_vec is not None and last_action is not None:
                reward = compute_reward(prev_game_state, game_state, agent_index)
                buffer.append((prev_state_vec, last_action, reward, current_state_vec, False))

            # === Movement ===
            action = heuristic_policy(current_state_vec)
            last_action = action
            prev_state_vec = current_state_vec
            prev_game_state = game_state
            controller.move_analog(*action)

            # === Input Heuristics ===
            # Footstep every 10 frames
            if frame_count % 10 == 0:
                input_controller.press_foot()

            # Try pickup if very close
            for pickup_pos in game_state.pickup_positions():
                if np.linalg.norm(np.subtract(pickup_pos, player_pos)) < 50:
                    input_controller.press_pickup()
                    break

            # Try shoot if enemy close
            for pid, enemy in game_state.players.items():
                if pid != agent_index:
                    enemy_pos = np.array(enemy["position"])
                    if np.linalg.norm(enemy_pos - player_pos) < 100:
                        input_controller.press_shoot()
                        break

            # Clear inputs to avoid persistent "pressed" state
            input_controller.clear_input()

            frame_count += 1
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        controller.stop()
        input_controller.clear_input()
        print("📦 Collected data samples:", len(buffer))

        # You can save buffer to disk or return for training


def main():
    run_agent_loop(agent_index=1)


if __name__ == "__main__":
    main()
