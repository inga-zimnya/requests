import time
import math
import random
from typing import List, Tuple, Dict, Optional

from PlayerMovement import PlayerMovementController
from main import fetch_game_state, ParsedGameState


def get_player_target_direction(player_pos: Tuple[float, float], targets: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Returns normalized direction vector toward nearest target"""
    if not targets:
        return (0.0, 0.0)
    closest = min(targets, key=lambda p: (p[0] - player_pos[0]) ** 2 + (p[1] - player_pos[1]) ** 2)
    dx = closest[0] - player_pos[0]
    dy = closest[1] - player_pos[1]
    dist = math.hypot(dx, dy) or 1.0
    return (dx / dist, dy / dist)


def decide_direction(player_pos: Tuple[float, float],
                     pickups: List[Tuple[float, float]],
                     enemies: List[Tuple[float, float]],
                     explore_weight: float = 0.1) -> Tuple[float, float]:
    """Combines target seeking and random movement for exploration"""
    dx, dy = 0.0, 0.0

    if pickups:
        px, py = get_player_target_direction(player_pos, pickups)
        dx += px
        dy += py

    if enemies:
        ex, ey = get_player_target_direction(player_pos, enemies)
        dx += ex
        dy += ey

    # Add small random exploration
    dx += (random.random() - 0.5) * 2 * explore_weight
    dy += (random.random() - 0.5) * 2 * explore_weight

    mag = math.hypot(dx, dy)
    return (dx / mag, dy / mag) if mag > 0 else (0.0, 0.0)


def run_agent_loop(agent_index: int = 1, server_url: str = "http://127.0.0.1:15702/"):
    print(f"🎮 Starting agent control loop for player {agent_index}...")
    controller = PlayerMovementController(server_url=server_url, player_index=agent_index)

    try:
        while True:
            state: Optional[ParsedGameState] = fetch_game_state()
            if not state:
                print("⚠️ No game state, retrying...")
                time.sleep(0.2)
                continue

            if agent_index not in state.players:
                print(f"⚠️ Agent player {agent_index} not found")
                time.sleep(0.2)
                continue

            agent_data = state.players[agent_index]
            agent_pos = agent_data["position"]

            # Treat all other players as enemies
            enemy_positions = [p["position"] for pid, p in state.players.items() if pid != agent_index]
            pickups = state.pickup_positions()

            direction = decide_direction(agent_pos, pickups, enemy_positions)
            controller.move_analog(*direction)

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("🛑 Agent stopped by user.")
        controller.stop()
    except Exception as e:
        print(f"❌ Error during agent loop: {e}")
        controller.stop()


def main():
    run_agent_loop(agent_index=1)


if __name__ == "__main__":
    main()
