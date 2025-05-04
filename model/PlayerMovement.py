import random
import time
from typing import Optional, List, Tuple, Dict, Any
import requests

from parse_player import fetch_game_state  # Make sure this import is correct


class PlayerMovementController:
    def __init__(self, server_url: str = "http://127.0.0.1:15702/", player_index: int = 1):
        self.server_url = server_url
        self.player_index = player_index
        self.base_speed = 200.0  # Matches Rust default
        self.current_direction = [0.0, 0.0]
        self.is_moving = False

    def _send_movement_command(self, direction: List[float], speed: Optional[float] = None) -> bool:
        """Send movement command to server with proper error handling"""
        speed = speed if speed is not None else self.base_speed

        try:
            game_state = fetch_game_state()
            if game_state is None:
                print("⚠️ Could not fetch game state")
                return False

            if len(game_state.players) <= self.player_index:
                print(f"⚠️ Player index {self.player_index} out of range (total players: {len(game_state.players)})")
                return False

            player = game_state.players[self.player_index]
            entity_id = player["entity"]

            request = {
                "id": 3,
                "jsonrpc": "2.0",
                "method": "bevy/insert",
                "params": {
                    "entity": entity_id,
                    "components": {
                        "hotline_miami_like::player::movement::Movement": {
                            "norm_direction": direction,
                            "speed": speed
                        }
                    }
                }
            }

            resp = requests.post(self.server_url, json=request, timeout=1.0)
            resp.raise_for_status()
            self.current_direction = direction
            self.is_moving = any(d != 0 for d in direction)
            return True

        except Exception as e:
            print(f"❌ Movement command failed: {e}")
            return False

    # Movement methods
    def move_up(self, speed: Optional[float] = None) -> bool:
        print("🔼 Moving UP")
        return self._send_movement_command([0.0, 1.0], speed)

    def move_down(self, speed: Optional[float] = None) -> bool:
        print("🔽 Moving DOWN")
        return self._send_movement_command([0.0, -1.0], speed)

    def move_left(self, speed: Optional[float] = None) -> bool:
        print("◀️ Moving LEFT")
        return self._send_movement_command([-1.0, 0.0], speed)

    def move_right(self, speed: Optional[float] = None) -> bool:
        print("▶️ Moving RIGHT")
        return self._send_movement_command([1.0, 0.0], speed)

    def stop(self) -> bool:
        print("🛑 STOPPING")
        return self._send_movement_command([0.0, 0.0], 0.0)

    def move_analog(self, stick_x: float, stick_y: float, speed: Optional[float] = None) -> bool:
        """Gamepad-style non-normalized movement"""
        stick_x = max(-1.0, min(1.0, stick_x))
        stick_y = max(-1.0, min(1.0, stick_y))
        print(f"🎮 Analog move: X={stick_x:.2f}, Y={stick_y:.2f}")
        return self._send_movement_command([stick_x, stick_y], speed)

    def _get_player_position(self) -> Optional[Dict[str, float]]:
        """Get player position with proper error handling"""
        try:
            game_state = fetch_game_state()
            if not game_state:
                print("⚠️ No game state available")
                return None

            if len(game_state.players) <= self.player_index:
                print(f"⚠️ Player index {self.player_index} out of range")
                return None

            player = game_state.players[self.player_index]

            # Handle position as tuple (which is what your structure shows)
            if hasattr(player, "position") and isinstance(player.position, tuple):
                x, y = player.position
                return {"x": float(x), "y": float(y)}
            elif isinstance(player.get("position"), tuple):
                x, y = player["position"]
                return {"x": float(x), "y": float(y)}

            # Fallback for other formats
            if hasattr(player, "x") and hasattr(player, "y"):
                return {"x": float(player.x), "y": float(player.y)}
            elif isinstance(player, dict) and "x" in player and "y" in player:
                return {"x": float(player["x"]), "y": float(player["y"])}

            print(f"⚠️ Could not determine position format. Player data: {player}")
            return None

        except Exception as e:
            print(f"❌ Position fetch failed: {e}")
            return None


def main():
    print("🚀 Starting Player Movement Demo")
    controller = PlayerMovementController()

    try:
        # 1. Basic movement demo
        print("\n=== BASIC MOVEMENT ===")
        for direction in [controller.move_right, controller.move_down,
                          controller.move_left, controller.move_up]:
            direction()
            time.sleep(1)
            controller.stop()
            time.sleep(0.5)

        # 2. Analog movement demo
        print("\n=== ANALOG MOVEMENT ===")
        for x, y in [(0.5, 0), (0, 0.7), (-0.3, -0.3), (0.9, 0.1)]:
            controller.move_analog(x, y)
            time.sleep(1)
        controller.stop()

        # 3. Position tracking demo
        print("\n=== POSITION TRACKING ===")
        for _ in range(3):
            pos = controller._get_player_position()
            if pos:
                print(f"📍 Current position: X={pos['x']:.1f}, Y={pos['y']:.1f}")
            controller.move_right(100.0)
            time.sleep(0.5)
            controller.stop()
            time.sleep(0.5)

        print("\n✅ Demo complete!")

    except KeyboardInterrupt:
        print("\n🛑 User stopped the demo")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()