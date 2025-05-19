import time
import math
from main import fetch_game_state
from model.PlayerMovement import PlayerMovementController
from model.PlayerInputController import PlayerInputController

GAME_SERVER_URL = "http://127.0.0.1:15702/"
AGENT_ID = 1
MOVE_DURATION = 0.5
WAIT_AFTER_PICKUP = 1.0


def pickup_loop():
    print(f"🎮 Agent {AGENT_ID} trying to move toward nearest pickup...")

    movement = PlayerMovementController(GAME_SERVER_URL, AGENT_ID)
    input_ctrl = PlayerInputController(GAME_SERVER_URL, AGENT_ID)

    try:
        while True:
            game_state = fetch_game_state()
            if not game_state or AGENT_ID not in game_state.players:
                print("⚠️ No valid game state. Retrying...")
                time.sleep(0.5)
                continue

            player = game_state.players[AGENT_ID]
            player_pos = player["position"]
            inventory_before = set(player.get("inventory", []))
            pickup_positions = game_state.pickup_positions()
            if not pickup_positions:
                print("🔍 No pickups found.")
                time.sleep(1.0)
                continue

            target = min(pickup_positions, key=lambda p: math.dist(p, player_pos))
            dx = target[0] - player_pos[0]
            dy = target[1] - player_pos[1]
            distance = math.hypot(dx, dy)

            if distance < 0.2:
                print("📍 Close enough to pickup, not moving.")
            else:
                norm_dx, norm_dy = dx / distance, dy / distance  # ✅ DO NOT FLIP dy
                print(f"🚶 Moving toward pickup: dx={norm_dx:.3f}, dy={norm_dy:.3f}")
                movement.move_analog(norm_dx, norm_dy)
                time.sleep(MOVE_DURATION)
                movement.stop()

            # Try pickup
            print("🛎️ Attempting pickup...")
            input_ctrl.press_pickup()
            time.sleep(WAIT_AFTER_PICKUP)

            new_state = fetch_game_state()
            if not new_state or AGENT_ID not in new_state.players:
                print("⚠️ Failed to fetch updated state.")
                continue

            inventory_after = set(new_state.players[AGENT_ID].get("inventory", []))
            print(f"📦 Inventory before: {inventory_before}, after: {inventory_after}")

            if inventory_after != inventory_before:
                print("✅ Inventory changed. Pickup likely successful.")
                input_ctrl.clear_input()
            else:
                print("❌ No inventory change.")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("🛑 Interrupted. Stopping...")
    finally:
        movement.stop()
        input_ctrl.clear_input()


if __name__ == "__main__":
    pickup_loop()