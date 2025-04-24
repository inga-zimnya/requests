# example_run.py
import time
from actions import ActionClient
from main import (fetch_game_state, PlayerCharacter)
#, pickup_positions)

GAME_SERVER = "http://127.0.0.1:15702/"


def main():
    # 1) grab a fresh game state
    state = fetch_game_state()
    if state is None:
        print("❌ No game state available.")
        return

    # 2) find the 'Lemon' player
    lemon_id = None
    for pid, pdata in state.players.items():
        if pdata["character"] == PlayerCharacter.LEMON:
            lemon_id = pid
            break

    if lemon_id is None:
        print("❌ Could not find any Lemon player in state.players.")
        return

    print(f"➡️ Found Lemon with entity ID = {lemon_id}")

    # 3) initialize your action client
    client = ActionClient(GAME_SERVER)

    # 4) register all current pickup positions so we get stable item_ids
    """
    pickup_positions = state.pickup_positions
    client.update_pickup_ids()
    print("Pickup position → item_id map:")
    for pos, iid in client._position_to_id.items():
        print(f"  {pos} → {iid}")
    """
    # 5) example: move Lemon towards the center of the map
    target = (50.0, 50.0)
    ok = client.move_to(lemon_id, target, speed=1.2)
    print(f"move_to {target} → {ok}")

    # 6) example: shoot at first other player in the list
    enemies = [pid for pid in state.players if pid != lemon_id]
    if enemies:
        enemy_id = enemies[0]
        enemy_pos = state.players[enemy_id]["position"]
        ok = client.shoot_at(lemon_id, enemy_pos, force=2.0)
        print(f"shoot_at player {enemy_id} at {enemy_pos} → {ok}")
    else:
        print("ℹ️ No enemies to shoot at.")

    # 7) example: pick up every pickup on the ground
    """
    for pos in pickup_positions:
        ok = client.send_action(
            lemon_id,
            "pickup",
            {"position": list(pos)}
        )
        print(f"pickup at {pos} → {ok}")
    """


if __name__ == "__main__":
    # small delay so your server can spin up
    time.sleep(0.2)
    main()
