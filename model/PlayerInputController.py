from main import fetch_game_state
import requests
from typing import Optional, Dict
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

class PlayerInputController:
    def __init__(self, server_url: str = "http://127.0.0.1:15702/", player_index: int = 1):
        self.server_url = server_url
        self.player_index = player_index

    def _send_input_state(self, input_state: Dict[str, bool]) -> bool:
        try:
            game_state = fetch_game_state()
            if game_state is None:
                print("⚠️ Could not fetch game state")
                return False

            if len(game_state.players) <= self.player_index:
                print(f"⚠️ Player index {self.player_index} out of range")
                return False

            player = game_state.players[self.player_index]
            entity_id = player["entity"]

            request = {
                "id": 4,
                "jsonrpc": "2.0",
                "method": "bevy/insert",
                "params": {
                    "entity": entity_id,
                    "components": {
                        "hotline_miami_like::player::input::PlayerInput": input_state
                    }
                }
            }

            resp = requests.post(self.server_url, json=request, timeout=1.0)
            resp.raise_for_status()
            print(f"📤 Sent PlayerInput: {input_state}. Response: {resp.json()}")
            return True

        except Exception as e:
            print(f"❌ Input command failed: {e}")
            return False

    def press_foot(self) -> bool:
        return self._send_input_state({
            "is_shoot_button_pressed": False,
            "is_shoot_button_just_pressed": False,
            "is_foot_button_just_pressed": True,
            "is_pickup_button_just_pressed": False,
            "is_shoot_button_just_released": False,
            "is_any_move_button_pressed": False,
        })

    def clear_input(self) -> bool:
        return self._send_input_state({
            "is_shoot_button_pressed": False,
            "is_shoot_button_just_pressed": False,
            "is_foot_button_just_pressed": False,
            "is_pickup_button_just_pressed": False,
            "is_shoot_button_just_released": False,
            "is_any_move_button_pressed": False,
        })

    def press_shoot(self) -> bool:
        return self._send_input_state({
            "is_shoot_button_pressed": True,
            "is_shoot_button_just_pressed": True,
            "is_foot_button_just_pressed": False,
            "is_pickup_button_just_pressed": False,
            "is_shoot_button_just_released": False,
            "is_any_move_button_pressed": False,
        })

    def press_pickup(self) -> bool:
        return self._send_input_state({
            "is_shoot_button_pressed": False,
            "is_shoot_button_just_pressed": False,
            "is_foot_button_just_pressed": False,
            "is_pickup_button_just_pressed": True,
            "is_shoot_button_just_released": False,
            "is_any_move_button_pressed": False,
        })


