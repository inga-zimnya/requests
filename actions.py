from typing import Tuple
import requests
from typing import List, Tuple, Dict, Optional


class ActionClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._position_to_id: Dict[Tuple[int, int], int] = {}  # Maps (x,y) positions to item_ids
        self._next_id = 1  # Starting ID (adjust if Rust uses specific ranges)

    """
    def update_pickup_ids(self) -> None:
        from main import ParsedGameState
        parse_game_state = ParsedGameState()
        positions = parse_game_state.pickup_positions()

        self._position_to_id.clear()
        for pos in positions:
            if pos not in self._position_to_id:
                self._position_to_id[pos] = self._next_id
                self._next_id += 1

    """
    def send_action(self, entity_id: int, action_type: str, action_data: Optional[dict] = None) -> bool:
        """Modified to handle position-based pickups"""
        if action_data is None:
            action_data = {}
            #self.update_pickup_ids()

        # Convert position to item_id if needed
        if action_type == 'pickup' and 'position' in action_data:
            pos = tuple(action_data['position'])
            if pos in self._position_to_id:
                action_data['item_id'] = self._position_to_id[pos]
            else:
                raise ValueError(f"No item_id found for position {pos}")

        action_map = {
            'move': {
                'method': 'bevy/move_entity',
                'params': {
                    'entity': entity_id,
                    'direction': action_data.get('direction', [0, 0]),
                    'speed': action_data.get('speed', 1.0)
                }
            },
            'shoot': {
                'method': 'bevy/shoot',
                'params': {
                    'entity': entity_id,
                    'direction': action_data.get('direction', [0, 1]),
                    'force': action_data.get('force', 1.0)
                }
            },
            'kick': {
                'method': 'bevy/kick',
                'params': {
                    'entity': entity_id,
                    'direction': action_data.get('direction', [0, 1])
                }
            },
            'pickup': {
                'method': 'bevy/pickup_item',
                'params': {
                    'entity': entity_id,
                    'item_id': action_data['item_id']  # Required
                }
            },
            'reload': {
                'method': 'bevy/reload_weapon',
                'params': {
                    'entity': entity_id
                }
            }
        }

        if action_type not in action_map:
            raise ValueError(f"Invalid action_type '{action_type}'")

        if action_type == 'pickup' and 'item_id' not in action_data:
            raise ValueError("Pickup requires either item_id or position in action_data")

        action_config = action_map[action_type]
        try:
            response = requests.post(
                f"{self.base_url}{action_config['method']}",
                json=action_config['params'],
                timeout=1.0
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def move_to(self, entity_id: int, target_pos: Tuple[float, float], speed: float = 1.0):
        from main import fetch_game_state
        current_state = fetch_game_state()
        if not current_state or entity_id not in current_state.players:
            return False

        current_pos = current_state.players[entity_id]['position']
        direction = [
            target_pos[0] - current_pos[0],
            target_pos[1] - current_pos[1]
        ]
        return self.send_action(entity_id, 'move', {
            'direction': direction,
            'speed': speed
        })

    def shoot_at(self, entity_id: int, target_pos: Tuple[float, float], force: float = 1.0):
        from main import fetch_game_state
        current_state = fetch_game_state()
        if not current_state or entity_id not in current_state.players:
            return False

        current_pos = current_state.players[entity_id]['position']
        direction = [
            target_pos[0] - current_pos[0],
            target_pos[1] - current_pos[1]
        ]
        return self.send_action(entity_id, 'shoot', {
            'direction': direction,
            'force': force
        })

    def kick_at(self, entity_id: int, target_pos: Tuple[float, float]):
        from main import fetch_game_state
        current_state = fetch_game_state()
        if not current_state or entity_id not in current_state.players:
            return False

        current_pos = current_state.players[entity_id]['position']
        direction = [
            target_pos[0] - current_pos[0],
            target_pos[1] - current_pos[1]
        ]
        return self.send_action(entity_id, 'kick', {
            'direction': direction
        })
