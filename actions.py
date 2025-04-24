from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass
import math


@dataclass
class GameState:
    players: Dict[int, Dict[str, Tuple[float, float]]]
    pickups: List[Tuple[int, int]]


class ActionClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._position_to_id: Dict[Tuple[int, int], int] = {}
        self._next_id = 1

    def update_pickup_ids(self, positions: List[Tuple[int, int]]) -> None:
        """Update the position-to-ID mapping with current pickup positions"""
        self._position_to_id.clear()
        for pos in positions:
            if pos not in self._position_to_id:
                self._position_to_id[pos] = self._next_id
                self._next_id += 1

    def _calculate_direction(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[float]:
        """Calculate normalized direction vector between two points"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(math.sqrt(dx * dx + dy * dy), 0.0001)  # Avoid division by zero
        return [dx / length, dy / length]

    def send_action(self, entity_id: int, action_type: str, action_data: Optional[dict] = None) -> bool:
        """Send an action to the game server"""
        if action_data is None:
            action_data = {}

        # Handle position-based pickups
        if action_type == 'pickup' and 'position' in action_data:
            pos = tuple(action_data['position'])
            if pos in self._position_to_id:
                action_data['item_id'] = self._position_to_id[pos]
            else:
                raise ValueError(f"No pickup at position {pos}")

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
                    'item_id': action_data['item_id']
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
            raise ValueError(f"Invalid action: {action_type}")

        if action_type == 'pickup' and 'item_id' not in action_data:
            raise ValueError("Pickup requires item_id or position")

        try:
            response = requests.post(
                f"{self.base_url}{action_map[action_type]['method']}",
                json=action_map[action_type]['params'],
                timeout=1.0
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def move_to(self, entity_id: int, target_pos: Tuple[float, float], speed: float = 1.0) -> bool:
        """Move entity to target position"""
        if not hasattr(self, '_game_state'):
            raise RuntimeError("Game state not set. Call set_game_state() first")

        if entity_id not in self._game_state.players:
            return False

        current_pos = self._game_state.players[entity_id]['position']
        direction = self._calculate_direction(current_pos, target_pos)
        return self.send_action(entity_id, 'move', {
            'direction': direction,
            'speed': speed
        })

    def shoot_at(self, entity_id: int, target_pos: Tuple[float, float], force: float = 1.0) -> bool:
        """Shoot towards target position"""
        if not hasattr(self, '_game_state'):
            raise RuntimeError("Game state not set. Call set_game_state() first")

        if entity_id not in self._game_state.players:
            return False

        current_pos = self._game_state.players[entity_id]['position']
        direction = self._calculate_direction(current_pos, target_pos)
        return self.send_action(entity_id, 'shoot', {
            'direction': direction,
            'force': force
        })

    def kick_at(self, entity_id: int, target_pos: Tuple[float, float]) -> bool:
        """Kick towards target position"""
        if not hasattr(self, '_game_state'):
            raise RuntimeError("Game state not set. Call set_game_state() first")

        if entity_id not in self._game_state.players:
            return False

        current_pos = self._game_state.players[entity_id]['position']
        direction = self._calculate_direction(current_pos, target_pos)
        return self.send_action(entity_id, 'kick', {
            'direction': direction
        })

    def set_game_state(self, state: GameState) -> None:
        """Update the client's game state reference"""
        self._game_state = state
        self.update_pickup_ids(state.pickups)