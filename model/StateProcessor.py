import numpy as np
from typing import Dict, List, Tuple
from collections import deque

from parse_player import ParsedGameState  # Adjust import if needed


class StateProcessor:
    def __init__(self, map_size: Tuple[int, int] = (160, 112)):
        self.map_size = map_size
        self.history_length = 4  # Number of past states to remember
        self.state_history = deque(maxlen=self.history_length)
        self.base_speed = 200.0  # Default value used for speed normalization

    def process_state(self, game_state: ParsedGameState, player_id: int) -> np.ndarray:
        """Convert game state to feature vector for RL model"""
        if game_state is None:
            return np.zeros(self.get_state_shape())

        # Get player state
        player = game_state.players.get(player_id, None)
        if player is None:
            return np.zeros(self.get_state_shape())

        # 1. Player features
        player_features = [
            player['position'][0] / self.map_size[0],  # Normalized x
            player['position'][1] / self.map_size[1],  # Normalized y
            player['rotation'] / (2 * np.pi),          # Normalized rotation
            player['health'] / 100.0,                  # Normalized health
            float(player['is_shooting']),
            float(player['is_kicking']),
            float(player['is_moving']),
            player['calculated_speed'] / self.base_speed if player['calculated_speed'] is not None else 0.0
        ]

        # 2. Nearby objects
        nearby_features = self._get_nearby_features(game_state, player_id, player['position'])

        # Combine all features
        state_vector = np.concatenate([np.array(player_features), nearby_features])

        # Add to history
        self.state_history.append(state_vector)

        # Fill the history buffer if it's not yet full
        while len(self.state_history) < self.history_length:
            self.state_history.appendleft(np.zeros_like(state_vector))

        return np.concatenate(self.state_history)

    def _get_nearby_features(self, game_state: ParsedGameState, player_id: int, player_pos: Tuple[float, float]) -> np.ndarray:
        """Extract features about nearby objects (pickups, enemies, etc.)"""
        features = []

        # --- Closest pickup ---
        pickups = game_state.pickup_positions()
        if pickups:
            closest_pickup = min(pickups, key=lambda p: self._distance(player_pos, p))
            features.extend([
                closest_pickup[0] / self.map_size[0],
                closest_pickup[1] / self.map_size[1],
                self._distance(player_pos, closest_pickup) / max(self.map_size)
            ])
        else:
            features.extend([0.0, 0.0, 1.0])  # No pickup = max distance

        # --- Closest enemy player ---
        enemy_features = []
        for pid, pdata in game_state.players.items():
            if pid != player_id:
                enemy_features.extend([
                    pdata['position'][0] / self.map_size[0],
                    pdata['position'][1] / self.map_size[1],
                    self._distance(player_pos, pdata['position']) / max(self.map_size),
                    pdata['health'] / 100.0
                ])

        if not enemy_features:
            enemy_features = [0.0] * 4  # No enemy present

        features.extend(enemy_features[:4])  # Only use first enemy

        return np.array(features)

    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_state_shape(self) -> int:
        # Player features: 8
        # Nearby features: 3 (pickup) + 4 (enemy) = 7
        # Total per frame: 15
        # With history: 15 * history_length
        return 15 * self.history_length

    def reset(self):
        self.state_history.clear()
