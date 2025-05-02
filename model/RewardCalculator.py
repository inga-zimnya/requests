class RewardCalculator:
    def __init__(self):
        self.last_positions = {}  # player_id: last_position
        self.last_health = {}  # player_id: last_health
        self.pickup_counts = {}  # player_id: pickup_count

    def calculate_reward(self, game_state: ParsedGameState, player_id: int) -> float:
        """Calculate reward for player's current state"""
        if game_state is None or player_id not in game_state.players:
            return 0.0

        player = game_state.players[player_id]
        reward = 0.0

        # 1. Movement reward (small reward for moving)
        if player['is_moving']:
            reward += 0.01

        # 2. Health change
        if player_id in self.last_health:
            health_diff = player['health'] - self.last_health[player_id]
            reward += health_diff * 0.1  # Scale health changes

        self.last_health[player_id] = player['health']

        # 3. Pickup collection (detect changes in inventory)
        current_pickups = len(player.get('inventory', []))
        if player_id in self.pickup_counts:
            if current_pickups > self.pickup_counts[player_id]:
                reward += 1.0  # Reward for collecting pickup
        self.pickup_counts[player_id] = current_pickups

        # 4. Distance to nearest pickup (encourage exploration)
        pickups = game_state.pickup_positions()
        if pickups:
            closest_pickup = min(pickups, key=lambda p: self._distance(player['position'], p))
            dist = self._distance(player['position'], closest_pickup)
            max_dist = max(game_state.map_size)
            reward += (1 - dist / max_dist) * 0.1  # Small reward for getting closer

        # 5. Survival reward
        reward += 0.02  # Small reward for staying alive

        return reward

    def _distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def reset(self, player_id: int):
        """Reset tracking for player"""
        self.last_positions.pop(player_id, None)
        self.last_health.pop(player_id, None)
        self.pickup_counts.pop(player_id, None)