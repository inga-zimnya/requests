import os
import time
from typing import Optional

from actions import ActionClient
from parse_player import ParsedGameState, fetch_game_state


class RLController:
    def __init__(self, server_url: str = "http://127.0.0.1:15702/"):
        self.server_url = server_url
        self.state_processor = StateProcessor()
        self.reward_calculator = RewardCalculator()
        self.agents: Dict[int, RLAgent] = {}  # player_id: agent
        self.action_client = ActionClient(server_url)

    def update(self):
        """Main update loop called from your game loop"""
        game_state = fetch_game_state()
        if game_state is None:
            return

        # Update all agents
        for player_id in game_state.players.keys():
            if player_id not in self.agents:
                self._init_agent(player_id)

            agent = self.agents[player_id]

            # Process state
            state = self.state_processor.process_state(game_state, player_id)

            # Get action
            action_probs = agent.get_action(state)

            # Execute action
            self._execute_action(player_id, action_probs)

            # Calculate reward
            reward = self.reward_calculator.calculate_reward(game_state, player_id)

            # Get next state
            next_state = self.state_processor.process_state(fetch_game_state(), player_id)

            # Store experience
            agent.store_transition(state, action_probs, reward, next_state, False)

            # Train agent
            agent.update_policy()

    def _init_agent(self, player_id: int):
        """Initialize new agent for player"""
        self.agents[player_id] = RLAgent(player_id, self.state_processor)
        self.reward_calculator.reset(player_id)

    def _execute_action(self, player_id: int, action_probs: np.ndarray):
        """Convert action probabilities to game actions"""
        # Thresholds for action execution
        move_threshold = 0.7
        shoot_threshold = 0.8

        # Movement
        move_x, move_y = 0.0, 0.0

        if action_probs[0] > move_threshold:  # Up
            move_y += action_probs[0]
        if action_probs[1] > move_threshold:  # Down
            move_y -= action_probs[1]
        if action_probs[2] > move_threshold:  # Left
            move_x -= action_probs[2]
        if action_probs[3] > move_threshold:  # Right
            move_x += action_probs[3]

        # Normalize movement vector
        move_mag = np.sqrt(move_x ** 2 + move_y ** 2)
        if move_mag > 1.0:
            move_x /= move_mag
            move_y /= move_mag

        # Shooting
        is_shooting = action_probs[4] > shoot_threshold

        # Send movement command
        self.action_client.move_analog(player_id, move_x, move_y)

        # Send shoot command if needed
        if is_shooting:
            self.action_client.shoot(player_id)


if __name__ == "__main__":
    action_client = ActionClient(GAME_SERVER_URL)
    rl_controller = RLController(GAME_SERVER_URL)

    previous_game_state: Optional[ParsedGameState] = None
    previous_time: Optional[float] = None
    first_run = True

    while True:
        start_time = time.monotonic()

        try:
            current_game_state = fetch_game_state()
            current_time = time.monotonic()

            if current_game_state is None:
                if first_run:
                    print("\nWaiting for game state from server...", end="\r")
                time.sleep(0.5)
                continue
            elif first_run:
                print("Game state received. Starting updates...")
                first_run = False

            delta_time = (current_time - previous_time) if previous_time is not None else 0.0
            update_calculated_velocity(current_game_state, previous_game_state, delta_time)

            # Update RL controller
            rl_controller.update()

            os.system('cls' if os.name == 'nt' else 'clear')
            print_map_legend()

            print("\n=== Game State Update ===")
            print(f"Active players: {len(current_game_state.players)}")
            if delta_time > 0:
                print(f"Time since last update: {delta_time:.3f}s")

            if not current_game_state.players:
                print("\nNo active players detected.")
            else:
                sorted_player_ids = sorted(current_game_state.players.keys())
                for player_id in sorted_player_ids:
                    player_data = current_game_state.players[player_id]
                    print_player_details(player_data, player_id)

            print("\n=== Full Map View ===")
            print(show_ascii_map(current_game_state))

            previous_game_state = current_game_state
            previous_time = current_time

        except KeyboardInterrupt:
            print("\nStopping game state monitor.")
            break
        except Exception as e:
            print(f"\n--- ERROR IN MAIN LOOP ---")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            time.sleep(1.0)

        processing_time = time.monotonic() - start_time
        sleep_time = max(0, POLLING_INTERVAL_SECONDS - processing_time)
        time.sleep(sleep_time)