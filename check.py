# -*- coding: utf-8 -*-
# ... (keep all your existing imports)

# Add these new imports at the top
import numpy as np
from StateProcessor import StateProcessor, RewardCalculator, RLAgent, RLController
from RewardCalculator import RewardCalculator
from MultiAgentMLP import MultiAgentMLP
from RLController import RLController

def main():
    # Initialize components
    action_client = ActionClient(GAME_SERVER_URL)
    state_processor = StateProcessor()
    reward_calculator = RewardCalculator()

    # Initialize RL controller with your existing action client
    rl_controller = RLController(GAME_SERVER_URL)

    previous_game_state = None
    previous_time = None

    try:
        while True:
            start_time = time.monotonic()

            # 1. Fetch current game state
            current_game_state = fetch_game_state()
            current_time = time.monotonic()

            if current_game_state is None:
                time.sleep(0.1)
                continue

            # 2. Calculate time delta
            delta_time = current_time - previous_time if previous_time else 0.0

            # 3. Update calculated velocities
            update_calculated_velocity(current_game_state, previous_game_state, delta_time)

            # 4. Update RL controller (does all the processing)
            rl_controller.update()

            # 5. Optional: Print debug info
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=== RL Agent Running ===")
            print(f"Players: {len(current_game_state.players)}")

            # Print basic info for each player
            for player_id, player in current_game_state.players.items():
                print(f"\nPlayer {player_id}:")
                print(f"Position: {player['position']}")
                print(f"Health: {player['health']}")
                print(f"Moving: {'Yes' if player['is_moving'] else 'No'}")

            # 6. Store previous state and time
            previous_game_state = current_game_state
            previous_time = current_time

            # 7. Control loop speed
            processing_time = time.monotonic() - start_time
            sleep_time = max(0, POLLING_INTERVAL_SECONDS - processing_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping RL controller...")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()