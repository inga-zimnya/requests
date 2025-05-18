import time
import math
import random
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from main import fetch_game_state, ParsedGameState
from PlayerMovement import PlayerMovementController
from PlayerInputController import PlayerInputController

# === Constants ===
STATE_SIZE = 15  # Make sure this matches your StateProcessor output
ACTION_SIZE = 4


# === Neural Network ===
class DQN(nn.Module):
    def __init__(self, input_size=STATE_SIZE, output_size=ACTION_SIZE):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# === Multi-Agent RL ===
class MultiAgentRL:
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory = deque(maxlen=10000)

        self.policy_nets = [DQN() for _ in range(num_agents)]
        self.target_nets = [DQN() for _ in range(num_agents)]
        self.optimizers = [optim.Adam(net.parameters(), lr=0.001) for net in self.policy_nets]

        for i in range(num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())

    def act(self, state: np.ndarray, agent_id: int) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(ACTION_SIZE)

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_nets[agent_id](state)
        return torch.argmax(q_values).item()

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool, agent_id: int):
        self.memory.append((state, action, reward, next_state, done, agent_id))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        # Separate experiences by agent
        agent_batches = [[] for _ in range(self.num_agents)]
        for state, action, reward, next_state, done, agent_id in minibatch:
            agent_batches[agent_id].append((state, action, reward, next_state, done))

        # Train each agent
        for agent_id in range(self.num_agents):
            if len(agent_batches[agent_id]) == 0:
                continue

            states, actions, rewards, next_states, dones = zip(*agent_batches[agent_id])

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            # Verify dimensions
            if states.shape[1] != STATE_SIZE:
                print(f"Error: Expected state size {STATE_SIZE}, got {states.shape[1]}")
                continue

            current_q = self.policy_nets[agent_id](states).gather(1, actions.unsqueeze(1))

            with torch.no_grad():
                next_q = self.target_nets[agent_id](next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q

            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            self.optimizers[agent_id].zero_grad()
            loss.backward()
            self.optimizers[agent_id].step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_networks(self):
        for i in range(self.num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())


# === State Processor ===
class StateProcessor:
    def __init__(self):
        self.prev_positions = {}

    def process_state(self, game_state: ParsedGameState, agent_id: int) -> np.ndarray:
        if agent_id not in game_state.players:
            return np.zeros(STATE_SIZE)

        player = game_state.players[agent_id]
        features = []

        # Player features (5)
        features.extend(player["position"])  # x, y
        features.append(player["rotation"])
        features.append(1 if "gun" in player.get("inventory", []) else 0)
        features.append(1 if player.get("is_shooting", False) else 0)
        features.append(1 if player.get("is_kicking", False) else 0)

        # Pickup features (2)
        pickup_pos = game_state.pickup_positions()
        if pickup_pos:
            closest = min(pickup_pos, key=lambda p: math.dist(p, player["position"]))
            features.extend(closest)
        else:
            features.extend([-1, -1])

        # Enemy features (3)
        enemies = [p for pid, p in game_state.players.items() if pid != agent_id]
        if enemies:
            closest = min(enemies, key=lambda e: math.dist(e["position"], player["position"]))
            features.extend(closest["position"])
            features.append(1 if "gun" in closest.get("inventory", []) else 0)
        else:
            features.extend([-1, -1, 0])

        # Glass features (5)
        features.append(sum(row.count('Glass') for row in game_state.walls))

        # Ensure we have exactly STATE_SIZE features
        if len(features) != STATE_SIZE:
            print(f"Warning: Expected {STATE_SIZE} features, got {len(features)}")
            features.extend([0] * (STATE_SIZE - len(features)))

        return np.array(features[:STATE_SIZE], dtype=np.float32)


# === Reward Calculation ===
def compute_reward(prev_state: ParsedGameState, curr_state: ParsedGameState, agent_id: int) -> float:
    if agent_id not in prev_state.players or agent_id not in curr_state.players:
        return 0.0

    prev_player = prev_state.players[agent_id]
    curr_player = curr_state.players[agent_id]
    reward = 0.0

    # Movement
    position_change = math.dist(prev_player["position"], curr_player["position"])
    reward += position_change * 0.05

    # Weapon pickup
    prev_gun = "gun" in prev_player.get("inventory", [])
    curr_gun = "gun" in curr_player.get("inventory", [])
    if not prev_gun and curr_gun:
        reward += 10.0

    # Combat
    if curr_gun:
        prev_enemies = {pid: p for pid, p in prev_state.players.items() if pid != agent_id}
        curr_enemies = {pid: p for pid, p in curr_state.players.items() if pid != agent_id}

        for pid in prev_enemies:
            if pid not in curr_enemies:
                reward += 15.0

        for pid, enemy in curr_enemies.items():
            if pid in prev_enemies and "health" in prev_enemies[pid] and "health" in enemy:
                reward += (prev_enemies[pid]["health"] - enemy["health"]) * 0.5

    # Penalties
    if not curr_gun and curr_player.get("is_shooting", False):
        reward -= 5.0

    # Survival
    reward += 0.1

    return reward


# === Action Mapping ===
def map_action(action_idx: int, movement: PlayerMovementController,
               input_ctrl: PlayerInputController, has_gun: bool):
    # Movement
    if action_idx == 0:
        movement.move_analog(0, 1)
    elif action_idx == 1:
        movement.move_analog(0, -1)
    elif action_idx == 2:
        movement.move_analog(-1, 0)
    elif action_idx == 3:
        movement.move_analog(1, 0)

    # Combat
    if random.random() < 0.3:
        input_ctrl.press_foot()

    if has_gun and random.random() < 0.1 and hasattr(input_ctrl, 'press_shoot'):
        input_ctrl.press_shoot()


# === Main Training Loop ===
def run_training_loop(num_agents=2, server_url="http://127.0.0.1:15702/"):
    print(f"🤖 Starting training with {num_agents} agents")

    controllers = [
        {
            'movement': PlayerMovementController(server_url, i),
            'input': PlayerInputController(server_url, i),
            'processor': StateProcessor(),
            'prev_state': None
        }
        for i in range(num_agents)
    ]

    rl_agent = MultiAgentRL(num_agents=num_agents)
    episode = 0

    try:
        while True:
            # Wait for valid state
            while True:
                game_state = fetch_game_state()
                if game_state and all(i in game_state.players for i in range(num_agents)):
                    break
                time.sleep(0.1)

            # Initialize states
            states = [controllers[i]['processor'].process_state(game_state, i)
                      for i in range(num_agents)]

            # Episode loop
            episode_rewards = [0] * num_agents
            episode_steps = 0

            while True:
                actions = []

                # Get actions
                for i in range(num_agents):
                    action = rl_agent.act(states[i], i)
                    actions.append(action)

                    # Execute action
                    has_gun = "gun" in game_state.players[i].get("inventory", [])
                    map_action(action, controllers[i]['movement'],
                               controllers[i]['input'], has_gun)

                # Wait for next state
                time.sleep(0.1)
                next_game_state = fetch_game_state()

                if not next_game_state or not all(i in next_game_state.players for i in range(num_agents)):
                    print("⚠️ Invalid game state, skipping")
                    continue

                # Process rewards
                next_states = []
                rewards = []

                for i in range(num_agents):
                    reward = compute_reward(game_state, next_game_state, i)
                    rewards.append(reward)
                    episode_rewards[i] += reward

                    next_state = controllers[i]['processor'].process_state(next_game_state, i)
                    next_states.append(next_state)

                    # Store experience
                    if controllers[i]['prev_state'] is not None:
                        rl_agent.remember(
                            controllers[i]['prev_state'],
                            actions[i],
                            reward,
                            next_state,
                            False,
                            i
                        )

                # Train
                rl_agent.replay()

                # Update states
                for i in range(num_agents):
                    controllers[i]['prev_state'] = states[i]
                    states[i] = next_states[i]

                game_state = next_game_state
                episode_steps += 1

                # Check end condition
                if episode_steps >= 100 or any(game_state.players[i].get("health", 1) <= 0 for i in range(num_agents)):
                    break

            # End of episode
            episode += 1
            print(f"Episode {episode} completed. Steps: {episode_steps}")
            print(f"Agent rewards: {episode_rewards}")

            if episode % 10 == 0:
                rl_agent.update_target_networks()
                print("Updated target networks")

            # Reset
            for i in range(num_agents):
                controllers[i]['movement'].stop()
                controllers[i]['input'].clear_input()
                controllers[i]['prev_state'] = None

    except KeyboardInterrupt:
        print("🛑 Training stopped")
    finally:
        for i in range(num_agents):
            controllers[i]['movement'].stop()
            controllers[i]['input'].clear_input()


if __name__ == "__main__":
    run_training_loop(num_agents=2)