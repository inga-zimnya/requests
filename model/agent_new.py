import time
import math
import random
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from main import fetch_game_state, ParsedGameState
from model.PlayerMovement import PlayerMovementController
from model.PlayerInputController import PlayerInputController


# === Neural Network Models ===
class DQN(nn.Module):
    """Deep Q-Network for decision making"""

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MultiAgentRL:
    def __init__(self, num_agents=2, state_size=15, action_size=5):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory = deque(maxlen=10000)

        # Create networks for each agent
        self.policy_nets = [DQN(state_size, action_size) for _ in range(num_agents)]
        self.target_nets = [DQN(state_size, action_size) for _ in range(num_agents)]
        self.optimizers = [optim.Adam(net.parameters(), lr=0.001) for net in self.policy_nets]

        # Sync target networks
        for i in range(num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())

    def act(self, state_vec: np.ndarray, agent_id: int) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state_vec).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_nets[agent_id](state)
        return torch.argmax(q_values).item()

    def remember(self, states: List[np.ndarray], actions: List[int],
                 rewards: List[float], next_states: List[np.ndarray], dones: List[bool]):
        for i in range(self.num_agents):
            self.memory.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Train each agent's network
        for agent_id in range(self.num_agents):
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


# === State Processing ===
class StateProcessor:
    def __init__(self):
        self.prev_positions = {}

    def process_state(self, game_state: ParsedGameState, agent_id: int) -> np.ndarray:
        if agent_id not in game_state.players:
            return np.zeros(15)

        player = game_state.players[agent_id]
        features = []

        features.extend(player["position"])
        features.append(player["rotation"])
        features.append(player["health"])
        features.append(1 if player["is_shooting"] else 0)
        features.append(1 if player["is_kicking"] else 0)

        pickup_pos = game_state.pickup_positions()
        if pickup_pos:
            closest_pickup = min(pickup_pos, key=lambda p: math.dist(p, player["position"]))
            features.extend(closest_pickup)
        else:
            features.extend([-1, -1])

        enemies = [p for pid, p in game_state.players.items() if pid != agent_id]
        if enemies:
            closest_enemy = min(enemies, key=lambda e: math.dist(e["position"], player["position"]))
            features.extend(closest_enemy["position"])
            features.append(closest_enemy["health"])
        else:
            features.extend([-1, -1, -1])

        if len(features) < 15:
            features.extend([0.0] * (15 - len(features)))
        elif len(features) > 15:
            features = features[:15]

        return np.array(features, dtype=np.float32)


# === Reward Calculation ===
def compute_reward(prev_state: ParsedGameState, curr_state: ParsedGameState, agent_id: int) -> float:
    if agent_id not in prev_state.players or agent_id not in curr_state.players:
        return 0.0

    prev_player = prev_state.players[agent_id]
    curr_player = curr_state.players[agent_id]
    reward = 0.0

    position_change = math.dist(prev_player["position"], curr_player["position"])
    reward += position_change * 0.05

    prev_has_gun = "gun" in prev_player.get("inventory", [])
    curr_has_gun = "gun" in curr_player.get("inventory", [])

    if not prev_has_gun and curr_has_gun:
        reward += 10.0
    elif curr_has_gun:
        reward += 0.2

    if curr_has_gun:
        prev_enemies = {pid: p for pid, p in prev_state.players.items() if pid != agent_id}
        curr_enemies = {pid: p for pid, p in curr_state.players.items() if pid != agent_id}

        for pid in prev_enemies:
            if pid not in curr_enemies:
                reward += 15.0

        for pid, curr_enemy in curr_enemies.items():
            if pid in prev_enemies:
                damage_dealt = prev_enemies[pid].get("health", 100) - curr_enemy.get("health", 100)
                reward += damage_dealt * 0.5

    if not curr_has_gun and curr_player.get("is_shooting", False):
        reward -= 5.0

    prev_glass = sum(row.count('Glass') for row in prev_state.walls)
    curr_glass = sum(row.count('Glass') for row in curr_state.walls)
    if curr_glass < prev_glass:
        reward += (prev_glass - curr_glass) * 0.3

    reward += 0.05

    return reward


# === Action Mapping ===
def map_action(action_idx: int, movement_ctrl: PlayerMovementController,
               input_ctrl: PlayerInputController) -> None:
    if action_idx == 0:
        movement_ctrl.move_analog(0, 1)
    elif action_idx == 1:
        movement_ctrl.move_analog(0, -1)
    elif action_idx == 2:
        movement_ctrl.move_analog(-1, 0)
    elif action_idx == 3:
        movement_ctrl.move_analog(1, 0)
    elif action_idx == 4:
        input_ctrl.press_pickup()

    if random.random() < 0.3:
        input_ctrl.press_foot()
    if random.random() < 0.1:
        input_ctrl.press_shoot()


# === Main Training Loop ===
def run_training_loop(num_agents=2, server_url="http://127.0.0.1:15702/"):
    print(f"🤖 Starting multi-agent training with {num_agents} agents")

    movement_controllers = [PlayerMovementController(server_url, i) for i in range(num_agents)]
    input_controllers = [PlayerInputController(server_url, i) for i in range(num_agents)]
    state_processor = StateProcessor()
    rl_agent = MultiAgentRL(num_agents=num_agents)

    episode_rewards = [[] for _ in range(num_agents)]
    episode = 0

    try:
        while True:
            prev_states = [None] * num_agents
            states = [None] * num_agents
            total_rewards = [0] * num_agents
            done = False

            while True:
                game_state = fetch_game_state()
                if game_state and all(i in game_state.players for i in range(num_agents)):
                    break
                time.sleep(0.1)

            for i in range(num_agents):
                states[i] = state_processor.process_state(game_state, i)

            while not done:
                actions = []
                next_states = [None] * num_agents
                rewards = [0] * num_agents

                for i in range(num_agents):
                    action = rl_agent.act(states[i], i)
                    actions.append(action)
                    map_action(action, movement_controllers[i], input_controllers[i])

                time.sleep(0.1)
                next_game_state = fetch_game_state()

                for i in range(num_agents):
                    if next_game_state and i in next_game_state.players:
                        next_states[i] = state_processor.process_state(next_game_state, i)
                        rewards[i] = compute_reward(game_state, next_game_state, i)
                        total_rewards[i] += rewards[i]

                        if next_game_state.players[i]["health"] <= 0:
                            done = True

                if prev_states[0] is not None:
                    rl_agent.remember(states, actions, rewards, next_states, [done] * num_agents)

                rl_agent.replay()

                prev_states = states
                states = next_states
                game_state = next_game_state

            episode += 1
            for i in range(num_agents):
                episode_rewards[i].append(total_rewards[i])
                avg_reward = np.mean(episode_rewards[i][-10:]) if episode_rewards[i] else 0
                print(f"Agent {i} - Episode {episode}: Reward={total_rewards[i]:.1f}, Avg={avg_reward:.1f}")

            if episode % 10 == 0:
                rl_agent.update_target_networks()

            for i in range(num_agents):
                movement_controllers[i].stop()
                input_controllers[i].clear_input()

    except KeyboardInterrupt:
        print("🚩 Training stopped by user")
    finally:
        for i in range(num_agents):
            movement_controllers[i].stop()
            input_controllers[i].clear_input()


if __name__ == "__main__":
    run_training_loop(num_agents=2)
