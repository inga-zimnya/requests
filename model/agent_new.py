import time
import math
import random
from enum import Enum

import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from main import fetch_game_state, ParsedGameState
from model.PlayerMovement import PlayerMovementController
from model.PlayerInputController import PlayerInputController


# --- Constants ---
class PlayerCharacter(str, Enum):
    ORANGE = "Orange"
    LIME = "Lime"
    VITELOT = "Vitelot"
    LEMON = "Lemon"


DESIRED_CHARACTERS: List[str] = [PlayerCharacter.LIME, PlayerCharacter.ORANGE]


# --- Helper Functions ---
def has_gun(player_state: dict) -> bool:
    return 'gun' in player_state.get('inventory', [])


def find_character_ids(gs, desired_characters: List[str]) -> Dict[str, int]:
    """Map character names to player IDs."""
    character_map = {}
    for pid, pdata in gs.players.items():
        character = pdata.get('character')
        if character in desired_characters:
            character_map[character] = pid
    return character_map


def setup_agents(server_url: str, characters: List[str]):
    """Find characters in game and setup controllers."""
    while True:
        gs = fetch_game_state()
        if gs:
            char_id_map = find_character_ids(gs, characters)
            if len(char_id_map) == len(characters):
                break
        time.sleep(0.1)

    agent_ids = [char_id_map[c] for c in characters]
    mov_ctrls = [PlayerMovementController(server_url, pid) for pid in agent_ids]
    inp_ctrls = [PlayerInputController(server_url, pid) for pid in agent_ids]
    print(f"✅ Found agents: {list(zip(characters, agent_ids))}")
    return agent_ids, mov_ctrls, inp_ctrls


# --- DQN Network ---
class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# --- Multi-agent RL ---
class MultiAgentRL:
    def __init__(self, num_agents: int, state_size: int = 13, action_size: int = 5):
        self.num_agents = num_agents
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory = deque(maxlen=10000)

        self.policy_nets = [DQN(state_size, action_size) for _ in range(num_agents)]
        self.target_nets = [DQN(state_size, action_size) for _ in range(num_agents)]
        self.optimizers = [optim.Adam(net.parameters(), lr=0.001) for net in self.policy_nets]
        for i in range(num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())

    def act(self, state_vec: np.ndarray, agent_idx: int) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(len(self.policy_nets[agent_idx].fc3.weight))
        state = torch.FloatTensor(state_vec).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_nets[agent_idx](state)
        return torch.argmax(q).item()

    def remember(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.memory.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        for i in range(self.num_agents):
            q_vals = self.policy_nets[i](states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = self.target_nets[i](next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q
            loss = nn.MSELoss()(q_vals, target_q)
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_networks(self):
        for i in range(self.num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())


# --- State Processor ---
class StateProcessor:
    def process_state(self, game_state: ParsedGameState, agent_id: int) -> np.ndarray:
        if agent_id not in game_state.players:
            return np.zeros(13, dtype=np.float32)
        p = game_state.players[agent_id]
        feat = []
        feat.extend(p['position'])
        feat.append(p['rotation'])
        feat.append(1 if p['is_shooting'] else 0)
        feat.append(1 if p['is_kicking'] else 0)
        pickups = game_state.pickup_positions()
        if pickups:
            closest = min(pickups, key=lambda x: math.dist(x, p['position']))
            feat.extend(closest)
        else:
            feat.extend([-1, -1])
        enemies = [v for pid, v in game_state.players.items() if pid != agent_id]
        if enemies:
            ce = min(enemies, key=lambda x: math.dist(x['position'], p['position']))
            feat.extend(ce['position'])
        else:
            feat.extend([-1, -1])
        if len(feat) < 13:
            feat += [0.0] * (13 - len(feat))
        return np.array(feat[:13], dtype=np.float32)


# --- Reward Calculation ---
def compute_reward(prev_state: ParsedGameState, curr_state: ParsedGameState, agent_id: int) -> float:
    if agent_id not in prev_state.players or agent_id not in curr_state.players:
        return 0.0

    prev_p = prev_state.players[agent_id]
    curr_p = curr_state.players[agent_id]
    r = 0.0

    movement_distance = math.dist(prev_p['position'], curr_p['position'])
    r += movement_distance * 0.1

    pickups = curr_state.pickup_positions()
    if pickups:
        prev_dist = min(math.dist(p, prev_p['position']) for p in pickups)
        curr_dist = min(math.dist(p, curr_p['position']) for p in pickups)
        if curr_dist < prev_dist:
            r += (prev_dist - curr_dist) * 0.5
        elif curr_dist > prev_dist:
            r -= (curr_dist - prev_dist) * 0.2

    prev_gun = has_gun(prev_p)
    curr_gun = has_gun(curr_p)
    if not prev_gun and curr_gun:
        r += 30.0
    elif prev_gun and not curr_gun:
        r -= 25.0
    elif curr_gun:
        r += 0.8

    if curr_gun:
        prev_enemies = set(prev_state.players) - {agent_id}
        curr_enemies = set(curr_state.players) - {agent_id}
        for pid in prev_enemies - curr_enemies:
            r += 40.0
        if curr_enemies:
            closest_enemy = min(curr_enemies, key=lambda pid: math.dist(
                curr_state.players[pid]['position'],
                curr_p['position']
            ))
            enemy_pos = curr_state.players[closest_enemy]['position']
            r += 0.2 if is_facing_enemy(curr_p, enemy_pos) else 0

    if not curr_gun and curr_p.get('is_shooting', False):
        r -= 8.0
    if curr_p.get('is_kicking', False):
        r -= 0.5

    prev_glass = sum(row.count('Glass') for row in prev_state.walls)
    curr_glass = sum(row.count('Glass') for row in curr_state.walls)
    if curr_glass < prev_glass:
        r += (prev_glass - curr_glass) * 0.7

    r += 0.1
    return r


def is_facing_enemy(player: dict, enemy_pos: Tuple[float, float]) -> bool:
    px, py = player['position']
    ex, ey = enemy_pos
    angle_to_enemy = math.atan2(ey - py, ex - px)
    return abs(angle_to_enemy - player['rotation']) < math.pi / 4


# --- Action Mapping ---
def map_action(action_idx: int, movement_ctrl: PlayerMovementController,
               input_ctrl: PlayerInputController, player_state: dict):
    input_ctrl.clear_input()

    if action_idx == 0:
        movement_ctrl.move_analog(0, 1)
    elif action_idx == 1:
        movement_ctrl.move_analog(0, -1)
    elif action_idx == 2:
        movement_ctrl.move_analog(-1, 0)
    elif action_idx == 3:
        movement_ctrl.move_analog(1, 0)
    elif action_idx == 4:
        if not has_gun(player_state):
            input_ctrl.press_pickup()

    if not has_gun(player_state):
        if random.random() < 0.3:
            input_ctrl.press_foot()
    if random.random() < 0.1:
        input_ctrl.press_shoot()


# --- Main Training Loop ---
def run_training_loop(server_url: str = "http://127.0.0.1:15702/"):
    print(f"🤖 Starting training for agents: {DESIRED_CHARACTERS}")
    agent_ids, mov_ctrls, inp_ctrls = setup_agents(server_url, DESIRED_CHARACTERS)
    state_proc = StateProcessor()
    rl_agent = MultiAgentRL(num_agents=len(agent_ids))
    ep_rewards = [[] for _ in agent_ids]
    episode = 0
    prev_states: List[Optional[np.ndarray]] = [None] * len(agent_ids)

    try:
        while True:
            gs = fetch_game_state()
            if not gs or any(pid not in gs.players for pid in agent_ids):
                time.sleep(0.1)
                continue

            states = [state_proc.process_state(gs, pid) for pid in agent_ids]
            total_r = [0.0] * len(agent_ids)
            done = False

            while not done:
                actions = []
                for idx, pid in enumerate(agent_ids):
                    a = rl_agent.act(states[idx], idx)
                    actions.append(a)
                    player_state = gs.players[pid]
                    map_action(a, mov_ctrls[idx], inp_ctrls[idx], player_state)

                time.sleep(0.1)
                next_gs = fetch_game_state()
                next_states, rewards = [None] * len(agent_ids), [0.0] * len(agent_ids)

                for idx, pid in enumerate(agent_ids):
                    if next_gs and pid in next_gs.players:
                        next_states[idx] = state_proc.process_state(next_gs, pid)
                        rewards[idx] = compute_reward(gs, next_gs, pid)
                        total_r[idx] += rewards[idx]

                if prev_states[0] is not None:
                    rl_agent.remember(states, actions, rewards, next_states, [done] * len(agent_ids))
                rl_agent.replay()
                prev_states, states, gs = states, next_states, next_gs
                episode += 1

            for i in range(len(agent_ids)):
                ep_rewards[i].append(total_r[i])
                avg = np.mean(ep_rewards[i][-10:]) if ep_rewards[i] else 0
                print(f"Agent {agent_ids[i]} ep {episode}: R={total_r[i]:.1f}, Avg={avg:.1f}")

            if episode % 10 == 0:
                rl_agent.update_target_networks()

            for c in mov_ctrls: c.stop()
            for c in inp_ctrls: c.clear_input()

    except KeyboardInterrupt:
        print("🚩 Training stopped by user")
    finally:
        for c in mov_ctrls: c.stop()
        for c in inp_ctrls: c.clear_input()


if __name__ == "__main__":
    run_training_loop()
