import json
import os
import time
import math
import random
from datetime import datetime
from enum import Enum

from unified_logger import UnifiedLogger
logger = UnifiedLogger(config_path="logging_config.json")

import threading
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

from main import fetch_game_state, ParsedGameState
from model.PlayerMovement import PlayerMovementController
from model.PlayerInputController import PlayerInputController

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Shortcut references
general_cfg = config["general"]
dqn_cfg = config["dqn"]
rewards_cfg = config["rewards"]
actions_cfg = config["actions"]
training_cfg = config["training"]
logging_cfg = config["logging"]

# --- Constants / Config values ---
DEBUG = general_cfg["debug"]
SERVER_URL = general_cfg["server_url"]
DESIRED_CHARACTERS: List[str] = general_cfg["desired_characters"]

# DQN params
STATE_SIZE = dqn_cfg["state_size"]
ACTION_SIZE = dqn_cfg["action_size"]
HIDDEN_LAYERS = dqn_cfg["hidden_layers"]
LEARNING_RATE = dqn_cfg["learning_rate"]
GAMMA = dqn_cfg["gamma"]
EPSILON_START = dqn_cfg["epsilon_start"]
EPSILON_MIN = dqn_cfg["epsilon_min"]
EPSILON_DECAY = dqn_cfg["epsilon_decay"]
BATCH_SIZE = dqn_cfg["batch_size"]
REPLAY_MEMORY_SIZE = dqn_cfg["replay_memory_size"]
TARGET_UPDATE_INTERVAL = dqn_cfg["target_update_interval"]

# Reward weights
MOVEMENT_REWARD_WEIGHT = rewards_cfg["movement_reward_weight"]
PICKUP_DISTANCE_REWARD_WEIGHT = rewards_cfg["pickup_distance_reward_weight"]
GUN_PICKUP_REWARD = rewards_cfg["gun_pickup_reward"]
GUN_POSSESSION_REWARD = rewards_cfg["gun_possession_reward"]
ENEMY_ELIMINATION_REWARD = rewards_cfg["enemy_elimination_reward"]
SURVIVAL_REWARD = rewards_cfg["survival_reward"]
ROTATION_REWARD_WEIGHT = rewards_cfg.get("rotation_reward_weight")
SHOOTING_REWARD = rewards_cfg.get("shooting_reward", 0.0)
KICK_NEAR_GLASS_REWARD = rewards_cfg.get("kick_near_glass_reward", 0.0)
LOOK_TOWARD_TARGET_REWARD_WEIGHT = rewards_cfg.get("look_toward_target_reward_weight", 0.0)


# Actions config
PICKUP_ATTEMPTS = actions_cfg["pickup_attempts"]
PICKUP_DELAY = actions_cfg["pickup_delay"]
SHOOT_PROBABILITY = actions_cfg["shoot_probability"]
MIN_DISTANCE_TO_PICKUP_TO_SKIP_MOVE = actions_cfg["min_distance_to_pickup_to_skip_move"]

# Training config
MAX_STEPS_PER_EPISODE = training_cfg["max_steps_per_episode"]


def extract_input_parameters(config: dict) -> dict:
    return {
        "characters": config.get("general", {}).get("desired_characters", []),
        "server_url": config.get("general", {}).get("server_url", ""),
        "rewards": config.get("rewards", {}),
        "epsilon_decay": config.get("training", {}).get("epsilon_decay"),
        "learning_rate": config.get("training", {}).get("learning_rate"),
        "batch_size": config.get("training", {}).get("batch_size"),
        "target_update_interval": config.get("training", {}).get("target_update_interval")
    }


# --- Constants ---
class PlayerCharacter(str, Enum):
    ORANGE = "Orange"
    LIME = "Lime"
    VITELOT = "Vitelot"
    LEMON = "Lemon"


# --- Helper Functions ---
def has_gun(player_state: dict) -> bool:
    return 'gun' in player_state.get('inventory', [])


def find_character_ids(gs, desired_characters: List[str]) -> Dict[str, int]:
    character_map = {}
    for pid, pdata in gs.players.items():
        character = pdata.get('character')
        if character in desired_characters:
            character_map[character] = pid
    return character_map


def setup_agents(server_url: str, characters: List[str]):
    while True:
        gs = fetch_game_state()
        if gs:
            char_id_map = find_character_ids(gs, characters)
            if len(char_id_map) == len(characters):
                break

    agent_ids = [char_id_map[c] for c in characters]
    mov_ctrls = [PlayerMovementController(server_url, pid) for pid in agent_ids]
    inp_ctrls = [PlayerInputController(server_url, pid) for pid in agent_ids]
    print(f"✅ Found agents: {list(zip(characters, agent_ids))}")
    return agent_ids, mov_ctrls, inp_ctrls


# --- DQN Network ---
class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        # Use config for hidden layers
        self.fc1 = nn.Linear(input_size, HIDDEN_LAYERS[0])
        self.fc2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
        self.fc3 = nn.Linear(HIDDEN_LAYERS[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# --- Multi-agent RL ---
class MultiAgentRL:
    def __init__(self, num_agents: int, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE):
        self.num_agents = num_agents
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.batch_size = BATCH_SIZE
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.policy_nets = [DQN(state_size, action_size).to(device) for _ in range(num_agents)]
        self.target_nets = [DQN(state_size, action_size).to(device) for _ in range(num_agents)]

        self.optimizers = [optim.Adam(net.parameters(), lr=LEARNING_RATE) for net in self.policy_nets]
        for i in range(num_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())

    def act(self, state_vec: np.ndarray, agent_idx: int) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(len(self.policy_nets[agent_idx].fc3.weight))
        state = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
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
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

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
    def __init__(self):
        self.last_targets = {}
        self.last_positions = {}

    def process_state(self, game_state: ParsedGameState, agent_id: int) -> np.ndarray:
        if agent_id not in game_state.players:
            return np.zeros(13, dtype=np.float32)

        p = game_state.players[agent_id]
        feat = []
        feat.extend(p['position'])
        feat.append(p['rotation'])
        feat.append(1 if p['is_shooting'] else 0)
        feat.append(1 if p['is_kicking'] else 0)

        # Cache pickup positions
        pickups = game_state.pickup_positions()
        if pickups:
            closest = min(pickups, key=lambda x: math.dist(x, p['position']))
            feat.extend(closest)
        else:
            feat.extend([-1, -1])

        # Cache enemy positions
        enemies = [v for pid, v in game_state.players.items() if pid != agent_id]
        if enemies:
            ce = min(enemies, key=lambda x: math.dist(x['position'], p['position']))
            feat.extend(ce['position'])
        else:
            feat.extend([-1, -1])

        if len(feat) < 13:
            feat += [0.0] * (13 - len(feat))

        return np.array(feat[:13], dtype=np.float32)


# --- Action Mapping ---
def map_action(action_idx: int, movement_ctrl: PlayerMovementController,
               input_ctrl: PlayerInputController, player_state: dict, game_state: ParsedGameState):
    pos = player_state.get('position', [0, 0])
    has_weapon = has_gun(player_state)

    current_gs = fetch_game_state()
    if not current_gs:
        return

    # Safety check - avoid movement if too close to pickup to prevent accidental kicks
    pickups = current_gs.pickup_positions()
    if any(math.dist(pos, pu) < MIN_DISTANCE_TO_PICKUP_TO_SKIP_MOVE for pu in pickups):
        if DEBUG and action_idx < 4:
            print("🚫 Too close to pickup, skipping movement to avoid kick/throw")
        if action_idx == 4 and not has_weapon:
            for attempt in range(PICKUP_ATTEMPTS):
                if input_ctrl.press_pickup():
                    if DEBUG:
                        print(f"✅ Pickup success on attempt {attempt + 1}")
                    time.sleep(PICKUP_DELAY)
                    break
        return

    # Movement handling
    if action_idx < 4:
        vectors = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        movement_ctrl.move_analog(*vectors[action_idx])

    # Pickup handling if not near pickup but action requested pickup
    elif action_idx == 4 and not has_weapon:
        for attempt in range(PICKUP_ATTEMPTS):
            if input_ctrl.press_pickup():
                if DEBUG:
                    print(f"✅ Pickup success on attempt {attempt + 1}")
                time.sleep(PICKUP_DELAY)
                break

    # 🔁 Always rotate toward closest target, even without weapon
    targets = []
    targets += current_gs.pickup_positions()
    for y, row in enumerate(current_gs.walls):
        for x, tile in enumerate(row):
            if tile == 'Glass':
                targets.append([x, y])
    for pid, enemy in current_gs.players.items():
        if pid != player_state.get('id'):
            targets.append(enemy.get('position', [0, 0]))

    if targets:
        closest = min(targets, key=lambda t: math.dist(t, pos))
        dx, dy = closest[0] - pos[0], closest[1] - pos[1]
        angle = math.atan2(dy, dx)
        movement_ctrl.set_ai_rotation(angle)

        # Action 5: Shoot (if agent has a gun)
    elif action_idx == 5 and has_weapon:
        input_ctrl.press_shoot()
        if DEBUG:
            print("🔫 Shooting action executed")

    # Action 6: Kick (only if agent is unarmed and not too close to a pickup)
    elif action_idx == 6 and not has_weapon:
        too_close_to_gun = any(math.dist(pos, gun) < 1.5 for gun in pickups)
        if not too_close_to_gun:
            input_ctrl.press_foot()
            if DEBUG:
                print("🦶 Kicking (no gun nearby)")
        elif DEBUG:
            print("❌ Skipping kick — too close to gun or already armed")


# --- Reward Calculation ---
def compute_reward(prev_state: ParsedGameState, curr_state: ParsedGameState, agent_id: int) -> float:
    if agent_id not in prev_state.players or agent_id not in curr_state.players:
        return 0.0

    prev_p = prev_state.players[agent_id]
    curr_p = curr_state.players[agent_id]
    r = 0.0

    movement_distance = math.dist(prev_p['position'], curr_p['position'])
    r += movement_distance * MOVEMENT_REWARD_WEIGHT

    pickups = curr_state.pickup_positions()
    if pickups:
        prev_dist = min(math.dist(p, prev_p['position']) for p in pickups)
        curr_dist = min(math.dist(p, curr_p['position']) for p in pickups)
        if curr_dist < prev_dist:
            r += (prev_dist - curr_dist) * PICKUP_DISTANCE_REWARD_WEIGHT

    prev_gun = has_gun(prev_p)
    curr_gun = has_gun(curr_p)
    if not prev_gun and curr_gun:
        r += GUN_PICKUP_REWARD
    elif curr_gun:
        r += GUN_POSSESSION_REWARD

    if curr_gun:
        prev_enemies = set(prev_state.players) - {agent_id}
        curr_enemies = set(curr_state.players) - {agent_id}
        for pid in prev_enemies - curr_enemies:
            r += ENEMY_ELIMINATION_REWARD

    r += SURVIVAL_REWARD

    # 🔁 New: Rotation reward (yaw difference in radians or degrees)
    # Reward for rotating towards closest target (pickup, glass, or enemy)
    if 'rotation' in curr_p:
        agent_pos = curr_p['position']
        agent_yaw = curr_p['rotation']

        # Collect targets
        targets = []
        targets += curr_state.pickup_positions()

        for y, row in enumerate(curr_state.walls):
            for x, tile in enumerate(row):
                if tile == 'Glass':
                    targets.append([x, y])

        for pid, p in curr_state.players.items():
            if pid != agent_id:
                targets.append(p.get("position", [0, 0]))

        if targets:
            # Find closest target
            closest = min(targets, key=lambda t: math.dist(t, agent_pos))
            dx, dy = closest[0] - agent_pos[0], closest[1] - agent_pos[1]
            target_angle = math.atan2(dy, dx)

            # Compute angle difference
            angle_diff = abs((agent_yaw - target_angle + math.pi) % (2 * math.pi) - math.pi)
            alignment_score = max(0, (math.pi - angle_diff) / math.pi)  # 1 when aligned, 0 when opposite
            r += alignment_score * ROTATION_REWARD_WEIGHT

    if curr_p.get('is_shooting', False):
        r += SHOOTING_REWARD

    # Reward for kicking near glass tiles
    if curr_p.get("is_kicking", False):
        px, py = curr_p['position']
        for y, row in enumerate(curr_state.walls):
            for x, tile in enumerate(row):
                if tile == "Glass":
                    distance = math.dist([x, y], [px, py])
                    if distance < 1.5:
                        r += KICK_NEAR_GLASS_REWARD
                        break

    return r


# --- Main Training Loop ---
def run_training_loop(server_url: str = SERVER_URL, logger: UnifiedLogger = logger):
    print(f"🤖 Starting training for agents: {DESIRED_CHARACTERS}")
    agent_ids, mov_ctrls, inp_ctrls = setup_agents(server_url, DESIRED_CHARACTERS)
    state_proc = StateProcessor()
    rl_agent = MultiAgentRL(num_agents=len(agent_ids))
    ep_rewards = [[] for _ in agent_ids]
    episode = 0
    episode_count = 0
    prev_states: List[Optional[np.ndarray]] = [None] * len(agent_ids)
    os.makedirs("summaries", exist_ok=True)

    try:
        while True:
            episode_id = episode  # use integer for experiment_number
            start_time = time.time()
            gs = fetch_game_state()
            if not gs or any(pid not in gs.players for pid in agent_ids):
                time.sleep(0.05)
                continue

            # Log experiment start as a step log (or you can skip this if you want)
            logger.log_step({
                "type": "experiment_start",
                "experiment_id": episode_id,
                "agent_ids": agent_ids,
                "start_time": start_time,
                "params": {
                    "characters": DESIRED_CHARACTERS,
                    "server_url": server_url
                }
            }, force=True)

            states = [state_proc.process_state(gs, pid) for pid in agent_ids]
            total_r = [0.0] * len(agent_ids)
            done = False
            step_count = 0
            snapshots = []

            while not done:
                threads = []
                actions = []

                for idx, pid in enumerate(agent_ids):
                    action = rl_agent.act(states[idx], idx)
                    actions.append(action)
                    t = threading.Thread(
                        target=map_action,
                        args=(action, mov_ctrls[idx], inp_ctrls[idx], gs.players[pid], gs)
                    )
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join()

                next_gs = fetch_game_state()
                next_states, rewards = [None] * len(agent_ids), [0.0] * len(agent_ids)

                for idx, pid in enumerate(agent_ids):
                    if next_gs and pid in next_gs.players:
                        next_states[idx] = state_proc.process_state(next_gs, pid)
                        rewards[idx] = compute_reward(gs, next_gs, pid)
                        total_r[idx] += rewards[idx]

                        player = next_gs.players[pid]
                        position = player.get("position", (0.0, 0.0))
                        movement = player.get("movement", {})
                        calc_velocity = player.get("calculated_velocity", None)

                        # Log each step's state data
                        logger.log_step({
                            "type": "state",
                            "experiment_id": episode_id,
                            "agent_id": pid,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "character": str(player.get("character", "Unknown")),
                            "device": str(player.get("device", "Unknown")),
                            "position": {
                                "x": position[0],
                                "y": position[1]
                            },
                            "rotation": player.get("rotation", 0.0),
                            "reward": rewards[idx],
                            "action": actions[idx],
                            "health": player.get("health", -1),
                            "movement": {
                                "speed": movement.get("speed", 0.0),
                                "direction": {
                                    "x": movement.get("direction", (0.0, 0.0))[0],
                                    "y": movement.get("direction", (0.0, 0.0))[1]
                                }
                            },
                            "calculated_velocity": {
                                "x": calc_velocity[0] if calc_velocity else None,
                                "y": calc_velocity[1] if calc_velocity else None
                            },
                            "calculated_speed": player.get("calculated_speed", None),
                            "is_shooting": bool(player.get("is_shooting", False)),
                            "is_kicking": bool(player.get("is_kicking", False)),
                            "is_moving": bool(player.get("is_moving", False)),
                            "inventory": player.get("inventory", [])
                        })

                        max_reward = max(total_r)
                        wins = [1 if r == max_reward else 0 for r in total_r]

                step_count += 1
                if step_count >= MAX_STEPS_PER_EPISODE:
                    done = True

                if prev_states[0] is not None:
                    rl_agent.remember(states, actions, rewards, next_states, [done] * len(agent_ids))
                    rl_agent.replay()

                states = next_states
                gs = next_gs

                # You might want to define when done=True here,
                # otherwise this is an infinite inner loop
                # For example, break after some number of steps or a terminal condition

            input_params = extract_input_parameters(config)

            summary_entry = {"type": "experiment_summary", "timestamp": datetime.utcnow().isoformat() + "Z",
                             "experiment_number": episode_id, "parameters": {
                    "input_parameters": input_params,
                    "characters": DESIRED_CHARACTERS,
                    "server_url": server_url
                }, "metrics": {
                    "total_rewards": total_r,
                    "duration": time.time() - start_time,
                    "episode_count": episode_count,
                    "wins": wins
                }, "snapshots": snapshots}

            logger.log_experiment_summary(
                experiment_number=summary_entry["experiment_number"],
                parameters=summary_entry["parameters"],
                metrics=summary_entry["metrics"],
                snapshots=summary_entry["snapshots"]
            )

            summary_path = os.path.join("summaries", f"summary_episode_{episode_id}.json")

            with open(summary_path, "w") as f:
                json.dump(summary_entry, f, indent=2)

            episode += 1
            episode_count += 1

    except KeyboardInterrupt:
        print("🛑 Training interrupted by user")
        # Log training stop as a forced step log
        logger.log_step({
            "type": "training_stopped",
            "timestamp": time.time(),
            "last_episode": episode
        }, force=True)


if __name__ == "__main__":
    run_training_loop()