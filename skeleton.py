import math
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from typing import Dict

from main import (
    fetch_game_state,
    ParsedGameState,
    show_ascii_map,
    print_map_legend, PlayerState
)

class ShooterEnv:
    def __init__(self, server_url: str, polling_interval: float = 0.2):
        self.server_url = server_url
        self.poll_interval = polling_interval
        self.prev_state: ParsedGameState = None
        self.prev_health = {}
        self.prev_kill_counts = {}

    def reset(self) -> ParsedGameState:
        st = None
        while st is None:
            st = fetch_game_state()
            time.sleep(self.poll_interval)
        self.prev_state = st
        for pid, p in st.players.items():
            self.prev_health[pid] = p['health']
            self.prev_kill_counts[pid] = 0
        return st

    def step(self, actions: dict) -> (ParsedGameState, dict):
        for pid, act in actions.items():
            self._send_action(pid, act)
        time.sleep(self.poll_interval)
        next_state = fetch_game_state()
        if next_state is None:
            return self.prev_state, {pid: 0.0 for pid in actions}
        rewards = self._compute_rewards(self.prev_state, next_state)
        self.prev_state = next_state
        return next_state, rewards

    def _send_action(self, player_id, act):
        dx, dy, shoot, kick = act
        cmd = {
            "id": 42,
            "jsonrpc": "2.0",
            "method": "bevy/command",
            "params": {
                "entity": player_id,
                "commands": [
                    {"name": "move", "args": {"dx": float(dx), "dy": float(dy)}},
                    {"name": "shoot", "args": {"enabled": bool(shoot)}},
                    {"name": "kick", "args": {"enabled": bool(kick)}},
                ]
            }
        }
        try:
            requests.post(self.server_url, json=cmd, timeout=0.1).raise_for_status()
        except Exception as e:
            print(f"[WARN] failed to send action for {player_id}: {e}")

    def _compute_rewards(self, prev: ParsedGameState, curr: ParsedGameState) -> dict:
        rwd = {}
        # +1 за каждый новый килл (ai_state true→false)
        for prev_alive, curr_alive in zip(prev.ai_states.values(), curr.ai_states.values()):
            if prev_alive and not curr_alive:
                for pid in curr.players:
                    rwd.setdefault(pid, 0.0)
                    rwd[pid] += 1.0 / max(1, len(curr.players))
        for pid, p_curr in curr.players.items():
            p_prev = prev.players.get(pid)
            base = rwd.setdefault(pid, 0.0)
            # штраф за смерть
            if p_curr['health'] <= 0 < (p_prev['health'] if p_prev else 1):
                base -= 1.0
            # бонус за выживание
            base += 0.1
            # бонус за подборки
            prev_pickups = len(prev.pickup_positions)
            curr_pickups = len(curr.pickup_positions)
            delta = prev_pickups - curr_pickups
            base += 0.5 * max(0, delta)
            # штраф за бездействие
            if not p_curr['is_moving'] and not p_curr['is_shooting'] and not p_curr['is_kicking']:
                base -= 0.2
            rwd[pid] = base
        return rwd

class MLPPolicy(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class MultiAgentTrainer:
    def __init__(self, obs_dim, act_dim, lr=1e-3, exploration_start=1.0, exploration_end=0.1, exploration_decay=0.995):
        self.policies = {}
        self.optimizers = {}
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.current_learning_rate = lr
        self.exploration_rate = exploration_start
        self.exploration_min = exploration_end
        self.exploration_decay = exploration_decay
        self.steps = 0

    def set_learning_rate(self, new_lr):
        for pid in self.optimizers:
            for param_group in self.optimizers[pid].param_groups:
                param_group['lr'] = new_lr
        self.current_learning_rate = new_lr

    def ensure_agent(self, pid):
        if pid not in self.policies:
            pol = MLPPolicy(self.obs_dim, 128, self.act_dim)
            opt = optim.Adam(pol.parameters(), lr=self.lr)
            self.policies[pid] = pol
            self.optimizers[pid] = opt

    def select_actions(self, obs):
        acts = {}
        for pid, ob in obs.items():
            self.ensure_agent(pid)
            tensor = torch.from_numpy(ob).float().unsqueeze(0)

            # Exploration-exploitation tradeoff
            if random.random() < self.exploration_rate:
                # Random action (exploration)
                out = np.random.uniform(-1, 1, size=2)
                shoot = random.random() > 0.5
                kick = random.random() > 0.5
            else:
                # Policy action (exploitation)
                with torch.no_grad():
                    logits = self.policies[pid](tensor)
                    out = torch.tanh(logits[:, :2]).numpy().flatten()
                    shoot = (torch.sigmoid(logits[:, 2]) > 0.5).item()
                    kick = (torch.sigmoid(logits[:, 3]) > 0.5).item()

            acts[pid] = np.array([*out, shoot, kick], dtype=np.float32)

        # Decay exploration rate
        self.exploration_rate = max(
            self.exploration_min,
            self.exploration_rate * self.exploration_decay
        )
        self.steps += 1

        return acts

    def update(self, obss, acts, rewards, next_obss):
        losses = {}
        for pid in rewards:
            self.ensure_agent(pid)
            pol = self.policies[pid]
            opt = self.optimizers[pid]
            tensor = torch.from_numpy(obss[pid]).float().unsqueeze(0)
            logits = pol(tensor)
            loss_tensor = -rewards[pid] * logits.abs().mean()
            loss = loss_tensor.item()
            opt.zero_grad()
            loss_tensor.backward()
            opt.step()
            losses[pid] = loss
        return losses

def extract_obs(state: ParsedGameState) -> dict:
    obs = {}
    for pid, p in state.players.items():
        x, y = p['position']
        h = p['health'] / 100.0
        # nearest enemy
        best = (None, float('inf'))
        enn = None
        for qid, q in state.players.items():
            if qid == pid:
                continue
            dx = q['position'][0] - x
            dy = q['position'][1] - y
            d2 = dx*dx + dy*dy
            if d2 < best[1]:
                best = ([dx, dy], d2)
                enn = q
        if best[0] is None:
            dx, dy, he = 0.0, 0.0, 0.0
        else:
            dx, dy = best[0]
            he = enn['health'] / 100.0
        pickups = len(state.pickup_positions) / 50.0
        obs[pid] = np.array([x/160, y/112, h, dx/160, dy/112, he, pickups], dtype=np.float32)
    return obs

def print_players(players: Dict[int, PlayerState]):
    """Prints player information in a structured, readable format."""
    print("\n=== PLAYER STATES ===")
    for player_id, player in players.items():
        print(f"\nPlayer {player_id} ({player['character'].value}):")
        print(f"  Position: ({player['position'][0]:.1f}, {player['position'][1]:.1f})")
        print(f"  Rotation: {math.degrees(player['rotation']):.1f}°")
        print(f"  Device: {player['device'].value}")
        print(f"  State: {'SHOOTING ' if player['is_shooting'] else ''}"
              f"{'KICKING ' if player['is_kicking'] else ''}"
              f"{'MOVING' if player['is_moving'] else 'IDLE'}")
        print(f"  Health: {player['health']:.1f}")
        print(f"  Velocity: ({player['velocity'][0]:.1f}, {player['velocity'][1]:.1f})")

        if player['calculated_velocity'] is not None:
            calc_vx, calc_vy = player['calculated_velocity']
            calc_speed = player['calculated_speed']
            print(f"  Calculated Velocity: ({calc_vx:.1f}, {calc_vy:.1f})")
            print(f"  Calculated Speed: {calc_speed:.1f}")


def main():
    env = ShooterEnv(server_url="http://127.0.0.1:15702/")
    trainer = MultiAgentTrainer(obs_dim=7, act_dim=4)
    state = env.reset()
    obs = extract_obs(state)

    # Initialize episode logging
    episode_log = {
        'total_rewards': [],
        'episodes': [],
        'avg_losses': [],
        'player_stats': {},  # Will store {pid: {reward_history: [], avg_loss: [], ...}}
        'steps_alive': []
    }

    # Initialize player stats tracking
    for pid in obs:
        episode_log['player_stats'][pid] = {
            'reward_history': [],
            'loss_history': [],
            'survival_rates': [],
            'color': None
        }

    for episode in range(1, 10):
        total_reward = {pid: 0.0 for pid in obs}
        step_losses = {pid: [] for pid in obs}
        steps_alive = {pid: 0 for pid in obs}

        print(f"\n=== Starting Episode {episode} ===")
        print(f"Initial player states:")
        print_players(state.players)
        time.sleep(1.0)

        for t in range(1, 3):
            acts = trainer.select_actions(obs)
            next_state, rewards = env.step(acts)
            next_obs = extract_obs(next_state)
            losses = trainer.update(obs, acts, rewards, next_obs)

            # Update metrics
            for pid in obs:
                total_reward[pid] += rewards.get(pid, 0.0)
                step_losses[pid].append(losses.get(pid, 0.0))
                if next_state.players.get(pid, {}).get('health', 0) > 0:
                    steps_alive[pid] += 1

            #os.system('cls' if os.name == 'nt' else 'clear')

            # Enhanced Episode Header
            print(f"Episode {episode:03d} | Step {t:03d}")
            print("=" * 40)

            # Map Visualization
            # print_map_legend()
            # print(show_ascii_map(next_state))

            # Enhanced Agent Stats
            print("\n--- AGENT PERFORMANCE METRICS ---")
            if next_state.players:
                for pid, pdata in next_state.players.items():
                    r = rewards.get(pid, 0.0)
                    l = losses.get(pid, 0.0)
                    cum = total_reward.get(pid, 0.0)
                    hp = pdata['health']
                    alive_pct = (steps_alive[pid] / t) * 100

                    print(f"Player {pid} ({pdata['character'].value}):")
                    print(f"  Reward: {r:+.2f} (Cumulative: {cum:+.2f})")
                    print(f"  Health: {hp:.1f} | Alive: {alive_pct:.1f}% of steps")
                    print(f"  Loss: {l:.4f} (Avg: {np.mean(step_losses[pid]):.4f})")
                    print(f"  Actions: {'SHOOT' if pdata['is_shooting'] else ''} "
                          f"{'KICK' if pdata['is_kicking'] else ''} "
                          f"{'MOVE' if pdata['is_moving'] else ''}")
                    print("-" * 30)
            else:
                print(" No players data available.")

            # Enhanced Extras Section
            print("\n--- ENVIRONMENT ANALYSIS ---")
            if next_state.players and len(next_state.players) > 1:
                for player_id, pdata in next_state.players.items():
                    x, y = pdata['position']
                    best_d2, target = float('inf'), None

                    # Find nearest enemy
                    for qid, q in next_state.players.items():
                        if qid == player_id:
                            continue
                        dx = q['position'][0] - x
                        dy = q['position'][1] - y
                        d2 = dx * dx + dy * dy
                        if d2 < best_d2:
                            best_d2, target = d2, qid

                    dist = np.sqrt(best_d2) if best_d2 != float('inf') else None
                    pickups = len(next_state.pickup_positions)

                    print(f" Player {player_id}:")
                    if target is not None and dist is not None:
                        print(f"  Nearest enemy: {target} at {dist:.1f} units")
                    elif target is not None:
                        print(f"  Nearest enemy: {target} (distance not available)")
                    else:
                        print("  No enemies detected")
                    print(f"  Pickups remaining: {pickups}")
                    print(f"  Velocity: {pdata['velocity'][0]:.1f}, {pdata['velocity'][1]:.1f}")
            else:
                print(" No environmental data to display.")

            print("\n--- TRAINING PROGRESS ---")
            print(f"Current learning rate: {trainer.current_learning_rate:.2e}")
            print(f"Exploration rate: {trainer.exploration_rate:.2f}")

            time.sleep(env.poll_interval)
            obs = next_obs

        # Episode summary
        episode_log['total_rewards'].append(total_reward)
        episode_log['avg_losses'].append({pid: np.mean(l) for pid, l in step_losses.items()})
        episode_log['steps_alive'].append(steps_alive)

        episode_log['episodes'].append(episode)
        #os.system('cls' if os.name == 'nt' else 'clear')

        print("\n=== EPISODE LOG ===")
        print("=" * 40)
        for pid in obs:
            # Store player color on first encounter
            if episode_log['player_stats'][pid]['color'] is None:
                episode_log['player_stats'][pid]['color'] = state.players[pid]['character'].value

            # Update logs
            avg_loss = np.mean(step_losses[pid])
            survival_rate = (steps_alive[pid] / 200) * 100

            episode_log['player_stats'][pid]['reward_history'].append(total_reward[pid])
            episode_log['player_stats'][pid]['loss_history'].append(avg_loss)
            episode_log['player_stats'][pid]['survival_rates'].append(survival_rate)

            # Print current player log
            print(f"Player {pid} ({episode_log['player_stats'][pid]['color']}):")
            print(f"  Reward History: {episode_log['player_stats'][pid]['reward_history']}")
            print(f"  Avg Loss History: {[f'{x:.4f}%' for x in episode_log['player_stats'][pid]['loss_history']]}")
            print(f"  Survival Rates: {[f'{x:.1f}' for x in episode_log['player_stats'][pid]['survival_rates']]}")
            print("-" * 40)

            # Print formatted episode log table
        print("\n=== EPISODE LOG SUMMARY ===")
        print("Ep. | " + " | ".join(f"P{pid} Reward | P{pid} Loss | P{pid} Survival" for pid in obs))
        for i, ep in enumerate(episode_log['episodes']):
            row =[f"{ep:2d}"]
            for pid in obs:
                row.append(f"{episode_log['player_stats'][pid]['reward_history'][i]:+7.2f}")
                row.append(f"{episode_log['player_stats'][pid]['loss_history'][i]:7.4f}")
                row.append(f"{episode_log['player_stats'][pid]['survival_rates'][i]:6.1f}%")
            print(" | ".join(row))

        # Periodic training report
        if episode % 10 == 0:
            print("\n=== TRAINING REPORT (Last 10 Episodes) ===")
            recent_rewards = episode_log['total_rewards'][-10:]
            avg_rewards = {pid: np.mean([r[pid] for r in recent_rewards]) for pid in obs}
            for pid, avg in avg_rewards.items():
                print(f"Player {pid} Avg Reward: {avg:+.2f}")

        time.sleep(2.0)
        state = env.reset()
        obs = extract_obs(state)

if __name__ == "__main__":
    main()