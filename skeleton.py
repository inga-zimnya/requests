
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests

from main import (
    fetch_game_state,
    ParsedGameState,
    show_ascii_map,
    print_map_legend,
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
    def __init__(self, obs_dim, act_dim, lr=1e-3):
        self.policies = {}
        self.optimizers = {}
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr

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
            logits = self.policies[pid](tensor)
            out = torch.tanh(logits[:, :2]).detach().numpy().flatten()
            shoot = (torch.sigmoid(logits[:, 2]) > 0.5).item()
            kick = (torch.sigmoid(logits[:, 3]) > 0.5).item()
            acts[pid] = np.array([*out, shoot, kick], dtype=np.float32)
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

def main():
    env = ShooterEnv(server_url="http://127.0.0.1:15702/")
    trainer = MultiAgentTrainer(obs_dim=7, act_dim=4)
    state = env.reset()
    obs = extract_obs(state)
    for episode in range(1, 501):
        total_reward = {pid: 0.0 for pid in obs}
        print(f"\n=== Starting Episode {episode} ===")
        time.sleep(1.0)
        for t in range(1, 201):
            acts = trainer.select_actions(obs)
            next_state, rewards = env.step(acts)
            next_obs = extract_obs(next_state)
            losses = trainer.update(obs, acts, rewards, next_obs)
            for pid, r in rewards.items():
                total_reward[pid] += r
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Episode {episode:03d} | Step {t:03d}")
            print_map_legend()
            print(show_ascii_map(next_state))

            # Agent Stats
            print("\n--- Agent Stats ---")
            print(f"next_state.players : {next_state.players}")
            if next_state.players:
                for pid, pdata in next_state.players.items():
                    r = rewards.get(pid, 0.0)
                    l = losses.get(pid, 0.0)
                    cum = total_reward.get(pid, 0.0)
                    hp = pdata['health']
                    print(f"Player {pid}: ΔR={r:+.2f} | Loss={l:.4f} | CumR={cum:+.2f} | HP={hp:.1f}")
            else:
                print(" No players data available.")

            # Extras
            print("\n--- Extras ---")
            if next_state.players and len(next_state.players) > 1:
                for pid, pdata in next_state.players.items():
                    x, y = pdata['position']
                    best_d2, target = float('inf'), None
                    for qid, q in next_state.players.items():
                        if qid == pid:
                            continue
                        dx = q['position'][0] - x
                        dy = q['position'][1] - y
                        d2 = dx*dx + dy*dy
                        if d2 < best_d2:
                            best_d2, target = d2, qid
                    dist = np.sqrt(best_d2) if target else None
                    pickups = len(next_state.pickup_positions)
                    print(f" Player {pid}: nearest enemy={target} at {dist} units; pickups left={pickups}")
            else:
                print(" No extras to show.")

            time.sleep(env.poll_interval)
            obs = next_obs

        print(f"\n>>> Episode {episode} completed. Total rewards: " +
              ", ".join(f"{pid}:{r:+.2f}" for pid, r in total_reward.items()))
        time.sleep(2.0)
        state = env.reset()
        obs = extract_obs(state)

if __name__ == "__main__":
    main()