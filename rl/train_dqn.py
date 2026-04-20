"""
rl/train_dqn.py

Compact DQN trainer for smart traffic signal control using the shared simulator.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.sim_env import SimConfig, TrafficSimEnv


@dataclass
class DQNConfig:
    episodes: int = 2500
    max_steps_per_episode: int = 120
    batch_size: int = 128
    replay_capacity: int = 50000
    learning_rate: float = 1e-3
    discount_factor: float = 0.97

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    target_update_steps: int = 250

    lane_count: int = 4
    max_lane_count: int = 20
    max_wait_age: int = 30

    service_capacity_per_step: int = 4
    arrival_min: int = 0
    arrival_max: int = 2
    min_green_steps: int = 3
    yellow_steps: int = 1

    hidden_size: int = 64

    queue_weight: float = 1.05
    queue_delta_weight: float = 1.55
    fairness_weight: float = 0.15
    wait_age_weight: float = 0.35
    switch_penalty_weight: float = 0.30
    invalid_action_penalty_weight: float = 0.50

    seed: int = 42


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNTrafficLearner:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.device = torch.device("cpu")

        self.env = TrafficSimEnv(
            SimConfig(
                lane_count=cfg.lane_count,
                max_lane_count=cfg.max_lane_count,
                service_capacity_per_step=cfg.service_capacity_per_step,
                arrival_min=cfg.arrival_min,
                arrival_max=cfg.arrival_max,
                min_green_steps=cfg.min_green_steps,
                yellow_steps=cfg.yellow_steps,
                max_wait_age=cfg.max_wait_age,
            )
        )

        self.input_dim = cfg.lane_count * 3 + 3
        self.online_net = QNetwork(self.input_dim, cfg.hidden_size, cfg.lane_count).to(self.device)
        self.target_net = QNetwork(self.input_dim, cfg.hidden_size, cfg.lane_count).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(
            maxlen=cfg.replay_capacity
        )
        self.global_step = 0

    def _state_vector(
        self,
        lane_counts: List[int],
        waiting_ages: List[int],
        previous_action: int,
        steps_since_switch: int,
        in_yellow: bool,
        can_switch: bool,
    ) -> np.ndarray:
        counts = np.array(lane_counts, dtype=np.float32) / max(1.0, float(self.cfg.max_lane_count))
        ages = np.array(waiting_ages, dtype=np.float32) / max(1.0, float(self.cfg.max_wait_age))

        prev_one_hot = np.zeros(self.cfg.lane_count, dtype=np.float32)
        prev_idx = max(0, min(self.cfg.lane_count - 1, int(previous_action)))
        prev_one_hot[prev_idx] = 1.0

        phase = np.array(
            [
                float(max(0, steps_since_switch)) / max(1.0, float(self.cfg.max_wait_age)),
                1.0 if in_yellow else 0.0,
                1.0 if can_switch else 0.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate([counts, ages, prev_one_hot, phase], axis=0)

    def _valid_action_mask(self, valid_actions: List[int]) -> np.ndarray:
        mask = np.zeros(self.cfg.lane_count, dtype=np.float32)
        if not valid_actions:
            mask[:] = 1.0
            return mask
        for idx in valid_actions:
            mask[max(0, min(self.cfg.lane_count - 1, int(idx)))] = 1.0
        return mask

    def _choose_action(self, state_vec: np.ndarray, valid_actions: List[int], epsilon: float) -> int:
        candidates = valid_actions if valid_actions else list(range(self.cfg.lane_count))
        if random.random() < epsilon:
            return int(random.choice(candidates))

        with torch.no_grad():
            state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t).squeeze(0).cpu().numpy()

        masked = q_values.copy()
        invalid = [idx for idx in range(self.cfg.lane_count) if idx not in candidates]
        for idx in invalid:
            masked[idx] = -1e9
        return int(np.argmax(masked))

    def _compute_reward(
        self,
        current_counts: List[int],
        next_counts: List[int],
        next_wait_ages: List[int],
        action: int,
        previous_action: Optional[int],
        in_yellow: bool,
        invalid_request: bool,
    ) -> float:
        queue_after = float(sum(next_counts))
        queue_delta = float(sum(current_counts) - sum(next_counts))

        mean = sum(next_counts) / max(1, len(next_counts))
        variance = sum((float(x) - mean) ** 2 for x in next_counts) / max(1, len(next_counts))
        imbalance = variance ** 0.5

        avg_wait_age = float(sum(next_wait_ages)) / max(1, len(next_wait_ages))

        switched = 1.0 if previous_action is not None and action != previous_action else 0.0
        yellow_penalty = 0.25 if in_yellow else 0.0
        invalid_penalty = self.cfg.invalid_action_penalty_weight if invalid_request else 0.0

        return (
            -(self.cfg.queue_weight * queue_after)
            + (self.cfg.queue_delta_weight * queue_delta)
            - (self.cfg.fairness_weight * imbalance)
            - (self.cfg.wait_age_weight * avg_wait_age)
            - (self.cfg.switch_penalty_weight * switched)
            - yellow_penalty
            - invalid_penalty
        )

    def _optimize_step(self) -> Optional[float]:
        if len(self.replay) < self.cfg.batch_size:
            return None

        batch = random.sample(self.replay, self.cfg.batch_size)

        states = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_masks = torch.tensor(np.stack([b[5] for b in batch]), dtype=torch.float32, device=self.device)

        q_pred = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            target_q_all = self.target_net(next_states)
            masked_target_q = target_q_all.masked_fill(next_masks <= 0, -1e9)
            max_next_q = masked_target_q.max(dim=1, keepdim=True).values
            max_next_q = torch.where(torch.isfinite(max_next_q), max_next_q, torch.zeros_like(max_next_q))
            q_target = rewards + (1.0 - dones) * self.cfg.discount_factor * max_next_q

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.cfg.target_update_steps == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def train(self) -> None:
        epsilon = self.cfg.epsilon_start
        rng = random.Random(self.cfg.seed)

        for episode in range(1, self.cfg.episodes + 1):
            self.env.reset(rng)

            lane_counts = self.env.lane_counts.copy()
            waiting_ages = self.env.waiting_ages.copy()
            previous_action = int(self.env.current_green)

            state_vec = self._state_vector(
                lane_counts,
                waiting_ages,
                previous_action,
                self.env.steps_since_switch,
                self.env.yellow_remaining > 0,
                self.env.can_switch(),
            )

            last_loss: Optional[float] = None

            for step in range(self.cfg.max_steps_per_episode):
                valid_actions = self.env.valid_actions()
                action = self._choose_action(state_vec, valid_actions, epsilon)

                info = self.env.step(action, rng)
                next_counts = info["lane_counts"]
                next_wait_ages = info["waiting_ages"]
                effective_action = int(info["current_green"])

                reward = self._compute_reward(
                    lane_counts,
                    next_counts,
                    next_wait_ages,
                    effective_action,
                    previous_action,
                    bool(info["in_yellow"]),
                    bool(info["invalid_request"]),
                )

                next_state_vec = self._state_vector(
                    next_counts,
                    next_wait_ages,
                    effective_action,
                    int(info["steps_since_switch"]),
                    bool(info["in_yellow"]),
                    bool(info["can_switch"]),
                )

                done = step == (self.cfg.max_steps_per_episode - 1)
                next_mask = self._valid_action_mask(list(info["valid_actions"]))
                self.replay.append((state_vec, action, reward, next_state_vec, done, next_mask))

                loss = self._optimize_step()
                if loss is not None:
                    last_loss = loss

                lane_counts = next_counts
                waiting_ages = next_wait_ages
                previous_action = effective_action
                state_vec = next_state_vec

            epsilon = max(self.cfg.epsilon_min, epsilon * self.cfg.epsilon_decay)

            if episode % 250 == 0 or episode == 1:
                loss_text = "n/a" if last_loss is None else f"{last_loss:.4f}"
                print(
                    f"Episode {episode}/{self.cfg.episodes} | epsilon={epsilon:.4f} | "
                    f"replay={len(self.replay)} | loss={loss_text}"
                )

    def export_model(self, output_path: str) -> None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload: Dict[str, object] = {
            "state_dict": self.online_net.state_dict(),
            "meta": {
                "lane_count": self.cfg.lane_count,
                "max_lane_count": self.cfg.max_lane_count,
                "max_wait_age": self.cfg.max_wait_age,
                "hidden_size": self.cfg.hidden_size,
                "service_capacity_per_step": self.cfg.service_capacity_per_step,
                "arrival_min": self.cfg.arrival_min,
                "arrival_max": self.cfg.arrival_max,
                "min_green_steps": self.cfg.min_green_steps,
                "yellow_steps": self.cfg.yellow_steps,
                "seed": self.cfg.seed,
            },
        }
        torch.save(payload, output_path)
        print(f"Saved DQN model to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN model for traffic signal control")
    parser.add_argument("--episodes", type=int, default=2500, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=120, help="Max steps per episode")
    parser.add_argument("--batch-size", type=int, default=128, help="Replay batch size")
    parser.add_argument("--replay-capacity", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")

    parser.add_argument("--arrival-min", type=int, default=0, help="Minimum arrivals per lane per step")
    parser.add_argument("--arrival-max", type=int, default=1, help="Maximum arrivals per lane per step")
    parser.add_argument("--service-capacity", type=int, default=5, help="Vehicles served per step on active green")
    parser.add_argument("--min-green-steps", type=int, default=3, help="Minimum green hold steps")
    parser.add_argument("--yellow-steps", type=int, default=0, help="Yellow transition steps")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="rl/dqn_model.pt", help="Output model path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DQNConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        arrival_min=args.arrival_min,
        arrival_max=args.arrival_max,
        service_capacity_per_step=args.service_capacity,
        min_green_steps=args.min_green_steps,
        yellow_steps=args.yellow_steps,
        seed=args.seed,
    )

    learner = DQNTrafficLearner(cfg)
    print("Starting DQN training...")
    learner.train()
    learner.export_model(args.output)
    print("DQN training complete.")


if __name__ == "__main__":
    main()
