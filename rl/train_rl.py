"""
rl/train_rl.py

Q-learning training for smart traffic signal control.
- State: 4 lane vehicle counts (discretized)
- Actions: 0,1,2,3 (lane index to set GREEN)
- Reward: negative total waiting vehicles after serving selected lane
- Trains on simulated random traffic and saves Q-table to pickle
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

State = Tuple[int, int, int, int]
QTable = Dict[State, List[float]]


@dataclass
class TrainConfig:
    episodes: int = 5000
    max_steps_per_episode: int = 50
    learning_rate: float = 0.12
    discount_factor: float = 0.92
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    lane_count: int = 4
    max_lane_count: int = 20
    bucket_size: int = 3

    # When a lane gets green, this many vehicles can pass in one step.
    service_capacity_per_step: int = 4


class TrafficQLearner:
    def __init__(self, config: TrainConfig):
        self.cfg = config

        # New states default to 0.0 Q-value for each action.
        self.q_table: DefaultDict[State, List[float]] = defaultdict(
            lambda: [0.0] * self.cfg.lane_count
        )

    def discretize_state(self, raw_counts: List[int]) -> State:
        """
        Convert raw lane counts into discrete buckets for a compact Q-table.
        Example with bucket_size=3: 0..2 -> 0, 3..5 -> 1, etc.
        """
        discretized = []
        for c in raw_counts:
            capped = max(0, min(self.cfg.max_lane_count, c))
            bucket = capped // self.cfg.bucket_size
            discretized.append(bucket)
        return tuple(discretized)  # type: ignore[return-value]

    def undiscretized_random_state(self) -> List[int]:
        return [random.randint(0, self.cfg.max_lane_count) for _ in range(self.cfg.lane_count)]

    def choose_action(self, state: State, epsilon: float) -> int:
        # Epsilon-greedy: explore with probability epsilon.
        if random.random() < epsilon:
            return random.randint(0, self.cfg.lane_count - 1)

        q_values = self.q_table[state]
        max_q = max(q_values)

        # Break ties randomly for stable learning behavior.
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def step_environment(self, lane_counts: List[int], action: int) -> Tuple[List[int], float]:
        """
        Simulated traffic dynamics for one step:
        1) Selected green lane serves vehicles (decreases queue)
        2) New vehicles arrive randomly in all lanes
        3) Reward is negative total queue after transition
        """
        next_counts = lane_counts.copy()

        # Serve selected lane
        next_counts[action] = max(0, next_counts[action] - self.cfg.service_capacity_per_step)

        # Random arrivals (0..2 vehicles per lane per step)
        for i in range(self.cfg.lane_count):
            next_counts[i] += random.randint(0, 2)
            next_counts[i] = min(next_counts[i], self.cfg.max_lane_count)

        waiting_sum = sum(next_counts)
        reward = -float(waiting_sum)
        return next_counts, reward

    def train(self) -> None:
        epsilon = self.cfg.epsilon_start

        for episode in range(1, self.cfg.episodes + 1):
            lane_counts = self.undiscretized_random_state()
            state = self.discretize_state(lane_counts)

            for _ in range(self.cfg.max_steps_per_episode):
                action = self.choose_action(state, epsilon)
                next_counts, reward = self.step_environment(lane_counts, action)
                next_state = self.discretize_state(next_counts)

                # Q-learning update rule
                old_q = self.q_table[state][action]
                best_next_q = max(self.q_table[next_state])
                target = reward + self.cfg.discount_factor * best_next_q
                updated_q = old_q + self.cfg.learning_rate * (target - old_q)
                self.q_table[state][action] = updated_q

                lane_counts = next_counts
                state = next_state

            epsilon = max(self.cfg.epsilon_min, epsilon * self.cfg.epsilon_decay)

            if episode % 500 == 0 or episode == 1:
                print(
                    f"Episode {episode}/{self.cfg.episodes} | "
                    f"epsilon={epsilon:.4f} | "
                    f"states_learned={len(self.q_table)}"
                )

    def export_q_table(self, output_path: str) -> None:
        data = {
            "q_table": dict(self.q_table),
            "meta": {
                "lane_count": self.cfg.lane_count,
                "bucket_size": self.cfg.bucket_size,
                "max_lane_count": self.cfg.max_lane_count,
                "service_capacity_per_step": self.cfg.service_capacity_per_step,
            },
        }

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved Q-table to: {output_path}")
        print(f"Total learned states: {len(self.q_table)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-learning model for traffic signal control")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument(
        "--output",
        type=str,
        default="rl/q_table.pkl",
        help="Path to save trained Q-table pickle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(episodes=args.episodes)
    learner = TrafficQLearner(cfg)

    print("Starting Q-learning training...")
    learner.train()
    learner.export_q_table(args.output)
    print("Training complete.")


if __name__ == "__main__":
    main()
