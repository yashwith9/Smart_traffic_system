"""
rl/train_rl.py

Q-learning training for smart traffic signal control.
State includes:
- lane queue buckets (4)
- lane waiting-age buckets (4)
- previous green lane action (1)

Signal dynamics include:
- minimum green duration
- yellow transition on switching
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple

from rl.sim_env import SimConfig, TrafficSimEnv

State = Tuple[int, ...]
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
    age_bucket_size: int = 3
    phase_bucket_size: int = 2

    service_capacity_per_step: int = 4
    arrival_min: int = 0
    arrival_max: int = 2
    min_green_steps: int = 3
    yellow_steps: int = 1
    max_wait_age: int = 30

    queue_weight: float = 1.0
    queue_delta_weight: float = 1.2
    fairness_weight: float = 0.25
    wait_age_weight: float = 0.15
    switch_penalty_weight: float = 0.2
    invalid_action_penalty_weight: float = 0.75

    seed: int = 42


class TrafficQLearner:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        random.seed(self.cfg.seed)

        self.env = TrafficSimEnv(
            SimConfig(
                lane_count=self.cfg.lane_count,
                max_lane_count=self.cfg.max_lane_count,
                service_capacity_per_step=self.cfg.service_capacity_per_step,
                arrival_min=self.cfg.arrival_min,
                arrival_max=self.cfg.arrival_max,
                min_green_steps=self.cfg.min_green_steps,
                yellow_steps=self.cfg.yellow_steps,
                max_wait_age=self.cfg.max_wait_age,
            )
        )

        self.q_table: DefaultDict[State, List[float]] = defaultdict(
            lambda: [0.0] * self.cfg.lane_count
        )

    def discretize_state(
        self,
        raw_counts: List[int],
        waiting_ages: Optional[List[int]] = None,
        previous_action: Optional[int] = None,
        steps_since_switch: int = 0,
        in_yellow: bool = False,
        can_switch: bool = True,
    ) -> State:
        discretized: List[int] = []

        for count in raw_counts:
            capped = max(0, min(self.cfg.max_lane_count, count))
            discretized.append(capped // self.cfg.bucket_size)

        ages = waiting_ages or [0] * self.cfg.lane_count
        for age in ages:
            capped_age = max(0, min(self.cfg.max_wait_age, age))
            discretized.append(capped_age // self.cfg.age_bucket_size)

        prev = 0 if previous_action is None else max(0, min(self.cfg.lane_count - 1, previous_action))
        discretized.append(prev)

        capped_steps = max(0, min(self.cfg.max_wait_age, steps_since_switch))
        discretized.append(capped_steps // max(1, self.cfg.phase_bucket_size))
        discretized.append(1 if in_yellow else 0)
        discretized.append(1 if can_switch else 0)
        return tuple(discretized)

    def choose_action(self, state: State, epsilon: float, valid_actions: Optional[List[int]] = None) -> int:
        candidates = valid_actions or list(range(self.cfg.lane_count))
        if not candidates:
            candidates = list(range(self.cfg.lane_count))

        if random.random() < epsilon:
            return random.choice(candidates)

        q_values = self.q_table[state]
        candidate_q_values = [q_values[idx] for idx in candidates]
        max_q = max(candidate_q_values)
        best_actions = [idx for idx in candidates if q_values[idx] == max_q]
        return random.choice(best_actions)

    def compute_reward(
        self,
        current_counts: List[int],
        next_counts: List[int],
        current_wait_ages: Optional[List[int]] = None,
        next_wait_ages: Optional[List[int]] = None,
        action: int = 0,
        requested_action: Optional[int] = None,
        previous_action: Optional[int] = None,
        in_yellow: bool = False,
        invalid_request: bool = False,
    ) -> float:
        queue_after = float(sum(next_counts))
        queue_delta = float(sum(current_counts) - sum(next_counts))
        imbalance = float(statistics.pstdev(next_counts))

        ages = next_wait_ages or [0] * self.cfg.lane_count
        avg_wait_age = float(sum(ages)) / max(1, len(ages))

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

    def step_environment(
        self,
        lane_counts: List[int],
        action: int,
        previous_action: Optional[int] = None,
    ) -> Tuple[List[int], float]:
        """Backward-compatible one-step wrapper used by tests."""
        self.env.lane_counts = lane_counts.copy()
        self.env.waiting_ages = [0] * self.cfg.lane_count
        self.env.current_green = previous_action if previous_action is not None else 0
        self.env.steps_since_switch = self.cfg.min_green_steps
        self.env.yellow_remaining = 0
        self.env.pending_green = None

        info = self.env.step(action, random)
        next_counts = info["lane_counts"]
        reward = self.compute_reward(
            lane_counts,
            next_counts,
            next_wait_ages=info["waiting_ages"],
            action=action,
            requested_action=action,
            previous_action=previous_action,
            in_yellow=bool(info["in_yellow"]),
            invalid_request=bool(info["invalid_request"]),
        )
        return next_counts, reward

    def train(self) -> None:
        epsilon = self.cfg.epsilon_start
        rng = random.Random(self.cfg.seed)

        for episode in range(1, self.cfg.episodes + 1):
            self.env.reset(rng)

            lane_counts = self.env.lane_counts.copy()
            waiting_ages = self.env.waiting_ages.copy()
            previous_action: Optional[int] = self.env.current_green

            state = self.discretize_state(
                lane_counts,
                waiting_ages,
                previous_action,
                self.env.steps_since_switch,
                self.env.yellow_remaining > 0,
                self.env.can_switch(),
            )

            for _ in range(self.cfg.max_steps_per_episode):
                valid_actions = self.env.valid_actions()
                requested_action = self.choose_action(state, epsilon, valid_actions)
                info = self.env.step(requested_action, rng)

                next_counts = info["lane_counts"]
                next_wait_ages = info["waiting_ages"]
                effective_action = int(info["current_green"])

                reward = self.compute_reward(
                    lane_counts,
                    next_counts,
                    current_wait_ages=waiting_ages,
                    next_wait_ages=next_wait_ages,
                    action=effective_action,
                    requested_action=requested_action,
                    previous_action=previous_action,
                    in_yellow=bool(info["in_yellow"]),
                    invalid_request=bool(info["invalid_request"]),
                )

                next_state = self.discretize_state(
                    next_counts,
                    next_wait_ages,
                    effective_action,
                    int(info["steps_since_switch"]),
                    bool(info["in_yellow"]),
                    bool(info["can_switch"]),
                )

                old_q = self.q_table[state][requested_action]
                best_next_q = max(self.q_table[next_state])
                target = reward + self.cfg.discount_factor * best_next_q
                updated_q = old_q + self.cfg.learning_rate * (target - old_q)
                self.q_table[state][requested_action] = updated_q

                lane_counts = next_counts
                waiting_ages = next_wait_ages
                previous_action = effective_action
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
                "age_bucket_size": self.cfg.age_bucket_size,
                "phase_bucket_size": self.cfg.phase_bucket_size,
                "max_lane_count": self.cfg.max_lane_count,
                "service_capacity_per_step": self.cfg.service_capacity_per_step,
                "arrival_min": self.cfg.arrival_min,
                "arrival_max": self.cfg.arrival_max,
                "min_green_steps": self.cfg.min_green_steps,
                "yellow_steps": self.cfg.yellow_steps,
                "max_wait_age": self.cfg.max_wait_age,
                "queue_weight": self.cfg.queue_weight,
                "queue_delta_weight": self.cfg.queue_delta_weight,
                "fairness_weight": self.cfg.fairness_weight,
                "wait_age_weight": self.cfg.wait_age_weight,
                "switch_penalty_weight": self.cfg.switch_penalty_weight,
                "invalid_action_penalty_weight": self.cfg.invalid_action_penalty_weight,
                "seed": self.cfg.seed,
            },
        }

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "wb") as file_handle:
            pickle.dump(data, file_handle)

        print(f"Saved Q-table to: {output_path}")
        print(f"Total learned states: {len(self.q_table)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-learning model for traffic signal control")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")

    parser.add_argument("--age-bucket-size", type=int, default=3, help="Bucket size for waiting-age discretization")
    parser.add_argument("--phase-bucket-size", type=int, default=2, help="Bucket size for signal phase timer")
    parser.add_argument("--arrival-min", type=int, default=0, help="Minimum random arrivals per lane per step")
    parser.add_argument("--arrival-max", type=int, default=2, help="Maximum random arrivals per lane per step")
    parser.add_argument("--min-green-steps", type=int, default=3, help="Minimum steps to hold green before switching")
    parser.add_argument("--yellow-steps", type=int, default=1, help="Yellow transition steps during switching")
    parser.add_argument("--max-wait-age", type=int, default=30, help="Maximum waiting-age tracked per lane")

    parser.add_argument("--queue-weight", type=float, default=1.0, help="Weight for queue size penalty")
    parser.add_argument("--queue-delta-weight", type=float, default=1.2, help="Weight for queue reduction reward")
    parser.add_argument("--fairness-weight", type=float, default=0.25, help="Weight for queue imbalance penalty")
    parser.add_argument("--wait-age-weight", type=float, default=0.15, help="Weight for waiting-age penalty")
    parser.add_argument("--switch-penalty-weight", type=float, default=0.2, help="Penalty for changing green lane")
    parser.add_argument("--invalid-action-penalty-weight", type=float, default=0.75, help="Penalty for invalid switch requests")

    parser.add_argument(
        "--output",
        type=str,
        default="rl/q_table.pkl",
        help="Path to save trained Q-table pickle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        episodes=args.episodes,
        seed=args.seed,
        age_bucket_size=args.age_bucket_size,
        phase_bucket_size=args.phase_bucket_size,
        arrival_min=args.arrival_min,
        arrival_max=args.arrival_max,
        min_green_steps=args.min_green_steps,
        yellow_steps=args.yellow_steps,
        max_wait_age=args.max_wait_age,
        queue_weight=args.queue_weight,
        queue_delta_weight=args.queue_delta_weight,
        fairness_weight=args.fairness_weight,
        wait_age_weight=args.wait_age_weight,
        switch_penalty_weight=args.switch_penalty_weight,
        invalid_action_penalty_weight=args.invalid_action_penalty_weight,
    )
    learner = TrafficQLearner(cfg)

    print("Starting Q-learning training...")
    learner.train()
    learner.export_q_table(args.output)
    print("Training complete.")


if __name__ == "__main__":
    main()
