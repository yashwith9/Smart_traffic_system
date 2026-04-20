"""
rl/infer.py

Inference module for smart traffic signal control using a trained Q-table.
- Loads Q-table from pickle
- Accepts lane counts state [l1, l2, l3, l4]
- Returns best action (0,1,2,3) where action = lane to set GREEN
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

State = Tuple[int, ...]


@dataclass
class InferenceConfig:
    q_table_path: str = "rl/q_table.pkl"


class TrafficSignalInference:
    def __init__(self, config: InferenceConfig | None = None):
        self.cfg = config or InferenceConfig()
        self.q_table: Dict[State, List[float]] = {}
        self.state_dimensions = 4
        self.meta = {
            "lane_count": 4,
            "bucket_size": 3,
            "age_bucket_size": 3,
            "phase_bucket_size": 2,
            "max_lane_count": 20,
            "max_wait_age": 30,
            "service_capacity_per_step": 4,
            "min_green_steps": 3,
        }
        self.previous_action = 0
        self.waiting_ages = [0, 0, 0, 0]
        self.steps_since_switch = 0
        self.in_yellow = False

    def load(self) -> None:
        with open(self.cfg.q_table_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict) or "q_table" not in data:
            raise ValueError("Invalid Q-table file format. Expected dictionary with key 'q_table'.")

        self.q_table = data["q_table"]
        self.meta.update(data.get("meta", {}))

        # Determine state dimensionality dynamically for backward compatibility.
        if self.q_table:
            sample_state = next(iter(self.q_table.keys()))
            self.state_dimensions = len(sample_state)
        else:
            self.state_dimensions = 4

        lane_count = int(self.meta.get("lane_count", 4))
        self.previous_action = 0
        self.waiting_ages = [0] * lane_count
        self.steps_since_switch = 0
        self.in_yellow = False

    def discretize_state(self, raw_counts: List[int]) -> State:
        bucket_size = int(self.meta.get("bucket_size", 3))
        max_lane_count = int(self.meta.get("max_lane_count", 20))

        discretized = []
        for c in raw_counts:
            capped = max(0, min(max_lane_count, int(c)))
            bucket = capped // bucket_size
            discretized.append(bucket)

        return tuple(discretized)

    def discretize_state_extended(
        self,
        raw_counts: List[int],
        waiting_ages: List[int],
        previous_action: int,
        steps_since_switch: int,
        in_yellow: bool,
        can_switch: bool,
    ) -> State:
        bucket_size = int(self.meta.get("bucket_size", 3))
        age_bucket_size = int(self.meta.get("age_bucket_size", 3))
        phase_bucket_size = int(self.meta.get("phase_bucket_size", 2))
        max_lane_count = int(self.meta.get("max_lane_count", 20))
        max_wait_age = int(self.meta.get("max_wait_age", 30))
        lane_count = int(self.meta.get("lane_count", 4))

        state_values: List[int] = []
        for count in raw_counts:
            capped = max(0, min(max_lane_count, int(count)))
            state_values.append(capped // max(1, bucket_size))

        for age in waiting_ages:
            capped_age = max(0, min(max_wait_age, int(age)))
            state_values.append(capped_age // max(1, age_bucket_size))

        state_values.append(max(0, min(lane_count - 1, int(previous_action))))
        capped_steps = max(0, min(max_wait_age, int(steps_since_switch)))
        state_values.append(capped_steps // max(1, phase_bucket_size))
        state_values.append(1 if in_yellow else 0)
        state_values.append(1 if can_switch else 0)
        return tuple(state_values)

    def discretize_state_extended_legacy(
        self,
        raw_counts: List[int],
        waiting_ages: List[int],
        previous_action: int,
    ) -> State:
        bucket_size = int(self.meta.get("bucket_size", 3))
        age_bucket_size = int(self.meta.get("age_bucket_size", 3))
        max_lane_count = int(self.meta.get("max_lane_count", 20))
        max_wait_age = int(self.meta.get("max_wait_age", 30))
        lane_count = int(self.meta.get("lane_count", 4))

        state_values: List[int] = []
        for count in raw_counts:
            capped = max(0, min(max_lane_count, int(count)))
            state_values.append(capped // max(1, bucket_size))

        for age in waiting_ages:
            capped_age = max(0, min(max_wait_age, int(age)))
            state_values.append(capped_age // max(1, age_bucket_size))

        state_values.append(max(0, min(lane_count - 1, int(previous_action))))
        return tuple(state_values)

    def _best_action_for_state(self, state: State, raw_counts: List[int], valid_actions: Optional[List[int]] = None) -> int:
        lane_count = int(self.meta.get("lane_count", 4))
        candidates = valid_actions or list(range(lane_count))
        if not candidates:
            candidates = list(range(lane_count))

        if state not in self.q_table:
            return max(candidates, key=lambda idx: raw_counts[idx])

        q_values = self.q_table[state]
        if not q_values:
            return max(candidates, key=lambda idx: raw_counts[idx])

        candidate_q_values = [q_values[idx] for idx in candidates]
        max_q = max(candidate_q_values)
        best_actions = [idx for idx in candidates if q_values[idx] == max_q]
        return min(best_actions)

    def decide_with_context(
        self,
        raw_counts: List[int],
        waiting_ages: List[int],
        previous_action: int,
        steps_since_switch: int = 0,
        in_yellow: bool = False,
        can_switch: bool = True,
        valid_actions: Optional[List[int]] = None,
    ) -> int:
        lane_count = int(self.meta.get("lane_count", 4))
        if len(raw_counts) != lane_count:
            raise ValueError(f"Expected {lane_count} lane counts, got {len(raw_counts)}")

        if self.state_dimensions <= lane_count:
            state = self.discretize_state(raw_counts)
        elif self.state_dimensions == (lane_count * 2 + 1):
            state = self.discretize_state_extended_legacy(raw_counts, waiting_ages, previous_action)
        else:
            state = self.discretize_state_extended(
                raw_counts,
                waiting_ages,
                previous_action,
                steps_since_switch,
                in_yellow,
                can_switch,
            )

        action = self._best_action_for_state(state, raw_counts, valid_actions)
        if action == previous_action:
            self.steps_since_switch = max(0, steps_since_switch + 1)
        else:
            self.steps_since_switch = 0
        self.previous_action = action
        self.waiting_ages = waiting_ages.copy()
        self.in_yellow = in_yellow
        return action

    def decide(self, raw_counts: List[int]) -> int:
        lane_count = int(self.meta.get("lane_count", 4))
        if len(raw_counts) != lane_count:
            raise ValueError(f"Expected {lane_count} lane counts, got {len(raw_counts)}")

        # Legacy model with queue-only state.
        if self.state_dimensions <= lane_count:
            state = self.discretize_state(raw_counts)
            action = self._best_action_for_state(state, raw_counts)
            self.previous_action = action
            return action

        # Extended model: infer waiting ages online from observation history.
        max_wait_age = int(self.meta.get("max_wait_age", 30))
        min_green_steps = int(self.meta.get("min_green_steps", 3))
        updated_ages: List[int] = []
        for idx, count in enumerate(raw_counts):
            if count <= 0:
                updated_ages.append(0)
            elif idx == self.previous_action:
                updated_ages.append(max(0, self.waiting_ages[idx] - 1))
            else:
                updated_ages.append(min(max_wait_age, self.waiting_ages[idx] + 1))

        can_switch = (not self.in_yellow) and (self.steps_since_switch >= min_green_steps)
        if not can_switch:
            valid_actions = [self.previous_action]
        else:
            valid_actions = list(range(lane_count))

        return self.decide_with_context(
            raw_counts,
            updated_ages,
            self.previous_action,
            steps_since_switch=self.steps_since_switch,
            in_yellow=self.in_yellow,
            can_switch=can_switch,
            valid_actions=valid_actions,
        )


def parse_state(state_str: str) -> List[int]:
    """
    Parse CSV lane counts, e.g. "4,7,2,9" -> [4, 7, 2, 9].
    """
    parts = [p.strip() for p in state_str.split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise ValueError("State must contain exactly 4 comma-separated integers, e.g. 4,7,2,9")
    return [int(x) for x in parts]


def action_to_text(action: int) -> str:
    return f"Lane {action + 1} GREEN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using trained Q-table")
    parser.add_argument(
        "--model",
        type=str,
        default="rl/q_table.pkl",
        help="Path to Q-table pickle file",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="5,8,2,4",
        help="Lane counts as CSV: l1,l2,l3,l4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_state = parse_state(args.state)

    infer = TrafficSignalInference(InferenceConfig(q_table_path=args.model))

    try:
        infer.load()
    except FileNotFoundError:
        print(f"Model file not found: {args.model}")
        print("Train first using: python rl/train_rl.py")
        return

    action = infer.decide(raw_state)

    print(f"Input lane counts: {raw_state}")
    print(f"Selected action: {action}")
    print(f"Decision: {action_to_text(action)}")


if __name__ == "__main__":
    main()
