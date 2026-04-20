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
from typing import Dict, List, Tuple

State = Tuple[int, int, int, int]


@dataclass
class InferenceConfig:
    q_table_path: str = "rl/q_table.pkl"


class TrafficSignalInference:
    def __init__(self, config: InferenceConfig | None = None):
        self.cfg = config or InferenceConfig()
        self.q_table: Dict[State, List[float]] = {}
        self.meta = {
            "lane_count": 4,
            "bucket_size": 3,
            "max_lane_count": 20,
            "service_capacity_per_step": 4,
        }

    def load(self) -> None:
        with open(self.cfg.q_table_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict) or "q_table" not in data:
            raise ValueError("Invalid Q-table file format. Expected dictionary with key 'q_table'.")

        self.q_table = data["q_table"]
        self.meta.update(data.get("meta", {}))

    def discretize_state(self, raw_counts: List[int]) -> State:
        bucket_size = int(self.meta.get("bucket_size", 3))
        max_lane_count = int(self.meta.get("max_lane_count", 20))

        discretized = []
        for c in raw_counts:
            capped = max(0, min(max_lane_count, int(c)))
            bucket = capped // bucket_size
            discretized.append(bucket)

        return tuple(discretized)  # type: ignore[return-value]

    def decide(self, raw_counts: List[int]) -> int:
        lane_count = int(self.meta.get("lane_count", 4))
        if len(raw_counts) != lane_count:
            raise ValueError(f"Expected {lane_count} lane counts, got {len(raw_counts)}")

        state = self.discretize_state(raw_counts)

        # If state was never seen during training, choose busiest lane as fallback.
        if state not in self.q_table:
            return max(range(lane_count), key=lambda i: raw_counts[i])

        q_values = self.q_table[state]
        if not q_values:
            return max(range(lane_count), key=lambda i: raw_counts[i])

        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]

        # Deterministic tie-breaker for reproducible behavior.
        return min(best_actions)


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
