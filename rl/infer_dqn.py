"""
rl/infer_dqn.py

Inference module for DQN-based traffic signal control.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.train_dqn import QNetwork


@dataclass
class DQNInferenceConfig:
    model_path: str = "rl/dqn_model.pt"


class TrafficDQNInference:
    def __init__(self, config: DQNInferenceConfig | None = None):
        self.cfg = config or DQNInferenceConfig()
        self.meta: Dict[str, int] = {
            "lane_count": 4,
            "max_lane_count": 20,
            "max_wait_age": 30,
            "hidden_size": 64,
            "min_green_steps": 3,
        }
        self.device = torch.device("cpu")
        self.net: Optional[QNetwork] = None

    def load(self) -> None:
        payload = torch.load(self.cfg.model_path, map_location=self.device)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError("Invalid DQN model format. Expected dict with 'state_dict'.")

        meta_raw = payload.get("meta", {})
        if isinstance(meta_raw, dict):
            for key, val in meta_raw.items():
                if isinstance(val, (int, float)):
                    self.meta[key] = int(val)

        lane_count = int(self.meta.get("lane_count", 4))
        hidden_size = int(self.meta.get("hidden_size", 64))
        input_dim = lane_count * 3 + 3

        self.net = QNetwork(input_dim, hidden_size, lane_count).to(self.device)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()

    def _state_vector(
        self,
        lane_counts: List[int],
        waiting_ages: List[int],
        previous_action: int,
        steps_since_switch: int,
        in_yellow: bool,
        can_switch: bool,
    ) -> np.ndarray:
        lane_count = int(self.meta.get("lane_count", 4))
        max_lane_count = int(self.meta.get("max_lane_count", 20))
        max_wait_age = int(self.meta.get("max_wait_age", 30))

        counts = np.array(lane_counts, dtype=np.float32) / max(1.0, float(max_lane_count))
        ages = np.array(waiting_ages, dtype=np.float32) / max(1.0, float(max_wait_age))

        prev_one_hot = np.zeros(lane_count, dtype=np.float32)
        prev_idx = max(0, min(lane_count - 1, int(previous_action)))
        prev_one_hot[prev_idx] = 1.0

        phase = np.array(
            [
                float(max(0, steps_since_switch)) / max(1.0, float(max_wait_age)),
                1.0 if in_yellow else 0.0,
                1.0 if can_switch else 0.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate([counts, ages, prev_one_hot, phase], axis=0)

    def decide_with_context(
        self,
        raw_counts: List[int],
        waiting_ages: List[int],
        previous_action: int,
        steps_since_switch: int,
        in_yellow: bool,
        can_switch: bool,
        valid_actions: Optional[List[int]] = None,
    ) -> int:
        if self.net is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        lane_count = int(self.meta.get("lane_count", 4))
        if len(raw_counts) != lane_count:
            raise ValueError(f"Expected {lane_count} lane counts, got {len(raw_counts)}")

        if valid_actions is None or len(valid_actions) == 0:
            valid_actions = list(range(lane_count))

        state_vec = self._state_vector(
            raw_counts,
            waiting_ages,
            previous_action,
            steps_since_switch,
            in_yellow,
            can_switch,
        )

        with torch.no_grad():
            q_values = self.net(torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()

        masked = q_values.copy()
        for idx in range(lane_count):
            if idx not in valid_actions:
                masked[idx] = -1e9
        return int(np.argmax(masked))


def parse_state(state_str: str) -> List[int]:
    parts = [p.strip() for p in state_str.split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise ValueError("State must contain exactly 4 comma-separated integers, e.g. 4,7,2,9")
    return [int(x) for x in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using trained DQN model")
    parser.add_argument("--model", type=str, default="rl/dqn_model.pt", help="Path to DQN model file")
    parser.add_argument("--state", type=str, default="5,8,2,4", help="Lane counts CSV: l1,l2,l3,l4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_state = parse_state(args.state)

    infer = TrafficDQNInference(DQNInferenceConfig(model_path=args.model))

    try:
        infer.load()
    except FileNotFoundError:
        print(f"Model file not found: {args.model}")
        print("Train first using: python rl/train_dqn.py")
        return

    action = infer.decide_with_context(
        raw_counts=raw_state,
        waiting_ages=[0, 0, 0, 0],
        previous_action=0,
        steps_since_switch=3,
        in_yellow=False,
        can_switch=True,
        valid_actions=[0, 1, 2, 3],
    )
    print(f"Input lane counts: {raw_state}")
    print(f"Selected action: {action}")


if __name__ == "__main__":
    main()
