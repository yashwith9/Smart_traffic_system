"""
rl/sim_env.py

Shared traffic simulation environment for RL training and evaluation.
Includes:
- queue dynamics
- waiting-age per lane
- min-green signal hold
- yellow transition steps during lane switching
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SimConfig:
    lane_count: int = 4
    max_lane_count: int = 20
    service_capacity_per_step: int = 4

    arrival_min: int = 0
    arrival_max: int = 2

    min_green_steps: int = 3
    yellow_steps: int = 1
    max_wait_age: int = 30


class TrafficSimEnv:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.lane_counts: List[int] = [0] * self.cfg.lane_count
        self.waiting_ages: List[int] = [0] * self.cfg.lane_count

        self.current_green: int = 0
        self.steps_since_switch: int = 0

        self.yellow_remaining: int = 0
        self.pending_green: Optional[int] = None

    def reset(self, rng: random.Random, initial_green: Optional[int] = None) -> Dict[str, object]:
        self.lane_counts = [rng.randint(0, self.cfg.max_lane_count) for _ in range(self.cfg.lane_count)]
        self.waiting_ages = [0] * self.cfg.lane_count

        if initial_green is None:
            self.current_green = rng.randint(0, self.cfg.lane_count - 1)
        else:
            self.current_green = max(0, min(self.cfg.lane_count - 1, initial_green))

        self.steps_since_switch = 0
        self.yellow_remaining = 0
        self.pending_green = None

        return self.snapshot()

    def snapshot(self) -> Dict[str, object]:
        return {
            "lane_counts": self.lane_counts.copy(),
            "waiting_ages": self.waiting_ages.copy(),
            "current_green": self.current_green,
            "steps_since_switch": self.steps_since_switch,
            "yellow_remaining": self.yellow_remaining,
            "in_yellow": self.yellow_remaining > 0,
            "can_switch": self.can_switch(),
            "valid_actions": self.valid_actions(),
        }

    def can_switch(self) -> bool:
        if self.yellow_remaining > 0:
            return False
        return self.steps_since_switch >= self.cfg.min_green_steps

    def valid_actions(self) -> List[int]:
        if self.yellow_remaining > 0:
            return [self.current_green]
        if not self.can_switch():
            return [self.current_green]
        return list(range(self.cfg.lane_count))

    def step(self, requested_action: int, rng: random.Random) -> Dict[str, object]:
        requested_action = max(0, min(self.cfg.lane_count - 1, int(requested_action)))
        valid_before = self.valid_actions()
        invalid_request = requested_action not in valid_before
        switched = False

        # Handle active yellow phase first.
        if self.yellow_remaining > 0:
            self.yellow_remaining -= 1
            if self.yellow_remaining == 0 and self.pending_green is not None:
                self.current_green = self.pending_green
                self.pending_green = None
                self.steps_since_switch = 0
                switched = True
        else:
            can_switch = self.steps_since_switch >= self.cfg.min_green_steps
            if requested_action != self.current_green and can_switch:
                if self.cfg.yellow_steps > 0:
                    self.yellow_remaining = self.cfg.yellow_steps
                    self.pending_green = requested_action
                else:
                    self.current_green = requested_action
                    self.steps_since_switch = 0
                    switched = True

        in_yellow = self.yellow_remaining > 0
        served_per_lane = [0] * self.cfg.lane_count

        # Serve only if a green lane is active (not during yellow transition).
        if not in_yellow:
            lane = self.current_green
            served = min(self.cfg.service_capacity_per_step, self.lane_counts[lane])
            self.lane_counts[lane] -= served
            served_per_lane[lane] = served

        # Random arrivals.
        for i in range(self.cfg.lane_count):
            arrivals = rng.randint(self.cfg.arrival_min, self.cfg.arrival_max)
            self.lane_counts[i] = min(self.cfg.max_lane_count, self.lane_counts[i] + arrivals)

        # Waiting-age updates.
        for i in range(self.cfg.lane_count):
            if self.lane_counts[i] <= 0:
                self.waiting_ages[i] = 0
                continue

            if not in_yellow and i == self.current_green and served_per_lane[i] > 0:
                self.waiting_ages[i] = max(0, self.waiting_ages[i] - 1)
            else:
                self.waiting_ages[i] = min(self.cfg.max_wait_age, self.waiting_ages[i] + 1)

        self.steps_since_switch += 1

        return {
            "lane_counts": self.lane_counts.copy(),
            "waiting_ages": self.waiting_ages.copy(),
            "current_green": self.current_green,
            "requested_action": requested_action,
            "yellow_remaining": self.yellow_remaining,
            "steps_since_switch": self.steps_since_switch,
            "switched": switched,
            "in_yellow": in_yellow,
            "can_switch": self.can_switch(),
            "valid_actions": self.valid_actions(),
            "invalid_request": invalid_request,
            "served_per_lane": served_per_lane,
            "served_total": sum(served_per_lane),
        }
