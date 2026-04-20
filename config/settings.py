"""
config/settings.py

Centralized runtime settings loaded from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    model_type: str = "dqn"
    model_path: str = "rl/q_table.pkl"
    dqn_model_path: str = "rl/dqn_model_tuned_best.pt"
    serial_port: str = "COM5"
    serial_baud: int = 115200
    serial_timeout: float = 1.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        model_type = os.getenv("SMART_TRAFFIC_MODEL_TYPE", "dqn").strip().lower()
        if model_type not in {"qtable", "dqn"}:
            model_type = "dqn"

        return cls(
            model_type=model_type,
            model_path=os.getenv("SMART_TRAFFIC_MODEL_PATH", "rl/q_table.pkl"),
            dqn_model_path=os.getenv("SMART_TRAFFIC_DQN_MODEL_PATH", "rl/dqn_model_tuned_best.pt"),
            serial_port=os.getenv("SMART_TRAFFIC_SERIAL_PORT", "COM5"),
            serial_baud=int(os.getenv("SMART_TRAFFIC_SERIAL_BAUD", "115200")),
            serial_timeout=float(os.getenv("SMART_TRAFFIC_SERIAL_TIMEOUT", "1.0")),
            log_level=os.getenv("SMART_TRAFFIC_LOG_LEVEL", "INFO").upper(),
        )
