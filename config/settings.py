"""
config/settings.py

Centralized runtime settings loaded from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    model_path: str = "rl/q_table.pkl"
    serial_port: str = "COM5"
    serial_baud: int = 115200
    serial_timeout: float = 1.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            model_path=os.getenv("SMART_TRAFFIC_MODEL_PATH", "rl/q_table.pkl"),
            serial_port=os.getenv("SMART_TRAFFIC_SERIAL_PORT", "COM5"),
            serial_baud=int(os.getenv("SMART_TRAFFIC_SERIAL_BAUD", "115200")),
            serial_timeout=float(os.getenv("SMART_TRAFFIC_SERIAL_TIMEOUT", "1.0")),
            log_level=os.getenv("SMART_TRAFFIC_LOG_LEVEL", "INFO").upper(),
        )
