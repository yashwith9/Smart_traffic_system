"""
integration/serial_send.py

Serial communication helper for sending lane actions (0-3) to ESP32/Arduino.
Supports mock mode so the module can be tested without hardware.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Allow running this file directly via: python integration/serial_send.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import AppConfig
from utils.logging_utils import setup_logging


LOGGER = logging.getLogger(__name__)

try:
    import serial  # type: ignore
except ImportError:
    serial = None


@dataclass
class SerialConfig:
    port: str = "COM5"
    baudrate: int = 115200
    timeout: float = 1.0
    mock: bool = False


class SerialSender:
    def __init__(self, config: SerialConfig):
        self.cfg = config
        self.ser: Optional["serial.Serial"] = None

    def connect(self) -> bool:
        if self.cfg.mock:
            LOGGER.info("[Serial] Mock mode enabled. No hardware connection needed.")
            return True

        if serial is None:
            LOGGER.warning("[Serial] pyserial is not installed. Falling back to mock mode.")
            self.cfg.mock = True
            return True

        try:
            self.ser = serial.Serial(
                self.cfg.port,
                self.cfg.baudrate,
                timeout=self.cfg.timeout,
            )
            # Give MCU time to reset after opening serial.
            time.sleep(2)
            LOGGER.info("[Serial] Connected to %s @ %s", self.cfg.port, self.cfg.baudrate)
            return True
        except Exception as exc:
            LOGGER.error("[Serial] Connection failed: %s", exc)
            return False

    def send_action(self, action: int) -> None:
        if action not in (0, 1, 2, 3):
            raise ValueError("Action must be one of 0, 1, 2, 3")

        payload = f"{action}\n"

        if self.cfg.mock:
            LOGGER.info("[Serial-Mock] Sent: %s", payload.strip())
            return

        if self.ser is None:
            raise RuntimeError("Serial is not connected. Call connect() first.")

        self.ser.write(payload.encode("utf-8"))
        self.ser.flush()
        LOGGER.info("[Serial] Sent: %s", payload.strip())

    def close(self) -> None:
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            LOGGER.info("[Serial] Connection closed.")


def parse_args() -> argparse.Namespace:
    env_cfg = AppConfig.from_env()

    parser = argparse.ArgumentParser(description="Send lane action to ESP32/Arduino over serial")
    parser.add_argument("--port", type=str, default=env_cfg.serial_port, help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=env_cfg.serial_baud, help="Serial baudrate")
    parser.add_argument("--timeout", type=float, default=env_cfg.serial_timeout, help="Serial timeout in seconds")
    parser.add_argument("--action", type=int, default=0, help="Action to send: 0,1,2,3")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to send")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between sends")
    parser.add_argument("--mock", action="store_true", help="Run without real serial hardware")
    parser.add_argument("--random", action="store_true", help="Send random actions instead of --action")
    parser.add_argument("--log-level", type=str, default=env_cfg.log_level, help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    sender = SerialSender(
        SerialConfig(
            port=args.port,
            baudrate=args.baud,
            timeout=args.timeout,
            mock=args.mock,
        )
    )

    if not sender.connect():
        return

    try:
        for _ in range(max(1, args.repeat)):
            action = random.randint(0, 3) if args.random else args.action
            sender.send_action(action)
            time.sleep(max(0.0, args.interval))
    finally:
        sender.close()


if __name__ == "__main__":
    main()
