"""
integration/serial_send.py

Serial communication helper for sending lane actions (0-3) to ESP32/Arduino.
Supports mock mode so the module can be tested without hardware.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Optional

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
            print("[Serial] Mock mode enabled. No hardware connection needed.")
            return True

        if serial is None:
            print("[Serial] pyserial is not installed. Falling back to mock mode.")
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
            print(f"[Serial] Connected to {self.cfg.port} @ {self.cfg.baudrate}")
            return True
        except Exception as exc:
            print(f"[Serial] Connection failed: {exc}")
            return False

    def send_action(self, action: int) -> None:
        if action not in (0, 1, 2, 3):
            raise ValueError("Action must be one of 0, 1, 2, 3")

        payload = f"{action}\n"

        if self.cfg.mock:
            print(f"[Serial-Mock] Sent: {payload.strip()}")
            return

        if self.ser is None:
            raise RuntimeError("Serial is not connected. Call connect() first.")

        self.ser.write(payload.encode("utf-8"))
        self.ser.flush()
        print(f"[Serial] Sent: {payload.strip()}")

    def close(self) -> None:
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print("[Serial] Connection closed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send lane action to ESP32/Arduino over serial")
    parser.add_argument("--port", type=str, default="COM5", help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--action", type=int, default=0, help="Action to send: 0,1,2,3")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to send")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between sends")
    parser.add_argument("--mock", action="store_true", help="Run without real serial hardware")
    parser.add_argument("--random", action="store_true", help="Send random actions instead of --action")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sender = SerialSender(
        SerialConfig(
            port=args.port,
            baudrate=args.baud,
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
