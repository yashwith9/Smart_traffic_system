"""
main.py

Full pipeline:
capture frame -> detect lane counts -> RL decision -> send serial -> display

Works in two modes:
1) Camera/Video mode
2) Mock mode (no camera needed)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List

import cv2

# Keep project root in import path for direct script execution.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cv.detect import VehicleDetector, mock_lane_counts, open_capture
from config.settings import AppConfig
from integration.serial_send import SerialConfig, SerialSender
from rl.infer import InferenceConfig, TrafficSignalInference, action_to_text
from rl.infer_dqn import DQNInferenceConfig, TrafficDQNInference
from utils.logging_utils import setup_logging


LOGGER = logging.getLogger(__name__)


class SmartTrafficPipeline:
    def __init__(
        self,
        model_type: str,
        model_path: str,
        serial_port: str,
        serial_baud: int,
        serial_timeout: float,
        serial_mock: bool,
    ):
        self.model_type = model_type
        self.detector = VehicleDetector()
        self.infer_qtable: TrafficSignalInference | None = None
        self.infer_dqn: TrafficDQNInference | None = None
        if self.model_type == "dqn":
            self.infer_dqn = TrafficDQNInference(DQNInferenceConfig(model_path=model_path))
        else:
            self.infer_qtable = TrafficSignalInference(InferenceConfig(q_table_path=model_path))

        self.previous_action = 0
        self.waiting_ages = [0, 0, 0, 0]
        self.steps_since_switch = 0
        self.in_yellow = False

        self.sender = SerialSender(
            SerialConfig(
                port=serial_port,
                baudrate=serial_baud,
                timeout=serial_timeout,
                mock=serial_mock,
            )
        )

    def initialize(self) -> bool:
        try:
            if self.infer_dqn is not None:
                self.infer_dqn.load()
            elif self.infer_qtable is not None:
                self.infer_qtable.load()
            else:
                raise RuntimeError("No inference model initialized")
        except FileNotFoundError:
            if self.model_type == "dqn":
                LOGGER.error("Model not found. Train first with: python rl/train_dqn.py")
            else:
                LOGGER.error("Model not found. Train first with: python rl/train_rl.py")
            return False

        if not self.sender.connect():
            LOGGER.error("Serial initialization failed.")
            return False

        LOGGER.info("Pipeline initialized successfully.")
        return True

    def _decide_dqn(self, lane_counts: List[int]) -> int:
        if self.infer_dqn is None:
            raise RuntimeError("DQN inference not initialized")

        max_wait_age = int(self.infer_dqn.meta.get("max_wait_age", 30))
        min_green_steps = int(self.infer_dqn.meta.get("min_green_steps", 3))
        lane_count = int(self.infer_dqn.meta.get("lane_count", 4))

        updated_ages: List[int] = []
        for idx, count in enumerate(lane_counts):
            if count <= 0:
                updated_ages.append(0)
            elif idx == self.previous_action:
                updated_ages.append(max(0, self.waiting_ages[idx] - 1))
            else:
                updated_ages.append(min(max_wait_age, self.waiting_ages[idx] + 1))

        can_switch = (not self.in_yellow) and (self.steps_since_switch >= min_green_steps)
        valid_actions = [self.previous_action] if not can_switch else list(range(lane_count))

        action = self.infer_dqn.decide_with_context(
            raw_counts=lane_counts,
            waiting_ages=updated_ages,
            previous_action=self.previous_action,
            steps_since_switch=self.steps_since_switch,
            in_yellow=self.in_yellow,
            can_switch=can_switch,
            valid_actions=valid_actions,
        )

        if action == self.previous_action:
            self.steps_since_switch += 1
        else:
            self.steps_since_switch = 0
        self.previous_action = action
        self.waiting_ages = updated_ages
        self.in_yellow = False
        return action

    def process_state(self, lane_counts: List[int]) -> int:
        if self.model_type == "dqn":
            action = self._decide_dqn(lane_counts)
        else:
            if self.infer_qtable is None:
                raise RuntimeError("Q-table inference not initialized")
            action = self.infer_qtable.decide(lane_counts)
        self.sender.send_action(action)
        LOGGER.info(
            "State=%s | Action=%s | Decision=%s",
            lane_counts,
            action,
            action_to_text(action),
        )
        return action

    def run_mock(self, steps: int = 30, interval: float = 1.0) -> None:
        LOGGER.info("Running full pipeline in MOCK mode...")
        try:
            for _ in range(steps):
                state = mock_lane_counts()
                self.process_state(state)
                time.sleep(max(0.0, interval))
        finally:
            self.sender.close()

    def run_camera(self, source: str = "0", show_window: bool = True) -> None:
        cap = open_capture(source)
        if not cap.isOpened():
            LOGGER.error("Could not open camera/video source.")
            self.sender.close()
            return

        LOGGER.info("Running full pipeline in CAMERA mode. Press 'q' to quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    LOGGER.warning("Stream ended or frame read failed.")
                    break

                lane_counts, annotated = self.detector.detect_and_count(frame)
                action = self.process_state(lane_counts)

                if show_window:
                    cv2.putText(
                        annotated,
                        f"Decision: {action_to_text(action)}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Smart Traffic AI Pipeline", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.sender.close()


def parse_args() -> argparse.Namespace:
    env_cfg = AppConfig.from_env()

    parser = argparse.ArgumentParser(description="Smart Traffic AI full pipeline")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["qtable", "dqn"],
        default=env_cfg.model_type,
        help="Inference model type",
    )
    parser.add_argument("--model", type=str, default="", help="Path to model file (optional override)")

    parser.add_argument("--mock", action="store_true", help="Run without camera and serial hardware")
    parser.add_argument("--steps", type=int, default=20, help="Mock mode steps")
    parser.add_argument("--interval", type=float, default=1.0, help="Mock mode delay in seconds")

    parser.add_argument("--source", type=str, default="0", help="Camera index or video file path")
    parser.add_argument("--no-view", action="store_true", help="Disable display window")

    parser.add_argument("--serial-port", type=str, default=env_cfg.serial_port, help="ESP32/Arduino serial port")
    parser.add_argument("--serial-baud", type=int, default=env_cfg.serial_baud, help="Serial baudrate")
    parser.add_argument("--serial-timeout", type=float, default=env_cfg.serial_timeout, help="Serial timeout (seconds)")
    parser.add_argument("--serial-mock", action="store_true", help="Force mock serial sender")
    parser.add_argument("--log-level", type=str, default=env_cfg.log_level, help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    env_cfg = AppConfig.from_env()

    serial_mock = args.serial_mock or args.mock
    model_path = args.model
    if not model_path:
        model_path = env_cfg.dqn_model_path if args.model_type == "dqn" else env_cfg.model_path

    pipeline = SmartTrafficPipeline(
        model_type=args.model_type,
        model_path=model_path,
        serial_port=args.serial_port,
        serial_baud=args.serial_baud,
        serial_timeout=args.serial_timeout,
        serial_mock=serial_mock,
    )

    if not pipeline.initialize():
        return

    if args.mock:
        pipeline.run_mock(steps=args.steps, interval=args.interval)
    else:
        pipeline.run_camera(source=args.source, show_window=not args.no_view)


if __name__ == "__main__":
    main()
