"""
integration/controller.py

Controller module that connects:
- CV lane counting (cv/detect.py)
- RL inference (rl/infer.py)

Pipeline step:
state (lane counts) -> action (lane to set GREEN)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

import cv2

# Allow running this file directly via: python integration/controller.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cv.detect import VehicleDetector, mock_lane_counts, open_capture
from config.settings import AppConfig
from rl.infer import InferenceConfig, TrafficSignalInference, action_to_text
from rl.infer_dqn import DQNInferenceConfig, TrafficDQNInference
from utils.logging_utils import setup_logging


LOGGER = logging.getLogger(__name__)


class TrafficController:
    def __init__(self, model_path: str = "rl/q_table.pkl", model_type: str = "dqn"):
        self.model_path = model_path
        self.model_type = model_type
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

        self.detector = VehicleDetector()

    def load_model(self) -> bool:
        if not os.path.exists(self.model_path):
            LOGGER.error("Model file not found: %s", self.model_path)
            if self.model_type == "dqn":
                LOGGER.error("Run training first: python rl/train_dqn.py")
            else:
                LOGGER.error("Run training first: python rl/train_rl.py")
            return False
        if self.infer_dqn is not None:
            self.infer_dqn.load()
        elif self.infer_qtable is not None:
            self.infer_qtable.load()
        else:
            LOGGER.error("No inference model configured")
            return False
        LOGGER.info("Model loaded from %s", self.model_path)
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

    def decide_from_state(self, lane_counts: List[int]) -> int:
        if self.model_type == "dqn":
            return self._decide_dqn(lane_counts)
        if self.infer_qtable is None:
            raise RuntimeError("Q-table inference not initialized")
        return self.infer_qtable.decide(lane_counts)

    def run_mock(self, iterations: int = 20) -> None:
        LOGGER.info("Controller running in MOCK mode...")
        for step in range(1, iterations + 1):
            state = mock_lane_counts()
            action = self.decide_from_state(state)
            LOGGER.info(
                "[Step %s] State=%s | Action=%s | Decision=%s",
                step,
                state,
                action,
                action_to_text(action),
            )

    def run_camera(self, source: str = "0", show_window: bool = True) -> None:
        cap = open_capture(source)
        if not cap.isOpened():
            LOGGER.error("Could not open camera/video source.")
            return

        LOGGER.info("Controller running in CAMERA mode. Press 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                LOGGER.warning("Stream ended or frame read failed.")
                break

            lane_counts, annotated = self.detector.detect_and_count(frame)
            action = self.decide_from_state(lane_counts)

            LOGGER.info(
                "State=%s | Action=%s | Decision=%s",
                lane_counts,
                action,
                action_to_text(action),
            )

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
                cv2.imshow("Traffic Controller", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    env_cfg = AppConfig.from_env()

    parser = argparse.ArgumentParser(description="Connect CV lane counts to RL decision engine")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["qtable", "dqn"],
        default=env_cfg.model_type,
        help="Inference model type",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to trained model file (optional override)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use generated mock traffic states instead of camera",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of mock steps when --mock is used",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g. 0) or video file path",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Do not open display window in camera mode",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=env_cfg.log_level,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    env_cfg = AppConfig.from_env()

    model_path = args.model
    if not model_path:
        model_path = env_cfg.dqn_model_path if args.model_type == "dqn" else env_cfg.model_path

    controller = TrafficController(model_path=model_path, model_type=args.model_type)
    if not controller.load_model():
        return

    if args.mock:
        controller.run_mock(iterations=args.steps)
    else:
        controller.run_camera(source=args.source, show_window=not args.no_view)


if __name__ == "__main__":
    main()
