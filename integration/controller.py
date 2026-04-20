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
import os
import sys
from typing import List, Tuple

import cv2

# Allow running this file directly via: python integration/controller.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cv.detect import VehicleDetector, mock_lane_counts, open_capture
from rl.infer import InferenceConfig, TrafficSignalInference, action_to_text


class TrafficController:
    def __init__(self, model_path: str = "rl/q_table.pkl"):
        self.model_path = model_path
        self.infer = TrafficSignalInference(InferenceConfig(q_table_path=model_path))
        self.detector = VehicleDetector()

    def load_model(self) -> bool:
        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}")
            print("Run training first: python rl/train_rl.py")
            return False
        self.infer.load()
        return True

    def decide_from_state(self, lane_counts: List[int]) -> int:
        action = self.infer.decide(lane_counts)
        return action

    def run_mock(self, iterations: int = 20) -> None:
        print("Controller running in MOCK mode...")
        for step in range(1, iterations + 1):
            state = mock_lane_counts()
            action = self.decide_from_state(state)
            print(
                f"[Step {step}] State={state} | "
                f"Action={action} | Decision={action_to_text(action)}"
            )

    def run_camera(self, source: str = "0", show_window: bool = True) -> None:
        cap = open_capture(source)
        if not cap.isOpened():
            print("Could not open camera/video source.")
            return

        print("Controller running in CAMERA mode. Press 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream ended or frame read failed.")
                break

            lane_counts, annotated = self.detector.detect_and_count(frame)
            action = self.decide_from_state(lane_counts)

            print(f"State={lane_counts} | Action={action} | Decision={action_to_text(action)}")

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
    parser = argparse.ArgumentParser(description="Connect CV lane counts to RL decision engine")
    parser.add_argument(
        "--model",
        type=str,
        default="rl/q_table.pkl",
        help="Path to trained Q-table pickle",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    controller = TrafficController(model_path=args.model)
    if not controller.load_model():
        return

    if args.mock:
        controller.run_mock(iterations=args.steps)
    else:
        controller.run_camera(source=args.source, show_window=not args.no_view)


if __name__ == "__main__":
    main()
