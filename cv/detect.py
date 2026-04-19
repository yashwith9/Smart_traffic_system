"""
cv/detect.py

Beginner-friendly vehicle detection and lane-wise counting module.
This version uses OpenCV background subtraction (lightweight, no YOLO dependency).

Output format:
    [lane1_count, lane2_count, lane3_count, lane4_count]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Vehicle-like classes kept here for future YOLO integration compatibility
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}


@dataclass
class DetectionConfig:
    lane_count: int = 4
    min_box_width: int = 25
    min_box_height: int = 25
    blur_kernel: Tuple[int, int] = (5, 5)
    roi_start_ratio: float = 0.35  # Ignore top area to reduce noise
    debug_draw: bool = True


class VehicleDetector:
    """
    OpenCV-based detector using background subtraction + contour filtering.
    Works with webcam or video file.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

        # MOG2 handles moving-object extraction reasonably well for traffic scenes
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400,
            varThreshold=32,
            detectShadows=False,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, self.config.blur_kernel, 0)
        fg_mask = self.bg_subtractor.apply(blurred)

        # Morphological cleanup to remove scattered noise
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)
        return fg_mask

    def _lane_index(self, x_center: int, frame_width: int) -> int:
        lane_width = frame_width / self.config.lane_count
        lane = int(x_center // lane_width)
        return max(0, min(self.config.lane_count - 1, lane))

    def detect_and_count(self, frame: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Detect moving vehicles and count them lane-wise.

        Returns:
            counts: list[int] of length 4 -> [lane1, lane2, lane3, lane4]
            annotated_frame: frame with debug drawings
        """
        if frame is None or frame.size == 0:
            return [0] * self.config.lane_count, frame

        h, w = frame.shape[:2]
        roi_start_y = int(h * self.config.roi_start_ratio)

        roi = frame[roi_start_y:, :]
        fg_mask = self._preprocess(roi)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        counts = [0] * self.config.lane_count
        annotated = frame.copy()

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Filter very small blobs that are unlikely to be vehicles
            if bw < self.config.min_box_width or bh < self.config.min_box_height:
                continue

            # Convert ROI-local y back to full-frame y
            y_full = y + roi_start_y
            x_center = x + bw // 2
            lane = self._lane_index(x_center, w)
            counts[lane] += 1

            if self.config.debug_draw:
                cv2.rectangle(annotated, (x, y_full), (x + bw, y_full + bh), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"L{lane + 1}",
                    (x, max(20, y_full - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if self.config.debug_draw:
            self._draw_lane_overlay(annotated, counts, roi_start_y)

        return counts, annotated

    def _draw_lane_overlay(self, frame: np.ndarray, counts: List[int], roi_start_y: int) -> None:
        h, w = frame.shape[:2]
        lane_w = w // self.config.lane_count

        # Draw lane separators
        for i in range(1, self.config.lane_count):
            x = i * lane_w
            cv2.line(frame, (x, 0), (x, h), (255, 180, 0), 2)

        # Draw ROI boundary
        cv2.line(frame, (0, roi_start_y), (w, roi_start_y), (80, 80, 255), 2)

        # Draw lane counts
        for i, c in enumerate(counts):
            text = f"Lane {i + 1}: {c}"
            x_text = i * lane_w + 10
            cv2.putText(
                frame,
                text,
                (x_text, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw compact state list expected by RL module
        cv2.putText(
            frame,
            f"State: {counts}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 255, 50),
            2,
            cv2.LINE_AA,
        )


def mock_lane_counts(max_count: int = 12) -> List[int]:
    """
    Fallback helper when camera/video is unavailable.
    Simulates lane traffic state for independent module testing.
    """
    return [random.randint(0, max_count) for _ in range(4)]


def open_capture(source: str) -> cv2.VideoCapture:
    """
    source can be:
      - "0" (or any integer string) for webcam index
      - path to a video file
    """
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vehicle detection + lane counting")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help='Video source: webcam index like "0" or video file path',
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run mock mode (no camera/video needed)",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Disable OpenCV display window",
    )
    args = parser.parse_args()

    if args.mock:
        print("Running in mock mode. Press Ctrl+C to stop.")
        try:
            while True:
                counts = mock_lane_counts()
                print(f"Lane counts: {counts}")
                # Simple delay without importing time globally in hot path
                cv2.waitKey(700)
        except KeyboardInterrupt:
            print("\nMock mode stopped.")
        return

    detector = VehicleDetector()
    cap = open_capture(args.source)

    if not cap.isOpened():
        print("Could not open video source.")
        print("Tip: run with --mock for simulation mode.")
        return

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("End of stream or failed to read frame.")
            break

        counts, annotated = detector.detect_and_count(frame)
        print(f"Lane counts: {counts}")

        if not args.no_view:
            cv2.imshow("Smart Traffic CV - Lane Count", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
