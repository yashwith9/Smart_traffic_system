"""
cv/detect.py

Vehicle detection + lane-wise counting for Smart Traffic AI.

Backends:
1) YOLO (optional): requires a YOLO ONNX model path.
2) OpenCV fallback: background subtraction (lightweight, no extra model).

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


COCO_VEHICLE_CLASS_IDS = {
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
}


@dataclass
class DetectionConfig:
    lane_count: int = 4
    # YOLO settings
    use_yolo: bool = False
    yolo_model_path: str = ""
    yolo_conf_threshold: float = 0.4
    yolo_nms_threshold: float = 0.45
    yolo_input_size: int = 640

    # Fallback OpenCV contour settings
    min_box_width: int = 25
    min_box_height: int = 25
    blur_kernel: Tuple[int, int] = (5, 5)
    roi_start_ratio: float = 0.35  # Ignore top area to reduce noise

    # General
    debug_draw: bool = True


class VehicleDetector:
    """
    Vehicle detector with optional YOLO backend and OpenCV fallback.

    Public API used by other modules:
        detect_and_count(frame) -> (counts, annotated_frame)
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.yolo_net: Optional[cv2.dnn.Net] = None
        self.yolo_enabled = False

        if self.config.use_yolo and self.config.yolo_model_path:
            try:
                self.yolo_net = cv2.dnn.readNetFromONNX(self.config.yolo_model_path)
                self.yolo_enabled = True
                print(f"[INFO] YOLO enabled with model: {self.config.yolo_model_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Could not load YOLO model. Falling back to OpenCV mode. Reason: {exc}")

        # Fallback detector for moving vehicles when YOLO is unavailable.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400,
            varThreshold=32,
            detectShadows=False,
        )

    def _lane_index(self, x_center: int, frame_width: int) -> int:
        lane_width = frame_width / self.config.lane_count
        lane = int(x_center // lane_width)
        return max(0, min(self.config.lane_count - 1, lane))

    def _count_from_boxes(self, boxes: List[Tuple[int, int, int, int]], frame_width: int) -> List[int]:
        counts = [0] * self.config.lane_count
        for x, _, bw, _ in boxes:
            x_center = x + bw // 2
            lane = self._lane_index(x_center, frame_width)
            counts[lane] += 1
        return counts

    def _detect_with_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return vehicle boxes as (x, y, w, h) using YOLO ONNX output."""
        if self.yolo_net is None:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.config.yolo_input_size, self.config.yolo_input_size),
            swapRB=True,
            crop=False,
        )
        self.yolo_net.setInput(blob)
        outputs = self.yolo_net.forward()

        # YOLOv5 ONNX common output: (1, N, 85) where 85 = [x,y,w,h,obj + 80 class scores]
        preds = outputs[0] if outputs.ndim == 3 else outputs

        boxes: List[List[int]] = []
        confidences: List[float] = []

        x_factor = w / float(self.config.yolo_input_size)
        y_factor = h / float(self.config.yolo_input_size)

        for det in preds:
            obj_conf = float(det[4])
            if obj_conf < self.config.yolo_conf_threshold:
                continue

            class_scores = det[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            score = obj_conf * class_conf

            if class_id not in COCO_VEHICLE_CLASS_IDS:
                continue
            if score < self.config.yolo_conf_threshold:
                continue

            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * x_factor)
            y = int((cy - bh / 2) * y_factor)
            ww = int(bw * x_factor)
            hh = int(bh * y_factor)

            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            ww = max(1, min(w - x, ww))
            hh = max(1, min(h - y, hh))

            boxes.append([x, y, ww, hh])
            confidences.append(score)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.config.yolo_conf_threshold,
            self.config.yolo_nms_threshold,
        )

        final_boxes: List[Tuple[int, int, int, int]] = []
        if len(indices) > 0:
            for idx in indices.flatten():
                final_boxes.append(tuple(boxes[int(idx)]))
        return final_boxes

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, self.config.blur_kernel, 0)
        fg_mask = self.bg_subtractor.apply(blurred)

        # Morphological cleanup to remove scattered noise
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)
        return fg_mask

    def _detect_with_opencv(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return moving-object boxes as (x, y, w, h) from the ROI region."""
        h, _ = frame.shape[:2]
        roi_start_y = int(h * self.config.roi_start_ratio)
        roi = frame[roi_start_y:, :]
        fg_mask = self._preprocess(roi)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < self.config.min_box_width or bh < self.config.min_box_height:
                continue

            # Convert ROI-local y to full-frame y.
            boxes.append((x, y + roi_start_y, bw, bh))
        return boxes

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
        annotated = frame.copy()
        roi_start_y = int(h * self.config.roi_start_ratio)

        if self.yolo_enabled:
            boxes = self._detect_with_yolo(frame)
        else:
            boxes = self._detect_with_opencv(frame)

        counts = self._count_from_boxes(boxes, w)

        if self.config.debug_draw:
            for x, y, bw, bh in boxes:
                lane = self._lane_index(x + bw // 2, w)
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"L{lane + 1}",
                    (x, max(20, y - 5)),
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
        "--use-yolo",
        action="store_true",
        help="Enable YOLO backend (requires --yolo-model)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="",
        help="Path to YOLOv5 ONNX model file",
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

    detector = VehicleDetector(
        DetectionConfig(
            use_yolo=args.use_yolo,
            yolo_model_path=args.yolo_model,
        )
    )
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
