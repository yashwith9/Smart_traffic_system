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
    lane_boundaries: Optional[List[float]] = None  # normalized x boundaries, e.g. [0.0,0.22,0.47,0.75,1.0]
    count_mode: str = "approach_zones"  # approach_zones | lane_splits
    intersection_center_x_ratio: float = 0.50
    intersection_center_y_ratio: float = 0.53
    intersection_half_width_ratio: float = 0.16
    intersection_half_height_ratio: float = 0.16
    # YOLO settings
    use_yolo: bool = False
    yolo_backend: str = "auto"  # auto | onnx | ultralytics
    yolo_model_path: str = ""
    yolo_conf_threshold: float = 0.15
    yolo_nms_threshold: float = 0.45
    yolo_input_size: int = 1280
    yolo_device: str = "cpu"
    yolo_tile_grid: int = 2
    yolo_tile_overlap_ratio: float = 0.12
    yolo_tiled_inference: bool = True

    # Fallback OpenCV contour settings
    min_box_width: int = 25
    min_box_height: int = 25
    min_contour_area: int = 220
    max_contour_area_ratio: float = 0.08
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 4.2
    blur_kernel: Tuple[int, int] = (5, 5)
    roi_start_ratio: float = 0.35  # Ignore top area to reduce noise
    bg_learning_rate: float = 0.0015
    track_iou_threshold: float = 0.30
    track_ttl_frames: int = 18
    dedupe_iou_threshold: float = 0.60

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
        self._lane_boundaries = self._resolve_lane_boundaries()
        self.yolo_net: Optional[cv2.dnn.Net] = None
        self.yolo_ultra = None
        self.yolo_backend = "none"
        self.yolo_enabled = False

        if self.config.use_yolo:
            self._init_yolo_backend()

        # Fallback detector for moving vehicles when YOLO is unavailable.
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400,
            varThreshold=32,
            detectShadows=False,
        )
        self._tracks: List[dict] = []

    def _init_yolo_backend(self) -> None:
        selected = self.config.yolo_backend.lower().strip()
        model_path = self.config.yolo_model_path.strip()

        # Prefer explicit ONNX model path for OpenCV DNN backend.
        if selected in {"auto", "onnx"} and model_path and model_path.lower().endswith(".onnx"):
            try:
                self.yolo_net = cv2.dnn.readNetFromONNX(model_path)
                self.yolo_enabled = True
                self.yolo_backend = "onnx"
                print(f"[INFO] YOLO enabled (onnx): {model_path}")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Could not load YOLO ONNX model: {exc}")

        # Ultralytics backend supports .pt and default pretrained models.
        if selected in {"auto", "ultralytics"}:
            try:
                from ultralytics import YOLO  # type: ignore

                ultra_model_name = model_path if model_path else "yolov8s.pt"
                self.yolo_ultra = YOLO(ultra_model_name)
                self.yolo_enabled = True
                self.yolo_backend = "ultralytics"
                print(f"[INFO] YOLO enabled (ultralytics): {ultra_model_name}")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Could not initialize Ultralytics YOLO: {exc}")

        # Final fallback: if backend is onnx but user passed non-onnx path, try as ONNX anyway.
        if selected == "onnx" and model_path:
            try:
                self.yolo_net = cv2.dnn.readNetFromONNX(model_path)
                self.yolo_enabled = True
                self.yolo_backend = "onnx"
                print(f"[INFO] YOLO enabled (onnx): {model_path}")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Could not load YOLO model. Falling back to OpenCV mode. Reason: {exc}")

        print("[WARN] YOLO requested but no backend initialized. Falling back to OpenCV mode.")

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = (aw * ah) + (bw * bh) - inter
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    def _update_tracks(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        matched_track_ids = set()
        for box in boxes:
            best_idx = -1
            best_iou = 0.0
            for idx, track in enumerate(self._tracks):
                if idx in matched_track_ids:
                    continue
                iou = self._iou(box, track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= self.config.track_iou_threshold:
                self._tracks[best_idx]["box"] = box
                self._tracks[best_idx]["ttl"] = self.config.track_ttl_frames
                matched_track_ids.add(best_idx)
            else:
                self._tracks.append({"box": box, "ttl": self.config.track_ttl_frames})
                matched_track_ids.add(len(self._tracks) - 1)

        next_tracks: List[dict] = []
        for idx, track in enumerate(self._tracks):
            if track["ttl"] <= 0:
                continue
            if idx not in matched_track_ids:
                track["ttl"] -= 1
            next_tracks.append(track)

        self._tracks = next_tracks
        return [t["box"] for t in self._tracks]

    def _deduplicate_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []

        # Keep larger boxes first, then reject near-duplicates by IoU.
        sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept: List[Tuple[int, int, int, int]] = []
        for box in sorted_boxes:
            is_duplicate = any(self._iou(box, k) >= self.config.dedupe_iou_threshold for k in kept)
            if not is_duplicate:
                kept.append(box)
        return kept

    def _merge_boxes_with_nms(
        self,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
    ) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []

        # Confidence-aware merge for full-frame + tiled predictions.
        nms_boxes = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in boxes]
        nms_scores = [float(s) for s in scores]
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            nms_scores,
            self.config.yolo_conf_threshold,
            self.config.yolo_nms_threshold,
        )
        merged: List[Tuple[int, int, int, int]] = []
        if len(indices) > 0:
            for idx in indices.flatten():
                merged.append(tuple(nms_boxes[int(idx)]))
        return self._deduplicate_boxes(merged)

    def _resolve_lane_boundaries(self) -> List[float]:
        if self.config.lane_boundaries is None:
            step = 1.0 / float(self.config.lane_count)
            return [i * step for i in range(self.config.lane_count)] + [1.0]

        boundaries = [float(v) for v in self.config.lane_boundaries]
        if len(boundaries) != (self.config.lane_count + 1):
            raise ValueError(
                f"lane_boundaries must have {self.config.lane_count + 1} values; got {len(boundaries)}"
            )

        if abs(boundaries[0] - 0.0) > 1e-6 or abs(boundaries[-1] - 1.0) > 1e-6:
            raise ValueError("lane_boundaries must start at 0.0 and end at 1.0")

        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                raise ValueError("lane_boundaries must be strictly increasing")

        return boundaries

    def _lane_index(self, x_center: int, frame_width: int) -> int:
        x_norm = max(0.0, min(1.0, float(x_center) / max(1.0, float(frame_width))))
        for i in range(self.config.lane_count):
            left = self._lane_boundaries[i]
            right = self._lane_boundaries[i + 1]
            if left <= x_norm < right:
                return i
        return self.config.lane_count - 1

    def _approach_index(self, x_center: int, y_center: int, frame_width: int, frame_height: int) -> Optional[int]:
        cx = float(x_center) / max(1.0, float(frame_width))
        cy = float(y_center) / max(1.0, float(frame_height))

        dx = cx - self.config.intersection_center_x_ratio
        dy = cy - self.config.intersection_center_y_ratio
        if (
            abs(dx) <= self.config.intersection_half_width_ratio
            and abs(dy) <= self.config.intersection_half_height_ratio
        ):
            return None

        # Assign to the nearest approach direction: N, E, S, W -> lanes 1..4.
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3
        return 2 if dy > 0 else 0

    def _assign_lane_for_box(
        self,
        x: int,
        y: int,
        bw: int,
        bh: int,
        frame_width: int,
        frame_height: int,
    ) -> Optional[int]:
        x_center = x + bw // 2
        y_center = y + bh // 2
        if self.config.count_mode == "approach_zones":
            return self._approach_index(x_center, y_center, frame_width, frame_height)
        return self._lane_index(x_center, frame_width)

    def _count_from_boxes(self, boxes: List[Tuple[int, int, int, int]], frame_width: int, frame_height: int) -> List[int]:
        counts = [0] * self.config.lane_count
        for x, y, bw, bh in boxes:
            lane = self._assign_lane_for_box(x, y, bw, bh, frame_width, frame_height)
            if lane is None:
                continue
            counts[lane] += 1
        return counts

    def _detect_with_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return vehicle boxes as (x, y, w, h) using YOLO ONNX output."""
        if self.yolo_backend == "ultralytics" and self.yolo_ultra is not None:
            h, w = frame.shape[:2]

            def _predict_boxes(
                img: np.ndarray,
                x_off: int = 0,
                y_off: int = 0,
            ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
                ih, iw = img.shape[:2]
                results = self.yolo_ultra.predict(
                    source=img,
                    conf=self.config.yolo_conf_threshold,
                    iou=self.config.yolo_nms_threshold,
                    classes=sorted(COCO_VEHICLE_CLASS_IDS),
                    verbose=False,
                    device=self.config.yolo_device,
                    imgsz=self.config.yolo_input_size,
                    agnostic_nms=True,
                    max_det=3000,
                )
                if not results:
                    return [], []

                boxes_xyxy = results[0].boxes.xyxy if results[0].boxes is not None else None
                boxes_conf = results[0].boxes.conf if results[0].boxes is not None else None
                if boxes_xyxy is None:
                    return [], []

                out: List[Tuple[int, int, int, int]] = []
                conf_out: List[float] = []
                conf_vals = boxes_conf.tolist() if boxes_conf is not None else [1.0] * len(boxes_xyxy)
                for row, conf in zip(boxes_xyxy.tolist(), conf_vals):
                    x1, y1, x2, y2 = row[:4]
                    x = int(max(0, min(iw - 1, x1))) + x_off
                    y = int(max(0, min(ih - 1, y1))) + y_off
                    xx2 = int(max(int(x1) + 1, min(iw, x2))) + x_off
                    yy2 = int(max(int(y1) + 1, min(ih, y2))) + y_off
                    x = max(0, min(w - 1, x))
                    y = max(0, min(h - 1, y))
                    xx2 = max(x + 1, min(w, xx2))
                    yy2 = max(y + 1, min(h, yy2))
                    out.append((x, y, xx2 - x, yy2 - y))
                    conf_out.append(float(conf))
                return out, conf_out

            final_boxes, final_scores = _predict_boxes(frame)

            if self.config.yolo_tiled_inference and self.config.yolo_tile_grid > 1:
                grid = max(2, self.config.yolo_tile_grid)
                overlap = max(0.0, min(0.35, self.config.yolo_tile_overlap_ratio))
                tile_w = max(64, int(w / grid))
                tile_h = max(64, int(h / grid))
                step_x = max(32, int(tile_w * (1.0 - overlap)))
                step_y = max(32, int(tile_h * (1.0 - overlap)))

                for y0 in range(0, h, step_y):
                    for x0 in range(0, w, step_x):
                        x1 = min(w, x0 + tile_w)
                        y1 = min(h, y0 + tile_h)
                        tile = frame[y0:y1, x0:x1]
                        if tile.size == 0:
                            continue
                        tile_boxes, tile_scores = _predict_boxes(tile, x_off=x0, y_off=y0)
                        final_boxes.extend(tile_boxes)
                        final_scores.extend(tile_scores)
                        if x1 == w:
                            break
                    if y1 == h:
                        break

            return self._merge_boxes_with_nms(final_boxes, final_scores)

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
        fg_mask = self.bg_subtractor.apply(blurred, learningRate=self.config.bg_learning_rate)

        # Morphological cleanup to remove scattered noise
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        return fg_mask

    def _detect_with_opencv(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return moving-object boxes as (x, y, w, h) from the ROI region."""
        h, w = frame.shape[:2]
        roi_start_y = int(h * self.config.roi_start_ratio)
        roi = frame[roi_start_y:, :]
        fg_mask = self._preprocess(roi)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: List[Tuple[int, int, int, int]] = []
        max_contour_area = int((h * w) * self.config.max_contour_area_ratio)
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area < self.config.min_contour_area or area > max_contour_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < self.config.min_box_width or bh < self.config.min_box_height:
                continue

            aspect_ratio = float(bw) / max(1.0, float(bh))
            if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
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
            boxes = self._deduplicate_boxes(boxes)
            # YOLO already detects static vehicles well; persistence here causes ghost boxes.
            self._tracks = []
        else:
            boxes = self._detect_with_opencv(frame)
            # Persist fallback detections for short periods to avoid dropping queued vehicles instantly.
            boxes = self._update_tracks(boxes)
            boxes = self._deduplicate_boxes(boxes)

        counts = self._count_from_boxes(boxes, w, h)

        if self.config.debug_draw:
            for x, y, bw, bh in boxes:
                lane = self._assign_lane_for_box(x, y, bw, bh, w, h)
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    "X" if lane is None else f"L{lane + 1}",
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

        if self.config.count_mode == "approach_zones":
            cx = int(self.config.intersection_center_x_ratio * w)
            cy = int(self.config.intersection_center_y_ratio * h)
            hw = int(self.config.intersection_half_width_ratio * w)
            hh = int(self.config.intersection_half_height_ratio * h)
            cv2.rectangle(frame, (cx - hw, cy - hh), (cx + hw, cy + hh), (255, 180, 0), 2)

            labels = ["L1 North", "L2 East", "L3 South", "L4 West"]
            positions = [
                (cx - 95, max(25, cy - hh - 18)),
                (min(w - 165, cx + hw + 10), cy - 10),
                (cx - 95, min(h - 20, cy + hh + 24)),
                (max(10, cx - hw - 165), cy - 10),
            ]
            for i, text in enumerate(labels):
                cv2.putText(
                    frame,
                    f"{text}: {counts[i]}",
                    positions[i],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

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
            return

        # Draw lane separators
        for i in range(1, self.config.lane_count):
            x = int(self._lane_boundaries[i] * w)
            cv2.line(frame, (x, 0), (x, h), (255, 180, 0), 2)

        # Draw ROI boundary
        if roi_start_y > 0:
            cv2.line(frame, (0, roi_start_y), (w, roi_start_y), (80, 80, 255), 2)

        # Draw lane counts
        for i, c in enumerate(counts):
            text = f"Lane {i + 1}: {c}"
            x_left = int(self._lane_boundaries[i] * w)
            x_right = int(self._lane_boundaries[i + 1] * w)
            x_text = x_left + max(10, (x_right - x_left) // 12)
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
