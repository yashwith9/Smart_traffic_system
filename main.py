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
from typing import List, Optional

import cv2
import numpy as np

# Keep project root in import path for direct script execution.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cv.detect import DetectionConfig, VehicleDetector, mock_lane_counts, open_capture
from config.settings import AppConfig
from integration.serial_send import SerialConfig, SerialSender
from rl.infer import InferenceConfig, TrafficSignalInference, action_to_text
from rl.infer_dqn import DQNInferenceConfig, TrafficDQNInference
from utils.logging_utils import setup_logging


LOGGER = logging.getLogger(__name__)
LANE_LABELS = ["North", "East", "South", "West"]


def parse_lane_boundaries(boundaries_str: str, lane_count: int) -> Optional[List[float]]:
    if not boundaries_str.strip():
        return None

    parts = [p.strip() for p in boundaries_str.split(",") if p.strip()]
    values = [float(p) for p in parts]
    if len(values) != lane_count + 1:
        raise ValueError(f"--lane-boundaries must have {lane_count + 1} values")

    if abs(values[0] - 0.0) > 1e-6 or abs(values[-1] - 1.0) > 1e-6:
        raise ValueError("--lane-boundaries must start at 0.0 and end at 1.0")
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            raise ValueError("--lane-boundaries values must be strictly increasing")

    return values


def draw_count_overlay(
    frame,
    lane_counts: List[int],
    active_lane: int,
    timer_sec: float,
    suggested_lane: int,
) -> None:
    _, w = frame.shape[:2]
    panel_w = min(520, w - 20)
    panel_h = 110
    px0 = w - panel_w - 10
    py0 = 10
    px1 = w - 10
    py1 = py0 + panel_h

    cv2.rectangle(frame, (px0, py0), (px1, py1), (20, 20, 20), -1)
    cv2.rectangle(frame, (px0, py0), (px1, py1), (230, 230, 230), 2)
    cv2.putText(
        frame,
        "Vehicle Count Dashboard",
        (px0 + 12, py0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    lane_count = max(1, len(lane_counts))
    slot_w = max(1, panel_w // lane_count)
    for idx, count in enumerate(lane_counts):
        sx = px0 + idx * slot_w + 10
        sy = py0 + 42
        is_active = idx == active_lane
        status = "GREEN" if is_active else "RED"
        status_color = (0, 220, 0) if is_active else (0, 0, 220)

        cv2.circle(frame, (sx + 8, sy + 7), 7, status_color, -1)
        cv2.putText(
            frame,
            f"L{idx + 1} {status}",
            (sx + 22, sy + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Lane {idx + 1}",
            (sx, sy + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Count: {count}",
            (sx, sy + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"Total Vehicles: {sum(lane_counts)}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"GREEN NOW: L{active_lane + 1} {LANE_LABELS[active_lane]}",
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"NEXT: L{suggested_lane + 1} {LANE_LABELS[suggested_lane]} | Timer: {timer_sec:0.1f}s",
        (12, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )

    # High-visibility decision banner for demos.
    banner_w = min(620, w - 30)
    bx0 = max(10, (w - banner_w) // 2)
    by0 = 10
    bx1 = bx0 + banner_w
    by1 = by0 + 54
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), (15, 60, 15), -1)
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), (40, 240, 40), 2)
    cv2.putText(
        frame,
        f"CURRENT GREEN: L{active_lane + 1} {LANE_LABELS[active_lane]}",
        (bx0 + 14, by0 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (220, 255, 220),
        2,
        cv2.LINE_AA,
    )


def resize_keep_aspect(frame, max_width: int = 1280, max_height: int = 720):
    h, w = frame.shape[:2]
    scale = min(float(max_width) / max(1.0, float(w)), float(max_height) / max(1.0, float(h)))
    if scale >= 1.0:
        return frame
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_signal_dashboard(
    frame,
    lane_counts: List[int],
    green_lane: int,
    timer_sec: float,
    suggested_lane: int,
    lane_boundaries: Optional[List[float]],
) -> None:
    # Keep lane_boundaries argument for compatibility.
    _ = lane_boundaries
    draw_count_overlay(frame, lane_counts, green_lane, timer_sec, suggested_lane)


class SmartTrafficPipeline:
    def __init__(
        self,
        model_type: str,
        model_path: str,
        output_video: str,
        use_yolo: bool,
        yolo_backend: str,
        yolo_model_path: str,
        yolo_device: str,
        yolo_conf_threshold: float,
        yolo_nms_threshold: float,
        yolo_input_size: int,
        yolo_tile_grid: int,
        yolo_tile_overlap_ratio: float,
        yolo_tiled_inference: bool,
        skip_dark_frames: int,
        dark_frame_threshold: float,
        intersection_center_x_ratio: float,
        intersection_center_y_ratio: float,
        intersection_half_width_ratio: float,
        intersection_half_height_ratio: float,
        lane_boundaries: Optional[List[float]],
        count_mode: str,
        serial_port: str,
        serial_baud: int,
        serial_timeout: float,
        serial_mock: bool,
    ):
        self.model_type = model_type
        self.output_video = output_video
        self.lane_boundaries = lane_boundaries
        self.skip_dark_frames = max(0, int(skip_dark_frames))
        self.dark_frame_threshold = max(0.0, float(dark_frame_threshold))
        self.detector = VehicleDetector(
            DetectionConfig(
                use_yolo=use_yolo,
                yolo_backend=yolo_backend,
                yolo_model_path=yolo_model_path,
                yolo_device=yolo_device,
                yolo_conf_threshold=yolo_conf_threshold,
                yolo_nms_threshold=yolo_nms_threshold,
                yolo_input_size=yolo_input_size,
                yolo_tile_grid=yolo_tile_grid,
                yolo_tile_overlap_ratio=yolo_tile_overlap_ratio,
                yolo_tiled_inference=yolo_tiled_inference,
                intersection_center_x_ratio=intersection_center_x_ratio,
                intersection_center_y_ratio=intersection_center_y_ratio,
                intersection_half_width_ratio=intersection_half_width_ratio,
                intersection_half_height_ratio=intersection_half_height_ratio,
                lane_boundaries=lane_boundaries,
                count_mode=count_mode,
                roi_start_ratio=0.0,
            )
        )
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

        # Dynamic signal phase state for more realistic and visible control.
        self.current_green = 0
        self.suggested_green = 0
        self.phase_remaining_sec = 0.0
        self.phase_elapsed_sec = 0.0
        self.min_green_sec = 4.0
        self.max_green_sec = 18.0
        self.preempt_margin = 5
        self.count_smooth_alpha = 0.35
        self.smoothed_counts: Optional[List[float]] = None

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

    def _smooth_lane_counts(self, lane_counts: List[int]) -> List[int]:
        if self.smoothed_counts is None:
            self.smoothed_counts = [float(c) for c in lane_counts]
        else:
            a = self.count_smooth_alpha
            self.smoothed_counts = [
                (a * float(cur)) + ((1.0 - a) * prev)
                for prev, cur in zip(self.smoothed_counts, lane_counts)
            ]
        return [max(0, int(round(v))) for v in self.smoothed_counts]

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

    def _dynamic_green_duration(self, lane_counts: List[int], lane_idx: int) -> float:
        total = max(1, sum(lane_counts))
        lane_pressure = lane_counts[lane_idx]
        pressure_ratio = float(lane_pressure) / float(total)
        duration = self.min_green_sec + (pressure_ratio * 10.0) + min(4.0, lane_pressure * 0.2)
        return max(self.min_green_sec, min(self.max_green_sec, duration))

    def process_state(self, lane_counts: List[int], delta_sec: float) -> int:
        dt = max(0.0, delta_sec)
        self.phase_remaining_sec = max(0.0, self.phase_remaining_sec - dt)
        self.phase_elapsed_sec += dt

        if self.model_type == "dqn":
            suggested = self._decide_dqn(lane_counts)
        else:
            if self.infer_qtable is None:
                raise RuntimeError("Q-table inference not initialized")
            suggested = self.infer_qtable.decide(lane_counts)

        self.suggested_green = suggested

        # If another lane is significantly heavier, preempt after minimum green hold.
        heaviest_lane = int(max(range(len(lane_counts)), key=lambda idx: lane_counts[idx]))
        active_count = lane_counts[self.current_green]
        heaviest_count = lane_counts[heaviest_lane]
        can_preempt = self.phase_elapsed_sec >= self.min_green_sec
        should_preempt = (
            heaviest_lane != self.current_green
            and can_preempt
            and (heaviest_count - active_count) >= self.preempt_margin
        )

        if should_preempt:
            self.current_green = heaviest_lane
            self.phase_remaining_sec = self._dynamic_green_duration(lane_counts, self.current_green)
            self.phase_elapsed_sec = 0.0
            self.suggested_green = heaviest_lane

        # Keep current green until timer expires, then apply next suggestion.
        if self.phase_remaining_sec <= 0.0:
            self.current_green = suggested
            self.phase_remaining_sec = self._dynamic_green_duration(lane_counts, self.current_green)
            self.phase_elapsed_sec = 0.0

        action = self.current_green
        self.sender.send_action(action)
        LOGGER.info(
            "State=%s | Suggested=%s | Active=%s | Timer=%.1fs | Decision=%s",
            lane_counts,
            suggested,
            action,
            self.phase_remaining_sec,
            action_to_text(action),
        )
        return action

    def run_mock(self, steps: int = 30, interval: float = 1.0) -> None:
        LOGGER.info("Running full pipeline in MOCK mode...")
        try:
            for _ in range(steps):
                state = mock_lane_counts()
                self.process_state(state, interval)
                time.sleep(max(0.0, interval))
        finally:
            self.sender.close()

    def run_camera(self, source: str = "0", show_window: bool = True) -> None:
        cap = open_capture(source)
        if not cap.isOpened():
            LOGGER.error("Could not open camera/video source.")
            self.sender.close()
            return

        video_writer: Optional[cv2.VideoWriter] = None
        if self.output_video:
            out_dir = os.path.dirname(self.output_video)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 1 else 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))
            if video_writer.isOpened():
                LOGGER.info("Saving annotated output video to: %s", self.output_video)
            else:
                LOGGER.warning("Could not initialize video writer for: %s", self.output_video)
                video_writer = None

        LOGGER.info("Running full pipeline in CAMERA mode. Press 'q' to quit.")

        # Skip very dark intro frames often present in compressed drone videos.
        if self.skip_dark_frames > 0:
            skipped = 0
            while skipped < self.skip_dark_frames:
                ok, probe = cap.read()
                if not ok:
                    break
                if float(np.mean(probe)) >= self.dark_frame_threshold:
                    # Rewind one frame-equivalent by using this frame as first processed frame.
                    frame = probe
                    break
                skipped += 1
            else:
                frame = None
            if skipped > 0:
                LOGGER.info("Skipped %d dark intro frames", skipped)
        else:
            frame = None

        last_ts = time.time()
        try:
            while True:
                if frame is None:
                    ok, frame = cap.read()
                    if not ok:
                        LOGGER.warning("Stream ended or frame read failed.")
                        break

                now_ts = time.time()
                delta_sec = max(0.001, now_ts - last_ts)
                last_ts = now_ts

                raw_counts, annotated = self.detector.detect_and_count(frame)
                lane_counts = self._smooth_lane_counts(raw_counts)
                action = self.process_state(lane_counts, delta_sec)
                draw_signal_dashboard(
                    annotated,
                    lane_counts,
                    action,
                    self.phase_remaining_sec,
                    self.suggested_green,
                    self.lane_boundaries,
                )

                if show_window:
                    display_frame = resize_keep_aspect(annotated)
                    cv2.imshow("Smart Traffic AI Pipeline", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if video_writer is not None:
                    video_writer.write(annotated)

                frame = None
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
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
    parser.add_argument("--output-video", type=str, default="", help="Optional path to save annotated output video")
    parser.add_argument(
        "--lane-boundaries",
        type=str,
        default="",
        help="Comma-separated normalized lane x-boundaries (e.g. 0.0,0.22,0.47,0.75,1.0)",
    )
    parser.add_argument(
        "--count-mode",
        type=str,
        choices=["approach_zones", "lane_splits"],
        default="approach_zones",
        help="Vehicle counting mode: approach_zones (recommended for intersections) or lane_splits",
    )
    parser.add_argument("--use-yolo", action="store_true", help="Use YOLO ONNX detector (recommended for stopped traffic)")
    parser.add_argument("--yolo-model", type=str, default="", help="Path to YOLO ONNX model file")
    parser.add_argument(
        "--yolo-backend",
        type=str,
        choices=["auto", "onnx", "ultralytics"],
        default="auto",
        help="YOLO runtime backend: auto, onnx, or ultralytics",
    )
    parser.add_argument("--yolo-device", type=str, default="cpu", help="YOLO device, e.g. cpu or 0")
    parser.add_argument("--yolo-conf", type=float, default=0.18, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--yolo-imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--yolo-tile-grid", type=int, default=2, help="Tile grid for dense-scene YOLO inference")
    parser.add_argument("--yolo-tile-overlap", type=float, default=0.12, help="Tile overlap ratio (0.0-0.35)")
    parser.add_argument("--no-yolo-tiling", action="store_true", help="Disable tiled YOLO inference")
    parser.add_argument("--skip-dark-frames", type=int, default=80, help="Skip up to N very dark intro frames")
    parser.add_argument("--dark-threshold", type=float, default=12.0, help="Mean pixel threshold for dark-frame skipping")
    parser.add_argument("--intersection-cx", type=float, default=0.51, help="Intersection center X ratio (0-1)")
    parser.add_argument("--intersection-cy", type=float, default=0.56, help="Intersection center Y ratio (0-1)")
    parser.add_argument("--intersection-hw", type=float, default=0.14, help="Intersection half-width ratio (0-1)")
    parser.add_argument("--intersection-hh", type=float, default=0.12, help="Intersection half-height ratio (0-1)")

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
    lane_boundaries = parse_lane_boundaries(args.lane_boundaries, lane_count=4)

    pipeline = SmartTrafficPipeline(
        model_type=args.model_type,
        model_path=model_path,
        output_video=args.output_video,
        use_yolo=args.use_yolo,
        yolo_backend=args.yolo_backend,
        yolo_model_path=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_conf_threshold=args.yolo_conf,
        yolo_nms_threshold=args.yolo_iou,
        yolo_input_size=args.yolo_imgsz,
        yolo_tile_grid=args.yolo_tile_grid,
        yolo_tile_overlap_ratio=args.yolo_tile_overlap,
        yolo_tiled_inference=not args.no_yolo_tiling,
        skip_dark_frames=args.skip_dark_frames,
        dark_frame_threshold=args.dark_threshold,
        intersection_center_x_ratio=args.intersection_cx,
        intersection_center_y_ratio=args.intersection_cy,
        intersection_half_width_ratio=args.intersection_hw,
        intersection_half_height_ratio=args.intersection_hh,
        lane_boundaries=lane_boundaries,
        count_mode=args.count_mode,
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
