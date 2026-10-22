"""Phase 4: real-time BPM overlay from the webcam.

Run from anywhere:  python scripts/phase4_realtime_pipeline.py
Keys: q = quit, d = toggle landmark debug view.

Keeps a rolling buffer of the last WINDOW_S seconds of ROI colour and
recomputes the BPM every REFRESH_S seconds via vitalvision.pipeline, which
runs the same resample -> detrend -> band-pass -> FFT pipeline as Phase 3.
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from vitalvision.face import (
    create_landmarker,
    detect_landmarks,
    forehead_polygon,
    landmarks_to_pixels,
    next_timestamp_ms,
    open_camera,
    roi_mean_rgb,
)
from vitalvision.pipeline import REFRESH_S, WINDOW_S, estimate_bpm_from_buffer

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
WINDOW_NAME = "VitalVision - Phase 4"
METHOD = "pos"

ROI_COLOR = (0, 255, 0)
ROI_FILL_ALPHA = 0.3
LANDMARK_COLOR = (0, 200, 255)
TEXT_COLOR = (255, 255, 255)
NO_FACE_COLOR = (0, 0, 255)
BPM_COLOR = (0, 215, 255)
FPS_SMOOTHING = 0.9
CAMERA_RETRY_S = 2.0
CAMERA_RETRY_INTERVAL_S = 0.03


def draw_roi(frame, polygon) -> None:
    """Draw the ROI as a semi-transparent fill with a solid outline, in place."""
    fill = frame.copy()
    cv2.fillPoly(fill, [polygon], ROI_COLOR)
    cv2.addWeighted(fill, ROI_FILL_ALPHA, frame, 1 - ROI_FILL_ALPHA, 0, dst=frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=ROI_COLOR, thickness=2)


def draw_overlay(frame, fps: float, face_found: bool, bpm: float | None) -> None:
    """Draw FPS, face status and the current BPM estimate in the top-left corner."""
    if face_found:
        status, status_color = "Face detected", ROI_COLOR
    else:
        status, status_color = "No face", NO_FACE_COLOR
    bpm_text = f"BPM: {bpm:.0f}" if bpm else "BPM: --"
    lines = [
        (f"FPS: {fps:.1f}", TEXT_COLOR),
        (status, status_color),
        (bpm_text, BPM_COLOR),
    ]
    for row, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text, (10, 30 + row * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )


def smooth_fps(previous_fps: float, frame_interval_s: float) -> float:
    """Blend the latest frame interval into an exponentially smoothed FPS."""
    instant_fps = 1.0 / max(frame_interval_s, 1e-6)
    if previous_fps == 0.0:
        return instant_fps
    return FPS_SMOOTHING * previous_fps + (1 - FPS_SMOOTHING) * instant_fps


def main() -> None:
    """Run the live webcam loop, overlaying a periodically-refreshed BPM estimate."""
    landmarker = create_landmarker(MODEL_PATH)
    cap = open_camera(CAMERA_INDEX)
    show_debug = False
    fps = 0.0
    last_ts_ms = -1
    last_frame_time = time.monotonic()
    last_good_read = time.monotonic()
    start_time = time.monotonic()
    last_bpm_update = 0.0
    current_bpm: float | None = None

    buffer_timestamps: deque[float] = deque()
    buffer_rgb: deque[tuple[float, float, float]] = deque()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if time.monotonic() - last_good_read > CAMERA_RETRY_S:
                    print("Error: failed to read a frame from the camera.")
                    break
                time.sleep(CAMERA_RETRY_INTERVAL_S)
                continue
            last_good_read = time.monotonic()

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            last_ts_ms = next_timestamp_ms(last_ts_ms)
            landmarks = detect_landmarks(landmarker, frame, last_ts_ms)

            now = time.monotonic()
            elapsed = now - start_time

            if landmarks is not None:
                points = landmarks_to_pixels(landmarks, width, height)
                polygon = forehead_polygon(points)
                rgb = roi_mean_rgb(frame, polygon)
                buffer_timestamps.append(elapsed)
                buffer_rgb.append(rgb)
                draw_roi(frame, polygon)
                if show_debug:
                    for x, y in points.astype(int):
                        cv2.circle(frame, (x, y), 1, LANDMARK_COLOR, -1)

            while buffer_timestamps and elapsed - buffer_timestamps[0] > WINDOW_S:
                buffer_timestamps.popleft()
                buffer_rgb.popleft()

            if elapsed - last_bpm_update >= REFRESH_S and len(buffer_timestamps) > 1:
                bpm = estimate_bpm_from_buffer(
                    np.array(buffer_timestamps), np.array(buffer_rgb), method=METHOD
                )
                if bpm is not None:
                    current_bpm = bpm
                last_bpm_update = elapsed

            fps = smooth_fps(fps, now - last_frame_time)
            last_frame_time = now

            draw_overlay(frame, fps, landmarks is not None, current_bpm)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_debug = not show_debug
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
