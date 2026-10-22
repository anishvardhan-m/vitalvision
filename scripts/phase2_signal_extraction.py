"""Phase 2: extract the raw forehead-ROI colour signal from the webcam.

Run from anywhere:  python scripts/phase2_signal_extraction.py
Keys: q = quit and save, d = toggle landmark debug view.

On quit, writes a CSV of (timestamp_s, mean_r, mean_g, mean_b) -- one row per
frame where a face was found -- to data/raw/signal_<unix_time>.csv.
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"
DATA_DIR = REPO_ROOT / "data" / "raw"

CAMERA_INDEX = 0
WINDOW_NAME = "VitalVision - Phase 2"

# See scripts/phase1_face_roi.py for the landmark layout this mirrors.
FOREHEAD_TOP_ROW = (67, 109, 10, 338, 297)
FOREHEAD_BROW_ROW = (66, 107, 9, 336, 296)
ROI_TOP_FRACTION = 0.25
ROI_BOTTOM_FRACTION = 0.75

ROI_COLOR = (0, 255, 0)
ROI_FILL_ALPHA = 0.3
LANDMARK_COLOR = (0, 200, 255)
TEXT_COLOR = (255, 255, 255)
NO_FACE_COLOR = (0, 0, 255)
FPS_SMOOTHING = 0.9
CAMERA_RETRY_S = 2.0
CAMERA_RETRY_INTERVAL_S = 0.03


def open_camera(index: int) -> cv2.VideoCapture:
    """Open the webcam or exit with a clear error."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Error: could not open camera {index}.")
    return cap


def create_landmarker(model_path: Path) -> FaceLandmarker:
    """Create a CPU, video-mode Face Landmarker tracking a single face."""
    if not model_path.exists():
        sys.exit(f"Model not found at {model_path}.\nRun: scripts/download_model.sh")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return FaceLandmarker.create_from_options(options)


def next_timestamp_ms(last_ms: int) -> int:
    """Return a monotonic millisecond timestamp strictly greater than last_ms."""
    return max(int(time.monotonic() * 1000), last_ms + 1)


def detect_landmarks(
    landmarker: FaceLandmarker, frame_bgr: np.ndarray, timestamp_ms: int
) -> list | None:
    """Run the landmarker on a BGR frame; return the face's landmarks or None."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)
    return result.face_landmarks[0] if result.face_landmarks else None


def landmarks_to_pixels(landmarks: list, width: int, height: int) -> np.ndarray:
    """Convert normalised landmarks to an (N, 2) float array of pixel coordinates."""
    return np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float32)


def forehead_polygon(points: np.ndarray) -> np.ndarray:
    """Build the forehead ROI polygon (int32, clockwise) from pixel landmarks."""
    top_row = points[list(FOREHEAD_TOP_ROW)]
    brow_row = points[list(FOREHEAD_BROW_ROW)]
    upper_edge = top_row + ROI_TOP_FRACTION * (brow_row - top_row)
    lower_edge = top_row + ROI_BOTTOM_FRACTION * (brow_row - top_row)
    return np.round(np.vstack([upper_edge, lower_edge[::-1]])).astype(np.int32)


def roi_mean_color(frame_bgr: np.ndarray, polygon: np.ndarray) -> tuple[float, float, float]:
    """Return the ROI's mean (R, G, B) over the polygon, 0-255 scale."""
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
    b, g, r = mean_bgr
    return r, g, b


def draw_roi(frame: np.ndarray, polygon: np.ndarray) -> None:
    """Draw the ROI as a semi-transparent fill with a solid outline, in place."""
    fill = frame.copy()
    cv2.fillPoly(fill, [polygon], ROI_COLOR)
    cv2.addWeighted(fill, ROI_FILL_ALPHA, frame, 1 - ROI_FILL_ALPHA, 0, dst=frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=ROI_COLOR, thickness=2)


def draw_overlay(frame: np.ndarray, fps: float, face_found: bool, sample_count: int) -> None:
    """Draw FPS, face status and the number of signal samples collected so far."""
    if face_found:
        status, status_color = "Face detected", ROI_COLOR
    else:
        status, status_color = "No face", NO_FACE_COLOR
    lines = [
        (f"FPS: {fps:.1f}", TEXT_COLOR),
        (status, status_color),
        (f"Samples: {sample_count}", TEXT_COLOR),
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


def save_signal(samples: list[tuple[float, float, float, float]], path: Path) -> None:
    """Write (timestamp_s, mean_r, mean_g, mean_b) rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "mean_r", "mean_g", "mean_b"])
        writer.writerows(samples)
    print(f"Saved {len(samples)} samples to {path}")


def main() -> None:
    """Run the live webcam loop, collecting the ROI signal until the user quits."""
    landmarker = create_landmarker(MODEL_PATH)
    cap = open_camera(CAMERA_INDEX)
    show_debug = False
    fps = 0.0
    last_ts_ms = -1
    last_frame_time = time.monotonic()
    last_good_read = time.monotonic()
    start_time = time.monotonic()
    samples: list[tuple[float, float, float, float]] = []

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

            if landmarks is not None:
                points = landmarks_to_pixels(landmarks, width, height)
                polygon = forehead_polygon(points)
                r, g, b = roi_mean_color(frame, polygon)
                samples.append((time.monotonic() - start_time, r, g, b))
                draw_roi(frame, polygon)
                if show_debug:
                    for x, y in points.astype(int):
                        cv2.circle(frame, (x, y), 1, LANDMARK_COLOR, -1)

            now = time.monotonic()
            fps = smooth_fps(fps, now - last_frame_time)
            last_frame_time = now

            draw_overlay(frame, fps, landmarks is not None, len(samples))
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

    if samples:
        out_path = DATA_DIR / f"signal_{int(time.time())}.csv"
        save_signal(samples, out_path)
    else:
        print("No samples collected; nothing saved.")


if __name__ == "__main__":
    main()
