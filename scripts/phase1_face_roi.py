"""Phase 1: live face landmarks and a forehead ROI from the webcam.

Run from anywhere:  python scripts/phase1_face_roi.py
Keys: q = quit, d = toggle landmark debug view, s = save screenshot.
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"
SCREENSHOT_PATH = REPO_ROOT / "docs" / "images" / "phase1_roi.png"

CAMERA_INDEX = 0
WINDOW_NAME = "VitalVision - Phase 1"

# The forehead ROI is built from two column-aligned rows of landmarks, ordered
# subject's right -> subject's left. "Right"/"left" are from the subject's point
# of view, so after mirroring they appear on the same side of the screen.
FOREHEAD_TOP_ROW = (
    67,  # upper forehead, right of centre
    109,  # upper forehead, just right of centre
    10,  # top-centre of forehead (top of the face mesh, near the hairline)
    338,  # upper forehead, just left of centre
    297,  # upper forehead, left of centre
)
FOREHEAD_BROW_ROW = (
    66,  # top edge of right eyebrow, middle
    107,  # top edge of right eyebrow, inner end
    9,  # glabella, between the eyebrows
    336,  # top edge of left eyebrow, inner end
    296,  # top edge of left eyebrow, middle
)
# Where the ROI edges sit, as a fraction of the way from the top row (0.0) down
# to the brow row (1.0). Insets keep clear of the hairline and the eyebrows.
ROI_TOP_FRACTION = 0.25
ROI_BOTTOM_FRACTION = 0.75

ROI_COLOR = (0, 255, 0)
ROI_FILL_ALPHA = 0.3
LANDMARK_COLOR = (0, 200, 255)
TEXT_COLOR = (255, 255, 255)
NO_FACE_COLOR = (0, 0, 255)
FPS_SMOOTHING = 0.9  # EMA weight on the previous FPS estimate
# macOS cameras can return empty frames while warming up; keep retrying this long.
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
    # Walk the upper edge one way and the lower edge back to close the shape.
    return np.round(np.vstack([upper_edge, lower_edge[::-1]])).astype(np.int32)


def draw_roi(frame: np.ndarray, polygon: np.ndarray) -> None:
    """Draw the ROI as a semi-transparent fill with a solid outline, in place."""
    fill = frame.copy()
    cv2.fillPoly(fill, [polygon], ROI_COLOR)
    cv2.addWeighted(fill, ROI_FILL_ALPHA, frame, 1 - ROI_FILL_ALPHA, 0, dst=frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=ROI_COLOR, thickness=2)


def draw_debug_landmarks(frame: np.ndarray, points: np.ndarray) -> None:
    """Draw every landmark as a dot labelled with its index, in place."""
    for index, (x, y) in enumerate(points.astype(int)):
        cv2.circle(frame, (x, y), 1, LANDMARK_COLOR, -1)
        cv2.putText(
            frame, str(index), (x + 2, y - 2), cv2.FONT_HERSHEY_PLAIN, 0.6, LANDMARK_COLOR, 1
        )


def draw_overlay(frame: np.ndarray, fps: float, face_found: bool, roi_area: float) -> None:
    """Draw FPS, face status and ROI area in the top-left corner, in place."""
    if face_found:
        status, status_color = "Face detected", ROI_COLOR
    else:
        status, status_color = "No face", NO_FACE_COLOR
    lines = [
        (f"FPS: {fps:.1f}", TEXT_COLOR),
        (status, status_color),
        (f"ROI area: {roi_area:.0f} px", TEXT_COLOR),
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


def save_screenshot(frame: np.ndarray, path: Path) -> None:
    """Write the frame to disk, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    print(f"Saved screenshot to {path}")


def main() -> None:
    """Run the live webcam loop until the user presses q."""
    landmarker = create_landmarker(MODEL_PATH)
    cap = open_camera(CAMERA_INDEX)
    show_debug = False
    fps = 0.0
    last_ts_ms = -1
    last_frame_time = time.monotonic()
    last_good_read = time.monotonic()

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

            # Mirror before detection so landmarks are already in display coordinates.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            last_ts_ms = next_timestamp_ms(last_ts_ms)
            landmarks = detect_landmarks(landmarker, frame, last_ts_ms)

            roi_area = 0.0
            if landmarks is not None:
                points = landmarks_to_pixels(landmarks, width, height)
                polygon = forehead_polygon(points)
                roi_area = cv2.contourArea(polygon)
                draw_roi(frame, polygon)
                if show_debug:
                    draw_debug_landmarks(frame, points)

            now = time.monotonic()
            fps = smooth_fps(fps, now - last_frame_time)
            last_frame_time = now

            draw_overlay(frame, fps, landmarks is not None, roi_area)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_debug = not show_debug
            elif key == ord("s"):
                save_screenshot(frame, SCREENSHOT_PATH)
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
