"""Face landmark detection and forehead ROI construction (MediaPipe + OpenCV)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

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


def open_camera(index: int = 0) -> cv2.VideoCapture:
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


def roi_mean_rgb(frame_bgr: np.ndarray, polygon: np.ndarray) -> tuple[float, float, float]:
    """Return the ROI's mean (R, G, B) over the polygon, 0-255 scale."""
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    b, g, r = cv2.mean(frame_bgr, mask=mask)[:3]
    return r, g, b
