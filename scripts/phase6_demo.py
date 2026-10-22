"""Phase 6: demo app - live BPM with a method switch and trend sparkline.

Run from anywhere:  python scripts/phase6_demo.py
Keys: q = quit and save, g/c/p = switch to Green/CHROM/POS, d = debug landmarks.

Builds on Phase 4's rolling-buffer pipeline, adding the three Phase 5
methods (switchable live), a sparkline of recent BPM estimates drawn
directly on the frame, and a session log (timestamp_s, bpm, method) saved
to data/raw/session_<unix_time>.csv on quit.
"""

import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker
from scipy import signal as sp_signal

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"
DATA_DIR = REPO_ROOT / "data" / "raw"

CAMERA_INDEX = 0
WINDOW_NAME = "VitalVision - Demo"

FOREHEAD_TOP_ROW = (67, 109, 10, 338, 297)
FOREHEAD_BROW_ROW = (66, 107, 9, 336, 296)
ROI_TOP_FRACTION = 0.25
ROI_BOTTOM_FRACTION = 0.75

ROI_COLOR = (0, 255, 0)
ROI_FILL_ALPHA = 0.3
LANDMARK_COLOR = (0, 200, 255)
TEXT_COLOR = (255, 255, 255)
NO_FACE_COLOR = (0, 0, 255)
BPM_COLOR = (0, 215, 255)
SPARKLINE_COLOR = (0, 215, 255)
FPS_SMOOTHING = 0.9
CAMERA_RETRY_S = 2.0
CAMERA_RETRY_INTERVAL_S = 0.03

WINDOW_S = 10.0
REFRESH_S = 1.0
RESAMPLE_HZ = 30.0
BAND_LOW_BPM = 42.0
BAND_HIGH_BPM = 180.0
FILTER_ORDER = 3
MIN_SAMPLES_FOR_ESTIMATE = int(WINDOW_S * RESAMPLE_HZ * 0.5)
POS_WINDOW_S = 1.6

BPM_HISTORY_MAXLEN = 120  # ~2 minutes at one estimate per second
SPARKLINE_SIZE = (240, 60)  # (width, height) px
SPARKLINE_MARGIN = (10, 110)  # (x, y) top-left corner on the frame

CHROM_X_WEIGHTS = (3.0, -2.0, 0.0)
CHROM_Y_WEIGHTS = (1.5, 1.0, -1.5)
POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

METHOD_KEYS = {ord("g"): "green", ord("c"): "chrom", ord("p"): "pos"}


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


def roi_mean_rgb(frame_bgr: np.ndarray, polygon: np.ndarray) -> tuple[float, float, float]:
    """Return the ROI's mean (R, G, B), 0-255 scale."""
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    b, g, r = cv2.mean(frame_bgr, mask=mask)[:3]
    return r, g, b


def draw_roi(frame: np.ndarray, polygon: np.ndarray) -> None:
    """Draw the ROI as a semi-transparent fill with a solid outline, in place."""
    fill = frame.copy()
    cv2.fillPoly(fill, [polygon], ROI_COLOR)
    cv2.addWeighted(fill, ROI_FILL_ALPHA, frame, 1 - ROI_FILL_ALPHA, 0, dst=frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=ROI_COLOR, thickness=2)


def draw_overlay(
    frame: np.ndarray, fps: float, face_found: bool, bpm: float | None, method: str
) -> None:
    """Draw FPS, face status, the active method and the current BPM estimate."""
    if face_found:
        status, status_color = "Face detected", ROI_COLOR
    else:
        status, status_color = "No face", NO_FACE_COLOR
    bpm_text = f"BPM: {bpm:.0f} ({method})" if bpm else f"BPM: -- ({method})"
    lines = [
        (f"FPS: {fps:.1f}", TEXT_COLOR),
        (status, status_color),
        (bpm_text, BPM_COLOR),
    ]
    for row, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text, (10, 30 + row * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )


def draw_sparkline(frame: np.ndarray, bpm_history: deque[float]) -> None:
    """Draw a small BPM trend line, scaled to its own min/max, in place."""
    if len(bpm_history) < 2:
        return
    width, height = SPARKLINE_SIZE
    x0, y0 = SPARKLINE_MARGIN
    values = np.array(bpm_history)
    lo, hi = values.min(), values.max()
    span = max(hi - lo, 1e-6)

    xs = np.linspace(x0, x0 + width, len(values))
    ys = y0 + height - (values - lo) / span * height
    points = np.column_stack([xs, ys]).astype(np.int32)
    cv2.polylines(frame, [points], isClosed=False, color=SPARKLINE_COLOR, thickness=2)


def smooth_fps(previous_fps: float, frame_interval_s: float) -> float:
    """Blend the latest frame interval into an exponentially smoothed FPS."""
    instant_fps = 1.0 / max(frame_interval_s, 1e-6)
    if previous_fps == 0.0:
        return instant_fps
    return FPS_SMOOTHING * previous_fps + (1 - FPS_SMOOTHING) * instant_fps


def bandpass_filter(values: np.ndarray, fs_hz: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter, in the plausible heart-rate band."""
    nyquist = fs_hz / 2.0
    low = (BAND_LOW_BPM / 60.0) / nyquist
    high = (BAND_HIGH_BPM / 60.0) / nyquist
    b, a = sp_signal.butter(FILTER_ORDER, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, values)


def bpm_from_pulse(pulse: np.ndarray, fs_hz: float) -> float | None:
    """FFT-peak BPM estimate from a (band-passed) pulse signal."""
    windowed = pulse * np.hanning(len(pulse))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs_hz = np.fft.rfftfreq(len(windowed), d=1.0 / fs_hz)
    band = (freqs_hz * 60.0 >= BAND_LOW_BPM) & (freqs_hz * 60.0 <= BAND_HIGH_BPM)
    if not np.any(band):
        return None
    peak_freq_hz = freqs_hz[band][np.argmax(spectrum[band])]
    return peak_freq_hz * 60.0


def green_method(rgb: np.ndarray, fs_hz: float) -> np.ndarray:
    """The naive baseline: band-passed green channel only."""
    return bandpass_filter(sp_signal.detrend(rgb[:, 1], type="linear"), fs_hz)


def chrom_method(rgb: np.ndarray, fs_hz: float) -> np.ndarray:
    """CHROM (de Haan & Jeanne, 2013): a fixed linear combination of R, G, B."""
    normalised = rgb / rgb.mean(axis=0)
    x = normalised @ np.array(CHROM_X_WEIGHTS)
    y = normalised @ np.array(CHROM_Y_WEIGHTS)
    alpha = np.std(x) / np.std(y)
    return bandpass_filter(sp_signal.detrend(x - alpha * y, type="linear"), fs_hz)


def pos_method(rgb: np.ndarray, fs_hz: float) -> np.ndarray:
    """POS (Wang et al., 2017): sliding-window projection, overlap-added."""
    n_samples = rgb.shape[0]
    window_len = max(int(round(POS_WINDOW_S * fs_hz)), 2)
    pulse = np.zeros(n_samples)

    for end in range(window_len, n_samples + 1):
        start = end - window_len
        normalised = rgb[start:end] / rgb[start:end].mean(axis=0)
        projected = POS_PROJECTION @ normalised.T
        s1, s2 = projected[0], projected[1]
        h = s1 + (np.std(s1) / np.std(s2)) * s2
        pulse[start:end] += h - h.mean()

    return bandpass_filter(sp_signal.detrend(pulse, type="linear"), fs_hz)


METHODS = {"green": green_method, "chrom": chrom_method, "pos": pos_method}


def estimate_bpm_from_buffer(
    timestamps: np.ndarray, rgb: np.ndarray, method: str
) -> float | None:
    """Resample the buffered RGB signal and run the selected method's BPM estimate."""
    if len(timestamps) < MIN_SAMPLES_FOR_ESTIMATE:
        return None
    uniform_t = np.arange(timestamps[0], timestamps[-1], 1.0 / RESAMPLE_HZ)
    if len(uniform_t) < MIN_SAMPLES_FOR_ESTIMATE:
        return None
    uniform_rgb = np.column_stack(
        [np.interp(uniform_t, timestamps, rgb[:, i]) for i in range(3)]
    )
    pulse = METHODS[method](uniform_rgb, RESAMPLE_HZ)
    return bpm_from_pulse(pulse, RESAMPLE_HZ)


def save_session(rows: list[tuple[float, float, str]], path: Path) -> None:
    """Write (timestamp_s, bpm, method) rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "bpm", "method"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} BPM readings to {path}")


def main() -> None:
    """Run the live demo: switchable method, sparkline, and a saved session log."""
    landmarker = create_landmarker(MODEL_PATH)
    cap = open_camera(CAMERA_INDEX)
    show_debug = False
    method = "pos"
    fps = 0.0
    last_ts_ms = -1
    last_frame_time = time.monotonic()
    last_good_read = time.monotonic()
    start_time = time.monotonic()
    last_bpm_update = 0.0
    current_bpm: float | None = None

    buffer_timestamps: deque[float] = deque()
    buffer_rgb: deque[tuple[float, float, float]] = deque()
    bpm_history: deque[float] = deque(maxlen=BPM_HISTORY_MAXLEN)
    session_rows: list[tuple[float, float, str]] = []

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
                    np.array(buffer_timestamps), np.array(buffer_rgb), method
                )
                if bpm is not None:
                    current_bpm = bpm
                    bpm_history.append(bpm)
                    session_rows.append((elapsed, bpm, method))
                last_bpm_update = elapsed

            fps = smooth_fps(fps, now - last_frame_time)
            last_frame_time = now

            draw_overlay(frame, fps, landmarks is not None, current_bpm, method)
            draw_sparkline(frame, bpm_history)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_debug = not show_debug
            elif key in METHOD_KEYS:
                new_method = METHOD_KEYS[key]
                if new_method != method:
                    method = new_method
                    bpm_history.clear()
                    current_bpm = None
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    if session_rows:
        out_path = DATA_DIR / f"session_{int(time.time())}.csv"
        save_session(session_rows, out_path)
    else:
        print("No BPM readings collected; nothing saved.")


if __name__ == "__main__":
    main()
