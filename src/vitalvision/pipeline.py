"""Rolling-buffer live BPM estimation, combining resampling and an rPPG method."""

from __future__ import annotations

import numpy as np

from .methods import METHODS
from .signal import RESAMPLE_HZ, bpm_from_pulse

WINDOW_S = 10.0  # how much signal history to estimate BPM from
REFRESH_S = 1.0  # how often to recompute the BPM
MIN_WINDOW_FRACTION = 0.5  # fraction of a full window required before estimating


def min_samples_for_window(window_s: float = WINDOW_S, fs_hz: float = RESAMPLE_HZ) -> int:
    """Minimum resampled sample count before an estimate is attempted."""
    return int(window_s * fs_hz * MIN_WINDOW_FRACTION)


def estimate_bpm_from_buffer(
    timestamps: np.ndarray,
    rgb: np.ndarray,
    method: str = "pos",
    fs_hz: float = RESAMPLE_HZ,
    min_samples: int | None = None,
) -> float | None:
    """Resample a buffered (timestamps, RGB) trace and run the named method's BPM estimate.

    Returns None if there isn't enough signal yet for a reliable estimate.
    """
    if min_samples is None:
        min_samples = min_samples_for_window(fs_hz=fs_hz)
    if len(timestamps) < min_samples:
        return None

    uniform_t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs_hz)
    if len(uniform_t) < min_samples:
        return None
    uniform_rgb = np.column_stack(
        [np.interp(uniform_t, timestamps, rgb[:, i]) for i in range(3)]
    )

    pulse = METHODS[method](uniform_rgb, fs_hz)
    return bpm_from_pulse(pulse, fs_hz)
