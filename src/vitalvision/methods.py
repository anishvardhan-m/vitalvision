"""rPPG methods: turn an ROI's RGB trace into a pulse signal.

- ``green``: the naive baseline, just the band-passed green channel.
- ``chrom``: de Haan & Jeanne (2013), a single fixed linear projection of R/G/B.
- ``pos``: Wang et al. (2017), a projection re-fit on every short sliding
  window and overlap-added, which tracks changing light and motion better.
"""

from __future__ import annotations

import numpy as np

from .signal import RESAMPLE_HZ, bandpass_filter, detrend

CHROM_X_WEIGHTS = (3.0, -2.0, 0.0)  # 3R - 2G
CHROM_Y_WEIGHTS = (1.5, 1.0, -1.5)  # 1.5R + G - 1.5B

# Rows act on normalised (R, G, B): row 0 is (G - B), row 1 is (-2R + G + B).
POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
POS_WINDOW_S = 1.6  # per Wang et al. 2017


def green(rgb: np.ndarray, fs_hz: float = RESAMPLE_HZ) -> np.ndarray:
    """The naive baseline: band-passed green channel only."""
    return bandpass_filter(detrend(rgb[:, 1]), fs_hz)


def chrom(rgb: np.ndarray, fs_hz: float = RESAMPLE_HZ) -> np.ndarray:
    """CHROM (de Haan & Jeanne, 2013): a fixed linear combination of R, G, B."""
    normalised = rgb / rgb.mean(axis=0)
    x = normalised @ np.array(CHROM_X_WEIGHTS)
    y = normalised @ np.array(CHROM_Y_WEIGHTS)
    alpha = np.std(x) / np.std(y)
    return bandpass_filter(detrend(x - alpha * y), fs_hz)


def pos(rgb: np.ndarray, fs_hz: float = RESAMPLE_HZ) -> np.ndarray:
    """POS (Wang et al., 2017): sliding-window projection, overlap-added."""
    n_samples = rgb.shape[0]
    window_len = max(int(round(POS_WINDOW_S * fs_hz)), 2)
    pulse = np.zeros(n_samples)

    for end in range(window_len, n_samples + 1):
        start = end - window_len
        normalised = rgb[start:end] / rgb[start:end].mean(axis=0)
        projected = POS_PROJECTION @ normalised.T  # (2, window_len)
        s1, s2 = projected[0], projected[1]
        h = s1 + (np.std(s1) / np.std(s2)) * s2
        pulse[start:end] += h - h.mean()

    return bandpass_filter(detrend(pulse), fs_hz)


METHODS = {"green": green, "chrom": chrom, "pos": pos}
