"""Resampling, filtering, and FFT-based BPM estimation for rPPG signals."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

RESAMPLE_HZ = 30.0
# Narrowed from (42, 180) to exclude a recurring ~50 BPM camera auto-exposure
# artifact seen on the M4 Mac's built-in webcam that otherwise dominates the
# FFT peak. Widen back out if evaluating someone whose true rate falls
# outside 55-150.
BAND_LOW_BPM = 55.0
BAND_HIGH_BPM = 150.0
FILTER_ORDER = 3


def resample_uniform(
    timestamps_s: np.ndarray, values: np.ndarray, hz: float = RESAMPLE_HZ
) -> np.ndarray:
    """Linearly interpolate an irregularly-sampled signal onto a uniform time grid."""
    uniform_t = np.arange(timestamps_s[0], timestamps_s[-1], 1.0 / hz)
    return np.interp(uniform_t, timestamps_s, values)


def detrend(values: np.ndarray) -> np.ndarray:
    """Remove the linear trend (slow lighting drift) from a signal."""
    return sp_signal.detrend(values, type="linear")


def bandpass_filter(
    values: np.ndarray,
    fs_hz: float = RESAMPLE_HZ,
    low_bpm: float = BAND_LOW_BPM,
    high_bpm: float = BAND_HIGH_BPM,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter, in the plausible heart-rate band."""
    nyquist = fs_hz / 2.0
    low = (low_bpm / 60.0) / nyquist
    high = (high_bpm / 60.0) / nyquist
    b, a = sp_signal.butter(FILTER_ORDER, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, values)


def bpm_from_pulse(
    pulse: np.ndarray,
    fs_hz: float = RESAMPLE_HZ,
    low_bpm: float = BAND_LOW_BPM,
    high_bpm: float = BAND_HIGH_BPM,
) -> float | None:
    """Return the dominant frequency of a (band-passed) pulse signal, as BPM.

    Returns None if no frequency bin falls inside the plausible heart-rate band.
    """
    windowed = pulse * np.hanning(len(pulse))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs_hz = np.fft.rfftfreq(len(windowed), d=1.0 / fs_hz)

    band = (freqs_hz * 60.0 >= low_bpm) & (freqs_hz * 60.0 <= high_bpm)
    if not np.any(band):
        return None
    peak_freq_hz = freqs_hz[band][np.argmax(spectrum[band])]
    return peak_freq_hz * 60.0
