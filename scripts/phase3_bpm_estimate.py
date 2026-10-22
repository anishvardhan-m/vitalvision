"""Phase 3: offline BPM estimation from a Phase 2 signal CSV.

Usage:  python scripts/phase3_bpm_estimate.py data/raw/signal_<id>.csv [--plot]

Pipeline: resample the (possibly irregular) frame timestamps to a uniform
rate, detrend, band-pass filter to the plausible heart-rate band, then take
the dominant frequency of the FFT as the BPM.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

RESAMPLE_HZ = 30.0
BAND_LOW_BPM = 42.0
BAND_HIGH_BPM = 180.0
FILTER_ORDER = 3


def load_signal(path: Path) -> pd.DataFrame:
    """Load a Phase 2 CSV (timestamp_s, mean_r, mean_g, mean_b)."""
    df = pd.read_csv(path)
    required = {"timestamp_s", "mean_r", "mean_g", "mean_b"}
    if not required.issubset(df.columns):
        sys.exit(f"Error: {path} is missing columns {required - set(df.columns)}")
    return df


def resample_uniform(timestamps_s: np.ndarray, values: np.ndarray, hz: float) -> np.ndarray:
    """Linearly interpolate an irregularly-sampled signal onto a uniform time grid."""
    uniform_t = np.arange(timestamps_s[0], timestamps_s[-1], 1.0 / hz)
    return np.interp(uniform_t, timestamps_s, values)


def detrend(values: np.ndarray) -> np.ndarray:
    """Remove the linear trend (slow lighting drift) from a signal."""
    return sp_signal.detrend(values, type="linear")


def bandpass_filter(values: np.ndarray, fs_hz: float, low_bpm: float, high_bpm: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter, in the plausible heart-rate band."""
    nyquist = fs_hz / 2.0
    low = (low_bpm / 60.0) / nyquist
    high = (high_bpm / 60.0) / nyquist
    b, a = sp_signal.butter(FILTER_ORDER, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, values)


def estimate_bpm(filtered: np.ndarray, fs_hz: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (bpm, freqs_hz, magnitude) from the FFT of a filtered signal."""
    windowed = filtered * np.hanning(len(filtered))
    spectrum = np.fft.rfft(windowed)
    freqs_hz = np.fft.rfftfreq(len(windowed), d=1.0 / fs_hz)
    magnitude = np.abs(spectrum)

    band = (freqs_hz * 60.0 >= BAND_LOW_BPM) & (freqs_hz * 60.0 <= BAND_HIGH_BPM)
    if not np.any(band):
        return 0.0, freqs_hz, magnitude
    peak_index = np.argmax(magnitude[band])
    peak_freq_hz = freqs_hz[band][peak_index]
    return peak_freq_hz * 60.0, freqs_hz, magnitude


def plot_pipeline(
    raw: np.ndarray, filtered: np.ndarray, freqs_hz: np.ndarray, magnitude: np.ndarray, bpm: float
) -> None:
    """Show the raw vs. filtered signal and its spectrum, with the BPM peak marked."""
    import matplotlib.pyplot as plt

    fig, (ax_signal, ax_spectrum) = plt.subplots(2, 1, figsize=(9, 6))
    ax_signal.plot(raw, label="raw (detrended) green channel", alpha=0.5)
    ax_signal.plot(filtered, label="band-passed", linewidth=2)
    ax_signal.set_xlabel("sample")
    ax_signal.legend()

    bpm_axis = freqs_hz * 60.0
    ax_spectrum.plot(bpm_axis, magnitude)
    ax_spectrum.axvline(bpm, color="red", linestyle="--", label=f"peak = {bpm:.1f} BPM")
    ax_spectrum.set_xlim(0, BAND_HIGH_BPM + 60)
    ax_spectrum.set_xlabel("BPM")
    ax_spectrum.set_ylabel("magnitude")
    ax_spectrum.legend()

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Phase 2 signal CSV")
    parser.add_argument("--plot", action="store_true", help="show the signal/spectrum plot")
    args = parser.parse_args()

    df = load_signal(args.csv_path)
    timestamps_s = df["timestamp_s"].to_numpy()
    green = df["mean_g"].to_numpy()

    uniform_green = resample_uniform(timestamps_s, green, RESAMPLE_HZ)
    detrended = detrend(uniform_green)
    filtered = bandpass_filter(detrended, RESAMPLE_HZ, BAND_LOW_BPM, BAND_HIGH_BPM)
    bpm, freqs_hz, magnitude = estimate_bpm(filtered, RESAMPLE_HZ)

    print(f"Duration:     {timestamps_s[-1] - timestamps_s[0]:.1f} s")
    print(f"Resampled to: {RESAMPLE_HZ:.0f} Hz ({len(uniform_green)} samples)")
    print(f"Estimated BPM: {bpm:.1f}")

    if args.plot:
        plot_pipeline(detrended, filtered, freqs_hz, magnitude, bpm)


if __name__ == "__main__":
    main()
