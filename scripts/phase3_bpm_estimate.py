"""Phase 3: offline BPM estimation from a Phase 2 signal CSV.

Usage:  python scripts/phase3_bpm_estimate.py data/raw/signal_<id>.csv [--plot]

Pipeline: resample the (possibly irregular) frame timestamps to a uniform
rate, detrend, band-pass filter to the plausible heart-rate band, then take
the dominant frequency of the FFT as the BPM.
"""

import argparse
from pathlib import Path

import numpy as np

from vitalvision.io import load_signal
from vitalvision.signal import (
    BAND_HIGH_BPM,
    RESAMPLE_HZ,
    bandpass_filter,
    bpm_from_pulse,
    detrend,
    resample_uniform,
)


def plot_pipeline(
    raw: np.ndarray, filtered: np.ndarray, fs_hz: float, bpm: float
) -> None:
    """Show the raw vs. filtered signal and its spectrum, with the BPM peak marked."""
    import matplotlib.pyplot as plt

    windowed = filtered * np.hanning(len(filtered))
    magnitude = np.abs(np.fft.rfft(windowed))
    freqs_hz = np.fft.rfftfreq(len(windowed), d=1.0 / fs_hz)

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
    filtered = bandpass_filter(detrended, RESAMPLE_HZ)
    bpm = bpm_from_pulse(filtered, RESAMPLE_HZ)

    print(f"Duration:     {timestamps_s[-1] - timestamps_s[0]:.1f} s")
    print(f"Resampled to: {RESAMPLE_HZ:.0f} Hz ({len(uniform_green)} samples)")
    print(f"Estimated BPM: {bpm:.1f}" if bpm else "Estimated BPM: could not detect a peak")

    if args.plot:
        plot_pipeline(detrended, filtered, RESAMPLE_HZ, bpm or 0.0)


if __name__ == "__main__":
    main()
