"""Phase 5: compare Green / CHROM / POS rPPG methods against ground truth.

Usage:
    python scripts/phase5_evaluate.py data/raw/signal_<id>.csv
    python scripts/phase5_evaluate.py data/raw/signal_<id>.csv --ground-truth data/raw/gt_<id>.csv

The signal CSV is a Phase 2 recording (timestamp_s, mean_r, mean_g, mean_b).
The optional ground-truth CSV is a pulse-oximeter log with columns
(timestamp_s, bpm); its mean BPM over the clip is used as the reference
against which each method's error is reported. Without --ground-truth, only
the three methods' BPM estimates are printed side by side.
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
POS_WINDOW_S = 1.6  # per Wang et al. 2017

# CHROM projection (de Haan & Jeanne, 2013).
CHROM_X_WEIGHTS = (3.0, -2.0, 0.0)  # 3R - 2G
CHROM_Y_WEIGHTS = (1.5, 1.0, -1.5)  # 1.5R + G - 1.5B

# POS projection matrix (Wang et al., 2017): rows act on normalised (R, G, B).
POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def load_signal(path: Path) -> pd.DataFrame:
    """Load a Phase 2 CSV (timestamp_s, mean_r, mean_g, mean_b)."""
    df = pd.read_csv(path)
    required = {"timestamp_s", "mean_r", "mean_g", "mean_b"}
    if not required.issubset(df.columns):
        sys.exit(f"Error: {path} is missing columns {required - set(df.columns)}")
    return df


def load_ground_truth(path: Path) -> float:
    """Load a (timestamp_s, bpm) ground-truth CSV and return its mean BPM."""
    df = pd.read_csv(path)
    if "bpm" not in df.columns:
        sys.exit(f"Error: {path} must have a 'bpm' column")
    return float(df["bpm"].mean())


def resample_uniform(timestamps_s: np.ndarray, values: np.ndarray, hz: float) -> np.ndarray:
    """Linearly interpolate an irregularly-sampled signal onto a uniform time grid."""
    uniform_t = np.arange(timestamps_s[0], timestamps_s[-1], 1.0 / hz)
    return np.interp(uniform_t, timestamps_s, values)


def bandpass_filter(values: np.ndarray, fs_hz: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter, in the plausible heart-rate band."""
    nyquist = fs_hz / 2.0
    low = (BAND_LOW_BPM / 60.0) / nyquist
    high = (BAND_HIGH_BPM / 60.0) / nyquist
    b, a = sp_signal.butter(FILTER_ORDER, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, values)


def bpm_from_pulse(pulse: np.ndarray, fs_hz: float) -> float:
    """FFT-peak BPM estimate from a (band-passed) pulse signal."""
    windowed = pulse * np.hanning(len(pulse))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs_hz = np.fft.rfftfreq(len(windowed), d=1.0 / fs_hz)
    band = (freqs_hz * 60.0 >= BAND_LOW_BPM) & (freqs_hz * 60.0 <= BAND_HIGH_BPM)
    if not np.any(band):
        return 0.0
    peak_freq_hz = freqs_hz[band][np.argmax(spectrum[band])]
    return peak_freq_hz * 60.0


def green_method(rgb: np.ndarray, fs_hz: float) -> np.ndarray:
    """The naive baseline: band-passed green channel only."""
    green = sp_signal.detrend(rgb[:, 1], type="linear")
    return bandpass_filter(green, fs_hz)


def chrom_method(rgb: np.ndarray, fs_hz: float) -> np.ndarray:
    """CHROM (de Haan & Jeanne, 2013): a fixed linear combination of R, G, B."""
    mean_rgb = rgb.mean(axis=0)
    normalised = rgb / mean_rgb  # (N, 3), each channel's own DC removed multiplicatively

    x = normalised @ np.array(CHROM_X_WEIGHTS)
    y = normalised @ np.array(CHROM_Y_WEIGHTS)
    alpha = np.std(x) / np.std(y)
    chrom_signal = x - alpha * y
    return bandpass_filter(sp_signal.detrend(chrom_signal, type="linear"), fs_hz)


def pos_method(rgb: np.ndarray, fs_hz: float) -> np.ndarray:
    """POS (Wang et al., 2017): sliding-window projection, overlap-added.

    Unlike CHROM's single fixed projection over the whole clip, POS
    re-normalises and re-projects every short window, which tracks changing
    lighting and motion better.
    """
    n_samples = rgb.shape[0]
    window_len = max(int(round(POS_WINDOW_S * fs_hz)), 2)
    pulse = np.zeros(n_samples)

    for end in range(window_len, n_samples + 1):
        start = end - window_len
        window = rgb[start:end]  # (window_len, 3)
        mean_rgb = window.mean(axis=0)
        normalised = window / mean_rgb

        projected = (POS_PROJECTION @ normalised.T)  # (2, window_len)
        s1, s2 = projected[0], projected[1]
        std_ratio = np.std(s1) / np.std(s2)
        h = s1 + std_ratio * s2
        pulse[start:end] += h - h.mean()

    return bandpass_filter(sp_signal.detrend(pulse, type="linear"), fs_hz)


METHODS = {
    "green": green_method,
    "chrom": chrom_method,
    "pos": pos_method,
}


def plot_methods(pulses: dict[str, np.ndarray], bpms: dict[str, float]) -> None:
    """Show each method's filtered pulse signal, labelled with its BPM estimate."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(pulses), 1, figsize=(9, 2.5 * len(pulses)), sharex=True)
    for ax, (name, pulse) in zip(axes, pulses.items()):
        ax.plot(pulse)
        ax.set_ylabel(name)
        ax.set_title(f"{name}: {bpms[name]:.1f} BPM")
    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Phase 2 signal CSV")
    parser.add_argument("--ground-truth", type=Path, default=None, help="pulse-oximeter CSV")
    parser.add_argument("--plot", action="store_true", help="show each method's pulse signal")
    args = parser.parse_args()

    df = load_signal(args.csv_path)
    timestamps_s = df["timestamp_s"].to_numpy()
    raw_rgb = df[["mean_r", "mean_g", "mean_b"]].to_numpy()

    uniform_rgb = np.column_stack(
        [resample_uniform(timestamps_s, raw_rgb[:, i], RESAMPLE_HZ) for i in range(3)]
    )

    pulses = {name: method(uniform_rgb, RESAMPLE_HZ) for name, method in METHODS.items()}
    bpms = {name: bpm_from_pulse(pulse, RESAMPLE_HZ) for name, pulse in pulses.items()}

    print(f"{'method':<8} {'BPM':>8}", end="")
    reference_bpm = None
    if args.ground_truth is not None:
        reference_bpm = load_ground_truth(args.ground_truth)
        print(f" {'error':>8}")
    else:
        print()

    for name, bpm in bpms.items():
        print(f"{name:<8} {bpm:>8.1f}", end="")
        if reference_bpm is not None:
            print(f" {bpm - reference_bpm:>+8.1f}")
        else:
            print()

    if reference_bpm is not None:
        print(f"\nground truth: {reference_bpm:.1f} BPM")

    if args.plot:
        plot_methods(pulses, bpms)


if __name__ == "__main__":
    main()
