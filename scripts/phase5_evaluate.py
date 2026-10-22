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
from pathlib import Path

import numpy as np

from vitalvision.io import load_ground_truth, load_signal
from vitalvision.methods import METHODS
from vitalvision.signal import RESAMPLE_HZ, bpm_from_pulse, resample_uniform


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
    bpms = {name: bpm_from_pulse(pulse, RESAMPLE_HZ) or 0.0 for name, pulse in pulses.items()}

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
