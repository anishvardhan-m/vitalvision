"""CSV I/O for recorded signals, session logs, and ground-truth readings."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd


def load_signal(path: Path) -> pd.DataFrame:
    """Load a recorded signal CSV (timestamp_s, mean_r, mean_g, mean_b)."""
    df = pd.read_csv(path)
    required = {"timestamp_s", "mean_r", "mean_g", "mean_b"}
    if not required.issubset(df.columns):
        sys.exit(f"Error: {path} is missing columns {required - set(df.columns)}")
    return df


def save_signal(samples: list[tuple[float, float, float, float]], path: Path) -> None:
    """Write (timestamp_s, mean_r, mean_g, mean_b) rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "mean_r", "mean_g", "mean_b"])
        writer.writerows(samples)


def save_session(rows: list[tuple[float, float, str]], path: Path) -> None:
    """Write (timestamp_s, bpm, method) rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "bpm", "method"])
        writer.writerows(rows)


def load_ground_truth(path: Path) -> float:
    """Load a (timestamp_s, bpm) pulse-oximeter CSV and return its mean BPM."""
    df = pd.read_csv(path)
    if "bpm" not in df.columns:
        sys.exit(f"Error: {path} must have a 'bpm' column")
    return float(df["bpm"].mean())
