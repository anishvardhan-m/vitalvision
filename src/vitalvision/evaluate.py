"""Error metrics for comparing BPM estimates against ground truth."""

from __future__ import annotations

import numpy as np


def mae(estimates: np.ndarray | float, reference: np.ndarray | float) -> float:
    """Mean absolute error between estimated and reference BPM."""
    return float(np.mean(np.abs(np.asarray(estimates) - np.asarray(reference))))


def rmse(estimates: np.ndarray | float, reference: np.ndarray | float) -> float:
    """Root-mean-square error between estimated and reference BPM."""
    return float(np.sqrt(np.mean((np.asarray(estimates) - np.asarray(reference)) ** 2)))
