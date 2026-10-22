import numpy as np
import pytest

from vitalvision.pipeline import estimate_bpm_from_buffer, min_samples_for_window


def _synthetic_rgb(
    true_bpm: float, fs_hz: float, duration_s: float
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0, duration_s, 1.0 / fs_hz)
    pulse = np.sin(2 * np.pi * (true_bpm / 60.0) * t)
    rng = np.random.default_rng(0)
    r = 150 + 1.0 * pulse + 0.3 * rng.standard_normal(len(t))
    g = 120 + 2.0 * pulse + 0.3 * rng.standard_normal(len(t))
    b = 100 + 0.5 * pulse + 0.3 * rng.standard_normal(len(t))
    return t, np.column_stack([r, g, b])


def test_returns_none_before_the_buffer_fills_up():
    t, rgb = _synthetic_rgb(true_bpm=75.0, fs_hz=30.0, duration_s=1.0)

    assert estimate_bpm_from_buffer(t, rgb, method="pos") is None


@pytest.mark.parametrize("method", ["green", "chrom", "pos"])
def test_recovers_known_bpm_once_the_buffer_is_full(method):
    fs_hz = 30.0
    t, rgb = _synthetic_rgb(true_bpm=85.0, fs_hz=fs_hz, duration_s=10.0)

    bpm = estimate_bpm_from_buffer(t, rgb, method=method, fs_hz=fs_hz)

    assert bpm is not None
    assert bpm == pytest.approx(85.0, abs=3.0)


def test_min_samples_for_window_scales_with_window_and_rate():
    assert min_samples_for_window(window_s=10.0, fs_hz=30.0) == 150
    assert min_samples_for_window(window_s=5.0, fs_hz=30.0) == 75
