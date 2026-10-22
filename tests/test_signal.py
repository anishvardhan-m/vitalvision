import numpy as np
import pytest

from vitalvision.signal import bandpass_filter, bpm_from_pulse, detrend, resample_uniform


def test_resample_uniform_produces_evenly_spaced_grid():
    timestamps = np.array([0.0, 0.31, 0.62, 1.0, 1.5])
    values = np.array([1.0, 2.0, 1.0, 2.0, 1.0])

    resampled = resample_uniform(timestamps, values, hz=10.0)

    assert len(resampled) == pytest.approx(15, abs=1)
    assert resampled.min() >= values.min() - 1e-6
    assert resampled.max() <= values.max() + 1e-6


def test_detrend_removes_linear_drift():
    t = np.linspace(0, 10, 300)
    drifting = 5.0 * t + np.sin(2 * np.pi * 1.2 * t)

    detrended = detrend(drifting)

    # a linear fit to the detrended signal should have ~zero slope left
    slope = np.polyfit(t, detrended, 1)[0]
    assert abs(slope) < 0.05


@pytest.mark.parametrize("true_bpm", [48.0, 72.0, 110.0, 160.0])
def test_bandpass_and_fft_recover_known_bpm(true_bpm):
    fs_hz = 30.0
    duration_s = 20.0
    t = np.arange(0, duration_s, 1.0 / fs_hz)
    freq_hz = true_bpm / 60.0

    rng = np.random.default_rng(0)
    pulse = np.sin(2 * np.pi * freq_hz * t) + 0.05 * rng.standard_normal(len(t))

    filtered = bandpass_filter(pulse, fs_hz)
    bpm = bpm_from_pulse(filtered, fs_hz)

    assert bpm is not None
    assert bpm == pytest.approx(true_bpm, abs=2.0)


def test_bpm_from_pulse_returns_none_when_no_bin_falls_in_band():
    # Too few samples at too low a rate: every FFT bin lands below BAND_LOW_BPM.
    fs_hz = 0.5
    pulse = np.array([1.0, -1.0, 1.0, -1.0])

    assert bpm_from_pulse(pulse, fs_hz) is None
