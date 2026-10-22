import numpy as np
import pytest

from vitalvision.methods import METHODS
from vitalvision.signal import bpm_from_pulse


def _synthetic_rgb(true_bpm: float, fs_hz: float = 30.0, duration_s: float = 20.0) -> np.ndarray:
    """A plausible pulsatile RGB trace: green most sensitive, blue least, plus noise."""
    t = np.arange(0, duration_s, 1.0 / fs_hz)
    pulse = np.sin(2 * np.pi * (true_bpm / 60.0) * t)
    rng = np.random.default_rng(0)
    r = 150 + 1.0 * pulse + 0.3 * rng.standard_normal(len(t))
    g = 120 + 2.0 * pulse + 0.3 * rng.standard_normal(len(t))
    b = 100 + 0.5 * pulse + 0.3 * rng.standard_normal(len(t))
    return np.column_stack([r, g, b])


@pytest.mark.parametrize("method_name", list(METHODS))
@pytest.mark.parametrize("true_bpm", [60.0, 90.0, 130.0])
def test_each_method_recovers_known_bpm(method_name, true_bpm):
    fs_hz = 30.0
    rgb = _synthetic_rgb(true_bpm, fs_hz=fs_hz)

    pulse = METHODS[method_name](rgb, fs_hz)
    bpm = bpm_from_pulse(pulse, fs_hz)

    assert bpm is not None
    assert bpm == pytest.approx(true_bpm, abs=3.0)
