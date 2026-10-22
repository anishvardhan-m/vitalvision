import numpy as np
import pytest

from vitalvision.evaluate import mae, rmse


def test_mae_zero_for_identical_values():
    assert mae(np.array([70.0, 80.0]), np.array([70.0, 80.0])) == 0.0


def test_mae_matches_hand_computed_value():
    estimates = np.array([70.0, 82.0, 60.0])
    reference = np.array([72.0, 80.0, 65.0])

    assert mae(estimates, reference) == pytest.approx((2.0 + 2.0 + 5.0) / 3.0)


def test_rmse_matches_hand_computed_value():
    estimates = np.array([70.0, 82.0])
    reference = np.array([72.0, 80.0])

    assert rmse(estimates, reference) == pytest.approx(np.sqrt((4.0 + 4.0) / 2.0))


def test_rmse_at_least_mae_for_nonuniform_errors():
    estimates = np.array([70.0, 100.0])
    reference = np.array([72.0, 80.0])

    assert rmse(estimates, reference) >= mae(estimates, reference)


def test_metrics_accept_scalars():
    assert mae(70.0, 72.0) == pytest.approx(2.0)
    assert rmse(70.0, 72.0) == pytest.approx(2.0)
