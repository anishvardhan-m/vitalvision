import numpy as np

from vitalvision.face import (
    FOREHEAD_BROW_ROW,
    FOREHEAD_TOP_ROW,
    ROI_BOTTOM_FRACTION,
    ROI_TOP_FRACTION,
    forehead_polygon,
)


def _fake_landmarks() -> np.ndarray:
    """468 points on a flat 2D grid; only indices used by the ROI rows matter."""
    points = np.zeros((468, 2), dtype=np.float32)
    for index in FOREHEAD_TOP_ROW:
        points[index] = (index, 0.0)
    for index in FOREHEAD_BROW_ROW:
        points[index] = (index, 100.0)
    return points


def test_forehead_polygon_has_two_rows_of_points():
    polygon = forehead_polygon(_fake_landmarks())
    assert polygon.shape == (2 * len(FOREHEAD_TOP_ROW), 2)


def test_forehead_polygon_edges_sit_between_top_and_brow_rows():
    polygon = forehead_polygon(_fake_landmarks())
    upper_edge_y = polygon[: len(FOREHEAD_TOP_ROW), 1]
    lower_edge_y = polygon[len(FOREHEAD_TOP_ROW) :, 1]

    assert np.allclose(upper_edge_y, 100.0 * ROI_TOP_FRACTION, atol=1)
    assert np.allclose(lower_edge_y, 100.0 * ROI_BOTTOM_FRACTION, atol=1)
    assert upper_edge_y.max() < lower_edge_y.min()
