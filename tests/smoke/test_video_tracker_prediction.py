from __future__ import annotations

import numpy as np
import pytest

from visual_servo_tracker.tracking_runtime.video_tracker import _prediction_to_int_delta


def test_prediction_to_int_delta_from_column_vector() -> None:
    predicted = np.array([[1.8], [-2.2]], dtype=np.float64)
    dx, dy = _prediction_to_int_delta(predicted)
    assert dx == 1
    assert dy == -2


def test_prediction_to_int_delta_from_flat_vector() -> None:
    predicted = np.array([3.1, 4.9], dtype=np.float64)
    dx, dy = _prediction_to_int_delta(predicted)
    assert dx == 3
    assert dy == 4


def test_prediction_to_int_delta_invalid_shape() -> None:
    with pytest.raises(ValueError):
        _prediction_to_int_delta(np.array([[1.0]]))
