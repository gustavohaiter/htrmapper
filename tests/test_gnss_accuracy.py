from __future__ import annotations

import numpy as np
import pytest

from htrmapper.gnss.accuracy import (
    MIN_SIGMA_M,
    CameraAccuracy,
    gnss_position_residual_cost,
)


def test_weight_is_inverse_variance():
    accuracy = CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.05)

    weights = accuracy.weight_diagonal()

    assert weights[0] == pytest.approx(1.0 / 0.02**2)
    assert weights[1] == pytest.approx(1.0 / 0.02**2)
    assert weights[2] == pytest.approx(1.0 / 0.05**2)


def test_smaller_sigma_means_larger_weight():
    tight = CameraAccuracy(xy_sigma_m=0.01, z_sigma_m=0.01)
    loose = CameraAccuracy(xy_sigma_m=0.10, z_sigma_m=0.10)

    assert tight.weight_diagonal()[0] > loose.weight_diagonal()[0]


def test_zero_sigma_is_rejected():
    with pytest.raises(ValueError):
        CameraAccuracy(xy_sigma_m=0.0, z_sigma_m=0.02)
    with pytest.raises(ValueError):
        CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.0)


def test_negative_sigma_is_rejected():
    with pytest.raises(ValueError):
        CameraAccuracy(xy_sigma_m=-0.01, z_sigma_m=0.02)


def test_extremely_small_sigma_is_floored_not_infinite():
    accuracy = CameraAccuracy(xy_sigma_m=1e-12, z_sigma_m=1e-12)

    weights = accuracy.weight_diagonal()

    assert np.all(np.isfinite(weights))
    assert accuracy.effective_xy_sigma_m == MIN_SIGMA_M


def test_weight_matrix_is_diagonal_3x3():
    accuracy = CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.05)

    matrix = accuracy.weight_matrix()

    assert matrix.shape == (3, 3)
    off_diagonal = matrix - np.diag(np.diagonal(matrix))
    assert np.allclose(off_diagonal, 0.0)


def test_residual_cost_is_zero_when_estimate_matches_gnss_exactly():
    accuracy = CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.02)
    position = np.array([100.0, 200.0, 850.0])

    cost = gnss_position_residual_cost(position, position, accuracy)

    assert cost == pytest.approx(0.0)


def test_residual_cost_scales_with_squared_error_and_weight():
    accuracy = CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.02)
    estimated = np.array([100.10, 200.0, 850.0])  # 10 cm off in X
    gnss = np.array([100.0, 200.0, 850.0])

    cost = gnss_position_residual_cost(estimated, gnss, accuracy)

    expected = (0.10**2) / (0.02**2)
    assert cost == pytest.approx(expected)


def test_looser_accuracy_produces_lower_cost_for_same_residual():
    tight = CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.02)
    loose = CameraAccuracy(xy_sigma_m=0.10, z_sigma_m=0.10)
    estimated = np.array([100.10, 200.0, 850.0])
    gnss = np.array([100.0, 200.0, 850.0])

    cost_tight = gnss_position_residual_cost(estimated, gnss, tight)
    cost_loose = gnss_position_residual_cost(estimated, gnss, loose)

    assert cost_tight > cost_loose


def test_to_dict_round_trip():
    accuracy = CameraAccuracy(xy_sigma_m=0.03, z_sigma_m=0.06)

    restored = CameraAccuracy.from_dict(accuracy.to_dict())

    assert restored == accuracy
