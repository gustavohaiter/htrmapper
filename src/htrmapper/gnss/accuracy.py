"""GNSS/PPK observation accuracy for camera positions.

This module defines how the precision of a camera's GNSS/PPK position is
represented and turned into a *weight* for use as a weighted observation in
the bundle adjustment (implemented in a later phase). It intentionally does
NOT implement the bundle adjustment itself -- only the data model and the
weighting math, which is independent of the solver used later.

Mathematical basis (standard weighted least squares / Gauss-Markov model):

    weight_i = 1 / sigma_i^2

where sigma_i is the 1-sigma standard deviation of observation i. This is
the same principle used in geodetic adjustment software: a GNSS/PPK
position is treated as an *observation with uncertainty*, not as a fixed
("hard") constraint. A small sigma (e.g. 0.02 m from a good PPK fixed
solution) yields a large weight, pulling the adjusted camera position
strongly toward the GNSS value; a large sigma yields a small weight, letting
the reprojection-error term dominate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A sigma of exactly zero would produce an infinite weight (a hard
# constraint) and is numerically unsound in a weighted least squares solver
# (singular / ill-conditioned normal equations). We floor sigma at this
# value, which is far below any real GNSS/PPK accuracy (sub-millimeter),
# so it never meaningfully changes the result -- it only prevents division
# by zero.
MIN_SIGMA_M = 1.0e-4

# Common presets mentioned in the project brief. These are convenience
# values for the UI; any positive float is accepted.
COMMON_SIGMA_PRESETS_M: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.10)


@dataclass(frozen=True)
class CameraAccuracy:
    """1-sigma accuracy of a camera's GNSS/PPK position observation.

    xy_sigma_m: horizontal (X, Y) standard deviation, in meters, in the
        camera's local ENU frame (or equivalently, in the project's
        projected CRS, since it is a planar projection).
    z_sigma_m: vertical (Z) standard deviation, in meters. Vertical GNSS
        accuracy is typically 1.5-3x worse than horizontal, so this is
        deliberately a separate field, not derived from xy_sigma_m.
    """

    xy_sigma_m: float
    z_sigma_m: float

    def __post_init__(self) -> None:
        if self.xy_sigma_m <= 0 or self.z_sigma_m <= 0:
            raise ValueError(
                "camera accuracy sigmas must be positive "
                f"(got xy={self.xy_sigma_m}, z={self.z_sigma_m}); "
                "a zero sigma would imply infinite confidence and produce "
                "a singular weight in the adjustment"
            )

    @property
    def effective_xy_sigma_m(self) -> float:
        return max(self.xy_sigma_m, MIN_SIGMA_M)

    @property
    def effective_z_sigma_m(self) -> float:
        return max(self.z_sigma_m, MIN_SIGMA_M)

    def weight_diagonal(self) -> np.ndarray:
        """Return the diagonal weight vector [w_x, w_y, w_z] = 1/sigma^2.

        This is the diagonal of the weight matrix W used in the GNSS
        position residual term of the bundle adjustment:

            residual = t_camera_estimated - t_camera_gnss
            cost += residual^T @ W @ residual   (W = diag(w_x, w_y, w_z))
        """
        wx = 1.0 / (self.effective_xy_sigma_m**2)
        wy = 1.0 / (self.effective_xy_sigma_m**2)
        wz = 1.0 / (self.effective_z_sigma_m**2)
        return np.array([wx, wy, wz], dtype=np.float64)

    def weight_matrix(self) -> np.ndarray:
        """Return the 3x3 diagonal weight matrix W."""
        return np.diag(self.weight_diagonal())

    def to_dict(self) -> dict:
        return {"xy_sigma_m": self.xy_sigma_m, "z_sigma_m": self.z_sigma_m}

    @classmethod
    def from_dict(cls, data: dict) -> "CameraAccuracy":
        return cls(xy_sigma_m=float(data["xy_sigma_m"]), z_sigma_m=float(data["z_sigma_m"]))


def gnss_position_residual_cost(
    estimated_position: np.ndarray,
    gnss_position: np.ndarray,
    accuracy: CameraAccuracy,
) -> float:
    """Weighted squared-error cost of a camera position vs. its GNSS/PPK observation.

    cost = sum_k  w_k * (estimated_k - gnss_k)^2

    This is the scalar cost contributed by one camera's GNSS observation
    term in the bundle adjustment objective described in the project
    architecture (Sigma reprojection^2 + Sigma GNSS_error^2 / sigma^2).
    It is exposed here, ahead of the bundle-adjustment implementation,
    so the weighting model can be unit-tested independently of the solver.
    """
    estimated_position = np.asarray(estimated_position, dtype=np.float64)
    gnss_position = np.asarray(gnss_position, dtype=np.float64)
    if estimated_position.shape != (3,) or gnss_position.shape != (3,):
        raise ValueError("positions must be 3-vectors [x, y, z]")
    residual = estimated_position - gnss_position
    weights = accuracy.weight_diagonal()
    return float(np.sum(weights * residual**2))
