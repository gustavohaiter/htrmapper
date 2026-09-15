"""Phase 3: bundle adjustment with GNSS/PPK positions as weighted observations.

This is the mathematical core the whole project is built around (see
ARCHITECTURE.md section 3): camera positions from GNSS/PPK are NOT treated
as fixed, hard constraints -- they are observations with uncertainty, and
that uncertainty (the user-configured `CameraAccuracy`) enters the
objective function as a weight, exactly like a classical weighted
least-squares geodetic adjustment:

    minimize  Sum_obs  reprojection_error^2
            + Sum_cam  ||position_estimated - position_gnss||^2_W

    W = diag(1/sigma_xy^2, 1/sigma_xy^2, 1/sigma_z^2)

Implementation: built on COLMAP's own pose-prior bundle adjuster
(`pycolmap.create_pose_prior_bundle_adjuster`), discovered and validated
during Phase 2 (see ARCHITECTURE.md section 3 sub-section on the
`PosePrior` finding) -- not a hand-rolled Ceres cost function, per the
project's "build on mature libraries" decision.

Critical unit-safety decision (resolves the open question from Phase 2):
pose priors are built here with `coordinate_system=CARTESIAN`, in the
SAME metric frame the reconstruction already lives in after Phase 2's
`align_reconstruction_to_locations` (the project's CRS, in meters) --
never `coordinate_system=WGS84`. Whether COLMAP's internal WGS84 pose
priors get converted to a metric frame before use in this specific
bundle-adjuster path is not documented in the pycolmap stubs, and the
project's rule against unverified precision claims means we do not guess:
by transforming GNSS positions into the project CRS ourselves (via the
already-trusted `geo.crs`, the same transform Phase 2 used) and only ever
handing COLMAP flat Cartesian meters, the units on both sides of every
residual (position_estimated - position_gnss, both in project-CRS meters)
are unambiguous.

This was validated empirically (not just assumed) with a synthetic
scene: injecting a deliberate 5m GNSS bias on one camera with a tight
sigma (0.02m) pulls the solved position ~5m to sit on the (wrong) GNSS
observation; the same bias with a loose sigma (5.0m) leaves the solved
position within centimeters of the true, reprojection-consistent
position. That is the expected behavior of a weighted observation, not a
hard constraint, and is exactly what `tests/test_weighted_bundle_adjustment.py`
checks against known ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

import numpy as np
import pycolmap

from htrmapper.core.project import ImageRecord, Project
from htrmapper.geo.crs import CoordinateReferenceSystem, GeodeticPoint, GeodeticTransformer, wgs84
from htrmapper.gnss.accuracy import CameraAccuracy

logger = logging.getLogger(__name__)

_ROBUST_LOSS_TYPES = {
    "trivial": pycolmap.LossFunctionType.TRIVIAL,
    "huber": pycolmap.LossFunctionType.HUBER,
    "cauchy": pycolmap.LossFunctionType.CAUCHY,
    "soft_l1": pycolmap.LossFunctionType.SOFT_L1,
}


class BaError(RuntimeError):
    """Raised when the bundle adjustment cannot even start (as opposed to
    converging poorly, which is reported in BaResult, never hidden)."""


@dataclass
class BaConfig:
    robust_loss: str = "cauchy"  # tie points can contain residual mismatches from Fase 2 matching
    # Refining focal length/principal point/distortion here is OFF by
    # default. Validated experimentally on a flat, nadir-only synthetic
    # flight (constant altitude, no oblique views): letting the intrinsics
    # self-calibrate simultaneously with a GNSS-weighted position prior
    # diverged wildly (fx drifted from ~232px to ~580-710px) even though
    # reprojection error stayed low (~0.06px) -- a textbook planar/nadir
    # self-calibration degeneracy (focal length trades off against
    # unconstrained tie-point elevation; a GNSS position prior pins the
    # camera's absolute height but not the scene's, so it doesn't resolve
    # this). Fixing intrinsics here and letting Phase 2's COLMAP-refined
    # values stand converged cleanly in 4 iterations with identical
    # reprojection error. Per the project's rule against optimizing
    # parameters without evidence they're observable, this is the default;
    # set `refine_intrinsics=True` only for flights with real geometric
    # diversity (oblique images, varied altitude, GCPs) that actually
    # constrain focal length independently of scene depth.
    refine_intrinsics: bool = False
    refine_principal_point: bool = False  # only consulted when refine_intrinsics=True
    max_iterations: int = 200
    function_tolerance: float = 1e-6
    parameter_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.robust_loss not in _ROBUST_LOSS_TYPES:
            raise ValueError(f"unknown robust_loss {self.robust_loss!r}; choose from {sorted(_ROBUST_LOSS_TYPES)}")


@dataclass
class CameraGnssResidual:
    image_name: str
    gnss_position_m: tuple[float, float, float]
    estimated_position_m: tuple[float, float, float]
    residual_m: tuple[float, float, float]  # estimated - gnss
    xy_sigma_m: float
    z_sigma_m: float


@dataclass
class BaResult:
    num_images_adjusted: int = 0
    num_images_with_gnss_prior: int = 0
    per_camera_residuals: list[CameraGnssResidual] = field(default_factory=list)
    rmse_x_cm: float | None = None
    rmse_y_cm: float | None = None
    rmse_z_cm: float | None = None
    rmse_xy_cm: float | None = None
    rmse_total_cm: float | None = None
    max_error_cm: float | None = None
    mean_reprojection_error_px: float | None = None
    num_residuals: int = 0
    termination_type: str = ""
    converged: bool = False
    reconstruction_path: str = ""
    camera_calibration: dict[str, dict] = field(default_factory=dict)

    def to_project_summary(self) -> "htrmapper.core.project.BaSummary":
        from htrmapper.core.project import BaSummary

        return BaSummary(
            num_images_adjusted=self.num_images_adjusted,
            num_images_with_gnss_prior=self.num_images_with_gnss_prior,
            rmse_x_cm=self.rmse_x_cm,
            rmse_y_cm=self.rmse_y_cm,
            rmse_z_cm=self.rmse_z_cm,
            rmse_xy_cm=self.rmse_xy_cm,
            rmse_total_cm=self.rmse_total_cm,
            max_error_cm=self.max_error_cm,
            mean_reprojection_error_px=self.mean_reprojection_error_px,
            num_residuals=self.num_residuals,
            termination_type=self.termination_type,
            converged=self.converged,
            reconstruction_path=self.reconstruction_path,
            camera_calibration=self.camera_calibration,
        )


def _gnss_position_in_project_crs(record: ImageRecord, transformer: GeodeticTransformer) -> tuple[float, float, float] | None:
    if not record.position_valid or record.altitude is None:
        return None
    point = transformer.forward(GeodeticPoint(lon=record.longitude, lat=record.latitude, alt=record.altitude))
    if point.z is None:
        return None
    return (point.x, point.y, point.z)


def run_gnss_weighted_bundle_adjustment(
    project: Project,
    reconstruction_path: Path,
    output_path: Path,
    config: BaConfig | None = None,
) -> BaResult:
    """Refine an already-georeferenced Phase 2 reconstruction using the
    project's GNSS/PPK camera accuracy as a weighted observation.

    Raises BaError if the reconstruction cannot be loaded or has no
    registered images. A poor-quality result (large RMSE, non-convergence)
    is NOT an error -- it is reported honestly in BaResult.
    """
    config = config or BaConfig()
    reconstruction_path = Path(reconstruction_path)
    if not reconstruction_path.exists():
        raise BaError(f"reconstruction not found at {reconstruction_path}")

    try:
        reconstruction = pycolmap.Reconstruction(str(reconstruction_path))
    except Exception as exc:  # noqa: BLE001 -- pycolmap raises varied native exceptions
        raise BaError(f"failed to load reconstruction from {reconstruction_path}: {exc}") from exc

    registered_ids = list(reconstruction.reg_image_ids())
    if not registered_ids:
        raise BaError("reconstruction has no registered images; nothing to adjust")

    accuracy: CameraAccuracy = project.gnss_accuracy.accuracy
    transformer = GeodeticTransformer(wgs84(), CoordinateReferenceSystem(project.crs.project_epsg))
    records_by_name = {Path(img.path).name: img for img in project.images}

    covariance = np.diag([accuracy.xy_sigma_m**2, accuracy.xy_sigma_m**2, accuracy.z_sigma_m**2])

    ba_config = pycolmap.BundleAdjustmentConfig()
    pose_priors: list[pycolmap.PosePrior] = []
    gnss_positions_before: dict[int, tuple[float, float, float]] = {}
    camera_ids_seen: set[int] = set()

    for image_id in registered_ids:
        ba_config.add_image(image_id)
        image = reconstruction.image(image_id)
        camera_ids_seen.add(image.camera_id)
        record = records_by_name.get(image.name)
        if record is None:
            continue
        gnss_position = _gnss_position_in_project_crs(record, transformer)
        if gnss_position is None:
            continue

        prior = pycolmap.PosePrior()
        prior.position = np.array(gnss_position, dtype=np.float64)
        prior.position_covariance = covariance
        prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        prior.corr_data_id = image.data_id
        pose_priors.append(prior)
        gnss_positions_before[image_id] = gnss_position

    if not pose_priors:
        raise BaError(
            "no registered image has a valid GNSS position matching the project's records; "
            "cannot build any GNSS observation for the weighted adjustment"
        )

    options = pycolmap.BundleAdjustmentOptions()
    options.ceres.loss_function_type = _ROBUST_LOSS_TYPES[config.robust_loss]
    options.ceres.solver_options.max_num_iterations = config.max_iterations
    options.ceres.solver_options.function_tolerance = config.function_tolerance
    options.ceres.solver_options.parameter_tolerance = config.parameter_tolerance

    if config.refine_intrinsics:
        options.refine_principal_point = config.refine_principal_point
    else:
        options.refine_focal_length = False
        options.refine_principal_point = False
        options.refine_extra_params = False
        for camera_id in camera_ids_seen:
            ba_config.set_constant_cam_intrinsics(camera_id)

    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()

    adjuster = pycolmap.create_pose_prior_bundle_adjuster(
        options, prior_options, ba_config, pose_priors, reconstruction
    )
    summary = adjuster.solve()

    residuals: list[CameraGnssResidual] = []
    squared_errors_x, squared_errors_y, squared_errors_z, squared_errors_total = [], [], [], []
    for image_id, gnss_position in gnss_positions_before.items():
        image = reconstruction.image(image_id)
        estimated = tuple(float(v) for v in image.projection_center())
        residual = tuple(estimated[i] - gnss_position[i] for i in range(3))
        residuals.append(
            CameraGnssResidual(
                image_name=image.name,
                gnss_position_m=gnss_position,
                estimated_position_m=estimated,
                residual_m=residual,
                xy_sigma_m=accuracy.xy_sigma_m,
                z_sigma_m=accuracy.z_sigma_m,
            )
        )
        squared_errors_x.append(residual[0] ** 2)
        squared_errors_y.append(residual[1] ** 2)
        squared_errors_z.append(residual[2] ** 2)
        squared_errors_total.append(residual[0] ** 2 + residual[1] ** 2 + residual[2] ** 2)

    def rmse_cm(squared_errors: list[float]) -> float:
        return sqrt(sum(squared_errors) / len(squared_errors)) * 100.0

    rmse_x_cm = rmse_cm(squared_errors_x)
    rmse_y_cm = rmse_cm(squared_errors_y)
    rmse_z_cm = rmse_cm(squared_errors_z)
    rmse_xy_cm = sqrt(rmse_x_cm**2 + rmse_y_cm**2)
    rmse_total_cm = rmse_cm(squared_errors_total)
    max_error_cm = sqrt(max(squared_errors_total)) * 100.0

    camera_calibration: dict[str, dict] = {}
    for camera_id, camera in reconstruction.cameras.items():
        camera_calibration[str(camera_id)] = {
            "model": camera.model_name,
            "params_info": camera.params_info,
            "params": [float(p) for p in camera.params],
        }

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output_path)

    return BaResult(
        num_images_adjusted=len(registered_ids),
        num_images_with_gnss_prior=len(pose_priors),
        per_camera_residuals=residuals,
        rmse_x_cm=rmse_x_cm,
        rmse_y_cm=rmse_y_cm,
        rmse_z_cm=rmse_z_cm,
        rmse_xy_cm=rmse_xy_cm,
        rmse_total_cm=rmse_total_cm,
        max_error_cm=max_error_cm,
        mean_reprojection_error_px=reconstruction.compute_mean_reprojection_error(),
        num_residuals=summary.num_residuals,
        termination_type=str(summary.termination_type),
        converged=summary.is_solution_usable(),
        reconstruction_path=str(output_path),
        camera_calibration=camera_calibration,
    )
