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

Camera intrinsics are refined here, with the GNSS positions as reference
(ARCHITECTURE.md section 27). On the first real flight (56 Mavic 3M
photos, RTK FIX at ~5 mm) keeping the lens fixed at the values Phase 2
self-calibrated WITHOUT GNSS left the RTK FIX cameras 19-30 cm (max
115 cm) away from their own GNSS positions. Freeing the principal point
alone brought that to ~2-3 cm (max 7.5 cm). The reason: at 100 m flight
height a principal point that is off by 27 px tilts every ray by
27/3721 rad, i.e. moves the ground ~73 cm. Opposite flight lines see that
shift in opposite directions, so the block bends to compromise and the
orthomosaic lands ~70 cm away from where it belongs. Freeing focal length
too converged to 3723.9 px, within 0.03% of DJI's own factory
`CalibratedFocalLength` (3725.15 px) for that camera.

The degeneracy that originally kept the lens fixed is real, but it shows
up on small synthetic flights (6 images, perfectly flat, constant
altitude), where focal length and principal point drift far away while
reprojection error stays tiny. So each refined parameter is guarded: if
focal length moves more than `max_focal_change_ratio` or the principal
point ends up further than `max_principal_point_offset_ratio` of the
image size from the image center (physically implausible for a real
drone camera), the adjustment is re-run with that parameter fixed and
the reason is reported in `BaResult.intrinsics_notes`. Nothing is
silently dropped.

Per-image GNSS weights: DJI RTK images carry their own position
standard deviations (`RtkStdLon/Lat/Hgt`). The configured
`CameraAccuracy` acts as a floor (it reflects error sources the receiver
does not report, e.g. base coordinates), and an image whose receiver
reports a larger deviation (an RTK FLOAT image, at 17-47 cm on that same
flight) gets that larger deviation. Weighting those like FIX images
pulled the start of the block by meters. GNSS residual RMSE is reported
over RTK FIX images only (or all images when the flight carries no RTK
flag, e.g. PPK), since a FLOAT image's residual measures its bad GNSS,
not the adjustment.
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

# DJI XMP `RtkFlag`: 0 = no position, 16 = single point, 34 = RTK FLOAT,
# 50 = RTK FIX.
RTK_FIX_FLAG = "50"
MIN_FIX_PRIORS = 3

# Guards against the planar-flight self-calibration degeneracy (see module
# docstring). Real values on the first real flight: focal moved 0.07%,
# principal point 0.5% of the image size.
DEFAULT_MAX_FOCAL_CHANGE_RATIO = 0.05
DEFAULT_MAX_PRINCIPAL_POINT_OFFSET_RATIO = 0.02

# COLMAP's pose-prior adjuster first re-aligns the reconstruction to the
# priors with a RANSAC Sim3, and when that fails it DROPS every GNSS prior
# and runs a plain reprojection-only adjustment (logging only a warning).
# With COLMAP's automatic threshold (~2.8 sigma, i.e. ~6 cm at a 2 cm
# sigma) that happened on a 6-image test flight with one bad GNSS fix and
# left the cameras 21 m off. The input is already Sim3-aligned to the same
# GNSS in Phase 2 (Phase 2 models deviated up to ~1.2 m on the real
# flight), so this step only has to reject gross blunders: a few meters.
PRIOR_ALIGNMENT_MAX_ERROR_M = 5.0

# Tie point cleanup between adjustment passes, the automated equivalent of
# Metashape's "gradual selection" + "Optimize Cameras" loop that the USGS
# workflow scripts (ARCHITECTURE.md section 29). Each pass drops
# observations reprojecting worse than max(factor x median point error,
# floor), points triangulated at a very small angle (Metashape's
# "reconstruction uncertainty") and points seen in too few images, then
# re-adjusts. The floor keeps it from shaving a clean block: it removes
# blunders, it does not chase a nicer number.
DEFAULT_TIE_POINT_CLEANUP_MAX_PASSES = 3
DEFAULT_TIE_POINT_ERROR_FACTOR = 3.0
DEFAULT_TIE_POINT_MIN_ERROR_PX = 1.0
DEFAULT_TIE_POINT_MIN_TRI_ANGLE_DEG = 1.5
DEFAULT_TIE_POINT_MIN_TRACK_LENGTH = 3


class BaError(RuntimeError):
    """Raised when the bundle adjustment cannot even start (as opposed to
    converging poorly, which is reported in BaResult, never hidden)."""


@dataclass
class BaConfig:
    robust_loss: str = "cauchy"  # tie points can contain residual mismatches from Fase 2 matching
    refine_focal_length: bool = True
    refine_principal_point: bool = True
    refine_distortion: bool = True
    use_per_image_rtk_sigma: bool = True
    # False: photos without an RTK FIX get no GNSS observation (see
    # run_gnss_weighted_bundle_adjustment), as long as MIN_FIX_PRIORS have one.
    use_non_fix_priors: bool = False
    tie_point_cleanup: bool = True
    tie_point_cleanup_max_passes: int = DEFAULT_TIE_POINT_CLEANUP_MAX_PASSES
    tie_point_error_factor: float = DEFAULT_TIE_POINT_ERROR_FACTOR
    tie_point_min_error_px: float = DEFAULT_TIE_POINT_MIN_ERROR_PX
    tie_point_min_tri_angle_deg: float = DEFAULT_TIE_POINT_MIN_TRI_ANGLE_DEG
    tie_point_min_track_length: int = DEFAULT_TIE_POINT_MIN_TRACK_LENGTH
    max_focal_change_ratio: float = DEFAULT_MAX_FOCAL_CHANGE_RATIO
    max_principal_point_offset_ratio: float = DEFAULT_MAX_PRINCIPAL_POINT_OFFSET_RATIO
    # Validated on the real flight with these tolerances (the looser
    # 1e-6 function tolerance stopped well before the prior term settled).
    max_iterations: int = 500
    function_tolerance: float = 1e-10
    gradient_tolerance: float = 1e-12
    parameter_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.robust_loss not in _ROBUST_LOSS_TYPES:
            raise ValueError(f"unknown robust_loss {self.robust_loss!r}; choose from {sorted(_ROBUST_LOSS_TYPES)}")
        if self.max_focal_change_ratio <= 0 or self.max_principal_point_offset_ratio <= 0:
            raise ValueError("intrinsics guard ratios must be positive")
        if self.tie_point_cleanup_max_passes < 0 or self.tie_point_min_track_length < 2:
            raise ValueError("tie point cleanup needs max_passes >= 0 and min_track_length >= 2")

    @classmethod
    def with_fixed_intrinsics(cls, **kwargs) -> "BaConfig":
        return cls(refine_focal_length=False, refine_principal_point=False, refine_distortion=False, **kwargs)


@dataclass
class CameraGnssResidual:
    image_name: str
    gnss_position_m: tuple[float, float, float]
    estimated_position_m: tuple[float, float, float]
    residual_m: tuple[float, float, float]  # estimated - gnss
    xy_sigma_m: float  # the sigma this image was actually weighted with
    z_sigma_m: float
    rtk_flag: str | None = None


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
    intrinsics_refined: list[str] = field(default_factory=list)
    intrinsics_notes: list[str] = field(default_factory=list)
    images_excluded_from_rmse: list[str] = field(default_factory=list)
    tie_points_before_cleanup: int = 0
    reprojection_error_before_cleanup_px: float | None = None
    tie_point_cleanup_passes: list[dict] = field(default_factory=list)

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
            intrinsics_refined=self.intrinsics_refined,
            intrinsics_notes=self.intrinsics_notes,
            images_excluded_from_rmse=self.images_excluded_from_rmse,
            tie_points_before_cleanup=self.tie_points_before_cleanup,
            reprojection_error_before_cleanup_px=self.reprojection_error_before_cleanup_px,
            tie_point_cleanup_passes=self.tie_point_cleanup_passes,
        )


def _gnss_position_in_project_crs(record: ImageRecord, transformer: GeodeticTransformer) -> tuple[float, float, float] | None:
    if not record.position_valid or record.altitude is None:
        return None
    point = transformer.forward(GeodeticPoint(lon=record.longitude, lat=record.latitude, alt=record.altitude))
    if point.z is None:
        return None
    return (point.x, point.y, point.z)


def _prior_sigmas(record: ImageRecord, accuracy: CameraAccuracy, use_rtk_sigma: bool) -> tuple[float, float]:
    """(xy, z) sigma for one image: the configured accuracy, raised to the
    receiver's own reported deviation when that is larger."""
    xy_sigma, z_sigma = accuracy.xy_sigma_m, accuracy.z_sigma_m
    if use_rtk_sigma:
        horizontal = [s for s in (record.rtk_std_lon_m, record.rtk_std_lat_m) if s is not None]
        if horizontal:
            xy_sigma = max(xy_sigma, *horizontal)
        if record.rtk_std_hgt_m is not None:
            z_sigma = max(z_sigma, record.rtk_std_hgt_m)
    return xy_sigma, z_sigma


def _is_rtk_fix_or_unflagged(record: ImageRecord) -> bool:
    return record.rtk_flag is None or str(record.rtk_flag).strip() == RTK_FIX_FLAG


def _solve(
    reconstruction: "pycolmap.Reconstruction",
    priors_by_name: dict[str, tuple[tuple[float, float, float], float, float]],
    config: BaConfig,
    refine_focal_length: bool,
    refine_principal_point: bool,
    refine_distortion: bool,
):
    """One pose-prior bundle adjustment run, in place on `reconstruction`.
    Returns (reconstruction, summary, gnss_by_image_id)."""

    ba_config = pycolmap.BundleAdjustmentConfig()
    pose_priors: list[pycolmap.PosePrior] = []
    gnss_by_image_id: dict[int, tuple[float, float, float]] = {}
    for image_id in reconstruction.reg_image_ids():
        ba_config.add_image(image_id)
        image = reconstruction.image(image_id)
        if image.name not in priors_by_name:
            continue
        position, xy_sigma, z_sigma = priors_by_name[image.name]
        prior = pycolmap.PosePrior()
        prior.position = np.array(position, dtype=np.float64)
        prior.position_covariance = np.diag([xy_sigma**2, xy_sigma**2, z_sigma**2])
        prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        prior.corr_data_id = image.data_id
        pose_priors.append(prior)
        gnss_by_image_id[image_id] = position

    if not pose_priors:
        raise BaError(
            "no registered image has a valid GNSS position matching the project's records; "
            "cannot build any GNSS observation for the weighted adjustment"
        )

    options = pycolmap.BundleAdjustmentOptions()
    options.ceres.loss_function_type = _ROBUST_LOSS_TYPES[config.robust_loss]
    options.ceres.solver_options.max_num_iterations = config.max_iterations
    options.ceres.solver_options.function_tolerance = config.function_tolerance
    options.ceres.solver_options.gradient_tolerance = config.gradient_tolerance
    options.ceres.solver_options.parameter_tolerance = config.parameter_tolerance
    options.refine_focal_length = refine_focal_length
    options.refine_principal_point = refine_principal_point
    options.refine_extra_params = refine_distortion
    if not (refine_focal_length or refine_principal_point or refine_distortion):
        for camera_id in reconstruction.cameras:
            ba_config.set_constant_cam_intrinsics(camera_id)

    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
    alignment_ransac = prior_options.alignment_ransac
    alignment_ransac.max_error = PRIOR_ALIGNMENT_MAX_ERROR_M
    prior_options.alignment_ransac = alignment_ransac

    adjuster = pycolmap.create_pose_prior_bundle_adjuster(
        options, prior_options, ba_config, pose_priors, reconstruction
    )
    summary = adjuster.solve()

    # Each prior adds 3 residuals and each observation 2; anything else
    # means COLMAP silently dropped the priors (see PRIOR_ALIGNMENT_MAX_ERROR_M).
    prior_residuals = summary.num_residuals - 2 * reconstruction.compute_num_observations()
    if prior_residuals != 3 * len(pose_priors):
        raise BaError(
            f"COLMAP did not use the GNSS observations ({prior_residuals // 3} of {len(pose_priors)} in the "
            "adjustment): its alignment of the reconstruction to the GNSS positions failed. The Fase 2 "
            "reconstruction is probably not consistent with the GNSS positions (check the alignment and the "
            "image positions)."
        )
    return reconstruction, summary, gnss_by_image_id


def _mean_reprojection_error_px(reconstruction: "pycolmap.Reconstruction") -> float:
    """Mean reprojection error of the reconstruction as it is NOW.

    `compute_mean_reprojection_error()` averages the per-point errors
    cached in the reconstruction, and the bundle adjuster moves cameras
    and points without refreshing that cache: the adjusted result kept
    reporting the Phase 2 value (0.499 px on every variant of the real
    flight). Refresh first."""
    reconstruction.update_point_3d_errors()
    return reconstruction.compute_mean_reprojection_error()


def _clean_tie_points(reconstruction: "pycolmap.Reconstruction", config: BaConfig) -> tuple[int, float]:
    """One cleanup pass (see DEFAULT_TIE_POINT_* above), in place.
    Returns (observations removed, reprojection threshold used in px)."""
    reconstruction.update_point_3d_errors()
    errors = np.array([point.error for point in reconstruction.points3D.values()])
    if errors.size == 0:
        return 0, 0.0
    threshold_px = max(config.tie_point_min_error_px, config.tie_point_error_factor * float(np.median(errors)))
    manager = pycolmap.ObservationManager(reconstruction)
    removed = manager.filter_all_points3D(threshold_px, config.tie_point_min_tri_angle_deg)
    removed += manager.filter_points3D_with_short_tracks(config.tie_point_min_track_length)
    return int(removed), threshold_px


def _implausible_intrinsics(
    initial_cameras: dict[int, "pycolmap.Camera"], reconstruction: "pycolmap.Reconstruction", config: BaConfig
) -> tuple[bool, bool, list[str]]:
    """(focal_implausible, principal_point_implausible, notes)."""
    focal_bad = principal_point_bad = False
    notes: list[str] = []
    for camera_id, camera in reconstruction.cameras.items():
        initial = initial_cameras[camera_id]
        focal_change = abs(camera.mean_focal_length() / initial.mean_focal_length() - 1.0)
        if focal_change > config.max_focal_change_ratio:
            focal_bad = True
            notes.append(
                f"câmera {camera_id}: a focal variou {focal_change:.1%} "
                f"({initial.mean_focal_length():.1f} -> {camera.mean_focal_length():.1f} px), acima do limite de "
                f"{config.max_focal_change_ratio:.0%}; reajustada com a focal fixa"
            )
        offset_x = abs(camera.principal_point_x - camera.width / 2.0) / camera.width
        offset_y = abs(camera.principal_point_y - camera.height / 2.0) / camera.height
        if max(offset_x, offset_y) > config.max_principal_point_offset_ratio:
            principal_point_bad = True
            notes.append(
                f"câmera {camera_id}: o centro óptico ({camera.principal_point_x:.1f}, "
                f"{camera.principal_point_y:.1f}) ficou a {max(offset_x, offset_y):.1%} do tamanho da imagem a partir "
                f"do centro, acima do limite de {config.max_principal_point_offset_ratio:.0%}; reajustada com o "
                "centro óptico fixo"
            )
    return focal_bad, principal_point_bad, notes


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
        initial_reconstruction = pycolmap.Reconstruction(str(reconstruction_path))
    except Exception as exc:  # noqa: BLE001 -- pycolmap raises varied native exceptions
        raise BaError(f"failed to load reconstruction from {reconstruction_path}: {exc}") from exc

    registered_ids = list(initial_reconstruction.reg_image_ids())
    if not registered_ids:
        raise BaError("reconstruction has no registered images; nothing to adjust")
    initial_cameras = dict(initial_reconstruction.cameras.items())

    accuracy: CameraAccuracy = project.gnss_accuracy.accuracy
    transformer = GeodeticTransformer(wgs84(), CoordinateReferenceSystem(project.crs.project_epsg))
    records_by_name = {Path(img.path).name: img for img in project.images}

    priors_by_name: dict[str, tuple[tuple[float, float, float], float, float]] = {}
    for image_id in registered_ids:
        name = initial_reconstruction.image(image_id).name
        record = records_by_name.get(name)
        if record is None:
            continue
        gnss_position = _gnss_position_in_project_crs(record, transformer)
        if gnss_position is None:
            continue
        priors_by_name[name] = (gnss_position, *_prior_sigmas(record, accuracy, config.use_per_image_rtk_sigma))

    # Photos without an RTK FIX carry no GNSS observation: their reported
    # deviation is not trustworthy (the first two photos of the real flight,
    # taken before the fix, reported 17 cm and were 4.2 m off), and with that
    # weight a meters-off position pulls the block. Tie points place them.
    # Kept only if fewer than MIN_FIX_PRIORS photos have a FIX at all.
    non_fix_priors: dict[str, tuple[tuple[float, float, float], float, float]] = {}
    if not config.use_non_fix_priors:
        fix_names = [n for n in priors_by_name if _is_rtk_fix_or_unflagged(records_by_name[n])]
        if len(fix_names) >= MIN_FIX_PRIORS:
            non_fix_priors = {n: v for n, v in priors_by_name.items() if n not in set(fix_names)}
            priors_by_name = {n: priors_by_name[n] for n in fix_names}
            for name in sorted(non_fix_priors):
                logger.info("%s: no RTK FIX (flag %s), not used as a GNSS observation", name, records_by_name[name].rtk_flag)

    refine_focal = config.refine_focal_length
    refine_principal_point = config.refine_principal_point
    intrinsics_notes: list[str] = []
    def solve_fresh():
        # Always from the Phase 2 state, so a rejected run leaves no trace.
        return _solve(
            pycolmap.Reconstruction(str(reconstruction_path)),
            priors_by_name,
            config,
            refine_focal,
            refine_principal_point,
            config.refine_distortion,
        )

    reconstruction, summary, gnss_by_image_id = solve_fresh()
    # Fixing one parameter can push the other out of bounds, hence the loop;
    # at most two re-runs, since each one fixes at least one more parameter.
    while refine_focal or refine_principal_point:
        focal_bad, principal_point_bad, notes = _implausible_intrinsics(initial_cameras, reconstruction, config)
        focal_bad = focal_bad and refine_focal
        principal_point_bad = principal_point_bad and refine_principal_point
        if not (focal_bad or principal_point_bad):
            break
        intrinsics_notes.extend(notes)
        for note in notes:
            logger.warning(note)
        refine_focal = refine_focal and not focal_bad
        refine_principal_point = refine_principal_point and not principal_point_bad
        reconstruction, summary, gnss_by_image_id = solve_fresh()

    tie_points_before = reconstruction.num_points3D()
    reprojection_error_before = _mean_reprojection_error_px(reconstruction)
    cleanup_passes: list[dict] = []
    if config.tie_point_cleanup:
        for _ in range(config.tie_point_cleanup_max_passes):
            removed, threshold_px = _clean_tie_points(reconstruction, config)
            if removed == 0:
                break
            reconstruction, summary, gnss_by_image_id = _solve(
                reconstruction, priors_by_name, config, refine_focal, refine_principal_point, config.refine_distortion
            )
            cleanup_passes.append(
                {
                    "max_error_px": threshold_px,
                    "observations_removed": removed,
                    "tie_points_left": reconstruction.num_points3D(),
                    "mean_reprojection_error_px": _mean_reprojection_error_px(reconstruction),
                }
            )
        if cleanup_passes and (refine_focal or refine_principal_point):
            focal_bad, principal_point_bad, notes = _implausible_intrinsics(initial_cameras, reconstruction, config)
            if (focal_bad and refine_focal) or (principal_point_bad and refine_principal_point):
                # Thinner tie points made the lens drift: keep the result
                # without cleanup rather than an implausible calibration.
                intrinsics_notes.append(
                    "limpeza de tie points descartada, ela deixou a calibração sair dos limites: " + "; ".join(notes)
                )
                cleanup_passes = []
                reconstruction, summary, gnss_by_image_id = solve_fresh()

    intrinsics_refined = [
        name
        for name, enabled in (
            ("focal_length", refine_focal),
            ("principal_point", refine_principal_point),
            ("distortion", config.refine_distortion),
        )
        if enabled
    ]

    residuals: list[CameraGnssResidual] = []
    for image_id, gnss_position in gnss_by_image_id.items():
        image = reconstruction.image(image_id)
        record = records_by_name[image.name]
        _, xy_sigma, z_sigma = priors_by_name[image.name]
        estimated = tuple(float(v) for v in image.projection_center())
        residuals.append(
            CameraGnssResidual(
                image_name=image.name,
                gnss_position_m=gnss_position,
                estimated_position_m=estimated,
                residual_m=tuple(estimated[i] - gnss_position[i] for i in range(3)),
                xy_sigma_m=xy_sigma,
                z_sigma_m=z_sigma,
                rtk_flag=record.rtk_flag,
            )
        )

    # Photos left out of the adjustment's GNSS observations: their difference
    # to the GNSS is still reported, never part of the RMSE.
    ids_by_name = {reconstruction.image(i).name: i for i in reconstruction.reg_image_ids()}
    for name, (gnss_position, xy_sigma, z_sigma) in sorted(non_fix_priors.items()):
        image = reconstruction.image(ids_by_name[name])
        estimated = tuple(float(v) for v in image.projection_center())
        residuals.append(
            CameraGnssResidual(
                image_name=name,
                gnss_position_m=gnss_position,
                estimated_position_m=estimated,
                residual_m=tuple(estimated[i] - gnss_position[i] for i in range(3)),
                xy_sigma_m=xy_sigma,
                z_sigma_m=z_sigma,
                rtk_flag=records_by_name[name].rtk_flag,
            )
        )

    rmse_set = [r for r in residuals if _is_rtk_fix_or_unflagged(records_by_name[r.image_name])] or residuals
    rmse_names = {r.image_name for r in rmse_set}
    images_excluded_from_rmse = sorted(r.image_name for r in residuals if r.image_name not in rmse_names)

    def rmse_cm(squared_errors: list[float]) -> float:
        return sqrt(sum(squared_errors) / len(squared_errors)) * 100.0

    rmse_x_cm = rmse_cm([r.residual_m[0] ** 2 for r in rmse_set])
    rmse_y_cm = rmse_cm([r.residual_m[1] ** 2 for r in rmse_set])
    rmse_z_cm = rmse_cm([r.residual_m[2] ** 2 for r in rmse_set])
    rmse_xy_cm = sqrt(rmse_x_cm**2 + rmse_y_cm**2)
    squared_totals = [sum(v**2 for v in r.residual_m) for r in rmse_set]
    rmse_total_cm = rmse_cm(squared_totals)
    max_error_cm = sqrt(max(squared_totals)) * 100.0

    camera_calibration: dict[str, dict] = {}
    for camera_id, camera in reconstruction.cameras.items():
        camera_calibration[str(camera_id)] = {
            "model": camera.model_name,
            "params_info": camera.params_info,
            "params": [float(p) for p in camera.params],
            "initial_params": [float(p) for p in initial_cameras[camera_id].params],
        }

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output_path)

    return BaResult(
        num_images_adjusted=len(registered_ids),
        num_images_with_gnss_prior=len(gnss_by_image_id),
        per_camera_residuals=residuals,
        rmse_x_cm=rmse_x_cm,
        rmse_y_cm=rmse_y_cm,
        rmse_z_cm=rmse_z_cm,
        rmse_xy_cm=rmse_xy_cm,
        rmse_total_cm=rmse_total_cm,
        max_error_cm=max_error_cm,
        mean_reprojection_error_px=_mean_reprojection_error_px(reconstruction),
        num_residuals=summary.num_residuals,
        termination_type=str(summary.termination_type),
        converged=summary.is_solution_usable(),
        reconstruction_path=str(output_path),
        camera_calibration=camera_calibration,
        intrinsics_refined=intrinsics_refined,
        intrinsics_notes=intrinsics_notes,
        images_excluded_from_rmse=images_excluded_from_rmse,
        tie_points_before_cleanup=tie_points_before,
        reprojection_error_before_cleanup_px=reprojection_error_before,
        tie_point_cleanup_passes=cleanup_passes,
    )
