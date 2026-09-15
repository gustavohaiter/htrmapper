"""Phase 2: feature extraction, matching, and initial Structure-from-Motion.

Built on COLMAP (via `pycolmap`), per the architecture decision in
ARCHITECTURE.md: no custom feature detector, matcher, or SfM solver is
reimplemented here. What this module does that is specific to HTRMapper:

- Derives initial camera intrinsics from EXIF (crop-factor trick, see
  `sfm.camera_model`) instead of trusting COLMAP's built-in EXIF/sensor
  guess for every camera model, and groups images by camera model so each
  physical camera gets its own shared intrinsics (matching the project's
  multi-camera-model requirement, e.g. RGB + future multispectral).
- Uses the user's GNSS/PPK positions to restrict candidate matching pairs
  via COLMAP's spatial matching (`match_spatial`), falling back to
  exhaustive matching when there are too few positioned images for spatial
  matching to be meaningful -- GNSS narrows the search, it is never a hard
  requirement (per the project brief).
- Georeferences the resulting (otherwise arbitrary-scale) SfM
  reconstruction by a similarity (Sim3) alignment to the same GNSS
  positions, converted through `geo.crs` into the project's CRS -- never a
  naive linear rescale.
- This is the *initial* geometry estimate. The GNSS-weighted bundle
  adjustment (Phase 3) later re-optimizes camera poses using the
  configured `CameraAccuracy` as a proper weighted observation, on top of
  what COLMAP produces here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

from htrmapper.core.project import ImageRecord, Project
from htrmapper.geo.crs import CoordinateReferenceSystem, GeodeticPoint, GeodeticTransformer, wgs84
from htrmapper.sfm.camera_model import InsufficientExifError, derive_initial_intrinsics

logger = logging.getLogger(__name__)

DEFAULT_KEY_POINT_LIMIT = 40_000
MIN_IMAGES_FOR_SPATIAL_MATCHING = 3
MIN_IMAGES_FOR_GEOREFERENCING = 3


@dataclass
class SfmConfig:
    key_point_limit: int = DEFAULT_KEY_POINT_LIMIT
    spatial_max_neighbors: int = 50
    spatial_max_distance_m: float = 150.0
    use_gpu: bool | None = None  # None = auto-detect (pycolmap.has_cuda)

    def device(self) -> "pycolmap.Device":
        if self.use_gpu is None:
            return pycolmap.Device.auto
        return pycolmap.Device.cuda if self.use_gpu else pycolmap.Device.cpu

    def extraction_options(self) -> "pycolmap.FeatureExtractionOptions":
        options = pycolmap.FeatureExtractionOptions()
        options.sift.max_num_features = self.key_point_limit
        return options


@dataclass
class CameraGroupResult:
    camera_model: str
    num_images: int
    intrinsics_source: str  # "exif_crop_factor" or "colmap_auto_exif"
    fx_px: float | None = None
    fy_px: float | None = None


@dataclass
class SfmResult:
    num_images_input: int = 0
    num_registered: int = 0
    unregistered_image_names: list[str] = field(default_factory=list)
    num_points3d: int = 0
    num_observations: int = 0
    mean_reprojection_error_px: float | None = None
    matching_strategy: str = ""
    camera_groups: list[CameraGroupResult] = field(default_factory=list)
    georeferenced: bool = False
    georeferencing_note: str = ""
    database_path: str = ""
    reconstruction_path: str = ""
    num_reconstruction_components: int = 1

    @property
    def success(self) -> bool:
        return self.num_registered > 0

    def to_project_summary(self) -> "htrmapper.core.project.SfmSummary":
        from htrmapper.core.project import SfmSummary

        return SfmSummary(
            num_images_input=self.num_images_input,
            num_registered=self.num_registered,
            unregistered_image_names=self.unregistered_image_names,
            num_points3d=self.num_points3d,
            num_observations=self.num_observations,
            mean_reprojection_error_px=self.mean_reprojection_error_px,
            matching_strategy=self.matching_strategy,
            georeferenced=self.georeferenced,
            georeferencing_note=self.georeferencing_note,
            database_path=self.database_path,
            reconstruction_path=self.reconstruction_path,
        )


class SfmError(RuntimeError):
    """Raised when the pipeline cannot proceed at all (as opposed to a
    partial result, e.g. some images unregistered, which is reported in
    SfmResult rather than raised)."""


def _validate_common_root(images: list[ImageRecord]) -> Path:
    parents = {Path(img.path).resolve().parent for img in images}
    if len(parents) != 1:
        raise SfmError(
            f"images span {len(parents)} different folders ({sorted(str(p) for p in parents)}); "
            "COLMAP requires a single image root directory for a reconstruction"
        )
    return next(iter(parents))


def _group_by_camera_model(images: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    groups: dict[str, list[ImageRecord]] = {}
    for img in images:
        key = img.camera_model or "unknown"
        groups.setdefault(key, []).append(img)
    return groups


def _extract_features_for_group(
    database_path: Path,
    image_root: Path,
    group_name: str,
    group_images: list[ImageRecord],
    config: SfmConfig,
) -> CameraGroupResult:
    image_names = [Path(img.path).name for img in group_images]

    try:
        intrinsics = derive_initial_intrinsics(group_images)
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "OPENCV"
        reader_options.camera_params = intrinsics.as_opencv_params_string()
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_root,
            image_names=image_names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=config.extraction_options(),
            device=config.device(),
        )
        return CameraGroupResult(
            camera_model=group_name,
            num_images=len(group_images),
            intrinsics_source="exif_crop_factor",
            fx_px=intrinsics.fx_px,
            fy_px=intrinsics.fy_px,
        )
    except InsufficientExifError as exc:
        logger.warning(
            "camera group %r: %s -- falling back to COLMAP's own EXIF-based camera guess",
            group_name,
            exc,
        )
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_root,
            image_names=image_names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            extraction_options=config.extraction_options(),
            device=config.device(),
        )
        return CameraGroupResult(
            camera_model=group_name, num_images=len(group_images), intrinsics_source="colmap_auto_exif"
        )


def _run_matching(database_path: Path, images: list[ImageRecord], config: SfmConfig) -> str:
    num_positioned = sum(1 for img in images if img.position_valid)
    if num_positioned >= MIN_IMAGES_FOR_SPATIAL_MATCHING:
        pairing_options = pycolmap.SpatialPairingOptions()
        pairing_options.max_num_neighbors = config.spatial_max_neighbors
        pairing_options.max_distance = config.spatial_max_distance_m
        pycolmap.match_spatial(
            database_path=database_path, pairing_options=pairing_options, device=config.device()
        )
        return "spatial (restricted by GNSS position)"

    pycolmap.match_exhaustive(database_path=database_path, device=config.device())
    return (
        f"exhaustive (only {num_positioned} image(s) with valid GNSS position, "
        f"below the {MIN_IMAGES_FOR_SPATIAL_MATCHING} needed for spatial matching)"
    )


def _georeference(
    reconstruction: "pycolmap.Reconstruction",
    images_by_name: dict[str, ImageRecord],
    project_crs_epsg: int,
) -> tuple[bool, str]:
    transformer = GeodeticTransformer(wgs84(), CoordinateReferenceSystem(project_crs_epsg))

    tgt_names: list[str] = []
    tgt_locations: list[list[float]] = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        record = images_by_name.get(image.name)
        if record is None or not record.position_valid:
            continue
        point = transformer.forward(GeodeticPoint(lon=record.longitude, lat=record.latitude, alt=record.altitude))
        if point.z is None:
            continue
        tgt_names.append(image.name)
        tgt_locations.append([point.x, point.y, point.z])

    if len(tgt_names) < MIN_IMAGES_FOR_GEOREFERENCING:
        return False, (
            f"only {len(tgt_names)} registered image(s) have a valid GNSS position "
            f"(need >= {MIN_IMAGES_FOR_GEOREFERENCING} for a similarity alignment); "
            "reconstruction kept in its arbitrary-scale SfM frame"
        )

    sim3d = pycolmap.align_reconstruction_to_locations(
        src=reconstruction,
        tgt_image_names=tgt_names,
        tgt_locations=np.array(tgt_locations, dtype=np.float64),
        min_common_images=MIN_IMAGES_FOR_GEOREFERENCING,
        ransac_options=pycolmap.RANSACOptions(),
    )
    if sim3d is None:
        return False, (
            "similarity alignment to GNSS positions failed (RANSAC found no consistent transform); "
            "reconstruction kept in its arbitrary-scale SfM frame"
        )

    reconstruction.transform(sim3d)
    return True, f"aligned to EPSG:{project_crs_epsg} using {len(tgt_names)} GNSS-positioned camera(s)"


def run_structure_from_motion(project: Project, workdir: Path, config: SfmConfig | None = None) -> SfmResult:
    """Run feature extraction, matching, and incremental SfM for a project.

    Raises SfmError if the pipeline cannot even start (e.g. no images, or
    images spread across multiple folders). A reconstruction that only
    partially aligns is NOT an error -- it is reported in SfmResult with
    the unregistered image names, per the project's "never hide failures,
    say exactly why" rule.
    """
    config = config or SfmConfig()
    images = project.images
    if not images:
        raise SfmError("project has no imported images; run import before alignment")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    database_path = workdir / "database.db"
    if database_path.exists():
        database_path.unlink()
    sparse_path = workdir / "sparse"
    sparse_path.mkdir(exist_ok=True)

    image_root = _validate_common_root(images)
    groups = _group_by_camera_model(images)

    result = SfmResult(num_images_input=len(images), database_path=str(database_path))

    for group_name, group_images in groups.items():
        group_result = _extract_features_for_group(database_path, image_root, group_name, group_images, config)
        result.camera_groups.append(group_result)

    result.matching_strategy = _run_matching(database_path, images, config)

    pipeline_options = pycolmap.IncrementalPipelineOptions()
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_root,
        output_path=sparse_path,
        options=pipeline_options,
    )

    if not reconstructions:
        result.unregistered_image_names = [Path(img.path).name for img in images]
        return result

    primary_id = max(reconstructions, key=lambda i: reconstructions[i].num_reg_images())
    primary = reconstructions[primary_id]
    result.num_reconstruction_components = len(reconstructions)

    registered_names = {primary.image(i).name for i in primary.reg_image_ids()}
    result.unregistered_image_names = sorted(
        Path(img.path).name for img in images if Path(img.path).name not in registered_names
    )
    result.num_registered = primary.num_reg_images()
    result.num_points3d = primary.num_points3D()
    result.num_observations = primary.compute_num_observations()
    result.mean_reprojection_error_px = primary.compute_mean_reprojection_error()

    images_by_name = {Path(img.path).name: img for img in images}
    result.georeferenced, result.georeferencing_note = _georeference(primary, images_by_name, project.crs.project_epsg)

    primary.write(sparse_path)
    result.reconstruction_path = str(sparse_path)

    return result
