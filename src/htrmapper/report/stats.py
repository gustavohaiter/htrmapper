"""Quality statistics for the processing report, measured from the real
outputs on disk (COLMAP database and reconstructions, DEM, orthomosaic) --
the same quantities Pix4D's quality report and Metashape's processing
report show. Each function returns None when its input does not exist,
never a made-up value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from htrmapper.core.project import Project


@dataclass
class ImageStats:
    name: str
    keypoints: int | None = None
    tie_points: int = 0  # observations of a 3D point (matched and triangulated keypoints)
    rtk_flag: str | None = None
    d_east_cm: float | None = None
    d_north_cm: float | None = None
    d_up_cm: float | None = None
    in_rmse: bool = True  # False: no RTK FIX, weighted by its own std-dev and left out of the RMSE


@dataclass
class ReportStats:
    images: list[ImageStats] = field(default_factory=list)
    median_keypoints: float | None = None
    median_tie_points: float | None = None
    track_length_counts: dict[int, int] = field(default_factory=dict)  # photos per 3D point -> number of points
    num_points3d: int = 0
    num_observations: int = 0
    mean_track_length: float | None = None
    mean_flying_height_m: float | None = None
    gsd_cm: float | None = None
    covered_area_ha: float | None = None
    dense_density_per_m2: float | None = None
    # Mean image residual per cell of a grid over the image (Metashape's
    # "image residuals" plot): (grid_x, grid_y, mean_dx, mean_dy, count)
    residual_grid: tuple | None = None
    residual_image_size: tuple[int, int] | None = None
    # Number of cameras seeing each cell over the DEM extent (Pix4D overlap map).
    overlap: np.ndarray | None = None
    overlap_extent: tuple[float, float, float, float] | None = None


def _keypoints_per_image(database_path: str) -> dict[str, int]:
    import pycolmap

    if not database_path or not Path(database_path).is_file():
        return {}
    db = pycolmap.Database.open(database_path)
    try:
        return {img.name: int(db.num_keypoints_for_image(img.image_id)) for img in db.read_all_images()}
    finally:
        db.close()


def _camera_position_errors(project: Project, reconstruction) -> dict[str, tuple[float, float, float, bool]]:
    """Adjusted camera center minus its GNSS position, in cm (E, N, up),
    computed exactly as the bundle adjustment builds its priors."""
    from htrmapper.ba.weighted_bundle_adjustment import _gnss_position_in_project_crs, _is_rtk_fix_or_unflagged
    from htrmapper.geo.crs import CoordinateReferenceSystem, GeodeticTransformer, wgs84

    transformer = GeodeticTransformer(wgs84(), CoordinateReferenceSystem(project.crs.project_epsg))
    records = {r.file_name: r for r in project.images}
    errors = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        record = records.get(image.name)
        if record is None:
            continue
        gnss = _gnss_position_in_project_crs(record, transformer)
        if gnss is None:
            continue
        delta = (np.asarray(image.projection_center()) - np.asarray(gnss)) * 100.0
        errors[image.name] = (float(delta[0]), float(delta[1]), float(delta[2]), _is_rtk_fix_or_unflagged(record))
    return errors


def _residual_grid(reconstruction, cells: int = 12) -> tuple[tuple, tuple[int, int]] | None:
    """Mean reprojection residual (projected minus observed) per image cell,
    over every observation of every image of the (single) camera."""
    cameras = list(reconstruction.cameras.values())
    if not cameras:
        return None
    width, height = cameras[0].width, cameras[0].height
    sum_dx = np.zeros((cells, cells))
    sum_dy = np.zeros((cells, cells))
    count = np.zeros((cells, cells))
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        if image.camera_id != cameras[0].camera_id:
            continue
        # Only the keypoints that became a tie point (a fraction of all).
        observations = image.get_observation_points2D()
        observed = [p.xy for p in observations]
        xyz = [reconstruction.points3D[p.point3D_id].xyz for p in observations]
        if not observed:
            continue
        observed = np.asarray(observed)
        projected = image.camera.img_from_cam(image.cam_from_world() * np.asarray(xyz))
        ok = np.all(np.isfinite(projected), axis=1)
        delta = projected[ok] - observed[ok]
        gx = np.clip((observed[ok, 0] / width * cells).astype(int), 0, cells - 1)
        gy = np.clip((observed[ok, 1] / height * cells).astype(int), 0, cells - 1)
        np.add.at(sum_dx, (gy, gx), delta[:, 0])
        np.add.at(sum_dy, (gy, gx), delta[:, 1])
        np.add.at(count, (gy, gx), 1)
    if count.sum() == 0:
        return None
    with np.errstate(invalid="ignore"):
        mean_dx = sum_dx / count
        mean_dy = sum_dy / count
    centers_x = (np.arange(cells) + 0.5) * width / cells
    centers_y = (np.arange(cells) + 0.5) * height / cells
    return (centers_x, centers_y, mean_dx, mean_dy, count), (width, height)


def _overlap_and_height(project: Project, max_cells: int = 300):
    """Cameras seeing each DEM cell (sampled to at most max_cells per side),
    and the mean camera height above the DEM."""
    import pycolmap
    import rasterio

    if project.mvs is None or project.dem is None:
        return None, None, None
    recon_path = Path(project.mvs.undistorted_reconstruction_path)
    if not recon_path.exists() or not Path(project.dem.raster_path).is_file():
        return None, None, None
    reconstruction = pycolmap.Reconstruction(str(recon_path))
    with rasterio.open(project.dem.raster_path) as src:
        step = max(1, int(np.ceil(max(src.width, src.height) / max_cells)))
        rows = np.arange(step // 2, src.height, step)
        cols = np.arange(step // 2, src.width, step)
        elevation = src.read(1, masked=True).astype(float).filled(np.nan)
        transform = src.transform
        width, height = src.width, src.height
    grid_z = elevation[np.ix_(rows, cols)]
    cc, rr = np.meshgrid(cols + 0.5, rows + 0.5)
    xs = transform.c + cc * transform.a
    ys = transform.f + rr * transform.e
    valid = np.isfinite(grid_z)
    points = np.column_stack([xs[valid], ys[valid], grid_z[valid]])
    counts = np.zeros(points.shape[0], dtype=np.int32)
    heights = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        camera = image.camera
        cam_points = image.cam_from_world() * points
        uv = camera.img_from_cam(cam_points)
        inside = (
            (cam_points[:, 2] > 0)
            & np.isfinite(uv[:, 0])
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < camera.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < camera.height)
        )
        counts += inside
        center = image.projection_center()
        row, col = rasterio.transform.rowcol(transform, center[0], center[1])
        if 0 <= row < elevation.shape[0] and 0 <= col < elevation.shape[1] and np.isfinite(elevation[row, col]):
            heights.append(center[2] - elevation[row, col])
    overlap = np.full(grid_z.shape, np.nan)
    overlap[valid] = counts
    extent = (transform.c, transform.c + transform.a * width, transform.f + transform.e * height, transform.f)
    return overlap, extent, (float(np.mean(heights)) if heights else None)


def compute_report_stats(project: Project, ortho_resolution_scale: float = 1.0) -> ReportStats:
    import pycolmap

    stats = ReportStats()
    keypoints = _keypoints_per_image(project.sfm.database_path) if project.sfm else {}
    recon_path = (
        project.ba.reconstruction_path if project.ba else (project.sfm.reconstruction_path if project.sfm else "")
    )
    reconstruction = pycolmap.Reconstruction(recon_path) if recon_path and Path(recon_path).exists() else None

    per_image = {r.file_name: ImageStats(name=r.file_name, rtk_flag=r.rtk_flag) for r in project.images}
    for name, value in keypoints.items():
        if name in per_image:
            per_image[name].keypoints = value

    if reconstruction is not None:
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.image(image_id)
            if image.name in per_image:
                per_image[image.name].tie_points = int(image.num_points3D)
        lengths = np.array([p.track.length() for p in reconstruction.points3D.values()], dtype=int)
        if lengths.size:
            values, counts = np.unique(lengths, return_counts=True)
            stats.track_length_counts = {int(v): int(c) for v, c in zip(values, counts)}
            stats.num_points3d = int(lengths.size)
            stats.num_observations = int(lengths.sum())
            stats.mean_track_length = float(lengths.mean())
        for name, (de, dn, du, in_rmse) in _camera_position_errors(project, reconstruction).items():
            item = per_image[name]
            item.d_east_cm, item.d_north_cm, item.d_up_cm, item.in_rmse = de, dn, du, in_rmse
        residuals = _residual_grid(reconstruction)
        if residuals is not None:
            stats.residual_grid, stats.residual_image_size = residuals

    stats.images = sorted(per_image.values(), key=lambda s: s.name)
    kp = [s.keypoints for s in stats.images if s.keypoints is not None]
    stats.median_keypoints = float(np.median(kp)) if kp else None
    tp = [s.tie_points for s in stats.images if s.tie_points]
    stats.median_tie_points = float(np.median(tp)) if tp else None

    stats.overlap, stats.overlap_extent, stats.mean_flying_height_m = _overlap_and_height(project)

    if project.ortho is not None:
        stats.covered_area_ha = project.ortho.num_valid_pixels * project.ortho.resolution_m**2 / 1e4
        if project.ortho.resolution_source.startswith("automatic (camera GSD)"):
            stats.gsd_cm = project.ortho.resolution_m / ortho_resolution_scale * 100.0
    if project.dem is not None and project.mvs is not None:
        dem = project.dem
        covered_m2 = dem.width_px * dem.height_px * (1.0 - dem.nodata_fraction) * dem.resolution_m**2
        if covered_m2 > 0:
            stats.dense_density_per_m2 = project.mvs.num_points / covered_m2
    return stats
