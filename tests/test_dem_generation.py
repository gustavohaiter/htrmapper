from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pytest
import rasterio

from htrmapper.dem.generation import DemConfig, DemError, run_dem_generation


def _true_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A smooth, known terrain function: a gentle slope plus a broad,
    low-amplitude undulation -- large wavelength relative to point
    spacing, so linear interpolation between scattered samples should
    track it closely."""
    return 700.0 + 0.02 * x + 0.01 * y + 0.5 * np.sin(x / 40.0) * np.cos(y / 40.0)


def _write_synthetic_point_cloud(path: Path, seed: int = 0, num_points: int = 6000) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 120.0, num_points)
    y = rng.uniform(0.0, 100.0, num_points)
    z = _true_surface(x, y) + rng.normal(0.0, 0.01, num_points)  # 1cm noise

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([x.min(), y.min(), z.min()])
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)


def test_dem_matches_known_surface_within_tolerance(tmp_path: Path):
    las_path = tmp_path / "cloud.las"
    _write_synthetic_point_cloud(las_path)

    result = run_dem_generation(las_path, project_epsg=31983, output_path=tmp_path / "dem.tif")

    assert result.resolution_source == "automatic"
    assert result.num_points_used > 0

    with rasterio.open(result.raster_path) as src:
        assert src.crs.to_epsg() == 31983
        band = src.read(1)
        transform = src.transform

        rng = np.random.default_rng(1)
        sample_x = rng.uniform(10.0, 110.0, 200)
        sample_y = rng.uniform(10.0, 90.0, 200)
        errors = []
        for xw, yw in zip(sample_x, sample_y):
            row, col = rasterio.transform.rowcol(transform, xw, yw)
            raster_z = band[row, col]
            true_z = _true_surface(np.array([xw]), np.array([yw]))[0]
            errors.append(abs(raster_z - true_z))

        mean_abs_error = float(np.mean(errors))
        max_error = float(np.max(errors))

    # Dense, smooth synthetic surface -> interpolation should track it
    # closely, well within the sub-decimeter range, not just "roughly right".
    assert mean_abs_error < 0.15
    assert max_error < 0.5


def test_user_defined_resolution_is_respected(tmp_path: Path):
    las_path = tmp_path / "cloud.las"
    _write_synthetic_point_cloud(las_path)

    result = run_dem_generation(
        las_path, project_epsg=31983, output_path=tmp_path / "dem.tif", config=DemConfig(resolution_m=1.0)
    )

    assert result.resolution_m == pytest.approx(1.0)
    assert result.resolution_source == "user-defined"


def test_geotransform_and_nodata_are_correct(tmp_path: Path):
    las_path = tmp_path / "cloud.las"
    _write_synthetic_point_cloud(las_path)

    result = run_dem_generation(
        las_path, project_epsg=31983, output_path=tmp_path / "dem.tif", config=DemConfig(resolution_m=2.0)
    )

    with rasterio.open(result.raster_path) as src:
        assert src.nodata == -9999.0
        assert src.width == result.width_px
        assert src.height == result.height_px
        # North-up: pixel height in the affine transform is negative.
        assert src.transform.e < 0
        assert src.transform.a == pytest.approx(2.0)


def test_no_holes_left_in_output_raster(tmp_path: Path):
    """Hole-filling (nearest-neighbor beyond the linear-interpolation
    convex hull) should mean no NaN/NoData pixels remain inside the point
    cloud's bounding box."""
    las_path = tmp_path / "cloud.las"
    _write_synthetic_point_cloud(las_path)

    result = run_dem_generation(las_path, project_epsg=31983, output_path=tmp_path / "dem.tif")

    with rasterio.open(result.raster_path) as src:
        band = src.read(1)
        assert not np.any(band == -9999.0)
        assert not np.any(np.isnan(band))


def test_outlier_filtering_removes_extreme_points(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 100.0, 2000)
    y = rng.uniform(0.0, 100.0, 2000)
    z = _true_surface(x, y)
    # Inject a cluster of wildly wrong points (e.g. a matching blunder).
    # A naive fixed-percentile clip would only ever trim a fixed ~0.2% of
    # points regardless of how many of these there are; a robust
    # (MAD-based) filter should catch all of them since they deviate by
    # ~500m from a median around 700m.
    z[:10] += 500.0

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([x.min(), y.min(), z.min()])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las_path = tmp_path / "cloud.las"
    las.write(las_path)

    result = run_dem_generation(las_path, project_epsg=31983, output_path=tmp_path / "dem.tif")

    assert result.num_points_filtered_as_outliers == 10
    assert result.max_elevation_m < 800.0  # the +500m blunders were dropped


def test_raises_when_point_cloud_missing(tmp_path: Path):
    with pytest.raises(DemError, match="not found"):
        run_dem_generation(tmp_path / "nope.las", project_epsg=31983, output_path=tmp_path / "dem.tif")


def test_config_rejects_non_positive_resolution():
    with pytest.raises(ValueError):
        DemConfig(resolution_m=0)
    with pytest.raises(ValueError):
        DemConfig(resolution_m=-1.0)
