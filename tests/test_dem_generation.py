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


def test_no_holes_left_where_the_cloud_covers_the_whole_area(tmp_path: Path):
    """Empty cells between sparse points are interpolated: a cloud that
    covers its whole bounding box leaves no NaN/NoData pixels."""
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


def _write_points(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([x.min(), y.min(), z.min()])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.write(path)


def _read_dem(path: Path):
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float64)
        transform = src.transform
        band[band == src.nodata] = np.nan
    rows, cols = np.mgrid[0 : band.shape[0], 0 : band.shape[1]]
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    return band, x, y, transform


def test_local_spikes_are_removed_but_a_narrow_real_feature_is_kept(tmp_path: Path):
    # A dense cloud over a plane with a 1 m wide, 50 cm high embankment
    # (narrower than the 5x5-cell noise window) plus isolated blunders of
    # +-1.5 m: the blunders are local, so the global MAD filter keeps them,
    # and the earlier linear interpolation spread each one into a spike.
    rng = np.random.default_rng(3)
    n = 120_000
    x = rng.uniform(0.0, 100.0, n)
    y = rng.uniform(0.0, 100.0, n)
    embankment = (x >= 50.0) & (x < 51.0)
    z = 600.0 + 0.01 * x + np.where(embankment, 0.5, 0.0) + rng.normal(0.0, 0.02, n)
    blunders = (rng.random(n) < 0.003) & ~embankment
    z[blunders] += rng.choice([-1.5, 1.5], blunders.sum())
    las_path = tmp_path / "cloud.las"
    _write_points(las_path, x, y, z)

    result = run_dem_generation(las_path, 31983, tmp_path / "dem.tif", DemConfig(resolution_m=0.5))

    band, gx, gy, _ = _read_dem(tmp_path / "dem.tif")
    expected = 600.0 + 0.01 * gx + np.where((gx >= 50.0) & (gx < 51.0), 0.5, 0.0)
    error = np.abs(band - expected)
    assert np.nanmax(error[(gx < 49.0) | (gx > 52.0)]) < 0.15
    assert np.nanmedian(band[(gx > 50.0) & (gx < 51.0)] - 600.0 - 0.01 * gx[(gx > 50.0) & (gx < 51.0)]) > 0.4
    assert result.num_cells_despiked > 0


def test_no_terrain_is_invented_outside_the_reconstructed_area(tmp_path: Path):
    # An L-shaped cloud: the missing quadrant is inside the bounding box
    # but has no points. It must be NoData (beyond the small gap margin),
    # not nearest-neighbor terrain extrapolated out to the bounding box.
    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 100.0, 60_000)
    y = rng.uniform(0.0, 100.0, 60_000)
    keep = ~((x > 50.0) & (y > 50.0))
    x, y = x[keep], y[keep]
    las_path = tmp_path / "cloud.las"
    _write_points(las_path, x, y, 600.0 + 0.0 * x)

    result = run_dem_generation(las_path, 31983, tmp_path / "dem.tif", DemConfig(resolution_m=0.5))

    band, gx, gy, _ = _read_dem(tmp_path / "dem.tif")
    assert np.all(np.isnan(band[(gx > 55.0) & (gy > 55.0)]))
    assert not np.any(np.isnan(band[(gx < 45.0) | (gy < 45.0)]))
    assert 0.2 < result.nodata_fraction < 0.3


def test_a_hole_enclosed_by_data_is_interpolated_across(tmp_path: Path):
    # A 20 m wide gap in the middle of a tilted plane (water, a bare
    # homogeneous patch): enclosed by reconstructed surface, so it is
    # filled, and linearly -- it must follow the slope, not plateau.
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 100.0, 60_000)
    y = rng.uniform(0.0, 100.0, 60_000)
    keep = (x - 50.0) ** 2 + (y - 50.0) ** 2 > 10.0**2
    x, y = x[keep], y[keep]
    las_path = tmp_path / "cloud.las"
    _write_points(las_path, x, y, 600.0 + 0.05 * x + 0.02 * y)

    run_dem_generation(las_path, 31983, tmp_path / "dem.tif", DemConfig(resolution_m=0.5))

    band, gx, gy, _ = _read_dem(tmp_path / "dem.tif")
    hole = (gx - 50.0) ** 2 + (gy - 50.0) ** 2 < 9.0**2
    assert not np.any(np.isnan(band[hole]))
    assert np.max(np.abs(band[hole] - (600.0 + 0.05 * gx[hole] + 0.02 * gy[hole]))) < 0.05


def test_grid_is_snapped_to_multiples_of_the_resolution(tmp_path: Path):
    las_path = tmp_path / "cloud.las"
    _write_synthetic_point_cloud(las_path)

    run_dem_generation(las_path, 31983, tmp_path / "dem.tif", DemConfig(resolution_m=0.5))

    _, _, _, transform = _read_dem(tmp_path / "dem.tif")
    assert transform.c / 0.5 == pytest.approx(round(transform.c / 0.5))
    assert transform.f / 0.5 == pytest.approx(round(transform.f / 0.5))
