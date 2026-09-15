from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image as PILImage
from rasterio.transform import from_origin

from htrmapper.ortho.orthomosaic import OrthoConfig, OrthoError, run_orthomosaic_generation
from tests.synthetic_scene import (
    TEXTURE_PX_PER_METER,
    SyntheticFlightSpec,
    _camera_grid_positions,
    _required_terrain_size_px,
    build_ground_truth_reconstruction,
    generate_smooth_gradient_terrain,
    generate_synthetic_flight,
)


def _build_flat_dem(spec: SyntheticFlightSpec, positions: list[tuple[float, float]], path: Path, resolution_m: float = 0.5) -> None:
    # Must exceed a single camera's footprint reach beyond the outermost
    # camera center, or the "outside coverage" corners used below would
    # actually still be seen by that camera.
    margin = max(spec.footprint_width_m, spec.footprint_height_m) * 1.5
    min_x = spec.origin_x_m + min(p[0] for p in positions) - spec.footprint_width_m / 2 - margin
    max_x = spec.origin_x_m + max(p[0] for p in positions) + spec.footprint_width_m / 2 + margin
    min_y = spec.origin_y_m + min(p[1] for p in positions) - spec.footprint_height_m / 2 - margin
    max_y = spec.origin_y_m + max(p[1] for p in positions) + spec.footprint_height_m / 2 + margin

    width = int((max_x - min_x) / resolution_m)
    height = int((max_y - min_y) / resolution_m)
    elevation = np.full((height, width), spec.terrain_elevation_m, dtype=np.float32)
    transform = from_origin(min_x, max_y, resolution_m, resolution_m)

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=rasterio.float32, crs="EPSG:31983", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(elevation, 1)


def _setup_ground_truth_scene(tmp_path: Path, spec: SyntheticFlightSpec | None = None):
    spec = spec or SyntheticFlightSpec()
    positions = _camera_grid_positions(spec)
    terrain_size = _required_terrain_size_px(spec, positions)
    terrain = generate_smooth_gradient_terrain(size_px=terrain_size, seed=0)

    images_dir = tmp_path / "images"
    truths = generate_synthetic_flight(images_dir, spec, terrain=terrain)

    reconstruction = build_ground_truth_reconstruction(spec, truths)
    reconstruction_path = tmp_path / "sparse"
    reconstruction_path.mkdir()
    reconstruction.write(reconstruction_path)

    dem_path = tmp_path / "dem.tif"
    _build_flat_dem(spec, positions, dem_path)

    return spec, terrain, images_dir, reconstruction_path, dem_path


def _terrain_color_at(terrain: PILImage.Image, spec: SyntheticFlightSpec, world_x: float, world_y: float) -> np.ndarray:
    terrain_arr = np.asarray(terrain)
    terrain_h, terrain_w = terrain_arr.shape[:2]
    x_m = world_x - spec.origin_x_m
    y_m = world_y - spec.origin_y_m
    px_x = int(round(terrain_w / 2.0 + x_m * TEXTURE_PX_PER_METER))
    px_y = int(round(terrain_h / 2.0 - y_m * TEXTURE_PX_PER_METER))
    return terrain_arr[px_y, px_x].astype(np.float64)


def test_orthomosaic_reconstructs_known_terrain_colors(tmp_path: Path):
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)

    result = run_orthomosaic_generation(reconstruction_path, images_dir, dem_path, tmp_path / "ortho.tif")

    assert result.num_cameras_used == 6
    assert result.num_valid_pixels > 0

    with rasterio.open(result.raster_path) as src:
        assert src.count == 4
        rgb = src.read([1, 2, 3])
        alpha = src.read(4)
        transform = src.transform

        rng = np.random.default_rng(0)
        # Sample well inside the flight's covered area (away from the
        # ragged edges of the camera grid's footprint union).
        sample_x = rng.uniform(spec.origin_x_m + 15, spec.origin_x_m + 20, 60)
        sample_y = rng.uniform(spec.origin_y_m + 10, spec.origin_y_m + 15, 60)

        errors = []
        for wx, wy in zip(sample_x, sample_y):
            row, col = rasterio.transform.rowcol(transform, wx, wy)
            if alpha[row, col] == 0:
                continue
            ortho_color = rgb[:, row, col].astype(np.float64)
            true_color = _terrain_color_at(terrain, spec, wx, wy)
            errors.append(np.abs(ortho_color - true_color).mean())

        assert len(errors) > 10
        mean_error = float(np.mean(errors))

    # JPEG compression + resampling (crop->resize->orthorectify->resample)
    # mean a small amount of color error is expected; this is not a
    # lossless pipeline. With a smooth (low-frequency) validation texture,
    # sub-pixel misalignment barely changes the sampled color, so a tight
    # tolerance here is a real geometric-correctness signal.
    assert mean_error < 8.0


def test_orthomosaic_has_transparent_nodata_outside_coverage(tmp_path: Path):
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)

    result = run_orthomosaic_generation(reconstruction_path, images_dir, dem_path, tmp_path / "ortho.tif")

    with rasterio.open(result.raster_path) as src:
        alpha = src.read(4)
        # The DEM extends with a margin beyond the actual camera coverage,
        # so the far corners should be nodata (alpha == 0).
        assert alpha[0, 0] == 0
        assert alpha[-1, -1] == 0
        assert np.any(alpha == 255)


def test_ortho_config_rejects_invalid_feather_fraction():
    with pytest.raises(ValueError):
        OrthoConfig(feather_fraction=0.0)
    with pytest.raises(ValueError):
        OrthoConfig(feather_fraction=0.6)


def test_raises_when_reconstruction_missing(tmp_path: Path):
    (tmp_path / "images").mkdir()
    with pytest.raises(OrthoError, match="reconstruction not found"):
        run_orthomosaic_generation(tmp_path / "nope", tmp_path / "images", tmp_path / "dem.tif", tmp_path / "out.tif")


def test_raises_when_dem_missing(tmp_path: Path):
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)

    with pytest.raises(OrthoError, match="DEM not found"):
        run_orthomosaic_generation(reconstruction_path, images_dir, tmp_path / "no_dem.tif", tmp_path / "out.tif")
