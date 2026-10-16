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
        assert src.count == 3
        rgb = src.read([1, 2, 3])
        alpha = src.read_masks(1)
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
        alpha = src.read_masks(1)
        # The DEM extends with a margin beyond the actual camera coverage,
        # so the far corners should be masked out.
        assert alpha[0, 0] == 0
        assert alpha[-1, -1] == 0
        assert np.any(alpha == 255)


def test_ortho_config_rejects_invalid_max_view_angle():
    with pytest.raises(ValueError):
        OrthoConfig(max_view_angle_deg=0.0)
    with pytest.raises(ValueError):
        OrthoConfig(max_view_angle_deg=90.0)


def test_raises_when_reconstruction_missing(tmp_path: Path):
    (tmp_path / "images").mkdir()
    with pytest.raises(OrthoError, match="reconstruction not found"):
        run_orthomosaic_generation(tmp_path / "nope", tmp_path / "images", tmp_path / "dem.tif", tmp_path / "out.tif")


def test_raises_when_dem_missing(tmp_path: Path):
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)

    with pytest.raises(OrthoError, match="DEM not found"):
        run_orthomosaic_generation(reconstruction_path, images_dir, tmp_path / "no_dem.tif", tmp_path / "out.tif")


def test_ortho_config_rejects_non_positive_resolution():
    with pytest.raises(ValueError):
        OrthoConfig(resolution_m=0.0)
    with pytest.raises(ValueError):
        OrthoConfig(resolution_m=-1.0)


def test_automatic_resolution_is_independent_of_a_much_coarser_dem(tmp_path: Path):
    # Regression test for a real bug (found 2026-10-07 on a real 56-photo
    # flight): the orthomosaic used to simply reuse the DEM's own raster
    # grid, so a coarse DEM (e.g. from a sparse dense cloud, or a
    # deliberately low DEM quality tier) silently forced the orthomosaic
    # coarse too. A DEM resolution 10x coarser than the camera's real GSD
    # must not change the orthomosaic's automatic resolution at all.
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)

    coarse_dem_path = tmp_path / "dem_coarse.tif"
    positions = _camera_grid_positions(spec)
    _build_flat_dem(spec, positions, coarse_dem_path, resolution_m=spec.gsd_m_per_px * 10)

    fine_result = run_orthomosaic_generation(reconstruction_path, images_dir, dem_path, tmp_path / "ortho_fine_dem.tif")
    coarse_result = run_orthomosaic_generation(
        reconstruction_path, images_dir, coarse_dem_path, tmp_path / "ortho_coarse_dem.tif"
    )

    assert fine_result.resolution_source == "automatic (camera GSD)"
    assert coarse_result.resolution_source == "automatic (camera GSD)"
    # Same automatic resolution regardless of the DEM's own resolution --
    # the whole point of the fix.
    assert coarse_result.resolution_m == pytest.approx(fine_result.resolution_m, rel=0.05)


def test_automatic_resolution_matches_known_camera_gsd(tmp_path: Path):
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)

    result = run_orthomosaic_generation(reconstruction_path, images_dir, dem_path, tmp_path / "ortho.tif")

    assert result.resolution_source == "automatic (camera GSD)"
    # Estimated from real (bundle-adjusted-equivalent, here ground-truth)
    # camera height above DEM terrain and calibrated focal length -- should
    # land close to the synthetic flight's known, designed-in GSD.
    assert result.resolution_m == pytest.approx(spec.gsd_m_per_px, rel=0.1)


def test_tile_size_never_changes_the_output_pixel_for_pixel(tmp_path: Path, monkeypatch):
    # Regression test for the tiled-rendering rewrite (real bug found
    # 2026-10-07: the previous whole-grid implementation OOM-killed on a
    # real ~168-million-pixel flight once the resolution-decoupling fix
    # above made the orthomosaic render near native image GSD). Tiling
    # must be purely an implementation/memory detail -- a tiny tile size
    # (forcing many tile boundaries to cross right through the covered
    # area) must produce a byte-for-byte identical raster to one giant
    # tile (the old, whole-grid-at-once behavior), proving no seam or
    # edge artifact is introduced at tile boundaries.
    import htrmapper.ortho.orthomosaic as orthomosaic_module

    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)

    monkeypatch.setattr(orthomosaic_module, "TILE_SIZE_PX", 100_000)
    whole_grid_result = run_orthomosaic_generation(
        reconstruction_path, images_dir, dem_path, tmp_path / "ortho_whole.tif"
    )

    monkeypatch.setattr(orthomosaic_module, "TILE_SIZE_PX", 8)
    tiled_result = run_orthomosaic_generation(reconstruction_path, images_dir, dem_path, tmp_path / "ortho_tiled.tif")

    assert tiled_result.width_px == whole_grid_result.width_px
    assert tiled_result.height_px == whole_grid_result.height_px
    assert tiled_result.num_valid_pixels == whole_grid_result.num_valid_pixels
    assert tiled_result.num_cameras_used == whole_grid_result.num_cameras_used

    with rasterio.open(whole_grid_result.raster_path) as whole_src, rasterio.open(tiled_result.raster_path) as tiled_src:
        assert np.array_equal(whole_src.read(), tiled_src.read())


def test_explicit_resolution_overrides_automatic_estimate(tmp_path: Path):
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)

    forced_resolution = spec.gsd_m_per_px * 3
    result = run_orthomosaic_generation(
        reconstruction_path, images_dir, dem_path, tmp_path / "ortho.tif", OrthoConfig(resolution_m=forced_resolution)
    )

    assert result.resolution_source == "user-defined"
    assert result.resolution_m == pytest.approx(forced_resolution)


def _camera_centers(reconstruction_path: Path) -> dict[str, np.ndarray]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(reconstruction_path))
    return {
        reconstruction.image(i).name: np.array(reconstruction.image(i).projection_center())
        for i in reconstruction.reg_image_ids()
    }


def _pixel_at(src, x: float, y: float) -> tuple[np.ndarray, int]:
    row, col = rasterio.transform.rowcol(src.transform, x, y)
    window = rasterio.windows.Window(col, row, 1, 1)
    return src.read(window=window)[:, 0, 0].astype(np.float64), int(src.read_masks(1, window=window)[0, 0])


def test_each_pixel_comes_from_the_most_nadir_camera_not_an_average(tmp_path: Path):
    # Regression test for the blur on the first real flight (ARCHITECTURE.md
    # seção 26): averaging every overlapping view ghosts any DEM height
    # error. One camera's image is replaced by solid red: at its own nadir
    # point the mosaic must be red (it is the most vertical view there),
    # and at a neighbor's nadir point -- still inside the red camera's
    # footprint, but seen more vertically by the neighbor -- it must be the
    # true terrain color, with no red tint. A plain average would tint both.
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)
    centers = _camera_centers(reconstruction_path)
    red_name = sorted(centers)[0]
    neighbor_name = min(
        (n for n in centers if n != red_name), key=lambda n: np.linalg.norm(centers[n][:2] - centers[red_name][:2])
    )
    with PILImage.open(images_dir / red_name) as img:
        size = img.size
    PILImage.new("RGB", size, (255, 0, 0)).save(images_dir / red_name, quality=95)

    result = run_orthomosaic_generation(reconstruction_path, images_dir, dem_path, tmp_path / "ortho.tif")

    with rasterio.open(result.raster_path) as src:
        red_color, red_mask = _pixel_at(src, *centers[red_name][:2])
        neighbor_color, neighbor_mask = _pixel_at(src, *centers[neighbor_name][:2])

    assert red_mask == 255 and neighbor_mask == 255
    assert np.abs(red_color - [255, 0, 0]).max() < 25
    true_color = _terrain_color_at(terrain, spec, *centers[neighbor_name][:2])
    assert np.abs(neighbor_color - true_color).mean() < 10


def test_pixels_seen_only_obliquely_are_cropped(tmp_path: Path):
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)
    centers = np.array(list(_camera_centers(reconstruction_path).values()))

    loose = run_orthomosaic_generation(
        reconstruction_path, images_dir, dem_path, tmp_path / "loose.tif", OrthoConfig(max_view_angle_deg=None)
    )
    default = run_orthomosaic_generation(reconstruction_path, images_dir, dem_path, tmp_path / "default.tif")
    tight = run_orthomosaic_generation(
        reconstruction_path, images_dir, dem_path, tmp_path / "tight.tif", OrthoConfig(max_view_angle_deg=10.0)
    )

    assert tight.num_valid_pixels < default.num_valid_pixels < loose.num_valid_pixels

    # Geometric check on the tight result: on this flat terrain, every kept
    # pixel must lie within tan(10 deg) * flying height of some camera's
    # nadir point (plus half a pixel diagonal for the pixel-center grid).
    with rasterio.open(tight.raster_path) as src:
        mask = src.read_masks(1) > 0
        rows, cols = np.nonzero(mask)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        tolerance = src.res[0]
    xy = np.column_stack([xs, ys])
    nearest = np.min(np.linalg.norm(xy[:, None, :] - centers[None, :, :2], axis=2), axis=1)
    height = centers[0, 2] - spec.terrain_elevation_m
    assert nearest.max() <= np.tan(np.radians(10.0)) * height + tolerance


def test_output_is_tiled_jpeg_with_internal_mask_and_overviews(tmp_path: Path):
    spec, terrain, images_dir, reconstruction_path, dem_path = _setup_ground_truth_scene(tmp_path)

    result = run_orthomosaic_generation(reconstruction_path, images_dir, dem_path, tmp_path / "ortho.tif")

    with rasterio.open(result.raster_path) as src:
        assert max(src.width, src.height) > 512  # so overviews are actually expected here
        assert src.count == 3
        assert src.compression == rasterio.enums.Compression.jpeg
        assert src.photometric == rasterio.enums.PhotometricInterp.ycbcr
        assert src.block_shapes[0] == (512, 512)
        assert src.mask_flag_enums[0] == [rasterio.enums.MaskFlags.per_dataset]
        assert src.overviews(1)
        mask = src.read_masks(1)
    assert int((mask > 0).sum()) == result.num_valid_pixels
