"""Fase 7: cancellation and progress-reporting plumbing.

The GUI (Fase 7) needs two things pycolmap already supports natively: a
`CancellationToken` that a background worker can cancel from the main
thread, and phase-level progress callbacks. These tests validate that
contract directly against the pipeline functions, independent of any Qt
code -- a pre-cancelled token must raise `InterruptedError` (never a
generic pipeline error, so a caller/GUI can tell "cancelled" apart from
"failed"), and progress callbacks must actually fire with real values.
"""

from __future__ import annotations

from pathlib import Path

import pycolmap
import pytest
import rasterio
from rasterio.transform import from_origin
import numpy as np

from htrmapper.core.project import Project, ProjectCrsConfig
from htrmapper.dem.generation import run_dem_generation
from htrmapper.io.image_import import import_folder
from htrmapper.ortho.orthomosaic import run_orthomosaic_generation
from htrmapper.sfm.pipeline import SfmConfig, run_structure_from_motion
from tests.synthetic_scene import (
    SyntheticFlightSpec,
    _camera_grid_positions,
    _required_terrain_size_px,
    build_ground_truth_reconstruction,
    generate_smooth_gradient_terrain,
    generate_synthetic_flight,
)


def _build_project(tmp_path: Path, spec: SyntheticFlightSpec | None = None):
    truths = generate_synthetic_flight(tmp_path, spec)
    records, report = import_folder(tmp_path)
    assert not report.has_problems, report.summary_lines()
    project = Project(
        name="synthetic_flight",
        crs=ProjectCrsConfig(source_epsg=4326, project_epsg=31983),
        images=records,
    )
    return project, truths


def test_sfm_pre_cancelled_token_raises_interrupted_error(tmp_path: Path):
    project, _truths = _build_project(tmp_path)
    token = pycolmap.CancellationToken()
    token.cancel()

    with pytest.raises(InterruptedError):
        run_structure_from_motion(project, tmp_path / "work", SfmConfig(key_point_limit=8000), cancellation_token=token)


def test_sfm_progress_callback_reports_phases(tmp_path: Path):
    project, _truths = _build_project(tmp_path)
    phases: list[str] = []

    run_structure_from_motion(
        project, tmp_path / "work", SfmConfig(key_point_limit=8000), progress_callback=phases.append
    )

    assert "Extraindo features (SIFT)" in phases
    assert "Reconstrução incremental (SfM)" in phases
    # Phases fire in pipeline order, not just as an unordered set.
    assert phases.index("Extraindo features (SIFT)") < phases.index("Reconstrução incremental (SfM)")


def _build_flat_dem(spec, positions, path, resolution_m: float = 0.5) -> None:
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


def _setup_ortho_scene(tmp_path: Path):
    spec = SyntheticFlightSpec()
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
    return reconstruction_path, images_dir, dem_path


def test_orthomosaic_pre_cancelled_token_raises_interrupted_error(tmp_path: Path):
    reconstruction_path, images_dir, dem_path = _setup_ortho_scene(tmp_path)
    token = pycolmap.CancellationToken()
    token.cancel()

    with pytest.raises(InterruptedError):
        run_orthomosaic_generation(
            reconstruction_path, images_dir, dem_path, tmp_path / "out.tif", cancellation_token=token
        )


def test_orthomosaic_progress_callback_reports_every_camera(tmp_path: Path):
    reconstruction_path, images_dir, dem_path = _setup_ortho_scene(tmp_path)
    calls: list[tuple[int, int]] = []

    run_orthomosaic_generation(
        reconstruction_path, images_dir, dem_path, tmp_path / "out.tif", progress_callback=lambda done, total: calls.append((done, total))
    )

    assert len(calls) == 6  # one call per registered camera in the synthetic scene
    assert calls[-1] == (6, 6)
    assert [c[0] for c in calls] == [1, 2, 3, 4, 5, 6]


class _FakeCancellationToken:
    """DEM generation has no pycolmap call, so it accepts anything with an
    `is_cancelled` property rather than requiring a real
    `pycolmap.CancellationToken` -- this stand-in proves that contract."""

    def __init__(self, cancelled: bool) -> None:
        self.is_cancelled = cancelled


def test_dem_pre_cancelled_token_raises_interrupted_error(tmp_path: Path):
    las_path = tmp_path / "points.las"
    import laspy

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x = [0.0, 1.0, 1.0, 0.0]
    las.y = [0.0, 0.0, 1.0, 1.0]
    las.z = [10.0, 10.1, 9.9, 10.0]
    las.write(las_path)

    with pytest.raises(InterruptedError):
        run_dem_generation(
            las_path, 31983, tmp_path / "dem.tif", cancellation_token=_FakeCancellationToken(True)
        )


def test_dem_progress_callback_reports_phases(tmp_path: Path):
    las_path = tmp_path / "points.las"
    import laspy

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    rng_x = [float(i % 10) for i in range(200)]
    rng_y = [float(i // 10) for i in range(200)]
    las.x = rng_x
    las.y = rng_y
    las.z = [10.0 + 0.01 * i for i in range(200)]
    las.write(las_path)

    phases: list[str] = []
    run_dem_generation(las_path, 31983, tmp_path / "dem.tif", progress_callback=phases.append)

    assert "Carregando nuvem de pontos (LAS)" in phases
    assert "Escrevendo GeoTIFF" in phases
    assert phases.index("Carregando nuvem de pontos (LAS)") < phases.index("Escrevendo GeoTIFF")
