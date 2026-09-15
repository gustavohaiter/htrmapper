"""CLI-level regression tests.

Not a full CLI test suite (each subcommand's actual computation is already
covered where it lives: `sfm.pipeline`, `ba.weighted_bundle_adjustment`,
`mvs.dense`, `dem.generation`, `ortho.orthomosaic`) -- this file exists for
bugs that only show up at the CLI's own glue-code layer (argument wiring,
which `Project` field feeds which function parameter).
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import rasterio

from htrmapper.cli.main import main
from htrmapper.core.project import MvsSummary, Project, ProjectCrsConfig


def _write_synthetic_point_cloud(path: Path, num_points: int = 200) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 50.0, num_points)
    y = rng.uniform(0.0, 50.0, num_points)
    z = 700.0 + 0.01 * x + 0.01 * y

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([x.min(), y.min(), z.min()])
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)


def test_dem_command_honors_a_configured_export_epsg(tmp_path: Path):
    # Regression test for a real bug: `htrmapper dem` used to pass
    # `project.crs.project_epsg` straight to `run_dem_generation`,
    # ignoring `export_epsg` entirely -- a user who configured a distinct
    # export CRS would silently get their project's working CRS instead.
    las_path = tmp_path / "cloud.las"
    _write_synthetic_point_cloud(las_path)

    project = Project(
        name="export_epsg_test",
        crs=ProjectCrsConfig(source_epsg=4326, project_epsg=31983, export_epsg=4674),
        mvs=MvsSummary(point_cloud_las_path=str(las_path)),
    )
    project_path = tmp_path / "project.json"
    project.save(project_path)

    output_path = tmp_path / "dem.tif"
    exit_code = main(["dem", str(project_path), "--output", str(output_path)])

    assert exit_code == 0
    with rasterio.open(output_path) as src:
        assert src.crs.to_epsg() == 4674
