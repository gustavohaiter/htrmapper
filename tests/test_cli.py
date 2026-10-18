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
import pytest
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


def test_dem_is_written_in_the_crs_its_coordinates_are_in_even_with_an_export_crs(tmp_path: Path, capsys):
    # The DEM is computed from project-CRS (UTM, meters) points and nothing
    # reprojects it. An earlier fix labeled it with the configured export
    # CRS instead (EPSG:4674 here, geographic degrees), which put UTM
    # coordinates into a degrees CRS: the raster lands in the wrong place.
    # It must declare the project CRS and say that reprojection is not done.
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
        assert src.crs.to_epsg() == 31983
    assert "reprojection is not implemented" in capsys.readouterr().err


def test_orthometric_command_writes_an_extra_dem_and_keeps_the_project_dem_ellipsoidal(tmp_path: Path):
    from rasterio.transform import from_origin

    from htrmapper.core.project import DemSummary

    grid = tmp_path / "grid.txt"
    grid.write_text(
        "\n".join(
            f"{lon:.7f} {lat:.7f} -6.0000"
            for lat in np.arange(-22.5, -21.79, 1 / 12)
            for lon in np.arange(312.4, 313.01, 1 / 12)
        )
    )
    dem = tmp_path / "dem.tif"
    with rasterio.open(
        dem, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32", crs="EPSG:31983",
        transform=from_origin(260300.0, 7548700.0, 1.0, 1.0), nodata=-9999.0,
    ) as dst:
        dst.write(np.full((1, 4, 4), 590.0, dtype=np.float32))
    project = Project(name="vertical", dem=DemSummary(raster_path=str(dem)))
    project_path = tmp_path / "project.json"
    project.save(project_path)

    out = tmp_path / "dem_normal.tif"
    assert main(["orthometric", str(project_path), "--output", str(out), "--geoid-grid", str(grid)]) == 0

    with rasterio.open(out) as src:
        assert src.read(1)[0, 0] == pytest.approx(596.0)
    saved = Project.load(project_path)
    assert saved.dem.raster_path == str(dem)
    assert saved.dem.orthometric_raster_path == str(out)


def test_checkpoints_command_prints_errors_and_writes_the_report(tmp_path: Path, capsys):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_checkpoints import ORIGIN_X, ORIGIN_Y, _write_dem, _write_ortho

    from htrmapper.core.project import DemSummary, OrthoSummary

    x, y = ORIGIN_X + 25.0, ORIGIN_Y - 25.0
    _write_ortho(tmp_path / "ortho.tif", [(x + 0.10, y, 0.6)])
    _write_dem(tmp_path / "dem.tif")
    points = tmp_path / "pontos.csv"
    points.write_text(f"nome;E;N;h\nT1;{x};{y};580,50\n")
    project = Project(
        name="cp",
        dem=DemSummary(raster_path=str(tmp_path / "dem.tif")),
        ortho=OrthoSummary(raster_path=str(tmp_path / "ortho.tif")),
    )
    project.save(tmp_path / "project.json")

    assert main(["checkpoints", str(tmp_path / "project.json"), "--points", str(points)]) == 0

    out = capsys.readouterr().out
    assert "T1" in out and "auto" in out and "+10.0" in out
    assert (tmp_path / "pontos_relatorio.csv").is_file()
