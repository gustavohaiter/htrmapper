from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from htrmapper.geo.checkpoints import (
    CheckpointError,
    evaluate_checkpoints,
    read_field_points,
    write_report_csv,
)

ORIGIN_X, ORIGIN_Y, GSD = 260000.0, 7549000.0, 0.025
SIZE = 2000  # 50 m x 50 m


def _write_ortho(path: Path, targets: list[tuple[float, float, float]], bright: bool = True) -> None:
    """Green, textured field (like cane rows) with square targets of side
    `side` m centered at the given world positions."""
    rng = np.random.default_rng(0)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    base = 90 + 25 * np.sin(cols / 20.0) + rng.normal(0, 8, (SIZE, SIZE))
    rgb = np.stack([base * 0.6, base, base * 0.5]).clip(0, 255)
    x = ORIGIN_X + (cols + 0.5) * GSD
    y = ORIGIN_Y - (rows + 0.5) * GSD
    for cx, cy, side in targets:
        square = (np.abs(x - cx) <= side / 2) & (np.abs(y - cy) <= side / 2)
        rgb[:, square] = 245 if bright else 15
    with rasterio.open(
        path, "w", driver="GTiff", height=SIZE, width=SIZE, count=3, dtype="uint8", crs="EPSG:31983",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, GSD, GSD),
    ) as dst:
        dst.write(rgb.astype(np.uint8))


def _write_dem(path: Path) -> None:
    rows, cols = np.mgrid[0:100, 0:100]
    z = 580.0 + 0.02 * (ORIGIN_X + (cols + 0.5) * 0.5 - ORIGIN_X)
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32", crs="EPSG:31983",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, 0.5, 0.5), nodata=-9999.0,
    ) as dst:
        dst.write(z.astype(np.float32), 1)


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_automatic_detection_measures_a_known_offset(tmp_path: Path):
    # The mosaic shows each target 12 cm east and 7 cm south of where the
    # rover measured it: exactly the error the report must find.
    field = [(ORIGIN_X + 10.0, ORIGIN_Y - 10.0), (ORIGIN_X + 35.0, ORIGIN_Y - 20.0), (ORIGIN_X + 20.0, ORIGIN_Y - 40.0)]
    _write_ortho(tmp_path / "ortho.tif", [(x + 0.12, y - 0.07, 0.6) for x, y in field])
    csv_path = _write_csv(
        tmp_path / "pontos.csv",
        "nome;E;N;h\n" + "".join(f"P{i};{x:.3f};{y:.3f};580,5\n" for i, (x, y) in enumerate(field, start=1)),
    )

    report = evaluate_checkpoints(read_field_points(csv_path), tmp_path / "ortho.tif")

    assert [r.method for r in report.results] == ["auto"] * 3
    for result in report.results:
        assert result.dx == pytest.approx(0.12, abs=GSD)
        assert result.dy == pytest.approx(-0.07, abs=GSD)
    assert report.rmse("d2d") == pytest.approx(np.hypot(0.12, 0.07), abs=GSD)


def test_dark_targets_and_manual_measurements(tmp_path: Path):
    _write_ortho(tmp_path / "ortho.tif", [(ORIGIN_X + 25.0, ORIGIN_Y - 25.0, 0.8)], bright=False)
    csv_path = _write_csv(
        tmp_path / "pontos.csv",
        "nome,E,N,h,E_ortho,N_ortho\n"
        f"A,{ORIGIN_X + 25.0},{ORIGIN_Y - 25.0},,,\n"
        f"B,{ORIGIN_X + 5.0},{ORIGIN_Y - 5.0},,{ORIGIN_X + 5.04},{ORIGIN_Y - 4.97}\n",
    )

    report = evaluate_checkpoints(read_field_points(csv_path), tmp_path / "ortho.tif", bright_targets=False)

    a, b = report.results
    assert a.method == "auto" and a.d2d < GSD
    assert b.method == "manual"
    assert b.dx == pytest.approx(0.04) and b.dy == pytest.approx(0.03)


def test_no_target_is_reported_as_not_found_never_guessed(tmp_path: Path):
    _write_ortho(tmp_path / "ortho.tif", [])
    csv_path = _write_csv(tmp_path / "pontos.csv", f"P1;{ORIGIN_X + 25.0};{ORIGIN_Y - 25.0}\n")

    report = evaluate_checkpoints(read_field_points(csv_path), tmp_path / "ortho.tif")

    assert report.results[0].method == "not found"
    assert report.results[0].d2d is None
    assert report.rmse("d2d") is None


def test_vertical_error_against_the_dem_and_csv_report(tmp_path: Path):
    _write_ortho(tmp_path / "ortho.tif", [(ORIGIN_X + 25.0, ORIGIN_Y - 25.0, 0.6)])
    _write_dem(tmp_path / "dem.tif")
    # DEM at E = origin + 25 m is 580.5; the rover says 580.42.
    csv_path = _write_csv(tmp_path / "pontos.csv", f"P1;{ORIGIN_X + 25.0};{ORIGIN_Y - 25.0};580,42\n")

    report = evaluate_checkpoints(read_field_points(csv_path), tmp_path / "ortho.tif", tmp_path / "dem.tif")
    write_report_csv(report, tmp_path / "relatorio.csv")

    assert report.results[0].dz == pytest.approx(0.08, abs=0.005)
    text = (tmp_path / "relatorio.csv").read_text()
    assert text.splitlines()[0].startswith("nome;E_campo;N_campo")
    assert "RMSE" in text


def test_unreadable_rows_are_an_error(tmp_path: Path):
    csv_path = _write_csv(tmp_path / "pontos.csv", "nome;E;N\nP1;abc;7549000\n")
    with pytest.raises(CheckpointError, match="row 1"):
        read_field_points(csv_path)


def test_the_target_nearest_to_the_field_coordinate_wins_over_a_distractor(tmp_path: Path):
    # A second bright object (a bag, a stone) inside the search radius,
    # farther from the rover coordinate than the real target.
    x, y = ORIGIN_X + 25.0, ORIGIN_Y - 25.0
    _write_ortho(tmp_path / "ortho.tif", [(x + 0.15, y + 0.05, 0.6), (x - 0.7, y - 0.6, 0.5)])
    csv_path = _write_csv(tmp_path / "pontos.csv", f"P1;{x};{y}\n")

    report = evaluate_checkpoints(read_field_points(csv_path), tmp_path / "ortho.tif")

    assert report.results[0].dx == pytest.approx(0.15, abs=GSD)
    assert report.results[0].dy == pytest.approx(0.05, abs=GSD)
