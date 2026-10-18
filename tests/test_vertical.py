from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from htrmapper.geo.vertical import (
    VERTICAL_REFERENCE_TAG,
    VerticalError,
    load_conversion_grid,
    write_orthometric_dem,
)

# Rows of IBGE's own hgeoHNOR2020 Imbituba grid around the first real
# flight (fazenda 20046, lat -22.15, lon -47.32), unchanged from
# hgeoHNOR2020__IMBITUBA__fator-conversao.txt.
REAL_EXCERPT = Path(__file__).parent / "data" / "hgeoHNOR2020_IMBITUBA_excerpt_fazenda20046.txt"


def _write_dem(path: Path, heights: np.ndarray, origin_x: float, origin_y: float, resolution: float) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=heights.shape[0], width=heights.shape[1], count=1,
        dtype="float32", crs="EPSG:31983", transform=from_origin(origin_x, origin_y, resolution, resolution),
        nodata=-9999.0,
    ) as dst:
        dst.write(heights.astype(np.float32), 1)


def _write_grid(path: Path, factor) -> None:
    # 5' grid in IBGE's layout (lon 0..360, lat, FC) around the test DEM.
    lines = []
    for lat in np.arange(-22.5, -21.79, 1 / 12):
        for lon in np.arange(312.4, 313.01, 1 / 12):
            lines.append(f"{lon:.7f} {lat:.7f} {factor(lon - 360.0, lat):.4f}")
    path.write_text("\n".join(lines))


def test_grid_reads_ibge_longitudes_in_0_to_360(tmp_path: Path):
    interpolator = load_conversion_grid(REAL_EXCERPT)
    value = interpolator([[-22.15121122, -47.32407528]])[0]
    # Interior of Sao Paulo: the geoid is below the ellipsoid, FC ~ -6 m.
    assert -7.0 < value < -5.0


def test_normal_height_is_ellipsoidal_minus_conversion_factor(tmp_path: Path):
    grid = tmp_path / "grid.txt"
    _write_grid(grid, lambda lon, lat: -6.0 + 0.5 * (lat + 22.15))
    heights = np.full((20, 30), 590.0)
    heights[0, 0] = -9999.0
    dem = tmp_path / "dem.tif"
    _write_dem(dem, heights, 260300.0, 7548700.0, 1.0)  # UTM 23S at the real flight

    result = write_orthometric_dem(dem, tmp_path / "dem_normal.tif", 31983, grid_path=grid)

    with rasterio.open(result.raster_path) as src:
        normal = src.read(1)
        tags = src.tags()
    assert normal[0, 0] == -9999.0
    lat_center = -22.1496  # latitude of 7548690 N on this meridian
    assert normal[10, 15] == pytest.approx(590.0 + 6.0 - 0.5 * (lat_center + 22.15), abs=0.01)
    assert tags[VERTICAL_REFERENCE_TAG].startswith("normal heights")


def test_cells_outside_the_model_are_an_error_not_left_ellipsoidal(tmp_path: Path):
    grid = tmp_path / "grid.txt"
    _write_grid(grid, lambda lon, lat: -6.0)
    dem = tmp_path / "dem.tif"
    _write_dem(dem, np.full((5, 5), 590.0), 500000.0, 7000000.0, 1.0)  # far south of the grid

    with pytest.raises(VerticalError, match="outside"):
        write_orthometric_dem(dem, tmp_path / "out.tif", 31983, grid_path=grid)


def test_an_already_converted_dem_is_not_converted_twice(tmp_path: Path):
    grid = tmp_path / "grid.txt"
    _write_grid(grid, lambda lon, lat: -6.0)
    dem = tmp_path / "dem.tif"
    _write_dem(dem, np.full((5, 5), 590.0), 260300.0, 7548700.0, 1.0)
    first = write_orthometric_dem(dem, tmp_path / "once.tif", 31983, grid_path=grid)

    with pytest.raises(VerticalError, match="already"):
        write_orthometric_dem(Path(first.raster_path), tmp_path / "twice.tif", 31983, grid_path=grid)
