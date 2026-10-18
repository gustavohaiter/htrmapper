"""Ellipsoidal -> orthometric (normal) heights with IBGE's hgeoHNOR2020.

GNSS (and therefore the DJI RTK geotags, see ARCHITECTURE.md section 28)
gives heights above the SIRGAS 2000 ellipsoid. Brazilian maps, the RAAP
benchmark network and most field references use normal heights above the
Imbituba (or, in Amapa, Santana) vertical datum. IBGE's hgeoHNOR2020
model gives, on a 5' grid, the conversion factor FC between the two:

    H (normal height) = h (ellipsoidal height) - FC

FC is about -6 m in the interior of Sao Paulo, so the normal height is
about 6 m above the ellipsoidal one there.

This module only writes an EXTRA raster. The project's DEM stays
ellipsoidal on purpose: the cameras are positioned with ellipsoidal
heights, and the orthomosaic must intersect their rays with a surface in
that same reference -- feeding it normal heights would put the ground
~6 m off and shift edge pixels by up to ~3.5 m (6 m x tan 30 deg).

The grids are IBGE's own text files (columns: longitude in 0..360 deg
east, latitude in deg, FC in m), downloaded once from IBGE's public FTP
into the user's cache, or read from a local copy.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

from htrmapper.geo.crs import CoordinateReferenceSystem, GeodeticTransformer, wgs84

HGEOHNOR2020_BASE_URL = (
    "https://geoftp.ibge.gov.br/modelos_digitais_de_superficie/modelo_de_ondulacao_geoidal/grades_hgeoHNOR2020/"
)
HGEOHNOR2020_DATUMS = {
    "imbituba": ("hgeoHNOR2020__grades-IMBITUBA.zip", "hgeoHNOR2020__IMBITUBA__fator-conversao.txt"),
    "santana": ("hgeoHNOR2020__grades-SANTANA.zip", "hgeoHNOR2020__SANTANA__fator-conversao.txt"),
}
VERTICAL_REFERENCE_TAG = "HTRMAPPER_VERTICAL_REFERENCE"


class VerticalError(RuntimeError):
    """Raised when the conversion cannot be done (missing grid, area
    outside the model)."""


@dataclass
class OrthometricResult:
    raster_path: str
    datum: str
    min_conversion_factor_m: float
    max_conversion_factor_m: float
    min_elevation_m: float
    max_elevation_m: float


def _grid_cache_path(datum: str) -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "htrmapper"
    return cache_root / "hgeoHNOR2020" / HGEOHNOR2020_DATUMS[datum][1]


def ensure_hgeohnor2020_grid(datum: str = "imbituba") -> Path:
    """Path to the conversion-factor grid for `datum`, downloading IBGE's
    zip into the cache on first use."""
    if datum not in HGEOHNOR2020_DATUMS:
        raise VerticalError(f"unknown vertical datum {datum!r}; choose from {sorted(HGEOHNOR2020_DATUMS)}")
    path = _grid_cache_path(datum)
    if path.is_file():
        return path
    zip_name, member = HGEOHNOR2020_DATUMS[datum]
    try:
        request = urllib.request.Request(HGEOHNOR2020_BASE_URL + zip_name, headers={"User-Agent": "htrmapper"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except OSError as exc:
        raise VerticalError(
            f"could not download the hgeoHNOR2020 grid from IBGE ({exc}); download {zip_name} from "
            f"{HGEOHNOR2020_BASE_URL} and pass the extracted '{member}' with --geoid-grid"
        ) from exc
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        data = archive.read(member)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def load_conversion_grid(path: Path) -> RegularGridInterpolator:
    """Bilinear interpolator (lat, lon in -180..180) -> FC in meters, from
    an IBGE hgeoHNOR2020 text grid."""
    values = np.loadtxt(path)
    if values.ndim != 2 or values.shape[1] != 3:
        raise VerticalError(f"{path} is not an hgeoHNOR2020 grid (expected 3 columns: lon lat FC)")
    lons = np.where(values[:, 0] > 180.0, values[:, 0] - 360.0, values[:, 0])
    lats = values[:, 1]
    unique_lons = np.unique(lons)
    unique_lats = np.unique(lats)
    grid = np.full((unique_lats.size, unique_lons.size), np.nan)
    grid[np.searchsorted(unique_lats, lats), np.searchsorted(unique_lons, lons)] = values[:, 2]
    return RegularGridInterpolator((unique_lats, unique_lons), grid, bounds_error=False, fill_value=np.nan)


def write_orthometric_dem(
    ellipsoidal_dem_path: Path,
    output_path: Path,
    project_epsg: int,
    datum: str = "imbituba",
    grid_path: Path | None = None,
) -> OrthometricResult:
    """Write a copy of an ellipsoidal DEM converted to normal heights
    (H = h - FC). NoData stays NoData; every valid cell must fall inside
    the model, never silently left ellipsoidal."""
    grid_path = Path(grid_path) if grid_path is not None else ensure_hgeohnor2020_grid(datum)
    interpolator = load_conversion_grid(grid_path)

    with rasterio.open(ellipsoidal_dem_path) as src:
        profile = src.profile
        heights = src.read(1).astype(np.float64)
        transform = src.transform
        nodata = src.nodata
        tags = src.tags()
    if tags.get(VERTICAL_REFERENCE_TAG, "").startswith("normal"):
        raise VerticalError(f"{ellipsoidal_dem_path} is already in normal heights")

    valid = np.isfinite(heights) if nodata is None else (heights != nodata) & np.isfinite(heights)
    rows, cols = np.nonzero(valid)
    x = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    y = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    lons, lats = GeodeticTransformer(wgs84(), CoordinateReferenceSystem(project_epsg)).inverse_xy(x, y)
    factor = interpolator(np.column_stack([lats, lons]))
    if np.isnan(factor).any():
        raise VerticalError(
            f"{int(np.isnan(factor).sum())} DEM cell(s) fall outside the hgeoHNOR2020 {datum} model; "
            "use the other datum (imbituba / santana) or check the project CRS"
        )

    converted = np.where(valid, heights, profile.get("nodata", -9999.0))
    converted[rows, cols] = heights[rows, cols] - factor

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(converted.astype(profile["dtype"]), 1)
        dst.update_tags(**{VERTICAL_REFERENCE_TAG: f"normal heights, hgeoHNOR2020 {datum} (H = h - FC)"})

    return OrthometricResult(
        raster_path=str(output_path),
        datum=datum,
        min_conversion_factor_m=float(factor.min()),
        max_conversion_factor_m=float(factor.max()),
        min_elevation_m=float(converted[rows, cols].min()),
        max_elevation_m=float(converted[rows, cols].max()),
    )
