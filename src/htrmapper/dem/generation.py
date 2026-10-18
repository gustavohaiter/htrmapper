"""Phase 5: DEM/DSM generation from the dense point cloud.

Rasterizes the Phase 4 dense point cloud into a georeferenced elevation
raster, per the project brief's requirements (automatic or user-defined
resolution, interpolation, hole filling, filtering, GeoTIFF export with
correct CRS/geotransform/NoData).

This is a Digital Surface Model (DSM), not a bare-earth DEM: the dense
point cloud comes from image-based multi-view stereo, so every point is
already a "first surface" sample (canopy, structures, ground) -- there is
no LiDAR ground-vs-non-ground return classification to separate here.
Ground classification/filtering to a true bare-earth DEM is listed as a
future capability in the project brief ("classificação de solo
futuramente") and is not implemented in this phase.

Built on GDAL (via `rasterio`) for the raster I/O, per the architecture
decision to never hand-roll GeoTIFF writing, and `scipy.interpolate` for
the interpolation -- no custom triangulation/rasterization algorithm is
implemented here either.

Rasterization (ARCHITECTURE.md section 28), modeled on what Pix4D and
Metashape document for their DSMs:

1. Each cell takes the MEDIAN elevation of the points falling in it. A
   wrong point inside a populated cell is simply outvoted, where the
   earlier linear interpolation of raw points let every blunder pull a
   triangle fan around it (2.8 m spikes survived on the benchmark).
2. Noise filter (Pix4D: "corrects the altitude of noisy points with the
   median altitude of the neighboring points"): a cell whose median
   deviates from the median of its 5x5 neighborhood by more than
   DESPIKE_ROBUST_SIGMAS robust sigmas (and at least DESPIKE_MIN_M) is
   discarded and refilled from its neighbors. Real relief (crop rows,
   ditches, slopes) is far inside that band and is left untouched -- this
   removes spikes, it does not smooth.
3. Holes are filled only where the surface was actually reconstructed
   around them (Metashape's default "Enabled" interpolation, not its
   "Extrapolated" mode): gaps up to COVERAGE_MARGIN_M wide and any hole
   fully enclosed by data. Outside that footprint the DEM is NoData,
   instead of nearest-neighbor values invented out to the bounding box
   (4,600 m2 of made-up terrain on the benchmark's uncovered field tip).
   The orthomosaic extends the DEM by itself where cameras still see the
   ground (see `ortho.orthomosaic`).
4. The grid is snapped to multiples of the resolution, so DEMs of the
   same area from different runs align cell for cell.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.interpolate import griddata

from htrmapper.geo.crs import CoordinateReferenceSystem

NODATA_VALUE = -9999.0

# Automatic cell size, as a multiple of the average point spacing. Each
# cell needs a few points for its median to mean anything (at 2x the
# spacing a cell holds ~4 points on average and only ~2% of cells come out
# empty). Pix4D defaults its DSM to 1x GSD and Metashape to the dense
# cloud's own spacing, but both interpolate from the raw points; a median
# per cell needs the slightly coarser grid. Measured on the benchmark in
# ARCHITECTURE.md section 28; always overridable (`DemConfig.resolution_m`).
AUTO_RESOLUTION_SPACING_FACTOR = 2.0

# Noise filter (see module docstring). The window covers 5x5 cells; the
# threshold is relative to the robust spread of cell-vs-neighborhood
# differences over the whole DEM, with a floor so a very clean cloud does
# not start shaving real micro-relief.
DESPIKE_WINDOW_CELLS = 5
DESPIKE_ROBUST_SIGMAS = 4.0
DESPIKE_MIN_M = 0.15
# Only small, isolated groups of deviating cells are spikes. A real narrow
# feature (a ditch, a road edge, a terrace) also deviates from a 5x5
# median along its edges, but it does so along a long connected line and
# is kept -- the same connected-component idea Metashape documents for its
# depth-map filtering. Measured: a 1 m wide, 50 cm high embankment at 0.5 m
# cells was erased entirely by the plain threshold and survives intact
# with this guard, while 99% of injected 2 m spikes are still removed.
DESPIKE_MAX_SPIKE_AREA_M2 = 1.0

# Gaps narrower than this (on each side of the data, and never less than
# COVERAGE_MARGIN_CELLS cells) are interpolated; anything beyond it that is
# not enclosed by data is NoData.
COVERAGE_MARGIN_M = 2.0
COVERAGE_MARGIN_CELLS = 2

# Empty-cell groups up to this many cells (the ordinary gaps between
# sparse points) take the mean of their neighbors; larger holes are
# interpolated linearly from their rim (see `_fill_holes`).
SMALL_GAP_MAX_CELLS = 4

# Outlier filter: median absolute deviation (MAD), not a fixed-percentile
# clip. A percentile clip always trims the same ~0.1% fraction of points
# regardless of how many true outliers exist -- a cluster of several
# correlated blunders (e.g. a matching error affecting a handful of
# points) mostly survives it, because percentile rank doesn't measure
# *how far* a point deviates from the bulk of the data, only its rank.
# MAD scales with actual deviation from the median (a robust center
# estimate, unlike the mean, which the outliers themselves would pull),
# so it correctly flags any number of anomalous points as long as they
# remain a minority. 1.4826 is the standard scale factor that makes MAD
# consistent with the standard deviation for normally-distributed data;
# a threshold of 6x that robust sigma is generous enough to keep genuine
# terrain relief while still catching gross blunders (order of meters,
# not the sub-meter relief a real DSM has).
OUTLIER_MAD_SCALE = 1.4826
OUTLIER_MAD_THRESHOLD = 6.0


class DemError(RuntimeError):
    """Raised when DEM generation cannot proceed at all (missing/empty
    point cloud). A low-quality result (large holes, coarse resolution)
    is never an error -- it is reported in DemResult."""


@dataclass
class DemConfig:
    resolution_m: float | None = None  # None = automatic, from point density
    filter_outliers: bool = True

    def __post_init__(self) -> None:
        if self.resolution_m is not None and self.resolution_m <= 0:
            raise ValueError(f"resolution_m must be positive, got {self.resolution_m}")


@dataclass
class DemResult:
    raster_path: str = ""
    resolution_m: float = 0.0
    resolution_source: str = ""  # "automatic" or "user-defined"
    width_px: int = 0
    height_px: int = 0
    min_elevation_m: float = 0.0
    max_elevation_m: float = 0.0
    num_points_used: int = 0
    num_points_filtered_as_outliers: int = 0
    point_density_per_m2: float = 0.0
    num_cells_despiked: int = 0
    num_cells_interpolated: int = 0
    nodata_fraction: float = 0.0


def _load_points(las_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    if x.size == 0:
        raise DemError(f"point cloud at {las_path} has zero points")
    return x, y, z


def _filter_outliers(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    median_z = np.median(z)
    mad = np.median(np.abs(z - median_z))
    robust_sigma = OUTLIER_MAD_SCALE * mad
    if robust_sigma == 0:
        # All points share (near-)identical elevation -- nothing to filter.
        return x, y, z, 0
    keep = np.abs(z - median_z) <= OUTLIER_MAD_THRESHOLD * robust_sigma
    num_removed = int((~keep).sum())
    return x[keep], y[keep], z[keep], num_removed


def _auto_resolution(x: np.ndarray, y: np.ndarray) -> float:
    area_m2 = (x.max() - x.min()) * (y.max() - y.min())
    if area_m2 <= 0 or len(x) == 0:
        raise DemError("point cloud has zero spatial extent; cannot determine an automatic resolution")
    avg_spacing_m = np.sqrt(area_m2 / len(x))
    return avg_spacing_m * AUTO_RESOLUTION_SPACING_FACTOR


def _cell_medians(cols: np.ndarray, rows: np.ndarray, z: np.ndarray, width: int, height: int) -> np.ndarray:
    """(height, width) grid of per-cell median elevation, NaN where empty."""
    flat = rows * width + cols
    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    z_sorted = z[order]
    cells, starts, counts = np.unique(flat_sorted, return_index=True, return_counts=True)
    lower = starts + (counts - 1) // 2
    upper = starts + counts // 2
    grid = np.full(width * height, np.nan)
    grid[cells] = 0.5 * (z_sorted[lower] + z_sorted[upper])
    return grid.reshape(height, width)


def _fill_from_neighbors(grid: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Fill NaN cells inside `region` ring by ring with the mean of their
    already-known 3x3 neighbors (normalized convolution). Known cells are
    never changed."""
    filled = grid.copy()
    kernel = np.ones((3, 3))
    todo = region & np.isnan(filled)
    while todo.any():
        known = np.isfinite(filled)
        sums = ndimage.convolve(np.where(known, filled, 0.0), kernel, mode="constant")
        counts = ndimage.convolve(known.astype(float), kernel, mode="constant")
        ready = todo & (counts > 0)
        if not ready.any():
            break  # cells in `region` with no known cell anywhere around them
        filled[ready] = sums[ready] / counts[ready]
        todo &= ~ready
    return filled


def _fill_holes(grid: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Interpolate every NaN cell inside `region`. Small gaps take the mean
    of their neighbors; larger holes are interpolated linearly across from
    the populated cells around their rim. Growing a large hole inward from
    one side instead biases it by the local slope at every ring (23.7 cm
    RMSE inside the benchmark's 1,000 m2 hole, 13.4 cm linearly)."""
    filled = grid.copy()
    empty = region & np.isnan(filled)
    if not empty.any():
        return filled
    labels, _ = ndimage.label(empty, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    small = sizes <= SMALL_GAP_MAX_CELLS
    small[0] = False
    large_holes = empty & ~small[labels]
    filled = _fill_from_neighbors(filled, empty & small[labels])
    if large_holes.any():
        rim = np.isfinite(filled) & ndimage.binary_dilation(large_holes, structure=np.ones((3, 3)))
        rim_rows, rim_cols = np.nonzero(rim)
        hole_rows, hole_cols = np.nonzero(large_holes)
        if rim_rows.size >= 3:
            filled[hole_rows, hole_cols] = griddata(
                (rim_rows, rim_cols), filled[rim_rows, rim_cols], (hole_rows, hole_cols), method="linear"
            )
        # Whatever linear interpolation cannot reach (outside the rim's
        # convex hull, or a degenerate rim) falls back to neighbor growth.
        filled = _fill_from_neighbors(filled, region)
    return filled

def _coverage(data: np.ndarray, resolution_m: float) -> np.ndarray:
    """Cells where the surface counts as reconstructed: within the coverage
    margin of a populated cell, plus every hole enclosed by them."""
    distance_m = ndimage.distance_transform_edt(~data) * resolution_m
    margin_m = max(COVERAGE_MARGIN_M, COVERAGE_MARGIN_CELLS * resolution_m)
    return ndimage.binary_fill_holes(distance_m <= margin_m)


def _despike(grid: np.ndarray, coverage: np.ndarray, resolution_m: float) -> np.ndarray:
    """Boolean mask of populated cells that are spikes (module docstring)."""
    data = np.isfinite(grid)
    complete = _fill_from_neighbors(grid, coverage)
    # The median filter needs a value everywhere; outside the coverage use
    # the nearest covered value, so the edges are judged against real data.
    outside = ~np.isfinite(complete)
    if outside.any():
        nearest = ndimage.distance_transform_edt(outside, return_distances=False, return_indices=True)
        complete = complete[tuple(nearest)]
    neighborhood = ndimage.median_filter(complete, size=DESPIKE_WINDOW_CELLS, mode="nearest")
    deviation = np.abs(grid - neighborhood)
    robust_sigma = OUTLIER_MAD_SCALE * float(np.median(deviation[data])) if data.any() else 0.0
    threshold = max(DESPIKE_ROBUST_SIGMAS * robust_sigma, DESPIKE_MIN_M)
    candidates = data & (deviation > threshold)
    labels, num_labels = ndimage.label(candidates, structure=np.ones((3, 3)))
    if num_labels == 0:
        return candidates
    sizes = np.bincount(labels.ravel())
    max_cells = max(1, int(DESPIKE_MAX_SPIKE_AREA_M2 / resolution_m**2))
    small = sizes <= max_cells
    small[0] = False
    return small[labels]


def run_dem_generation(
    las_path: Path,
    project_epsg: int,
    output_path: Path,
    config: DemConfig | None = None,
    cancellation_token: Any = None,
    progress_callback: "Callable[[str], None] | None" = None,
) -> DemResult:
    """Rasterize a dense point cloud (LAS) into a georeferenced DSM GeoTIFF.

    Unlike the COLMAP-backed phases, there is no native cancellation hook
    here (this is plain numpy/scipy/rasterio code) -- `cancellation_token`
    (anything exposing an `is_cancelled` property, so the same
    `pycolmap.CancellationToken` used elsewhere can be reused as a simple
    thread-safe flag) is only checked between the phases below, never
    mid-computation, and raises `InterruptedError` when set, matching
    pycolmap's own convention for a cancelled operation.
    `progress_callback`, when given, is called with a short phase label
    before each phase -- real phase-boundary progress, not a fabricated
    smooth percentage (this pipeline has no meaningful finer granularity).
    """
    config = config or DemConfig()
    las_path = Path(las_path)
    if not las_path.exists():
        raise DemError(f"point cloud not found at {las_path}")

    def _report(phase: str) -> None:
        if cancellation_token is not None and cancellation_token.is_cancelled:
            raise InterruptedError("Operation cancelled")
        if progress_callback is not None:
            progress_callback(phase)

    _report("Carregando nuvem de pontos (LAS)")
    x, y, z = _load_points(las_path)
    num_points_input = len(x)

    num_filtered = 0
    if config.filter_outliers:
        _report("Filtrando outliers (mediana + MAD)")
        x, y, z, num_filtered = _filter_outliers(x, y, z)
        if len(x) < 3:
            raise DemError(
                f"outlier filtering left only {len(x)} point(s) (of {num_points_input}); "
                "cannot interpolate a surface from fewer than 3 points"
            )

    if config.resolution_m is not None:
        resolution_m = config.resolution_m
        resolution_source = "user-defined"
    else:
        resolution_m = _auto_resolution(x, y)
        resolution_source = "automatic"

    # Grid snapped to multiples of the resolution (module docstring, item 4).
    min_x = np.floor(x.min() / resolution_m) * resolution_m
    max_y = np.ceil(y.max() / resolution_m) * resolution_m
    width_px = max(1, int(np.floor((x.max() - min_x) / resolution_m)) + 1)
    height_px = max(1, int(np.floor((max_y - y.min()) / resolution_m)) + 1)
    cols = np.clip(((x - min_x) / resolution_m).astype(np.int64), 0, width_px - 1)
    rows = np.clip(((max_y - y) / resolution_m).astype(np.int64), 0, height_px - 1)

    _report("Rasterizando (mediana por célula)")
    grid = _cell_medians(cols, rows, z, width_px, height_px)
    coverage = _coverage(np.isfinite(grid), resolution_m)

    _report("Removendo picos (filtro de ruído)")
    spikes = _despike(grid, coverage, resolution_m)
    grid[spikes] = np.nan

    _report("Preenchendo buracos (interpolação)")
    data_cells = np.isfinite(grid)
    grid = _fill_holes(grid, coverage)
    grid[~coverage] = np.nan
    num_interpolated = int((coverage & ~data_cells).sum())

    transform = from_origin(min_x, max_y, resolution_m, resolution_m)
    crs_wkt = CoordinateReferenceSystem(project_epsg).to_wkt()

    _report("Escrevendo GeoTIFF")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height_px,
        width=width_px,
        count=1,
        dtype=rasterio.float32,
        crs=crs_wkt,
        transform=transform,
        nodata=NODATA_VALUE,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="DEFLATE",
        predictor=3,
    ) as dst:
        dst.write(np.where(np.isfinite(grid), grid, NODATA_VALUE).astype(np.float32), 1)

    valid_z = grid[np.isfinite(grid)]
    area_m2 = (x.max() - x.min()) * (y.max() - y.min())
    point_density = len(x) / area_m2 if area_m2 > 0 else 0.0

    return DemResult(
        raster_path=str(output_path),
        resolution_m=resolution_m,
        resolution_source=resolution_source,
        width_px=width_px,
        height_px=height_px,
        min_elevation_m=float(valid_z.min()),
        max_elevation_m=float(valid_z.max()),
        num_points_used=len(x),
        num_points_filtered_as_outliers=num_filtered,
        point_density_per_m2=point_density,
        num_cells_despiked=int(spikes.sum()),
        num_cells_interpolated=num_interpolated,
        nodata_fraction=float(1.0 - coverage.mean()),
    )
