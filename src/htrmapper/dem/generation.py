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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata

from htrmapper.geo.crs import CoordinateReferenceSystem

NODATA_VALUE = -9999.0

# Heuristic, not a physical law: a DEM cell needs support from multiple
# dense-cloud points to interpolate reliably, so the automatic resolution
# is set coarser than the raw average point spacing. A factor of ~2.5x
# the average nearest-neighbor spacing is common practice in aerial
# photogrammetry (DEM/orthomosaic resolution is typically recommended at
# 2-4x the source GSD/point spacing) -- documented here as a heuristic,
# not asserted as an exact optimum, and always overridable by the user
# (`DemConfig.resolution_m`).
AUTO_RESOLUTION_SPACING_FACTOR = 2.5

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


def run_dem_generation(
    las_path: Path,
    project_epsg: int,
    output_path: Path,
    config: DemConfig | None = None,
) -> DemResult:
    """Rasterize a dense point cloud (LAS) into a georeferenced DSM GeoTIFF."""
    config = config or DemConfig()
    las_path = Path(las_path)
    if not las_path.exists():
        raise DemError(f"point cloud not found at {las_path}")

    x, y, z = _load_points(las_path)
    num_points_input = len(x)

    num_filtered = 0
    if config.filter_outliers:
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

    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()
    width_px = max(1, int(np.ceil((max_x - min_x) / resolution_m)))
    height_px = max(1, int(np.ceil((max_y - min_y) / resolution_m)))

    # Raster cell centers, row 0 = north (top), per the standard GeoTIFF
    # north-up convention.
    col_centers = min_x + (np.arange(width_px) + 0.5) * resolution_m
    row_centers = max_y - (np.arange(height_px) + 0.5) * resolution_m
    grid_x, grid_y = np.meshgrid(col_centers, row_centers)

    # Linear interpolation inside the point cloud's convex hull, then
    # nearest-neighbor to fill the remaining holes (outside the hull, or
    # sparse gaps linear interpolation can't reach) -- the "interpolação
    # + preenchimento de buracos" the project brief asks for, without
    # extrapolating with an unconstrained method.
    grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")
    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z[nan_mask] = griddata((x, y), z, (grid_x[nan_mask], grid_y[nan_mask]), method="nearest")

    transform = from_origin(min_x, max_y, resolution_m, resolution_m)
    crs_wkt = CoordinateReferenceSystem(project_epsg).to_wkt()

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
    ) as dst:
        dst.write(grid_z.astype(np.float32), 1)

    area_m2 = (max_x - min_x) * (max_y - min_y)
    point_density = len(x) / area_m2 if area_m2 > 0 else 0.0

    return DemResult(
        raster_path=str(output_path),
        resolution_m=resolution_m,
        resolution_source=resolution_source,
        width_px=width_px,
        height_px=height_px,
        min_elevation_m=float(z.min()),
        max_elevation_m=float(z.max()),
        num_points_used=len(x),
        num_points_filtered_as_outliers=num_filtered,
        point_density_per_m2=point_density,
    )
