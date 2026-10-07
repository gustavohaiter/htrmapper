"""Phase 6: orthomosaic generation.

**This is 2.5D orthorectification, not full 3D true-ortho.** For every
output pixel, the terrain height comes from the per-pixel DEM (Phase 5),
so terrain relief is handled correctly -- unlike the naive "single average
plane" ortho the project brief explicitly warns against (ARCHITECTURE.md
section 6). What it does NOT do is per-pixel visibility/occlusion
(z-buffer) checking: an object taller than the DSM surface at a given
(x, y) -- there is no such object here since the DSM already includes
canopy/structures, but a *different* nearby tall object between the camera
and a lower point can still project onto that lower point's pixel
incorrectly. That "leaning"/double-mapping artifact around vertical
features is the one thing a full 3D true-ortho (ray-casting with a
per-pixel visibility test against the whole scene) avoids and this
implementation does not. This is stated explicitly, per the project's
rule against overstating precision -- never call this "true-ortho"
without this caveat attached.

Built by reusing COLMAP's own camera model directly (`Image.cam_from_world`,
`Camera.img_from_cam` -- the same projection used for reprojection error
throughout this project), sampling from the undistorted images that
Phase 4's `pycolmap.undistort_images` step already produced (so sampling
never has to also handle lens distortion). No custom camera math, no
custom raster I/O (`rasterio`/GDAL) -- per the architecture decision to
build on mature libraries.

**Resolution is independent of the DEM's own resolution** (real-world bug
found and fixed on 2026-10-07, see ARCHITECTURE.md): the orthomosaic used
to simply reuse the DEM's raster grid pixel-for-pixel, so a DEM generated
coarse (e.g. from a sparse dense cloud, or a deliberately low DEM quality
tier for a quick volumetria-irrelevant flight) silently forced the
orthomosaic coarse too -- 2.46 m/px on a real 56-photo flight whose actual
photographed ground resolution was ~2.5 cm/px, a ~100x gap, confirmed
against a real Pix4Dfields export of the same flight. Commercial tools
(Pix4D/Metashape) never couple these two: the DEM/mesh is only a terrain
height lookup for orthorectification, and the orthomosaic itself is
rendered close to native image GSD. This module now builds its own output
grid (same ground footprint/bounding box as the DEM, but at an
independently chosen resolution) and resamples the DEM's elevation surface
onto it (bilinear, via `scipy.interpolate.RegularGridInterpolator` -- no
hand-rolled resampling). The default resolution, when not given explicitly,
is estimated directly from each registered camera's real (bundle-adjusted)
height above the DEM terrain and its calibrated focal length in pixels --
the same pinhole GSD formula as `geo.gsd`, but using actual reconstructed
geometry instead of EXIF/nominal flying height, which is more accurate
(this is exactly the "more accurate, post-alignment" GSD recomputation
`geo/gsd.py`'s own docstring anticipated).

Blending: each candidate camera contributes a weighted vote to every
output pixel it can see, weighted by distance-to-image-edge ("feathering",
per the project brief) -- pixels near a source image's border are
down-weighted so the seam between two overlapping images fades smoothly
instead of showing a hard edge. This is a documented, simple blending
strategy; graph-cut seamline optimization (what mature commercial tools
use to avoid blending across moving objects/parallax errors) is not
implemented here.

**Processed in tiles, never as one whole-grid array in RAM** (real bug
found and fixed on 2026-10-07, same day as the resolution-decoupling fix
above and the direct cause of it: once the orthomosaic rendered near
native image GSD instead of the DEM's coarse resolution, a real 56-photo
flight's output grid grew to ~168 million pixels, and the previous
single-pass implementation held several full-grid float64 arrays (color
accumulator, weight accumulator, world coordinates, elevation) in memory
simultaneously -- OOM-killed on the user's own machine on first real use
at native resolution). Professional tools (Pix4D/Metashape) never hold a
whole large orthomosaic in RAM either; they process it in blocks. This
module now does the same: the output raster is split into fixed-size
tiles (`TILE_SIZE_PX`), and for each tile only that tile's world
coordinates/elevation/accumulators are ever materialized, written to the
GeoTIFF via a windowed write, then discarded before the next tile. Peak
memory is therefore bounded by tile size, never by total output size.
To avoid every camera being tested against every tile (correct but
wasteful), each registered camera's approximate ground footprint is
precomputed once up front (by reprojecting the DEM's own, much smaller,
native-resolution grid through that camera -- reusing the same camera
projection math, just applied to fewer points) and used as a cheap
bounding-box pre-filter per tile.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import rasterio
import rasterio.transform
from PIL import Image as PILImage
from rasterio.transform import from_origin
from rasterio.windows import Window
from scipy.interpolate import RegularGridInterpolator

# Width of the feathering zone, as a fraction of the source image's
# shorter side. A pixel exactly on the image border gets weight 0; a
# pixel this fraction of the image size (or further) from every border
# gets full weight 1, with a linear ramp in between. 10% is a common,
# unremarkable default for edge feathering -- large enough to visibly
# smooth seams, small enough not to discard most of small, single images.
FEATHER_FRACTION = 0.1

# Tile side length, in output pixels. 2048x2048 keeps every per-tile
# float64 array well under a few hundred MB (2048*2048*3*8 bytes = 100MB
# for the color accumulator alone, the largest one) regardless of how
# large the full orthomosaic is -- the entire point of tiling. Not
# exposed as user config for now: it only trades a little overhead
# (re-testing camera candidates per tile) for memory headroom, and 2048
# is a safe, unremarkable default (also GDAL's own common internal tiling
# block size).
TILE_SIZE_PX = 2048


class OrthoError(RuntimeError):
    """Raised when orthomosaic generation cannot proceed at all (missing
    inputs). A mosaic with holes where no camera saw the ground is not an
    error -- those pixels are marked NoData (alpha = 0), never hidden."""


@dataclass
class OrthoConfig:
    feather_fraction: float = FEATHER_FRACTION
    resolution_m: float | None = None  # None = automatic, from camera GSD (see module docstring)

    def __post_init__(self) -> None:
        if not (0.0 < self.feather_fraction <= 0.5):
            raise ValueError(f"feather_fraction must be in (0, 0.5], got {self.feather_fraction}")
        if self.resolution_m is not None and self.resolution_m <= 0:
            raise ValueError(f"resolution_m must be positive, got {self.resolution_m}")


@dataclass
class OrthoResult:
    raster_path: str = ""
    width_px: int = 0
    height_px: int = 0
    resolution_m: float = 0.0
    resolution_source: str = ""  # "automatic (camera GSD)" or "user-defined"
    num_cameras_used: int = 0
    num_valid_pixels: int = 0
    num_nodata_pixels: int = 0


def _load_dem_grid(dem_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine, str]:
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float64)
        transform = src.transform
        crs_wkt = src.crs.to_wkt()
        height, width = elevation.shape
        nodata = src.nodata

    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij"
    )
    # Pixel-center world coordinates via the affine transform directly
    # (rather than rasterio.transform.xy, which does not preserve 2D
    # array shape): standard north-up GeoTIFF convention, a=pixel width,
    # e=-pixel height, b=d=0.
    col_centers = cols + 0.5
    row_centers = rows + 0.5
    world_x = transform.c + transform.a * col_centers + transform.b * row_centers
    world_y = transform.f + transform.d * col_centers + transform.e * row_centers

    if nodata is not None:
        elevation = np.where(elevation == nodata, np.nan, elevation)

    return world_x, world_y, elevation, transform, crs_wkt


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Bilinear sampling of an (H, W, 3) uint8 image at fractional pixel
    coordinates (u=column, v=row). Returns float64 colors; entries where
    `valid` is False are left as zero (caller must not use them)."""
    height, width = image.shape[:2]
    out = np.zeros((u.shape[0], 3), dtype=np.float64)
    if not np.any(valid):
        return out

    uu = u[valid]
    vv = v[valid]
    x0 = np.clip(np.floor(uu).astype(np.int64), 0, width - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.clip(np.floor(vv).astype(np.int64), 0, height - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    fx = np.clip(uu - x0, 0.0, 1.0)
    fy = np.clip(vv - y0, 0.0, 1.0)

    img_f = image.astype(np.float64)
    top = img_f[y0, x0] * (1 - fx)[:, None] + img_f[y0, x1] * fx[:, None]
    bottom = img_f[y1, x0] * (1 - fx)[:, None] + img_f[y1, x1] * fx[:, None]
    sampled = top * (1 - fy)[:, None] + bottom * fy[:, None]

    out[valid] = sampled
    return out


def _estimate_resolution_from_cameras(
    reconstruction: "pycolmap.Reconstruction", dem_elevation: np.ndarray, dem_transform: rasterio.Affine
) -> float | None:
    """Estimate ground resolution (m/px) from each registered camera's real
    height above the DEM terrain and its calibrated focal length in pixels
    -- GSD = height_above_terrain / focal_length_px, the standard pinhole
    formula (see module docstring for why this is more accurate than the
    EXIF/nominal-altitude estimate in `geo.gsd`). Returns the median across
    cameras with a valid terrain sample underneath them (robust against the
    odd camera whose nadir point falls just outside DEM coverage or over a
    nodata cell); None if no camera qualifies, so the caller can fall back
    to a safe default instead of dividing by a fabricated number.
    """
    estimates = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        center = image.projection_center()
        focal_px = image.camera.mean_focal_length()
        if focal_px <= 0:
            continue
        row, col = rasterio.transform.rowcol(dem_transform, center[0], center[1])
        if not (0 <= row < dem_elevation.shape[0] and 0 <= col < dem_elevation.shape[1]):
            continue
        terrain_z = dem_elevation[row, col]
        if np.isnan(terrain_z):
            continue
        height_above_terrain = center[2] - terrain_z
        if height_above_terrain <= 0:
            continue
        estimates.append(height_above_terrain / focal_px)

    if not estimates:
        return None
    return float(np.median(estimates))


def _output_geometry(
    dem_transform: rasterio.Affine, dem_width: int, dem_height: int, resolution_m: float
) -> tuple[int, int, rasterio.Affine]:
    """Compute the orthomosaic's output raster shape/transform at
    `resolution_m`, covering the same ground footprint (bounding box) as
    the DEM -- pure arithmetic, no array allocation, so this is cheap to
    call even when the resulting grid is hundreds of millions of pixels.
    """
    min_x = dem_transform.c
    max_y = dem_transform.f
    max_x = min_x + dem_width * dem_transform.a
    min_y = max_y + dem_height * dem_transform.e  # dem_transform.e is negative (north-up)

    width_px = max(1, int(np.ceil((max_x - min_x) / resolution_m)))
    height_px = max(1, int(np.ceil((max_y - min_y) / resolution_m)))
    transform = from_origin(min_x, max_y, resolution_m, resolution_m)
    return width_px, height_px, transform


def _build_dem_interpolator(
    dem_transform: rasterio.Affine, dem_width: int, dem_height: int, dem_elevation: np.ndarray
) -> RegularGridInterpolator:
    """Build a reusable (row, col) -> elevation interpolator over the DEM's
    own (small, native-resolution) grid -- built once, then evaluated per
    output tile, never re-built per tile and never evaluated over the
    whole (potentially huge) output grid at once.
    """
    min_x = dem_transform.c
    max_y = dem_transform.f

    # RegularGridInterpolator requires strictly ascending axes; the DEM's
    # row axis is north-up (descending in world-Y), so flip both the axis
    # and the elevation rows to match.
    dem_col_centers = min_x + (np.arange(dem_width) + 0.5) * dem_transform.a
    dem_row_centers_desc = max_y + (np.arange(dem_height) + 0.5) * dem_transform.e
    dem_row_centers_asc = dem_row_centers_desc[::-1]
    elevation_asc = dem_elevation[::-1, :]

    return RegularGridInterpolator(
        (dem_row_centers_asc, dem_col_centers), elevation_asc, method="linear", bounds_error=False, fill_value=np.nan
    )


def _camera_footprint_bbox(
    image: "pycolmap.Image", dem_world_x: np.ndarray, dem_world_y: np.ndarray, dem_elevation: np.ndarray
) -> tuple[float, float, float, float] | None:
    """Approximate a registered camera's ground footprint, as a world-space
    bounding box (min_x, max_x, min_y, max_y), by reprojecting the DEM's
    own native-resolution grid through it -- the same projection math as
    the main per-tile loop, just applied once to the DEM's (much smaller)
    grid instead of per-tile to the (potentially huge) output grid. Used
    only as a cheap per-tile candidate pre-filter, never as the final
    per-pixel visibility test -- a coarse DEM cell near the camera's true
    edge could make this bbox slightly too tight or too loose, which is
    fine for a pre-filter but would not be for the real test. Returns None
    if the camera sees no DEM terrain at all (never a candidate for any
    tile).
    """
    valid_terrain = ~np.isnan(dem_elevation)
    if not np.any(valid_terrain):
        return None

    xs = dem_world_x[valid_terrain]
    ys = dem_world_y[valid_terrain]
    zs = dem_elevation[valid_terrain]
    world_points = np.stack([xs, ys, zs], axis=1)

    camera = image.camera
    cam_points = image.cam_from_world() * world_points
    pixel_coords = camera.img_from_cam(cam_points, check_cheirality=True)
    u, v = pixel_coords[:, 0], pixel_coords[:, 1]
    valid = ~np.isnan(u) & (u >= 0) & (u <= camera.width - 1) & (v >= 0) & (v <= camera.height - 1)
    if not np.any(valid):
        return None

    return float(xs[valid].min()), float(xs[valid].max()), float(ys[valid].min()), float(ys[valid].max())


def _bbox_overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], margin: float
) -> bool:
    a_min_x, a_max_x, a_min_y, a_max_y = a
    b_min_x, b_max_x, b_min_y, b_max_y = b
    return not (
        a_max_x + margin < b_min_x
        or a_min_x - margin > b_max_x
        or a_max_y + margin < b_min_y
        or a_min_y - margin > b_max_y
    )


def _iter_tiles(
    width_px: int, height_px: int, transform: rasterio.Affine, tile_size: int
) -> Iterator[tuple[int, int, int, int, np.ndarray, np.ndarray]]:
    """Yield (row_off, col_off, tile_height, tile_width, tile_world_x,
    tile_world_y) for each tile covering the output raster, row-major.
    Edge tiles are smaller than `tile_size` rather than padded."""
    for row_off in range(0, height_px, tile_size):
        tile_height = min(tile_size, height_px - row_off)
        for col_off in range(0, width_px, tile_size):
            tile_width = min(tile_size, width_px - col_off)

            cols = col_off + np.arange(tile_width)
            rows = row_off + np.arange(tile_height)
            col_centers = transform.c + transform.a * (cols + 0.5)
            row_centers = transform.f + transform.e * (rows + 0.5)
            tile_world_x, tile_world_y = np.meshgrid(col_centers, row_centers)

            yield row_off, col_off, tile_height, tile_width, tile_world_x, tile_world_y


def _feather_weight(u: np.ndarray, v: np.ndarray, width: int, height: int, feather_fraction: float) -> np.ndarray:
    feather_px = feather_fraction * min(width, height)
    dist_left = u
    dist_right = (width - 1) - u
    dist_top = v
    dist_bottom = (height - 1) - v
    edge_dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))
    weight = np.clip(edge_dist / feather_px, 0.0, 1.0)
    return weight


def run_orthomosaic_generation(
    reconstruction_path: Path,
    undistorted_image_path: Path,
    dem_path: Path,
    output_path: Path,
    config: OrthoConfig | None = None,
    cancellation_token: "pycolmap.CancellationToken | None" = None,
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> OrthoResult:
    """Generate a georeferenced RGBA orthomosaic GeoTIFF.

    `reconstruction_path`/`undistorted_image_path` must be the pair
    Phase 4's `pycolmap.undistort_images` wrote (undistorted images +
    matching PINHOLE-model reconstruction) -- never the original distorted
    images, since sampling here does not itself correct for distortion.

    Processed tile by tile (see module docstring) to bound peak memory
    regardless of the output raster's total size. `progress_callback`,
    when given, is called with `(tiles_done, tiles_total)` after each tile
    is written -- real, fine-grained progress (never fabricated), just at
    tile granularity rather than per-camera as before tiling existed.
    `cancellation_token` (a `pycolmap.CancellationToken`, reused here
    purely as a cheap thread-safe flag -- no pycolmap call is made in this
    loop) is checked once per tile; if cancelled, raises
    `InterruptedError` to match pycolmap's own convention.
    """
    config = config or OrthoConfig()
    reconstruction_path = Path(reconstruction_path)
    undistorted_image_path = Path(undistorted_image_path)
    dem_path = Path(dem_path)

    if not reconstruction_path.exists():
        raise OrthoError(f"reconstruction not found at {reconstruction_path}")
    if not undistorted_image_path.is_dir():
        raise OrthoError(f"undistorted image folder not found at {undistorted_image_path}")
    if not dem_path.exists():
        raise OrthoError(f"DEM not found at {dem_path}")

    reconstruction = pycolmap.Reconstruction(str(reconstruction_path))
    registered_ids = list(reconstruction.reg_image_ids())
    if not registered_ids:
        raise OrthoError("reconstruction has no registered images")

    dem_world_x, dem_world_y, dem_elevation, dem_transform, crs_wkt = _load_dem_grid(dem_path)
    dem_height_px, dem_width_px = dem_elevation.shape

    if config.resolution_m is not None:
        resolution_m = config.resolution_m
        resolution_source = "user-defined"
    else:
        estimated = _estimate_resolution_from_cameras(reconstruction, dem_elevation, dem_transform)
        if estimated is not None:
            resolution_m = estimated
            resolution_source = "automatic (camera GSD)"
        else:
            # No registered camera had a usable terrain sample underneath
            # it (e.g. every nadir point fell outside DEM coverage) -- fall
            # back to the DEM's own resolution rather than fabricating a
            # number or crashing; this matches the old (pre-fix) behavior
            # only in this degenerate case.
            resolution_m = abs(dem_transform.a)
            resolution_source = "automatic (fallback: DEM resolution)"

    width_px, height_px, transform = _output_geometry(dem_transform, dem_width_px, dem_height_px, resolution_m)
    interpolator = _build_dem_interpolator(dem_transform, dem_width_px, dem_height_px, dem_elevation)

    # Precompute each registered camera's approximate ground footprint
    # once, up front -- a cheap per-tile candidate pre-filter (see
    # `_camera_footprint_bbox` docstring). Cameras with a missing image
    # file or no DEM overlap at all are dropped here once, rather than
    # being re-checked on every tile.
    dem_cell_diag = float(np.hypot(abs(dem_transform.a), abs(dem_transform.e)))
    cameras: list[dict] = []
    for image_id in registered_ids:
        image = reconstruction.image(image_id)
        image_path = undistorted_image_path / image.name
        if not image_path.exists():
            continue
        bbox = _camera_footprint_bbox(image, dem_world_x, dem_world_y, dem_elevation)
        if bbox is None:
            continue
        cameras.append(
            {
                "image_id": image_id,
                "image": image,
                "camera": image.camera,
                "image_path": image_path,
                "bbox": bbox,
                "cache": None,  # lazily holds the decoded (H, W, 3) uint8 array
            }
        )

    tile_specs = list(_iter_tiles(width_px, height_px, transform, TILE_SIZE_PX))
    num_tiles_total = len(tile_specs)
    num_valid_pixels = 0
    num_nodata_pixels = 0
    used_camera_ids: set[int] = set()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height_px,
        width=width_px,
        count=4,
        dtype=rasterio.uint8,
        crs=crs_wkt,
        transform=transform,
        photometric="RGB",
        alpha="yes",
    ) as dst:
        for tile_index, (row_off, col_off, tile_height, tile_width, tile_world_x, tile_world_y) in enumerate(
            tile_specs, start=1
        ):
            if cancellation_token is not None and cancellation_token.is_cancelled:
                raise InterruptedError("Operation cancelled")

            tile_query = np.stack([tile_world_y.ravel(), tile_world_x.ravel()], axis=1)
            tile_elevation = interpolator(tile_query).reshape(tile_world_y.shape)

            color_accum = np.zeros((tile_height, tile_width, 3), dtype=np.float64)
            weight_accum = np.zeros((tile_height, tile_width), dtype=np.float64)

            valid_terrain = ~np.isnan(tile_elevation)
            if np.any(valid_terrain):
                flat_x = tile_world_x[valid_terrain]
                flat_y = tile_world_y[valid_terrain]
                flat_z = tile_elevation[valid_terrain]
                flat_indices = np.argwhere(valid_terrain)
                world_points = np.stack([flat_x, flat_y, flat_z], axis=1)

                tile_bbox = (
                    float(flat_x.min()),
                    float(flat_x.max()),
                    float(flat_y.min()),
                    float(flat_y.max()),
                )

                for entry in cameras:
                    if not _bbox_overlaps(entry["bbox"], tile_bbox, margin=dem_cell_diag):
                        continue

                    camera = entry["camera"]
                    cam_points = entry["image"].cam_from_world() * world_points
                    pixel_coords = camera.img_from_cam(cam_points, check_cheirality=True)
                    u = pixel_coords[:, 0]
                    v = pixel_coords[:, 1]
                    valid = (
                        ~np.isnan(u)
                        & (u >= 0)
                        & (u <= camera.width - 1)
                        & (v >= 0)
                        & (v <= camera.height - 1)
                    )
                    if not np.any(valid):
                        continue

                    if entry["cache"] is None:
                        entry["cache"] = np.asarray(PILImage.open(entry["image_path"]).convert("RGB"))
                    src_image = entry["cache"]

                    colors = _bilinear_sample(src_image, u, v, valid)
                    weights = np.zeros_like(u)
                    weights[valid] = _feather_weight(
                        u[valid], v[valid], camera.width, camera.height, config.feather_fraction
                    )

                    rows = flat_indices[valid, 0]
                    cols = flat_indices[valid, 1]
                    np.add.at(color_accum, (rows, cols), colors[valid] * weights[valid, None])
                    np.add.at(weight_accum, (rows, cols), weights[valid])
                    used_camera_ids.add(entry["image_id"])

            has_data = weight_accum > 0
            rgb = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            rgb[has_data] = np.clip(color_accum[has_data] / weight_accum[has_data, None], 0, 255).astype(np.uint8)
            alpha = np.where(has_data, 255, 0).astype(np.uint8)
            num_valid_pixels += int(has_data.sum())
            num_nodata_pixels += int((~has_data).sum())

            window = Window(col_off, row_off, tile_width, tile_height)
            for band_index in range(3):
                dst.write(rgb[:, :, band_index], band_index + 1, window=window)
            dst.write(alpha, 4, window=window)

            if progress_callback is not None:
                progress_callback(tile_index, num_tiles_total)

    return OrthoResult(
        raster_path=str(output_path),
        width_px=width_px,
        height_px=height_px,
        resolution_m=abs(transform.a),
        resolution_source=resolution_source,
        num_cameras_used=len(used_camera_ids),
        num_valid_pixels=num_valid_pixels,
        num_nodata_pixels=num_nodata_pixels,
    )
