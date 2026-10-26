"""Phase 6: orthomosaic generation.

**This is 2.5D orthorectification, not full 3D true-ortho.** For every
output pixel, the terrain height comes from the per-pixel DEM (Phase 5),
so terrain relief is handled correctly -- unlike the naive "single average
plane" ortho the project brief explicitly warns against (ARCHITECTURE.md
section 6). What it does NOT do is per-pixel visibility/occlusion
(z-buffer) checking: a nearby tall object between the camera and a lower
point can still project onto that lower point's pixel incorrectly. That
"leaning"/double-mapping artifact around vertical features is the one
thing a full 3D true-ortho (ray-casting with a per-pixel visibility test
against the whole scene) avoids and this implementation does not. Choosing
the most-nadir view per pixel (below) keeps that artifact as small as the
flight allows, but does not eliminate it -- never call this "true-ortho".

Built by reusing COLMAP's own camera model directly (`Image.cam_from_world`,
`Camera.img_from_cam` -- the same projection used for reprojection error
throughout this project), sampling from the undistorted images that
Phase 4's `pycolmap.undistort_images` step already produced (so sampling
never has to also handle lens distortion). No custom camera math, no
custom raster I/O (`rasterio`/GDAL) -- per the architecture decision to
build on mature libraries.

**Resolution is independent of the DEM's own resolution** (ARCHITECTURE.md
section 17): the output grid covers the DEM's footprint at its own
resolution, and the DEM's elevation surface is resampled onto it
(bilinear, `scipy.interpolate.RegularGridInterpolator`). The default
resolution is estimated from each registered camera's real height above
the DEM terrain and its calibrated focal length in pixels.

**Processed in tiles, never as one whole-grid array in RAM** (ARCHITECTURE.md
section 17): only one `TILE_SIZE_PX` tile's coordinates/elevation/colors
exist at a time, each written to the GeoTIFF via a windowed write. Each
camera's approximate ground footprint is precomputed once (by projecting
the DEM's own small grid) and used as a per-tile candidate pre-filter.

**Most-nadir view per pixel ("mosaic" blending), not an average**
(ARCHITECTURE.md section 26). Each pixel takes its color from the camera
that sees it closest to vertical (smallest angle between the camera ray
and the vertical), the same idea as Metashape's default "Mosaic"
blending. Averaging every overlapping view -- what this module did before
-- turns any height error in the DEM (crop canopy, trees, a sparse cloud)
into ghosting, because each view displaces the misplaced pixel in a
different direction; the most-nadir view is the one that displacement is
smallest for. Only near the seam between two cameras, where their view
angles differ by less than `SEAM_BLEND_DEG`, are the two best views mixed,
so seams fade instead of showing a hard edge.

**Cropped to well-observed ground** (ARCHITECTURE.md section 26): a pixel
whose most-nadir view is still more oblique than
`OrthoConfig.max_view_angle_deg` is left empty. Those are the fringes only
seen at the very edge of the outermost photos, where relief displacement
is largest and the result is smeared streaks -- the area Pix4D/Metashape
also crop away.

**Output format** (ARCHITECTURE.md section 26): 3-band RGB GeoTIFF, tiled
512x512, JPEG-compressed (YCbCr), with an internal lossless mask marking
the empty area and internal overviews -- the layout the user's own QGIS
optimizer script produced as an extra step. The mask, not NoData, marks
empty pixels: JPEG is lossy, so with NoData=0 dark-but-valid pixels would
turn transparent and the empty border would turn into a gray frame.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import radians
from pathlib import Path

import numpy as np
import pycolmap
import rasterio
import rasterio.transform
from PIL import Image as PILImage
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

# Processing tile side, in output pixels. Small tiles are much faster: a
# camera whose footprint box touches a tile is tested on every pixel of it,
# and with tiles much smaller than a photo's footprint (5280 px wide at the
# GSD) most cameras touch only the tiles they really cover. Measured on a
# synthetic flight with the same footprint/tile ratio: ~3x faster than 2048
# px tiles, same output. Must stay a multiple of OUTPUT_BLOCK_PX, so every
# JPEG block of the output file is written exactly once (a block written in
# two halves would be decompressed and recompressed, losing quality twice).
TILE_SIZE_PX = 512

# GeoTIFF internal tile size: 512x512 is the modern GDAL/COG default and
# what the user's QGIS optimizer script uses.
OUTPUT_BLOCK_PX = 512

# Visually lossless for aerial RGB, and roughly what Pix4Dfields' JPEG
# exports correspond to in bytes per pixel.
JPEG_QUALITY = 90

# Two cameras whose view angles to a pixel differ by less than this are
# mixed (50/50 at equal angles, ramping to the better one at this
# difference). At ~100 m flying height, the angle difference between two
# neighboring cameras changes by ~1 degree per meter across their seam, so
# 2 degrees gives a ~3-4 m soft transition.
SEAM_BLEND_DEG = 2.0

# Pixels whose most-nadir view is more oblique than this are left empty.
# A nadir Mavic 3M frame reaches ~28 degrees off-vertical along its short
# side, so 30 degrees keeps everything inside the flight block and about
# one footprint-half beyond the outermost line, and drops the extreme-edge
# fringe that came out as smeared streaks on the first real flight.
DEFAULT_MAX_VIEW_ANGLE_DEG = 30.0


class OrthoError(RuntimeError):
    """Raised when orthomosaic generation cannot proceed at all (missing
    inputs). A mosaic with holes where no camera saw the ground is not an
    error -- those pixels are masked out, never hidden."""


@dataclass
class OrthoConfig:
    resolution_m: float | None = None  # None = automatic, from camera GSD (see module docstring)
    # None = keep every pixel any camera sees, however obliquely.
    max_view_angle_deg: float | None = DEFAULT_MAX_VIEW_ANGLE_DEG
    # Multiplies the automatic resolution only (ignored with resolution_m):
    # 2.0 gives a pixel twice the camera GSD, a quarter of the pixels.
    resolution_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.resolution_m is not None and self.resolution_m <= 0:
            raise ValueError(f"resolution_m must be positive, got {self.resolution_m}")
        if self.resolution_scale <= 0:
            raise ValueError(f"resolution_scale must be positive, got {self.resolution_scale}")
        if self.max_view_angle_deg is not None and not (0.0 < self.max_view_angle_deg < 90.0):
            raise ValueError(f"max_view_angle_deg must be in (0, 90), got {self.max_view_angle_deg}")


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

    # The DEM is NoData where no surface was reconstructed (typically the
    # outer edge, seen by a single photo). Cameras may still see that ground
    # within the view-angle limit, so the mosaic extends the nearest known
    # elevation there instead of losing the strip; the DEM product itself
    # stays honest (ARCHITECTURE.md section 28).
    missing = np.isnan(elevation)
    if missing.any() and not missing.all():
        nearest = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        elevation = elevation[tuple(nearest)]

    return world_x, world_y, elevation, transform, crs_wkt


def _array_index_coords(pixel_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP image coordinates -> (column, row) array-index coordinates.

    COLMAP puts the image origin at the top-left CORNER of the top-left
    pixel, so that pixel's center is (0.5, 0.5) (which is why its default
    principal point is width/2, height/2). An image array indexes pixel
    centers at integers. Sampling with COLMAP's coordinates directly
    reads every photo half a pixel off (~1 cm at native GSD), in a
    direction that turns with each camera's heading, so overlapping
    photos disagree at the seams (ARCHITECTURE.md section 27).
    """
    return pixel_coords[:, 0] - 0.5, pixel_coords[:, 1] - 0.5


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sampling of an (H, W, 3) uint8 image at fractional pixel
    coordinates (u=column, v=row), all assumed inside the image."""
    height, width = image.shape[:2]
    x0 = np.clip(np.floor(u).astype(np.int64), 0, width - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.clip(np.floor(v).astype(np.int64), 0, height - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    fx = np.clip(u - x0, 0.0, 1.0).astype(np.float32)[:, None]
    fy = np.clip(v - y0, 0.0, 1.0).astype(np.float32)[:, None]

    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def _estimate_resolution_from_cameras(
    reconstruction: "pycolmap.Reconstruction", dem_elevation: np.ndarray, dem_transform: rasterio.Affine
) -> float | None:
    """Estimate ground resolution (m/px) from each registered camera's real
    height above the DEM terrain and its calibrated focal length in pixels
    -- GSD = height_above_terrain / focal_length_px, the standard pinhole
    formula. Returns the median across cameras with a valid terrain sample
    underneath them; None if no camera qualifies, so the caller can fall
    back to a safe default instead of dividing by a fabricated number.
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
    """Output raster shape/transform at `resolution_m` over the DEM's
    footprint -- pure arithmetic, no array allocation."""
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
    """Reusable (y, x) -> elevation interpolator over the DEM's own grid --
    built once, evaluated per output tile."""
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


FOOTPRINT_SAMPLE_SIDE = 256


def _footprint_sample_points(
    dem_world_x: np.ndarray, dem_world_y: np.ndarray, dem_elevation: np.ndarray
) -> tuple[np.ndarray, int]:
    """DEM terrain points used to find each camera's footprint: the DEM grid
    subsampled to at most FOOTPRINT_SAMPLE_SIDE per side (a big DEM projected
    whole through every camera would dominate the run for nothing), plus the
    subsampling step, which widens the footprint margin accordingly."""
    step = max(1, int(np.ceil(max(dem_elevation.shape) / FOOTPRINT_SAMPLE_SIDE)))
    zs = dem_elevation[::step, ::step]
    valid = ~np.isnan(zs)
    points = np.stack([dem_world_x[::step, ::step][valid], dem_world_y[::step, ::step][valid], zs[valid]], axis=1)
    return points, step


def _camera_footprint_bbox(image: "pycolmap.Image", world_points: np.ndarray) -> tuple[float, float, float, float] | None:
    """Approximate world-space bounding box (min_x, max_x, min_y, max_y) of
    a camera's ground footprint, by projecting the sampled DEM terrain
    points through it. Only a cheap per-tile pre-filter, never the final
    per-pixel test. None if the camera sees no DEM terrain at all."""
    if world_points.size == 0:
        return None
    camera = image.camera
    cam_points = image.cam_from_world() * world_points
    u, v = _array_index_coords(camera.img_from_cam(cam_points, check_cheirality=True))
    valid = ~np.isnan(u) & (u >= 0) & (u <= camera.width - 1) & (v >= 0) & (v <= camera.height - 1)
    if not np.any(valid):
        return None
    xs, ys = world_points[valid, 0], world_points[valid, 1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


class _ImageCache:
    """Decoded undistorted photos, least-recently-used first out, capped by
    memory: a full-resolution 20 MP photo is ~60 MB decoded, so keeping
    every photo of a large flight would exhaust the RAM. Tiles run row by
    row, north to south, so photos are also dropped as soon as the rows
    still to come are all south of their footprint."""

    def __init__(self, bytes_per_image: int, memory_fraction: float = 0.3, minimum: int = 8):
        import psutil

        budget = psutil.virtual_memory().available * memory_fraction
        self.capacity = max(minimum, int(budget // max(1, bytes_per_image)))
        self._images: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, image_id: int, path: Path) -> np.ndarray:
        image = self._images.get(image_id)
        if image is None:
            with PILImage.open(path) as pil:
                image = np.asarray(pil.convert("RGB"))
            self._images[image_id] = image
            while len(self._images) > self.capacity:
                self._images.popitem(last=False)
        else:
            self._images.move_to_end(image_id)
        return image

    def drop(self, image_id: int) -> None:
        self._images.pop(image_id, None)


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


def _overview_levels(width_px: int, height_px: int) -> list[int]:
    """Factors 2, 4, 8, ... until the smallest overview fits in one block."""
    max_dim = max(width_px, height_px)
    levels: list[int] = []
    factor = 2
    while max_dim > OUTPUT_BLOCK_PX:
        levels.append(factor)
        if -(-max_dim // factor) <= OUTPUT_BLOCK_PX:
            break
        factor *= 2
    return levels


def run_orthomosaic_generation(
    reconstruction_path: Path,
    undistorted_image_path: Path,
    dem_path: Path,
    output_path: Path,
    config: OrthoConfig | None = None,
    cancellation_token: "pycolmap.CancellationToken | None" = None,
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> OrthoResult:
    """Generate a georeferenced orthomosaic GeoTIFF (RGB + internal mask).

    `reconstruction_path`/`undistorted_image_path` must be the pair
    Phase 4's `pycolmap.undistort_images` wrote (undistorted images +
    matching PINHOLE-model reconstruction) -- never the original distorted
    images, since sampling here does not itself correct for distortion.

    `progress_callback`, when given, is called with `(tiles_done,
    tiles_total)` after each tile is written. `cancellation_token` (reused
    purely as a thread-safe flag) is checked once per tile; if cancelled,
    raises `InterruptedError` to match pycolmap's own convention.
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
            # it -- fall back to the DEM's own resolution rather than
            # fabricating a number or crashing.
            resolution_m = abs(dem_transform.a)
            resolution_source = "automatic (fallback: DEM resolution)"
        if config.resolution_scale != 1.0:
            resolution_m *= config.resolution_scale
            resolution_source += f" x{config.resolution_scale:g}"

    width_px, height_px, transform = _output_geometry(dem_transform, dem_width_px, dem_height_px, resolution_m)
    interpolator = _build_dem_interpolator(dem_transform, dem_width_px, dem_height_px, dem_elevation)

    footprint_points, footprint_step = _footprint_sample_points(dem_world_x, dem_world_y, dem_elevation)
    dem_cell_diag = float(np.hypot(abs(dem_transform.a), abs(dem_transform.e))) * footprint_step
    del dem_world_x, dem_world_y
    cameras: list[dict] = []
    for image_id in registered_ids:
        image = reconstruction.image(image_id)
        image_path = undistorted_image_path / image.name
        if not image_path.exists():
            continue
        bbox = _camera_footprint_bbox(image, footprint_points)
        if bbox is None:
            continue
        cameras.append(
            {
                "image_id": image_id,
                "image": image,
                "camera": image.camera,
                "center": np.asarray(image.projection_center(), dtype=np.float64),
                "image_path": image_path,
                "bbox": bbox,
            }
        )
    image_cache = _ImageCache(
        max((c["camera"].width * c["camera"].height * 3 for c in cameras), default=1)
    )

    max_angle_rad = radians(config.max_view_angle_deg) if config.max_view_angle_deg is not None else None
    seam_rad = radians(SEAM_BLEND_DEG)

    # Generated one at a time: each tile's coordinate grids are ~67 MB, and
    # materializing all of them up front cost gigabytes of RAM.
    num_tiles_total = -(-height_px // TILE_SIZE_PX) * -(-width_px // TILE_SIZE_PX)
    current_row_off = -1
    num_valid_pixels = 0
    num_nodata_pixels = 0
    used_camera_ids: set[int] = set()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdal_env = rasterio.Env(
        GDAL_TIFF_INTERNAL_MASK=True,
        COMPRESS_OVERVIEW="JPEG",
        PHOTOMETRIC_OVERVIEW="YCBCR",
        JPEG_QUALITY_OVERVIEW=str(JPEG_QUALITY),
        INTERLEAVE_OVERVIEW="PIXEL",
    )
    with gdal_env, rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height_px,
        width=width_px,
        count=3,
        dtype=rasterio.uint8,
        crs=crs_wkt,
        transform=transform,
        tiled=True,
        blockxsize=OUTPUT_BLOCK_PX,
        blockysize=OUTPUT_BLOCK_PX,
        compress="JPEG",
        photometric="YCBCR",
        jpeg_quality=JPEG_QUALITY,
        interleave="pixel",
        bigtiff="IF_SAFER",
    ) as dst:
        for tile_index, (row_off, col_off, tile_height, tile_width, tile_world_x, tile_world_y) in enumerate(
            _iter_tiles(width_px, height_px, transform, TILE_SIZE_PX), start=1
        ):
            if cancellation_token is not None and cancellation_token.is_cancelled:
                raise InterruptedError("Operation cancelled")
            if row_off != current_row_off:
                # New row of tiles: photos whose footprint lies entirely north
                # of it are not needed any more.
                current_row_off = row_off
                row_top_y = transform.f + transform.e * row_off
                for entry in cameras:
                    if entry["bbox"][2] - dem_cell_diag > row_top_y:
                        image_cache.drop(entry["image_id"])

            num_pixels = tile_height * tile_width
            tile_query = np.stack([tile_world_y.ravel(), tile_world_x.ravel()], axis=1)
            tile_elevation = interpolator(tile_query)

            # Per pixel: the best and second-best view (highest score =
            # smallest angle from vertical), their colors and cameras.
            best1 = np.full(num_pixels, -np.inf)
            best2 = np.full(num_pixels, -np.inf)
            color1 = np.zeros((num_pixels, 3), dtype=np.float32)
            color2 = np.zeros((num_pixels, 3), dtype=np.float32)
            camera1 = np.full(num_pixels, -1, dtype=np.int64)
            camera2 = np.full(num_pixels, -1, dtype=np.int64)

            terrain_idx = np.flatnonzero(~np.isnan(tile_elevation))
            if terrain_idx.size:
                world_points = np.stack(
                    [tile_world_x.ravel()[terrain_idx], tile_world_y.ravel()[terrain_idx], tile_elevation[terrain_idx]],
                    axis=1,
                )
                tile_bbox = (
                    float(world_points[:, 0].min()),
                    float(world_points[:, 0].max()),
                    float(world_points[:, 1].min()),
                    float(world_points[:, 1].max()),
                )

                tile_center = np.array([(tile_bbox[0] + tile_bbox[1]) / 2, (tile_bbox[2] + tile_bbox[3]) / 2])
                overlapping = [e for e in cameras if _bbox_overlaps(e["bbox"], tile_bbox, margin=dem_cell_diag)]
                # Nearest (most nadir) photos first: they win most pixels, so the
                # angle test below lets the later ones skip most of the tile.
                overlapping.sort(key=lambda e: float(np.hypot(*(e["center"][:2] - tile_center))))
                tile_min_z = float(world_points[:, 2].min())
                for entry in overlapping:
                    # Once every pixel of the tile has two views, a photo can only
                    # matter if its best possible angle over the tile (nearest
                    # point of the tile box, lowest terrain) beats the worst
                    # second view; otherwise it is skipped whole -- an exact
                    # bound, the output does not change.
                    worst_second = float(best2[terrain_idx].min())
                    if np.isfinite(worst_second):
                        cx, cy, cz = entry["center"]
                        dx = max(tile_bbox[0] - cx, 0.0, cx - tile_bbox[1])
                        dy = max(tile_bbox[2] - cy, 0.0, cy - tile_bbox[3])
                        height = cz - tile_min_z
                        if height > 0 and -np.arctan2(np.hypot(dx, dy), height) <= worst_second:
                            continue
                    offsets = world_points - entry["center"]
                    score_all = -np.arctan2(np.hypot(offsets[:, 0], offsets[:, 1]), -offsets[:, 2])
                    # A photo can only change a pixel whose second-best view it
                    # beats (the condition applied below), and the view angle
                    # needs no projection: project only those pixels.
                    candidates = np.flatnonzero(score_all > best2[terrain_idx])
                    if candidates.size == 0:
                        continue

                    camera = entry["camera"]
                    cam_points = entry["image"].cam_from_world() * world_points[candidates]
                    u, v = _array_index_coords(camera.img_from_cam(cam_points, check_cheirality=True))
                    visible = (
                        ~np.isnan(u)
                        & (u >= 0)
                        & (u <= camera.width - 1)
                        & (v >= 0)
                        & (v <= camera.height - 1)
                    )
                    if not np.any(visible):
                        continue

                    score = score_all[candidates[visible]]
                    idx = terrain_idx[candidates[visible]]

                    becomes_first = score > best1[idx]
                    becomes_second = ~becomes_first & (score > best2[idx])
                    if not (np.any(becomes_first) or np.any(becomes_second)):
                        continue

                    # Colors only where this photo wins a place (most visible
                    # pixels are already better served by an earlier photo).
                    take = becomes_first | becomes_second
                    photo = image_cache.get(entry["image_id"], entry["image_path"])
                    colors = _bilinear_sample(photo, u[visible][take], v[visible][take])
                    first_in_take = becomes_first[take]

                    first = idx[becomes_first]
                    best2[first] = best1[first]
                    color2[first] = color1[first]
                    camera2[first] = camera1[first]
                    best1[first] = score[becomes_first]
                    color1[first] = colors[first_in_take]
                    camera1[first] = entry["image_id"]

                    second = idx[becomes_second]
                    best2[second] = score[becomes_second]
                    color2[second] = colors[~first_in_take]
                    camera2[second] = entry["image_id"]

            has_data = np.isfinite(best1)
            if max_angle_rad is not None:
                has_data &= best1 >= -max_angle_rad

            blended = has_data & np.isfinite(best2)
            weight1 = np.ones(num_pixels, dtype=np.float32)
            weight1[blended] = np.clip(0.5 + (best1[blended] - best2[blended]) / (2 * seam_rad), 0.5, 1.0)
            rgb = color1 * weight1[:, None] + color2 * (1 - weight1)[:, None]
            rgb[~has_data] = 0

            used_camera_ids.update(np.unique(camera1[has_data]).tolist())
            used_camera_ids.update(np.unique(camera2[blended & (weight1 < 1)]).tolist())
            num_valid_pixels += int(has_data.sum())
            num_nodata_pixels += int((~has_data).sum())

            window = Window(col_off, row_off, tile_width, tile_height)
            tile_rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8).reshape(tile_height, tile_width, 3)
            dst.write(np.moveaxis(tile_rgb, 2, 0), window=window)
            dst.write_mask((has_data.reshape(tile_height, tile_width) * 255).astype(np.uint8), window=window)

            if progress_callback is not None:
                progress_callback(tile_index, num_tiles_total)

        levels = _overview_levels(width_px, height_px)
        if levels:
            dst.build_overviews(levels, Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    used_camera_ids.discard(-1)
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
