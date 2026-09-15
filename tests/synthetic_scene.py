"""Textured synthetic aerial dataset generator, with known ground truth.

Unlike the flat-color fixtures in conftest.py (fine for EXIF/XMP parsing
tests, useless for SIFT -- a flat color has zero keypoints), this renders
a textured "terrain" and crops a true nadir pinhole photo of it from each
of a set of known camera positions, at a known altitude and focal length.
Because the scene is a flat plane viewed straight down with no rotation,
the photo of it *is* exactly a crop+resize of the terrain texture at the
scale implied by the pinhole GSD formula -- no perspective warp needed for
that special case, which keeps this generator simple and exact rather
than an approximation.

This gives an SfM integration test with actual, verifiable ground truth:
known camera spacing, known altitude, and (via EXIF GPS on each image) a
known geographic position -- so `sfm.pipeline.run_structure_from_motion`
can be checked against real expected outcomes (images register, reprojection
error is small, recovered camera spacing matches truth after
georeferencing) instead of merely "it didn't crash".
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from htrmapper.geo.crs import GeodeticTransformer, ProjectedPoint, sirgas2000_utm23s, wgs84
from tests.conftest import SyntheticImageSpec, make_synthetic_dji_jpeg

TEXTURE_PX_PER_METER = 40.0


def generate_textured_terrain(size_px: int = 3000, seed: int = 0, num_shapes: int | None = None) -> Image.Image:
    """A high-texture synthetic terrain: random colored shapes on a base
    color, giving SIFT abundant corners/edges to detect -- standing in for
    a real field/soil/vegetation photo, which real SIFT also handles well
    precisely because it isn't visually uniform.
    """
    if num_shapes is None:
        num_shapes = max(2000, int(size_px * size_px / 1500))
    rng = random.Random(seed)
    img = Image.new("RGB", (size_px, size_px), color=(90, 110, 60))
    draw = ImageDraw.Draw(img)
    for _ in range(num_shapes):
        x0 = rng.randint(0, size_px - 1)
        y0 = rng.randint(0, size_px - 1)
        w = rng.randint(3, 40)
        h = rng.randint(3, 40)
        color = (rng.randint(20, 220), rng.randint(20, 220), rng.randint(20, 220))
        if rng.random() < 0.5:
            draw.rectangle([x0, y0, x0 + w, y0 + h], fill=color)
        else:
            draw.ellipse([x0, y0, x0 + w, y0 + h], fill=color)
    return img


@dataclass(frozen=True)
class SyntheticCameraTruth:
    file_name: str
    x_m: float
    y_m: float
    altitude_agl_m: float
    lat: float
    lon: float


@dataclass(frozen=True)
class SyntheticFlightSpec:
    focal_length_mm: float = 12.29
    sensor_width_mm: float = 17.3  # Four-Thirds-ish sensor, arbitrary but self-consistent
    image_width_px: int = 320
    image_height_px: int = 240
    altitude_agl_m: float = 40.0
    terrain_elevation_m: float = 700.0
    front_overlap: float = 0.7
    side_overlap: float = 0.6
    num_along_track: int = 3
    num_cross_track: int = 2
    camera_model: str = "SYNTH-CAM"
    # Local origin, in the project CRS (SIRGAS 2000 / UTM 23S), that the
    # synthetic flight grid is centered on. Arbitrary but must fall inside
    # the CRS's valid area -- this point is within zone 23S.
    origin_x_m: float = 200_000.0
    origin_y_m: float = 8_200_000.0

    @property
    def focal_length_35mm_equiv_mm(self) -> float:
        return self.focal_length_mm * (36.0 / self.sensor_width_mm)

    @property
    def pixel_pitch_mm(self) -> float:
        return self.sensor_width_mm / self.image_width_px

    @property
    def gsd_m_per_px(self) -> float:
        return (self.pixel_pitch_mm / 1000.0) * self.altitude_agl_m / (self.focal_length_mm / 1000.0)

    @property
    def footprint_width_m(self) -> float:
        return self.image_width_px * self.gsd_m_per_px

    @property
    def footprint_height_m(self) -> float:
        return self.image_height_px * self.gsd_m_per_px


def _camera_grid_positions(spec: SyntheticFlightSpec) -> list[tuple[float, float]]:
    along_spacing_m = spec.footprint_height_m * (1.0 - spec.front_overlap)
    cross_spacing_m = spec.footprint_width_m * (1.0 - spec.side_overlap)

    positions = []
    for row in range(spec.num_cross_track):
        for col in range(spec.num_along_track):
            x = col * along_spacing_m
            y = row * cross_spacing_m
            positions.append((x, y))
    return positions


def _render_nadir_crop(terrain: Image.Image, spec: SyntheticFlightSpec, x_m: float, y_m: float) -> Image.Image:
    terrain_w, terrain_h = terrain.size
    center_px_x = terrain_w / 2.0 + x_m * TEXTURE_PX_PER_METER
    center_px_y = terrain_h / 2.0 + y_m * TEXTURE_PX_PER_METER

    crop_w_px = spec.footprint_width_m * TEXTURE_PX_PER_METER
    crop_h_px = spec.footprint_height_m * TEXTURE_PX_PER_METER

    left = center_px_x - crop_w_px / 2.0
    top = center_px_y - crop_h_px / 2.0
    right = left + crop_w_px
    bottom = top + crop_h_px

    if left < 0 or top < 0 or right > terrain_w or bottom > terrain_h:
        raise ValueError(
            f"camera footprint at ({x_m}, {y_m}) m falls outside the {terrain_w}x{terrain_h}px "
            "terrain texture; generate a larger terrain or a tighter flight grid"
        )

    crop = terrain.crop((left, top, right, bottom))
    return crop.resize((spec.image_width_px, spec.image_height_px), Image.Resampling.LANCZOS)


def _required_terrain_size_px(spec: SyntheticFlightSpec, positions: list[tuple[float, float]]) -> int:
    max_x = max(abs(x) for x, _ in positions)
    max_y = max(abs(y) for _, y in positions)
    half_extent_m = max(
        max_x + spec.footprint_width_m / 2.0,
        max_y + spec.footprint_height_m / 2.0,
    )
    margin_factor = 1.15
    return int(2 * half_extent_m * margin_factor * TEXTURE_PX_PER_METER)


def generate_synthetic_flight(
    output_dir: Path,
    spec: SyntheticFlightSpec | None = None,
    terrain: Image.Image | None = None,
    seed: int = 0,
) -> list[SyntheticCameraTruth]:
    """Render a synthetic nadir drone flight into `output_dir` as JPEGs with
    real EXIF/XMP (GPS position, altitude, camera/focal-length fields),
    returning the ground-truth camera positions used to render them.
    """
    spec = spec or SyntheticFlightSpec()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = _camera_grid_positions(spec)
    if terrain is None:
        terrain_size_px = _required_terrain_size_px(spec, positions)
        terrain = generate_textured_terrain(size_px=terrain_size_px, seed=seed)

    transformer = GeodeticTransformer(wgs84(), sirgas2000_utm23s())

    truths = []
    for i, (x_m, y_m) in enumerate(positions):
        crop = _render_nadir_crop(terrain, spec, x_m, y_m)

        world_x = spec.origin_x_m + x_m
        world_y = spec.origin_y_m + y_m
        world_z = spec.terrain_elevation_m + spec.altitude_agl_m
        geodetic = transformer.inverse(ProjectedPoint(x=world_x, y=world_y, z=world_z))

        file_name = f"SYNTH_{i:04d}.JPG"
        image_spec = SyntheticImageSpec(
            latitude=geodetic.lat,
            longitude=geodetic.lon,
            altitude=geodetic.alt,
            focal_length_mm=spec.focal_length_mm,
            focal_length_35mm_equiv_mm=spec.focal_length_35mm_equiv_mm,
            camera_make="SYNTH",
            camera_model=spec.camera_model,
            width=spec.image_width_px,
            height=spec.image_height_px,
            gimbal_yaw_deg=0.0,
            gimbal_pitch_deg=-90.0,
            gimbal_roll_deg=0.0,
        )
        make_synthetic_dji_jpeg(output_dir / file_name, image_spec, image=crop)

        truths.append(
            SyntheticCameraTruth(
                file_name=file_name,
                x_m=x_m,
                y_m=y_m,
                altitude_agl_m=spec.altitude_agl_m,
                lat=geodetic.lat,
                lon=geodetic.lon,
            )
        )

    return truths
