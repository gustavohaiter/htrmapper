"""Ground Sample Distance (GSD) estimation.

GSD is the standard photogrammetric measure of "resolution on the ground"
reported in every processing report (Metashape reports it as
"Ground resolution", e.g. "4.03 cm/pix"). The correct pinhole-camera
formula is:

    GSD (m/px) = (sensor_pixel_pitch_m * flying_height_m) / focal_length_m

This requires the sensor's physical pixel pitch, which is rarely present
in EXIF directly. Rather than depending on a hand-maintained database of
camera sensor sizes per model (fragile, goes stale, and silently wrong
for an unlisted camera), we derive the sensor width from the "35mm
equivalent focal length" that cameras (including the DJI Mavic 3M / M3M)
routinely write to EXIF (tag FocalLengthIn35mmFilm):

    crop_factor = focal_length_35mm_equiv_mm / focal_length_mm
    sensor_width_mm = 36.0 / crop_factor          # 36 mm = full-frame width
    pixel_pitch_mm = sensor_width_mm / image_width_px

This is the same trick photography tools use to recover sensor size from
EXIF alone, and it is self-consistent for any camera as long as both
focal length fields are present -- no per-model lookup table needed.

This is a *pre-alignment estimate*: it uses the nominal flying height from
GNSS/EXIF, not the bundle-adjusted camera height above the reconstructed
terrain. Once the dense point cloud / DEM exist (Phase 4/5), the
per-image GSD can be recomputed from the actual camera-to-terrain
distance, which is more accurate over sloped terrain. Both values will be
surfaced in the report labeled accordingly -- never silently swapped.
"""

from __future__ import annotations

from dataclasses import dataclass

FULL_FRAME_SENSOR_WIDTH_MM = 36.0


@dataclass(frozen=True)
class GsdEstimate:
    gsd_cm_per_px: float
    sensor_width_mm: float
    pixel_pitch_mm: float
    crop_factor: float


def sensor_width_mm_from_crop_factor(focal_length_mm: float, focal_length_35mm_equiv_mm: float) -> float:
    """Derive physical sensor width from the crop factor implied by EXIF's
    35mm-equivalent focal length. Shared by GSD estimation here and by the
    initial camera-intrinsics guess in `sfm.camera_model` -- one formula,
    reused, instead of two copies drifting apart.
    """
    if focal_length_mm <= 0:
        raise ValueError(f"focal_length_mm must be positive, got {focal_length_mm}")
    if focal_length_35mm_equiv_mm <= 0:
        raise ValueError(
            f"focal_length_35mm_equiv_mm must be positive, got {focal_length_35mm_equiv_mm}"
        )
    crop_factor = focal_length_35mm_equiv_mm / focal_length_mm
    return FULL_FRAME_SENSOR_WIDTH_MM / crop_factor


def estimate_gsd(
    flying_height_m: float,
    focal_length_mm: float,
    focal_length_35mm_equiv_mm: float,
    image_width_px: int,
) -> GsdEstimate:
    """Estimate ground sample distance from flying height and EXIF focal lengths.

    Raises ValueError if any input is non-positive -- these are physical
    quantities that cannot be zero or negative for a valid image.
    """
    if flying_height_m <= 0:
        raise ValueError(f"flying_height_m must be positive, got {flying_height_m}")
    if image_width_px <= 0:
        raise ValueError(f"image_width_px must be positive, got {image_width_px}")

    sensor_width_mm = sensor_width_mm_from_crop_factor(focal_length_mm, focal_length_35mm_equiv_mm)
    crop_factor = FULL_FRAME_SENSOR_WIDTH_MM / sensor_width_mm
    pixel_pitch_mm = sensor_width_mm / image_width_px

    gsd_m_per_px = (pixel_pitch_mm / 1000.0) * flying_height_m / (focal_length_mm / 1000.0)
    gsd_cm_per_px = gsd_m_per_px * 100.0

    return GsdEstimate(
        gsd_cm_per_px=gsd_cm_per_px,
        sensor_width_mm=sensor_width_mm,
        pixel_pitch_mm=pixel_pitch_mm,
        crop_factor=crop_factor,
    )


def estimate_gsd_from_pixel_pitch(
    flying_height_m: float,
    focal_length_mm: float,
    pixel_pitch_um: float,
) -> float:
    """Estimate GSD (cm/px) when the sensor's physical pixel pitch (in
    micrometers) is known directly -- e.g. from a camera calibration
    report or manufacturer spec sheet -- rather than derived from the
    35mm-equivalent focal length. More accurate when available.
    """
    if flying_height_m <= 0 or focal_length_mm <= 0 or pixel_pitch_um <= 0:
        raise ValueError("flying_height_m, focal_length_mm and pixel_pitch_um must be positive")
    pixel_pitch_m = pixel_pitch_um * 1e-6
    focal_length_m = focal_length_mm / 1000.0
    gsd_m = pixel_pitch_m * flying_height_m / focal_length_m
    return gsd_m * 100.0
