"""Initial camera intrinsics for the SfM/BA camera model.

Per the project brief's camera model (f, cx, cy, k1, k2, k3, p1, p2), the
initial intrinsics used to seed feature extraction/reconstruction come
from OpenCV's/COLMAP's "OPENCV" model: fx, fy, cx, cy, k1, k2, p1, p2
(k3/k4/rolling-shutter parameters are reserved for later, per the
project's "don't optimize unobserved parameters" rule).

fx/fy in pixels are derived the same way as `geo.gsd`: from EXIF's focal
length (mm) and the sensor width implied by the 35mm-equivalent focal
length (crop factor trick) -- no per-camera-model sensor database. cx/cy
default to the image center (a standard, safe initial guess; refined by
the bundle adjustment). k1/k2/p1/p2 start at zero (undistorted pinhole
guess), refined by the bundle adjustment (COLMAP's own initially, then
the project's GNSS-weighted BA in Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from htrmapper.core.project import ImageRecord
from htrmapper.geo.gsd import sensor_width_mm_from_crop_factor


@dataclass(frozen=True)
class InitialIntrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    image_width_px: int
    image_height_px: int

    def as_opencv_params_string(self) -> str:
        """COLMAP's OPENCV camera model parameter order: fx,fy,cx,cy,k1,k2,p1,p2."""
        return f"{self.fx_px},{self.fy_px},{self.cx_px},{self.cy_px},0,0,0,0"


class InsufficientExifError(ValueError):
    """Raised when an image group lacks the EXIF fields needed to derive
    initial intrinsics (focal length + 35mm-equivalent focal length +
    pixel dimensions). Never silently guessed -- the caller decides the
    fallback (e.g. letting COLMAP's own EXIF-based auto camera mode take
    over for that group)."""


def derive_initial_intrinsics(images: list[ImageRecord]) -> InitialIntrinsics:
    """Derive one shared initial intrinsics estimate for a group of images
    assumed to come from the same physical camera (e.g. all images of one
    `camera_model`). Averages over images that carry the needed EXIF
    fields; raises if none do.
    """
    usable = [
        img
        for img in images
        if img.focal_length_mm and img.focal_length_35mm_equiv and img.pixel_width and img.pixel_height
    ]
    if not usable:
        raise InsufficientExifError(
            "no image in this group has focal_length_mm + focal_length_35mm_equiv + "
            "pixel dimensions in EXIF; cannot derive initial intrinsics"
        )

    widths = {img.pixel_width for img in usable}
    heights = {img.pixel_height for img in usable}
    if len(widths) > 1 or len(heights) > 1:
        raise InsufficientExifError(
            f"images in this group have inconsistent resolutions ({widths} x {heights}); "
            "they should be split into separate camera groups"
        )
    width_px = usable[0].pixel_width
    height_px = usable[0].pixel_height

    fx_values = []
    for img in usable:
        sensor_width_mm = sensor_width_mm_from_crop_factor(img.focal_length_mm, img.focal_length_35mm_equiv)
        pixel_pitch_mm = sensor_width_mm / width_px
        fx_values.append(img.focal_length_mm / pixel_pitch_mm)

    fx = mean(fx_values)
    # Square pixels assumed (true for all consumer/drone CMOS sensors in scope).
    fy = fx

    return InitialIntrinsics(
        fx_px=fx,
        fy_px=fy,
        cx_px=width_px / 2.0,
        cy_px=height_px / 2.0,
        image_width_px=width_px,
        image_height_px=height_px,
    )
