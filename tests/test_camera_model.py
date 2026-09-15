from __future__ import annotations

import pytest

from htrmapper.core.project import ImageRecord
from htrmapper.sfm.camera_model import InsufficientExifError, derive_initial_intrinsics


def _record(focal_mm=12.29, focal_35mm=24.0, width=5472, height=3648) -> ImageRecord:
    return ImageRecord(
        path="/data/img.jpg",
        file_name="img.jpg",
        focal_length_mm=focal_mm,
        focal_length_35mm_equiv=focal_35mm,
        pixel_width=width,
        pixel_height=height,
    )


def test_fx_matches_manual_crop_factor_formula():
    images = [_record(focal_mm=12.29, focal_35mm=24.0, width=5472, height=3648)]

    intrinsics = derive_initial_intrinsics(images)

    crop_factor = 24.0 / 12.29
    sensor_width_mm = 36.0 / crop_factor
    pixel_pitch_mm = sensor_width_mm / 5472
    expected_fx = 12.29 / pixel_pitch_mm

    assert intrinsics.fx_px == pytest.approx(expected_fx)
    assert intrinsics.fy_px == intrinsics.fx_px  # square pixels assumed


def test_principal_point_defaults_to_image_center():
    images = [_record(width=1000, height=600)]

    intrinsics = derive_initial_intrinsics(images)

    assert intrinsics.cx_px == 500.0
    assert intrinsics.cy_px == 300.0


def test_averages_across_multiple_images_in_group():
    # fx_px = width_px * focal_35mm_equiv_mm / 36mm (the raw focal_length_mm
    # cancels out of the crop-factor formula), so vary focal_35mm_equiv_mm
    # to actually change fx and exercise the averaging.
    images = [_record(focal_35mm=20.0), _record(focal_35mm=28.0)]

    intrinsics = derive_initial_intrinsics(images)

    fx_low = derive_initial_intrinsics([_record(focal_35mm=20.0)]).fx_px
    fx_high = derive_initial_intrinsics([_record(focal_35mm=28.0)]).fx_px
    assert fx_low < intrinsics.fx_px < fx_high
    assert intrinsics.fx_px == pytest.approx((fx_low + fx_high) / 2)


def test_raises_when_no_image_has_required_exif_fields():
    images = [
        ImageRecord(path="/data/img.jpg", file_name="img.jpg", focal_length_mm=None, focal_length_35mm_equiv=None)
    ]

    with pytest.raises(InsufficientExifError):
        derive_initial_intrinsics(images)


def test_skips_incomplete_images_and_uses_complete_ones():
    complete = _record()
    incomplete = ImageRecord(path="/data/other.jpg", file_name="other.jpg", focal_length_mm=None)

    intrinsics = derive_initial_intrinsics([complete, incomplete])

    assert intrinsics.fx_px == pytest.approx(derive_initial_intrinsics([complete]).fx_px)


def test_raises_on_inconsistent_resolutions_in_group():
    images = [_record(width=1000, height=600), _record(width=2000, height=1200)]

    with pytest.raises(InsufficientExifError):
        derive_initial_intrinsics(images)
