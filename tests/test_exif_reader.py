from __future__ import annotations

from pathlib import Path

import pytest

from htrmapper.io.exif_reader import read_exif
from tests.conftest import SyntheticImageSpec, make_synthetic_dji_jpeg


def test_reads_gps_position_within_dms_rounding_tolerance(tmp_path: Path):
    spec = SyntheticImageSpec(latitude=-15.793889, longitude=-47.882778, altitude=850.0)
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    exif = read_exif(img_path)

    assert exif.has_position
    assert exif.position_is_valid
    # DMS with 1/1000 arcsecond rational precision -> sub-mm level rounding.
    assert exif.latitude == pytest.approx(spec.latitude, abs=1e-6)
    assert exif.longitude == pytest.approx(spec.longitude, abs=1e-6)
    assert exif.altitude == pytest.approx(spec.altitude, abs=0.01)


def test_southern_western_hemisphere_signs_are_negative(tmp_path: Path):
    spec = SyntheticImageSpec(latitude=-15.5, longitude=-47.5)
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    exif = read_exif(img_path)

    assert exif.latitude is not None and exif.latitude < 0
    assert exif.longitude is not None and exif.longitude < 0


def test_missing_gps_is_reported_as_absent_not_zero(tmp_path: Path):
    spec = SyntheticImageSpec(include_gps=False)
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    exif = read_exif(img_path)

    assert not exif.has_position
    assert exif.latitude is None
    assert exif.longitude is None


def test_camera_and_focal_length_metadata(tmp_path: Path):
    spec = SyntheticImageSpec(camera_make="DJI", camera_model="M3M", focal_length_mm=12.29)
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    exif = read_exif(img_path)

    assert exif.camera_make == "DJI"
    assert exif.camera_model == "M3M"
    assert exif.focal_length_mm == pytest.approx(12.29, abs=0.01)
    assert exif.pixel_width == spec.width
    assert exif.pixel_height == spec.height


def test_timestamp_is_parsed(tmp_path: Path):
    spec = SyntheticImageSpec(timestamp="2025:07:10 09:15:30")
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    exif = read_exif(img_path)

    assert exif.timestamp is not None
    assert exif.timestamp.year == 2025
    assert exif.timestamp.month == 7
    assert exif.timestamp.day == 10
    assert exif.timestamp.hour == 9
    assert exif.timestamp.minute == 15
