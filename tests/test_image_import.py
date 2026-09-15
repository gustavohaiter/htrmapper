from __future__ import annotations

from pathlib import Path

import pytest

from htrmapper.io.image_import import build_image_record, find_images, import_folder
from tests.conftest import SyntheticImageSpec, make_synthetic_dji_jpeg


def test_find_images_only_matches_supported_extensions(tmp_path: Path):
    make_synthetic_dji_jpeg(tmp_path / "a.jpg")
    make_synthetic_dji_jpeg(tmp_path / "b.JPG")
    (tmp_path / "notes.txt").write_text("not an image")

    found = find_images(tmp_path)

    assert [p.name for p in found] == ["a.jpg", "b.JPG"]


def test_build_image_record_combines_exif_and_xmp(tmp_path: Path):
    spec = SyntheticImageSpec(latitude=-15.5, longitude=-47.5, gimbal_yaw_deg=90.0)
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    record = build_image_record(img_path)

    assert record.position_valid
    assert record.latitude == pytest.approx(spec.latitude)
    assert record.gimbal_yaw_deg == pytest.approx(90.0)
    assert record.warnings == []


def test_import_folder_reports_missing_and_valid_positions(tmp_path: Path):
    make_synthetic_dji_jpeg(tmp_path / "good1.jpg", SyntheticImageSpec(latitude=-15.0, longitude=-47.0))
    make_synthetic_dji_jpeg(tmp_path / "good2.jpg", SyntheticImageSpec(latitude=-15.1, longitude=-47.1))
    make_synthetic_dji_jpeg(tmp_path / "no_gps.jpg", SyntheticImageSpec(include_gps=False))

    records, report = import_folder(tmp_path)

    assert report.total_images == 3
    assert len(records) == 3
    assert report.num_with_valid_position == 2
    assert report.images_without_position == ["no_gps.jpg"]
    assert report.min_latitude == pytest.approx(-15.1)
    assert report.max_latitude == pytest.approx(-15.0)


def test_import_folder_counts_rtk_std_dev_and_camera_models(tmp_path: Path):
    make_synthetic_dji_jpeg(
        tmp_path / "rtk.jpg",
        SyntheticImageSpec(camera_model="M3M", rtk_std_lon_m=0.01, rtk_std_lat_m=0.01, rtk_std_hgt_m=0.02),
    )
    make_synthetic_dji_jpeg(
        tmp_path / "no_rtk.jpg",
        SyntheticImageSpec(camera_model="M3M", rtk_std_lon_m=None, rtk_std_lat_m=None, rtk_std_hgt_m=None),
    )

    _, report = import_folder(tmp_path)

    assert report.images_with_rtk_std_dev == 1
    assert report.camera_models == {"M3M"}


def test_import_folder_empty_directory(tmp_path: Path):
    records, report = import_folder(tmp_path)

    assert records == []
    assert report.total_images == 0
    assert report.num_with_valid_position == 0
