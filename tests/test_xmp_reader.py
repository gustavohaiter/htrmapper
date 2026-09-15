from __future__ import annotations

from pathlib import Path

import pytest

from htrmapper.io.xmp_reader import read_dji_xmp
from tests.conftest import SyntheticImageSpec, make_synthetic_dji_jpeg


def test_gimbal_orientation_is_parsed(tmp_path: Path):
    spec = SyntheticImageSpec(gimbal_yaw_deg=45.0, gimbal_pitch_deg=-90.0, gimbal_roll_deg=0.0)
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    dji = read_dji_xmp(img_path)

    assert dji.has_gimbal_orientation
    assert dji.gimbal_yaw_deg == pytest.approx(45.0)
    assert dji.gimbal_pitch_deg == pytest.approx(-90.0)
    assert dji.gimbal_roll_deg == pytest.approx(0.0)


def test_flight_orientation_is_distinct_from_gimbal(tmp_path: Path):
    spec = SyntheticImageSpec(
        gimbal_yaw_deg=45.0, flight_yaw_deg=44.5, flight_pitch_deg=1.2, flight_roll_deg=-0.3
    )
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    dji = read_dji_xmp(img_path)

    assert dji.flight_yaw_deg == pytest.approx(44.5)
    assert dji.flight_yaw_deg != dji.gimbal_yaw_deg


def test_rtk_std_dev_metadata_is_parsed(tmp_path: Path):
    spec = SyntheticImageSpec(rtk_std_lon_m=0.015, rtk_std_lat_m=0.012, rtk_std_hgt_m=0.025, rtk_flag="50")
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    dji = read_dji_xmp(img_path)

    assert dji.has_rtk_std_dev
    assert dji.rtk_std_lon_m == pytest.approx(0.015)
    assert dji.rtk_std_lat_m == pytest.approx(0.012)
    assert dji.rtk_std_hgt_m == pytest.approx(0.025)
    assert dji.rtk_flag == "50"
    assert dji.gps_status == "RTK"


def test_no_xmp_segment_returns_empty_data_not_error(tmp_path: Path):
    spec = SyntheticImageSpec(include_xmp=False)
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)

    dji = read_dji_xmp(img_path)

    assert not dji.has_gimbal_orientation
    assert not dji.has_rtk_std_dev
    assert dji.gimbal_yaw_deg is None
