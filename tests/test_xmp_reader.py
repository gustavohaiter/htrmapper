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


def test_xmp_found_without_reading_the_whole_photo(tmp_path: Path, monkeypatch):
    """Only the head of the file is read (XMP_SEARCH_BYTES), not the
    10-20 MB photo; a packet that runs past the head is still read whole."""
    import htrmapper.io.xmp_reader as xmp_reader

    spec = SyntheticImageSpec(gimbal_yaw_deg=12.5, rtk_flag="50")
    img_path = make_synthetic_dji_jpeg(tmp_path / "img.jpg", spec)
    head = img_path.read_bytes()
    start = head.find(b"<?xpacket begin=")
    assert start != -1
    img_path.write_bytes(head + b"\0" * 2_000_000)  # a big image body after the metadata
    assert read_dji_xmp(img_path).gimbal_yaw_deg == pytest.approx(12.5)

    monkeypatch.setattr(xmp_reader, "XMP_SEARCH_BYTES", start + 40)  # packet cut in the middle
    dji = read_dji_xmp(img_path)
    assert dji.gimbal_yaw_deg == pytest.approx(12.5)
    assert dji.rtk_flag == "50"


def test_fields_written_as_elements_are_read_too():
    from htrmapper.io.xmp_reader import parse_dji_xmp

    packet = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" drone-dji:GimbalYawDegree="+10.0">'
        "<drone-dji:RtkFlag>50</drone-dji:RtkFlag><drone-dji:RtkStdLon> 0.012 </drone-dji:RtkStdLon>"
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
    )
    dji = parse_dji_xmp(packet)
    assert dji.gimbal_yaw_deg == pytest.approx(10.0)
    assert dji.rtk_flag == "50"
    assert dji.rtk_std_lon_m == pytest.approx(0.012)
