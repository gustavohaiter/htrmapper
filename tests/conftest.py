"""Shared test fixtures: synthetic DJI-style JPEGs with EXIF + XMP.

No real drone imagery is used or required for tests -- every fixture
builds an in-memory JPEG with EXIF GPS/camera tags (via piexif) and a
hand-crafted DJI XMP packet (spliced in as a second APP1 segment),
matching what a real Mavic 3M photo contains. This lets the whole EXIF/XMP
parsing and import pipeline be tested deterministically with known ground
truth values.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import piexif
import pytest
from PIL import Image

XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


@dataclass
class SyntheticImageSpec:
    latitude: float = -15.793889  # Brasilia area, matches SIRGAS2000/UTM23S use case
    longitude: float = -47.882778
    altitude: float = 850.0
    gimbal_yaw_deg: float = 45.0
    gimbal_pitch_deg: float = -90.0
    gimbal_roll_deg: float = 0.0
    flight_yaw_deg: float = 44.5
    flight_pitch_deg: float = 1.2
    flight_roll_deg: float = -0.3
    focal_length_mm: float = 12.29
    focal_length_35mm_equiv_mm: float = 24.0
    camera_make: str = "DJI"
    camera_model: str = "M3M"
    timestamp: str = "2025:07:10 09:15:30"
    rtk_flag: str | None = "50"
    rtk_std_lon_m: float | None = 0.015
    rtk_std_lat_m: float | None = 0.012
    rtk_std_hgt_m: float | None = 0.025
    gps_status: str | None = "RTK"
    include_gps: bool = True
    include_xmp: bool = True
    width: int = 64
    height: int = 48


def _deg_to_dms_rational(value: float) -> tuple:
    value = abs(value)
    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return ((degrees, 1), (minutes, 1), (int(round(seconds * 1000)), 1000))


def _build_exif_bytes(spec: SyntheticImageSpec) -> bytes:
    zeroth_ifd = {
        piexif.ImageIFD.Make: spec.camera_make,
        piexif.ImageIFD.Model: spec.camera_model,
        piexif.ImageIFD.Orientation: 1,
    }
    exif_ifd = {
        piexif.ExifIFD.FocalLength: (int(round(spec.focal_length_mm * 100)), 100),
        piexif.ExifIFD.FocalLengthIn35mmFilm: int(round(spec.focal_length_35mm_equiv_mm)),
        piexif.ExifIFD.DateTimeOriginal: spec.timestamp,
        piexif.ExifIFD.PixelXDimension: spec.width,
        piexif.ExifIFD.PixelYDimension: spec.height,
    }
    gps_ifd = {}
    if spec.include_gps:
        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: "N" if spec.latitude >= 0 else "S",
            piexif.GPSIFD.GPSLatitude: _deg_to_dms_rational(spec.latitude),
            piexif.GPSIFD.GPSLongitudeRef: "E" if spec.longitude >= 0 else "W",
            piexif.GPSIFD.GPSLongitude: _deg_to_dms_rational(spec.longitude),
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: (int(round(spec.altitude * 1000)), 1000),
        }
    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": {}, "thumbnail": None}
    return piexif.dump(exif_dict)


def _build_xmp_packet(spec: SyntheticImageSpec) -> bytes:
    def attr(name: str, value) -> str:
        if value is None:
            return ""
        return f'\n    drone-dji:{name}="{value}"'

    attrs = "".join(
        [
            attr("GimbalYawDegree", f"{spec.gimbal_yaw_deg:+.2f}"),
            attr("GimbalPitchDegree", f"{spec.gimbal_pitch_deg:+.2f}"),
            attr("GimbalRollDegree", f"{spec.gimbal_roll_deg:+.2f}"),
            attr("FlightYawDegree", f"{spec.flight_yaw_deg:+.2f}"),
            attr("FlightPitchDegree", f"{spec.flight_pitch_deg:+.2f}"),
            attr("FlightRollDegree", f"{spec.flight_roll_deg:+.2f}"),
            attr("RelativeAltitude", f"{spec.altitude:+.2f}"),
            attr("AbsoluteAltitude", f"{spec.altitude:+.2f}"),
            attr("RtkFlag", spec.rtk_flag),
            attr("RtkStdLon", spec.rtk_std_lon_m),
            attr("RtkStdLat", spec.rtk_std_lat_m),
            attr("RtkStdHgt", spec.rtk_std_hgt_m),
            attr("GpsStatus", spec.gps_status),
        ]
    )
    xmp = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"' + attrs + ">\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )
    return xmp.encode("utf-8")


def _insert_app1_segment(jpeg_bytes: bytes, payload: bytes) -> bytes:
    """Insert a new APP1 segment right after any existing leading APPn/COM
    segments (JFIF APP0, Exif APP1, ...), preserving JPEG segment-order
    validity (APP0 JFIF, when present, must stay the first segment)."""
    assert jpeg_bytes[0:2] == b"\xff\xd8"
    pos = 2
    while True:
        marker = jpeg_bytes[pos : pos + 2]
        is_appn = 0xFFE0 <= struct.unpack(">H", marker)[0] <= 0xFFEF
        is_com = marker == b"\xff\xfe"
        if not (is_appn or is_com):
            break
        length = struct.unpack(">H", jpeg_bytes[pos + 2 : pos + 4])[0]
        pos += 2 + length
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return jpeg_bytes[:pos] + segment + jpeg_bytes[pos:]


def make_synthetic_dji_jpeg(
    path: Path, spec: SyntheticImageSpec | None = None, image: "Image.Image | None" = None
) -> Path:
    """Write a synthetic JPEG with EXIF (+ optional DJI XMP) to `path`.

    `image`, if given, replaces the default flat-color fill -- used by the
    textured synthetic-scene generator (tests/synthetic_scene.py), since a
    flat color has no texture for SIFT to find any keypoints in.
    """
    spec = spec or SyntheticImageSpec()
    img = image if image is not None else Image.new("RGB", (spec.width, spec.height), color=(120, 130, 90))

    exif_bytes = _build_exif_bytes(spec)
    jpeg_bytes_io = path.open("wb")
    img.save(jpeg_bytes_io, format="JPEG", exif=exif_bytes)
    jpeg_bytes_io.close()

    if spec.include_xmp:
        raw = path.read_bytes()
        xmp_payload = XMP_HEADER + _build_xmp_packet(spec)
        raw = _insert_app1_segment(raw, xmp_payload)
        path.write_bytes(raw)

    return path


@pytest.fixture
def synthetic_spec() -> SyntheticImageSpec:
    return SyntheticImageSpec()


@pytest.fixture
def synthetic_dji_image(tmp_path: Path, synthetic_spec: SyntheticImageSpec) -> Path:
    return make_synthetic_dji_jpeg(tmp_path / "DJI_0001.JPG", synthetic_spec)
