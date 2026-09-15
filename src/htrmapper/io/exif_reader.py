"""EXIF metadata extraction for aerial imagery.

Reads what is actually present in the file and nothing more: every field is
Optional, because in practice not every image carries every tag (a camera
without a GPS fix at capture time, a re-saved/edited JPEG with stripped
EXIF, etc.). Callers must not assume completeness -- this is enforced by
typing every field as ``Optional`` and by the import-report layer
(``io.image_import``) that explicitly counts and surfaces missing/invalid
fields to the user instead of silently defaulting them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational

# Reverse lookup: tag name -> tag id, and the GPS sub-IFD tag names -> ids.
_TAG_NAME_TO_ID = {name: tag_id for tag_id, name in ExifTags.TAGS.items()}
_GPS_TAG_NAME_TO_ID = {name: tag_id for tag_id, name in ExifTags.GPSTAGS.items()}


@dataclass
class ExifData:
    """Subset of EXIF fields relevant to photogrammetric processing."""

    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None  # meters, GPS altitude (typically MSL or ellipsoidal per tag 6)
    altitude_ref_below_sea_level: bool | None = None
    focal_length_mm: float | None = None
    focal_length_35mm_equiv: float | None = None
    pixel_width: int | None = None
    pixel_height: int | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    timestamp: datetime | None = None
    orientation: int | None = None

    @property
    def has_position(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def position_is_valid(self) -> bool:
        """Basic sanity range check -- does NOT validate accuracy, only plausibility."""
        if not self.has_position:
            return False
        return -90.0 <= self.latitude <= 90.0 and -180.0 <= self.longitude <= 180.0


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, IFDRational):
        if value.denominator == 0:
            return None
        return float(value)
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        if den == 0:
            return None
        return num / den
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dms_to_decimal(dms, ref: str | None) -> float | None:
    """Convert EXIF GPS (degrees, minutes, seconds) tuple to signed decimal degrees."""
    if dms is None or len(dms) != 3:
        return None
    degrees = _to_float(dms[0])
    minutes = _to_float(dms[1])
    seconds = _to_float(dms[2])
    if degrees is None or minutes is None or seconds is None:
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _parse_gps_ifd(gps_ifd: dict) -> tuple[float | None, float | None, float | None, bool | None]:
    lat_ref = gps_ifd.get(_GPS_TAG_NAME_TO_ID.get("GPSLatitudeRef"))
    lat_dms = gps_ifd.get(_GPS_TAG_NAME_TO_ID.get("GPSLatitude"))
    lon_ref = gps_ifd.get(_GPS_TAG_NAME_TO_ID.get("GPSLongitudeRef"))
    lon_dms = gps_ifd.get(_GPS_TAG_NAME_TO_ID.get("GPSLongitude"))

    latitude = _dms_to_decimal(lat_dms, lat_ref)
    longitude = _dms_to_decimal(lon_dms, lon_ref)

    alt_ref_raw = gps_ifd.get(_GPS_TAG_NAME_TO_ID.get("GPSAltitudeRef"))
    altitude = _to_float(gps_ifd.get(_GPS_TAG_NAME_TO_ID.get("GPSAltitude")))
    below_sea_level: bool | None = None
    if alt_ref_raw is not None:
        # GPSAltitudeRef is an EXIF BYTE; Pillow may hand it back as a
        # single-byte `bytes` object (e.g. b'\x00') rather than an int,
        # depending on how the IFD was written.
        alt_ref_value = alt_ref_raw[0] if isinstance(alt_ref_raw, bytes) else int(alt_ref_raw)
        # EXIF spec: 0 = above sea level, 1 = below sea level.
        below_sea_level = alt_ref_value == 1
        if below_sea_level and altitude is not None:
            altitude = -altitude

    return latitude, longitude, altitude, below_sea_level


def _parse_timestamp(exif: dict) -> datetime | None:
    for tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        raw = exif.get(_TAG_NAME_TO_ID.get(tag_name))
        if not raw:
            continue
        try:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
    return None


def read_exif(image_path: Path) -> ExifData:
    """Read EXIF metadata from an image file.

    Never raises for missing/malformed individual tags -- returns an
    ``ExifData`` with the corresponding fields left as ``None``. Raises
    only if the file cannot be opened as an image at all.
    """
    with Image.open(image_path) as img:
        pixel_width, pixel_height = img.size
        exif = img.getexif()
        # Pillow exposes GPS and other sub-IFDs via get_ifd(); GPS IFD tag
        # id is 0x8825 (34853) in the base EXIF IFD.
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo) if exif else {}
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif) if exif else {}

    latitude = longitude = altitude = None
    below_sea_level = None
    if gps_ifd:
        latitude, longitude, altitude, below_sea_level = _parse_gps_ifd(gps_ifd)

    focal_length = _to_float(exif_ifd.get(_TAG_NAME_TO_ID.get("FocalLength")))
    focal_length_35mm = _to_float(exif_ifd.get(_TAG_NAME_TO_ID.get("FocalLengthIn35mmFilm")))

    merged_exif = {**exif} if exif else {}
    merged_for_timestamp = {**merged_exif, **exif_ifd}

    return ExifData(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        altitude_ref_below_sea_level=below_sea_level,
        focal_length_mm=focal_length,
        focal_length_35mm_equiv=focal_length_35mm,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        camera_make=(str(merged_exif.get(_TAG_NAME_TO_ID.get("Make"))).strip() if merged_exif.get(_TAG_NAME_TO_ID.get("Make")) else None),
        camera_model=(str(merged_exif.get(_TAG_NAME_TO_ID.get("Model"))).strip() if merged_exif.get(_TAG_NAME_TO_ID.get("Model")) else None),
        timestamp=_parse_timestamp(merged_for_timestamp),
        orientation=merged_exif.get(_TAG_NAME_TO_ID.get("Orientation")),
    )
