"""XMP metadata extraction, focused on DJI drone-dji fields.

DJI drones (including the Mavic 3M) embed flight/gimbal orientation and,
when RTK is used, RTK fix-quality metadata as an XMP packet inside the
JPEG, in addition to the standard EXIF block. This data is NOT exposed by
EXIF and is essential for photogrammetry:

- ``drone-dji:GimbalYawDegree/GimbalPitchDegree/GimbalRollDegree``: the
  orientation of the camera itself (gimbal-stabilized), which is what
  actually determines the camera's exterior orientation -- not the
  aircraft body orientation.
- ``drone-dji:FlightYawDegree/FlightPitchDegree/FlightRollDegree``: the
  aircraft body orientation (useful for QA / sanity checks, and for
  future rolling-shutter modeling that needs the aircraft's angular
  velocity).
- ``drone-dji:RelativeAltitude`` / ``AbsoluteAltitude``: altitude above
  takeoff point / above the reference the GPS altitude, respectively.
- ``drone-dji:RtkFlag``, ``RtkStdLon``, ``RtkStdLat``, ``RtkStdHgt``: when
  the drone's own RTK engine produced the position, DJI stores the fix
  status and standard deviations directly. These are DIFFERENT from the
  user-configured "Camera Accuracy" for PPK-processed positions (PPK is
  computed after the flight by external software, e.g. RTKLIB, and
  overwrites the geotag with a more accurate position that does not
  carry per-image sigma in EXIF/XMP). Both are kept, so the import
  report can tell the user which per-image accuracy information is
  actually available, rather than assuming a single global blanket
  value applies uniformly.

The XMP packet is a plain-text (UTF-8) RDF/XML block; DJI writes most
fields as bare XML attributes on the ``rdf:Description`` element (e.g.
``drone-dji:GimbalYawDegree="+83.30"``), so we parse it as XML rather than
with regular expressions, which is robust to attribute ordering and
whitespace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

_XMP_START = b"<?xpacket begin="
_XMP_END = b"<?xpacket end="

_DJI_NS = "http://www.dji.com/drone-dji/1.0/"


@dataclass
class DjiXmpData:
    gimbal_yaw_deg: float | None = None
    gimbal_pitch_deg: float | None = None
    gimbal_roll_deg: float | None = None
    flight_yaw_deg: float | None = None
    flight_pitch_deg: float | None = None
    flight_roll_deg: float | None = None
    relative_altitude_m: float | None = None
    absolute_altitude_m: float | None = None
    rtk_flag: str | None = None
    rtk_std_lon_m: float | None = None
    rtk_std_lat_m: float | None = None
    rtk_std_hgt_m: float | None = None
    gps_status: str | None = None

    @property
    def has_gimbal_orientation(self) -> bool:
        return None not in (self.gimbal_yaw_deg, self.gimbal_pitch_deg, self.gimbal_roll_deg)

    @property
    def has_rtk_std_dev(self) -> bool:
        return None not in (self.rtk_std_lon_m, self.rtk_std_lat_m, self.rtk_std_hgt_m)


def extract_xmp_packet(image_bytes: bytes) -> str | None:
    """Extract the raw XMP packet (as text) embedded in a JPEG's APP1 segment."""
    start = image_bytes.find(_XMP_START)
    if start == -1:
        return None
    end = image_bytes.find(_XMP_END, start)
    if end == -1:
        return None
    # Include the trailing "<?xpacket end=...?>" processing instruction.
    end_tag_end = image_bytes.find(b"?>", end)
    if end_tag_end == -1:
        return None
    packet = image_bytes[start : end_tag_end + 2]
    return packet.decode("utf-8", errors="replace")


def _find_attr(root: ET.Element, local_name: str) -> str | None:
    """Find an attribute by local name (namespace-agnostic) anywhere in the XMP tree."""
    for elem in root.iter():
        for attr_name, attr_value in elem.attrib.items():
            # attr_name is "{namespace}LocalName" once parsed by ElementTree.
            if attr_name.endswith("}" + local_name) or attr_name == local_name:
                return attr_value
    return None


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_dji_xmp(xmp_text: str) -> DjiXmpData:
    """Parse the DJI-specific fields out of an XMP packet's text content."""
    try:
        root = ET.fromstring(xmp_text)
    except ET.ParseError:
        return DjiXmpData()

    return DjiXmpData(
        gimbal_yaw_deg=_as_float(_find_attr(root, "GimbalYawDegree")),
        gimbal_pitch_deg=_as_float(_find_attr(root, "GimbalPitchDegree")),
        gimbal_roll_deg=_as_float(_find_attr(root, "GimbalRollDegree")),
        flight_yaw_deg=_as_float(_find_attr(root, "FlightYawDegree")),
        flight_pitch_deg=_as_float(_find_attr(root, "FlightPitchDegree")),
        flight_roll_deg=_as_float(_find_attr(root, "FlightRollDegree")),
        relative_altitude_m=_as_float(_find_attr(root, "RelativeAltitude")),
        absolute_altitude_m=_as_float(_find_attr(root, "AbsoluteAltitude")),
        rtk_flag=_find_attr(root, "RtkFlag"),
        rtk_std_lon_m=_as_float(_find_attr(root, "RtkStdLon")),
        rtk_std_lat_m=_as_float(_find_attr(root, "RtkStdLat")),
        rtk_std_hgt_m=_as_float(_find_attr(root, "RtkStdHgt")),
        gps_status=_find_attr(root, "GpsStatus"),
    )


def read_dji_xmp(image_path: Path) -> DjiXmpData:
    """Read and parse DJI XMP metadata from a JPEG file. Returns empty data if absent."""
    data = image_path.read_bytes()
    packet = extract_xmp_packet(data)
    if packet is None:
        return DjiXmpData()
    return parse_dji_xmp(packet)


# Some DJI firmware versions omit the numeric sign explicitly but always
# include it for east/north-positive values (e.g. "+123.40"); float()
# handles the leading '+' natively, but keep this regex available in case
# a future format quirk needs a fallback text scan instead of XML parsing.
_NUMERIC_ATTR_RE = re.compile(r'([\w:-]+)="([+-]?[\d.]+)"')
