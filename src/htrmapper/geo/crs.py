"""Coordinate reference system handling, backed by PROJ (via pyproj).

Per the project rules: coordinate transformation is NEVER done by naive
arithmetic (e.g. treating longitude/latitude degrees as if they were
linear meters, or a hand-rolled UTM formula). All geodetic transforms go
through PROJ, which correctly accounts for the ellipsoid, datum, and
projection definition of both the source and target CRS.

Three CRS roles are kept explicit, matching the project brief:

- ``source_crs``: the CRS the raw coordinates were captured in (for
  EXIF GPS and typical PPK exports, this is geographic WGS84, EPSG:4326,
  with ellipsoidal height).
- ``project_crs``: the CRS the whole project is processed and stored in
  internally (a projected, metric CRS -- e.g. EPSG:31983, SIRGAS 2000 /
  UTM zone 23S, for the user's standard Brazilian workflow).
- ``export_crs``: the CRS of a given output product; defaults to
  ``project_crs`` but a user may re-export to a different CRS.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS as ProjCRS
from pyproj import Transformer

# WGS84 geographic, the near-universal CRS for raw camera EXIF GPS tags.
WGS84_EPSG = 4326

# The user's standard working CRS for Brazilian precision-agriculture
# projects: SIRGAS 2000 / UTM zone 23S.
SIRGAS2000_UTM23S_EPSG = 31983


@dataclass(frozen=True)
class GeodeticPoint:
    """A point in a geographic CRS: longitude/latitude in degrees, ellipsoidal height in meters."""

    lon: float
    lat: float
    alt: float | None = None


@dataclass(frozen=True)
class ProjectedPoint:
    """A point in a projected, metric CRS."""

    x: float
    y: float
    z: float | None = None


class CoordinateReferenceSystem:
    """Thin, explicit wrapper around a pyproj CRS.

    Kept as a distinct class (rather than passing raw EPSG ints/pyproj
    objects around the codebase) so every module that touches CRS is
    forced to go through validated, named objects -- this is what makes
    "source CRS" vs "project CRS" vs "export CRS" a hard distinction
    instead of an easily-confused convention.
    """

    def __init__(self, epsg: int):
        try:
            self._crs = ProjCRS.from_epsg(epsg)
        except Exception as exc:  # pragma: no cover - pyproj raises varied types
            raise ValueError(f"invalid or unknown EPSG code: {epsg}") from exc
        self.epsg = epsg

    @property
    def is_geographic(self) -> bool:
        return self._crs.is_geographic

    @property
    def is_projected(self) -> bool:
        return self._crs.is_projected

    @property
    def name(self) -> str:
        return self._crs.name

    @property
    def proj_crs(self) -> ProjCRS:
        return self._crs

    def to_wkt(self) -> str:
        return self._crs.to_wkt()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CoordinateReferenceSystem(EPSG:{self.epsg}, {self.name!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CoordinateReferenceSystem) and self.epsg == other.epsg

    def __hash__(self) -> int:
        return hash(self.epsg)


def wgs84() -> CoordinateReferenceSystem:
    return CoordinateReferenceSystem(WGS84_EPSG)


def sirgas2000_utm23s() -> CoordinateReferenceSystem:
    return CoordinateReferenceSystem(SIRGAS2000_UTM23S_EPSG)


class GeodeticTransformer:
    """Correct geodetic transform between two CRSs, using PROJ.

    Uses ``always_xy=True`` so coordinate order is consistently
    (longitude/easting, latitude/northing) regardless of the axis order
    defined by the authority for a given CRS (some EPSG definitions use
    lat/lon axis order internally; this is a common source of silent
    50000+ km errors if bypassed).
    """

    def __init__(self, source: CoordinateReferenceSystem, target: CoordinateReferenceSystem):
        self.source = source
        self.target = target
        self._transformer = Transformer.from_crs(
            source.proj_crs, target.proj_crs, always_xy=True
        )
        self._inverse_transformer = Transformer.from_crs(
            target.proj_crs, source.proj_crs, always_xy=True
        )

    def forward(self, point: GeodeticPoint) -> ProjectedPoint:
        if point.alt is not None:
            x, y, z = self._transformer.transform(point.lon, point.lat, point.alt)
            return ProjectedPoint(x=x, y=y, z=z)
        x, y = self._transformer.transform(point.lon, point.lat)
        return ProjectedPoint(x=x, y=y, z=None)

    def inverse(self, point: ProjectedPoint) -> GeodeticPoint:
        if point.z is not None:
            lon, lat, alt = self._inverse_transformer.transform(point.x, point.y, point.z)
            return GeodeticPoint(lon=lon, lat=lat, alt=alt)
        lon, lat = self._inverse_transformer.transform(point.x, point.y)
        return GeodeticPoint(lon=lon, lat=lat, alt=None)
