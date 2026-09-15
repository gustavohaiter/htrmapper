"""Project model and on-disk (JSON) project file format.

The project file is the single source of truth for reproducibility: every
parameter that affects the pipeline's output (CRS, GNSS accuracy, per-image
metadata) is captured here, versioned, and can be saved/reloaded to resume
processing later, per the project brief's reproducibility requirement.

This module intentionally only models what Phase 1 needs (images + CRS +
GNSS accuracy config). Later phases (tie points, dense cloud, DEM,
orthomosaic, processing history) will extend ``Project`` with additional
sections, bumping ``SCHEMA_VERSION`` and adding forward-compatible loading.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from htrmapper.gnss.accuracy import CameraAccuracy

SCHEMA_VERSION = 1


@dataclass
class ImageRecord:
    """All metadata extracted for a single source image."""

    path: str  # absolute path, as imported
    file_name: str

    # Position (WGS84 geographic, as captured by EXIF GPS / PPK geotagging).
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    position_valid: bool = False

    # Height above the takeoff point / ground, as reported by the drone's
    # own flight controller (drone-dji:RelativeAltitude in XMP). This is
    # the correct "flying height" for GSD estimation -- unlike the EXIF
    # GPS altitude, which is typically MSL/ellipsoidal and includes the
    # terrain elevation itself, not the height above it.
    relative_altitude_m: float | None = None

    # Orientation (degrees). Gimbal = actual camera orientation (preferred
    # for exterior orientation); flight = aircraft body orientation.
    gimbal_yaw_deg: float | None = None
    gimbal_pitch_deg: float | None = None
    gimbal_roll_deg: float | None = None
    flight_yaw_deg: float | None = None
    flight_pitch_deg: float | None = None
    flight_roll_deg: float | None = None

    # Camera / sensor.
    camera_make: str | None = None
    camera_model: str | None = None
    focal_length_mm: float | None = None
    focal_length_35mm_equiv: float | None = None
    pixel_width: int | None = None
    pixel_height: int | None = None

    timestamp: str | None = None  # ISO 8601

    # Per-image RTK metadata reported by the drone itself (if present).
    # Distinct from the project-level CameraAccuracy the user sets manually
    # for PPK-processed positions -- see io.xmp_reader module docstring.
    rtk_flag: str | None = None
    rtk_std_lon_m: float | None = None
    rtk_std_lat_m: float | None = None
    rtk_std_hgt_m: float | None = None
    gps_status: str | None = None

    # Populated by import-time validation; explains WHY position_valid is
    # False, or flags other issues (missing focal length, etc). Never
    # silently dropped -- surfaced in the import report.
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ImageRecord":
        return cls(**data)


@dataclass
class GnssAccuracyConfig:
    """Project-wide default GNSS/PPK camera accuracy.

    This is the value the bundle adjustment (Phase 3) will use to build
    the weight matrix for camera position observations, unless a future
    phase adds per-image overrides (e.g. from RTK std-dev metadata when
    available per image).
    """

    accuracy: CameraAccuracy = field(default_factory=lambda: CameraAccuracy(0.02, 0.02))

    def to_dict(self) -> dict:
        return self.accuracy.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> "GnssAccuracyConfig":
        return cls(accuracy=CameraAccuracy.from_dict(data))


@dataclass
class ProjectCrsConfig:
    """The three CRS roles kept distinct per the architecture (see geo.crs)."""

    source_epsg: int = 4326  # raw EXIF/PPK geotag CRS (WGS84 geographic)
    project_epsg: int = 31983  # SIRGAS 2000 / UTM zone 23S, the user's default
    export_epsg: int | None = None  # defaults to project_epsg if None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectCrsConfig":
        return cls(**data)


@dataclass
class Project:
    """Top-level project state, serializable to/from a JSON project file."""

    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    schema_version: int = SCHEMA_VERSION
    crs: ProjectCrsConfig = field(default_factory=ProjectCrsConfig)
    gnss_accuracy: GnssAccuracyConfig = field(default_factory=GnssAccuracyConfig)
    images: list[ImageRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "created_at": self.created_at,
            "crs": self.crs.to_dict(),
            "gnss_accuracy": self.gnss_accuracy.to_dict(),
            "images": [img.to_dict() for img in self.images],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        schema_version = data.get("schema_version", 1)
        if schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"project file schema_version={schema_version} is newer than this "
                f"htrmapper build supports (max {SCHEMA_VERSION}); upgrade htrmapper"
            )
        return cls(
            name=data["name"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            schema_version=schema_version,
            crs=ProjectCrsConfig.from_dict(data["crs"]),
            gnss_accuracy=GnssAccuracyConfig.from_dict(data["gnss_accuracy"]),
            images=[ImageRecord.from_dict(img) for img in data.get("images", [])],
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Project":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
