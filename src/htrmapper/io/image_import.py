"""Folder-based image import: scan, extract metadata, build a validated report.

Per the project brief, importing must never silently assume completeness.
This module explicitly counts and lists: images with no coordinates,
images with implausible/invalid coordinates, the detected spatial extent,
and altitude range -- so the user can see data quality issues before
spending compute on alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from htrmapper.core.project import ImageRecord
from htrmapper.io.exif_reader import read_exif
from htrmapper.io.xmp_reader import read_dji_xmp

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".tif", ".tiff"})


@dataclass
class ImportReport:
    total_images: int = 0
    images_without_position: list[str] = field(default_factory=list)
    images_with_invalid_position: list[str] = field(default_factory=list)
    images_without_orientation: list[str] = field(default_factory=list)
    images_without_focal_length: list[str] = field(default_factory=list)
    camera_models: set[str] = field(default_factory=set)
    min_altitude: float | None = None
    max_altitude: float | None = None
    min_latitude: float | None = None
    max_latitude: float | None = None
    min_longitude: float | None = None
    max_longitude: float | None = None
    images_with_rtk_std_dev: int = 0
    errors: dict[str, str] = field(default_factory=dict)  # file -> error message

    @property
    def num_with_valid_position(self) -> int:
        return self.total_images - len(self.images_without_position) - len(
            self.images_with_invalid_position
        )

    @property
    def has_problems(self) -> bool:
        return bool(
            self.errors
            or self.images_without_position
            or self.images_with_invalid_position
            or self.images_without_orientation
            or self.images_without_focal_length
        )

    def short_summary(self) -> str:
        """One/two-line status for a quick "did it work?" glance.

        Full detail (per-field counts, per-file errors) stays in
        `summary_lines()` for whoever wants to dig in -- this is only the
        headline: OK, or what specifically went wrong.
        """
        if self.total_images == 0:
            return "Nenhuma imagem suportada (JPG/TIFF) encontrada na pasta."

        if not self.has_problems:
            return (
                f"Importação concluída sem problemas: {self.total_images} imagens, "
                f"todas com posição GPS válida."
            )

        problems = []
        if self.errors:
            problems.append(f"{len(self.errors)} com erro de leitura")
        if self.images_without_position:
            problems.append(f"{len(self.images_without_position)} sem posição GPS")
        if self.images_with_invalid_position:
            problems.append(f"{len(self.images_with_invalid_position)} com posição GPS inválida")
        if self.images_without_orientation:
            problems.append(f"{len(self.images_without_orientation)} sem orientação do gimbal")
        if self.images_without_focal_length:
            problems.append(f"{len(self.images_without_focal_length)} sem focal length")

        return (
            f"Importação concluída com avisos ({self.total_images} imagens): "
            + "; ".join(problems)
            + ". Veja \"Show Details\" para a lista completa."
        )

    def summary_lines(self) -> list[str]:
        lines = [
            f"Total images found: {self.total_images}",
            f"Images successfully read: {self.total_images - len(self.errors)}",
            f"Images with valid position: {self.num_with_valid_position}",
            f"Images without any position: {len(self.images_without_position)}",
            f"Images with invalid/implausible position: {len(self.images_with_invalid_position)}",
            f"Images without gimbal orientation: {len(self.images_without_orientation)}",
            f"Images without focal length: {len(self.images_without_focal_length)}",
            f"Images with per-image RTK std-dev metadata: {self.images_with_rtk_std_dev}",
            f"Camera models detected: {sorted(self.camera_models) or 'none'}",
        ]
        if self.min_altitude is not None:
            lines.append(f"Altitude range (GPS, m): {self.min_altitude:.1f} .. {self.max_altitude:.1f}")
        if self.min_latitude is not None:
            lines.append(
                f"Spatial extent (WGS84): lon [{self.min_longitude:.6f}, {self.max_longitude:.6f}], "
                f"lat [{self.min_latitude:.6f}, {self.max_latitude:.6f}]"
            )
        if self.errors:
            lines.append(f"Errors reading {len(self.errors)} file(s):")
            for fname, msg in self.errors.items():
                lines.append(f"  - {fname}: {msg}")
        return lines


def find_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_image_record(image_path: Path) -> ImageRecord:
    """Extract EXIF + DJI XMP metadata for one image into an ImageRecord.

    Never raises for missing individual fields (see exif_reader/xmp_reader
    docstrings); only propagates exceptions if the file cannot be read at
    all (caller is expected to catch and record those in the ImportReport).
    """
    exif = read_exif(image_path)
    dji = read_dji_xmp(image_path)

    warnings: list[str] = []
    if not exif.has_position:
        warnings.append("no GPS position found in EXIF")
    elif not exif.position_is_valid:
        warnings.append(
            f"GPS position out of valid range (lat={exif.latitude}, lon={exif.longitude})"
        )
    if not dji.has_gimbal_orientation:
        warnings.append("no gimbal orientation (yaw/pitch/roll) found in XMP")
    if exif.focal_length_mm is None:
        warnings.append("no focal length found in EXIF")

    return ImageRecord(
        path=str(image_path.resolve()),
        file_name=image_path.name,
        latitude=exif.latitude,
        longitude=exif.longitude,
        altitude=exif.altitude,
        position_valid=exif.position_is_valid,
        relative_altitude_m=dji.relative_altitude_m,
        gimbal_yaw_deg=dji.gimbal_yaw_deg,
        gimbal_pitch_deg=dji.gimbal_pitch_deg,
        gimbal_roll_deg=dji.gimbal_roll_deg,
        flight_yaw_deg=dji.flight_yaw_deg,
        flight_pitch_deg=dji.flight_pitch_deg,
        flight_roll_deg=dji.flight_roll_deg,
        camera_make=exif.camera_make,
        camera_model=exif.camera_model,
        focal_length_mm=exif.focal_length_mm,
        focal_length_35mm_equiv=exif.focal_length_35mm_equiv,
        pixel_width=exif.pixel_width,
        pixel_height=exif.pixel_height,
        timestamp=exif.timestamp.isoformat() if exif.timestamp else None,
        rtk_flag=dji.rtk_flag,
        rtk_std_lon_m=dji.rtk_std_lon_m,
        rtk_std_lat_m=dji.rtk_std_lat_m,
        rtk_std_hgt_m=dji.rtk_std_hgt_m,
        gps_status=dji.gps_status,
        warnings=warnings,
    )


def import_folder(folder: Path) -> tuple[list[ImageRecord], ImportReport]:
    """Scan a folder for supported images and build records + a quality report."""
    image_paths = find_images(folder)
    records: list[ImageRecord] = []
    report = ImportReport(total_images=len(image_paths))

    for image_path in image_paths:
        try:
            record = build_image_record(image_path)
        except Exception as exc:  # noqa: BLE001 - we want to record and continue, not abort the batch
            report.errors[image_path.name] = str(exc)
            continue

        records.append(record)

        if not record.position_valid:
            if record.latitude is None or record.longitude is None:
                report.images_without_position.append(record.file_name)
            else:
                report.images_with_invalid_position.append(record.file_name)
        else:
            if report.min_latitude is None or record.latitude < report.min_latitude:
                report.min_latitude = record.latitude
            if report.max_latitude is None or record.latitude > report.max_latitude:
                report.max_latitude = record.latitude
            if report.min_longitude is None or record.longitude < report.min_longitude:
                report.min_longitude = record.longitude
            if report.max_longitude is None or record.longitude > report.max_longitude:
                report.max_longitude = record.longitude
            if record.altitude is not None:
                if report.min_altitude is None or record.altitude < report.min_altitude:
                    report.min_altitude = record.altitude
                if report.max_altitude is None or record.altitude > report.max_altitude:
                    report.max_altitude = record.altitude

        if record.gimbal_yaw_deg is None:
            report.images_without_orientation.append(record.file_name)
        if record.focal_length_mm is None:
            report.images_without_focal_length.append(record.file_name)
        if record.camera_model:
            report.camera_models.add(record.camera_model)
        if record.rtk_std_lon_m is not None:
            report.images_with_rtk_std_dev += 1

    return records, report
