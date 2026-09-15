from __future__ import annotations

from pathlib import Path

import pytest

from htrmapper.core.project import GnssAccuracyConfig, ImageRecord, Project, ProjectCrsConfig
from htrmapper.gnss.accuracy import CameraAccuracy


def _sample_project() -> Project:
    return Project(
        name="fazenda_teste",
        crs=ProjectCrsConfig(source_epsg=4326, project_epsg=31983),
        gnss_accuracy=GnssAccuracyConfig(accuracy=CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.03)),
        images=[
            ImageRecord(
                path="/data/DJI_0001.JPG",
                file_name="DJI_0001.JPG",
                latitude=-15.5,
                longitude=-47.5,
                altitude=850.0,
                position_valid=True,
                warnings=[],
            )
        ],
    )


def test_save_and_load_round_trip(tmp_path: Path):
    project = _sample_project()
    out_path = tmp_path / "project.json"

    project.save(out_path)
    loaded = Project.load(out_path)

    assert loaded.name == project.name
    assert loaded.crs.project_epsg == 31983
    assert loaded.gnss_accuracy.accuracy == project.gnss_accuracy.accuracy
    assert len(loaded.images) == 1
    assert loaded.images[0].file_name == "DJI_0001.JPG"
    assert loaded.images[0].latitude == pytest.approx(-15.5)


def test_default_crs_is_sirgas2000_utm23s():
    project = Project(name="p")

    assert project.crs.project_epsg == 31983
    assert project.crs.source_epsg == 4326


def test_future_schema_version_is_rejected(tmp_path: Path):
    project = _sample_project()
    out_path = tmp_path / "project.json"
    project.save(out_path)

    data = out_path.read_text(encoding="utf-8")
    data = data.replace('"schema_version": 1', '"schema_version": 999')
    out_path.write_text(data, encoding="utf-8")

    with pytest.raises(ValueError):
        Project.load(out_path)


def test_image_record_warnings_survive_round_trip(tmp_path: Path):
    project = _sample_project()
    project.images[0].warnings = ["no gimbal orientation (yaw/pitch/roll) found in XMP"]
    out_path = tmp_path / "project.json"

    project.save(out_path)
    loaded = Project.load(out_path)

    assert loaded.images[0].warnings == ["no gimbal orientation (yaw/pitch/roll) found in XMP"]
