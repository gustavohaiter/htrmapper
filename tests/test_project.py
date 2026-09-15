from __future__ import annotations

from pathlib import Path

import pytest

from htrmapper.core.project import GnssAccuracyConfig, ImageRecord, Project, ProjectCrsConfig, SfmSummary
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


def test_effective_export_epsg_falls_back_to_project_epsg_when_unset():
    crs = ProjectCrsConfig(source_epsg=4326, project_epsg=31983)

    assert crs.effective_export_epsg == 31983


def test_effective_export_epsg_uses_export_epsg_when_set():
    # Regression test: `export_epsg` used to be defined on the model (with
    # a docstring promising it "defaults to project_epsg if None") but was
    # never actually read by any export code path -- every LAS/DEM export
    # silently used `project_epsg` regardless of what a user configured
    # here. `effective_export_epsg` is the one property every export call
    # site must go through instead.
    crs = ProjectCrsConfig(source_epsg=4326, project_epsg=31983, export_epsg=4674)

    assert crs.effective_export_epsg == 4674


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


def test_sfm_summary_is_none_by_default():
    project = Project(name="p")

    assert project.sfm is None
    assert project.to_dict()["sfm"] is None


def test_sfm_summary_round_trips(tmp_path: Path):
    project = _sample_project()
    project.sfm = SfmSummary(
        num_images_input=6,
        num_registered=6,
        num_points3d=1012,
        num_observations=3893,
        mean_reprojection_error_px=0.085,
        matching_strategy="spatial (restricted by GNSS position)",
        georeferenced=True,
        georeferencing_note="aligned to EPSG:31983 using 6 GNSS-positioned camera(s)",
        reconstruction_path="/work/sparse",
    )
    out_path = tmp_path / "project.json"

    project.save(out_path)
    loaded = Project.load(out_path)

    assert loaded.sfm is not None
    assert loaded.sfm.num_registered == 6
    assert loaded.sfm.mean_reprojection_error_px == pytest.approx(0.085)
    assert loaded.sfm.georeferenced is True
