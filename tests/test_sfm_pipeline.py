from __future__ import annotations

import math
from pathlib import Path

import pytest

from htrmapper.core.project import ImageRecord, Project, ProjectCrsConfig
from htrmapper.io.image_import import import_folder
from htrmapper.sfm.pipeline import SfmConfig, SfmError, run_structure_from_motion
from tests.synthetic_scene import SyntheticFlightSpec, generate_synthetic_flight


def _build_project(tmp_path: Path, spec: SyntheticFlightSpec | None = None):
    truths = generate_synthetic_flight(tmp_path, spec)
    records, report = import_folder(tmp_path)
    assert not report.has_problems, report.summary_lines()
    project = Project(
        name="synthetic_flight",
        crs=ProjectCrsConfig(source_epsg=4326, project_epsg=31983),
        images=records,
    )
    return project, truths


def test_full_pipeline_registers_all_images_with_low_reprojection_error(tmp_path: Path):
    project, truths = _build_project(tmp_path)

    result = run_structure_from_motion(project, tmp_path / "work", SfmConfig(key_point_limit=8000))

    assert result.num_images_input == len(truths)
    assert result.num_registered == len(truths)
    assert result.unregistered_image_names == []
    assert result.num_points3d > 0
    assert result.mean_reprojection_error_px is not None
    # Noiseless synthetic scene with exact pinhole geometry: reprojection
    # error should be a small fraction of a pixel, not just "not huge".
    assert result.mean_reprojection_error_px < 1.0
    assert result.matching_strategy.startswith("spatial")


def test_full_pipeline_georeferences_to_known_camera_positions(tmp_path: Path):
    spec = SyntheticFlightSpec()
    project, truths = _build_project(tmp_path, spec)

    result = run_structure_from_motion(project, tmp_path / "work", SfmConfig(key_point_limit=8000))

    assert result.georeferenced, result.georeferencing_note

    import pycolmap

    reconstruction = pycolmap.Reconstruction(result.reconstruction_path)
    truths_by_name = {t.file_name: t for t in truths}

    max_error_m = 0.0
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        truth = truths_by_name[image.name]
        expected_x = spec.origin_x_m + truth.x_m
        expected_y = spec.origin_y_m + truth.y_m
        expected_z = spec.terrain_elevation_m + spec.altitude_agl_m

        center = image.projection_center()
        error = math.dist(center, [expected_x, expected_y, expected_z])
        max_error_m = max(max_error_m, error)

    # Noiseless synthetic scene: recovered camera positions, after
    # similarity alignment to the same GNSS positions, should match the
    # known ground truth to well under a meter.
    assert max_error_m < 1.0


def test_camera_intrinsics_source_is_exif_crop_factor(tmp_path: Path):
    project, _ = _build_project(tmp_path)

    result = run_structure_from_motion(project, tmp_path / "work", SfmConfig(key_point_limit=8000))

    assert len(result.camera_groups) == 1
    assert result.camera_groups[0].intrinsics_source == "exif_crop_factor"
    assert result.camera_groups[0].fx_px is not None


def test_falls_back_to_exhaustive_matching_with_few_gnss_positions(tmp_path: Path):
    project, _ = _build_project(tmp_path)
    # Simulate most images lacking a usable GNSS fix (e.g. a PPK gap).
    for img in project.images[2:]:
        img.position_valid = False
        img.latitude = None
        img.longitude = None

    result = run_structure_from_motion(project, tmp_path / "work", SfmConfig(key_point_limit=8000))

    assert result.matching_strategy.startswith("exhaustive")


def test_raises_when_images_span_multiple_folders(tmp_path: Path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate_synthetic_flight(dir_a, SyntheticFlightSpec(num_along_track=2, num_cross_track=1))
    generate_synthetic_flight(dir_b, SyntheticFlightSpec(num_along_track=2, num_cross_track=1))
    records_a, _ = import_folder(dir_a)
    records_b, _ = import_folder(dir_b)
    project = Project(name="mixed", images=records_a + records_b)

    with pytest.raises(SfmError):
        run_structure_from_motion(project, tmp_path / "work")


def test_raises_on_empty_project(tmp_path: Path):
    project = Project(name="empty", images=[])

    with pytest.raises(SfmError):
        run_structure_from_motion(project, tmp_path / "work")
