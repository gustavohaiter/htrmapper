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


def test_sfm_config_extraction_options_carries_max_image_size():
    assert SfmConfig().extraction_options().max_image_size == 2000
    assert SfmConfig(max_image_size=-1).extraction_options().max_image_size == -1
    assert SfmConfig(max_image_size=1600).extraction_options().max_image_size == 1600


def test_default_spatial_max_neighbors_is_30_not_the_old_50():
    # Regression test for the value itself, not just that the field
    # exists: a post-match "keep at most N matches" cap
    # (`FeatureMatchingOptions.max_num_matches`) measured zero effect on
    # matching wall-clock time in direct testing (it only trims the result
    # after the expensive brute-force comparison already ran) -- the
    # lever that actually controls it is how many candidate neighbor
    # images are matched against at all. COLMAP's own FAQ recommends
    # lowering exactly this to control spatial-matching runtime; 30
    # matches the value used in published pipeline examples for real
    # aerial datasets.
    assert SfmConfig().spatial_max_neighbors == 30


def test_default_max_image_size_meaningfully_reduces_features_on_a_large_image(tmp_path: Path):
    """Regression test for a real performance bug found on a user's actual
    56-photo, 21MP (5280x3956) flight: feature matching took minutes per
    image pair instead of seconds. Root cause: `SfmConfig` never set
    `max_image_size`, so pycolmap's own default (-1, no downscaling)
    extracted SIFT at full native resolution -- COLMAP's own log even
    warns about this ("Consider reducing the maximum image size"). This
    test proves the fix actually changes behavior, at the resolution that
    matters (measured against elapsed time, not feature count -- an
    earlier version of this test compared feature counts on random noise
    and was itself wrong: noise is adversarial for a scale-space detector,
    so its keypoint count does not vary monotonically with the requested
    max size, unlike real photographic content; elapsed extraction time is
    the reliable, monotonic signal for "how many pixels did this actually
    process").

    It also guards a second, easy-to-repeat mistake: COLMAP's CPU SIFT
    extractor only actually resamples when the requested size crosses one
    of its internal pyramid ("octave") boundaries, power-of-2 steps from
    the native resolution -- a value that looks like a reasonable
    reduction (e.g. 3200 from a 5280px-wide photo, ~1.65x) can measure
    byte-for-byte identical to no limit at all, a silent no-op. This test
    uses the project's own real 5280px-wide drone photo resolution (from a
    real 56-photo user flight) specifically so a future "reasonable but
    ineffective" value regresses here instead of only in the field."""
    import sqlite3
    import time

    import numpy as np
    import pycolmap
    from PIL import Image

    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, size=(3956, 5280, 3), dtype=np.uint8)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(pixels).save(image_dir / "big.jpg", quality=95)

    def _extract_and_time(max_image_size: int) -> float:
        db_path = tmp_path / f"db_{max_image_size}.sqlite"
        options = pycolmap.FeatureExtractionOptions()
        options.max_image_size = max_image_size
        options.num_threads = 1  # deterministic timing, not a wall-clock race against other cores
        start = time.time()
        pycolmap.extract_features(
            database_path=db_path, image_path=image_dir, image_names=["big.jpg"],
            camera_mode=pycolmap.CameraMode.SINGLE, extraction_options=options, device=pycolmap.Device.cpu,
        )
        elapsed = time.time() - start
        con = sqlite3.connect(db_path)
        (num_features,) = con.execute("select rows from keypoints").fetchone()
        con.close()
        assert num_features > 0  # sanity: extraction actually ran, not silently skipped
        return elapsed

    config = SfmConfig()
    assert config.max_image_size == 2000  # the fixed, validated default -- never -1 again by accident

    elapsed_default = _extract_and_time(config.max_image_size)
    elapsed_native = _extract_and_time(-1)

    assert elapsed_default < elapsed_native * 0.7
