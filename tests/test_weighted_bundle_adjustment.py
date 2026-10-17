from __future__ import annotations

from pathlib import Path

import pytest

import pycolmap

from htrmapper.ba.weighted_bundle_adjustment import (
    DEFAULT_MAX_FOCAL_CHANGE_RATIO,
    DEFAULT_MAX_PRINCIPAL_POINT_OFFSET_RATIO,
    BaConfig,
    BaError,
    _prior_sigmas,
    run_gnss_weighted_bundle_adjustment,
)
from htrmapper.core.project import GnssAccuracyConfig, ImageRecord, Project, ProjectCrsConfig
from htrmapper.geo.crs import CoordinateReferenceSystem, GeodeticPoint, GeodeticTransformer, ProjectedPoint, wgs84
from htrmapper.gnss.accuracy import CameraAccuracy
from htrmapper.io.image_import import import_folder
from htrmapper.sfm.pipeline import SfmConfig, run_structure_from_motion
from tests.synthetic_scene import SyntheticFlightSpec, generate_synthetic_flight


def _aligned_project_and_reconstruction(tmp_path: Path):
    generate_synthetic_flight(tmp_path / "images", SyntheticFlightSpec())
    records, report = import_folder(tmp_path / "images")
    assert not report.has_problems
    project = Project(
        name="synthetic_flight",
        crs=ProjectCrsConfig(source_epsg=4326, project_epsg=31983),
        gnss_accuracy=GnssAccuracyConfig(accuracy=CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.02)),
        images=records,
    )
    sfm_result = run_structure_from_motion(project, tmp_path / "work", SfmConfig(key_point_limit=8000))
    assert sfm_result.success and sfm_result.georeferenced
    return project, Path(sfm_result.reconstruction_path)


def _shift_image_position(project: Project, file_name: str, bias_xyz_m: tuple[float, float, float]) -> None:
    """Simulate a bad PPK fix on one image: shift its recorded GNSS position
    by a known offset in the project CRS, round-tripping through the same
    geo.crs transform the pipeline itself uses."""
    record = next(img for img in project.images if img.file_name == file_name)
    project_crs = CoordinateReferenceSystem(project.crs.project_epsg)
    transformer = GeodeticTransformer(wgs84(), project_crs)
    point = transformer.forward(GeodeticPoint(lon=record.longitude, lat=record.latitude, alt=record.altitude))
    shifted = ProjectedPoint(x=point.x + bias_xyz_m[0], y=point.y + bias_xyz_m[1], z=point.z + bias_xyz_m[2])
    geodetic = transformer.inverse(shifted)
    record.longitude = geodetic.lon
    record.latitude = geodetic.lat
    record.altitude = geodetic.alt


def test_converges_with_clean_gnss(tmp_path: Path):
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)

    result = run_gnss_weighted_bundle_adjustment(project, reconstruction_path, tmp_path / "ba_out", BaConfig())

    assert result.converged
    assert result.num_images_with_gnss_prior == result.num_images_adjusted
    # Clean, noiseless synthetic scene + tight sigma -> sub-decimeter RMSE.
    assert result.rmse_total_cm < 10.0
    assert result.mean_reprojection_error_px < 1.0
    assert "distortion" in result.intrinsics_refined


def test_intrinsics_guard_fixes_parameters_that_drift_on_a_degenerate_flight(tmp_path: Path):
    # The 6-image synthetic flight is perfectly flat at constant altitude:
    # the textbook case where focal length and principal point are not
    # observable and drift (fx ~231 -> ~331 px, principal point ~90 px off
    # a 320 px image) while reprojection stays tiny. The guard must catch
    # that, re-run with those parameters fixed, and say so.
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)
    initial = next(iter(pycolmap.Reconstruction(str(reconstruction_path)).cameras.values()))

    result = run_gnss_weighted_bundle_adjustment(project, reconstruction_path, tmp_path / "ba_out", BaConfig())

    assert result.intrinsics_notes
    calibration = next(iter(result.camera_calibration.values()))
    final = pycolmap.Camera.create_from_model_name(1, calibration["model"], 1.0, initial.width, initial.height)
    final.params = calibration["params"]
    assert abs(final.mean_focal_length() / initial.mean_focal_length() - 1.0) <= DEFAULT_MAX_FOCAL_CHANGE_RATIO
    assert abs(final.principal_point_x - initial.width / 2) / initial.width <= DEFAULT_MAX_PRINCIPAL_POINT_OFFSET_RATIO
    assert abs(final.principal_point_y - initial.height / 2) / initial.height <= DEFAULT_MAX_PRINCIPAL_POINT_OFFSET_RATIO
    assert calibration["initial_params"] == pytest.approx(list(initial.params))


def test_fixed_intrinsics_option_keeps_the_phase2_lens(tmp_path: Path):
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)
    initial = next(iter(pycolmap.Reconstruction(str(reconstruction_path)).cameras.values()))

    result = run_gnss_weighted_bundle_adjustment(
        project, reconstruction_path, tmp_path / "ba_out", BaConfig.with_fixed_intrinsics()
    )

    assert result.intrinsics_refined == []
    assert next(iter(result.camera_calibration.values()))["params"] == pytest.approx(list(initial.params), abs=1e-9)


def test_rtk_float_image_is_weighted_by_its_own_deviation_and_left_out_of_rmse(tmp_path: Path):
    # A real flight's first two images were RTK FLOAT (std 17-47 cm) and
    # several meters off; weighted like the 2 cm FIX images they pulled
    # the block. Simulate one: 5 m off, flagged FLOAT with a 5 m std-dev.
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)
    _shift_image_position(project, "SYNTH_0000.JPG", (5.0, 0.0, 0.0))
    record = next(img for img in project.images if img.file_name == "SYNTH_0000.JPG")
    record.rtk_flag = "34"
    record.rtk_std_lon_m = record.rtk_std_lat_m = record.rtk_std_hgt_m = 5.0

    result = run_gnss_weighted_bundle_adjustment(project, reconstruction_path, tmp_path / "ba_out", BaConfig())

    float_residual = next(r for r in result.per_camera_residuals if r.image_name == "SYNTH_0000.JPG")
    assert float_residual.xy_sigma_m == pytest.approx(5.0)
    assert sum(v**2 for v in float_residual.residual_m) ** 0.5 > 2.0
    assert result.images_excluded_from_rmse == ["SYNTH_0000.JPG"]
    assert result.rmse_xy_cm < 10.0


def test_prior_sigma_uses_the_configured_accuracy_as_a_floor():
    accuracy = CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.03)
    fix = ImageRecord(path="a.jpg", file_name="a.jpg", rtk_std_lon_m=0.0045, rtk_std_lat_m=0.0049, rtk_std_hgt_m=0.011)
    floating = ImageRecord(path="b.jpg", file_name="b.jpg", rtk_std_lon_m=0.17, rtk_std_lat_m=0.16, rtk_std_hgt_m=0.47)
    no_metadata = ImageRecord(path="c.jpg", file_name="c.jpg")

    assert _prior_sigmas(fix, accuracy, use_rtk_sigma=True) == (0.02, 0.03)
    assert _prior_sigmas(floating, accuracy, use_rtk_sigma=True) == (0.17, 0.47)
    assert _prior_sigmas(floating, accuracy, use_rtk_sigma=False) == (0.02, 0.03)
    assert _prior_sigmas(no_metadata, accuracy, use_rtk_sigma=True) == (0.02, 0.03)


def test_tight_sigma_pulls_solution_toward_biased_gnss(tmp_path: Path):
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)
    bias = (5.0, 0.0, 0.0)
    _shift_image_position(project, "SYNTH_0000.JPG", bias)

    result = run_gnss_weighted_bundle_adjustment(
        project, reconstruction_path, tmp_path / "ba_out",
        BaConfig(),
    )

    biased = next(r for r in result.per_camera_residuals if r.image_name == "SYNTH_0000.JPG")
    # With the default tight sigma (0.02 m), the solved position should sit
    # close to the (wrong) GNSS observation it was given -- most of the 5 m
    # bias absorbed, not left as reprojection-consistent residual.
    residual_norm = sum(v**2 for v in biased.residual_m) ** 0.5
    assert residual_norm < 0.5


def test_loose_sigma_ignores_biased_gnss(tmp_path: Path):
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)
    bias = (5.0, 0.0, 0.0)
    _shift_image_position(project, "SYNTH_0000.JPG", bias)
    project.gnss_accuracy = GnssAccuracyConfig(accuracy=CameraAccuracy(xy_sigma_m=5.0, z_sigma_m=5.0))

    result = run_gnss_weighted_bundle_adjustment(project, reconstruction_path, tmp_path / "ba_out", BaConfig())

    biased = next(r for r in result.per_camera_residuals if r.image_name == "SYNTH_0000.JPG")
    residual_norm = sum(v**2 for v in biased.residual_m) ** 0.5
    # A loose sigma (applied uniformly to the whole flight, as a real
    # CameraAccuracy setting would be) means the solver trusts reprojection
    # more: most of the 5 m bias should remain unabsorbed in the residual,
    # in clear contrast to the tight-sigma case above.
    assert residual_norm > 2.0
    assert result.mean_reprojection_error_px < 1.0


def test_tighter_sigma_produces_smaller_residual_than_looser_sigma(tmp_path: Path):
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)
    _shift_image_position(project, "SYNTH_0000.JPG", (2.0, 0.0, 0.0))

    def residual_norm_for(xy_sigma: float) -> float:
        project.gnss_accuracy = GnssAccuracyConfig(accuracy=CameraAccuracy(xy_sigma_m=xy_sigma, z_sigma_m=xy_sigma))
        result = run_gnss_weighted_bundle_adjustment(project, reconstruction_path, tmp_path / f"ba_{xy_sigma}", BaConfig())
        r = next(r for r in result.per_camera_residuals if r.image_name == "SYNTH_0000.JPG")
        return sum(v**2 for v in r.residual_m) ** 0.5

    tight_residual = residual_norm_for(0.02)
    loose_residual = residual_norm_for(2.0)
    assert tight_residual < loose_residual


def test_raises_when_reconstruction_missing(tmp_path: Path):
    project = Project(name="p")

    with pytest.raises(BaError):
        run_gnss_weighted_bundle_adjustment(project, tmp_path / "does_not_exist", tmp_path / "out")


def test_raises_when_no_registered_images_have_matching_gnss(tmp_path: Path):
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)
    for img in project.images:
        new_name = "unmatched_" + Path(img.path).name
        img.path = str(Path(img.path).with_name(new_name))
        img.file_name = new_name

    with pytest.raises(BaError):
        run_gnss_weighted_bundle_adjustment(project, reconstruction_path, tmp_path / "out")


def test_invalid_robust_loss_is_rejected():
    with pytest.raises(ValueError):
        BaConfig(robust_loss="not_a_real_loss")


def test_raises_instead_of_silently_dropping_gnss_when_colmap_cannot_align(tmp_path: Path):
    # When COLMAP's internal re-alignment to the GNSS positions fails it
    # drops every GNSS prior and runs a reprojection-only adjustment,
    # logging only a warning (once left a test flight 21 m off). GNSS
    # positions scrambled by tens of meters, inconsistent with any
    # similarity of the reconstruction, must fail loudly instead.
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)
    scramble = [(40.0, -30.0, 0.0), (-35.0, 45.0, 10.0), (60.0, 20.0, -15.0), (-50.0, -40.0, 5.0), (25.0, 70.0, 0.0), (-70.0, 10.0, -5.0)]
    for record, offset in zip(sorted(project.images, key=lambda r: r.file_name), scramble):
        _shift_image_position(project, record.file_name, offset)

    with pytest.raises(BaError, match="did not use the GNSS"):
        run_gnss_weighted_bundle_adjustment(project, reconstruction_path, tmp_path / "ba_out", BaConfig())
