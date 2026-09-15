from __future__ import annotations

from pathlib import Path

import pytest

from htrmapper.ba.weighted_bundle_adjustment import BaConfig, BaError, run_gnss_weighted_bundle_adjustment
from htrmapper.core.project import GnssAccuracyConfig, Project, ProjectCrsConfig
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


def test_converges_with_clean_gnss_and_keeps_intrinsics_fixed_by_default(tmp_path: Path):
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)

    import pycolmap

    fx_before = next(iter(pycolmap.Reconstruction(str(reconstruction_path)).cameras.items()))[1].params[0]

    result = run_gnss_weighted_bundle_adjustment(project, reconstruction_path, tmp_path / "ba_out", BaConfig())

    assert result.converged
    assert result.num_images_with_gnss_prior == result.num_images_adjusted
    # Clean, noiseless synthetic scene + tight sigma -> sub-decimeter RMSE.
    assert result.rmse_total_cm < 10.0
    assert result.mean_reprojection_error_px < 1.0

    # refine_intrinsics defaults to False -- fx must be exactly unchanged
    # from whatever Fase 2 (COLMAP's own incremental BA) produced, not
    # re-optimized here (see BaConfig docstring on the self-calibration
    # degeneracy this avoids).
    calib = next(iter(result.camera_calibration.values()))
    assert calib["params"][0] == pytest.approx(fx_before, abs=1e-6)


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


def test_refine_intrinsics_true_is_opt_in_and_documented_as_risky(tmp_path: Path):
    project, reconstruction_path = _aligned_project_and_reconstruction(tmp_path)

    result = run_gnss_weighted_bundle_adjustment(
        project, reconstruction_path, tmp_path / "ba_out", BaConfig(refine_intrinsics=True)
    )

    # Not asserting a specific outcome here -- the point of this test is
    # only that the opt-in path runs and returns a result; the default
    # (refine_intrinsics=False) is what's asserted stable elsewhere.
    assert result.num_images_adjusted > 0


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
