from __future__ import annotations

from pathlib import Path

from htrmapper.core.project import BaSummary, GnssAccuracyConfig, Project, ProjectCrsConfig, SfmSummary
from htrmapper.core.report import build_report_from_project, render_html
from htrmapper.gnss.accuracy import CameraAccuracy
from htrmapper.io.image_import import import_folder
from tests.conftest import SyntheticImageSpec, make_synthetic_dji_jpeg


def _project_with_synthetic_images(tmp_path: Path, n: int = 4) -> Project:
    for i in range(n):
        spec = SyntheticImageSpec(
            latitude=-15.5 + i * 0.0006,
            longitude=-47.5 + i * 0.0006,
            altitude=850.0 + i,
        )
        make_synthetic_dji_jpeg(tmp_path / f"DJI_{i:04d}.JPG", spec)
    records, _ = import_folder(tmp_path)
    return Project(
        name="fazenda_teste",
        crs=ProjectCrsConfig(source_epsg=4326, project_epsg=31983),
        gnss_accuracy=GnssAccuracyConfig(accuracy=CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.03)),
        images=records,
    )


def test_survey_data_counts_match_project(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=5)

    report = build_report_from_project(project)

    assert report.survey_data.num_images == 5
    assert report.survey_data.camera_stations == 5


def test_flying_altitude_is_computed_from_relative_altitude(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=3)

    report = build_report_from_project(project)

    assert report.survey_data.flying_altitude_m.is_available
    assert report.survey_data.flying_altitude_m.value > 0


def test_ground_resolution_is_computed_when_data_available(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=3)

    report = build_report_from_project(project)

    assert report.survey_data.ground_resolution_cm.is_available


def test_coverage_area_requires_at_least_three_positions(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=2)

    report = build_report_from_project(project)

    assert not report.survey_data.coverage_area_km2.is_available
    assert "Fase 1" in report.survey_data.coverage_area_km2.render_text()


def test_coverage_area_is_computed_with_enough_positions(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=4)

    report = build_report_from_project(project)

    assert report.survey_data.coverage_area_km2.is_available
    assert report.survey_data.coverage_area_km2.value >= 0


def test_phase2_and_beyond_metrics_are_explicitly_pending(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=3)

    report = build_report_from_project(project)

    assert not report.survey_data.tie_points.is_available
    assert report.survey_data.tie_points.pending_phase == 2
    assert not report.survey_data.reprojection_error_px.is_available
    assert report.survey_data.reprojection_error_px.pending_phase == 3
    assert not report.camera_locations.total_error_cm.is_available
    assert report.camera_locations.total_error_cm.pending_phase == 3
    assert not report.dem.resolution_cm_per_px.is_available
    assert not report.orthomosaic.size.is_available


def test_camera_model_with_mixed_resolutions_is_flagged_not_silently_picked(tmp_path: Path):
    # Regression test for dead code that looked like an unfinished check:
    # `widths`/`heights` sets were computed per camera-model group but
    # never actually consulted -- the resolution column just showed the
    # FIRST image's pixel dimensions even when other images sharing that
    # camera_model name had different ones, silently hiding a real
    # data-quality problem (mismatched sensors under one model name, or a
    # grouping bug).
    make_synthetic_dji_jpeg(
        tmp_path / "DJI_0000.JPG", SyntheticImageSpec(camera_model="M3M", width=64, height=48)
    )
    make_synthetic_dji_jpeg(
        tmp_path / "DJI_0001.JPG", SyntheticImageSpec(camera_model="M3M", width=128, height=96)
    )
    records, _ = import_folder(tmp_path)
    project = Project(name="mixed_res", images=records)

    report = build_report_from_project(project)

    assert len(report.survey_data.cameras) == 1
    assert "MISTA" in report.survey_data.cameras[0].resolution


def test_gnss_accuracy_summary_reflects_project_config(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=3)

    report = build_report_from_project(project)

    assert report.gnss_accuracy.xy_sigma_m == 0.02
    assert report.gnss_accuracy.z_sigma_m == 0.03
    assert report.gnss_accuracy.total_images == 3
    assert report.gnss_accuracy.images_with_per_image_rtk_std_dev == 3


def test_camera_model_table_is_grouped_correctly(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=3)

    report = build_report_from_project(project)

    assert len(report.survey_data.cameras) == 1
    assert report.survey_data.cameras[0].model == "M3M"
    assert report.survey_data.cameras[0].count == 3


def test_system_info_is_populated():
    project = Project(name="empty")

    report = build_report_from_project(project)

    assert report.processing_parameters.system is not None
    assert report.processing_parameters.cameras == 0


def test_render_html_contains_real_counts_and_pending_markers(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=3)

    report = build_report_from_project(project)
    html_text = render_html(report)

    assert "fazenda_teste" in html_text
    assert "<td>3</td>" in html_text  # num_images
    assert "Fase 2" in html_text
    assert "Fase 3" in html_text
    assert "Não disponível" in html_text


def test_render_html_never_fabricates_reprojection_error(tmp_path: Path):
    project = _project_with_synthetic_images(tmp_path, n=3)

    report = build_report_from_project(project)
    html_text = render_html(report)

    assert "0.0 pix" not in html_text
    assert "Reprojection error" in html_text


def test_empty_project_renders_without_error():
    project = Project(name="vazio")

    report = build_report_from_project(project)
    html_text = render_html(report)

    assert "<html" in html_text
    assert "vazio" in html_text


def test_gnss_section_does_not_claim_no_adjustment_ran_once_ba_has_run(tmp_path: Path):
    # Regression test: this section's text used to be a hardcoded string
    # claiming "Nenhum ajuste foi executado ainda" (no adjustment has run
    # yet) unconditionally, even once a real Fase 3 (GNSS-weighted bundle
    # adjustment) result -- with real RMSE values already shown in the
    # Camera Locations section -- was present on the project.
    project = _project_with_synthetic_images(tmp_path, n=3)
    project.sfm = SfmSummary(reconstruction_path=str(tmp_path / "sparse"))
    project.ba = BaSummary(
        rmse_x_cm=1.0, rmse_y_cm=1.0, rmse_z_cm=1.0, rmse_xy_cm=1.4, rmse_total_cm=1.7,
        converged=True, termination_type="CONVERGENCE", num_images_with_gnss_prior=3,
    )

    report = build_report_from_project(project)
    html_text = render_html(report)

    assert "Nenhum ajuste foi executado ainda" not in html_text
    assert "já executado" in html_text


def test_processing_parameters_paragraph_is_not_styled_pending_once_available(tmp_path: Path):
    # Regression test: the Tie Points/Depth Maps/Point Cloud/DEM/
    # Orthomosaic paragraphs under "Processing Parameters" used to always
    # render with the amber "pending" CSS class, even once their phase had
    # actually run and the paragraph showed a real computed value.
    project = _project_with_synthetic_images(tmp_path, n=3)
    project.sfm = SfmSummary(
        reconstruction_path=str(tmp_path / "sparse"), num_points3d=100, num_observations=300,
        matching_strategy="exhaustive",
    )

    report = build_report_from_project(project)

    assert report.processing_parameters.tie_points_section.is_available
    html_text = render_html(report)

    assert '<p class="pending">100 tie points' not in html_text
    assert "<p>100 tie points" in html_text
