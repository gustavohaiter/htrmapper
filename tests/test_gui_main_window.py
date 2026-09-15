"""Fase 7: interface completa (tree UI, background workers, previews).

Runs Qt with the "offscreen" platform plugin (no real display needed, and
harmless if one is present) so this suite works in CI/sandboxes exactly
like the rest of the project's synthetic-data tests. These are the "did it
actually run and update state" smoke tests for the GUI; the pipeline math
itself is already covered by the Fase 2-6 test suites and
`test_cancellation_and_progress.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import rasterio
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
from rasterio.transform import from_origin

from htrmapper.core.project import DemSummary, MvsSummary, OrthoSummary, Project, ProjectCrsConfig
from htrmapper.gui.main_window import MainWindow, _TREE_SECTIONS
from htrmapper.gui.worker import PipelineWorker
from htrmapper.io.image_import import import_folder
from htrmapper.sfm.pipeline import SfmConfig, run_structure_from_motion
from tests.synthetic_scene import (
    SyntheticFlightSpec,
    _camera_grid_positions,
    _required_terrain_size_px,
    build_ground_truth_reconstruction,
    generate_smooth_gradient_terrain,
    generate_synthetic_flight,
)


@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    return existing or QApplication([])


def _pump_worker_to_completion(worker: PipelineWorker, timeout_ms: int = 30_000) -> str:
    """Runs a local Qt event loop until the worker reports a terminal
    signal, exactly the pattern the real GUI relies on (queued signal
    delivery across threads needs a running event loop to dispatch)."""
    loop = QEventLoop()
    outcome = {}

    def _mark(name):
        def _handler(*_args):
            outcome["result"] = name
            loop.quit()
        return _handler

    worker.finished_ok.connect(_mark("finished_ok"))
    worker.failed.connect(_mark("failed"))
    worker.cancelled.connect(_mark("cancelled"))

    QTimer.singleShot(timeout_ms, loop.quit)
    worker.start()
    loop.exec()
    assert "result" in outcome, "worker did not finish within timeout"
    return outcome["result"]


def test_window_builds_with_empty_project(app):
    window = MainWindow(Project(name="vazio"))

    assert window.project_item.text(0) == "vazio"
    assert window.align_button.isEnabled() is False
    assert window.save_project_button.isEnabled() is False


def test_tree_selection_switches_stacked_page(app):
    window = MainWindow(Project(name="teste"))

    for key, _label in _TREE_SECTIONS:
        window.tree.setCurrentItem(window.tree_items[key])
        assert window.pages.currentIndex() == window._page_index[key]


def test_import_enables_downstream_buttons_and_populates_table(app, tmp_path: Path):
    generate_synthetic_flight(tmp_path)
    window = MainWindow(Project(name="teste"))

    records, _report = import_folder(tmp_path)
    window.project.images = records
    window._sync_ui_to_project_state()

    assert window.table.rowCount() == len(records)
    assert window.align_button.isEnabled() is True
    assert window.save_project_button.isEnabled() is True
    assert window.adjust_button.isEnabled() is False  # no SfM result yet


def test_align_click_asks_for_key_point_limit_and_threads_it_into_sfm_config(
    app, tmp_path: Path, monkeypatch
):
    # Regression test: `_on_align_clicked` used to call `SfmConfig()` with
    # no way for a GUI user to change `key_point_limit` or
    # `spatial_max_neighbors` -- only the CLI's `--key-point-limit`/
    # `--spatial-max-neighbors` flags could. On a real user flight (56 real
    # 21MP photos) the fixed defaults made feature matching take minutes
    # per image, and the GUI offered no way to lower either without
    # editing code. Confirms each dialog's answer reaches the matching
    # SfmConfig field the worker runs with, not just that a dialog
    # appears -- two different values, in dialog call order, so a field
    # mix-up (e.g. key_point_limit accidentally receiving the neighbors
    # answer) would fail this test.
    answers = iter([(12345, True), (7, True)])
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(lambda *a, **k: next(answers)))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    monkeypatch.setattr(PipelineWorker, "start", lambda self: None)  # don't actually run the pipeline

    generate_synthetic_flight(tmp_path / "images")
    records, _report = import_folder(tmp_path / "images")
    window = MainWindow(Project(name="teste", images=records))
    window._sync_ui_to_project_state()

    window._on_align_clicked()

    assert window._active_worker is not None
    sfm_config = window._active_worker._args[2]
    assert isinstance(sfm_config, SfmConfig)
    assert sfm_config.key_point_limit == 12345
    assert sfm_config.spatial_max_neighbors == 7


def test_align_worker_updates_project_and_ui(app, tmp_path: Path, monkeypatch):
    # `_handle_align_result` (the worker's on_success callback) pops a
    # modal `QMessageBox.exec()` -- in real use a human clicks it, but a
    # headless/automated run has nobody to do that, so it would block
    # forever waiting for input that never arrives. This is not a bug in
    # the code under test (this exact modal dialog is what a real user
    # sees after a real alignment run); it is just what any Qt test needs
    # to stub out to run unattended, so we auto-accept it here.
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Ok)

    generate_synthetic_flight(tmp_path)
    records, _report = import_folder(tmp_path)
    project = Project(
        name="synthetic_flight", crs=ProjectCrsConfig(source_epsg=4326, project_epsg=31983), images=records
    )
    window = MainWindow(project)
    window._sync_ui_to_project_state()

    worker = PipelineWorker(run_structure_from_motion, window.project, tmp_path / "work", SfmConfig(key_point_limit=8000))
    window._start_worker(worker, "Alinhamento (SfM)", window._handle_align_result)

    assert window.align_button.isEnabled() is False  # disabled while busy
    assert worker.supports_cancellation is True
    # isVisible() also depends on the (never-shown-in-this-test) top-level
    # window's own visibility; isHidden() checks only this widget's own
    # explicit show/hide state, which is what `_start_worker` actually sets.
    assert window.cancel_button.isHidden() is False

    outcome = _pump_worker_to_completion(worker)

    assert outcome == "finished_ok"
    assert window.project.sfm is not None
    assert window.project.sfm.num_registered > 0
    assert window.align_button.isEnabled() is True  # re-enabled after completion
    assert window.adjust_button.isEnabled() is True  # unlocked by the new SfM result


def test_adjust_worker_has_no_cancel_button_no_native_cancellation_hook(app):
    # BA's underlying Ceres solve has no cancellation_token parameter in
    # this pycolmap version (see ba/weighted_bundle_adjustment.py), so the
    # worker must not offer a Cancelar button that would silently do
    # nothing if clicked.
    from htrmapper.ba.weighted_bundle_adjustment import run_gnss_weighted_bundle_adjustment

    worker = PipelineWorker(run_gnss_weighted_bundle_adjustment, object(), Path("."), Path("."))
    assert worker.supports_cancellation is False
    assert worker.supports_progress is False


def _build_flat_dem(spec, positions, path, resolution_m: float = 0.5) -> None:
    margin = max(spec.footprint_width_m, spec.footprint_height_m) * 1.5
    min_x = spec.origin_x_m + min(p[0] for p in positions) - spec.footprint_width_m / 2 - margin
    max_x = spec.origin_x_m + max(p[0] for p in positions) + spec.footprint_width_m / 2 + margin
    min_y = spec.origin_y_m + min(p[1] for p in positions) - spec.footprint_height_m / 2 - margin
    max_y = spec.origin_y_m + max(p[1] for p in positions) + spec.footprint_height_m / 2 + margin
    width = int((max_x - min_x) / resolution_m)
    height = int((max_y - min_y) / resolution_m)
    elevation = np.full((height, width), spec.terrain_elevation_m, dtype=np.float32)
    transform = from_origin(min_x, max_y, resolution_m, resolution_m)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=rasterio.float32, crs="EPSG:31983", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(elevation, 1)


def test_dem_and_orthomosaic_previews_render_without_crashing(app, tmp_path: Path):
    from htrmapper.ortho.orthomosaic import run_orthomosaic_generation

    spec = SyntheticFlightSpec()
    positions = _camera_grid_positions(spec)
    terrain_size = _required_terrain_size_px(spec, positions)
    terrain = generate_smooth_gradient_terrain(size_px=terrain_size, seed=0)
    images_dir = tmp_path / "images"
    truths = generate_synthetic_flight(images_dir, spec, terrain=terrain)
    reconstruction = build_ground_truth_reconstruction(spec, truths)
    reconstruction_path = tmp_path / "sparse"
    reconstruction_path.mkdir()
    reconstruction.write(reconstruction_path)
    dem_path = tmp_path / "dem.tif"
    _build_flat_dem(spec, positions, dem_path)
    ortho_path = tmp_path / "ortho.tif"
    ortho_result = run_orthomosaic_generation(reconstruction_path, images_dir, dem_path, ortho_path)

    project = Project(name="preview_test", crs=ProjectCrsConfig(source_epsg=4326, project_epsg=31983))
    project.mvs = MvsSummary(
        num_points=1,
        quality="media",
        point_cloud_las_path="unused.las",
        undistorted_image_path=str(images_dir),
        undistorted_reconstruction_path=str(reconstruction_path),
    )
    project.dem = DemSummary(raster_path=str(dem_path), resolution_m=0.5, resolution_source="user-defined")
    project.ortho = OrthoSummary(
        raster_path=ortho_result.raster_path,
        width_px=ortho_result.width_px,
        height_px=ortho_result.height_px,
        resolution_m=ortho_result.resolution_m,
        num_cameras_used=ortho_result.num_cameras_used,
        num_valid_pixels=ortho_result.num_valid_pixels,
        num_nodata_pixels=ortho_result.num_nodata_pixels,
    )

    window = MainWindow(project)
    window._sync_ui_to_project_state()  # must not raise

    assert window.ortho_button.isEnabled() is True
    assert "Câmeras usadas: 6" in window.ortho_summary.toPlainText()
    assert window.dem_axes.images or window.dem_axes.collections  # something was actually drawn
    assert window.ortho_axes.images
