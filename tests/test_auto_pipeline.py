"""Automatic processing (pipeline.auto), its levels, time estimates and
PDF report, and the automatic-processing window.

The sandbox has no GPU, so the dense cloud's patch-match is stubbed and
fusion hands back the undistorted sparse model's own points (the same
trick as test_dense_reconstruction); every other step runs for real on a
small synthetic flight.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pycolmap
import pytest
import rasterio

from htrmapper.core.project import Project
from htrmapper.geo.crs import sirgas2000_utm_epsg
from htrmapper.ortho.orthomosaic import OrthoConfig
from htrmapper.pipeline import auto, timing
from htrmapper.pipeline.auto import STEP_KEYS, AutoProcessingError, AutoProcessor, AutoRequest, Callbacks
from htrmapper.pipeline.levels import LEVELS
from htrmapper.sfm.pipeline import SfmConfig
from tests.synthetic_scene import generate_synthetic_flight


@pytest.fixture(scope="module")
def photos(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("voo") / "fotos_voo"
    generate_synthetic_flight(folder)
    return folder


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(pycolmap, "has_cuda", True)
    monkeypatch.setattr(pycolmap, "patch_match_stereo", lambda *a, **k: None)
    monkeypatch.setattr(
        pycolmap,
        "stereo_fusion",
        lambda *a, **k: pycolmap.Reconstruction(str(Path(k["workspace_path"]) / "sparse")),
    )


class Recorder(Callbacks):
    def __init__(self):
        self.steps: list[tuple[str, str]] = []
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        self.lines.append(message)

    def step(self, key: str, status: str, detail: str = "") -> None:
        self.steps.append((key, status))


def _processor(photos: Path, out: Path, level: str = "medio", callbacks=None, **kwargs) -> AutoProcessor:
    return AutoProcessor(
        AutoRequest(photos, out, level, **kwargs),
        callbacks or Recorder(),
        timing_history_path=out.parent / "tempos.json",
    )


# -- levels, CRS, timing --------------------------------------------------------


def test_medio_level_is_the_configuration_validated_against_pix4d():
    medio = LEVELS["medio"]
    assert medio.sfm == SfmConfig()
    assert medio.ba.tie_point_cleanup and medio.ba.refine_focal_length
    assert medio.mvs.quality == "media"
    assert medio.ortho == OrthoConfig()
    assert LEVELS["baixo"].ortho.resolution_scale == 2.0
    assert LEVELS["alto"].mvs.quality == "alta"


@pytest.mark.parametrize(
    "lon, lat, epsg",
    [(-47.3, -22.1, 31983), (-48.0, -22.1, 31983), (-51.0, -23.0, 31982), (-39.0, -10.0, 31984), (-60.0, 2.0, 31975)],
)
def test_sirgas2000_utm_zone_from_coordinates(lon, lat, epsg):
    assert sirgas2000_utm_epsg(lon, lat) == epsg


def test_time_estimate_uses_reference_then_this_pcs_history(tmp_path):
    history = tmp_path / "tempos.json"
    first = timing.estimate("medio", 100, "PC-A", ["densa"], history)["densa"]
    assert not first.from_history
    assert first.seconds == pytest.approx(timing.REFERENCE_SECONDS_PER_PHOTO["medio"]["densa"] * 100)

    timing.record_step("densa", "medio", 50, 500.0, "PC-A", history)
    timing.record_step("densa", "medio", 50, 900.0, "PC-B", history)
    learned = timing.estimate("medio", 100, "PC-A", ["densa"], history)["densa"]
    assert learned.from_history
    assert learned.seconds == pytest.approx(1000.0)  # 10 s/photo measured on PC-A, not PC-B's 18
    assert not timing.estimate("alto", 100, "PC-A", ["densa"], history)["densa"].from_history


def test_format_duration():
    assert timing.format_duration(42) == "42 s"
    assert timing.format_duration(125) == "2 min 05 s"
    assert timing.format_duration(3 * 3600 + 7 * 60) == "3 h 07 min"


def test_ortho_resolution_scale_is_validated():
    with pytest.raises(ValueError):
        OrthoConfig(resolution_scale=0)


def test_work_folder_moves_to_the_wsl_disk_only_for_windows_drives(monkeypatch, tmp_path):
    monkeypatch.setattr("htrmapper.core.platform_paths.is_running_under_wsl", lambda: True)
    windows = auto.default_work_folder(Path("/mnt/c/Users/x/Desktop/saida"), "voo")
    assert windows.parts[:2] != ("/", "mnt") and windows.name.startswith("voo_")
    assert auto.default_work_folder(tmp_path / "saida", "voo") == (tmp_path / "saida").resolve() / "trabalho"
    monkeypatch.setattr("htrmapper.core.platform_paths.is_running_under_wsl", lambda: False)
    assert auto.default_work_folder(Path("/mnt/c/x"), "voo") == Path("/mnt/c/x/trabalho")


# -- preflight ------------------------------------------------------------------------


def test_preflight_reports_missing_gpu_and_empty_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(pycolmap, "has_cuda", False)
    empty = tmp_path / "vazia"
    empty.mkdir()
    problems = _processor(empty, tmp_path / "saida").preflight()
    assert any("pelo menos 3" in p for p in problems)
    assert any("GPU NVIDIA" in p for p in problems)
    with pytest.raises(AutoProcessingError, match="Não é possível começar"):
        _processor(empty, tmp_path / "saida").run()


def test_preflight_refuses_to_overwrite_a_manual_project(photos, tmp_path, fake_gpu):
    out = tmp_path / "projeto_manual"
    out.mkdir()
    manual = Project(name="manual")
    manual.save(out / "projeto.json")
    before = (out / "projeto.json").read_text()
    problems = _processor(photos, out).preflight()
    assert any("modo manual" in p for p in problems)
    with pytest.raises(AutoProcessingError):
        _processor(photos, out).run(resume=False)
    assert (out / "projeto.json").read_text() == before


# -- full runs ----------------------------------------------------------------------------


def test_full_run_produces_products_report_and_log(photos, tmp_path, fake_gpu):
    out = tmp_path / "saida"
    recorder = Recorder()
    result = _processor(photos, out, callbacks=recorder).run()

    assert [k for k, s in recorder.steps if s == "concluido"] == STEP_KEYS
    for key, suffix in (("ortho", "_ortomosaico.tif"), ("dem", "_dem.tif"), ("relatorio", "_relatorio.pdf")):
        assert result.products[key] == str(out / f"fotos_voo{suffix}")
        assert Path(result.products[key]).is_file()
    pdf = Path(result.products["relatorio"]).read_bytes()
    assert pdf.startswith(b"%PDF") and b"%%EOF" in pdf[-1024:] and len(pdf) > 50_000
    # Compact, Metashape-like: cover, survey data, calibration, cameras, DEM, parameters.
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf))
    assert 5 <= pages <= 7

    project = Project.load(out / "projeto.json")
    assert project.crs.project_epsg == 31983  # picked from the photos' position
    assert project.processing.level == "medio"
    assert set(project.processing.steps_done) == set(STEP_KEYS)
    assert project.processing.finished_at
    with rasterio.open(result.products["ortho"]) as src:
        assert src.crs.to_epsg() == 31983
    log = (out / "processamento.log").read_text(encoding="utf-8")
    assert "nível Médio" in log and "PROCESSAMENTO CONCLUÍDO" in log
    assert any("Ortomosaico" in line for line in recorder.lines)
    assert list((out / "trabalho" / "logs").glob("colmap_*"))  # COLMAP's own log lands in the work folder
    assert not list(out.glob("ERRO_*.txt"))
    history = timing._load(out.parent / "tempos.json")
    assert {e["step"] for e in history} == set(STEP_KEYS)


def test_failure_writes_error_file_and_rerun_resumes_from_that_step(photos, tmp_path, fake_gpu, monkeypatch):
    out = tmp_path / "saida"
    import htrmapper.ortho.orthomosaic as ortho_module

    real = ortho_module.run_orthomosaic_generation

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ortho_module, "run_orthomosaic_generation", disk_full)
    recorder = Recorder()
    with pytest.raises(AutoProcessingError) as info:
        _processor(photos, out, callbacks=recorder).run()
    error = info.value
    assert error.step == "ortho"
    assert "disco encheu" in str(error)
    text = error.error_file.read_text(encoding="utf-8")
    assert "Etapa: Ortomosaico" in text and "Traceback" in text and "No space left on device" in text
    assert ("ortho", "erro") in recorder.steps

    monkeypatch.setattr(ortho_module, "run_orthomosaic_generation", real)
    recorder = Recorder()
    result = _processor(photos, out, callbacks=recorder).run()
    assert result.skipped_steps == ["importar", "alinhar", "ajustar", "densa", "dem"]
    assert list(result.step_seconds) == ["ortho", "relatorio"]
    assert Path(result.products["ortho"]).is_file()


def test_rerun_redoes_steps_whose_outputs_were_deleted(photos, tmp_path, fake_gpu):
    out = tmp_path / "saida"
    _processor(photos, out).run()
    Path(out / "fotos_voo_dem.tif").unlink()
    result = _processor(photos, out).run()
    assert result.skipped_steps == ["importar", "alinhar", "ajustar", "densa"]
    assert list(result.step_seconds) == ["dem", "ortho", "relatorio"]


def test_other_level_in_same_folder_is_refused_unless_restarting(photos, tmp_path, fake_gpu):
    out = tmp_path / "saida"
    _processor(photos, out, level="baixo").run()
    existing = _processor(photos, out, level="medio").inspect_existing()
    assert existing.level == "baixo" and existing.finished and existing.same_photos
    with pytest.raises(AutoProcessingError, match="outro processamento"):
        _processor(photos, out, level="medio").run()
    assert not list(out.glob("ERRO_*.txt"))  # refused before starting: nothing to report

    marker = out / "trabalho" / "densa" / "antigo.txt"
    marker.write_text("x")
    result = _processor(photos, out, level="medio").run(resume=False)
    assert not marker.exists()
    assert result.skipped_steps == []
    assert Project.load(out / "projeto.json").processing.level == "medio"


def test_baixo_level_doubles_the_orthomosaic_pixel(photos, tmp_path, fake_gpu):
    baixo = _processor(photos, tmp_path / "baixo", level="baixo").run()
    medio = _processor(photos, tmp_path / "medio", level="medio").run()
    with rasterio.open(baixo.products["ortho"]) as a, rasterio.open(medio.products["ortho"]) as b:
        assert a.res[0] == pytest.approx(2 * b.res[0], rel=0.05)


def test_cancel_stops_before_the_next_step(photos, tmp_path, fake_gpu):
    class CancelAfterImport(Recorder):
        def step(self, key, status, detail=""):
            super().step(key, status, detail)
            if key == "importar" and status == "concluido":
                token.cancel()

    token = pycolmap.CancellationToken()
    recorder = CancelAfterImport()
    processor = AutoProcessor(
        AutoRequest(photos, tmp_path / "saida"), recorder, token, timing_history_path=tmp_path / "t.json"
    )
    with pytest.raises(InterruptedError):
        processor.run()
    assert ("alinhar", "cancelado") in recorder.steps
    assert Project.load(tmp_path / "saida" / "projeto.json").processing.steps_done.keys() == {"importar"}


def test_delete_work_folder_on_success(photos, tmp_path, fake_gpu):
    out = tmp_path / "saida"
    result = _processor(photos, out, delete_workdir_on_success=True).run()
    assert not (out / "trabalho").exists()
    assert Path(result.products["ortho"]).is_file()


def test_cli_processar(photos, tmp_path, fake_gpu, monkeypatch, capsys):
    from htrmapper.cli.main import main

    monkeypatch.setattr(timing, "history_path", lambda: tmp_path / "tempos.json")
    out = tmp_path / "saida"
    assert main(["processar", str(photos), "--saida", str(out), "--nivel", "baixo"]) == 0
    printed = capsys.readouterr().out
    assert "Tempo estimado" in printed and "PROCESSAMENTO CONCLUÍDO" in printed
    assert (out / "fotos_voo_relatorio.pdf").is_file()


# -- window ---------------------------------------------------------------------------------


def test_automatic_window_runs_everything(photos, tmp_path, fake_gpu, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from htrmapper.gui.auto_window import AutoWindow

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(timing, "history_path", lambda: tmp_path / "tempos.json")
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    app = QApplication.instance() or QApplication([])
    window = AutoWindow()
    window.photos_edit.setText(str(photos))
    window._photos_changed()
    assert window.photos_info.text().startswith("6 foto")
    assert window.output_edit.text().endswith("HTRMapper_fotos_voo")
    assert all(label.text().startswith("Tempo estimado") for label in window.level_time.values())
    window.output_edit.setText(str(tmp_path / "saida"))

    window._start()
    deadline = time.time() + 300
    while window.worker is not None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()

    assert set(window.step_state.values()) == {"concluido"}
    assert window.overall_bar.value() == 1000
    assert window.last_report == tmp_path / "saida" / "fotos_voo_relatorio.pdf"
    assert "PROCESSAMENTO CONCLUÍDO" in window.log_view.toPlainText()
    window.close()
    shutil.rmtree(tmp_path / "saida")
