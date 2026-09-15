"""Fase 7: interface completa.

Árvore de projeto (Projeto / Imagens / Câmeras / Tie Points / Point Cloud /
DEM / Orthomosaic, conforme o briefing original) substituindo a janela de
página única das Fases 1-6. Cada etapa de longa duração (Alinhar, Ajustar,
Nuvem densa, DEM, Ortomosaico) agora roda em segundo plano
(`gui.worker.PipelineWorker`) em vez de bloquear o event loop do Qt, então
a árvore e o monitor de CPU/RAM/GPU/VRAM continuam responsivos durante todo
o processamento -- e, onde a chamada pycolmap subjacente suporta (ver o
docstring de `gui.worker`), o botão Cancelar de fato interrompe a operação
via `pycolmap.CancellationToken`, nunca um botão decorativo que não faz
nada (por isso ele simplesmente não aparece para o Ajuste GNSS, cujo
`Ceres::Solve` não expõe esse gancho nesta versão do pycolmap).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from htrmapper.ba.weighted_bundle_adjustment import BaConfig, run_gnss_weighted_bundle_adjustment
from htrmapper.core import theme
from htrmapper.core.project import DemSummary, ImageRecord, MvsSummary, OrthoSummary, Project
from htrmapper.core.report import build_report_from_project, render_html
from htrmapper.core.system_info import sample_cpu_ram_usage, sample_live_usage
from htrmapper.dem.generation import DemConfig, run_dem_generation
from htrmapper.geo.crs import (
    CoordinateReferenceSystem,
    GeodeticPoint,
    GeodeticTransformer,
    wgs84,
)
from htrmapper.gui.worker import PipelineWorker
from htrmapper.io.image_import import import_folder
from htrmapper.mvs.dense import MvsConfig, run_dense_reconstruction
from htrmapper.ortho.orthomosaic import OrthoConfig, run_orthomosaic_generation
from htrmapper.sfm.pipeline import SfmConfig, run_structure_from_motion

_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {theme.BACKGROUND};
    color: {theme.TEXT};
    font-size: 13px;
}}
QLabel#statusLabel {{
    color: {theme.TEXT_MUTED};
}}
QLabel.monitor {{
    color: {theme.TEXT_MUTED};
    padding: 0 8px;
}}
QPushButton {{
    background-color: {theme.PRIMARY_DARK};
    color: {theme.BACKGROUND};
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {theme.PRIMARY};
}}
QPushButton:pressed {{
    background-color: {theme.PRIMARY_DARK};
}}
QPushButton:disabled {{
    background-color: {theme.BORDER};
    color: {theme.TEXT_MUTED};
}}
QTableWidget, QTextEdit, QTreeWidget {{
    background-color: {theme.BACKGROUND};
    alternate-background-color: {theme.SURFACE};
    gridline-color: {theme.BORDER};
    border: 1px solid {theme.BORDER};
    selection-background-color: {theme.PRIMARY};
    selection-color: {theme.BACKGROUND};
}}
QHeaderView::section {{
    background-color: {theme.PRIMARY_DARK};
    color: {theme.BACKGROUND};
    padding: 4px;
    border: none;
    font-weight: bold;
}}
QTreeWidget::item {{
    padding: 3px;
}}
QSplitter::handle {{
    background-color: {theme.BORDER};
}}
QProgressBar {{
    border: 1px solid {theme.BORDER};
    border-radius: 4px;
    text-align: center;
    background-color: {theme.SURFACE};
}}
QProgressBar::chunk {{
    background-color: {theme.PRIMARY};
}}
"""

_TABLE_COLUMNS = [
    ("file_name", "File"),
    ("latitude", "Lat"),
    ("longitude", "Lon"),
    ("altitude", "Alt (m)"),
    ("gimbal_yaw_deg", "Yaw"),
    ("camera_model", "Camera"),
    ("position_valid", "GPS OK"),
]

_TREE_SECTIONS = [
    ("imagens", "Imagens"),
    ("cameras", "Câmeras"),
    ("tie_points", "Tie Points"),
    ("point_cloud", "Point Cloud"),
    ("dem", "DEM"),
    ("orthomosaic", "Orthomosaic"),
]


def _pending(phase: int, note: str = "") -> str:
    text = f"Não disponível — calculado na Fase {phase}"
    return f"{text} ({note})" if note else text


# Process-wide, not per-window: `fork()` (what `subprocess.run` uses for
# nvidia-smi on POSIX) forks the WHOLE process, so a pipeline worker
# running in ANY window's background thread is a hazard for a subprocess
# call issued from ANY OTHER window's monitor timer, not just its own --
# a per-instance guard is not enough. In the shipped app there is only
# ever one `MainWindow`, so this is equivalent to a per-instance flag in
# practice, but it stays correct if that ever changes.
_active_pipeline_worker_count = 0


class MainWindow(QMainWindow):
    def __init__(self, project: Project | None = None):
        super().__init__()
        self.setWindowTitle("HTRMapper — Interface completa")
        self.resize(1300, 750)

        self.project = project or Project(name="untitled")
        self._active_worker: PipelineWorker | None = None

        self.setStyleSheet(_STYLESHEET)
        self._build_ui()
        self._sync_ui_to_project_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        splitter = QSplitter()
        root_layout.addWidget(splitter, 1)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(220)
        self.project_item = QTreeWidgetItem([self.project.name])
        self.project_item.setData(0, Qt.ItemDataRole.UserRole, "projeto")
        self.tree.addTopLevelItem(self.project_item)
        self.tree_items: dict[str, QTreeWidgetItem] = {}
        for key, label in _TREE_SECTIONS:
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            self.project_item.addChild(item)
            self.tree_items[key] = item
        self.tree.expandAll()
        self.tree.currentItemChanged.connect(self._on_tree_selection_changed)
        splitter.addWidget(self.tree)

        self.pages = QStackedWidget()
        self._page_index: dict[str, int] = {}
        self._page_index["projeto"] = self.pages.addWidget(self._build_projeto_page())
        self._page_index["imagens"] = self.pages.addWidget(self._build_imagens_page())
        self._page_index["cameras"] = self.pages.addWidget(self._build_cameras_page())
        self._page_index["tie_points"] = self.pages.addWidget(self._build_tie_points_page())
        self._page_index["point_cloud"] = self.pages.addWidget(self._build_point_cloud_page())
        self._page_index["dem"] = self.pages.addWidget(self._build_dem_page())
        self._page_index["orthomosaic"] = self.pages.addWidget(self._build_orthomosaic_page())
        splitter.addWidget(self.pages)
        splitter.setSizes([260, 1040])

        self.tree.setCurrentItem(self.project_item)

        root_layout.addLayout(self._build_status_bar())

    def _build_status_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()

        self.status_label = QLabel("Nenhum projeto carregado.")
        self.status_label.setObjectName("statusLabel")
        bar.addWidget(self.status_label, 1)

        self.phase_label = QLabel("")
        self.phase_label.setObjectName("statusLabel")
        bar.addWidget(self.phase_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setVisible(False)
        bar.addWidget(self.progress_bar)

        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        bar.addWidget(self.cancel_button)

        self.cpu_label = QLabel("CPU: --")
        self.cpu_label.setProperty("class", "monitor")
        self.ram_label = QLabel("RAM: --")
        self.ram_label.setProperty("class", "monitor")
        self.gpu_label = QLabel("GPU: --")
        self.gpu_label.setProperty("class", "monitor")
        for label in (self.cpu_label, self.ram_label, self.gpu_label):
            label.setStyleSheet(f"color: {theme.TEXT_MUTED}; padding: 0 8px;")
            bar.addWidget(label)

        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self._update_system_monitor)
        self.monitor_timer.start(1000)
        self._update_system_monitor()

        return bar

    # -- Projeto page --------------------------------------------------

    def _build_projeto_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        buttons = QHBoxLayout()
        open_button = QPushButton("Importar pasta de imagens…")
        open_button.clicked.connect(self._on_import_clicked)
        buttons.addWidget(open_button)

        load_button = QPushButton("Abrir projeto…")
        load_button.clicked.connect(self._on_open_project_clicked)
        buttons.addWidget(load_button)

        self.save_project_button = QPushButton("Salvar projeto…")
        self.save_project_button.clicked.connect(self._on_save_project_clicked)
        buttons.addWidget(self.save_project_button)

        self.generate_report_button = QPushButton("Gerar relatório…")
        self.generate_report_button.clicked.connect(self._on_generate_report_clicked)
        buttons.addWidget(self.generate_report_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.project_summary = QTextEdit()
        self.project_summary.setReadOnly(True)
        layout.addWidget(self.project_summary)
        return page

    # -- Imagens page ----------------------------------------------------

    def _build_imagens_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.table = QTableWidget()
        self.table.setColumnCount(len(_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in _TABLE_COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        return page

    # -- Câmeras page -----------------------------------------------------

    def _build_cameras_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        buttons = QHBoxLayout()
        self.align_button = QPushButton("Alinhar (SfM)…")
        self.align_button.clicked.connect(self._on_align_clicked)
        buttons.addWidget(self.align_button)

        self.adjust_button = QPushButton("Ajustar (GNSS)…")
        self.adjust_button.clicked.connect(self._on_adjust_clicked)
        buttons.addWidget(self.adjust_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.cameras_figure = Figure(figsize=(5, 5), facecolor=theme.BACKGROUND)
        self.cameras_canvas = FigureCanvasQTAgg(self.cameras_figure)
        self.cameras_axes = self.cameras_figure.add_subplot(111)
        layout.addWidget(self.cameras_canvas)
        return page

    # -- Tie Points page --------------------------------------------------

    def _build_tie_points_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.tie_points_summary = QTextEdit()
        self.tie_points_summary.setReadOnly(True)
        layout.addWidget(self.tie_points_summary)
        return page

    # -- Point Cloud page ---------------------------------------------------

    def _build_point_cloud_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        buttons = QHBoxLayout()
        self.dense_button = QPushButton("Nuvem densa…")
        self.dense_button.clicked.connect(self._on_dense_clicked)
        buttons.addWidget(self.dense_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.point_cloud_summary = QTextEdit()
        self.point_cloud_summary.setReadOnly(True)
        layout.addWidget(self.point_cloud_summary)
        return page

    # -- DEM page -----------------------------------------------------------

    def _build_dem_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        buttons = QHBoxLayout()
        self.dem_button = QPushButton("Gerar DEM…")
        self.dem_button.clicked.connect(self._on_dem_clicked)
        buttons.addWidget(self.dem_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.dem_summary = QTextEdit()
        self.dem_summary.setReadOnly(True)
        self.dem_summary.setMaximumHeight(120)
        layout.addWidget(self.dem_summary)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.dem_figure = Figure(figsize=(5, 5), facecolor=theme.BACKGROUND)
        self.dem_canvas = FigureCanvasQTAgg(self.dem_figure)
        self.dem_axes = self.dem_figure.add_subplot(111)
        layout.addWidget(self.dem_canvas)
        return page

    # -- Orthomosaic page -------------------------------------------------

    def _build_orthomosaic_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        buttons = QHBoxLayout()
        self.ortho_button = QPushButton("Gerar Ortomosaico…")
        self.ortho_button.clicked.connect(self._on_ortho_clicked)
        buttons.addWidget(self.ortho_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.ortho_summary = QTextEdit()
        self.ortho_summary.setReadOnly(True)
        self.ortho_summary.setMaximumHeight(120)
        layout.addWidget(self.ortho_summary)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.ortho_figure = Figure(figsize=(5, 5), facecolor=theme.BACKGROUND)
        self.ortho_canvas = FigureCanvasQTAgg(self.ortho_figure)
        self.ortho_axes = self.ortho_figure.add_subplot(111)
        layout.addWidget(self.ortho_canvas)
        return page

    def _on_tree_selection_changed(self, current: QTreeWidgetItem, _previous: QTreeWidgetItem) -> None:
        if current is None:
            return
        key = current.data(0, Qt.ItemDataRole.UserRole)
        if key in self._page_index:
            self.pages.setCurrentIndex(self._page_index[key])

    # ------------------------------------------------------------------
    # Background worker plumbing
    # ------------------------------------------------------------------

    def _pipeline_buttons(self) -> list[QPushButton]:
        return [
            self.align_button,
            self.adjust_button,
            self.dense_button,
            self.dem_button,
            self.ortho_button,
        ]

    def _start_worker(self, worker: PipelineWorker, title: str, on_success) -> None:
        if self._active_worker is not None:
            QMessageBox.warning(self, "Operação em andamento", "Aguarde a operação atual terminar.")
            return

        global _active_pipeline_worker_count
        _active_pipeline_worker_count += 1
        self._active_worker = worker
        for button in self._pipeline_buttons():
            button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.phase_label.setText(title)
        self.cancel_button.setVisible(worker.supports_cancellation)
        self.cancel_button.setEnabled(True)

        worker.phase_changed.connect(self._on_phase_changed)
        worker.progress_changed.connect(self._on_progress_changed)
        worker.finished_ok.connect(lambda result: self._on_worker_finished(result, on_success))
        worker.failed.connect(lambda exc: self._on_worker_failed(title, exc))
        worker.cancelled.connect(lambda: self._on_worker_cancelled(title))
        worker.start()

    def _on_phase_changed(self, phase: str) -> None:
        self.progress_bar.setRange(0, 0)
        self.phase_label.setText(phase)

    def _on_progress_changed(self, done: int, total: int) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)
        self.phase_label.setText(f"{done}/{total}")

    def _end_worker(self) -> None:
        global _active_pipeline_worker_count
        _active_pipeline_worker_count -= 1
        self._active_worker = None
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)

    def _on_worker_finished(self, result, on_success) -> None:
        # Clear the busy visuals (progress bar/cancel button) before the
        # completion dialog and before `on_success` runs, so a user isn't
        # shown "operation finished" alongside stale "still running" UI.
        # `on_success` (e.g. `_handle_align_result`) is what actually
        # writes the new summary into `self.project`, so the final
        # `_sync_ui_to_project_state()` call -- which decides every
        # button-enablement and panel refresh -- must come after it, not
        # before, or it would act on the stale, pre-result project state.
        self._end_worker()
        on_success(result)
        self._sync_ui_to_project_state()

    def _on_worker_failed(self, title: str, exc: Exception) -> None:
        self._end_worker()
        self.phase_label.setText("")
        self._sync_ui_to_project_state()  # project unchanged, but buttons disabled for the run need re-enabling
        QMessageBox.critical(self, f"Falha: {title}", str(exc))

    def _on_worker_cancelled(self, title: str) -> None:
        self._end_worker()
        self.phase_label.setText("")
        self._sync_ui_to_project_state()
        self.status_label.setText(f"{title}: cancelado pelo usuário.")

    def _on_cancel_clicked(self) -> None:
        if self._active_worker is not None:
            self._active_worker.cancel()
            self.cancel_button.setEnabled(False)
            self.phase_label.setText("Cancelando…")

    def _update_system_monitor(self) -> None:
        if _active_pipeline_worker_count > 0:
            # Never call nvidia-smi (a subprocess -- forks the whole
            # process on POSIX) while ANY pipeline worker's background
            # thread, in this or any other window, may be deep inside
            # heavily multi-threaded native code (COLMAP/SIFT): forking a
            # multi-threaded process can deadlock if another thread holds
            # a lock at that instant, a real hang reproduced empirically
            # during Fase 7 testing. CPU/RAM (psutil reading /proc, no
            # subprocess) stay live.
            cpu_percent, ram_used_gb, ram_total_gb = sample_cpu_ram_usage()
            self.cpu_label.setText(f"CPU: {cpu_percent:.0f}%")
            self.ram_label.setText(f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f} GB")
            self.gpu_label.setText("GPU: -- (retomado ao final da operação)")
            return

        usage = sample_live_usage()
        self.cpu_label.setText(f"CPU: {usage.cpu_percent:.0f}%")
        self.ram_label.setText(f"RAM: {usage.ram_used_gb:.1f}/{usage.ram_total_gb:.1f} GB")
        if usage.gpus:
            gpu = usage.gpus[0]
            self.gpu_label.setText(
                f"GPU: {gpu.utilization_percent:.0f}% ({gpu.vram_used_mb:.0f}/{gpu.vram_total_mb:.0f} MB)"
            )
        else:
            self.gpu_label.setText("GPU: sem GPU NVIDIA")

    # ------------------------------------------------------------------
    # Project-level actions
    # ------------------------------------------------------------------

    def _on_import_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Selecionar pasta de imagens")
        if not folder:
            return
        records, report = import_folder(Path(folder))
        if not records:
            QMessageBox.warning(self, "Importação", "Nenhuma imagem suportada encontrada na pasta.")
            return
        self.project.images = records
        self.project.sfm = None
        self.project.ba = None
        self.project.mvs = None
        self.project.dem = None
        self.project.ortho = None

        box = QMessageBox(self)
        box.setWindowTitle("Importação concluída")
        box.setIcon(QMessageBox.Icon.Warning if report.has_problems else QMessageBox.Icon.Information)
        box.setText(report.short_summary())
        box.setDetailedText("\n".join(report.summary_lines()))
        box.exec()

        self._sync_ui_to_project_state()

    def _on_open_project_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Abrir projeto", "", "Projeto HTRMapper (*.json)")
        if not path:
            return
        try:
            self.project = Project.load(Path(path))
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Falha ao abrir projeto", str(exc))
            return
        self.project_item.setText(0, self.project.name)
        self._sync_ui_to_project_state()
        QMessageBox.information(self, "Projeto aberto", f"Projeto carregado de:\n{path}")

    def _on_save_project_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar projeto", f"{self.project.name}.json", "Projeto HTRMapper (*.json)"
        )
        if not path:
            return
        self.project.save(Path(path))
        QMessageBox.information(self, "Projeto salvo", f"Projeto salvo em:\n{path}")

    def _on_generate_report_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Gerar relatório", f"{self.project.name}_relatorio.html", "Relatório HTML (*.html)"
        )
        if not path:
            return
        report = build_report_from_project(self.project)
        Path(path).write_text(render_html(report), encoding="utf-8")
        QMessageBox.information(self, "Relatório gerado", f"Relatório salvo em:\n{path}")

    # ------------------------------------------------------------------
    # Câmeras: Fase 2 (Alinhar) / Fase 3 (Ajustar)
    # ------------------------------------------------------------------

    def _on_align_clicked(self) -> None:
        if not self.project.images:
            QMessageBox.warning(self, "Alinhamento (SfM)", "Importe imagens primeiro.")
            return
        workdir = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta de trabalho para o alinhamento (banco COLMAP, reconstrução)"
        )
        if not workdir:
            return

        worker = PipelineWorker(run_structure_from_motion, self.project, Path(workdir), SfmConfig())
        self._start_worker(worker, "Alinhamento (SfM)", self._handle_align_result)

    def _handle_align_result(self, result) -> None:
        self.project.sfm = result.to_project_summary()

        short = (
            f"Alinhamento concluído: {result.num_registered}/{result.num_images_input} imagens registradas, "
            f"{result.num_points3d} tie points, erro de reprojeção {result.mean_reprojection_error_px:.3f} px "
            f"(inicial, sem peso GNSS)."
            if result.mean_reprojection_error_px is not None
            else f"Alinhamento concluído: {result.num_registered}/{result.num_images_input} imagens registradas."
        )
        detail_lines = [
            f"Matching strategy: {result.matching_strategy}",
            f"Registered: {result.num_registered} / {result.num_images_input}",
            f"Unregistered images: {result.unregistered_image_names}",
            f"Tie points (3D): {result.num_points3d}",
            f"Observations (projections): {result.num_observations}",
            f"Georeferenced: {result.georeferenced} -- {result.georeferencing_note}",
        ]

        box = QMessageBox(self)
        box.setWindowTitle("Alinhamento (SfM)")
        box.setIcon(QMessageBox.Icon.Information if result.success else QMessageBox.Icon.Warning)
        box.setText(short)
        box.setDetailedText("\n".join(detail_lines))
        box.exec()

        if result.georeferenced:
            self._plot_aligned_positions(result.reconstruction_path)
        self.status_label.setText(f"{len(self.project.images)} imagem(ns) carregada(s). Alinhamento executado.")

    def _on_adjust_clicked(self) -> None:
        if self.project.sfm is None or not self.project.sfm.reconstruction_path:
            QMessageBox.warning(self, "Ajuste (GNSS)", "Execute o alinhamento (SfM) primeiro.")
            return

        workdir = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta de trabalho para o ajuste (reconstrução refinada)"
        )
        if not workdir:
            return

        worker = PipelineWorker(
            run_gnss_weighted_bundle_adjustment,
            self.project,
            Path(self.project.sfm.reconstruction_path),
            Path(workdir),
            BaConfig(),
        )
        self._start_worker(worker, "Ajuste (GNSS)", self._handle_adjust_result)

    def _handle_adjust_result(self, result) -> None:
        self.project.ba = result.to_project_summary()

        short = (
            f"Ajuste concluído: RMSE total {result.rmse_total_cm:.3f} cm "
            f"({result.num_images_with_gnss_prior} câmeras com observação GNSS), "
            f"erro de reprojeção final {result.mean_reprojection_error_px:.3f} px."
        )
        detail_lines = [
            f"Termination: {result.termination_type} (converged: {result.converged})",
            f"RMSE X={result.rmse_x_cm:.3f} cm, Y={result.rmse_y_cm:.3f} cm, Z={result.rmse_z_cm:.3f} cm",
            f"RMSE XY={result.rmse_xy_cm:.3f} cm, Total={result.rmse_total_cm:.3f} cm",
            f"Max error: {result.max_error_cm:.3f} cm",
        ]

        box = QMessageBox(self)
        box.setWindowTitle("Ajuste (GNSS)")
        box.setIcon(QMessageBox.Icon.Information if result.converged else QMessageBox.Icon.Warning)
        box.setText(short)
        box.setDetailedText("\n".join(detail_lines))
        box.exec()

        self._plot_aligned_positions(result.reconstruction_path, label="ajustado (GNSS ponderado)")
        self.status_label.setText(f"{len(self.project.images)} imagem(ns) carregada(s). Ajuste GNSS executado.")

    # ------------------------------------------------------------------
    # Point Cloud: Fase 4 (Nuvem densa)
    # ------------------------------------------------------------------

    def _on_dense_clicked(self) -> None:
        if self.project.sfm is None or not self.project.sfm.reconstruction_path:
            QMessageBox.warning(self, "Nuvem densa", "Execute o alinhamento (SfM) primeiro.")
            return

        quality, ok = QInputDialog.getItem(
            self, "Qualidade da nuvem densa", "Qualidade:", ["baixa", "media", "alta", "muito_alta"], 1, False
        )
        if not ok:
            return

        workdir = QFileDialog.getExistingDirectory(self, "Selecionar pasta de trabalho para a nuvem densa")
        if not workdir:
            return

        image_parents = {Path(img.path).resolve().parent for img in self.project.images}
        if len(image_parents) != 1:
            QMessageBox.critical(
                self, "Nuvem densa", "As imagens estão em pastas diferentes; não é possível determinar uma raiz única."
            )
            return
        image_root = next(iter(image_parents))

        reconstruction_path = Path(
            self.project.ba.reconstruction_path
            if self.project.ba and self.project.ba.reconstruction_path
            else self.project.sfm.reconstruction_path
        )

        worker = PipelineWorker(
            run_dense_reconstruction,
            self.project,
            reconstruction_path,
            image_root,
            Path(workdir),
            MvsConfig(quality=quality),
        )
        self._start_worker(worker, "Nuvem densa", self._handle_dense_result)

    def _handle_dense_result(self, result) -> None:
        self.project.mvs = MvsSummary(
            num_points=result.num_points,
            quality=result.quality,
            point_cloud_las_path=result.point_cloud_las_path,
            point_cloud_native_path=result.point_cloud_native_path,
            undistorted_image_path=result.undistorted_image_path,
            undistorted_reconstruction_path=result.undistorted_reconstruction_path,
        )

        QMessageBox.information(
            self,
            "Nuvem densa gerada",
            f"{result.num_points} pontos (qualidade '{result.quality}').\nLAS: {result.point_cloud_las_path}",
        )
        self.status_label.setText(f"{len(self.project.images)} imagem(ns) carregada(s). Nuvem densa gerada.")

    # ------------------------------------------------------------------
    # DEM: Fase 5
    # ------------------------------------------------------------------

    def _on_dem_clicked(self) -> None:
        if self.project.mvs is None or not self.project.mvs.point_cloud_las_path:
            QMessageBox.warning(self, "DEM", "Gere a nuvem densa primeiro.")
            return

        resolution_str, ok = QInputDialog.getText(
            self, "Resolução do DEM", "Resolução em m/pixel (deixe em branco para automático):"
        )
        if not ok:
            return
        resolution_m = None
        if resolution_str.strip():
            try:
                resolution_m = float(resolution_str.strip())
            except ValueError:
                QMessageBox.warning(self, "DEM", f"Resolução inválida: {resolution_str!r}")
                return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Salvar DEM", f"{self.project.name}_dem.tif", "GeoTIFF (*.tif)"
        )
        if not output_path:
            return

        worker = PipelineWorker(
            run_dem_generation,
            Path(self.project.mvs.point_cloud_las_path),
            self.project.crs.effective_export_epsg,
            Path(output_path),
            DemConfig(resolution_m=resolution_m),
        )
        self._start_worker(worker, "DEM", self._handle_dem_result)

    def _handle_dem_result(self, result) -> None:
        self.project.dem = DemSummary(
            raster_path=result.raster_path,
            resolution_m=result.resolution_m,
            resolution_source=result.resolution_source,
            width_px=result.width_px,
            height_px=result.height_px,
            min_elevation_m=result.min_elevation_m,
            max_elevation_m=result.max_elevation_m,
            num_points_used=result.num_points_used,
            num_points_filtered_as_outliers=result.num_points_filtered_as_outliers,
            point_density_per_m2=result.point_density_per_m2,
        )

        QMessageBox.information(
            self,
            "DEM gerado",
            f"{result.width_px}x{result.height_px}px, resolução {result.resolution_m:.3f} m/px "
            f"({result.resolution_source}).\nElevação: [{result.min_elevation_m:.2f}, {result.max_elevation_m:.2f}] m\n"
            f"GeoTIFF: {result.raster_path}",
        )
        self.status_label.setText(f"{len(self.project.images)} imagem(ns) carregada(s). DEM gerado.")

    # ------------------------------------------------------------------
    # Orthomosaic: Fase 6
    # ------------------------------------------------------------------

    def _on_ortho_clicked(self) -> None:
        if self.project.mvs is None or not self.project.mvs.undistorted_reconstruction_path:
            QMessageBox.warning(self, "Ortomosaico", "Gere a nuvem densa primeiro.")
            return
        if self.project.dem is None or not self.project.dem.raster_path:
            QMessageBox.warning(self, "Ortomosaico", "Gere o DEM primeiro.")
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Salvar ortomosaico", f"{self.project.name}_ortho.tif", "GeoTIFF (*.tif)"
        )
        if not output_path:
            return

        worker = PipelineWorker(
            run_orthomosaic_generation,
            Path(self.project.mvs.undistorted_reconstruction_path),
            Path(self.project.mvs.undistorted_image_path),
            Path(self.project.dem.raster_path),
            Path(output_path),
            OrthoConfig(),
        )
        self._start_worker(worker, "Ortomosaico", self._handle_ortho_result)

    def _handle_ortho_result(self, result) -> None:
        self.project.ortho = OrthoSummary(
            raster_path=result.raster_path,
            width_px=result.width_px,
            height_px=result.height_px,
            resolution_m=result.resolution_m,
            num_cameras_used=result.num_cameras_used,
            num_valid_pixels=result.num_valid_pixels,
            num_nodata_pixels=result.num_nodata_pixels,
        )

        QMessageBox.information(
            self,
            "Ortomosaico gerado",
            f"{result.width_px}x{result.height_px}px, resolução {result.resolution_m:.3f} m/px.\n"
            f"Câmeras usadas: {result.num_cameras_used}.\nGeoTIFF: {result.raster_path}",
        )
        self.status_label.setText(f"{len(self.project.images)} imagem(ns) carregada(s). Ortomosaico gerado.")

    # ------------------------------------------------------------------
    # Panel refresh -- keeps every tree section's content in sync with
    # `self.project`, whether it changed via a finished worker, a fresh
    # import, or opening a saved project file.
    # ------------------------------------------------------------------

    def _sync_ui_to_project_state(self) -> None:
        has_images = bool(self.project.images)
        has_sfm = self.project.sfm is not None and bool(self.project.sfm.reconstruction_path)
        has_mvs = self.project.mvs is not None and bool(self.project.mvs.point_cloud_las_path)
        has_undistorted = self.project.mvs is not None and bool(self.project.mvs.undistorted_reconstruction_path)
        has_dem = self.project.dem is not None and bool(self.project.dem.raster_path)

        self.save_project_button.setEnabled(has_images)
        self.generate_report_button.setEnabled(has_images)
        self.align_button.setEnabled(has_images)
        self.adjust_button.setEnabled(has_sfm)
        self.dense_button.setEnabled(has_sfm)
        self.dem_button.setEnabled(has_mvs)
        self.ortho_button.setEnabled(has_undistorted and has_dem)

        self.status_label.setText(f"{len(self.project.images)} imagem(ns) carregada(s).")

        self._refresh_projeto_page()
        self._populate_table(self.project.images)
        self._refresh_cameras_page()
        self._refresh_tie_points_page()
        self._refresh_point_cloud_page()
        self._refresh_dem_page()
        self._refresh_orthomosaic_page()

    def _refresh_projeto_page(self) -> None:
        lines = [
            f"Nome: {self.project.name}",
            f"CRS do projeto: EPSG:{self.project.crs.project_epsg}",
            f"CRS de origem (GNSS): EPSG:{self.project.crs.source_epsg}",
            f"Precisão GNSS/PPK configurada: XY sigma={self.project.gnss_accuracy.accuracy.xy_sigma_m} m, "
            f"Z sigma={self.project.gnss_accuracy.accuracy.z_sigma_m} m",
            f"Imagens importadas: {len(self.project.images)}",
        ]
        self.project_summary.setPlainText("\n".join(lines))

    def _refresh_cameras_page(self) -> None:
        if self.project.sfm is not None and self.project.sfm.georeferenced and self.project.sfm.reconstruction_path:
            try:
                label = "ajustado (GNSS ponderado)" if self.project.ba else "SfM alinhado"
                path = self.project.ba.reconstruction_path if self.project.ba else self.project.sfm.reconstruction_path
                self._plot_aligned_positions(path, label=label)
                return
            except Exception:
                pass  # reconstruction no longer on disk -- fall back to raw GNSS below
        self._plot_camera_positions(self.project.images)

    def _refresh_tie_points_page(self) -> None:
        if self.project.sfm is None:
            self.tie_points_summary.setPlainText(_pending(2, "execute o alinhamento (SfM)"))
            return
        sfm = self.project.sfm
        lines = [
            f"Estratégia de matching: {sfm.matching_strategy}",
            f"Imagens registradas: {sfm.num_registered} / {sfm.num_images_input}",
            f"Tie points (3D): {sfm.num_points3d}",
            f"Observações (projeções): {sfm.num_observations}",
            "Erro de reprojeção inicial (Fase 2, sem peso GNSS): "
            + (f"{sfm.mean_reprojection_error_px:.4f} px" if sfm.mean_reprojection_error_px is not None else "N/A"),
        ]
        if self.project.ba is not None:
            lines.append(
                "Erro de reprojeção final (Fase 3, com peso GNSS): "
                + (f"{self.project.ba.mean_reprojection_error_px:.4f} px" if self.project.ba.mean_reprojection_error_px is not None else "N/A")
            )
        else:
            lines.append(f"Erro de reprojeção final: {_pending(3, 'execute o ajuste GNSS')}")
        self.tie_points_summary.setPlainText("\n".join(lines))

    def _refresh_point_cloud_page(self) -> None:
        if self.project.mvs is None:
            self.point_cloud_summary.setPlainText(_pending(4, "gere a nuvem densa"))
            return
        mvs = self.project.mvs
        lines = [
            f"Pontos: {mvs.num_points}",
            f"Qualidade: {mvs.quality}",
            f"LAS: {mvs.point_cloud_las_path}",
        ]
        self.point_cloud_summary.setPlainText("\n".join(lines))

    def _refresh_dem_page(self) -> None:
        if self.project.dem is None:
            self.dem_summary.setPlainText(_pending(5, "gere o DEM"))
            self.dem_axes.clear()
            self.dem_canvas.draw_idle()
            return
        dem = self.project.dem
        lines = [
            f"{dem.width_px}x{dem.height_px}px, resolução {dem.resolution_m:.3f} m/px ({dem.resolution_source})",
            f"Elevação: [{dem.min_elevation_m:.2f}, {dem.max_elevation_m:.2f}] m",
            f"Pontos usados: {dem.num_points_used} ({dem.num_points_filtered_as_outliers} filtrados como outlier)",
            f"Densidade: {dem.point_density_per_m2:.2f} pontos/m²",
        ]
        self.dem_summary.setPlainText("\n".join(lines))
        self._preview_raster(dem.raster_path, self.dem_axes, self.dem_canvas, mode="elevation")

    def _refresh_orthomosaic_page(self) -> None:
        if self.project.ortho is None:
            self.ortho_summary.setPlainText(_pending(6, "gere o ortomosaico"))
            self.ortho_axes.clear()
            self.ortho_canvas.draw_idle()
            return
        ortho = self.project.ortho
        lines = [
            f"{ortho.width_px}x{ortho.height_px}px, resolução {ortho.resolution_m:.3f} m/px",
            f"Câmeras usadas: {ortho.num_cameras_used}",
            f"Pixels válidos: {ortho.num_valid_pixels} ({ortho.num_nodata_pixels} sem cobertura/NoData)",
        ]
        self.ortho_summary.setPlainText("\n".join(lines))
        self._preview_raster(ortho.raster_path, self.ortho_axes, self.ortho_canvas, mode="rgba")

    # ------------------------------------------------------------------
    # Table / plotting helpers
    # ------------------------------------------------------------------

    def _populate_table(self, records: list[ImageRecord]) -> None:
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = {
                "file_name": record.file_name,
                "latitude": f"{record.latitude:.6f}" if record.latitude is not None else "-",
                "longitude": f"{record.longitude:.6f}" if record.longitude is not None else "-",
                "altitude": f"{record.altitude:.1f}" if record.altitude is not None else "-",
                "gimbal_yaw_deg": f"{record.gimbal_yaw_deg:.1f}" if record.gimbal_yaw_deg is not None else "-",
                "camera_model": record.camera_model or "-",
                "position_valid": "sim" if record.position_valid else "não",
            }
            for col, (key, _label) in enumerate(_TABLE_COLUMNS):
                self.table.setItem(row, col, QTableWidgetItem(values[key]))

    def _plot_camera_positions(self, records: list[ImageRecord]) -> None:
        if not records:
            self.cameras_axes.clear()
            self.cameras_canvas.draw_idle()
            return
        project_crs = CoordinateReferenceSystem(self.project.crs.project_epsg)
        transformer = GeodeticTransformer(wgs84(), project_crs)

        xs, ys = [], []
        for record in records:
            if not record.position_valid:
                continue
            point = transformer.forward(
                GeodeticPoint(lon=record.longitude, lat=record.latitude, alt=record.altitude)
            )
            xs.append(point.x)
            ys.append(point.y)

        self._plot_points(xs, ys, f"Posições das câmeras — GNSS bruto ({len(xs)} válidas)")

    def _plot_aligned_positions(self, reconstruction_path: str, label: str = "SfM alinhado") -> None:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(reconstruction_path)
        xs, ys = [], []
        for image_id in reconstruction.reg_image_ids():
            center = reconstruction.image(image_id).projection_center()
            xs.append(float(center[0]))
            ys.append(float(center[1]))

        self._plot_points(xs, ys, f"Posições das câmeras — {label} ({len(xs)} registradas)")

    def _plot_points(self, xs: list[float], ys: list[float], title: str) -> None:
        from matplotlib.ticker import ScalarFormatter

        self.cameras_axes.clear()
        self.cameras_axes.set_facecolor(theme.BACKGROUND)
        if xs:
            self.cameras_axes.scatter(
                xs, ys, c=theme.PRIMARY, s=30, edgecolors=theme.PRIMARY_DARK, linewidths=0.6
            )
            self.cameras_axes.set_aspect("equal", adjustable="datalim")
        self.cameras_axes.set_xlabel(f"Este (m) — EPSG:{self.project.crs.project_epsg}", color=theme.PRIMARY_DARK)
        self.cameras_axes.set_ylabel("Norte (m)", color=theme.PRIMARY_DARK)
        self.cameras_axes.set_title(title, color=theme.PRIMARY_DARK)
        # UTM coordinates (easting/northing) are 6-7 digit numbers -- by
        # default matplotlib shows them as a small offset ("+7.548e6") plus
        # short relative tick labels, which hides the real coordinate
        # value. Show the real, full number on every tick instead (never a
        # value the user can't read off directly), rotated vertically so
        # those wider labels don't eat into the plot area.
        for axis in (self.cameras_axes.xaxis, self.cameras_axes.yaxis):
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
        self.cameras_axes.tick_params(axis="y", labelrotation=90)
        self.cameras_axes.tick_params(colors=theme.TEXT_MUTED)
        for spine in self.cameras_axes.spines.values():
            spine.set_color(theme.BORDER)
        self.cameras_axes.grid(True, linewidth=0.3, color=theme.BORDER)
        self.cameras_figure.tight_layout()
        self.cameras_canvas.draw_idle()

    def _preview_raster(self, raster_path: str, axes, canvas, mode: str, max_preview_px: int = 1500) -> None:
        """Render a downsampled preview of a DEM/orthomosaic GeoTIFF.

        Decimated purely for on-screen display (never used for any
        measurement) -- a multi-thousand-pixel real orthomosaic rendered
        at full resolution in a Qt/matplotlib widget would be slow and
        wasteful when the window itself is a few hundred pixels wide.
        """
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling

        axes.clear()
        try:
            with rasterio.open(raster_path) as src:
                scale = min(1.0, max_preview_px / max(src.height, src.width))
                out_height = max(1, int(src.height * scale))
                out_width = max(1, int(src.width * scale))
                if mode == "elevation":
                    band = src.read(1, out_shape=(out_height, out_width), resampling=Resampling.bilinear)
                    data = np.where(band == src.nodata, np.nan, band) if src.nodata is not None else band
                    axes.imshow(data, cmap="terrain")
                else:
                    data = src.read(
                        [1, 2, 3, 4], out_shape=(4, out_height, out_width), resampling=Resampling.bilinear
                    )
                    rgba = np.transpose(data, (1, 2, 0))
                    axes.imshow(rgba)
        except (OSError, rasterio.errors.RasterioIOError):
            pass  # raster no longer on disk -- leave the panel blank rather than crash
        axes.axis("off")
        canvas.draw_idle()


def run(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
