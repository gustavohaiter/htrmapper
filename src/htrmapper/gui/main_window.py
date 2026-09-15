"""Minimal Phase-1 desktop viewer.

Deliberately small: an image table plus a real map of camera positions
computed by transforming each image's WGS84 EXIF coordinates into the
project CRS via `geo.crs` -- driven entirely by actual imported data, not
placeholder/mock content. The full "Project / Images / Cameras / Tie
Points / Point Cloud / DEM / Orthomosaic" tree UI described in the project
brief is a Phase 7 deliverable; this window is the seed it will grow from
(the same `Project`/`ImageRecord` model is reused, not replaced).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from htrmapper.core import theme
from htrmapper.core.project import ImageRecord, Project
from htrmapper.core.report import build_report_from_project, render_html
from htrmapper.geo.crs import (
    CoordinateReferenceSystem,
    GeodeticPoint,
    GeodeticTransformer,
    wgs84,
)
from htrmapper.io.image_import import import_folder
from htrmapper.sfm.pipeline import SfmConfig, SfmError, run_structure_from_motion

_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {theme.BACKGROUND};
    color: {theme.TEXT};
    font-size: 13px;
}}
QLabel#statusLabel {{
    color: {theme.TEXT_MUTED};
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
QTableWidget {{
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
QSplitter::handle {{
    background-color: {theme.BORDER};
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


class MainWindow(QMainWindow):
    def __init__(self, project: Project | None = None):
        super().__init__()
        self.setWindowTitle("HTRMapper — Fase 1: Importação e visualização")
        self.resize(1100, 650)

        self.project = project or Project(name="untitled")

        self.setStyleSheet(_STYLESHEET)
        self._build_ui()
        if self.project.images:
            self._refresh(self.project.images)
            self.save_project_button.setEnabled(True)
            self.generate_report_button.setEnabled(True)
            self.align_button.setEnabled(True)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        open_button = QPushButton("Importar pasta de imagens…")
        open_button.clicked.connect(self._on_import_clicked)
        toolbar.addWidget(open_button)

        self.save_project_button = QPushButton("Salvar projeto…")
        self.save_project_button.setEnabled(False)
        self.save_project_button.clicked.connect(self._on_save_project_clicked)
        toolbar.addWidget(self.save_project_button)

        self.generate_report_button = QPushButton("Gerar relatório…")
        self.generate_report_button.setEnabled(False)
        self.generate_report_button.clicked.connect(self._on_generate_report_clicked)
        toolbar.addWidget(self.generate_report_button)

        self.align_button = QPushButton("Alinhar (SfM)…")
        self.align_button.setEnabled(False)
        self.align_button.clicked.connect(self._on_align_clicked)
        toolbar.addWidget(self.align_button)

        self.status_label = QLabel("Nenhum projeto carregado.")
        self.status_label.setObjectName("statusLabel")
        toolbar.addWidget(self.status_label)
        toolbar.addStretch(1)
        root_layout.addLayout(toolbar)

        splitter = QSplitter()
        root_layout.addWidget(splitter)

        self.table = QTableWidget()
        self.table.setColumnCount(len(_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in _TABLE_COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        splitter.addWidget(self.table)

        # Matplotlib is an optional GUI-extra dependency; imported lazily so
        # the rest of the GUI module still loads without it installed.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 5), facecolor=theme.BACKGROUND)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor(theme.BACKGROUND)
        splitter.addWidget(self.canvas)
        splitter.setSizes([650, 450])

    def _on_import_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Selecionar pasta de imagens")
        if not folder:
            return
        records, report = import_folder(Path(folder))
        if not records:
            QMessageBox.warning(self, "Importação", "Nenhuma imagem suportada encontrada na pasta.")
            return
        self.project.images = records
        self._refresh(records)

        box = QMessageBox(self)
        box.setWindowTitle("Importação concluída")
        box.setIcon(QMessageBox.Icon.Warning if report.has_problems else QMessageBox.Icon.Information)
        box.setText(report.short_summary())
        box.setDetailedText("\n".join(report.summary_lines()))
        box.exec()

        self.save_project_button.setEnabled(True)
        self.generate_report_button.setEnabled(True)
        self.align_button.setEnabled(True)

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

    def _on_align_clicked(self) -> None:
        workdir = QFileDialog.getExistingDirectory(
            self, "Selecionar pasta de trabalho para o alinhamento (banco COLMAP, reconstrução)"
        )
        if not workdir:
            return

        self.setCursor(Qt.CursorShape.WaitCursor)
        self.status_label.setText("Alinhando (features, matching, SfM)... isso pode demorar.")
        QApplication.processEvents()
        try:
            result = run_structure_from_motion(self.project, Path(workdir), SfmConfig())
        except SfmError as exc:
            self.unsetCursor()
            QMessageBox.critical(self, "Falha no alinhamento", str(exc))
            self._refresh(self.project.images)
            return
        finally:
            self.unsetCursor()

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

    def _refresh(self, records: list[ImageRecord]) -> None:
        self.status_label.setText(f"{len(records)} imagem(ns) carregada(s).")
        self._populate_table(records)
        self._plot_camera_positions(records)

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

    def _plot_aligned_positions(self, reconstruction_path: str) -> None:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(reconstruction_path)
        xs, ys = [], []
        for image_id in reconstruction.reg_image_ids():
            center = reconstruction.image(image_id).projection_center()
            xs.append(float(center[0]))
            ys.append(float(center[1]))

        self._plot_points(xs, ys, f"Posições das câmeras — SfM alinhado ({len(xs)} registradas)")

    def _plot_points(self, xs: list[float], ys: list[float], title: str) -> None:
        self.axes.clear()
        self.axes.set_facecolor(theme.BACKGROUND)
        if xs:
            self.axes.scatter(
                xs, ys, c=theme.PRIMARY, s=30, edgecolors=theme.PRIMARY_DARK, linewidths=0.6
            )
            self.axes.set_aspect("equal", adjustable="datalim")
        self.axes.set_xlabel(f"Este (m) — EPSG:{self.project.crs.project_epsg}", color=theme.PRIMARY_DARK)
        self.axes.set_ylabel("Norte (m)", color=theme.PRIMARY_DARK)
        self.axes.set_title(title, color=theme.PRIMARY_DARK)
        self.axes.tick_params(colors=theme.TEXT_MUTED)
        for spine in self.axes.spines.values():
            spine.set_color(theme.BORDER)
        self.axes.grid(True, linewidth=0.3, color=theme.BORDER)
        self.canvas.draw_idle()


def run(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
