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

from htrmapper.core.project import ImageRecord, Project
from htrmapper.geo.crs import (
    CoordinateReferenceSystem,
    GeodeticPoint,
    GeodeticTransformer,
    wgs84,
)
from htrmapper.io.image_import import import_folder

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

        self._build_ui()
        if self.project.images:
            self._refresh(self.project.images)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        open_button = QPushButton("Importar pasta de imagens…")
        open_button.clicked.connect(self._on_import_clicked)
        toolbar.addWidget(open_button)
        self.status_label = QLabel("Nenhum projeto carregado.")
        toolbar.addWidget(self.status_label)
        toolbar.addStretch(1)
        root_layout.addLayout(toolbar)

        splitter = QSplitter()
        root_layout.addWidget(splitter)

        self.table = QTableWidget()
        self.table.setColumnCount(len(_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in _TABLE_COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        splitter.addWidget(self.table)

        # Matplotlib is an optional GUI-extra dependency; imported lazily so
        # the rest of the GUI module still loads without it installed.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
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
        QMessageBox.information(
            self,
            "Importação concluída",
            "\n".join(report.summary_lines()),
        )

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

        xs, ys, labels = [], [], []
        for record in records:
            if not record.position_valid:
                continue
            point = transformer.forward(
                GeodeticPoint(lon=record.longitude, lat=record.latitude, alt=record.altitude)
            )
            xs.append(point.x)
            ys.append(point.y)
            labels.append(record.file_name)

        self.axes.clear()
        if xs:
            self.axes.scatter(xs, ys, c="tab:green", s=30, edgecolors="black", linewidths=0.5)
            self.axes.set_aspect("equal", adjustable="datalim")
        self.axes.set_xlabel(f"Este (m) — EPSG:{self.project.crs.project_epsg}")
        self.axes.set_ylabel("Norte (m)")
        self.axes.set_title(f"Posições das câmeras ({len(xs)} válidas)")
        self.axes.grid(True, linewidth=0.3)
        self.canvas.draw_idle()


def run(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
