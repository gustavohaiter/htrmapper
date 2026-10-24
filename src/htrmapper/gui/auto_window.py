"""Main window: automatic processing in one click.

Choose the photos folder, the output folder and the level (each with a
short description and the estimated time on this PC), press Processar.
The steps run one after another (`pipeline.auto`) while the window shows
each step's status, the overall progress, elapsed and remaining time and
the live log. A failure is shown on screen and saved as ERRO_<data>.txt
in the output folder. Running again on the same output folder resumes
from the last finished step.

The step-by-step window of earlier phases stays available under
Avançado > Etapas individuais.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pycolmap
from PySide6.QtCore import QSettings, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from htrmapper.core import theme
from htrmapper.core.platform_paths import is_running_under_wsl
from htrmapper.pipeline import timing
from htrmapper.pipeline.auto import (
    STEP_KEYS,
    STEPS,
    AutoProcessingError,
    AutoProcessor,
    AutoRequest,
    Callbacks,
    count_photos,
    project_name_from_folder,
    software_version,
)
from htrmapper.pipeline.levels import DEFAULT_LEVEL, LEVELS

STATUS_STYLE = {
    "aguardando": ("Aguardando", theme.TEXT_MUTED),
    "rodando": ("Em andamento", theme.PRIMARY),
    "concluido": ("Concluída", "#2E7D32"),
    "pulado": ("Reaproveitada", "#2E7D32"),
    "erro": ("ERRO", "#C62828"),
    "cancelado": ("Cancelada", theme.PENDING),
}

_STYLESHEET = f"""
QMainWindow, QWidget {{ background-color: {theme.BACKGROUND}; color: {theme.TEXT}; font-size: 13px; }}
QGroupBox {{ border: 1px solid {theme.BORDER}; border-radius: 6px; margin-top: 12px; padding: 8px; font-weight: bold;
             color: {theme.PRIMARY_DARK}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QPushButton {{ background-color: {theme.PRIMARY_DARK}; color: white; border: none; border-radius: 4px;
               padding: 6px 14px; font-weight: bold; }}
QPushButton:hover {{ background-color: {theme.PRIMARY}; }}
QPushButton:disabled {{ background-color: {theme.BORDER}; color: {theme.TEXT_MUTED}; }}
QPushButton#processar {{ font-size: 16px; padding: 10px 28px; }}
QPushButton#cancelar {{ background-color: #8E2A2A; }}
QLineEdit, QPlainTextEdit, QTableWidget {{ border: 1px solid {theme.BORDER}; background: white; }}
QProgressBar {{ border: 1px solid {theme.BORDER}; border-radius: 4px; text-align: center; background: {theme.SURFACE};
                height: 20px; }}
QProgressBar::chunk {{ background-color: {theme.PRIMARY}; }}
QLabel.muted {{ color: {theme.TEXT_MUTED}; }}
"""


def open_in_system(path: Path) -> None:
    """Open a file or folder with the default program. Inside WSL that is
    the Windows program (Explorer, PDF reader, ...), through explorer.exe."""
    path = Path(path)
    if is_running_under_wsl():
        try:
            windows_path = subprocess.run(
                ["wslpath", "-w", str(path)], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
            subprocess.Popen(["explorer.exe", windows_path])
            return
        except (OSError, subprocess.SubprocessError):
            pass
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _windows_home_folders() -> list[Path]:
    """Windows folders worth a shortcut in the file dialog when in WSL."""
    if not is_running_under_wsl():
        return []
    users = Path("/mnt/c/Users")
    homes = []
    try:
        homes = [
            p
            for p in users.iterdir()
            if p.is_dir() and p.name not in ("Public", "Default", "Default User", "All Users")
        ]
    except OSError:
        pass
    folders = []
    for home in homes:
        folders += [home / "Desktop", home / "Documents", home / "Downloads", home]
    folders += [Path(f"/mnt/{d}") for d in "cdef" if Path(f"/mnt/{d}").is_dir()]
    return [f for f in folders if f.is_dir()]


class _SignalCallbacks(Callbacks):
    def __init__(self, worker: AutoWorker):
        self.worker = worker

    def log(self, message: str) -> None:
        self.worker.log_line.emit(message)

    def step(self, key: str, status: str, detail: str = "") -> None:
        self.worker.step_changed.emit(key, status, detail)

    def phase(self, key: str, text: str) -> None:
        self.worker.phase_changed.emit(key, text)

    def progress(self, key: str, done: int, total: int) -> None:
        self.worker.progress_changed.emit(key, done, total)


class AutoWorker(QThread):
    log_line = Signal(str)
    step_changed = Signal(str, str, str)
    phase_changed = Signal(str, str)
    progress_changed = Signal(str, int, int)
    finished_ok = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(self, request: AutoRequest, resume: bool):
        super().__init__()
        self.request = request
        self.resume = resume
        self.token = pycolmap.CancellationToken()

    def cancel(self) -> None:
        self.token.cancel()

    def run(self) -> None:
        processor = AutoProcessor(self.request, _SignalCallbacks(self), self.token)
        try:
            result = processor.run(resume=self.resume)
        except InterruptedError:
            self.cancelled.emit()
        except AutoProcessingError as exc:
            self.failed.emit(exc)
        except Exception as exc:  # noqa: BLE001 -- shown to the user, never swallowed
            self.failed.emit(AutoProcessingError(f"Erro inesperado: {exc}"))
        else:
            self.finished_ok.emit(result)


class AutoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"HTRMapper {software_version()}")
        self.resize(1250, 820)
        self.setStyleSheet(_STYLESHEET)
        self.settings = QSettings("HTRMapper", "HTRMapper")
        self.worker: AutoWorker | None = None
        self.manual_window = None
        self.num_photos = 0
        self.estimates: dict[str, timing.StepEstimate] = {}
        self.step_started: dict[str, float] = {}
        self.step_fraction: dict[str, float] = {}
        self.step_seconds: dict[str, float] = {}
        self.step_state: dict[str, str] = {}
        self.run_started = 0.0
        self.last_output: Path | None = None
        self.last_report: Path | None = None
        self._build_menu()
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._restore_settings()

    # -- UI ----------------------------------------------------------------------

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("Arquivo")
        open_out = QAction("Abrir pasta de resultados", self)
        open_out.triggered.connect(self._open_output)
        file_menu.addAction(open_out)
        quit_action = QAction("Sair", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        advanced = menu.addMenu("Avançado")
        manual = QAction("Etapas individuais (modo manual)…", self)
        manual.triggered.connect(self._open_manual_window)
        advanced.addAction(manual)
        help_menu = menu.addMenu("Ajuda")
        about = QAction("Sobre o HTRMapper", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter()
        root.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        title = QLabel("Processamento automático")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {theme.PRIMARY_DARK};")
        left_layout.addWidget(title)

        photos_box = QGroupBox("1. Fotos do voo")
        photos_layout = QGridLayout(photos_box)
        self.photos_edit = QLineEdit()
        self.photos_edit.setPlaceholderText("Pasta com as fotos (JPG) do drone")
        self.photos_edit.editingFinished.connect(self._photos_changed)
        photos_button = QPushButton("Escolher…")
        photos_button.clicked.connect(self._choose_photos)
        self.photos_info = QLabel("")
        self.photos_info.setProperty("class", "muted")
        photos_layout.addWidget(self.photos_edit, 0, 0)
        photos_layout.addWidget(photos_button, 0, 1)
        photos_layout.addWidget(self.photos_info, 1, 0, 1, 2)
        left_layout.addWidget(photos_box)

        output_box = QGroupBox("2. Pasta de saída")
        output_layout = QGridLayout(output_box)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Onde salvar ortomosaico, DEM e relatório")
        output_button = QPushButton("Escolher…")
        output_button.clicked.connect(self._choose_output)
        output_layout.addWidget(self.output_edit, 0, 0)
        output_layout.addWidget(output_button, 0, 1)
        left_layout.addWidget(output_box)

        level_box = QGroupBox("3. Nível de processamento")
        level_layout = QVBoxLayout(level_box)
        self.level_group = QButtonGroup(self)
        self.level_buttons: dict[str, QRadioButton] = {}
        self.level_time: dict[str, QLabel] = {}
        for key, level in LEVELS.items():
            button = QRadioButton(f"{level.label} — {level.summary}")
            button.setStyleSheet("font-weight: bold;")
            self.level_group.addButton(button)
            self.level_buttons[key] = button
            description = QLabel(level.description)
            description.setWordWrap(True)
            description.setStyleSheet(f"color: {theme.TEXT_MUTED}; margin-left: 22px;")
            estimate = QLabel("")
            estimate.setStyleSheet(f"color: {theme.PRIMARY}; margin-left: 22px; margin-bottom: 6px;")
            self.level_time[key] = estimate
            level_layout.addWidget(button)
            level_layout.addWidget(description)
            level_layout.addWidget(estimate)
        left_layout.addWidget(level_box)

        self.delete_work = QCheckBox(
            "Apagar os dados de trabalho ao terminar (libera espaço; não dá para retomar depois)"
        )
        left_layout.addWidget(self.delete_work)

        buttons = QHBoxLayout()
        self.process_button = QPushButton("Processar")
        self.process_button.setObjectName("processar")
        self.process_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setObjectName("cancelar")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        buttons.addWidget(self.process_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        left_layout.addLayout(buttons)

        results = QHBoxLayout()
        self.open_folder_button = QPushButton("Abrir pasta de resultados")
        self.open_folder_button.clicked.connect(self._open_output)
        self.open_report_button = QPushButton("Abrir relatório PDF")
        self.open_report_button.clicked.connect(lambda: self.last_report and open_in_system(self.last_report))
        self.open_report_button.setEnabled(False)
        results.addWidget(self.open_folder_button)
        results.addWidget(self.open_report_button)
        results.addStretch(1)
        left_layout.addLayout(results)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        progress_box = QGroupBox("Andamento")
        progress_layout = QVBoxLayout(progress_box)
        self.steps_table = QTableWidget(len(STEPS), 3)
        self.steps_table.setHorizontalHeaderLabels(["Etapa", "Situação", "Tempo"])
        self.steps_table.verticalHeader().setVisible(False)
        self.steps_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.steps_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.steps_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for row, (_key, label) in enumerate(STEPS):
            self.steps_table.setItem(row, 0, QTableWidgetItem(label))
            self.steps_table.setItem(row, 1, QTableWidgetItem(""))
            self.steps_table.setItem(row, 2, QTableWidgetItem(""))
        self.steps_table.setFixedHeight(self.steps_table.verticalHeader().length() + 34)
        progress_layout.addWidget(self.steps_table)
        self.current_label = QLabel("Pronto para processar.")
        self.current_label.setWordWrap(True)
        progress_layout.addWidget(self.current_label)
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 1000)
        self.overall_bar.setFormat("%p%")
        progress_layout.addWidget(self.overall_bar)
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        progress_layout.addWidget(self.time_label)
        right_layout.addWidget(progress_box)

        log_box = QGroupBox("O que está acontecendo (log)")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20000)
        mono = QFont("DejaVu Sans Mono")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.log_view.setFont(mono)
        log_layout.addWidget(self.log_view)
        right_layout.addWidget(log_box, 1)
        splitter.addWidget(right)
        splitter.setSizes([520, 730])
        self._reset_steps()

    # -- settings, folders, estimates ----------------------------------------------

    def _restore_settings(self) -> None:
        self.photos_edit.setText(self.settings.value("fotos", "", str))
        self.output_edit.setText(self.settings.value("saida", "", str))
        level = self.settings.value("nivel", DEFAULT_LEVEL, str)
        self.level_buttons.get(level, self.level_buttons[DEFAULT_LEVEL]).setChecked(True)
        self._photos_changed()

    def _save_settings(self) -> None:
        self.settings.setValue("fotos", self.photos_edit.text())
        self.settings.setValue("saida", self.output_edit.text())
        self.settings.setValue("nivel", self._level())

    def _level(self) -> str:
        return next(k for k, b in self.level_buttons.items() if b.isChecked())

    def _dialog(self, title: str, start: str) -> str:
        dialog = QFileDialog(self, title, start)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        shortcuts = _windows_home_folders()
        if shortcuts:
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
            dialog.setSidebarUrls(
                [QUrl.fromLocalFile(str(p)) for p in shortcuts] + [QUrl.fromLocalFile(str(Path.home()))]
            )
        if dialog.exec():
            selected = dialog.selectedFiles()
            return selected[0] if selected else ""
        return ""

    def _start_folder(self, text: str) -> str:
        if text and Path(text).exists():
            return text
        shortcuts = _windows_home_folders()
        return str(shortcuts[0]) if shortcuts else str(Path.home())

    def _choose_photos(self) -> None:
        folder = self._dialog("Pasta com as fotos do voo", self._start_folder(self.photos_edit.text()))
        if folder:
            self.photos_edit.setText(folder)
            self._photos_changed()

    def _choose_output(self) -> None:
        folder = self._dialog("Pasta de saída", self._start_folder(self.output_edit.text() or self.photos_edit.text()))
        if folder:
            self.output_edit.setText(folder)

    def _photos_changed(self) -> None:
        folder = Path(self.photos_edit.text().strip()) if self.photos_edit.text().strip() else None
        self.num_photos = count_photos(folder) if folder and folder.is_dir() else 0
        if folder is None:
            self.photos_info.setText("")
        elif not folder.is_dir():
            self.photos_info.setText("Pasta não encontrada.")
        else:
            self.photos_info.setText(f"{self.num_photos} foto(s) encontrada(s).")
            if not self.output_edit.text().strip():
                self.output_edit.setText(str(folder.parent / f"HTRMapper_{project_name_from_folder(folder)}"))
        self._update_estimates()

    def _update_estimates(self) -> None:
        if not self.num_photos:
            for label in self.level_time.values():
                label.setText("")
            return
        hardware = timing.hardware_signature()
        for key in LEVELS:
            est = timing.estimate(key, self.num_photos, hardware, STEP_KEYS)
            total = sum(e.seconds for e in est.values())
            source = (
                "pelo histórico deste PC"
                if all(e.from_history for e in est.values())
                else "estimativa inicial; fica mais precisa a cada processamento neste PC"
            )
            self.level_time[key].setText(f"Tempo estimado: ~{timing.format_duration(total)} ({source})")

    # -- running -------------------------------------------------------------------

    def _request(self) -> AutoRequest | None:
        photos = self.photos_edit.text().strip()
        output = self.output_edit.text().strip()
        if not photos or not output:
            QMessageBox.warning(self, "HTRMapper", "Escolha a pasta das fotos e a pasta de saída.")
            return None
        return AutoRequest(Path(photos), Path(output), self._level(), self.delete_work.isChecked())

    def _start(self) -> None:
        request = self._request()
        if request is None:
            return
        self._save_settings()
        processor = AutoProcessor(request)
        problems = processor.preflight()
        if problems:
            QMessageBox.critical(self, "Não é possível começar", "\n\n".join(problems))
            return
        resume = True
        existing = processor.inspect_existing()
        if existing is not None:
            choice = self._ask_about_existing(existing, request.level)
            if choice is None:
                return
            resume = choice
        self.estimates = processor.estimate()
        self._reset_steps()
        self.log_view.clear()
        self.last_output = request.output_folder
        self.last_report = None
        self.open_report_button.setEnabled(False)
        self.run_started = time.monotonic()
        self.worker = AutoWorker(request, resume)
        self.worker.log_line.connect(self._append_log)
        self.worker.step_changed.connect(self._on_step)
        self.worker.phase_changed.connect(self._on_phase)
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self._set_running(True)
        self.worker.start()
        self.timer.start(1000)

    def _ask_about_existing(self, existing, level: str) -> bool | None:
        """True = resume, False = start over, None = do nothing."""
        box = QMessageBox(self)
        box.setWindowTitle("Já existe um processamento nesta pasta")
        done = ", ".join(dict(STEPS)[k] for k in existing.steps_done) or "nenhuma etapa"
        if existing.resumable_with(level) and not existing.finished:
            box.setText(
                f"Esta pasta já tem um processamento nível {LEVELS[existing.level].label} pela metade.\n\nJá concluído: {done}."
            )
            resume = box.addButton("Continuar de onde parou", QMessageBox.ButtonRole.AcceptRole)
            restart = box.addButton("Recomeçar do zero", QMessageBox.ButtonRole.DestructiveRole)
        elif existing.resumable_with(level):
            box.setText("Esta pasta já tem um processamento CONCLUÍDO com estas fotos e este nível.")
            resume = None
            restart = box.addButton("Processar de novo do zero", QMessageBox.ButtonRole.DestructiveRole)
        else:
            other = "outras fotos" if not existing.same_photos else f"o nível {LEVELS[existing.level].label}"
            box.setText(
                f"Esta pasta já tem um processamento com {other}.\n\n"
                "Recomeçar apaga os dados de trabalho dele e substitui os resultados. Para guardar os dois, escolha outra pasta."
            )
            resume = None
            restart = box.addButton("Recomeçar do zero aqui", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if resume is not None and clicked is resume:
            return True
        if clicked is restart:
            return False
        return None

    def _cancel(self) -> None:
        if self.worker is None:
            return
        answer = QMessageBox.question(
            self,
            "Cancelar",
            "Cancelar o processamento? O que já foi concluído fica salvo, e rodar de novo continua de onde parou.\n\n"
            "(O ajuste GNSS não pode ser interrompido no meio: se estiver nele, o cancelamento acontece quando terminar.)",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.current_label.setText("Cancelando…")

    def _set_running(self, running: bool) -> None:
        self.process_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        for widget in (self.photos_edit, self.output_edit, self.delete_work, *self.level_buttons.values()):
            widget.setEnabled(not running)

    def _reset_steps(self) -> None:
        self.step_state = {k: "aguardando" for k in STEP_KEYS}
        self.step_started.clear()
        self.step_fraction.clear()
        self.step_seconds.clear()
        for key in STEP_KEYS:
            self._paint_step(key, "")
        self.overall_bar.setValue(0)
        self.time_label.setText("")

    def _paint_step(self, key: str, elapsed: str) -> None:
        row = STEP_KEYS.index(key)
        text, color = STATUS_STYLE[self.step_state[key]]
        item = self.steps_table.item(row, 1)
        item.setText(text)
        item.setForeground(QColor(color))
        self.steps_table.item(row, 2).setText(elapsed)

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    def _on_step(self, key: str, status: str, detail: str) -> None:
        self.step_state[key] = status
        if status == "rodando":
            self.step_started[key] = time.monotonic()
            self.step_fraction[key] = 0.0
            self.current_label.setText(f"<b>{dict(STEPS)[key]}</b>…")
            self._paint_step(key, "")
        else:
            elapsed = ""
            if key in self.step_started:
                elapsed = timing.format_duration(time.monotonic() - self.step_started[key])
            self._paint_step(key, elapsed)
            if status in ("concluido", "pulado"):
                self.step_fraction[key] = 1.0
            if status == "concluido" and key in self.step_started:
                self.step_seconds[key] = time.monotonic() - self.step_started[key]
        self._update_overall()

    def _on_phase(self, key: str, text: str) -> None:
        self.current_label.setText(f"<b>{dict(STEPS)[key]}</b> — {text}")

    def _on_progress(self, key: str, done: int, total: int) -> None:
        if total > 0:
            self.step_fraction[key] = done / total
            self.current_label.setText(f"<b>{dict(STEPS)[key]}</b> — {done} de {total}")
        self._update_overall()

    def _tick(self) -> None:
        for key, state in self.step_state.items():
            if state == "rodando" and key in self.step_started:
                elapsed = time.monotonic() - self.step_started[key]
                self._paint_step(key, timing.format_duration(elapsed))
        self._update_overall()

    def _update_overall(self) -> None:
        """Progress weighted by each step's estimated time; the remaining
        time is rescaled by how this run's finished steps compared to their
        estimates."""
        if not self.estimates:
            return
        weights = {k: max(self.estimates[k].seconds, 1.0) for k in STEP_KEYS}
        now = time.monotonic()
        done = measured = estimated = 0.0
        for key in STEP_KEYS:
            state = self.step_state[key]
            if state in ("concluido", "pulado"):
                done += weights[key]
                if key in self.step_seconds:
                    measured += self.step_seconds[key]
                    estimated += weights[key]
            elif state == "rodando":
                elapsed = now - self.step_started.get(key, now)
                fraction = self.step_fraction.get(key, 0.0) or min(elapsed / weights[key], 0.95)
                done += weights[key] * fraction
        total = sum(weights.values())
        self.overall_bar.setValue(int(1000 * done / total))
        ratio = min(max(measured / estimated, 0.3), 3.0) if estimated else 1.0
        if self.worker is not None:
            self.time_label.setText(
                f"Decorrido: {timing.format_duration(now - self.run_started)}   •   "
                f"Restante estimado: ~{timing.format_duration(max(total - done, 0.0) * ratio)}"
            )

    def _end_run(self) -> None:
        self.timer.stop()
        self._set_running(False)
        self.worker = None

    def _on_finished(self, result) -> None:
        self._end_run()
        self.overall_bar.setValue(1000)
        total = timing.format_duration(result.total_seconds)
        self.time_label.setText(f"Concluído em {total}.")
        self.current_label.setText("<b>Processamento concluído.</b>")
        report = result.products.get("relatorio")
        self.last_report = Path(report) if report else None
        self.open_report_button.setEnabled(self.last_report is not None)
        lines = [f"Processamento concluído em {total}.", ""]
        names = {"ortho": "Ortomosaico", "dem": "DEM", "relatorio": "Relatório PDF", "log": "Log"}
        for key, label in names.items():
            if key in result.products:
                lines.append(f"{label}: {Path(result.products[key]).name}")
        if result.warnings:
            lines += ["", "Avisos:"] + [f"• {w}" for w in result.warnings]
        box = QMessageBox(self)
        box.setWindowTitle("Processamento concluído")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("\n".join(lines))
        open_folder = box.addButton("Abrir pasta de resultados", QMessageBox.ButtonRole.AcceptRole)
        open_report = box.addButton("Abrir relatório", QMessageBox.ButtonRole.ActionRole) if self.last_report else None
        box.addButton("Fechar", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_folder:
            self._open_output()
        elif open_report is not None and box.clickedButton() is open_report:
            open_in_system(self.last_report)

    def _on_failed(self, error: AutoProcessingError) -> None:
        self._end_run()
        self.current_label.setText(f"<b style='color:#C62828'>Erro:</b> {error}")
        box = QMessageBox(self)
        box.setWindowTitle("Erro no processamento")
        box.setIcon(QMessageBox.Icon.Critical)
        text = str(error)
        if error.error_file:
            text += f"\n\nOs detalhes técnicos foram salvos em:\n{error.error_file}\n\nCorrigido o problema, é só clicar em Processar de novo: continua de onde parou."
        box.setText(text)
        open_file = (
            box.addButton("Abrir arquivo do erro", QMessageBox.ButtonRole.ActionRole) if error.error_file else None
        )
        box.addButton("Fechar", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if open_file is not None and box.clickedButton() is open_file:
            open_in_system(error.error_file)

    def _on_cancelled(self) -> None:
        self._end_run()
        self.current_label.setText("Cancelado. Clique em Processar para continuar de onde parou.")

    # -- menus -------------------------------------------------------------------------

    def _open_output(self) -> None:
        folder = self.last_output or (Path(self.output_edit.text()) if self.output_edit.text().strip() else None)
        if folder and folder.is_dir():
            open_in_system(folder)
        else:
            QMessageBox.information(self, "HTRMapper", "A pasta de saída ainda não existe.")

    def _open_manual_window(self) -> None:
        from htrmapper.gui.main_window import MainWindow

        if self.manual_window is None:
            self.manual_window = MainWindow()
        self.manual_window.show()
        self.manual_window.raise_()

    def _about(self) -> None:
        cuda = "sim" if pycolmap.has_cuda else "não"
        QMessageBox.about(
            self,
            "Sobre o HTRMapper",
            f"HTRMapper {software_version()}\n\nFotogrametria para agricultura de precisão.\n"
            f"pycolmap {pycolmap.__version__} (GPU CUDA: {cuda})\nPython {sys.version.split()[0]}",
        )

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            answer = QMessageBox.question(
                self, "Sair", "Há um processamento em andamento. Cancelar e sair? (Dá para continuar depois.)"
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait()
        event.accept()


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("HTRMapper")
    window = AutoWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
