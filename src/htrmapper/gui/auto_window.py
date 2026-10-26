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
from PySide6.QtCore import QEvent, QRectF, QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QImage, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from htrmapper.core import theme
from htrmapper.core.platform_paths import is_running_under_wsl
from htrmapper.gui.branding import app_icon, logo_pixmap
from htrmapper.gui.file_dialogs import choose_folder
from htrmapper.io.image_import import find_images
from htrmapper.pipeline import timing
from htrmapper.pipeline.auto import (
    STEP_KEYS,
    STEPS,
    AutoProcessingError,
    AutoProcessor,
    AutoRequest,
    Callbacks,
    project_name_from_folder,
    software_version,
)
from htrmapper.pipeline.levels import DEFAULT_LEVEL, LEVELS

STATUS_STYLE = {
    "aguardando": ("Aguardando", theme.TEXT_MUTED),
    "rodando": ("Em andamento", theme.PRIMARY),
    "concluido": ("Concluída", "#2E7D32"),
    "pulado": ("Reaproveitada", "#2E7D32"),
    "erro": ("Erro", "#C62828"),
    "cancelado": ("Cancelada", theme.PENDING),
}
# What each step does, in plain words (shown under its name).
STEP_HELP = {
    "importar": "Lê as fotos e, de cada uma, a posição do GNSS/RTK e os dados da câmera.",
    "alinhar": "Acha pontos em comum entre as fotos e monta o bloco com todas elas.",
    "ajustar": "Encaixa o bloco nas posições do RTK e calibra a lente da câmera.",
    "densa": "Reconstrói o terreno em 3D, pixel a pixel. É a etapa mais longa (usa a placa de vídeo).",
    "dem": "Transforma a nuvem 3D no modelo de elevação (superfície) do terreno.",
    "ortho": "Projeta cada foto sobre o terreno e costura o ortomosaico.",
    "relatorio": "Gera o relatório PDF com a qualidade e a precisão do processamento.",
}
SPINNER = "◐◓◑◒"
BACKDROP = "#EEF3F8"
CARD_BORDER = "#D5E0EB"
GREEN = "#2E7D32"
RED = "#C62828"

_STYLESHEET = f"""
QMainWindow, QWidget#page, QScrollArea, QWidget#scroll_body {{ background: {BACKDROP}; }}
QWidget {{ color: {theme.TEXT}; font-size: 13px; }}
QLabel, QRadioButton, QCheckBox {{ background: transparent; }}
QMenuBar {{ background: white; border-bottom: 1px solid {CARD_BORDER}; }}
QMenuBar::item:selected {{ background: {theme.SURFACE}; }}
QMenu {{ background: white; border: 1px solid {CARD_BORDER}; }}
QMenu::item:selected {{ background: {theme.PRIMARY}; color: white; }}
QFrame#header {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {theme.PRIMARY_DARK}, stop:1 {theme.PRIMARY}); }}
QFrame#header QLabel {{ color: white; }}
QLabel#app_title {{ font-size: 22px; font-weight: bold; }}
QLabel#app_subtitle {{ font-size: 13px; color: #D6E4F2; }}
QLabel#chip_ok, QLabel#chip_warn {{ border-radius: 11px; padding: 3px 12px; font-size: 12px; font-weight: bold; }}
QLabel#chip_ok {{ background: rgba(255,255,255,0.16); color: white; }}
QLabel#chip_warn {{ background: #F9A825; color: #3A2A00; }}
QFrame#card {{ background: white; border: 1px solid {CARD_BORDER}; border-radius: 10px; }}
QLabel#card_title {{ font-size: 15px; font-weight: bold; color: {theme.PRIMARY_DARK}; }}
QLabel#badge {{ background: {theme.PRIMARY_DARK}; color: white; border-radius: 13px; font-weight: bold; font-size: 13px; }}
QLabel#help, QLabel.muted {{ color: {theme.TEXT_MUTED}; font-size: 12px; }}
QLabel#thumb {{ background: {theme.SURFACE}; border: 1px solid {CARD_BORDER}; border-radius: 6px; }}
QLineEdit {{ border: 1px solid {CARD_BORDER}; border-radius: 6px; padding: 6px 8px; background: white; }}
QLineEdit:focus {{ border: 1px solid {theme.PRIMARY}; }}
QPushButton {{ background: white; color: {theme.PRIMARY_DARK}; border: 1px solid {theme.PRIMARY}; border-radius: 6px;
               padding: 7px 14px; font-weight: bold; }}
QPushButton:hover {{ background: {theme.SURFACE}; }}
QPushButton:disabled {{ color: #9AA9B8; border-color: {CARD_BORDER}; background: #F6F8FA; }}
QPushButton#processar {{ background: {GREEN}; color: white; border: none; font-size: 17px; padding: 12px 34px;
                         border-radius: 8px; }}
QPushButton#processar:hover {{ background: #388E3C; }}
QPushButton#processar:disabled {{ background: #A5C8A7; color: white; }}
QPushButton#cancelar {{ color: {RED}; border-color: {RED}; }}
QPushButton#cancelar:hover {{ background: #FDECEC; }}
QPushButton#cancelar:disabled {{ color: #D9A6A6; border-color: #EBCACA; background: white; }}
QFrame#level {{ background: white; border: 2px solid {CARD_BORDER}; border-radius: 10px; }}
QFrame#level:hover {{ border-color: #9DB8D3; }}
QFrame#level[selected="true"] {{ background: #EAF2FB; border-color: {theme.PRIMARY}; }}
QFrame#level QRadioButton {{ font-weight: bold; font-size: 14px; color: {theme.PRIMARY_DARK}; }}
QLabel#level_summary {{ color: {theme.PRIMARY}; font-weight: bold; font-size: 12px; }}
QLabel#level_time {{ color: {theme.PRIMARY_DARK}; font-weight: bold; font-size: 12px; }}
QLabel#step_title {{ font-weight: bold; font-size: 13px; }}
QLabel#step_help {{ color: {theme.TEXT_MUTED}; font-size: 11px; }}
QLabel#step_time {{ color: {theme.TEXT_MUTED}; font-size: 12px; }}
QProgressBar {{ border: none; border-radius: 6px; background: #DFE8F1; text-align: center; color: {theme.PRIMARY_DARK};
                font-weight: bold; min-height: 22px; }}
QProgressBar::chunk {{ border-radius: 6px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {theme.PRIMARY},
                       stop:1 #3B82C4); }}
QProgressBar#mini {{ min-height: 5px; max-height: 5px; border-radius: 2px; }}
QProgressBar#mini::chunk {{ border-radius: 2px; }}
QLabel#current {{ font-size: 14px; color: {theme.PRIMARY_DARK}; }}
QLabel#time {{ color: {theme.TEXT_MUTED}; }}
QToolButton#toggle_log {{ border: none; color: {theme.PRIMARY}; font-weight: bold; padding: 4px; background: transparent; }}
QPlainTextEdit {{ border: 1px solid {CARD_BORDER}; border-radius: 6px; background: #FBFCFE; }}
QFrame#result {{ background: #F1F8F1; border: 1px solid #A5D6A7; border-radius: 10px; }}
QFrame#result[failed="true"] {{ background: #FDF1F1; border-color: #EF9A9A; }}
QLabel#result_title {{ font-size: 16px; font-weight: bold; color: {GREEN}; }}
QFrame#result[failed="true"] QLabel#result_title {{ color: {RED}; }}
QCheckBox {{ color: {theme.TEXT}; spacing: 8px; }}
QRadioButton::indicator, QCheckBox::indicator {{ width: 14px; height: 14px; border: 2px solid #9DB8D3; background: white; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator:checked {{ border-color: {theme.PRIMARY}; background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
    fx:0.5, fy:0.5, stop:0 {theme.PRIMARY}, stop:0.55 {theme.PRIMARY}, stop:0.65 white, stop:1 white); }}
QCheckBox::indicator:checked {{ border-color: {theme.PRIMARY}; background: {theme.PRIMARY}; }}
QRadioButton::indicator:disabled, QCheckBox::indicator:disabled {{ border-color: {CARD_BORDER}; }}
QMessageBox {{ background: white; }}
QScrollArea {{ border: none; }}
"""


def _repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _card(title: str, number: int | None = None, help_text: str = "") -> tuple[QFrame, QVBoxLayout]:
    """White rounded card with a numbered title and a help line."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)
    head = QHBoxLayout()
    head.setSpacing(10)
    if number is not None:
        badge = QLabel(str(number))
        badge.setObjectName("badge")
        badge.setFixedSize(26, 26)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head.addWidget(badge)
    label = QLabel(title)
    label.setObjectName("card_title")
    head.addWidget(label)
    head.addStretch(1)
    layout.addLayout(head)
    if help_text:
        help_label = QLabel(help_text)
        help_label.setObjectName("help")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
    return frame, layout


class _LevelBars(QWidget):
    """Signal-strength style icon: 1, 2 or 3 of 3 bars filled."""

    def __init__(self, filled: int):
        super().__init__()
        self.filled = filled
        self.setFixedSize(26, 22)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            height = 8 + 6 * i
            color = QColor(theme.PRIMARY) if i < self.filled else QColor(CARD_BORDER)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(2 + 8 * i, 22 - height, 6, height), 1.5, 1.5)
        painter.end()


class _LevelCard(QFrame):
    """A level as a clickable card: bars icon, name, summary, description
    and the estimated time. Clicking anywhere selects its radio button."""

    def __init__(self, key: str, index: int, button: QRadioButton):
        super().__init__()
        level = LEVELS[key]
        self.button = button
        self.setObjectName("level")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(170)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(button)
        top.addStretch(1)
        top.addWidget(_LevelBars(index + 1))
        layout.addLayout(top)
        summary = QLabel(level.summary)
        summary.setObjectName("level_summary")
        layout.addWidget(summary)
        description = QLabel(level.description)
        description.setObjectName("help")
        description.setWordWrap(True)
        layout.addWidget(description, 1)
        self.time_label = QLabel("")
        self.time_label.setObjectName("level_time")
        self.time_label.setWordWrap(True)
        layout.addWidget(self.time_label)
        button.toggled.connect(self._toggled)

    def _toggled(self, checked: bool) -> None:
        self.setProperty("selected", "true" if checked else "false")
        _repolish(self)

    def mousePressEvent(self, event) -> None:
        if self.button.isEnabled():
            self.button.setChecked(True)
        super().mousePressEvent(event)


class _StepRow(QWidget):
    """One step of the run: state icon, name, plain description, a thin
    progress bar while it runs, and its state and time on the right."""

    def __init__(self, number: int, key: str, label: str):
        super().__init__()
        self.number = number
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)
        self.icon = QLabel()
        self.icon.setFixedSize(30, 30)
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        middle = QVBoxLayout()
        middle.setSpacing(2)
        title = QLabel(label)
        title.setObjectName("step_title")
        middle.addWidget(title)
        help_label = QLabel(STEP_HELP.get(key, ""))
        help_label.setObjectName("step_help")
        help_label.setWordWrap(True)
        middle.addWidget(help_label)
        self.bar = QProgressBar()
        self.bar.setObjectName("mini")
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setVisible(False)
        middle.addWidget(self.bar)
        layout.addLayout(middle, 1)
        right = QVBoxLayout()
        right.setSpacing(2)
        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.time = QLabel("")
        self.time.setObjectName("step_time")
        self.time.setAlignment(Qt.AlignmentFlag.AlignRight)
        right.addWidget(self.status)
        right.addWidget(self.time)
        right.addStretch(1)
        layout.addLayout(right)

    def paint(self, state: str, elapsed: str, spin: int = 0) -> None:
        text, color = STATUS_STYLE[state]
        symbol, fill, fg = {
            "aguardando": (str(self.number), "white", "#9AA9B8"),
            "rodando": (SPINNER[spin % len(SPINNER)], theme.PRIMARY, "white"),
            "concluido": ("✓", GREEN, "white"),
            "pulado": ("✓", "#81C784", "white"),
            "erro": ("!", RED, "white"),
            "cancelado": ("■", theme.PENDING, "white"),
        }[state]
        border = "#C3D0DD" if state == "aguardando" else fill
        self.icon.setText(symbol)
        self.icon.setStyleSheet(
            f"background: {fill}; color: {fg}; border: 2px solid {border}; border-radius: 15px;"
            "font-weight: bold; font-size: 14px;"
        )
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
        self.time.setText(elapsed)
        self.bar.setVisible(state == "rodando")
        if state == "pulado":
            self.setToolTip("Já estava pronta de um processamento anterior nesta pasta.")


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
        self.setWindowIcon(app_icon())
        self.resize(1320, 880)
        self.setStyleSheet(_STYLESHEET)
        self.settings = QSettings("HTRMapper", "HTRMapper")
        # Detect the hardware now, on the GUI thread before any processing
        # thread exists (it spawns nvidia-smi; cached afterwards).
        from htrmapper.core.system_info import detect_system_info

        self.system_info = detect_system_info()
        self.worker: AutoWorker | None = None
        self.manual_window = None
        self.num_photos = 0
        self.estimates: dict[str, timing.StepEstimate] = {}
        self.step_started: dict[str, float] = {}
        self.step_fraction: dict[str, float] = {}
        self.step_seconds: dict[str, float] = {}
        self.step_state: dict[str, str] = {}
        self.run_started = 0.0
        self.spin = 0
        self.last_output: Path | None = None
        self.last_report: Path | None = None
        self.last_ortho: Path | None = None
        self.last_error_file: Path | None = None
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
        how = QAction("Como funciona", self)
        how.triggered.connect(self._how_it_works)
        help_menu.addAction(how)
        about = QAction("Sobre o HTRMapper", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(14)
        logo = QLabel()
        logo.setPixmap(logo_pixmap(52))
        layout.addWidget(logo)
        texts = QVBoxLayout()
        texts.setSpacing(0)
        title = QLabel("HTRMapper")
        title.setObjectName("app_title")
        subtitle = QLabel("Ortomosaico de precisão a partir das fotos do drone — em um clique")
        subtitle.setObjectName("app_subtitle")
        texts.addWidget(title)
        texts.addWidget(subtitle)
        layout.addLayout(texts)
        layout.addStretch(1)
        info = self.system_info
        if info.gpus:
            gpu = QLabel(f"✓ GPU {info.gpus[0].name}")
            gpu.setObjectName("chip_ok")
            gpu.setToolTip("Placa de vídeo NVIDIA encontrada: a nuvem densa roda nela.")
        else:
            gpu = QLabel("Sem GPU NVIDIA")
            gpu.setObjectName("chip_warn")
            gpu.setToolTip("A nuvem densa precisa de uma placa de vídeo NVIDIA com o driver atualizado no Windows.")
        layout.addWidget(gpu, 0, Qt.AlignmentFlag.AlignVCenter)
        version = QLabel(f"versão {software_version()}")
        version.setObjectName("app_subtitle")
        layout.addWidget(version)
        return header

    def _build_ui(self) -> None:
        page = QWidget()
        page.setObjectName("page")
        self.setCentralWidget(page)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(16)
        outer.addLayout(body, 1)

        # Left: what to process (scrolls on small screens).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left = QWidget()
        left.setObjectName("scroll_body")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(12)
        scroll.setWidget(left)
        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        left_column.addWidget(scroll, 1)
        body.addLayout(left_column, 11)

        photos_card, photos_layout = _card(
            "Fotos do voo", 1, "A pasta com as fotos JPG do drone. As imagens multiespectrais (_MS_) são ignoradas."
        )
        row = QHBoxLayout()
        self.photos_edit = QLineEdit()
        self.photos_edit.setPlaceholderText("Clique em Escolher pasta…")
        self.photos_edit.editingFinished.connect(self._photos_changed)
        photos_button = QPushButton("Escolher pasta…")
        photos_button.clicked.connect(self._choose_photos)
        row.addWidget(self.photos_edit, 1)
        row.addWidget(photos_button)
        photos_layout.addLayout(row)
        preview = QHBoxLayout()
        preview.setSpacing(12)
        self.photos_thumb = QLabel()
        self.photos_thumb.setObjectName("thumb")
        self.photos_thumb.setFixedSize(112, 84)
        self.photos_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photos_thumb.setVisible(False)
        preview.addWidget(self.photos_thumb)
        self.photos_info = QLabel("")
        self.photos_info.setWordWrap(True)
        self.photos_info.setTextFormat(Qt.TextFormat.PlainText)
        preview.addWidget(self.photos_info, 1)
        photos_layout.addLayout(preview)
        left_layout.addWidget(photos_card)

        output_card, output_layout = _card(
            "Pasta de saída",
            2,
            "Onde salvar o ortomosaico, o DEM e o relatório. Processar de novo na mesma pasta continua de onde parou.",
        )
        row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Sugerida ao escolher as fotos")
        output_button = QPushButton("Escolher pasta…")
        output_button.clicked.connect(self._choose_output)
        row.addWidget(self.output_edit, 1)
        row.addWidget(output_button)
        output_layout.addLayout(row)
        left_layout.addWidget(output_card)

        level_card, level_layout = _card("Nível de processamento", 3, "Clique no cartão do nível desejado.")
        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.level_group = QButtonGroup(self)
        self.level_buttons: dict[str, QRadioButton] = {}
        self.level_time: dict[str, QLabel] = {}
        self.level_cards: dict[str, _LevelCard] = {}
        for index, (key, level) in enumerate(LEVELS.items()):
            button = QRadioButton(level.label)
            self.level_group.addButton(button)
            self.level_buttons[key] = button
            card = _LevelCard(key, index, button)
            self.level_cards[key] = card
            self.level_time[key] = card.time_label
            cards.addWidget(card, 1)
        level_layout.addLayout(cards)
        self.level_source = QLabel("")
        self.level_source.setObjectName("help")
        self.level_source.setWordWrap(True)
        level_layout.addWidget(self.level_source)
        left_layout.addWidget(level_card)

        self.delete_work = QCheckBox("Apagar os dados de trabalho ao terminar (libera espaço; depois não dá para retomar)")
        left_column.addWidget(self.delete_work)

        buttons = QHBoxLayout()
        self.process_button = QPushButton("▶  Processar")
        self.process_button.setObjectName("processar")
        self.process_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.process_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setObjectName("cancelar")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        buttons.addWidget(self.process_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        left_column.addLayout(buttons)
        tip = QLabel(
            "Se der erro, faltar energia ou você cancelar, é só clicar em Processar de novo: "
            "o que já ficou pronto é aproveitado."
        )
        tip.setObjectName("help")
        tip.setWordWrap(True)
        left_column.addWidget(tip)
        left_layout.addStretch(1)

        # Right: progress, result and log.
        self.right_scroll = right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_body = QWidget()
        right_body.setObjectName("scroll_body")
        right = QVBoxLayout(right_body)
        right.setContentsMargins(0, 0, 6, 0)
        right.setSpacing(12)
        right_scroll.setWidget(right_body)
        body.addWidget(right_scroll, 10)
        progress_card, progress_layout = _card("Andamento")
        self.step_rows: dict[str, _StepRow] = {}
        for number, (key, label) in enumerate(STEPS, start=1):
            step_row = _StepRow(number, key, label)
            self.step_rows[key] = step_row
            progress_layout.addWidget(step_row)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {CARD_BORDER};")
        progress_layout.addWidget(line)
        self.current_label = QLabel("Pronto para processar.")
        self.current_label.setObjectName("current")
        self.current_label.setWordWrap(True)
        self.current_label.setMinimumHeight(40)
        progress_layout.addWidget(self.current_label)
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 1000)
        self.overall_bar.setFormat("%p%")
        progress_layout.addWidget(self.overall_bar)
        self.time_label = QLabel("")
        self.time_label.setObjectName("time")
        progress_layout.addWidget(self.time_label)
        right.addWidget(progress_card)

        self.result_card = QFrame()
        self.result_card.setObjectName("result")
        result_outer = QVBoxLayout(self.result_card)
        result_outer.setContentsMargins(14, 12, 14, 12)
        result_outer.setSpacing(10)
        result_layout = QHBoxLayout()
        result_layout.setSpacing(14)
        result_outer.addLayout(result_layout)
        self.result_thumb = QLabel()
        self.result_thumb.setObjectName("thumb")
        self.result_thumb.setFixedSize(150, 112)
        self.result_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.result_thumb, 0, Qt.AlignmentFlag.AlignTop)
        result_texts = QVBoxLayout()
        self.result_title = QLabel("")
        self.result_title.setObjectName("result_title")
        self.result_text = QLabel("")
        self.result_text.setWordWrap(True)
        self.result_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        result_texts.addWidget(self.result_title)
        result_texts.addWidget(self.result_text, 1)
        result_buttons = QHBoxLayout()
        self.open_ortho_button = QPushButton("Abrir ortomosaico")
        self.open_ortho_button.clicked.connect(lambda: self.last_ortho and open_in_system(self.last_ortho))
        self.open_report_button = QPushButton("Abrir relatório PDF")
        self.open_report_button.clicked.connect(lambda: self.last_report and open_in_system(self.last_report))
        self.open_report_button.setEnabled(False)
        self.open_folder_button = QPushButton("Abrir pasta")
        self.open_folder_button.clicked.connect(self._open_output)
        self.open_error_button = QPushButton("Abrir arquivo do erro")
        self.open_error_button.clicked.connect(
            lambda: self.last_error_file and open_in_system(self.last_error_file)
        )
        for button in (self.open_ortho_button, self.open_report_button, self.open_folder_button, self.open_error_button):
            result_buttons.addWidget(button)
        result_buttons.addStretch(1)
        result_layout.addLayout(result_texts, 1)
        result_outer.addLayout(result_buttons)
        self.result_card.setVisible(False)
        right.addWidget(self.result_card)

        self.log_toggle = QToolButton()
        self.log_toggle.setObjectName("toggle_log")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.log_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.log_toggle.setText("Detalhes técnicos (log)")
        self.log_toggle.toggled.connect(self._show_log)
        right.addWidget(self.log_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20000)
        mono = QFont("DejaVu Sans Mono")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.log_view.setFont(mono)
        self.log_view.setVisible(False)
        right.addWidget(self.log_view, 1)
        self.right_stretch = QWidget()
        right.addWidget(self.right_stretch, 1)
        self._reset_steps()

    def _show_log(self, visible: bool) -> None:
        self.log_toggle.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        self.log_view.setVisible(visible)
        self.right_stretch.setVisible(not visible)
        self.settings.setValue("mostrar_log", visible)

    # -- settings, folders, estimates ----------------------------------------------

    def _restore_settings(self) -> None:
        self.photos_edit.setText(self.settings.value("fotos", "", str))
        self.output_edit.setText(self.settings.value("saida", "", str))
        level = self.settings.value("nivel", DEFAULT_LEVEL, str)
        self.level_buttons.get(level, self.level_buttons[DEFAULT_LEVEL]).setChecked(True)
        self.log_toggle.setChecked(self.settings.value("mostrar_log", False, bool))
        self._photos_changed()

    def _save_settings(self) -> None:
        self.settings.setValue("fotos", self.photos_edit.text())
        self.settings.setValue("saida", self.output_edit.text())
        self.settings.setValue("nivel", self._level())

    def _level(self) -> str:
        return next(k for k, b in self.level_buttons.items() if b.isChecked())

    def _start_folder(self, text: str) -> str:
        if text and Path(text).exists():
            return text
        shortcuts = _windows_home_folders()
        return str(shortcuts[0]) if shortcuts else str(Path.home())

    def _choose_photos(self) -> None:
        folder = choose_folder(self, "Pasta com as fotos do voo", self._start_folder(self.photos_edit.text()))
        if folder:
            self.photos_edit.setText(folder)
            self._photos_changed()

    def _choose_output(self) -> None:
        start = self.output_edit.text() or self.photos_edit.text()
        if start and not Path(start).exists():
            start = str(Path(start).parent)
        folder = choose_folder(self, "Pasta de saída (resultados)", self._start_folder(start))
        if folder:
            self.output_edit.setText(folder)

    def _photos_changed(self) -> None:
        text = self.photos_edit.text().strip()
        folder = Path(text) if text else None
        photos = []
        if folder is not None and folder.is_dir():
            try:
                photos = find_images(folder)
            except OSError:
                photos = []
        self.num_photos = len(photos)
        self.photos_thumb.setVisible(bool(photos))
        if folder is None:
            self.photos_info.setText("")
        elif not folder.is_dir():
            self.photos_info.setText("Pasta não encontrada.")
        elif not photos:
            self.photos_info.setText("0 fotos encontradas nesta pasta.")
        else:
            size_gb = sum(_file_size(p) for p in photos) / 1024**3
            size_text = f"{size_gb:.1f}".replace(".", ",")
            self.photos_info.setText(
                f"{self.num_photos} foto(s) encontrada(s), {size_text} GB.\nDe {photos[0].name} a {photos[-1].name}."
            )
            self._show_thumbnail(photos[0])
            if not self.output_edit.text().strip():
                self.output_edit.setText(str(folder.parent / f"HTRMapper_{project_name_from_folder(folder)}"))
        self._update_estimates()

    def _show_thumbnail(self, path: Path) -> None:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(self.photos_thumb.size() - QSize(4, 4), Qt.AspectRatioMode.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            self.photos_thumb.setText("sem prévia")
        else:
            self.photos_thumb.setPixmap(QPixmap.fromImage(image))

    def _update_estimates(self) -> None:
        if not self.num_photos:
            for label in self.level_time.values():
                label.setText("")
            self.level_source.setText("")
            return
        hardware = timing.hardware_signature()
        from_history = True
        for key in LEVELS:
            est = timing.estimate(key, self.num_photos, hardware, STEP_KEYS)
            total = sum(e.seconds for e in est.values())
            from_history &= all(e.from_history for e in est.values())
            self.level_time[key].setText(f"Tempo estimado:\n~{timing.format_duration(total)}")
        self.level_source.setText(
            f"Tempos para {self.num_photos} fotos neste computador, "
            + (
                "pelo histórico de processamentos dele."
                if from_history
                else "por uma estimativa inicial que fica mais precisa a cada processamento."
            )
        )

    # -- running -------------------------------------------------------------------

    def _request(self) -> AutoRequest | None:
        photos = self.photos_edit.text().strip()
        output = self.output_edit.text().strip()
        if not photos or not output:
            QMessageBox.warning(self, "HTRMapper", "Escolha a pasta das fotos (1) e a pasta de saída (2).")
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
        self.result_card.setVisible(False)
        self.last_output = request.output_folder
        self.last_report = None
        self.last_ortho = None
        self.last_error_file = None
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
        self.current_label.setText("Começando…")
        self.worker.start()
        self.timer.start(500)

    def _ask_about_existing(self, existing, level: str) -> bool | None:
        """True = resume, False = start over, None = do nothing."""
        box = QMessageBox(self)
        box.setWindowTitle("Já existe um processamento nesta pasta")
        box.setIcon(QMessageBox.Icon.Question)
        done = ", ".join(dict(STEPS)[k] for k in existing.steps_done) or "nenhuma etapa"
        if existing.resumable_with(level) and not existing.finished:
            box.setText(
                f"Esta pasta tem um processamento nível {LEVELS[existing.level].label} pela metade.\n\n"
                f"Já concluído: {done}."
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
                "Recomeçar apaga os dados de trabalho dele e substitui os resultados. "
                "Para guardar os dois, escolha outra pasta de saída."
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
            "Cancelar o processamento? O que já foi concluído fica salvo, e processar de novo continua de onde "
            "parou.\n\n(O ajuste com o GNSS não pode ser interrompido no meio: se estiver nele, o cancelamento "
            "acontece quando ele terminar.)",
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
        self.step_rows[key].paint(self.step_state[key], elapsed, self.spin)

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
        self.spin += 1
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
                self.step_rows[key].bar.setValue(int(1000 * fraction))
        total = sum(weights.values())
        self.overall_bar.setValue(int(1000 * done / total))
        ratio = min(max(measured / estimated, 0.3), 3.0) if estimated else 1.0
        if self.worker is not None:
            self.time_label.setText(
                f"Decorrido: {timing.format_duration(now - self.run_started)}   •   "
                f"Falta: ~{timing.format_duration(max(total - done, 0.0) * ratio)}"
            )

    def _end_run(self) -> None:
        self.timer.stop()
        self._set_running(False)
        self.worker = None

    def _show_result(self, failed: bool, title: str, text: str) -> None:
        self.result_card.setProperty("failed", "true" if failed else "false")
        _repolish(self.result_card)
        for child in self.result_card.findChildren(QLabel):
            _repolish(child)
        self.result_title.setText(title)
        self.result_text.setText(text)
        self.open_ortho_button.setVisible(not failed and self.last_ortho is not None)
        self.open_report_button.setVisible(not failed)
        self.open_error_button.setVisible(failed and self.last_error_file is not None)
        self.result_thumb.setVisible(not failed)
        self.result_card.setVisible(True)
        QTimer.singleShot(50, lambda: self.right_scroll.ensureWidgetVisible(self.result_card))
        QApplication.alert(self)

    def _on_finished(self, result) -> None:
        self._end_run()
        self.overall_bar.setValue(1000)
        total = timing.format_duration(result.total_seconds)
        self.time_label.setText(f"Concluído em {total}.")
        self.current_label.setText("<b>Processamento concluído.</b>")
        report = result.products.get("relatorio")
        ortho = result.products.get("ortho")
        self.last_report = Path(report) if report else None
        self.last_ortho = Path(ortho) if ortho else None
        self.open_report_button.setEnabled(self.last_report is not None)
        names = {"ortho": "Ortomosaico", "dem": "DEM", "relatorio": "Relatório", "log": "Log"}
        lines = [f"{label}: {Path(result.products[key]).name}" for key, label in names.items() if key in result.products]
        if result.warnings:
            lines += ["", "Avisos:"] + [f"• {w}" for w in result.warnings]
        self.result_thumb.clear()
        if self.last_ortho is not None:
            thumb = _raster_thumbnail(self.last_ortho, self.result_thumb.size() - QSize(4, 4))
            if thumb is not None:
                self.result_thumb.setPixmap(thumb)
        self._show_result(False, f"✓  Pronto! Processado em {total}", "\n".join(lines))

    def _on_failed(self, error: AutoProcessingError) -> None:
        self._end_run()
        self.last_error_file = Path(error.error_file) if error.error_file else None
        self.current_label.setText(f"<b style='color:{RED}'>Parou com erro.</b>")
        hint = "Corrigido o problema, clique em Processar de novo: continua de onde parou."
        self._show_result(True, "Erro no processamento", f"{error}\n\n{hint}")
        if not self.log_toggle.isChecked():
            self.log_toggle.setChecked(True)
        box = QMessageBox(self)
        box.setWindowTitle("Erro no processamento")
        box.setIcon(QMessageBox.Icon.Critical)
        text = str(error)
        if error.error_file:
            text += f"\n\nOs detalhes técnicos foram salvos em:\n{error.error_file}"
        box.setText(text + "\n\n" + hint)
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

    def _how_it_works(self) -> None:
        steps = "".join(f"<li><b>{label}</b>: {STEP_HELP[key]}</li>" for key, label in STEPS)
        box = QMessageBox(self)
        box.setWindowTitle("Como funciona")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(
            "<h3>Como funciona</h3>"
            "<p>Escolha a pasta das fotos, a pasta de saída e o nível, e clique em <b>Processar</b>. "
            "As etapas rodam sozinhas, uma depois da outra:</p>"
            f"<ol>{steps}</ol>"
            "<p>Resultados na pasta de saída: <b>ortomosaico</b> e <b>DEM</b> (GeoTIFF, SIRGAS 2000 / UTM) "
            "e o <b>relatório PDF</b>. Se der erro, ele aparece na tela e num arquivo ERRO_….txt na mesma pasta.</p>"
        )
        box.exec()

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
        if self.manual_window is not None:
            self.manual_window.close()  # closing the main window ends the program
        event.accept()


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _raster_thumbnail(path: Path, size: QSize) -> QPixmap | None:
    """Small preview of the orthomosaic, read from its internal overviews."""
    try:
        import numpy as np
        import rasterio

        with rasterio.open(path) as src:
            scale = max(src.width / size.width(), src.height / size.height(), 1.0)
            shape = (max(1, int(src.height / scale)), max(1, int(src.width / scale)))
            rgb = src.read([1, 2, 3], out_shape=(3, *shape))
            mask = src.dataset_mask(out_shape=shape)
        rgba = np.ascontiguousarray(np.dstack([np.moveaxis(rgb, 0, -1), mask]).astype(np.uint8))
        image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(image.copy())
    except Exception:  # noqa: BLE001 -- a missing preview never fails the run
        return None


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("HTRMapper")
    app.setWindowIcon(app_icon())
    window = AutoWindow()
    window.show()
    code = app.exec()
    # Destroy the windows while the application still exists. Left to the
    # interpreter's shutdown, PySide destroys the QApplication first and
    # then the widgets: "pure virtual method called" / segfault on every
    # close (seen with PySide6 6.12), and a non-zero exit code that made the
    # Windows shortcut report an error after a normal session.
    window.settings.sync()
    for widget in QApplication.topLevelWidgets():
        widget.deleteLater()
    del window
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()
    return code


if __name__ == "__main__":
    sys.exit(run())
