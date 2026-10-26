"""HTRMapper logo, drawn in code (the same design as windows/htrmapper.ico:
navy rounded square, white "H", green crop rows), so the package needs no
image files."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap

from htrmapper.core import theme

GREEN = "#4CAF50"


def logo_pixmap(size: int = 64) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, size, size), size * 0.18, size * 0.18)
    painter.fillPath(path, QColor(theme.PRIMARY_DARK))
    font = QFont("DejaVu Sans")
    font.setBold(True)
    font.setPixelSize(int(size * 0.5))
    painter.setFont(font)
    painter.setPen(QColor("white"))
    painter.drawText(QRectF(0, size * 0.04, size, size * 0.52), Qt.AlignmentFlag.AlignCenter, "H")
    pen = QPen(QColor(GREEN))
    pen.setWidthF(max(1.0, size * 0.035))
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    painter.setPen(pen)
    for i in range(5):
        y = size * (0.6 + 0.075 * i)
        painter.drawLine(QPointF(size * 0.15, y + size * 0.05), QPointF(size * 0.85, y - size * 0.05))
    painter.end()
    return pixmap


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(logo_pixmap(size))
    return icon
