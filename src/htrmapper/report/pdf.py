"""PDF processing report, compact, laid out like Metashape's processing
report (ARCHITECTURE.md sections 31 and 34):

1. Cover: project, date, level and the orthomosaic.
2. Survey data: camera positions over the image-overlap map, the survey
   figures (photos, flying height, GSD, area, tie points, projections,
   reprojection error), the camera, and a quality check with Pix4D's
   traffic-light criteria.
3. Camera calibration: image residuals and the calibrated coefficients.
4. Camera locations: each photo's difference to its GNSS position drawn as
   an ellipse over the orthomosaic, and the RMS errors.
5. Digital elevation model.
6. Processing parameters, times and system.

Every value comes from the processing outputs (`report.stats`); a value
that could not be measured is printed as "não disponível", never guessed.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
from matplotlib.patches import Ellipse, Rectangle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from htrmapper.core import theme
from htrmapper.core.project import Project
from htrmapper.report.stats import ReportStats, compute_report_stats

GREEN = colors.HexColor("#2E7D32")
YELLOW = colors.HexColor("#F9A825")
RED = colors.HexColor("#C62828")
STATUS_COLORS = {"ok": GREEN, "atencao": YELLOW, "ruim": RED, "na": colors.HexColor("#9E9E9E")}
STATUS_TEXT = {"ok": "OK", "atencao": "ATENÇÃO", "ruim": "RUIM", "na": "N/D"}
PAGE_WIDTH = A4[0] - 3.6 * cm
NA = "não disponível"
_QUALITY = {"baixa": "Baixa", "media": "Média", "alta": "Alta"}
_REFINED = {"focal_length": "focal", "principal_point": "centro óptico", "distortion": "distorção"}
# Metashape's overlap palette: 1 (red) ... 9, > 9 (blue).
OVERLAP_COLORS = [
    "#D73027",
    "#F46D43",
    "#FDAE61",
    "#D9EF8B",
    "#91CF60",
    "#1A9850",
    "#35B5A5",
    "#3B9AD9",
    "#3366CC",
    "#2B3FC4",
]


def _register_fonts() -> tuple[str, str]:
    """DejaVu (shipped with matplotlib) covers every character we print;
    the PDF core fonts would drop symbols such as ≥ or σ."""
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    try:
        if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("DejaVu", str(font_dir / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
            # So that <b> inside a paragraph switches to the bold face.
            pdfmetrics.registerFontFamily(
                "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold"
            )
        return "DejaVu", "DejaVu-Bold"
    except Exception:  # noqa: BLE001 -- fall back to the PDF core fonts
        return "Helvetica", "Helvetica-Bold"


def _fmt(value, digits: int = 2, unit: str = "") -> str:
    """Brazilian number format (1.234,56), or 'não disponível'."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return NA
    if isinstance(value, (int, np.integer)):
        text = f"{int(value):,}".replace(",", ".")
    else:
        text = f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{text} {unit}".strip()


def _duration(seconds: float) -> str:
    from htrmapper.pipeline.timing import format_duration

    return format_duration(seconds)


def _png(fig, dpi: int = 170) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _image(buffer: io.BytesIO, width: float, max_height: float | None = None) -> Image:
    from reportlab.lib.utils import ImageReader

    reader = ImageReader(buffer)
    w, h = reader.getSize()
    buffer.seek(0)
    height = width * h / w
    if max_height is not None and height > max_height:
        width, height = max_height * w / h, max_height
    return Image(buffer, width=width, height=height)


def _focal_change(project: Project) -> float | None:
    if not project.ba:
        return None
    changes = []
    for data in project.ba.camera_calibration.values():
        names = [n.strip() for n in data["params_info"].split(",")]
        for name in ("f", "fx", "fy"):
            if name in names:
                i = names.index(name)
                a, b = data["initial_params"][i], data["params"][i]
                if a:
                    changes.append(abs(b - a) / a)
    return max(changes) if changes else None


def _camera_sensor(project: Project) -> dict:
    """Model, resolution, focal length (mm) and pixel size (um) of the
    camera, from the EXIF (sensor width via the 35 mm crop factor)."""
    from htrmapper.geo.gsd import sensor_width_mm_from_crop_factor

    images = [i for i in project.images if i.pixel_width and i.pixel_height]
    if not images:
        return {}
    first = images[0]
    info = {
        "model": first.camera_model or "câmera",
        "resolution": f"{first.pixel_width} x {first.pixel_height}",
        "focal_mm": first.focal_length_mm,
        "pixel_um": None,
        "count": len(project.images),
    }
    if first.focal_length_mm and first.focal_length_35mm_equiv:
        sensor = sensor_width_mm_from_crop_factor(first.focal_length_mm, first.focal_length_35mm_equiv)
        info["pixel_um"] = sensor / first.pixel_width * 1000.0
    return info


def _pixel_text(sensor: dict) -> str:
    return f"{_fmt(sensor['pixel_um'], 2)} x {_fmt(sensor['pixel_um'], 2)} µm" if sensor.get("pixel_um") else NA


# -- figures ---------------------------------------------------------------------


def _read_preview(path: str, bands, max_px: int = 1600):
    import rasterio

    with rasterio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / max_px)
        shape = (max(1, int(src.height / scale)), max(1, int(src.width / scale)))
        out_shape = (len(bands), *shape) if isinstance(bands, list) else shape
        data = src.read(bands, out_shape=out_shape, masked=True)
        mask = src.dataset_mask(out_shape=shape) > 0
        bounds = src.bounds
    return data, mask, (bounds.left, bounds.right, bounds.bottom, bounds.top)


def _ortho_rgba(path: str, max_px: int = 1600):
    rgb, mask, extent = _read_preview(path, [1, 2, 3], max_px)
    rgba = np.dstack([np.moveaxis(np.asarray(rgb.filled(0)), 0, -1), (mask * 255).astype(np.uint8)])
    return rgba, extent


def _nice_length(span: float) -> float:
    """Largest 1, 2 or 5 x 10^n not above span (a round scale-bar length)."""
    span = max(span, 1e-9)
    magnitude = 10 ** np.floor(np.log10(span))
    return float(max(m * magnitude for m in (1, 2, 5) if m * magnitude <= span))


def _scale_bar(ax, extent) -> None:
    """Metashape-style scale bar (bottom right) and a north arrow (top left)."""
    left, right, bottom, top = extent
    width, height = right - left, top - bottom
    length = _nice_length(width / 5)
    x1 = right - 0.04 * width
    x0 = x1 - length
    y = bottom + 0.05 * height
    tick = 0.012 * height
    ax.plot([x0, x0, x1, x1], [y + tick, y, y, y + tick], color="black", lw=1.2, zorder=10)
    label = f"{length:g} m" if length < 1000 else f"{length / 1000:g} km"
    ax.text((x0 + x1) / 2, y - 0.012 * height, label, ha="center", va="top", fontsize=7, zorder=10)
    nx, ny = left + 0.04 * width, top - 0.11 * height
    ax.annotate(
        "N",
        xy=(nx, ny + 0.08 * height),
        xytext=(nx, ny),
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
        zorder=10,
    )


def _map_figure(extent, width_in: float = 7.2):
    left, right, bottom, top = extent
    aspect = (top - bottom) / max(right - left, 1e-9)
    fig, ax = plt.subplots(figsize=(width_in, min(width_in * aspect, 9.0)))
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _cover_figure(rgba, extent) -> io.BytesIO:
    fig, ax = _map_figure(extent)
    ax.imshow(rgba, extent=extent, interpolation="lanczos")
    _scale_bar(ax, extent)
    return _png(fig)


def _overlap_figure(stats: ReportStats) -> io.BytesIO:
    extent = stats.overlap_extent
    fig, ax = _map_figure(extent)
    cmap = ListedColormap(OVERLAP_COLORS)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 1e6], cmap.N)
    overlap = np.ma.masked_invalid(stats.overlap)
    ax.imshow(np.ma.masked_less(overlap, 1), cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
    if stats.camera_xy:
        xy = np.array(list(stats.camera_xy.values()))
        ax.scatter(xy[:, 0], xy[:, 1], s=3, color="black", zorder=3)
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "> 9"]
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in OVERLAP_COLORS]
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        frameon=False,
        handlelength=1.0,
    )
    _scale_bar(ax, extent)
    return _png(fig)


def _residuals_figure(stats: ReportStats) -> io.BytesIO | None:
    if stats.residual_grid is None:
        return None
    cx, cy, mdx, mdy, count = stats.residual_grid
    w, h = stats.residual_image_size
    gx, gy = np.meshgrid(cx, cy)
    ok = count > 0
    magnitude = np.hypot(mdx, mdy)
    largest = max(float(np.nanmax(magnitude[ok])), 1e-6)
    scale = 0.06 * w / largest  # longest arrow = 6% of the image width
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * h / w))
    ax.add_patch(Rectangle((0, 0), w, h, fill=False, lw=0.8, color="black"))
    quiver = ax.quiver(
        gx[ok],
        gy[ok],
        mdx[ok] * scale,
        -mdy[ok] * scale,
        magnitude[ok],
        cmap="RdYlGn_r",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0028,
        clim=(0, largest),
    )
    ax.set_xlim(-0.02 * w, 1.02 * w)
    ax.set_ylim(1.13 * h, -0.02 * h)
    ax.set_aspect("equal")
    ax.axis("off")
    bar = _nice_length(largest)
    ax.plot([w - bar * scale, w], [1.06 * h, 1.06 * h], color="black", lw=1.4)
    ax.text(w - bar * scale / 2, 1.08 * h, f"{bar:g} pix".replace(".", ","), ha="center", va="top", fontsize=7)
    fig.colorbar(quiver, ax=ax, shrink=0.6, pad=0.02, label="resíduo médio (pix)")
    return _png(fig)


def _camera_locations_figure(stats: ReportStats, ortho) -> io.BytesIO | None:
    items = [i for i in stats.images if i.d_east_cm is not None and i.name in stats.camera_xy]
    if not items:
        return None
    xy = np.array([stats.camera_xy[i.name] for i in items])
    de = np.array([i.d_east_cm for i in items])
    dn = np.array([i.d_north_cm for i in items])
    du = np.array([i.d_up_cm for i in items])
    fix = np.array([i.in_rmse for i in items])
    ref = fix if fix.any() else np.ones_like(fix)
    pad = 0.08 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0)
    extent = (xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    if ortho is not None:
        oe = ortho[1]
        extent = (min(extent[0], oe[0]), max(extent[1], oe[1]), min(extent[2], oe[2]), max(extent[3], oe[3]))
    fig, ax = _map_figure(extent, width_in=6.6)
    if ortho is not None:
        ax.imshow(ortho[0], extent=ortho[1], alpha=0.6, interpolation="lanczos")
    # Metashape's figure: ellipse axes = |dE| and |dN| times a magnification,
    # color = dZ. The magnification makes the RMS XY error 2,5% of the map
    # width. Only RTK FIX photos set the scales (a photo before the fix can
    # be meters off and would shrink every other ellipse to nothing).
    span = extent[1] - extent[0]
    rms_xy = float(np.sqrt(np.mean(de[ref] ** 2 + dn[ref] ** 2))) / 100.0
    magnification = 0.025 * span / max(rms_xy, 1e-4)
    limit = max(float(np.abs(du[ref]).max()), 0.5)
    levels = np.linspace(-limit, limit, 12)
    cmap = plt.get_cmap("turbo", 11)
    norm = BoundaryNorm(levels, cmap.N)
    floor = 0.15 * rms_xy  # keeps a photo with ~0 error visible
    for k in np.flatnonzero(ref):
        width = 2 * max(abs(de[k]) / 100.0, floor) * magnification
        height = 2 * max(abs(dn[k]) / 100.0, floor) * magnification
        ax.add_patch(
            Ellipse(
                xy[k], width, height, facecolor=cmap(norm(du[k])), edgecolor="black", lw=0.3, alpha=0.95, zorder=3
            )
        )
        ax.plot(*xy[k], marker=".", color="black", markersize=1.5, zorder=4)
    middle = (extent[0] + extent[1]) / 2
    for n, k in enumerate(np.flatnonzero(~fix)):
        ax.scatter(*xy[k], s=40, marker="D", facecolors="none", edgecolors="#C62828", lw=1.2, zorder=5)
        left_side = xy[k, 0] > middle  # label towards the map center, never off the edge
        ax.annotate(
            f"sem FIX: {np.hypot(de[k], dn[k]):.0f} cm XY, {du[k]:+.0f} cm Z".replace(".", ","),
            xy[k],
            textcoords="offset points",
            xytext=(-6 if left_side else 6, 4 + 9 * (n % 3)),
            ha="right" if left_side else "left",
            fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
            color="#C62828",
            zorder=5,
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, ticks=levels[::2])
    bar.set_label("dZ (cm)")
    bar.ax.set_yticklabels([f"{v:.1f}".replace(".", ",") for v in levels[::2]])
    ax.text(
        0.99,
        0.99,
        f"elipses: dX e dY ampliados {magnification:,.0f} x".replace(",", "."),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
        zorder=10,
    )
    _scale_bar(ax, extent)
    return _png(fig)


def _dem_figure(path: str) -> tuple[io.BytesIO, float | None, float | None]:
    z, _mask, extent = _read_preview(path, 1, 1400)
    z = np.ma.masked_invalid(np.ma.asarray(z, dtype=float))
    fig, ax = _map_figure(extent)
    low = high = None
    if z.count():
        low, high = float(z.min()), float(z.max())
        cell = (extent[1] - extent[0]) / z.shape[1]
        shade = LightSource(azdeg=315, altdeg=45).hillshade(z.filled(float(z.mean())), dx=cell, dy=cell, vert_exag=2)
        image = ax.imshow(z, cmap="turbo", extent=extent, vmin=low, vmax=high, interpolation="bilinear")
        ax.imshow(np.ma.array(shade, mask=np.ma.getmaskarray(z)), cmap="gray", alpha=0.25, extent=extent)
        bar = fig.colorbar(image, ax=ax, shrink=0.55, pad=0.02, ticks=[low, high])
        bar.ax.set_yticklabels([f"{low:.0f} m", f"{high:.0f} m"])
    _scale_bar(ax, extent)
    return _png(fig), low, high


# -- document --------------------------------------------------------------------


class _Builder:
    def __init__(self, project: Project, level, stats: ReportStats, warnings: list[str]):
        self.project = project
        self.level = level
        self.stats = stats
        self.warnings = warnings
        self.font, self.bold = _register_fonts()
        base = getSampleStyleSheet()
        navy = colors.HexColor(theme.PRIMARY_DARK)
        self.title_style = ParagraphStyle(
            "title", parent=base["Title"], fontName=self.bold, fontSize=22, textColor=navy, alignment=TA_CENTER
        )
        self.subtitle = ParagraphStyle(
            "subtitle", parent=base["BodyText"], fontName=self.bold, fontSize=11, alignment=TA_CENTER, leading=16
        )
        self.h1 = ParagraphStyle(
            "h1", parent=base["Heading1"], fontName=self.bold, fontSize=18, textColor=navy, spaceAfter=12
        )
        self.h2 = ParagraphStyle(
            "h2", parent=base["Heading2"], fontName=self.bold, fontSize=11, textColor=navy, spaceBefore=12
        )
        self.body = ParagraphStyle("body", parent=base["BodyText"], fontName=self.font, fontSize=9, leading=12)
        self.caption = ParagraphStyle("caption", parent=self.body, alignment=TA_CENTER, fontSize=8.5, leading=11)
        self.small = ParagraphStyle("small", parent=self.body, fontSize=7.5, leading=9.5)
        self.cell = ParagraphStyle("cell", parent=self.body, fontSize=8.5, leading=10.5)
        self.story: list = []
        self.ortho = None
        if project.ortho and Path(project.ortho.raster_path).is_file():
            self.ortho = _ortho_rgba(project.ortho.raster_path)

    def p(self, text: str, style=None) -> None:
        self.story.append(Paragraph(text, style or self.body))

    def boxed_table(self, rows: list[list], widths=None, extra=None) -> Table:
        """Grid table with a bold header row, like Metashape's tables."""
        data = [[Paragraph(f"<b>{c}</b>", self.cell) for c in rows[0]]]
        data += [[c if isinstance(c, Paragraph) else Paragraph(str(c), self.cell) for c in row] for row in rows[1:]]
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#333333")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
                + (extra or [])
            )
        )
        return table

    def pairs_table(self, rows: list[tuple[str, str, str, str]]) -> Table:
        """Borderless two-pair summary (Metashape's "Survey Data" block)."""
        data = [[Paragraph(c, self.body) for c in row] for row in rows]
        widths = [0.22 * PAGE_WIDTH, 0.25 * PAGE_WIDTH, 0.27 * PAGE_WIDTH, 0.26 * PAGE_WIDTH]
        table = Table(data, colWidths=widths)
        table.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return table

    # -- pages ---------------------------------------------------------------

    def cover(self) -> None:
        p = self.project
        when = (p.processing.finished_at or p.processing.started_at) if p.processing else ""
        try:
            date = datetime.fromisoformat(when).strftime("%d/%m/%Y %H:%M") if when else ""
        except ValueError:
            date = ""
        date = date or datetime.now().strftime("%d/%m/%Y %H:%M")
        self.story += [
            Spacer(1, 1.4 * cm),
            Paragraph("Relatório de Processamento", self.title_style),
            Spacer(1, 0.3 * cm),
            Paragraph(f"Projeto {p.name}", self.subtitle),
            Paragraph(date, self.subtitle),
            Paragraph(f"Nível {self.level.label} — {self.level.summary}", self.subtitle),
            Spacer(1, 0.8 * cm),
        ]
        if self.ortho is not None:
            self.story.append(_image(_cover_figure(*self.ortho), PAGE_WIDTH, max_height=18.5 * cm))

    def survey_data(self) -> None:
        p, s = self.project, self.stats
        self.story += [PageBreak(), Paragraph("Dados do levantamento", self.h1)]
        if s.overlap is not None:
            self.story.append(_image(_overlap_figure(s), PAGE_WIDTH * 0.9, max_height=11 * cm))
            self.p("Fig. 1. Posições das câmeras e sobreposição das fotos (fotos por ponto do terreno).", self.caption)
            self.story.append(Spacer(1, 0.4 * cm))
        aligned = _fmt(p.sfm.num_registered) if p.sfm else NA
        self.story.append(
            self.pairs_table(
                [
                    ("Fotos:", _fmt(len(p.images)), "Fotos alinhadas:", aligned),
                    (
                        "Altura de voo:",
                        _fmt(s.mean_flying_height_m, 0, "m"),
                        "Tie points:",
                        _fmt(s.num_points3d) if s.num_points3d else NA,
                    ),
                    (
                        "Resolução:",
                        _fmt(s.gsd_cm, 2, "cm/pix"),
                        "Projeções:",
                        _fmt(s.num_observations) if s.num_observations else NA,
                    ),
                    (
                        "Área coberta:",
                        _fmt(s.covered_area_ha, 2, "ha"),
                        "Erro de reprojeção:",
                        _fmt(s.reprojection_rms_px, 3, "pix"),
                    ),
                ]
            )
        )
        sensor = _camera_sensor(p)
        if sensor:
            self.story.append(Spacer(1, 0.4 * cm))
            self.story.append(
                self.boxed_table(
                    [
                        ["Modelo da câmera", "Resolução", "Distância focal", "Tamanho do pixel", "Pré-calibrada"],
                        [
                            sensor["model"],
                            sensor["resolution"],
                            _fmt(sensor["focal_mm"], 2, "mm"),
                            _pixel_text(sensor),
                            "Não",
                        ],
                    ]
                )
            )
            self.p("Tabela 1. Câmera.", self.caption)
        self.quality_check()

    def quality_check(self) -> None:
        p, s = self.project, self.stats
        rows = [["Verificação de qualidade", "Valor", "Situação", "Critério (Pix4D)"]]
        statuses = []

        def add(item, value, status, rule):
            badge = Paragraph(f"<font color='white'><b>{STATUS_TEXT[status]}</b></font>", self.cell)
            rows.append([item, value, badge, rule])
            statuses.append(status)

        if s.median_keypoints is not None:
            k = s.median_keypoints
            add(
                "Pontos-chave por foto (mediana)",
                _fmt(int(k)),
                "ok" if k > 10000 else "atencao" if k >= 500 else "ruim",
                "verde > 10.000",
            )
        if p.sfm:
            frac = p.sfm.num_registered / max(1, p.sfm.num_images_input)
            add(
                "Fotos alinhadas",
                f"{frac:.0%}",
                "ok" if frac > 0.95 else "atencao" if frac >= 0.60 else "ruim",
                "verde > 95%",
            )
        change = _focal_change(p)
        if change is not None:
            add(
                "Variação da focal na calibração",
                f"{change:.2%}".replace(".", ","),
                "ok" if change < 0.05 else "atencao" if change <= 0.20 else "ruim",
                "verde < 5%",
            )
        if s.median_tie_points is not None:
            t = s.median_tie_points
            add(
                "Correspondências por foto (mediana)",
                _fmt(int(t)),
                "ok" if t > 1000 else "atencao" if t >= 100 else "ruim",
                "verde > 1.000",
            )
        mean_error = p.ba.mean_reprojection_error_px if p.ba else None
        if mean_error is not None:
            add(
                "Erro médio de reprojeção",
                _fmt(mean_error, 3, "pix"),
                "ok" if mean_error < 1.0 else "atencao" if mean_error <= 2.0 else "ruim",
                "verde < 1 pix",
            )
        if p.ba and p.ba.rmse_xy_cm is not None:
            sigma_cm = p.gnss_accuracy.accuracy.xy_sigma_m * 100
            add(
                "Câmeras x GNSS (RMS XY)",
                _fmt(p.ba.rmse_xy_cm, 1, "cm"),
                "atencao" if p.ba.rmse_xy_cm <= 3 * sigma_cm else "ruim",
                "sem pontos de controle: amarelo, validar em campo",
            )
        if len(rows) == 1:
            return
        extra = [("BACKGROUND", (2, i + 1), (2, i + 1), STATUS_COLORS[st]) for i, st in enumerate(statuses)]
        widths = [0.37 * PAGE_WIDTH, 0.15 * PAGE_WIDTH, 0.13 * PAGE_WIDTH, 0.35 * PAGE_WIDTH]
        self.story.append(Spacer(1, 0.4 * cm))
        self.story.append(self.boxed_table(rows, widths, extra))
        self.p("Tabela 2. Verificação de qualidade.", self.caption)

    def calibration(self) -> None:
        p, s = self.project, self.stats
        if not p.ba or not p.ba.camera_calibration:
            return
        self.story += [PageBreak(), Paragraph("Calibração da câmera", self.h1)]
        sensor = _camera_sensor(p)
        figure = _residuals_figure(s)
        if figure is not None:
            self.story.append(_image(figure, PAGE_WIDTH * 0.88, max_height=9.5 * cm))
            self.p(f"Fig. 2. Resíduos na imagem para {sensor.get('model', 'a câmera')}.", self.caption)
            self.story.append(Spacer(1, 0.4 * cm))
        if sensor:
            self.p(f"<b>{sensor['model']}</b>")
            self.p(
                f"{sensor['count']} fotos. Modelo OPENCV: focal, centro óptico, distorção radial (k1, k2) "
                "e tangencial (p1, p2), calibrados no ajuste junto com o RTK."
            )
            self.story.append(Spacer(1, 0.15 * cm))
            self.story.append(
                self.pairs_table(
                    [
                        ("<b>Tipo</b>", "<b>Resolução</b>", "<b>Distância focal</b>", "<b>Tamanho do pixel</b>"),
                        ("Frame", sensor["resolution"], _fmt(sensor["focal_mm"], 2, "mm"), _pixel_text(sensor)),
                    ]
                )
            )
            self.story.append(Spacer(1, 0.3 * cm))
        for data in p.ba.camera_calibration.values():
            names = [n.strip() for n in data["params_info"].split(",")]
            rows = [["", "Inicial", "Calibrado", "Variação"]]
            for name, a, b in zip(names, data["initial_params"], data["params"]):
                digits = 3 if name in ("f", "fx", "fy", "cx", "cy") else 6
                if name in ("f", "fx", "fy") and a:
                    change = f"{(b - a) / a:+.3%}".replace(".", ",")
                else:
                    change = _fmt(b - a, digits)
                rows.append([f"<b>{name}</b>", _fmt(float(a), digits), _fmt(float(b), digits), change])
            widths = [0.16 * PAGE_WIDTH, 0.27 * PAGE_WIDTH, 0.27 * PAGE_WIDTH, 0.22 * PAGE_WIDTH]
            self.story.append(self.boxed_table(rows, widths))
            self.p("Tabela 3. Coeficientes de calibração (focal e centro óptico em pixels).", self.caption)
        for note in p.ba.intrinsics_notes:
            self.p("Aviso: " + note, self.small)

    def camera_locations(self) -> None:
        p, s = self.project, self.stats
        if not p.ba:
            return
        self.story += [PageBreak(), Paragraph("Posição das câmeras", self.h1)]
        figure = _camera_locations_figure(s, self.ortho)
        if figure is not None:
            self.story.append(_image(figure, PAGE_WIDTH * 0.9, max_height=12.5 * cm))
            self.p("Fig. 3. Posições das câmeras e diferenças para o GNSS (RTK).", self.caption)
            self.p(
                "A cor da elipse indica a diferença em Z; a forma, as diferenças em X e Y. "
                "O ponto preto é a posição da câmera após o ajuste.",
                self.caption,
            )
            self.story.append(Spacer(1, 0.4 * cm))
        ba = p.ba
        self.story.append(
            self.boxed_table(
                [
                    ["Erro X (cm)", "Erro Y (cm)", "Erro Z (cm)", "Erro XY (cm)", "Total (cm)", "Máximo (cm)"],
                    [
                        _fmt(ba.rmse_x_cm, 2),
                        _fmt(ba.rmse_y_cm, 2),
                        _fmt(ba.rmse_z_cm, 2),
                        _fmt(ba.rmse_xy_cm, 2),
                        _fmt(ba.rmse_total_cm, 2),
                        _fmt(ba.max_error_cm, 2),
                    ],
                ]
            )
        )
        self.p("Tabela 4. Erro médio (RMS) das posições das câmeras, fotos com RTK FIX.", self.caption)
        self.p("X - Leste, Y - Norte, Z - Altitude elipsoidal.", self.caption)
        self.story.append(Spacer(1, 0.3 * cm))
        self.p(
            f"Precisão do GNSS usada no ajuste: {_fmt(p.gnss_accuracy.accuracy.xy_sigma_m * 100, 1, 'cm')} (XY) e "
            f"{_fmt(p.gnss_accuracy.accuracy.z_sigma_m * 100, 1, 'cm')} (Z), ou o desvio informado pelo RTK da foto, "
            f"se maior. {ba.num_images_with_gnss_prior} de {ba.num_images_adjusted} fotos entraram como observação GNSS.",
            self.small,
        )
        if ba.images_excluded_from_rmse:
            self.p(
                "Sem RTK FIX (posicionadas só pelos tie points, fora do RMSE): "
                + ", ".join(ba.images_excluded_from_rmse)
                + ".",
                self.small,
            )
        self.p(
            "Estes números medem a coerência entre as fotos e o RTK. A exatidão no terreno só é comprovada com "
            "pontos de verificação medidos em campo.",
            self.small,
        )

    def elevation_model(self) -> None:
        p, s = self.project, self.stats
        if not p.dem or not Path(p.dem.raster_path).is_file():
            return
        self.story += [PageBreak(), Paragraph("Modelo digital de elevação", self.h1)]
        figure, low, high = _dem_figure(p.dem.raster_path)
        self.story.append(_image(figure, PAGE_WIDTH * 0.9, max_height=14 * cm))
        self.p("Fig. 4. Modelo digital de elevação (superfície) reconstruído.", self.caption)
        self.story.append(Spacer(1, 0.4 * cm))
        d = p.dem
        covered = d.width_px * d.height_px * (1 - d.nodata_fraction)
        interpolated = f"{d.num_cells_interpolated / covered:.1%}".replace(".", ",") if covered else NA
        self.story.append(
            self.pairs_table(
                [
                    (
                        "Resolução:",
                        _fmt(d.resolution_m * 100, 0, "cm/pix"),
                        "Densidade de pontos:",
                        _fmt(s.dense_density_per_m2, 1, "pontos/m²"),
                    ),
                    (
                        "Altitudes:",
                        f"{_fmt(low, 1)} a {_fmt(high, 1, 'm')}" if low is not None else NA,
                        "Células interpoladas:",
                        interpolated,
                    ),
                ]
            )
        )
        self.story.append(Spacer(1, 0.2 * cm))
        self.p("Altitude elipsoidal (GRS80), a mesma referência do GNSS do drone.", self.small)

    def parameters(self) -> None:
        from htrmapper.geo.crs import CoordinateReferenceSystem
        from htrmapper.pipeline.auto import _system_lines

        p, s, level = self.project, self.stats, self.level
        self.story += [PageBreak(), Paragraph("Parâmetros de processamento", self.h1)]
        steps = p.processing.steps_done if p.processing else {}

        def took(key):
            return _duration(steps[key]["seconds"]) if key in steps else NA

        ba = p.ba
        size = level.sfm.max_image_size
        view = level.ortho.max_view_angle_deg
        rows: list[tuple[int, str, str]] = [
            (0, "Geral", ""),
            (1, "Fotos", _fmt(len(p.images))),
            (1, "Fotos alinhadas", _fmt(p.sfm.num_registered) if p.sfm else NA),
            (
                1,
                "Sistema de coordenadas",
                f"{CoordinateReferenceSystem(p.crs.project_epsg).name} (EPSG:{p.crs.project_epsg})",
            ),
            (1, "Referência vertical", "Altitude elipsoidal (GRS80)"),
            (1, "Georreferenciamento", "RTK das câmeras, sem pontos de controle"),
            (1, "Nível de processamento", f"{level.label} — {level.summary}"),
            (0, "Alinhamento", ""),
            (1, "Pontos-chave por foto (limite)", _fmt(level.sfm.key_point_limit)),
            (1, "Tamanho das fotos na extração", "original" if size < 0 else f"{size} pix"),
            (1, "Pares por foto (vizinhas, pelo GNSS)", _fmt(level.sfm.spatial_max_neighbors)),
            (1, "Tempo", took("alinhar")),
            (0, "Ajuste com GNSS (tie points)", ""),
            (
                1,
                "Tie points",
                f"{_fmt(s.num_points3d)} (de {_fmt(ba.tie_points_before_cleanup)} antes da limpeza)"
                if ba and ba.tie_points_before_cleanup
                else _fmt(s.num_points3d),
            ),
            (1, "Erro de reprojeção RMS", _fmt(s.reprojection_rms_px, 3, "pix")),
            (1, "Erro de reprojeção máximo", _fmt(s.reprojection_max_px, 2, "pix")),
            (1, "Fotos por tie point (média)", _fmt(s.mean_track_length, 2)),
            (1, "Parâmetros calibrados", ", ".join(_REFINED.get(n, n) for n in ba.intrinsics_refined) or "nenhum")
            if ba
            else (1, "Parâmetros calibrados", NA),
            (1, "Observações GNSS", f"{ba.num_images_with_gnss_prior} fotos (só RTK FIX)" if ba else NA),
            (
                1,
                "Limpeza de tie points",
                f"{len(ba.tie_point_cleanup_passes)} passes" if ba and ba.tie_point_cleanup_passes else "não",
            ),
            (1, "Convergência", ("sim" if ba.converged else "NÃO") if ba else NA),
            (1, "Tempo", took("ajustar")),
            (0, "Nuvem densa", ""),
            (1, "Pontos", _fmt(p.mvs.num_points) if p.mvs else NA),
            (1, "Qualidade", _QUALITY.get(p.mvs.quality, p.mvs.quality) if p.mvs else NA),
            (1, "Densidade", _fmt(s.dense_density_per_m2, 1, "pontos/m²")),
            (1, "Tempo", took("densa")),
            (0, "Modelo digital de elevação", ""),
            (1, "Tamanho", f"{_fmt(p.dem.width_px)} x {_fmt(p.dem.height_px)}" if p.dem else NA),
            (1, "Resolução", _fmt(p.dem.resolution_m * 100, 0, "cm/pix") if p.dem else NA),
            (1, "Picos removidos", f"{_fmt(p.dem.num_cells_despiked)} células" if p.dem else NA),
            (1, "Interpolação", "só buracos internos, sem extrapolar para fora da área"),
            (1, "Arquivo", Path(p.dem.raster_path).name if p.dem else NA),
            (1, "Tempo", took("dem")),
            (0, "Ortomosaico", ""),
            (1, "Tamanho", f"{_fmt(p.ortho.width_px)} x {_fmt(p.ortho.height_px)}" if p.ortho else NA),
            (1, "Resolução", _fmt(p.ortho.resolution_m * 100, 2, "cm/pix") if p.ortho else NA),
            (1, "Superfície", "DEM"),
            (
                1,
                "Mistura",
                "foto mais vertical em cada pixel" + (f", até {view:g}° da vertical" if view else ""),
            ),
            (1, "Fotos usadas", _fmt(p.ortho.num_cameras_used) if p.ortho else NA),
            (1, "Arquivo", Path(p.ortho.raster_path).name if p.ortho else NA),
            (1, "Tempo", took("ortho")),
            (0, "Sistema", ""),
            (1, "Software", f"HTRMapper {p.processing.software_version if p.processing else ''}".strip()),
        ]
        for line in _system_lines():
            # "CPU: ..." / "GPU: ..." / "Python 3.x em ..." / "pycolmap 4.x (CUDA: sim)"
            key, _, value = line.partition(": ") if line[:4] in ("CPU:", "GPU:") else line.partition(" ")
            rows.append((1, key, value))
        total = sum(v.get("seconds", 0) for v in steps.values())
        if total:
            rows.append((1, "Tempo total de processamento", _duration(total)))
        data = []
        for indent, key, value in rows:
            style = ParagraphStyle(f"k{indent}", parent=self.cell, leftIndent=14 * indent)
            data.append([Paragraph(f"<b>{key}</b>" if indent == 0 else key, style), Paragraph(value, self.cell)])
        table = Table(data, colWidths=[0.45 * PAGE_WIDTH, 0.55 * PAGE_WIDTH])
        table.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
        self.story.append(table)
        if self.warnings:
            self.story.append(Paragraph("Avisos", self.h2))
            for warning in self.warnings:
                self.p("• " + warning, self.small)


def write_pdf_report(
    project: Project, path: Path, level, log_path: Path | None = None, warnings: list[str] | None = None
) -> Path:
    stats = compute_report_stats(project, level.ortho.resolution_scale)
    builder = _Builder(project, level, stats, warnings or [])
    builder.cover()
    builder.survey_data()
    builder.calibration()
    builder.camera_locations()
    builder.elevation_model()
    builder.parameters()
    if log_path is not None:
        builder.p(f"Log completo do processamento: {Path(log_path).name}", builder.small)

    font = builder.font

    def decorate(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor(theme.TEXT_MUTED))
        if doc.page > 1:
            canvas.drawString(1.8 * cm, A4[1] - 1.0 * cm, f"Relatório de Processamento — {project.name}")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Página {doc.page}")
        canvas.drawString(1.8 * cm, 1.0 * cm, "HTRMapper")
        canvas.restoreState()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.6 * cm,
        title=f"Relatório de Processamento — {project.name}",
        author="HTRMapper",
    )
    doc.build(builder.story, onFirstPage=decorate, onLaterPages=decorate)
    return path
