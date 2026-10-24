"""PDF processing report, modelled on Pix4D's quality report and
Metashape's processing report (ARCHITECTURE.md section 31).

Sections: summary; quality check (Pix4D's traffic-light criteria plus
reprojection error and georeferencing); previews of the orthomosaic and
DEM; camera positions and their differences to the GNSS; image overlap
map; camera calibration (initial vs. optimized, distortion profile, image
residuals); bundle adjustment (tie points, track lengths, cleanup passes);
dense cloud, DEM and orthomosaic details; processing parameters and times;
hardware and software; warnings; and a per-photo table in the appendix.

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
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

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
STEP_LABELS = {
    "importar": "Importar fotos",
    "alinhar": "Alinhamento",
    "ajustar": "Ajuste com GNSS",
    "densa": "Nuvem densa",
    "dem": "DEM",
    "ortho": "Ortomosaico",
    "relatorio": "Relatório",
}


def _register_fonts() -> tuple[str, str]:
    """DejaVu (shipped with matplotlib) covers every character we print;
    the PDF core fonts would drop symbols such as ≥ or σ."""
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    try:
        if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("DejaVu", str(font_dir / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:  # noqa: BLE001 -- fall back to the PDF core fonts
        return "Helvetica", "Helvetica-Bold"


def _fmt(value, digits: int = 2, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return NA
    if isinstance(value, int):
        text = f"{value:,}".replace(",", ".")
    else:
        text = f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{text} {unit}".strip()


def _png(fig, dpi: int = 150) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _image(buffer: io.BytesIO, width: float) -> Image:
    from reportlab.lib.utils import ImageReader

    reader = ImageReader(buffer)
    w, h = reader.getSize()
    buffer.seek(0)
    return Image(buffer, width=width, height=width * h / w)


class _Builder:
    def __init__(self, project: Project, level, stats: ReportStats, warnings: list[str]):
        self.project = project
        self.level = level
        self.stats = stats
        self.warnings = warnings
        self.font, self.bold = _register_fonts()
        base = getSampleStyleSheet()
        self.h1 = ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=self.bold,
            fontSize=15,
            textColor=colors.HexColor(theme.PRIMARY_DARK),
        )
        self.h2 = ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=self.bold,
            fontSize=12,
            textColor=colors.HexColor(theme.PRIMARY_DARK),
            spaceBefore=10,
        )
        self.body = ParagraphStyle("body", parent=base["BodyText"], fontName=self.font, fontSize=9, leading=12)
        self.small = ParagraphStyle("small", parent=self.body, fontSize=7.5, leading=9.5)
        self.cell = ParagraphStyle("cell", parent=self.body, fontSize=8, leading=10)
        self.title = ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=self.bold,
            fontSize=20,
            textColor=colors.HexColor(theme.PRIMARY_DARK),
            alignment=TA_CENTER,
        )
        self.story: list = []

    # -- helpers -------------------------------------------------------------

    def p(self, text: str, style=None) -> None:
        self.story.append(Paragraph(text, style or self.body))

    def table(self, rows: list[list], widths: list[float] | None = None, header: bool = True, extra=None) -> Table:
        data = [[Paragraph(str(c), self.cell) if not isinstance(c, Paragraph) else c for c in row] for row in rows]
        if header:
            data[0] = [Paragraph(f"<font color='white'><b>{c}</b></font>", self.cell) for c in rows[0]]
        table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(theme.BORDER)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor(theme.SURFACE)]),
        ]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(theme.PRIMARY_DARK))]
        table.setStyle(TableStyle(style + (extra or [])))
        return table

    def kv(self, rows: list[tuple[str, str]]) -> Table:
        return self.table(
            [[f"<b>{k}</b>", v] for k, v in rows], widths=[0.38 * PAGE_WIDTH, 0.62 * PAGE_WIDTH], header=False
        )

    # -- sections ------------------------------------------------------------

    def cover(self) -> None:
        p = self.project
        self.story += [Spacer(1, 1.2 * cm), Paragraph("Relatório de Processamento", self.title), Spacer(1, 0.2 * cm)]
        self.p(f"<para alignment='center'><b>{p.name}</b> — HTRMapper</para>")
        self.story.append(Spacer(1, 0.6 * cm))
        processing = p.processing
        camera = ", ".join(sorted({i.camera_model or "?" for i in p.images}))
        aligned = f"{p.sfm.num_registered} de {p.sfm.num_images_input}" if p.sfm else NA
        times = processing.steps_done if processing else {}
        total_s = sum(v.get("seconds", 0) for v in times.values())
        from htrmapper.geo.crs import CoordinateReferenceSystem

        crs = CoordinateReferenceSystem(p.crs.project_epsg)
        self.story.append(
            self.kv(
                [
                    ("Projeto", p.name),
                    ("Processado em", datetime.now().strftime("%d/%m/%Y %H:%M")),
                    ("Nível de processamento", f"{self.level.label} — {self.level.summary}"),
                    ("Câmera", camera),
                    ("Fotos alinhadas", aligned),
                    ("GSD médio", _fmt(self.stats.gsd_cm, 2, "cm/pixel")),
                    ("Altura média de voo (sobre o terreno)", _fmt(self.stats.mean_flying_height_m, 1, "m")),
                    ("Área coberta pelo ortomosaico", _fmt(self.stats.covered_area_ha, 2, "ha")),
                    ("Sistema de coordenadas", f"EPSG:{p.crs.project_epsg} — {crs.name}"),
                    ("Referência vertical", "Altura elipsoidal (GRS80), a mesma do GNSS do drone"),
                    ("Georreferenciamento", "RTK das câmeras (sem pontos de controle no solo)"),
                    ("Tempo total de processamento", _duration(total_s) if total_s else NA),
                    ("Software", f"HTRMapper {processing.software_version if processing else ''}"),
                ]
            )
        )

    def quality_check(self) -> None:
        s, p = self.stats, self.project
        self.story.append(Paragraph("1. Verificação de qualidade", self.h1))
        self.p(
            "Critérios do relatório de qualidade do Pix4D (fotos, conjunto, otimização da câmera e correspondências), "
            "mais o erro de reprojeção e o georreferenciamento. Verde = ok, amarelo = atenção, vermelho = ruim."
        )
        rows = [["Item", "Valor", "Situação", "Critério"]]
        statuses = []

        def add(item, value, status, rule):
            rows.append(
                [item, value, Paragraph(f"<font color='white'><b>{STATUS_TEXT[status]}</b></font>", self.cell), rule]
            )
            statuses.append(status)

        if s.median_keypoints is not None:
            k = s.median_keypoints
            add(
                "Pontos-chave por foto (mediana)",
                _fmt(int(k)),
                "ok" if k > 10000 else "atencao" if k >= 500 else "ruim",
                "verde > 10.000; vermelho < 500",
            )
        else:
            add("Pontos-chave por foto (mediana)", NA, "na", "")
        if p.sfm:
            frac = p.sfm.num_registered / max(1, p.sfm.num_images_input)
            add(
                "Fotos alinhadas",
                f"{p.sfm.num_registered}/{p.sfm.num_images_input} ({frac:.0%})",
                "ok" if frac > 0.95 else "atencao" if frac >= 0.60 else "ruim",
                "verde > 95%; vermelho < 60%",
            )
        change = _focal_change(p)
        if change is not None:
            add(
                "Otimização da câmera (variação da focal)",
                f"{change:.2%}".replace(".", ","),
                "ok" if change < 0.05 else "atencao" if change <= 0.20 else "ruim",
                "verde < 5%; vermelho > 20%",
            )
        if s.median_tie_points is not None:
            t = s.median_tie_points
            add(
                "Correspondências por foto (mediana)",
                _fmt(int(t)),
                "ok" if t > 1000 else "atencao" if t >= 100 else "ruim",
                "verde > 1.000; vermelho < 100",
            )
        if p.ba and p.ba.mean_reprojection_error_px is not None:
            e = p.ba.mean_reprojection_error_px
            add(
                "Erro médio de reprojeção",
                _fmt(e, 3, "pixel"),
                "ok" if e < 1.0 else "atencao" if e <= 2.0 else "ruim",
                "verde < 1 px; vermelho > 2 px",
            )
        if p.ba and p.ba.rmse_xy_cm is not None:
            sigma_cm = p.gnss_accuracy.accuracy.xy_sigma_m * 100
            add(
                "Georreferenciamento (câmera × GNSS, XY)",
                f"RMSE {_fmt(p.ba.rmse_xy_cm, 1, 'cm')}",
                "atencao" if p.ba.rmse_xy_cm <= 3 * sigma_cm else "ruim",
                "sem pontos de controle no solo: amarelo, como no Pix4D; validar com alvos de campo",
            )
        extra = [("BACKGROUND", (2, i + 1), (2, i + 1), STATUS_COLORS[status]) for i, status in enumerate(statuses)]
        self.story.append(
            self.table(rows, [0.32 * PAGE_WIDTH, 0.2 * PAGE_WIDTH, 0.13 * PAGE_WIDTH, 0.35 * PAGE_WIDTH], extra=extra)
        )

    def previews(self) -> None:
        p = self.project
        self.story.append(Paragraph("2. Prévias", self.h1))
        if p.ortho and Path(p.ortho.raster_path).is_file():
            self.story.append(Paragraph("Ortomosaico", self.h2))
            self.story.append(_image(_ortho_preview(p.ortho.raster_path), PAGE_WIDTH * 0.9))
        if p.dem and Path(p.dem.raster_path).is_file():
            self.story.append(Paragraph("DEM (modelo digital de superfície)", self.h2))
            self.story.append(_image(_dem_preview(p.dem.raster_path), PAGE_WIDTH * 0.9))

    def cameras(self) -> None:
        p, s = self.project, self.stats
        self.story.append(PageBreak())
        self.story.append(Paragraph("3. Posições das câmeras", self.h1))
        errors = [i for i in s.images if i.d_east_cm is not None]
        if errors:
            self.p(
                "Posição de cada foto após o ajuste, com a diferença para a posição do GNSS (RTK) desenhada como "
                "seta (ampliada) e colorida pela diferença vertical. Fotos sem RTK FIX (losango) entram no ajuste "
                "com o próprio desvio do RTK e ficam fora do RMSE."
            )
            self.story.append(_image(_camera_error_plot(p, errors), PAGE_WIDTH * 0.85))
        if p.ba:
            ba = p.ba
            self.story.append(Paragraph("Diferença câmera × GNSS (fotos com RTK FIX)", self.h2))
            self.story.append(
                self.table(
                    [
                        ["Leste (X)", "Norte (Y)", "Vertical (Z)", "Horizontal (XY)", "Total", "Máximo"],
                        [
                            _fmt(ba.rmse_x_cm, 2, "cm"),
                            _fmt(ba.rmse_y_cm, 2, "cm"),
                            _fmt(ba.rmse_z_cm, 2, "cm"),
                            _fmt(ba.rmse_xy_cm, 2, "cm"),
                            _fmt(ba.rmse_total_cm, 2, "cm"),
                            _fmt(ba.max_error_cm, 2, "cm"),
                        ],
                    ]
                )
            )
            self.p(
                "RMSE das diferenças entre o centro de cada câmera ajustada e a posição do GNSS. Mostra a coerência "
                "entre as fotos e o RTK; a exatidão no terreno só é comprovada com pontos de verificação medidos em campo.",
                self.small,
            )
            bins = _variance_bins([i for i in errors if i.in_rmse], p.gnss_accuracy.accuracy.xy_sigma_m * 100)
            if bins:
                self.story.append(
                    Paragraph("Distribuição das diferenças (como a 'Absolute Geolocation Variance' do Pix4D)", self.h2)
                )
                self.story.append(self.table(bins))
        if s.overlap is not None:
            self.story.append(PageBreak())
            self.story.append(Paragraph("4. Sobreposição das fotos", self.h1))
            self.p(
                "Quantas fotos enxergam cada ponto do terreno. Verde (5 ou mais) é o recomendado pelo Pix4D; vermelho "
                "e laranja indicam pouca sobreposição, normalmente só nas bordas do voo, onde o resultado é menos confiável."
            )
            self.story.append(_image(_overlap_plot(s), PAGE_WIDTH * 0.85))

    def calibration(self) -> None:
        p, s = self.project, self.stats
        if not p.ba or not p.ba.camera_calibration:
            return
        self.story.append(PageBreak())
        self.story.append(Paragraph("5. Calibração da câmera", self.h1))
        self.p(
            "Parâmetros internos estimados no ajuste, junto com as posições do RTK (modelo OPENCV do COLMAP: focal, "
            "centro óptico, distorção radial k1, k2 e tangencial p1, p2), comparados com os valores iniciais."
        )
        for camera_id, data in p.ba.camera_calibration.items():
            names = [n.strip() for n in data["params_info"].split(",")]
            initial, final = data["initial_params"], data["params"]
            rows = [["Parâmetro", "Inicial", "Otimizado", "Variação"]]
            for name, a, b in zip(names, initial, final):
                digits = 3 if name in ("fx", "fy", "cx", "cy") else 6
                diff = f"{(b - a) / a:+.3%}".replace(".", ",") if name in ("fx", "fy") and a else _fmt(b - a, digits)
                rows.append([name, _fmt(float(a), digits), _fmt(float(b), digits), diff])
            self.story.append(Paragraph(f"Câmera {camera_id} ({data['model']})", self.h2))
            self.story.append(
                self.table(rows, [0.2 * PAGE_WIDTH, 0.27 * PAGE_WIDTH, 0.27 * PAGE_WIDTH, 0.26 * PAGE_WIDTH])
            )
            if p.ba.intrinsics_refined:
                self.p("Refinados no ajuste: " + ", ".join(p.ba.intrinsics_refined), self.small)
            for note in p.ba.intrinsics_notes:
                self.p("Aviso: " + note, self.small)
            figure = _distortion_and_residuals(names, final, s)
            if figure is not None:
                self.story.append(_image(figure, PAGE_WIDTH))

    def bundle_adjustment(self) -> None:
        p, s = self.project, self.stats
        self.story.append(PageBreak())
        self.story.append(Paragraph("6. Ajuste de feixes (bundle adjustment)", self.h1))
        rows = [
            ("Pontos 3D (tie points)", _fmt(s.num_points3d) if s.num_points3d else NA),
            ("Observações 2D usadas no ajuste", _fmt(s.num_observations) if s.num_observations else NA),
            ("Fotos por ponto 3D (média)", _fmt(s.mean_track_length, 2)),
            ("Pontos-chave por foto (mediana)", _fmt(int(s.median_keypoints)) if s.median_keypoints else NA),
            ("Correspondências por foto (mediana)", _fmt(int(s.median_tie_points)) if s.median_tie_points else NA),
        ]
        if p.sfm:
            rows += [
                ("Erro de reprojeção do alinhamento inicial", _fmt(p.sfm.mean_reprojection_error_px, 3, "pixel")),
                ("Estratégia de correspondência", p.sfm.matching_strategy or NA),
            ]
        if p.ba:
            ba = p.ba
            rows += [
                ("Erro de reprojeção final (após ajuste e limpeza)", _fmt(ba.mean_reprojection_error_px, 3, "pixel")),
                (
                    "Precisão do GNSS informada",
                    f"{_fmt(p.gnss_accuracy.accuracy.xy_sigma_m * 100, 1, 'cm')} (XY), "
                    f"{_fmt(p.gnss_accuracy.accuracy.z_sigma_m * 100, 1, 'cm')} (Z); por foto, o desvio do RTK quando maior",
                ),
                ("Convergência", f"{'sim' if ba.converged else 'NÃO'} ({ba.termination_type.split('.')[-1]})"),
            ]
            if ba.images_excluded_from_rmse:
                rows.append(("Fotos sem RTK FIX (fora do RMSE)", ", ".join(ba.images_excluded_from_rmse)))
        self.story.append(self.kv(rows))
        if p.ba and p.ba.tie_point_cleanup_passes:
            self.story.append(Paragraph("Limpeza de tie points", self.h2))
            table = [["Passe", "Limite de erro", "Observações removidas", "Tie points restantes", "Reprojeção"]]
            table.append(
                [
                    "antes",
                    "-",
                    "-",
                    _fmt(p.ba.tie_points_before_cleanup),
                    _fmt(p.ba.reprojection_error_before_cleanup_px, 3, "px"),
                ]
            )
            for n, step in enumerate(p.ba.tie_point_cleanup_passes, start=1):
                table.append(
                    [
                        str(n),
                        _fmt(step["max_error_px"], 2, "px"),
                        _fmt(int(step["observations_removed"])),
                        _fmt(int(step["tie_points_left"])),
                        _fmt(step["mean_reprojection_error_px"], 3, "px"),
                    ]
                )
            self.story.append(self.table(table))
        if s.track_length_counts:
            self.story.append(Paragraph("Pontos 3D por número de fotos que os enxergam", self.h2))
            self.p("Pontos vistos em só 2 ou 3 fotos são menos precisos; quanto mais à direita, melhor.", self.small)
            self.story.append(_image(_track_plot(s.track_length_counts), PAGE_WIDTH * 0.8))

    def products(self) -> None:
        p, s = self.project, self.stats
        self.story.append(PageBreak())
        self.story.append(Paragraph("7. Nuvem densa, DEM e ortomosaico", self.h1))
        if p.mvs:
            self.story.append(Paragraph("Nuvem densa", self.h2))
            self.story.append(
                self.kv(
                    [
                        ("Pontos", _fmt(p.mvs.num_points)),
                        ("Qualidade", p.mvs.quality),
                        ("Densidade média na área reconstruída", _fmt(s.dense_density_per_m2, 1, "pontos/m²")),
                        ("Método", "COLMAP patch-match stereo (GPU) com consistência geométrica e fusão"),
                    ]
                )
            )
        if p.dem:
            d = p.dem
            covered = d.width_px * d.height_px * (1 - d.nodata_fraction)
            self.story.append(Paragraph("DEM", self.h2))
            self.story.append(
                self.kv(
                    [
                        (
                            "Resolução",
                            _fmt(d.resolution_m * 100, 1, "cm/pixel") + f" ({_resolution_source(d.resolution_source)})",
                        ),
                        ("Tamanho", f"{d.width_px} x {d.height_px} pixels"),
                        (
                            "Altitudes (elipsoidais)",
                            f"{_fmt(d.min_elevation_m, 2)} a {_fmt(d.max_elevation_m, 2, 'm')}",
                        ),
                        ("Células com picos removidos", _fmt(d.num_cells_despiked)),
                        (
                            "Células interpoladas (sem pontos)",
                            f"{_fmt(d.num_cells_interpolated)} ({d.num_cells_interpolated / covered:.1%} da área)".replace(
                                ".", ","
                            )
                            if covered
                            else NA,
                        ),
                        ("Fora da área reconstruída (NoData)", f"{d.nodata_fraction:.1%}".replace(".", ",")),
                        ("Arquivo", Path(d.raster_path).name),
                    ]
                )
            )
        if p.ortho:
            o = p.ortho
            self.story.append(Paragraph("Ortomosaico", self.h2))
            self.story.append(
                self.kv(
                    [
                        (
                            "Resolução (pixel)",
                            _fmt(o.resolution_m * 100, 2, "cm") + f" ({_resolution_source(o.resolution_source)})",
                        ),
                        ("Tamanho", f"{o.width_px} x {o.height_px} pixels"),
                        ("Área com imagem", _fmt(s.covered_area_ha, 2, "ha")),
                        ("Fotos usadas", _fmt(o.num_cameras_used)),
                        ("Formato", "GeoTIFF RGB, JPEG q90 (YCbCr), tiles 512, máscara interna, pirâmides internas"),
                        ("Sistema de coordenadas", f"EPSG:{p.crs.project_epsg}"),
                        ("Arquivo", Path(o.raster_path).name),
                    ]
                )
            )

    def parameters(self, log_path: Path | None) -> None:
        p = self.project
        self.story.append(PageBreak())
        self.story.append(Paragraph("8. Parâmetros e tempos de processamento", self.h1))
        self.story.append(Paragraph(f"Nível {self.level.label}", self.h2))
        for line in self.level.parameters_text():
            self.p("• " + line)
        if p.processing and p.processing.steps_done:
            self.story.append(Paragraph("Tempo por etapa", self.h2))
            rows = [["Etapa", "Tempo", "Concluída em"]]
            total = 0.0
            for key, label in STEP_LABELS.items():
                info = p.processing.steps_done.get(key)
                if info:
                    total += info.get("seconds", 0)
                    rows.append(
                        [label, _duration(info.get("seconds", 0)), info.get("finished_at", "").replace("T", " ")]
                    )
            rows.append(["<b>Total</b>", f"<b>{_duration(total)}</b>", ""])
            self.story.append(self.table(rows, [0.4 * PAGE_WIDTH, 0.25 * PAGE_WIDTH, 0.35 * PAGE_WIDTH]))
        self.story.append(Paragraph("Computador e software", self.h2))
        from htrmapper.pipeline.auto import _system_lines

        for line in _system_lines():
            self.p("• " + line)
        if log_path is not None:
            self.p(f"Log completo do processamento: {Path(log_path).name}", self.small)
        if self.warnings:
            self.story.append(Paragraph("Avisos", self.h2))
            for warning in self.warnings:
                self.p("• " + warning)

    def appendix(self) -> None:
        s = self.stats
        if not s.images:
            return
        self.story.append(PageBreak())
        self.story.append(Paragraph("Anexo: dados por foto", self.h1))
        self.p(
            "dE, dN, dZ: diferença entre a câmera ajustada e a posição do GNSS (leste, norte, vertical).", self.small
        )
        rows = [["Foto", "RTK", "Pontos-chave", "Correspond.", "dE (cm)", "dN (cm)", "dZ (cm)"]]
        for i in s.images:
            rtk = {"50": "FIX", "34": "FLOAT", "16": "SINGLE"}.get(str(i.rtk_flag), str(i.rtk_flag or "-"))
            rows.append(
                [
                    i.name,
                    rtk + ("" if i.in_rmse else "*"),
                    _fmt(i.keypoints) if i.keypoints is not None else "-",
                    _fmt(i.tie_points) if i.tie_points else "não alinhou",
                    _fmt(i.d_east_cm, 1) if i.d_east_cm is not None else "-",
                    _fmt(i.d_north_cm, 1) if i.d_north_cm is not None else "-",
                    _fmt(i.d_up_cm, 1) if i.d_up_cm is not None else "-",
                ]
            )
        widths = [0.3, 0.1, 0.13, 0.13, 0.11, 0.11, 0.12]
        self.story.append(self.table(rows, [w * PAGE_WIDTH for w in widths]))
        self.p("* fora do RMSE (sem RTK FIX).", self.small)


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


def _duration(seconds: float) -> str:
    from htrmapper.pipeline.timing import format_duration

    return format_duration(seconds)


def _resolution_source(text: str) -> str:
    return (
        text.replace("automatic (camera GSD)", "automática, pelo GSD das câmeras")
        .replace("automatic (fallback: DEM resolution)", "automática, pela resolução do DEM")
        .replace("automatic", "automática")
        .replace("user-defined", "definida pelo usuário")
    )


def _variance_bins(images, sigma_cm: float) -> list[list[str]] | None:
    if not images:
        return None
    edges = [0, 1, 2, 3, np.inf]
    rows = [["Faixa do erro", "X (leste)", "Y (norte)", "Z (vertical)"]]
    data = {
        "X": np.abs([i.d_east_cm for i in images]),
        "Y": np.abs([i.d_north_cm for i in images]),
        "Z": np.abs([i.d_up_cm for i in images]),
    }
    for low, high in zip(edges[:-1], edges[1:]):
        label = f"{low:g}σ a {high:g}σ" if np.isfinite(high) else f"acima de {low:g}σ"
        cells = [label]
        for key in ("X", "Y", "Z"):
            values = data[key] / sigma_cm
            frac = np.mean((values >= low) & (values < high))
            cells.append(f"{frac:.1%}".replace(".", ","))
        rows.append(cells)
    rows.append([f"σ = precisão informada do GNSS = {sigma_cm:g} cm".replace(".", ","), "", "", ""])
    return rows


# -- figures ---------------------------------------------------------------------


def _read_preview(path: str, bands, max_px: int = 1400):
    import rasterio

    with rasterio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / max_px)
        shape = (max(1, int(src.height / scale)), max(1, int(src.width / scale)))
        data = src.read(bands, out_shape=(len(bands), *shape) if isinstance(bands, list) else shape, masked=True)
        mask = src.dataset_mask(out_shape=shape) > 0
        bounds = src.bounds
    return data, mask, (bounds.left, bounds.right, bounds.bottom, bounds.top)


def _ortho_preview(path: str) -> io.BytesIO:
    rgb, mask, extent = _read_preview(path, [1, 2, 3])
    rgba = np.dstack([np.moveaxis(np.asarray(rgb.filled(0)), 0, -1), (mask * 255).astype(np.uint8)])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(rgba, extent=extent)
    _map_axes(ax)
    return _png(fig)


def _dem_preview(path: str) -> io.BytesIO:
    z, _mask, extent = _read_preview(path, 1)
    z = np.ma.masked_invalid(z.astype(float))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if z.count():
        cell = (extent[1] - extent[0]) / z.shape[1]
        shade = LightSource(azdeg=315, altdeg=45).hillshade(z.filled(np.nanmean(z)), dx=cell, dy=cell, vert_exag=3)
        image = ax.imshow(z, cmap="terrain", extent=extent)
        ax.imshow(np.ma.array(shade, mask=z.mask), cmap="gray", alpha=0.35, extent=extent)
        fig.colorbar(image, ax=ax, shrink=0.75, label="Altitude elipsoidal (m)")
    _map_axes(ax)
    return _png(fig)


def _map_axes(ax) -> None:
    ax.set_xlabel("Leste (m)")
    ax.set_ylabel("Norte (m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


def _camera_error_plot(project: Project, images) -> io.BytesIO:
    import pycolmap

    centers = {}
    if project.ba and Path(project.ba.reconstruction_path).exists():
        r = pycolmap.Reconstruction(project.ba.reconstruction_path)
        centers = {r.image(i).name: r.image(i).projection_center() for i in r.reg_image_ids()}
    pts = [(centers[i.name], i) for i in images if i.name in centers]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if pts:
        xy = np.array([c[:2] for c, _ in pts])
        de = np.array([i.d_east_cm for _, i in pts])
        dn = np.array([i.d_north_cm for _, i in pts])
        du = np.array([i.d_up_cm for _, i in pts])
        fix = np.array([i.in_rmse for _, i in pts])
        ax.plot(xy[:, 0], xy[:, 1], color="#BBBBBB", lw=0.6, zorder=1)
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0)
        biggest = max(np.hypot(de, dn).max(), 0.1)
        scale = 0.06 * span / biggest
        ax.quiver(
            xy[:, 0],
            xy[:, 1],
            de * scale,
            dn * scale,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            color="#333333",
            zorder=3,
        )
        limit = max(np.abs(du).max(), 0.1)
        sc = ax.scatter(
            xy[fix, 0],
            xy[fix, 1],
            c=du[fix],
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            s=28,
            zorder=2,
            edgecolors="k",
            linewidths=0.3,
        )
        if (~fix).any():
            ax.scatter(
                xy[~fix, 0],
                xy[~fix, 1],
                c=du[~fix],
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                s=45,
                marker="D",
                zorder=2,
                edgecolors="k",
            )
        fig.colorbar(sc, ax=ax, shrink=0.75, label="Diferença vertical dZ (cm)")
        ax.set_title(f"Setas: diferença horizontal (maior = {biggest:.1f} cm)".replace(".", ","), fontsize=9)
    _map_axes(ax)
    return _png(fig)


def _overlap_plot(stats: ReportStats) -> io.BytesIO:
    overlap = np.ma.masked_invalid(stats.overlap)
    cmap = ListedColormap(["#C62828", "#EF6C00", "#F9A825", "#9CCC65", "#2E7D32"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 1000], cmap.N)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(np.clip(overlap, 0, 999), cmap=cmap, norm=norm, extent=stats.overlap_extent)
    bar = fig.colorbar(image, ax=ax, shrink=0.75, ticks=[1, 2, 3, 4, 5])
    bar.ax.set_yticklabels(["1", "2", "3", "4", "5 ou mais"])
    bar.set_label("Fotos que enxergam o ponto")
    _map_axes(ax)
    return _png(fig)


def _distortion_and_residuals(names, params, stats: ReportStats) -> io.BytesIO | None:
    values = dict(zip(names, params))
    has_distortion = "k1" in values and ("fx" in values or "f" in values)
    if not has_distortion and stats.residual_grid is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    if has_distortion and stats.residual_image_size:
        f = values.get("fx", values.get("f"))
        w, h = stats.residual_image_size
        r_max = np.hypot(w, h) / 2 / f
        r = np.linspace(0, r_max, 200)
        k1, k2 = values.get("k1", 0.0), values.get("k2", 0.0)
        axes[0].plot(r * f, f * r * (k1 * r**2 + k2 * r**4), color=theme.PRIMARY)
        axes[0].axhline(0, color="#999999", lw=0.6)
        axes[0].set_xlabel("Distância ao centro da imagem (px)")
        axes[0].set_ylabel("Deslocamento radial (px)")
        axes[0].set_title("Distorção radial da lente", fontsize=9)
    else:
        axes[0].axis("off")
    if stats.residual_grid is not None:
        cx, cy, mdx, mdy, count = stats.residual_grid
        gx, gy = np.meshgrid(cx, cy)
        ok = count > 0
        magnitude = np.hypot(mdx[ok], mdy[ok])
        w, h = stats.residual_image_size
        scale = 0.08 * w / max(magnitude.max(), 1e-6)
        axes[1].quiver(
            gx[ok],
            gy[ok],
            mdx[ok] * scale,
            -mdy[ok] * scale,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
            color=theme.PRIMARY_DARK,
        )
        axes[1].set_xlim(0, w)
        axes[1].set_ylim(h, 0)
        axes[1].set_aspect("equal")
        axes[1].set_title(f"Resíduos médios na imagem (maior = {magnitude.max():.3f} px)".replace(".", ","), fontsize=9)
        axes[1].tick_params(labelsize=7)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    return _png(fig)


def _track_plot(counts: dict[int, int]) -> io.BytesIO:
    keys = sorted(counts)
    labels = [str(k) if k < 10 else "10+" for k in keys if k < 10] + (["10+"] if any(k >= 10 for k in keys) else [])
    values = [counts[k] for k in keys if k < 10] + (
        [sum(v for k, v in counts.items() if k >= 10)] if any(k >= 10 for k in keys) else []
    )
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(labels, values, color=theme.PRIMARY)
    ax.set_xlabel("Número de fotos")
    ax.set_ylabel("Pontos 3D")
    ax.tick_params(labelsize=8)
    return _png(fig)


def write_pdf_report(
    project: Project, path: Path, level, log_path: Path | None = None, warnings: list[str] | None = None
) -> Path:
    stats = compute_report_stats(project, level.ortho.resolution_scale)
    builder = _Builder(project, level, stats, warnings or [])
    builder.cover()
    builder.story.append(PageBreak())
    builder.quality_check()
    builder.previews()
    builder.cameras()
    builder.calibration()
    builder.bundle_adjustment()
    builder.products()
    builder.parameters(log_path)
    builder.appendix()

    font = builder.font
    generated = datetime.now().strftime("%d/%m/%Y %H:%M")

    def decorate(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(font, 7.5)
        canvas.setFillColor(colors.HexColor(theme.TEXT_MUTED))
        canvas.drawString(1.8 * cm, A4[1] - 1.1 * cm, f"HTRMapper — Relatório de Processamento — {project.name}")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Página {doc.page}")
        canvas.drawString(1.8 * cm, 1.0 * cm, f"Gerado em {generated}")
        canvas.restoreState()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=f"Relatório de Processamento — {project.name}",
        author="HTRMapper",
    )
    doc.build(builder.story, onFirstPage=decorate, onLaterPages=decorate)
    return path


__all__ = ["KeepTogether", "write_pdf_report"]
