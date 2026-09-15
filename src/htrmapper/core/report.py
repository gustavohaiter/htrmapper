"""Processing (quality) report generation.

Mirrors the section structure of a standard photogrammetry processing
report (Survey Data / Camera Calibration / Camera Locations / DEM /
Orthomosaic / Processing Parameters / System) -- see ARCHITECTURE.md
section 11 for the field-by-field mapping to our own pipeline phases.

The guiding rule (project brief, rule 3: never claim centimetric
precision without validation) is enforced structurally here: every metric
that a later phase (SfM, bundle adjustment, dense cloud, DEM, orthomosaic)
must produce is represented as a `Metric` that is either a real computed
value, or `None` with an explicit `pending_phase` explaining which phase
will compute it. The HTML renderer never prints a bare "0" or blank cell
for those -- it prints the pending explanation.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean

from htrmapper.core import theme
from htrmapper.core.project import Project
from htrmapper.core.system_info import SystemInfo, detect_system_info
from htrmapper.geo.coverage import Point2D, convex_hull_area_km2
from htrmapper.geo.crs import CoordinateReferenceSystem, GeodeticPoint, GeodeticTransformer, wgs84
from htrmapper.geo.gsd import estimate_gsd


@dataclass(frozen=True)
class Metric:
    """A report value that may not be computable yet.

    `value is None` together with `pending_phase` means: this metric is a
    real photogrammetric quantity that HTRMapper has not computed yet
    because the phase that produces it hasn't run (or isn't implemented
    yet) -- it is never rendered as zero or omitted silently.
    """

    value: float | int | str | None = None
    unit: str = ""
    pending_phase: int | None = None
    pending_note: str | None = None

    @property
    def is_available(self) -> bool:
        return self.value is not None

    def render_text(self) -> str:
        if self.value is None:
            reason = self.pending_note or "Não disponível"
            if self.pending_phase is not None:
                return f"{reason} — calculado na Fase {self.pending_phase}"
            return reason
        if isinstance(self.value, float):
            text = f"{self.value:,.3f}".rstrip("0").rstrip(".")
        else:
            text = str(self.value)
        return f"{text} {self.unit}".strip()


def pending(phase: int, note: str | None = None) -> Metric:
    return Metric(value=None, pending_phase=phase, pending_note=note)


@dataclass
class CameraModelSummary:
    model: str
    count: int
    resolution: str
    focal_length_mm: Metric
    pixel_size_um: Metric
    precalibrated: str = "Não"


@dataclass
class SurveyData:
    num_images: int = 0
    camera_stations: int = 0
    aligned_cameras: Metric = field(default_factory=lambda: pending(2, "Alinhamento (SfM) ainda não executado"))
    flying_altitude_m: Metric = field(default_factory=lambda: pending(1, "Nenhuma imagem com altitude AGL"))
    ground_resolution_cm: Metric = field(default_factory=lambda: pending(1, "Dados insuficientes (focal/altitude)"))
    coverage_area_km2: Metric = field(default_factory=lambda: pending(1, "Menos de 3 posições válidas de câmera"))
    tie_points: Metric = field(default_factory=lambda: pending(2))
    projections: Metric = field(default_factory=lambda: pending(2))
    # COLMAP's own (unweighted) bundle adjustment error from Phase 2's
    # initial SfM -- distinct from `reprojection_error_px`, which stays
    # pending until the GNSS-weighted bundle adjustment (Phase 3) produces
    # the final, reported value. Never conflate the two.
    initial_reprojection_error_px: Metric = field(default_factory=lambda: pending(2))
    reprojection_error_px: Metric = field(default_factory=lambda: pending(3))
    matching_strategy: str = ""
    georeferencing_note: str = ""
    unregistered_images: list[str] = field(default_factory=list)
    cameras: list[CameraModelSummary] = field(default_factory=list)
    camera_map_png: bytes | None = None


@dataclass
class CameraCalibrationEntry:
    camera_id: str
    model: str
    params_info: str
    params: list[float]


@dataclass
class CameraCalibration:
    note: str = "Calibração de câmera (bundle adjustment) ainda não implementada."
    pending_phase: int | None = 3
    entries: list[CameraCalibrationEntry] = field(default_factory=list)
    correlation_matrix_note: str = "Matriz de correlação: não disponível nesta fase."


@dataclass
class CameraLocations:
    x_error_cm: Metric = field(default_factory=lambda: pending(3))
    y_error_cm: Metric = field(default_factory=lambda: pending(3))
    z_error_cm: Metric = field(default_factory=lambda: pending(3))
    xy_error_cm: Metric = field(default_factory=lambda: pending(3))
    total_error_cm: Metric = field(default_factory=lambda: pending(3))


@dataclass
class DigitalElevationModel:
    resolution_cm_per_px: Metric = field(default_factory=lambda: pending(5))
    point_density_per_m2: Metric = field(default_factory=lambda: pending(4))


@dataclass
class Orthomosaic:
    size: Metric = field(default_factory=lambda: pending(6))
    coordinate_system: Metric = field(default_factory=lambda: pending(6))


@dataclass
class GnssAccuracySummary:
    xy_sigma_m: float = 0.0
    z_sigma_m: float = 0.0
    images_with_per_image_rtk_std_dev: int = 0
    total_images: int = 0


@dataclass
class ProcessingParameters:
    cameras: int = 0
    aligned_cameras: Metric = field(default_factory=lambda: pending(2))
    project_crs: str = ""
    source_crs: str = ""
    rotation_angles: str = "Yaw, Pitch, Roll"
    tie_points_section: Metric = field(default_factory=lambda: pending(2))
    depth_maps_section: Metric = field(default_factory=lambda: pending(4))
    point_cloud_section: Metric = field(default_factory=lambda: pending(4))
    dem_section: Metric = field(default_factory=lambda: pending(5))
    orthomosaic_section: Metric = field(default_factory=lambda: pending(6))
    system: SystemInfo | None = None


@dataclass
class ProcessingReport:
    project_name: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    survey_data: SurveyData = field(default_factory=SurveyData)
    camera_calibration: CameraCalibration = field(default_factory=CameraCalibration)
    camera_locations: CameraLocations = field(default_factory=CameraLocations)
    dem: DigitalElevationModel = field(default_factory=DigitalElevationModel)
    orthomosaic: Orthomosaic = field(default_factory=Orthomosaic)
    gnss_accuracy: GnssAccuracySummary = field(default_factory=GnssAccuracySummary)
    processing_parameters: ProcessingParameters = field(default_factory=ProcessingParameters)


def _render_camera_position_map_png(
    points_xy: list[tuple[float, float]], project_epsg: int
) -> bytes | None:
    if not points_xy:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=110, facecolor=theme.BACKGROUND)
    ax.set_facecolor(theme.BACKGROUND)
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    ax.scatter(xs, ys, c=theme.PRIMARY, s=18, edgecolors=theme.PRIMARY_DARK, linewidths=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"Este (m) — EPSG:{project_epsg}", color=theme.PRIMARY_DARK)
    ax.set_ylabel("Norte (m)", color=theme.PRIMARY_DARK)
    ax.set_title(f"Posições das câmeras ({len(points_xy)})", color=theme.PRIMARY_DARK)
    ax.tick_params(colors=theme.TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_color(theme.BORDER)
    ax.grid(True, linewidth=0.3, color=theme.BORDER)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def build_report_from_project(project: Project) -> ProcessingReport:
    """Populate a ProcessingReport with everything computable from imported
    images alone (Phase 1). Every field that requires SfM/BA/MVS/DEM/
    orthomosaic output stays a `pending` Metric until those phases run.
    """
    images = project.images
    valid_images = [img for img in images if img.position_valid]

    project_crs = CoordinateReferenceSystem(project.crs.project_epsg)
    transformer = GeodeticTransformer(wgs84(), project_crs)
    projected_points: list[tuple[float, float]] = []
    for img in valid_images:
        p = transformer.forward(GeodeticPoint(lon=img.longitude, lat=img.latitude, alt=img.altitude))
        projected_points.append((p.x, p.y))

    survey = SurveyData(num_images=len(images), camera_stations=len(images))

    relative_altitudes = [img.relative_altitude_m for img in images if img.relative_altitude_m is not None]
    if relative_altitudes:
        survey.flying_altitude_m = Metric(value=round(mean(relative_altitudes), 1), unit="m")

    gsd_values = []
    for img in images:
        if (
            img.focal_length_mm
            and img.focal_length_35mm_equiv
            and img.pixel_width
            and img.relative_altitude_m
        ):
            try:
                gsd = estimate_gsd(
                    flying_height_m=img.relative_altitude_m,
                    focal_length_mm=img.focal_length_mm,
                    focal_length_35mm_equiv_mm=img.focal_length_35mm_equiv,
                    image_width_px=img.pixel_width,
                )
                gsd_values.append(gsd.gsd_cm_per_px)
            except ValueError:
                continue
    if gsd_values:
        survey.ground_resolution_cm = Metric(
            value=round(mean(gsd_values), 2), unit="cm/pix (estimativa pré-alinhamento)"
        )

    if len(projected_points) >= 3:
        hull_points = [Point2D(x, y) for x, y in projected_points]
        area = convex_hull_area_km2(hull_points)
        survey.coverage_area_km2 = Metric(value=round(area, 4), unit="km² (hull convexo, pré-ortomosaico)")

    by_model: dict[str, list] = {}
    for img in images:
        key = img.camera_model or "Desconhecido"
        by_model.setdefault(key, []).append(img)
    for model, imgs in sorted(by_model.items()):
        widths = {i.pixel_width for i in imgs if i.pixel_width}
        heights = {i.pixel_height for i in imgs if i.pixel_height}
        resolution = (
            f"{imgs[0].pixel_width} x {imgs[0].pixel_height}"
            if imgs[0].pixel_width and imgs[0].pixel_height
            else "-"
        )
        focal_lengths = [i.focal_length_mm for i in imgs if i.focal_length_mm]
        focal_metric = Metric(value=round(mean(focal_lengths), 2), unit="mm") if focal_lengths else pending(
            1, "Focal length ausente no EXIF"
        )

        pixel_size_metric = pending(1, "Dados insuficientes para estimar sensor (falta focal 35mm equiv.)")
        candidates = [
            i
            for i in imgs
            if i.focal_length_mm and i.focal_length_35mm_equiv and i.pixel_width
        ]
        if candidates:
            estimates = [
                estimate_gsd(1.0, i.focal_length_mm, i.focal_length_35mm_equiv, i.pixel_width).pixel_pitch_mm
                * 1000.0
                for i in candidates
            ]
            pixel_size_metric = Metric(value=round(mean(estimates), 2), unit="um (estimado via crop factor)")

        survey.cameras.append(
            CameraModelSummary(
                model=model,
                count=len(imgs),
                resolution=resolution,
                focal_length_mm=focal_metric,
                pixel_size_um=pixel_size_metric,
            )
        )

    survey.camera_map_png = _render_camera_position_map_png(projected_points, project.crs.project_epsg)

    gnss_summary = GnssAccuracySummary(
        xy_sigma_m=project.gnss_accuracy.accuracy.xy_sigma_m,
        z_sigma_m=project.gnss_accuracy.accuracy.z_sigma_m,
        images_with_per_image_rtk_std_dev=sum(1 for i in images if i.rtk_std_lon_m is not None),
        total_images=len(images),
    )

    params = ProcessingParameters(
        cameras=len(images),
        project_crs=f"EPSG:{project.crs.project_epsg}",
        source_crs=f"EPSG:{project.crs.source_epsg}",
        system=detect_system_info(),
    )

    if project.sfm is not None:
        sfm = project.sfm
        survey.aligned_cameras = Metric(value=sfm.num_registered, unit=f"/ {sfm.num_images_input}")
        survey.tie_points = Metric(value=sfm.num_points3d)
        survey.projections = Metric(value=sfm.num_observations)
        if sfm.mean_reprojection_error_px is not None:
            survey.initial_reprojection_error_px = Metric(
                value=round(sfm.mean_reprojection_error_px, 4),
                unit="pix (SfM inicial, COLMAP, sem peso GNSS)",
            )
        survey.matching_strategy = sfm.matching_strategy
        survey.georeferencing_note = sfm.georeferencing_note
        survey.unregistered_images = sfm.unregistered_image_names
        params.aligned_cameras = Metric(value=sfm.num_registered, unit=f"/ {sfm.num_images_input}")
        params.tie_points_section = Metric(
            value=(
                f"{sfm.num_points3d} tie points, {sfm.num_observations} observations, "
                f"matching: {sfm.matching_strategy}"
            )
        )

    camera_locations = CameraLocations()
    camera_calibration = CameraCalibration()

    if project.ba is not None:
        ba = project.ba
        camera_locations = CameraLocations(
            x_error_cm=Metric(value=round(ba.rmse_x_cm, 4), unit="cm") if ba.rmse_x_cm is not None else pending(3),
            y_error_cm=Metric(value=round(ba.rmse_y_cm, 4), unit="cm") if ba.rmse_y_cm is not None else pending(3),
            z_error_cm=Metric(value=round(ba.rmse_z_cm, 4), unit="cm") if ba.rmse_z_cm is not None else pending(3),
            xy_error_cm=Metric(value=round(ba.rmse_xy_cm, 4), unit="cm") if ba.rmse_xy_cm is not None else pending(3),
            total_error_cm=(
                Metric(value=round(ba.rmse_total_cm, 4), unit="cm") if ba.rmse_total_cm is not None else pending(3)
            ),
        )
        if ba.mean_reprojection_error_px is not None:
            survey.reprojection_error_px = Metric(
                value=round(ba.mean_reprojection_error_px, 4),
                unit="pix (final, bundle adjustment ponderado por GNSS)",
            )
        entries = [
            CameraCalibrationEntry(
                camera_id=camera_id,
                model=data["model"],
                params_info=data["params_info"],
                params=[round(p, 6) for p in data["params"]],
            )
            for camera_id, data in ba.camera_calibration.items()
        ]
        camera_calibration = CameraCalibration(
            note=(
                f"Convergiu: {ba.converged} ({ba.termination_type}). "
                f"{ba.num_images_with_gnss_prior} câmera(s) com observação GNSS ponderada."
            ),
            pending_phase=None,
            entries=entries,
        )

    dem = DigitalElevationModel()
    if project.mvs is not None:
        mvs = project.mvs
        params.point_cloud_section = Metric(
            value=f"{mvs.num_points} pontos, qualidade '{mvs.quality}' (COLMAP patch-match stereo + fusion)"
        )
        params.depth_maps_section = Metric(value=f"Qualidade '{mvs.quality}' (ver Point Cloud)")
        if survey.coverage_area_km2.is_available and survey.coverage_area_km2.value:
            density = mvs.num_points / (survey.coverage_area_km2.value * 1_000_000.0)
            dem.point_density_per_m2 = Metric(
                value=round(density, 2),
                unit="pontos/m² (estimativa: área do hull convexo de câmeras, pré-ortomosaico)",
            )

    if project.dem is not None:
        dem_summary = project.dem
        dem.resolution_cm_per_px = Metric(
            value=round(dem_summary.resolution_m * 100, 2),
            unit=f"cm/pix ({dem_summary.resolution_source})",
        )
        dem.point_density_per_m2 = Metric(
            value=round(dem_summary.point_density_per_m2, 2), unit="pontos/m² (nuvem densa real)"
        )
        params.dem_section = Metric(
            value=(
                f"{dem_summary.width_px}x{dem_summary.height_px}px, resolução "
                f"{dem_summary.resolution_m:.3f} m ({dem_summary.resolution_source}), "
                f"elevação [{dem_summary.min_elevation_m:.2f}, {dem_summary.max_elevation_m:.2f}] m, "
                f"{dem_summary.num_points_filtered_as_outliers} ponto(s) filtrado(s) como outlier"
            )
        )

    orthomosaic = Orthomosaic()
    if project.ortho is not None:
        ortho_summary = project.ortho
        orthomosaic.size = Metric(value=f"{ortho_summary.width_px} x {ortho_summary.height_px} px")
        orthomosaic.coordinate_system = Metric(value=f"EPSG:{project.crs.project_epsg}")
        params.orthomosaic_section = Metric(
            value=(
                f"{ortho_summary.width_px}x{ortho_summary.height_px}px, resolução "
                f"{ortho_summary.resolution_m:.3f} m/px, {ortho_summary.num_cameras_used} câmera(s) usada(s), "
                f"{ortho_summary.num_nodata_pixels} pixel(s) sem cobertura (NoData)"
            )
        )

    return ProcessingReport(
        project_name=project.name,
        survey_data=survey,
        gnss_accuracy=gnss_summary,
        processing_parameters=params,
        camera_locations=camera_locations,
        camera_calibration=camera_calibration,
        dem=dem,
        orthomosaic=orthomosaic,
    )


def _esc(value) -> str:
    return html.escape(str(value))


def render_html(report: ProcessingReport) -> str:
    survey = report.survey_data
    gnss = report.gnss_accuracy
    params = report.processing_parameters

    camera_rows = "\n".join(
        f"<tr><td>{_esc(c.model)}</td><td>{c.count}</td><td>{_esc(c.resolution)}</td>"
        f"<td>{_esc(c.focal_length_mm.render_text())}</td><td>{_esc(c.pixel_size_um.render_text())}</td>"
        f"<td>{_esc(c.precalibrated)}</td></tr>"
        for c in survey.cameras
    )

    calibration_rows = "\n".join(
        f"<tr><td>{_esc(e.camera_id)}</td><td>{_esc(e.model)}</td><td>{_esc(e.params_info)}</td>"
        f"<td>{_esc(', '.join(f'{p:.6g}' for p in e.params))}</td></tr>"
        for e in report.camera_calibration.entries
    )

    map_html = ""
    if survey.camera_map_png:
        b64 = base64.b64encode(survey.camera_map_png).decode("ascii")
        map_html = f'<img alt="Mapa de posições de câmera" src="data:image/png;base64,{b64}" style="max-width:500px;">'

    gpu_rows = (
        "".join(
            f"<li>{_esc(g.name)} — {g.vram_total_mb:.0f} MB VRAM (driver {g.driver_version})</li>"
            for g in params.system.gpus
        )
        if params.system and params.system.gpus
        else "<li>Nenhuma GPU NVIDIA detectada — processamento em CPU.</li>"
    )

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Relatório de Processamento — {_esc(report.project_name)}</title>
<style>
body {{
    font-family: "Segoe UI", Arial, sans-serif;
    margin: 0;
    padding: 0;
    color: {theme.TEXT};
    background-color: {theme.BACKGROUND};
}}
.banner {{
    background-color: {theme.PRIMARY_DARK};
    color: {theme.BACKGROUND};
    padding: 1.8rem 3rem;
}}
.banner h1 {{
    margin: 0;
    color: {theme.BACKGROUND};
    border-bottom: none;
}}
.banner p {{
    margin: 0.3rem 0 0 0;
    color: {theme.SURFACE};
    opacity: 0.85;
}}
.content {{
    padding: 2rem 3rem;
}}
h2 {{
    margin-top: 2.2rem;
    color: {theme.PRIMARY_DARK};
    border-left: 5px solid {theme.PRIMARY};
    padding-left: 0.6rem;
}}
h3 {{ color: {theme.PRIMARY}; margin-top: 1.2rem; }}
table {{ border-collapse: collapse; margin: 0.5rem 0 1rem 0; min-width: 320px; }}
th {{
    background-color: {theme.PRIMARY_DARK};
    color: {theme.BACKGROUND};
    border: 1px solid {theme.PRIMARY_DARK};
    padding: 6px 12px;
    text-align: left;
}}
td {{
    border: 1px solid {theme.BORDER};
    padding: 6px 12px;
    text-align: left;
}}
tr:nth-child(even) td {{ background-color: {theme.SURFACE}; }}
.pending {{ color: {theme.PENDING}; font-style: italic; }}
.section {{
    margin-bottom: 2rem;
    padding: 1rem 1.5rem;
    background-color: {theme.SURFACE};
    border: 1px solid {theme.BORDER};
    border-radius: 6px;
}}
</style>
</head>
<body>
<div class="banner">
<h1>Relatório de Processamento — {_esc(report.project_name)}</h1>
<p>Gerado em: {_esc(report.generated_at)}</p>
</div>
<div class="content">

<div class="section">
<h2>Survey Data</h2>
{map_html}
<table>
<tr><td>Número de imagens</td><td>{survey.num_images}</td></tr>
<tr><td>Estações de câmera</td><td>{survey.camera_stations}</td></tr>
<tr><td>Câmeras alinhadas</td><td>{_esc(survey.aligned_cameras.render_text())}</td></tr>
<tr><td>Altitude de voo (AGL, média)</td><td>{_esc(survey.flying_altitude_m.render_text())}</td></tr>
<tr><td>Resolução em solo (GSD)</td><td>{_esc(survey.ground_resolution_cm.render_text())}</td></tr>
<tr><td>Área de cobertura</td><td>{_esc(survey.coverage_area_km2.render_text())}</td></tr>
<tr><td>Tie points</td><td>{_esc(survey.tie_points.render_text())}</td></tr>
<tr><td>Projections</td><td>{_esc(survey.projections.render_text())}</td></tr>
<tr><td>Reprojection error (inicial, SfM/COLMAP)</td><td>{_esc(survey.initial_reprojection_error_px.render_text())}</td></tr>
<tr><td>Reprojection error (final, ponderado por GNSS)</td><td>{_esc(survey.reprojection_error_px.render_text())}</td></tr>
</table>
<table>
<tr><th>Câmera</th><th>Qtd. imagens</th><th>Resolução</th><th>Focal Length</th><th>Pixel Size</th><th>Precalibrada</th></tr>
{camera_rows}
</table>
{f'<p>Estratégia de matching: {_esc(survey.matching_strategy)}</p>' if survey.matching_strategy else ''}
{f'<p>Georreferenciamento: {_esc(survey.georeferencing_note)}</p>' if survey.georeferencing_note else ''}
{f'<p class="pending">Imagens não alinhadas: {_esc(", ".join(survey.unregistered_images))}</p>' if survey.unregistered_images else ''}
</div>

<div class="section">
<h2>Configuração de Precisão GNSS/PPK</h2>
<table>
<tr><td>Sigma XY configurado</td><td>{gnss.xy_sigma_m:.3f} m</td></tr>
<tr><td>Sigma Z configurado</td><td>{gnss.z_sigma_m:.3f} m</td></tr>
<tr><td>Imagens com desvio-padrão RTK próprio (XMP)</td><td>{gnss.images_with_per_image_rtk_std_dev} de {gnss.total_images}</td></tr>
</table>
<p class="pending">Estes valores serão usados como peso (1/sigma²) no bundle adjustment ponderado por GNSS (Fase 3). Nenhum ajuste foi executado ainda.</p>
</div>

<div class="section">
<h2>Camera Calibration</h2>
{f'<p class="pending">{_esc(report.camera_calibration.note)} (Fase {report.camera_calibration.pending_phase})</p>' if report.camera_calibration.pending_phase is not None else f'<p>{_esc(report.camera_calibration.note)}</p>'}
{f'''<table>
<tr><th>Câmera</th><th>Modelo</th><th>Parâmetros</th><th>Valores</th></tr>
{calibration_rows}
</table>
<p class="pending">{_esc(report.camera_calibration.correlation_matrix_note)}</p>''' if report.camera_calibration.entries else ''}
</div>

<div class="section">
<h2>Camera Locations</h2>
<table>
<tr><th>X error (cm)</th><th>Y error (cm)</th><th>Z error (cm)</th><th>XY error (cm)</th><th>Total error (cm)</th></tr>
<tr>
<td>{_esc(report.camera_locations.x_error_cm.render_text())}</td>
<td>{_esc(report.camera_locations.y_error_cm.render_text())}</td>
<td>{_esc(report.camera_locations.z_error_cm.render_text())}</td>
<td>{_esc(report.camera_locations.xy_error_cm.render_text())}</td>
<td>{_esc(report.camera_locations.total_error_cm.render_text())}</td>
</tr>
</table>
</div>

<div class="section">
<h2>Digital Elevation Model</h2>
<table>
<tr><td>Resolução</td><td>{_esc(report.dem.resolution_cm_per_px.render_text())}</td></tr>
<tr><td>Densidade de pontos</td><td>{_esc(report.dem.point_density_per_m2.render_text())}</td></tr>
</table>
</div>

<div class="section">
<h2>Orthomosaic</h2>
<table>
<tr><td>Tamanho</td><td>{_esc(report.orthomosaic.size.render_text())}</td></tr>
<tr><td>Sistema de coordenadas</td><td>{_esc(report.orthomosaic.coordinate_system.render_text())}</td></tr>
</table>
</div>

<div class="section">
<h2>Processing Parameters</h2>
<h3>General</h3>
<table>
<tr><td>Cameras</td><td>{params.cameras}</td></tr>
<tr><td>Aligned cameras</td><td>{_esc(params.aligned_cameras.render_text())}</td></tr>
<tr><td>Project CRS</td><td>{_esc(params.project_crs)}</td></tr>
<tr><td>Source CRS</td><td>{_esc(params.source_crs)}</td></tr>
<tr><td>Rotation angles</td><td>{_esc(params.rotation_angles)}</td></tr>
</table>
<h3>Tie Points</h3>
<p class="pending">{_esc(params.tie_points_section.render_text())}</p>
<h3>Depth Maps</h3>
<p class="pending">{_esc(params.depth_maps_section.render_text())}</p>
<h3>Point Cloud</h3>
<p class="pending">{_esc(params.point_cloud_section.render_text())}</p>
<h3>DEM</h3>
<p class="pending">{_esc(params.dem_section.render_text())}</p>
<h3>Orthomosaic</h3>
<p class="pending">{_esc(params.orthomosaic_section.render_text())}</p>
<h3>System</h3>
<table>
<tr><td>OS</td><td>{_esc(params.system.os_name if params.system else '-')}</td></tr>
<tr><td>CPU</td><td>{_esc(params.system.cpu_name if params.system else '-')} ({params.system.cpu_logical_cores if params.system else '-'} threads)</td></tr>
<tr><td>RAM</td><td>{params.system.ram_total_gb:.1f} GB</td></tr>
</table>
<p>GPU(s):</p>
<ul>{gpu_rows}</ul>
</div>

</div>
</body>
</html>
"""
