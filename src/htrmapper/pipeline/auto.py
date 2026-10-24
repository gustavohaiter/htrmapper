"""Automatic processing: photos folder in, orthomosaic + DEM + PDF report out.

One call runs every step in order -- import, alignment, GNSS adjustment,
dense cloud, DEM, orthomosaic, report -- with the parameters of the chosen
level (`pipeline.levels`). Nothing is asked along the way.

Output folder layout::

    <saída>/projeto.json                  project state (resume, advanced mode)
    <saída>/processamento.log             everything that happened, in order
    <saída>/<nome>_ortomosaico.tif
    <saída>/<nome>_dem.tif                ellipsoidal heights, as the ortho uses
    <saída>/<nome>_relatorio.pdf
    <saída>/ERRO_<data>.txt               only if a step failed
    <saída>/trabalho/                     intermediate data (alignment, dense cloud, COLMAP logs)

**Resume**: after every step the project file records it as done. Running
again on the same output folder, with the same photos and level, skips the
finished steps whose outputs are still on disk -- a failure at the
orthomosaic does not redo half an hour of dense cloud.

**Errors** never pass silently: the step is marked as failed, the full
context (traceback, versions, hardware, the end of the log) goes to
`ERRO_<data>.txt` in the output folder, and `AutoProcessingError` carries a
plain-language message plus that file's path for the GUI to show.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import platform
import re
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean

from htrmapper.core.project import (
    DemSummary,
    MvsSummary,
    OrthoSummary,
    ProcessingSummary,
    Project,
    ProjectCrsConfig,
)
from htrmapper.geo.crs import CoordinateReferenceSystem, sirgas2000_utm_epsg
from htrmapper.io.image_import import find_images, import_folder
from htrmapper.pipeline import timing
from htrmapper.pipeline.levels import DEFAULT_LEVEL, LEVELS, ProcessingLevel

STEPS: list[tuple[str, str]] = [
    ("importar", "Importar fotos"),
    ("alinhar", "Alinhamento das fotos"),
    ("ajustar", "Ajuste com o GNSS (RTK)"),
    ("densa", "Nuvem densa"),
    ("dem", "DEM (modelo de superfície)"),
    ("ortho", "Ortomosaico"),
    ("relatorio", "Relatório PDF"),
]
STEP_LABELS = dict(STEPS)
STEP_KEYS = [key for key, _ in STEPS]

MIN_PHOTOS = 3
# Below this share of aligned photos the mosaic would have large holes
# (Pix4D's quality check turns red under 60%).
MIN_ALIGNED_FRACTION = 0.60
# Disk needed per photo (working data, mostly the dense cloud's depth maps).
DISK_MB_PER_PHOTO = {"baixo": 15, "medio": 80, "alto": 300}
DISK_MARGIN_MB = 1024

LOG_FILE_NAME = "processamento.log"
PROJECT_FILE_NAME = "projeto.json"
WORK_DIR_NAME = "trabalho"

logger = logging.getLogger("htrmapper.auto")


class AutoProcessingError(RuntimeError):
    """A step failed. `message` is meant for the user; `error_file` holds
    the full technical context."""

    def __init__(self, message: str, step: str | None = None, error_file: Path | None = None):
        super().__init__(message)
        self.step = step
        self.error_file = error_file


@dataclass
class AutoRequest:
    images_folder: Path
    output_folder: Path
    level: str = DEFAULT_LEVEL
    delete_workdir_on_success: bool = False
    work_folder: Path | None = None  # None = default_work_folder(output_folder)

    def __post_init__(self) -> None:
        self.images_folder = Path(self.images_folder)
        self.output_folder = Path(self.output_folder)
        if self.work_folder is not None:
            self.work_folder = Path(self.work_folder)
        if self.level not in LEVELS:
            raise ValueError(f"nível desconhecido: {self.level!r} (use {', '.join(LEVELS)})")


@dataclass
class ExistingRun:
    """What an earlier automatic run left in the output folder."""

    level: str
    steps_done: list[str]
    same_photos: bool
    finished: bool

    @property
    def resumable_with(self) -> Callable[[str], bool]:
        return lambda level: self.same_photos and level == self.level


@dataclass
class AutoResult:
    project_path: Path
    products: dict[str, str]
    step_seconds: dict[str, float]
    skipped_steps: list[str]
    total_seconds: float
    warnings: list[str] = field(default_factory=list)


class Callbacks:
    """Hooks for a GUI (all optional, all called from the worker thread)."""

    def log(self, message: str) -> None: ...

    def step(self, key: str, status: str, detail: str = "") -> None:
        """status: 'rodando' | 'concluido' | 'pulado' | 'erro' | 'cancelado'."""

    def phase(self, key: str, text: str) -> None: ...

    def progress(self, key: str, done: int, total: int) -> None: ...


def software_version() -> str:
    try:
        from importlib.metadata import version

        return version("htrmapper")
    except Exception:  # noqa: BLE001 -- running from a plain checkout
        return "dev"


def default_work_folder(output_folder: Path, name: str) -> Path:
    """`<saída>/trabalho`, except in WSL with the output on a Windows drive
    (/mnt/c/...): the dense cloud writes gigabytes of depth maps, and the
    Windows drive is several times slower from inside WSL than WSL's own
    disk, so the working data goes to ~/.local/share/htrmapper/trabalho
    instead. The products still go to the chosen output folder."""
    from htrmapper.core.platform_paths import is_running_under_wsl

    output_folder = Path(output_folder).resolve()
    if is_running_under_wsl() and output_folder.parts[:2] == ("/", "mnt"):
        digest = hashlib.sha1(str(output_folder).encode("utf-8")).hexdigest()[:8]
        return Path.home() / ".local" / "share" / "htrmapper" / WORK_DIR_NAME / f"{name}_{digest}"
    return output_folder / WORK_DIR_NAME


def project_name_from_folder(folder: Path) -> str:
    name = re.sub(r"[^\w\-]+", "_", Path(folder).resolve().name, flags=re.UNICODE).strip("_")
    return name or "projeto"


def count_photos(folder: Path) -> int:
    try:
        return len(find_images(Path(folder)))
    except OSError:
        return 0


def gpu_available() -> bool:
    import pycolmap

    return bool(pycolmap.has_cuda)


class _CallbackLogHandler(logging.Handler):
    def __init__(self, callbacks: Callbacks):
        super().__init__(level=logging.INFO)
        self.callbacks = callbacks

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callbacks.log(self.format(record))
        except Exception:  # noqa: BLE001 -- a GUI hiccup must not kill the run
            pass


class AutoProcessor:
    def __init__(
        self,
        request: AutoRequest,
        callbacks: Callbacks | None = None,
        cancellation_token=None,
        require_gpu: bool = True,
        timing_history_path: Path | None = None,
    ):
        self.request = request
        self.level: ProcessingLevel = LEVELS[request.level]
        self.callbacks = callbacks or Callbacks()
        self.cancellation_token = cancellation_token
        self.require_gpu = require_gpu
        self.timing_history_path = timing_history_path
        self.name = project_name_from_folder(request.images_folder)
        self.warnings: list[str] = []
        self.project: Project | None = None

    # -- paths -------------------------------------------------------------

    @property
    def out(self) -> Path:
        return self.request.output_folder

    @property
    def project_path(self) -> Path:
        return self.out / PROJECT_FILE_NAME

    @property
    def log_path(self) -> Path:
        return self.out / LOG_FILE_NAME

    @property
    def work(self) -> Path:
        return self.request.work_folder or default_work_folder(self.out, self.name)

    def product_path(self, key: str) -> Path:
        suffix = {"ortho": "ortomosaico.tif", "dem": "dem.tif", "relatorio": "relatorio.pdf"}[key]
        return self.out / f"{self.name}_{suffix}"

    # -- before running ------------------------------------------------------

    def inspect_existing(self) -> ExistingRun | None:
        if not self.project_path.is_file():
            return None
        try:
            project = Project.load(self.project_path)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if project.processing is None:
            return None
        photos = (
            sorted(p.name for p in find_images(self.request.images_folder))
            if self.request.images_folder.is_dir()
            else []
        )
        done = [k for k in STEP_KEYS if k in project.processing.steps_done]
        return ExistingRun(
            level=project.processing.level,
            steps_done=done,
            same_photos=photos == sorted(project.processing.image_names),
            finished=bool(project.processing.finished_at),
        )

    def preflight(self) -> list[str]:
        """Problems that would make the run fail, found in seconds instead
        of after half an hour. Empty list = good to go."""
        problems = []
        folder = self.request.images_folder
        if not folder.is_dir():
            return [f"A pasta de fotos não existe: {folder}"]
        num_photos = count_photos(folder)
        if num_photos < MIN_PHOTOS:
            problems.append(
                f"A pasta de fotos tem {num_photos} foto(s) JPG/TIFF; são necessárias pelo menos {MIN_PHOTOS}."
            )
        try:
            self.out.mkdir(parents=True, exist_ok=True)
            probe = self.out / ".htrmapper_teste_escrita"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            problems.append(f"Não é possível gravar na pasta de saída {self.out}: {exc}")
            return problems
        if self.project_path.is_file() and self.inspect_existing() is None:
            problems.append(
                f"A pasta de saída já tem um {PROJECT_FILE_NAME} que não é de um processamento automático "
                "(por exemplo, um projeto do modo manual). Para não sobrescrevê-lo, escolha outra pasta de saída."
            )
        if self.require_gpu and not gpu_available():
            problems.append(
                "Nenhuma GPU NVIDIA com CUDA foi encontrada. A nuvem densa precisa dela: rode o HTRMapper "
                "dentro do WSL com o pycolmap-cuda12 instalado e o driver NVIDIA do Windows atualizado."
            )
        need_mb = DISK_MB_PER_PHOTO[self.level.key] * num_photos + DISK_MARGIN_MB
        try:
            self.work.mkdir(parents=True, exist_ok=True)
            free_mb = shutil.disk_usage(self.work).free / 1024**2
        except OSError as exc:
            problems.append(f"Não é possível criar a pasta de trabalho {self.work}: {exc}")
            return problems
        if free_mb < need_mb:
            problems.append(
                f"Pouco espaço em disco para os dados de trabalho ({self.work}): {free_mb / 1024:.1f} GB livres, "
                f"são necessários cerca de {need_mb / 1024:.1f} GB para {num_photos} fotos no nível {self.level.label}."
            )
        return problems

    def estimate(self) -> dict[str, timing.StepEstimate]:
        return timing.estimate(
            self.level.key,
            count_photos(self.request.images_folder),
            timing.hardware_signature(),
            STEP_KEYS,
            self.timing_history_path,
        )

    # -- running ---------------------------------------------------------------

    def run(self, resume: bool = True) -> AutoResult:
        self.out.mkdir(parents=True, exist_ok=True)
        handlers = self._start_logging()
        started = time.monotonic()
        try:
            problems = self.preflight()
            if problems:
                raise AutoProcessingError("Não é possível começar:\n- " + "\n- ".join(problems))
            done = self._prepare(resume)
            hardware = timing.hardware_signature()
            step_seconds: dict[str, float] = {}
            skipped = []
            for key in STEP_KEYS:
                if key in done:
                    skipped.append(key)
                    self.callbacks.step(key, "pulado", "já concluída antes")
                    logger.info("Etapa '%s': já concluída num processamento anterior, reaproveitada.", STEP_LABELS[key])
                    continue
                self._check_cancel(key)
                step_seconds[key] = self._run_step(key, hardware)
            self._finish()
            total = time.monotonic() - started
            logger.info("PROCESSAMENTO CONCLUÍDO em %s.", timing.format_duration(total))
            for key, path in self.project.processing.products.items():
                logger.info("  %s: %s", key, path)
            return AutoResult(
                project_path=self.project_path,
                products=dict(self.project.processing.products),
                step_seconds=step_seconds,
                skipped_steps=skipped,
                total_seconds=total,
                warnings=list(self.warnings),
            )
        except AutoProcessingError as exc:
            if exc.error_file is None:  # refused before any step ran: the message says it all
                logger.error("%s", exc)
            raise
        except InterruptedError:
            logger.warning("Processamento cancelado pelo usuário. Rode de novo para continuar de onde parou.")
            raise
        finally:
            self._stop_logging(handlers)

    def _check_cancel(self, key: str) -> None:
        token = self.cancellation_token
        if token is not None and token.is_cancelled:
            self.callbacks.step(key, "cancelado")
            raise InterruptedError("Operation cancelled")

    def _prepare(self, resume: bool) -> set[str]:
        """Load the earlier run to resume, or start clean. Returns the steps
        that can be skipped."""
        existing = self.inspect_existing()
        if existing is not None and resume:
            if not existing.resumable_with(self.request.level):
                raise AutoProcessingError(
                    f"A pasta de saída já tem outro processamento (nível '{LEVELS[existing.level].label}'"
                    + ("" if existing.same_photos else ", com outras fotos")
                    + "). Escolha outra pasta de saída ou mande recomeçar do zero."
                )
            self.project = Project.load(self.project_path)
            done = self._verified_done_steps()
            if done:
                logger.info(
                    "Retomando processamento anterior: %s já concluída(s).",
                    ", ".join(STEP_LABELS[k] for k in STEP_KEYS if k in done),
                )
            self.project.processing.finished_at = ""
            return done
        if existing is not None and not resume and self._remove_workdir():
            logger.info("Recomeçando do zero: dados de trabalho anteriores apagados (%s).", self.work)
        self.project = None
        return set()

    def _verified_done_steps(self) -> set[str]:
        """Steps marked done whose outputs still exist, up to the first one
        that does not (every later step depends on it, so it reruns too)."""
        p = self.project
        checks = {
            "importar": lambda: bool(p.images),
            "alinhar": lambda: p.sfm is not None and Path(p.sfm.reconstruction_path).exists(),
            "ajustar": lambda: p.ba is not None and Path(p.ba.reconstruction_path).exists(),
            "densa": lambda: (
                p.mvs is not None
                and Path(p.mvs.point_cloud_las_path).is_file()
                and Path(p.mvs.undistorted_reconstruction_path).exists()
            ),
            "dem": lambda: p.dem is not None and Path(p.dem.raster_path).is_file(),
            "ortho": lambda: p.ortho is not None and Path(p.ortho.raster_path).is_file(),
            "relatorio": lambda: Path(p.processing.products.get("relatorio", "/nonexistent")).is_file(),
        }
        done = set()
        for key in STEP_KEYS:
            if key in p.processing.steps_done and checks[key]():
                done.add(key)
            else:
                break
        for key in STEP_KEYS:
            if key not in done:
                p.processing.steps_done.pop(key, None)
        return done

    def _run_step(self, key: str, hardware: str) -> float:
        label = STEP_LABELS[key]
        logger.info("")
        logger.info("=== %s ===", label)
        self.callbacks.step(key, "rodando")
        started = time.monotonic()
        try:
            detail = getattr(self, f"_step_{key}")()
        except InterruptedError:
            self.callbacks.step(key, "cancelado")
            raise
        except AutoProcessingError as exc:
            exc.step = key
            exc.error_file = self._write_error_file(key, exc)
            logger.error("%s", exc)
            logger.error("Detalhes do erro em: %s", exc.error_file)
            self.callbacks.step(key, "erro", str(exc))
            raise
        except Exception as exc:
            message = _friendly_message(label, exc)
            error = AutoProcessingError(message, step=key)
            error.error_file = self._write_error_file(key, exc)
            logger.error("%s", message)
            logger.error("Detalhes do erro em: %s", error.error_file)
            self.callbacks.step(key, "erro", message)
            raise error from exc
        seconds = time.monotonic() - started
        self.project.processing.steps_done[key] = {
            "seconds": round(seconds, 1),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.project.save(self.project_path)
        timing.record_step(key, self.level.key, len(self.project.images), seconds, hardware, self.timing_history_path)
        logger.info("Concluído: %s, em %s. %s", label, timing.format_duration(seconds), detail or "")
        self.callbacks.step(key, "concluido", detail or "")
        return seconds

    def _phase(self, key: str):
        def callback(*args):
            if len(args) == 1:
                logger.info("  %s", args[0])
                self.callbacks.phase(key, str(args[0]))
            else:
                self.callbacks.progress(key, int(args[0]), int(args[1]))

        return callback

    # -- the steps ---------------------------------------------------------------

    def _step_importar(self) -> str:
        folder = self.request.images_folder
        records, report = import_folder(folder)
        for line in report.summary_lines():
            logger.info("  %s", line)
        valid = [r for r in records if r.position_valid]
        if len(valid) < MIN_PHOTOS:
            raise AutoProcessingError(
                f"Só {len(valid)} de {len(records)} foto(s) têm posição GPS válida; são necessárias pelo menos "
                f"{MIN_PHOTOS}. Confira se as fotos são as originais do drone (com os metadados de GPS)."
            )
        if len(valid) < len(records):
            self.warnings.append(f"{len(records) - len(valid)} foto(s) sem GPS válido.")
        epsg = sirgas2000_utm_epsg(mean(r.longitude for r in valid), mean(r.latitude for r in valid))
        crs = CoordinateReferenceSystem(epsg)
        self.project = Project(
            name=self.name,
            crs=ProjectCrsConfig(source_epsg=4326, project_epsg=epsg),
            images=records,
            processing=ProcessingSummary(
                level=self.level.key,
                images_folder=str(folder.resolve()),
                output_folder=str(self.out.resolve()),
                image_names=sorted(r.file_name for r in records),
                software_version=software_version(),
            ),
        )
        self.project.save(self.project_path)
        rtk = sum(1 for r in records if r.rtk_flag is not None)
        logger.info("  Sistema de coordenadas: EPSG:%s (%s), escolhido pela posição das fotos", epsg, crs.name)
        return f"{len(records)} fotos, {len(valid)} com GPS ({rtk} com dados RTK). EPSG:{epsg}."

    def _step_alinhar(self) -> str:
        from htrmapper.sfm.pipeline import run_structure_from_motion

        result = run_structure_from_motion(
            self.project, self.work / "alinhamento", self.level.sfm, self.cancellation_token, self._phase("alinhar")
        )
        self.project.sfm = result.to_project_summary()
        aligned = result.num_registered / max(1, result.num_images_input)
        if not result.georeferenced:
            raise AutoProcessingError(
                f"O alinhamento não conseguiu georreferenciar as fotos: {result.georeferencing_note}"
            )
        if aligned < MIN_ALIGNED_FRACTION:
            raise AutoProcessingError(
                f"Só {result.num_registered} de {result.num_images_input} fotos alinharam ({aligned:.0%}). "
                "Normalmente é pouca sobreposição ou fotos borradas/de outro voo misturadas na pasta."
            )
        if result.unregistered_image_names:
            self.warnings.append(
                f"{len(result.unregistered_image_names)} foto(s) não alinharam: {', '.join(result.unregistered_image_names)}"
            )
            logger.warning("  Fotos que não alinharam: %s", ", ".join(result.unregistered_image_names))
        return f"{result.num_registered}/{result.num_images_input} fotos alinhadas, {result.num_points3d} tie points."

    def _step_ajustar(self) -> str:
        from htrmapper.ba.weighted_bundle_adjustment import run_gnss_weighted_bundle_adjustment

        self._phase("ajustar")("Ajustando câmeras e lente com o GNSS (não pode ser cancelado no meio)")
        result = run_gnss_weighted_bundle_adjustment(
            self.project, Path(self.project.sfm.reconstruction_path), self.work / "ajuste", self.level.ba
        )
        self.project.ba = result.to_project_summary()
        if not result.converged:
            self.warnings.append(f"O ajuste GNSS terminou sem convergir ({result.termination_type}).")
            logger.warning("  O ajuste não convergiu (%s): confira o relatório.", result.termination_type)
        for note in result.intrinsics_notes:
            self.warnings.append(note)
            logger.warning("  %s", note)
        return (
            f"Erro câmera x GNSS: {result.rmse_xy_cm:.1f} cm XY, {result.rmse_z_cm:.1f} cm Z; "
            f"reprojeção {result.mean_reprojection_error_px:.2f} px."
        )

    def _step_densa(self) -> str:
        from htrmapper.mvs.dense import run_dense_reconstruction

        parents = {Path(img.path).resolve().parent for img in self.project.images}
        if len(parents) != 1:
            raise AutoProcessingError("As fotos estão em pastas diferentes; coloque todas na mesma pasta.")
        result = run_dense_reconstruction(
            self.project,
            Path(self.project.ba.reconstruction_path),
            next(iter(parents)),
            self.work / "densa",
            self.level.mvs,
            self.cancellation_token,
            self._phase("densa"),
        )
        self.project.mvs = MvsSummary(
            num_points=result.num_points,
            quality=result.quality,
            point_cloud_las_path=result.point_cloud_las_path,
            point_cloud_native_path=result.point_cloud_native_path,
            undistorted_image_path=result.undistorted_image_path,
            undistorted_reconstruction_path=result.undistorted_reconstruction_path,
            local_origin_m=result.local_origin_m,
        )
        return f"{result.num_points:,} pontos.".replace(",", ".")

    def _step_dem(self) -> str:
        from htrmapper.dem.generation import run_dem_generation

        result = run_dem_generation(
            Path(self.project.mvs.point_cloud_las_path),
            self.project.crs.project_epsg,
            self.product_path("dem"),
            self.level.dem,
            self.cancellation_token,
            self._phase("dem"),
        )
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
            num_cells_despiked=result.num_cells_despiked,
            num_cells_interpolated=result.num_cells_interpolated,
            nodata_fraction=result.nodata_fraction,
        )
        self.project.processing.products["dem"] = result.raster_path
        return f"Resolução {result.resolution_m * 100:.0f} cm, altitudes {result.min_elevation_m:.1f} a {result.max_elevation_m:.1f} m."

    def _step_ortho(self) -> str:
        from htrmapper.ortho.orthomosaic import run_orthomosaic_generation

        result = run_orthomosaic_generation(
            Path(self.project.mvs.undistorted_reconstruction_path),
            Path(self.project.mvs.undistorted_image_path),
            Path(self.project.dem.raster_path),
            self.product_path("ortho"),
            self.level.ortho,
            self.cancellation_token,
            self._phase("ortho"),
        )
        self.project.ortho = OrthoSummary(
            raster_path=result.raster_path,
            width_px=result.width_px,
            height_px=result.height_px,
            resolution_m=result.resolution_m,
            resolution_source=result.resolution_source,
            num_cameras_used=result.num_cameras_used,
            num_valid_pixels=result.num_valid_pixels,
            num_nodata_pixels=result.num_nodata_pixels,
        )
        self.project.processing.products["ortho"] = result.raster_path
        return f"{result.width_px} x {result.height_px} px, pixel de {result.resolution_m * 100:.2f} cm."

    def _step_relatorio(self) -> str:
        from htrmapper.report.pdf import write_pdf_report

        path = self.product_path("relatorio")
        write_pdf_report(self.project, path, self.level, log_path=self.log_path, warnings=self.warnings)
        self.project.processing.products["relatorio"] = str(path)
        return path.name

    def _finish(self) -> None:
        processing = self.project.processing
        processing.finished_at = datetime.now().isoformat(timespec="seconds")
        processing.products["log"] = str(self.log_path)
        processing.products["projeto"] = str(self.project_path)
        self.project.save(self.project_path)
        if self.request.delete_workdir_on_success and self._remove_workdir(reopen_log=False):
            logger.info(
                "Dados de trabalho apagados (%s), como pedido. A retomada e o modo avançado não terão mais as etapas intermediárias.",
                self.work,
            )

    def _remove_workdir(self, reopen_log: bool = True) -> bool:
        """Delete the working folder -- only if it looks like one this
        module created, never an arbitrary folder passed by mistake."""
        work = self.work
        ours = any((work / sub).exists() for sub in ("alinhamento", "ajuste", "densa", "logs"))
        if not (work.is_dir() and ours):
            return False
        self._redirect_colmap_log(False)
        shutil.rmtree(work, ignore_errors=True)
        if reopen_log:
            self._redirect_colmap_log(True)
        return True

    # -- logging and errors ----------------------------------------------------------

    def _start_logging(self) -> list[logging.Handler]:
        formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        gui_handler = _CallbackLogHandler(self.callbacks)
        gui_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logger.addHandler(file_handler)
        logger.addHandler(gui_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        self._redirect_colmap_log(True)

        logger.info("")
        logger.info("#" * 70)
        logger.info("HTRMapper %s -- processamento automático, nível %s", software_version(), self.level.label)
        logger.info("Fotos: %s", self.request.images_folder)
        logger.info("Saída: %s", self.out)
        logger.info("Dados de trabalho: %s", self.work)
        for line in self.level.parameters_text():
            logger.info("  %s", line)
        for line in _system_lines():
            logger.info("  %s", line)
        return [file_handler, gui_handler]

    def _stop_logging(self, handlers: list[logging.Handler]) -> None:
        self._redirect_colmap_log(False)
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

    def _redirect_colmap_log(self, on: bool) -> None:
        """COLMAP's own (glog) messages go to trabalho/logs/colmap_<data>.log."""
        try:
            import pycolmap

            log = pycolmap.logging
            if on:
                (self.work / "logs").mkdir(parents=True, exist_ok=True)
                log.set_log_destination(log.INFO, str(self.work / "logs" / "colmap_"))
                for level in (log.WARNING, log.ERROR, log.FATAL):
                    log.set_log_destination(level, "")
            else:
                log.set_log_destination(log.INFO, "")
        except Exception:  # noqa: BLE001 -- logging setup must never block processing
            pass

    def _write_error_file(self, step: str | None, exc: BaseException) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.out / f"ERRO_{stamp}.txt"
        counter = 2
        while path.exists():
            path = self.out / f"ERRO_{stamp}_{counter}.txt"
            counter += 1
        lines = [
            "HTRMapper -- relatório de erro",
            "=" * 60,
            f"Data: {datetime.now().isoformat(timespec='seconds')}",
            f"Etapa: {STEP_LABELS.get(step, step or '-')}",
            f"Mensagem: {exc}",
            f"Tipo: {type(exc).__module__}.{type(exc).__name__}",
            "",
            f"Fotos: {self.request.images_folder}",
            f"Saída: {self.out}",
            f"Nível: {self.level.label}",
            *[f"  {line}" for line in self.level.parameters_text()],
            "",
            "Sistema:",
            *[f"  {line}" for line in _system_lines()],
            "",
            "Rastreamento (traceback):",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
            "",
            f"Log do COLMAP: {self.work / 'logs'}",
            "",
            f"Final do {LOG_FILE_NAME}:",
        ]
        try:
            tail = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-150:]
        except OSError:
            tail = ["(não foi possível ler o log)"]
        lines += tail
        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            path = Path.home() / path.name
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _system_lines() -> list[str]:
    lines = [f"Python {platform.python_version()} em {platform.platform()}"]
    try:
        import pycolmap

        lines.append(f"pycolmap {pycolmap.__version__} (CUDA: {'sim' if pycolmap.has_cuda else 'não'})")
    except Exception:  # noqa: BLE001
        lines.append("pycolmap não disponível")
    try:
        from htrmapper.core.system_info import detect_system_info

        info = detect_system_info()
        lines.append(f"CPU: {info.cpu_name} ({info.cpu_logical_cores} núcleos), RAM {info.ram_total_gb:.0f} GB")
        lines += [f"GPU: {g.name}, {g.vram_total_mb / 1024:.0f} GB, driver {g.driver_version}" for g in info.gpus] or [
            "GPU: nenhuma NVIDIA detectada"
        ]
    except Exception:  # noqa: BLE001
        pass
    return lines


def _friendly_message(label: str, exc: BaseException) -> str:
    if isinstance(exc, MemoryError):
        return f"Falha em '{label}': memória RAM insuficiente. Feche outros programas ou use um nível mais baixo."
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return f"Falha em '{label}': o disco encheu. Libere espaço na pasta de saída e rode de novo (continua de onde parou)."
    text = str(exc) or type(exc).__name__
    return f"Falha em '{label}': {text}"


def main_for_cli(request: AutoRequest, resume: bool) -> int:
    """Terminal front-end (`htrmapper processar`): prints the log as it goes."""

    class Printer(Callbacks):
        def log(self, message: str) -> None:
            print(message, flush=True)

    processor = AutoProcessor(request, Printer())
    estimates = processor.estimate()
    total = sum(e.seconds for e in estimates.values())
    source = "histórico deste PC" if all(e.from_history for e in estimates.values()) else "estimativa inicial"
    print(f"Tempo estimado: {timing.format_duration(total)} ({source})", flush=True)
    try:
        result = processor.run(resume=resume)
    except AutoProcessingError as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        if exc.error_file:
            print(f"Detalhes em: {exc.error_file}", file=sys.stderr)
        return 1
    except InterruptedError:
        print("\nCancelado.", file=sys.stderr)
        return 130
    for warning in result.warnings:
        print(f"Aviso: {warning}")
    return 0
