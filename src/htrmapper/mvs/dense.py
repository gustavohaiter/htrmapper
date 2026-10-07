"""Phase 4: dense point cloud reconstruction (multi-view stereo).

Built on COLMAP's own patch-match stereo + stereo fusion
(`pycolmap.undistort_images` -> `pycolmap.patch_match_stereo` ->
`pycolmap.stereo_fusion`), per the architecture decision (ARCHITECTURE.md
section 4): no custom MVS algorithm is implemented here.

Important, empirically-confirmed constraint: COLMAP's patch-match stereo
has NO native CPU fallback -- it requires a CUDA (or HIP/AMD) GPU and
raises a clean `ValueError` ("Dense stereo reconstruction requires CUDA or
HIP...") otherwise. This module checks `pycolmap.has_cuda` up front and
fails fast with a clear, actionable message rather than letting that
native error surface partway through a long-running step, per the
project's "never hide why something failed" rule. Per the architecture's
GPU section, CUDA is not meant to be a hard requirement of HTRMapper as a
whole -- the documented (not yet implemented) fallback for GPU-less
machines is OpenMVS as an optional external process (already scoped that
way in ARCHITECTURE.md section 2 due to its AGPL license), never a custom
CPU MVS implementation.

**Runs in a local frame, not in absolute project-CRS coordinates**
(ARCHITECTURE.md section 25). After Phase 2/3 the sparse reconstruction
lives in absolute UTM meters (northing ~7.5 million in the user's area).
COLMAP's MVS keeps camera poses and fused points in single precision, and
float32 at a northing of 7.5e6 can only represent steps of 0.5 m -- a pose
error that, at ~100 m flying height, shifts every pixel by several pixels,
far beyond patch-match's 1-2 px consistency thresholds. The real-flight
symptoms matched exactly: only ~9 of 56 images contributed fused points,
and the lower "media" tier produced *more* points than "alta" (the same
metric error is fewer pixels at lower resolution). So the reconstruction
is shifted by a rounded local origin (`_local_origin`) before
undistortion, MVS runs entirely in that local frame, and the origin is
added back only at the boundaries: the exported LAS, and a georeferenced
copy of the undistorted model that Phase 6 consumes. The native `fused.ply`
and the dense workspace stay in the local frame; the origin is persisted
next to them (`local_origin.json`) and in `MvsResult.local_origin_m`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np
import pycolmap

from htrmapper.core.project import Project
from htrmapper.geo.crs import CoordinateReferenceSystem

logger = logging.getLogger(__name__)


class MvsError(RuntimeError):
    """Raised when dense reconstruction cannot proceed at all (missing GPU,
    missing input reconstruction, etc). Never raised for a merely sparse
    or low-density result -- that is reported in MvsResult."""


# max_image_size caps the longest image dimension fed to patch-match
# stereo; this is COLMAP's own knob for compute/quality tradeoff. The
# four tiers mirror the "Low/Medium/High/Ultra High" quality convention
# used by standard photogrammetry reports (the same terms the project
# brief asks for): "muito_alta" processes at native resolution (no cap);
# each tier below roughly halves the linear resolution (quarters the
# pixel count and, correspondingly, compute time), the standard tradeoff
# curve for MVS depth estimation. `geom_consistency` (a photometric
# pass followed by a cross-view consistency check) is disabled only at
# the lowest tier to keep it meaningfully faster, matching how these
# report tiers behave in mature tools.
_QUALITY_PRESETS: dict[str, dict] = {
    "baixa": dict(max_image_size=800, window_radius=4, num_samples=8, geom_consistency=False),
    "media": dict(max_image_size=1600, window_radius=5, num_samples=15, geom_consistency=True),
    "alta": dict(max_image_size=3200, window_radius=5, num_samples=15, geom_consistency=True),
    "muito_alta": dict(max_image_size=-1, window_radius=5, num_samples=15, geom_consistency=True),
}


@dataclass
class MvsConfig:
    quality: str = "media"
    # All three below default to None ("use pycolmap's own default,
    # unchanged") -- exposed, not changed, following a real but unresolved
    # finding (ARCHITECTURE.md section 24): the one real execution of this
    # module on real hardware so far (56-photo nadir UAV flight, RTX 3060
    # Ti, 2026-10-06/07) produced an unusually sparse dense cloud (~1-3
    # points/m^2, when MVS typically yields far more). Two plausible,
    # low-risk levers for the user to experiment with on their own GPU
    # (this sandbox has none to validate against):
    #
    # num_patch_match_src_images: how many candidate source images
    # `undistort_images` picks per reference image for stereo matching.
    # pycolmap's own default (-1) empirically resolved to 20 in development
    # logs -- more candidates is strictly more data for the
    # geometric-consistency filter to work with, never less.
    #
    # filter_min_triangulation_angle / filter_min_ncc: COLMAP's own FAQ
    # explicitly warns that long-distance, nadir-only aerial imagery (our
    # exact case, ~100m altitude) can have genuinely small but still valid
    # triangulation angles -- the default filter (currently 3 degrees) may
    # be discarding real terrain points, not just noise, on this kind of
    # flight.
    num_patch_match_src_images: int | None = None
    filter_min_triangulation_angle: float | None = None
    filter_min_ncc: float | None = None

    def __post_init__(self) -> None:
        if self.quality not in _QUALITY_PRESETS:
            raise ValueError(f"unknown quality {self.quality!r}; choose from {sorted(_QUALITY_PRESETS)}")


@dataclass
class MvsResult:
    num_points: int = 0
    quality: str = ""
    point_cloud_las_path: str = ""
    point_cloud_native_path: str = ""
    undistorted_image_path: str = ""
    undistorted_reconstruction_path: str = ""
    # Added to every local-frame coordinate to get back to the project CRS
    # (see module docstring). The LAS and `undistorted_reconstruction_path`
    # are already absolute; only `point_cloud_native_path` is local.
    local_origin_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


def _local_origin(reconstruction: "pycolmap.Reconstruction") -> np.ndarray:
    """Rounded mean of the registered camera centers, horizontal only.

    Rounded to whole meters so the offset is human-readable in logs and in
    `local_origin.json`; Z is left at 0 because elevations (~hundreds of
    meters) are already well inside float32's precise range.
    """
    centers = np.array([reconstruction.image(i).projection_center() for i in reconstruction.reg_image_ids()])
    origin = np.round(centers.mean(axis=0))
    origin[2] = 0.0
    return origin


def _translated(reconstruction: "pycolmap.Reconstruction", translation: np.ndarray) -> "pycolmap.Reconstruction":
    reconstruction.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(), np.asarray(translation, dtype=np.float64)))
    return reconstruction


def _export_to_las(
    reconstruction: "pycolmap.Reconstruction",
    project_epsg: int,
    las_path: Path,
    offset: np.ndarray | None = None,
) -> None:
    xyz = []
    rgb = []
    for point in reconstruction.points3D.values():
        xyz.append(point.xyz)
        rgb.append(point.color)

    if not xyz:
        raise MvsError("stereo fusion produced zero points; cannot export an empty point cloud")

    xyz_arr = np.array(xyz, dtype=np.float64)
    if offset is not None:
        xyz_arr += np.asarray(offset, dtype=np.float64)
    rgb_arr = np.array(rgb, dtype=np.uint8)

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.add_crs(CoordinateReferenceSystem(project_epsg).proj_crs)
    # LAS scale factor: sub-millimeter, appropriate for the project's
    # centimetric-or-better precision goal (never coarser than the
    # precision the rest of the pipeline claims).
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    header.offsets = xyz_arr.min(axis=0)

    las = laspy.LasData(header)
    las.x = xyz_arr[:, 0]
    las.y = xyz_arr[:, 1]
    las.z = xyz_arr[:, 2]
    # LAS RGB channels are 16-bit; scale up from the 8-bit COLMAP color.
    las.red = rgb_arr[:, 0].astype(np.uint16) * 257
    las.green = rgb_arr[:, 1].astype(np.uint16) * 257
    las.blue = rgb_arr[:, 2].astype(np.uint16) * 257

    las_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(las_path)


def run_dense_reconstruction(
    project: Project,
    reconstruction_path: Path,
    image_root: Path,
    workdir: Path,
    config: MvsConfig | None = None,
    cancellation_token: "pycolmap.CancellationToken | None" = None,
    progress_callback: "Callable[[str], None] | None" = None,
) -> MvsResult:
    """Run undistortion + patch-match stereo + fusion to produce a dense,
    colored, georeferenced point cloud.

    Raises MvsError if no CUDA/HIP GPU is available (see module docstring)
    or if the input reconstruction/images cannot be found -- these are
    "cannot even start" conditions, never silently downgraded.

    `cancellation_token`/`progress_callback`: see the equivalent parameters
    on `sfm.pipeline.run_structure_from_motion` -- same contract (coarse,
    phase-level progress; a cancelled token surfaces as `InterruptedError`,
    raised by pycolmap itself, never masked as `MvsError`).
    """
    config = config or MvsConfig()
    reconstruction_path = Path(reconstruction_path)
    image_root = Path(image_root)
    workdir = Path(workdir)

    def _report(phase: str) -> None:
        if progress_callback is not None:
            progress_callback(phase)

    if not reconstruction_path.exists():
        raise MvsError(f"reconstruction not found at {reconstruction_path}")
    if not image_root.is_dir():
        raise MvsError(f"image folder not found at {image_root}")
    if not pycolmap.has_cuda:
        raise MvsError(
            "Reconstrução densa (patch-match stereo) requer GPU NVIDIA (CUDA) ou AMD (HIP); "
            "nenhuma foi detectada nesta máquina. O COLMAP não possui fallback de CPU nativo "
            "para essa etapa. Se esta máquina tem uma GPU NVIDIA real e você está no Windows: "
            "o pacote pycolmap instalado via pip não traz suporte a CUDA em nenhuma plataforma "
            "Windows -- rode esta etapa (htrmapper dense/dem/ortho) de dentro do WSL2, com "
            "pycolmap-cuda12 instalado lá, apontando para o mesmo projeto.json (os caminhos são "
            "traduzidos automaticamente); ver ARCHITECTURE.md seções 21-23. Alternativa (não "
            "implementada ainda): OpenMVS externo para CPU."
        )

    dense_workspace = workdir / "dense"
    dense_workspace.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(str(reconstruction_path))
    if reconstruction.num_reg_images() == 0:
        raise MvsError(f"reconstruction at {reconstruction_path} has no registered images")
    local_origin = _local_origin(reconstruction)
    local_sparse_path = workdir / "sparse_local"
    local_sparse_path.mkdir(parents=True, exist_ok=True)
    _translated(reconstruction, -local_origin).write(local_sparse_path)
    (workdir / "local_origin.json").write_text(
        json.dumps({"local_origin_m": local_origin.tolist(), "project_epsg": project.crs.effective_export_epsg}),
        encoding="utf-8",
    )
    logger.info("dense reconstruction runs in a local frame; origin (project CRS) = %s", local_origin.tolist())

    _report("Desdistorcendo imagens")
    pycolmap.undistort_images(
        output_path=dense_workspace,
        input_path=local_sparse_path,
        image_path=image_root,
        num_patch_match_src_images=(
            config.num_patch_match_src_images if config.num_patch_match_src_images is not None else -1
        ),
        cancellation_token=cancellation_token,
    )

    # Phase 6 samples the undistorted images through this model and looks
    # terrain heights up in the (absolute) DEM, so it needs the undistorted
    # cameras back in the project CRS. Written in double precision, so the
    # round trip local -> absolute is exact.
    georef_sparse_path = dense_workspace / "sparse_georef"
    georef_sparse_path.mkdir(parents=True, exist_ok=True)
    _translated(pycolmap.Reconstruction(str(dense_workspace / "sparse")), local_origin).write(georef_sparse_path)

    preset = _QUALITY_PRESETS[config.quality]
    options = pycolmap.PatchMatchOptions()
    options.max_image_size = preset["max_image_size"]
    options.window_radius = preset["window_radius"]
    options.num_samples = preset["num_samples"]
    options.geom_consistency = preset["geom_consistency"]
    if config.filter_min_triangulation_angle is not None:
        options.filter_min_triangulation_angle = config.filter_min_triangulation_angle
    if config.filter_min_ncc is not None:
        options.filter_min_ncc = config.filter_min_ncc

    _report("Patch-match stereo (GPU)")
    pycolmap.patch_match_stereo(
        workspace_path=dense_workspace, options=options, cancellation_token=cancellation_token
    )

    input_type = "geometric" if preset["geom_consistency"] else "photometric"
    fused_path = workdir / "fused.ply"
    fusion_options = pycolmap.StereoFusionOptions()
    _report("Fusão estéreo")
    dense_reconstruction = pycolmap.stereo_fusion(
        output_path=fused_path,
        workspace_path=dense_workspace,
        input_type=input_type,
        options=fusion_options,
        # Without this, pycolmap's binding defaults output_type to "bin" and
        # tries to read the returned Reconstruction back from output_path as
        # if it were a COLMAP binary reconstruction directory (cameras.bin/
        # images.bin/points3D.bin), even though output_path is a .ply file --
        # confirmed via `pycolmap-cuda12` 4.2.1 on real hardware: it raises
        # `Check failed: colmap::ExistsDir(path_val)` on the .ply path itself.
        output_type="PLY",
        cancellation_token=cancellation_token,
    )

    las_path = workdir / "dense_point_cloud.las"
    _report("Exportando nuvem de pontos (LAS)")
    _export_to_las(dense_reconstruction, project.crs.effective_export_epsg, las_path, offset=local_origin)

    return MvsResult(
        num_points=len(dense_reconstruction.points3D),
        quality=config.quality,
        point_cloud_las_path=str(las_path),
        point_cloud_native_path=str(fused_path),
        # Phase 6 samples colors from exactly these undistorted images and
        # must use the exact model that matches them (never the original
        # distorted images/sparse model) -- the georeferenced copy, since
        # the workspace's own `sparse` is in the local frame.
        undistorted_image_path=str(dense_workspace / "images"),
        undistorted_reconstruction_path=str(georef_sparse_path),
        local_origin_m=local_origin.tolist(),
    )
