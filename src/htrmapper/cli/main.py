"""CLI entry point for htrmapper -- usable without the GUI (automation/CI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from htrmapper.core.project import GnssAccuracyConfig, MvsSummary, Project, ProjectCrsConfig
from htrmapper.core.report import build_report_from_project, render_html
from htrmapper.geo.crs import CoordinateReferenceSystem
from htrmapper.gnss.accuracy import CameraAccuracy
from htrmapper.io.image_import import import_folder
from htrmapper.ba.weighted_bundle_adjustment import BaConfig, BaError, run_gnss_weighted_bundle_adjustment
from htrmapper.mvs.dense import MvsConfig, MvsError, run_dense_reconstruction
from htrmapper.sfm.pipeline import SfmConfig, SfmError, run_structure_from_motion


def _cmd_import(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        return 1

    # Validate the project CRS eagerly (via PROJ), before doing any work,
    # so a bad EPSG code fails fast with a clear message.
    try:
        crs = CoordinateReferenceSystem(args.epsg)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not crs.is_projected:
        print(
            f"warning: EPSG:{args.epsg} ({crs.name}) is not a projected CRS; "
            "the project CRS should normally be a metric projection (e.g. UTM), "
            "not geographic lat/lon",
            file=sys.stderr,
        )

    records, report = import_folder(folder)

    print(f"Import folder: {folder}")
    print(f"Project CRS: EPSG:{args.epsg} ({crs.name})")
    print(f"Camera accuracy: xy={args.xy_sigma} m, z={args.z_sigma} m")
    print()
    for line in report.summary_lines():
        print(line)

    project = Project(
        name=args.name or folder.name,
        crs=ProjectCrsConfig(source_epsg=4326, project_epsg=args.epsg),
        gnss_accuracy=GnssAccuracyConfig(
            accuracy=CameraAccuracy(xy_sigma_m=args.xy_sigma, z_sigma_m=args.z_sigma)
        ),
        images=records,
    )

    if args.project_out:
        project.save(Path(args.project_out))
        print(f"\nProject saved to: {args.project_out}")

    if args.report_out:
        report = build_report_from_project(project)
        Path(args.report_out).write_text(render_html(report), encoding="utf-8")
        print(f"Report saved to: {args.report_out}")

    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    project_path = Path(args.project)
    if not project_path.is_file():
        print(f"error: project file not found: {project_path}", file=sys.stderr)
        return 1

    project = Project.load(project_path)
    if not project.images:
        print("error: project has no imported images; run 'htrmapper import' first", file=sys.stderr)
        return 1

    workdir = Path(args.workdir)
    config = SfmConfig(key_point_limit=args.key_point_limit)

    print(f"Aligning project: {project.name} ({len(project.images)} images)")
    print(f"Work directory: {workdir}")
    print()

    try:
        result = run_structure_from_motion(project, workdir, config)
    except SfmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Matching strategy: {result.matching_strategy}")
    for group in result.camera_groups:
        print(
            f"Camera group '{group.camera_model}': {group.num_images} image(s), "
            f"intrinsics from {group.intrinsics_source}"
            + (f" (fx={group.fx_px:.1f}px)" if group.fx_px else "")
        )
    print()
    print(f"Registered: {result.num_registered} / {result.num_images_input}")
    if result.unregistered_image_names:
        print(f"Unregistered images: {result.unregistered_image_names}")
    print(f"Tie points (3D): {result.num_points3d}")
    print(f"Observations (projections): {result.num_observations}")
    if result.mean_reprojection_error_px is not None:
        print(f"Mean reprojection error (COLMAP's own initial BA, unweighted): {result.mean_reprojection_error_px:.3f} px")
    print(f"Georeferenced: {result.georeferenced} -- {result.georeferencing_note}")

    project.sfm = result.to_project_summary()
    project.save(project_path)
    print(f"\nProject updated: {project_path}")

    return 0


def _cmd_adjust(args: argparse.Namespace) -> int:
    project_path = Path(args.project)
    if not project_path.is_file():
        print(f"error: project file not found: {project_path}", file=sys.stderr)
        return 1

    project = Project.load(project_path)
    if project.sfm is None or not project.sfm.reconstruction_path:
        print("error: project has no Fase 2 (SfM) result; run 'htrmapper align' first", file=sys.stderr)
        return 1

    workdir = Path(args.workdir)
    config = BaConfig(refine_intrinsics=args.refine_intrinsics)

    print(f"Adjusting project: {project.name}")
    print(f"GNSS accuracy: xy={project.gnss_accuracy.accuracy.xy_sigma_m} m, z={project.gnss_accuracy.accuracy.z_sigma_m} m")
    print()

    try:
        result = run_gnss_weighted_bundle_adjustment(project, Path(project.sfm.reconstruction_path), workdir, config)
    except BaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Termination: {result.termination_type} (converged: {result.converged})")
    print(f"Images adjusted: {result.num_images_adjusted} ({result.num_images_with_gnss_prior} with GNSS prior)")
    print(f"Mean reprojection error (final, GNSS-weighted): {result.mean_reprojection_error_px:.4f} px")
    print(
        f"GNSS residual RMSE: X={result.rmse_x_cm:.3f} cm, Y={result.rmse_y_cm:.3f} cm, "
        f"Z={result.rmse_z_cm:.3f} cm, XY={result.rmse_xy_cm:.3f} cm, Total={result.rmse_total_cm:.3f} cm"
    )
    print(f"Max error: {result.max_error_cm:.3f} cm")
    for camera_id, data in result.camera_calibration.items():
        print(f"Camera {camera_id} ({data['model']}, {data['params_info']}): {data['params']}")

    project.ba = result.to_project_summary()
    project.save(project_path)
    print(f"\nProject updated: {project_path}")

    return 0


def _cmd_dense(args: argparse.Namespace) -> int:
    project_path = Path(args.project)
    if not project_path.is_file():
        print(f"error: project file not found: {project_path}", file=sys.stderr)
        return 1

    project = Project.load(project_path)
    if project.sfm is None or not project.sfm.reconstruction_path:
        print("error: project has no Fase 2 (SfM) result; run 'htrmapper align' first", file=sys.stderr)
        return 1

    # Prefer the Fase 3 (GNSS-weighted) refined reconstruction when present;
    # fall back to the Fase 2 one otherwise.
    reconstruction_path = Path(
        project.ba.reconstruction_path if project.ba and project.ba.reconstruction_path else project.sfm.reconstruction_path
    )

    image_parents = {Path(img.path).resolve().parent for img in project.images}
    if len(image_parents) != 1:
        print(f"error: images span {len(image_parents)} folders; cannot determine a single image root", file=sys.stderr)
        return 1
    image_root = next(iter(image_parents))

    workdir = Path(args.workdir)
    config = MvsConfig(quality=args.quality)

    print(f"Dense reconstruction for project: {project.name}")
    print(f"Reconstruction: {reconstruction_path}")
    print(f"Quality: {config.quality}")
    print()

    try:
        result = run_dense_reconstruction(project, reconstruction_path, image_root, workdir, config)
    except MvsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Dense points: {result.num_points}")
    print(f"Point cloud (LAS): {result.point_cloud_las_path}")
    print(f"Point cloud (native/PLY): {result.point_cloud_native_path}")

    project.mvs = MvsSummary(
        num_points=result.num_points,
        quality=result.quality,
        point_cloud_las_path=result.point_cloud_las_path,
        point_cloud_native_path=result.point_cloud_native_path,
    )
    project.save(project_path)
    print(f"\nProject updated: {project_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="htrmapper", description="HTRMapper photogrammetry CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="Import a folder of images into a project")
    import_parser.add_argument("folder", help="Path to the folder containing source images")
    import_parser.add_argument(
        "--epsg", type=int, default=31983, help="Project CRS EPSG code (default: 31983, SIRGAS 2000 / UTM 23S)"
    )
    import_parser.add_argument("--xy-sigma", type=float, default=0.02, help="GNSS/PPK horizontal accuracy, meters")
    import_parser.add_argument("--z-sigma", type=float, default=0.02, help="GNSS/PPK vertical accuracy, meters")
    import_parser.add_argument("--name", default=None, help="Project name (default: folder name)")
    import_parser.add_argument("--project-out", default=None, help="Path to save the .json project file")
    import_parser.add_argument("--report-out", default=None, help="Path to save the HTML processing report")
    import_parser.set_defaults(func=_cmd_import)

    align_parser = subparsers.add_parser(
        "align", help="Fase 2: feature extraction, matching and initial SfM (requires a saved project)"
    )
    align_parser.add_argument("project", help="Path to a project .json file saved by 'htrmapper import'")
    align_parser.add_argument("--workdir", required=True, help="Directory for the COLMAP database/reconstruction")
    align_parser.add_argument(
        "--key-point-limit", type=int, default=40_000, help="Max SIFT features per image (default: 40000)"
    )
    align_parser.set_defaults(func=_cmd_align)

    adjust_parser = subparsers.add_parser(
        "adjust", help="Fase 3: bundle adjustment ponderado por GNSS/PPK (requires 'htrmapper align' first)"
    )
    adjust_parser.add_argument("project", help="Path to a project .json file with a Fase 2 (align) result")
    adjust_parser.add_argument("--workdir", required=True, help="Directory for the refined reconstruction")
    adjust_parser.add_argument(
        "--refine-intrinsics",
        action="store_true",
        help="Also refine focal length/principal point/distortion (risky for nadir-only flat flights; see BaConfig docstring)",
    )
    adjust_parser.set_defaults(func=_cmd_adjust)

    dense_parser = subparsers.add_parser(
        "dense", help="Fase 4: nuvem de pontos densa (requer GPU CUDA; 'htrmapper align' primeiro)"
    )
    dense_parser.add_argument("project", help="Path to a project .json file with a Fase 2 (align) result")
    dense_parser.add_argument("--workdir", required=True, help="Directory for the dense workspace/point cloud")
    dense_parser.add_argument(
        "--quality", choices=["baixa", "media", "alta", "muito_alta"], default="media", help="Dense quality tier"
    )
    dense_parser.set_defaults(func=_cmd_dense)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
