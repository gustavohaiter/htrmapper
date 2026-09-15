"""CLI entry point for htrmapper -- usable without the GUI (automation/CI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from htrmapper.core.project import GnssAccuracyConfig, Project, ProjectCrsConfig
from htrmapper.geo.crs import CoordinateReferenceSystem
from htrmapper.gnss.accuracy import CameraAccuracy
from htrmapper.io.image_import import import_folder


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

    if args.project_out:
        project = Project(
            name=args.name or folder.name,
            crs=ProjectCrsConfig(source_epsg=4326, project_epsg=args.epsg),
            gnss_accuracy=GnssAccuracyConfig(
                accuracy=CameraAccuracy(xy_sigma_m=args.xy_sigma, z_sigma_m=args.z_sigma)
            ),
            images=records,
        )
        project.save(Path(args.project_out))
        print(f"\nProject saved to: {args.project_out}")

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
    import_parser.set_defaults(func=_cmd_import)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
