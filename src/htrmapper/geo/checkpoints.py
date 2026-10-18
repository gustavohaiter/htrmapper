"""Check points: absolute accuracy of the delivered orthomosaic and DEM.

Agreeing with Pix4D proves consistency, not accuracy -- both start from
the same RTK geotags. The proof is ground truth: targets laid out in the
field and measured with an RTK rover, compared with where they appear in
the products (ARCHITECTURE.md section 29).

For every field point:

- Horizontal: where the target is in the orthomosaic. Either measured by
  the user (columns E_ortho / N_ortho, e.g. clicked in QGIS) or found
  automatically: within `search_radius_m` of the field coordinate, the
  target is the bright (or dark) blob of plausible size nearest to it,
  separated from the background halfway between the field's typical gray
  and the window's extreme, and its centroid is the measured position. A blob with weak contrast, or none at all, is
  reported as not found -- never guessed.
- Vertical: the DEM elevation at the field coordinate (bilinear) minus the
  field height. Field heights must be in the same reference as the DEM
  used: the project DEM is ellipsoidal; pass the orthometric DEM (see
  `geo.vertical`) when the rover delivers normal heights.

This measures the ORTHOMOSAIC, which is what goes to the harvester. Using
the targets as control points inside the bundle adjustment is a separate,
later step.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

import numpy as np
import rasterio
import rasterio.errors
from rasterio.windows import Window, from_bounds
from scipy import ndimage

DEFAULT_SEARCH_RADIUS_M = 1.0
DEFAULT_MIN_TARGET_AREA_M2 = 0.05
DEFAULT_MAX_TARGET_AREA_M2 = 4.0
# Gray levels (0-255) between the blob and a ring of TARGET_RING_M around
# it below which a detection is too weak to report. Measured against the
# immediate surroundings, not the whole search window: a dark target on
# green cane (~50 levels apart) would otherwise look weak next to the
# window's own dark texture.
MIN_TARGET_CONTRAST = 30.0
TARGET_RING_M = 0.2

_NAME_ALIASES = ("nome", "name", "id", "ponto", "point")
_X_ALIASES = ("e", "este", "x", "easting", "leste")
_Y_ALIASES = ("n", "norte", "y", "northing")
_Z_ALIASES = ("h", "z", "altitude", "alt", "cota", "elev", "elevation", "height")
_MEASURED_X_ALIASES = ("e_ortho", "x_ortho")
_MEASURED_Y_ALIASES = ("n_ortho", "y_ortho")


class CheckpointError(RuntimeError):
    """Raised for unusable input (unreadable CSV, missing columns)."""


@dataclass
class FieldPoint:
    name: str
    x: float
    y: float
    z: float | None = None
    measured_x: float | None = None
    measured_y: float | None = None


@dataclass
class CheckpointResult:
    name: str
    field_x: float
    field_y: float
    field_z: float | None
    measured_x: float | None = None
    measured_y: float | None = None
    method: str = "not found"  # "manual", "auto" or "not found"
    note: str = ""
    dem_z: float | None = None

    @property
    def dx(self) -> float | None:
        return None if self.measured_x is None else self.measured_x - self.field_x

    @property
    def dy(self) -> float | None:
        return None if self.measured_y is None else self.measured_y - self.field_y

    @property
    def d2d(self) -> float | None:
        return None if self.dx is None else sqrt(self.dx**2 + self.dy**2)

    @property
    def dz(self) -> float | None:
        return None if self.dem_z is None or self.field_z is None else self.dem_z - self.field_z


@dataclass
class CheckpointReport:
    results: list[CheckpointResult] = field(default_factory=list)

    def _values(self, attribute: str) -> np.ndarray:
        return np.array([v for r in self.results if (v := getattr(r, attribute)) is not None], dtype=float)

    def rmse(self, attribute: str) -> float | None:
        values = self._values(attribute)
        return float(np.sqrt(np.mean(values**2))) if values.size else None

    def mean(self, attribute: str) -> float | None:
        values = self._values(attribute)
        return float(values.mean()) if values.size else None

    def max_abs(self, attribute: str) -> float | None:
        values = self._values(attribute)
        return float(np.abs(values).max()) if values.size else None


def _parse_number(text: str) -> float:
    return float(text.strip().replace(",", "."))


def read_field_points(csv_path: Path) -> list[FieldPoint]:
    """Read rover points. A header row with recognizable names (nome/E/N/h,
    optionally E_ortho/N_ortho) is used when present; otherwise the
    columns are name, E, N[, h]. Accepts ',', ';' or tab as separator
    and decimal comma when the separator is not a comma."""
    text = Path(csv_path).read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise CheckpointError(f"{csv_path} is empty")
    delimiter = max((";", "\t", ","), key=lambda d: lines[0].count(d))
    rows = list(csv.reader(lines, delimiter=delimiter))

    header = [cell.strip().lower() for cell in rows[0]]

    def column(aliases: tuple[str, ...]) -> int | None:
        return next((i for i, name in enumerate(header) if name in aliases), None)

    if column(_X_ALIASES) is not None and column(_Y_ALIASES) is not None:
        indices = {
            "name": column(_NAME_ALIASES),
            "x": column(_X_ALIASES),
            "y": column(_Y_ALIASES),
            "z": column(_Z_ALIASES),
            "mx": column(_MEASURED_X_ALIASES),
            "my": column(_MEASURED_Y_ALIASES),
        }
        data_rows = rows[1:]
    else:
        indices = {"name": 0, "x": 1, "y": 2, "z": 3 if len(rows[0]) > 3 else None, "mx": None, "my": None}
        data_rows = rows

    def optional(row: list[str], index: int | None) -> float | None:
        if index is None or index >= len(row) or not row[index].strip():
            return None
        return _parse_number(row[index])

    points: list[FieldPoint] = []
    for number, row in enumerate(data_rows, start=1):
        try:
            name = row[indices["name"]].strip() if indices["name"] is not None else f"P{number}"
            points.append(
                FieldPoint(
                    name=name,
                    x=_parse_number(row[indices["x"]]),
                    y=_parse_number(row[indices["y"]]),
                    z=optional(row, indices["z"]),
                    measured_x=optional(row, indices["mx"]),
                    measured_y=optional(row, indices["my"]),
                )
            )
        except (IndexError, ValueError) as exc:
            raise CheckpointError(f"{csv_path}, data row {number}: cannot read {row!r} ({exc})") from exc
    if not points:
        raise CheckpointError(f"{csv_path} has no points")
    return points


def _target_threshold(values: np.ndarray, bright: bool) -> float:
    """Gray level halfway between the typical background (median) and the
    extreme of the window (1st / 99th percentile), where a target lies.
    Otsu's threshold looked like the natural choice but splits a textured
    background (cane rows) down the middle whenever the target covers only
    a small part of the window, merging a dark target with the darker
    half of the field."""
    extreme = np.percentile(values, 99.0 if bright else 1.0)
    return 0.5 * (float(np.median(values)) + float(extreme))


def detect_target_center(
    ortho: "rasterio.io.DatasetReader",
    x: float,
    y: float,
    search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
    bright: bool = True,
    min_area_m2: float = DEFAULT_MIN_TARGET_AREA_M2,
    max_area_m2: float = DEFAULT_MAX_TARGET_AREA_M2,
) -> tuple[float | None, float | None, str]:
    """(x, y, note) of the target centroid near (x, y), or (None, None,
    reason) when there is no convincing target (module docstring)."""
    window = from_bounds(x - search_radius_m, y - search_radius_m, x + search_radius_m, y + search_radius_m, ortho.transform)
    window = window.round_offsets().round_lengths()
    try:
        window = window.intersection(Window(0, 0, ortho.width, ortho.height))
    except rasterio.errors.WindowError:
        return None, None, "outside the orthomosaic"
    if window.width < 3 or window.height < 3:
        return None, None, "outside the orthomosaic"
    rgb = ortho.read([1, 2, 3], window=window).astype(np.float64)
    valid = ortho.read_masks(1, window=window) > 0
    if valid.mean() < 0.5:
        return None, None, "mostly outside the mosaic coverage"
    gray = rgb.mean(axis=0)
    threshold = _target_threshold(gray[valid], bright)
    blobs = valid & ((gray > threshold) if bright else (gray < threshold))
    labels, count = ndimage.label(blobs, structure=np.ones((3, 3)))
    if count == 0:
        return None, None, "no target-like blob"

    transform = ortho.window_transform(window)
    pixel_area = abs(transform.a * transform.e)
    sizes = np.bincount(labels.ravel())[1:] * pixel_area
    centroids = ndimage.center_of_mass(blobs, labels, range(1, count + 1))
    best, best_distance = None, np.inf
    for label, (area, (row, col)) in enumerate(zip(sizes, centroids), start=1):
        if not (min_area_m2 <= area <= max_area_m2):
            continue
        cx, cy = transform * (col + 0.5, row + 0.5)
        distance = np.hypot(cx - x, cy - y)
        if distance < best_distance:
            best, best_distance = (label, cx, cy, area), distance
    if best is None:
        return None, None, f"no blob between {min_area_m2} and {max_area_m2} m2"

    label, cx, cy, area = best
    target = labels == label
    ring_px = max(1, int(round(TARGET_RING_M / abs(transform.a))))
    ring = valid & ~target & ndimage.binary_dilation(target, structure=np.ones((3, 3)), iterations=ring_px)
    contrast = abs(gray[target].mean() - gray[ring].mean()) if ring.any() else 0.0
    if contrast < MIN_TARGET_CONTRAST:
        return None, None, f"weak contrast ({contrast:.0f} gray levels)"
    return float(cx), float(cy), f"auto: {area:.2f} m2 blob, contrast {contrast:.0f}"


def _sample_bilinear(dem: "rasterio.io.DatasetReader", x: float, y: float) -> float | None:
    col_f, row_f = ~dem.transform * (x, y)
    col_f -= 0.5
    row_f -= 0.5
    col0, row0 = int(np.floor(col_f)), int(np.floor(row_f))
    if col0 < 0 or row0 < 0 or col0 + 1 >= dem.width or row0 + 1 >= dem.height:
        return None
    block = dem.read(1, window=Window(col0, row0, 2, 2)).astype(np.float64)
    if dem.nodata is not None and np.any(block == dem.nodata):
        return None
    fx, fy = col_f - col0, row_f - row0
    top = block[0, 0] * (1 - fx) + block[0, 1] * fx
    bottom = block[1, 0] * (1 - fx) + block[1, 1] * fx
    return float(top * (1 - fy) + bottom * fy)


def evaluate_checkpoints(
    points: list[FieldPoint],
    ortho_path: Path,
    dem_path: Path | None = None,
    search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
    bright_targets: bool = True,
    min_area_m2: float = DEFAULT_MIN_TARGET_AREA_M2,
    max_area_m2: float = DEFAULT_MAX_TARGET_AREA_M2,
) -> CheckpointReport:
    report = CheckpointReport()
    dem = rasterio.open(dem_path) if dem_path else None
    try:
        with rasterio.open(ortho_path) as ortho:
            for point in points:
                result = CheckpointResult(name=point.name, field_x=point.x, field_y=point.y, field_z=point.z)
                if point.measured_x is not None and point.measured_y is not None:
                    result.measured_x, result.measured_y = point.measured_x, point.measured_y
                    result.method, result.note = "manual", "measured by the user"
                else:
                    mx, my, note = detect_target_center(
                        ortho, point.x, point.y, search_radius_m, bright_targets, min_area_m2, max_area_m2
                    )
                    result.note = note
                    if mx is not None:
                        result.measured_x, result.measured_y, result.method = mx, my, "auto"
                if dem is not None:
                    result.dem_z = _sample_bilinear(dem, point.x, point.y)
                report.results.append(result)
    finally:
        if dem is not None:
            dem.close()
    return report


def _fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_report_csv(report: CheckpointReport, path: Path) -> None:
    """Per-point table plus RMSE / mean / max rows, ';'-separated for a
    Portuguese-locale spreadsheet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            ["nome", "E_campo", "N_campo", "h_campo", "E_ortho", "N_ortho", "metodo", "dE", "dN", "d2D", "h_DEM", "dh", "obs"]
        )
        for r in report.results:
            writer.writerow(
                [
                    r.name, _fmt(r.field_x), _fmt(r.field_y), _fmt(r.field_z), _fmt(r.measured_x), _fmt(r.measured_y),
                    r.method, _fmt(r.dx), _fmt(r.dy), _fmt(r.d2d), _fmt(r.dem_z), _fmt(r.dz), r.note,
                ]
            )
        for label, function in (("RMSE", report.rmse), ("media", report.mean), ("max_abs", report.max_abs)):
            writer.writerow(
                [label, "", "", "", "", "", "", _fmt(function("dx")), _fmt(function("dy")), _fmt(function("d2d")), "",
                 _fmt(function("dz")), ""]
            )
