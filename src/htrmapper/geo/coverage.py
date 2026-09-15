"""Coverage-area estimation from camera positions.

Before alignment/orthomosaic exist, the only honest way to estimate the
survey's spatial extent is the convex hull of the camera stations
themselves (this is exactly what the "Coverage area" figure in a
Metashape-style report approximates before the true orthomosaic footprint
is known). This module implements a standard convex hull (Andrew's
monotone chain) and polygon area (shoelace formula) directly with
`numpy`/pure Python -- no `shapely` dependency needed for this one
computation, keeping the dependency footprint small.

Once the orthomosaic exists (Phase 6), the report should switch to the
orthomosaic's actual valid-pixel footprint area, which is more accurate
(the convex hull always overestimates a non-convex flight boundary such
as an L-shaped field). Both are surfaced with distinct labels in the
report -- never silently swapped for one another.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


def convex_hull(points: list[Point2D]) -> list[Point2D]:
    """Andrew's monotone chain convex hull. Returns hull vertices in
    counter-clockwise order, without duplicating the first point at the end.
    """
    pts = sorted(set((p.x, p.y) for p in points))
    if len(pts) <= 2:
        return [Point2D(x, y) for x, y in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [Point2D(x, y) for x, y in hull]


def polygon_area(vertices: list[Point2D]) -> float:
    """Shoelace formula. Returns 0 for fewer than 3 vertices (degenerate)."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i].x * vertices[j].y
        area -= vertices[j].x * vertices[i].y
    return abs(area) / 2.0


def convex_hull_area_km2(points: list[Point2D]) -> float:
    """Convex hull area of a set of projected (metric) points, in km^2."""
    hull = convex_hull(points)
    area_m2 = polygon_area(hull)
    return area_m2 / 1_000_000.0
