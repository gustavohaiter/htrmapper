from __future__ import annotations

import pytest

from htrmapper.geo.coverage import Point2D, convex_hull, convex_hull_area_km2, polygon_area


def test_convex_hull_of_square_returns_four_corners():
    points = [Point2D(0, 0), Point2D(0, 10), Point2D(10, 10), Point2D(10, 0), Point2D(5, 5)]

    hull = convex_hull(points)

    assert len(hull) == 4
    hull_set = {(round(p.x), round(p.y)) for p in hull}
    assert hull_set == {(0, 0), (0, 10), (10, 10), (10, 0)}


def test_polygon_area_of_unit_square_is_one():
    square = [Point2D(0, 0), Point2D(1, 0), Point2D(1, 1), Point2D(0, 1)]

    assert polygon_area(square) == pytest.approx(1.0)


def test_polygon_area_of_triangle():
    triangle = [Point2D(0, 0), Point2D(4, 0), Point2D(0, 3)]

    assert polygon_area(triangle) == pytest.approx(6.0)


def test_polygon_area_degenerate_returns_zero():
    assert polygon_area([Point2D(0, 0)]) == 0.0
    assert polygon_area([Point2D(0, 0), Point2D(1, 1)]) == 0.0


def test_convex_hull_area_km2_of_1km_square():
    # 1000 m x 1000 m square = 1 km^2.
    points = [Point2D(0, 0), Point2D(1000, 0), Point2D(1000, 1000), Point2D(0, 1000)]

    area = convex_hull_area_km2(points)

    assert area == pytest.approx(1.0)


def test_interior_points_do_not_affect_hull_area():
    corners = [Point2D(0, 0), Point2D(1000, 0), Point2D(1000, 1000), Point2D(0, 1000)]
    interior = [Point2D(500, 500), Point2D(300, 700)]

    area_with_interior = convex_hull_area_km2(corners + interior)
    area_without = convex_hull_area_km2(corners)

    assert area_with_interior == pytest.approx(area_without)
