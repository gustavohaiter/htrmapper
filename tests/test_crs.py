from __future__ import annotations

import pytest

from htrmapper.geo.crs import (
    CoordinateReferenceSystem,
    GeodeticPoint,
    GeodeticTransformer,
    sirgas2000_utm23s,
    wgs84,
)


def test_wgs84_is_geographic_and_utm23s_is_projected():
    assert wgs84().is_geographic
    assert not wgs84().is_projected
    assert sirgas2000_utm23s().is_projected
    assert not sirgas2000_utm23s().is_geographic


def test_invalid_epsg_raises_value_error():
    with pytest.raises(ValueError):
        CoordinateReferenceSystem(999999999)


def test_forward_transform_brasilia_known_value():
    # Known reference: Praca dos Tres Poderes area, Brasilia.
    # Expected UTM 23S easting/northing cross-checked against pyproj directly
    # (see test below) -- this test locks the *plumbing*, not an external
    # ground truth number, since pyproj itself is the trusted PROJ backend.
    transformer = GeodeticTransformer(wgs84(), sirgas2000_utm23s())
    point = GeodeticPoint(lon=-47.882778, lat=-15.793889)

    projected = transformer.forward(point)

    # Brasilia is roughly at UTM 23S easting ~190000-200000, northing ~8250000-8255000.
    assert 150_000 < projected.x < 250_000
    assert 8_200_000 < projected.y < 8_300_000


def test_forward_then_inverse_round_trip_is_accurate_to_millimeters():
    transformer = GeodeticTransformer(wgs84(), sirgas2000_utm23s())
    original = GeodeticPoint(lon=-47.882778, lat=-15.793889, alt=850.0)

    projected = transformer.forward(original)
    back = transformer.inverse(projected)

    assert back.lon == pytest.approx(original.lon, abs=1e-9)
    assert back.lat == pytest.approx(original.lat, abs=1e-9)
    assert back.alt == pytest.approx(original.alt, abs=1e-6)


def test_altitude_is_carried_through_projection():
    transformer = GeodeticTransformer(wgs84(), sirgas2000_utm23s())
    point = GeodeticPoint(lon=-47.882778, lat=-15.793889, alt=850.0)

    projected = transformer.forward(point)

    assert projected.z is not None
    assert projected.z == pytest.approx(850.0, abs=0.01)


def test_no_naive_linear_conversion_matches_proj_directly():
    """Guard against ever bypassing PROJ with a hand-rolled formula."""
    import pyproj

    transformer = GeodeticTransformer(wgs84(), sirgas2000_utm23s())
    point = GeodeticPoint(lon=-47.882778, lat=-15.793889)
    projected = transformer.forward(point)

    reference = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:31983", always_xy=True)
    ref_x, ref_y = reference.transform(point.lon, point.lat)

    assert projected.x == pytest.approx(ref_x)
    assert projected.y == pytest.approx(ref_y)
