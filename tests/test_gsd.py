from __future__ import annotations

import pytest

from htrmapper.geo.gsd import estimate_gsd, estimate_gsd_from_pixel_pitch


def test_pixel_pitch_formula_matches_known_sensor_case():
    # S.O.D.A camera (senseFly eBee): 5472x3648, f=10.6mm, pixel 2.4um,
    # flying at 175m -> Metashape reports ~4.03 cm/pix ground resolution.
    gsd_cm = estimate_gsd_from_pixel_pitch(flying_height_m=175, focal_length_mm=10.6, pixel_pitch_um=2.4)
    assert gsd_cm == pytest.approx(3.9623, abs=0.01)


def test_crop_factor_derivation_matches_pixel_pitch_method():
    # Sensor width implied by 5472 px * 2.4 um pixel pitch = 13.1328 mm.
    sensor_width_mm = 5472 * 2.4e-3
    crop_factor = 36.0 / sensor_width_mm
    focal_35mm_equiv = 10.6 * crop_factor

    result = estimate_gsd(
        flying_height_m=175,
        focal_length_mm=10.6,
        focal_length_35mm_equiv_mm=focal_35mm_equiv,
        image_width_px=5472,
    )

    direct = estimate_gsd_from_pixel_pitch(175, 10.6, 2.4)
    assert result.gsd_cm_per_px == pytest.approx(direct, abs=1e-9)
    assert result.pixel_pitch_mm == pytest.approx(2.4e-3, abs=1e-9)


def test_higher_altitude_means_coarser_gsd():
    low = estimate_gsd_from_pixel_pitch(100, 10.6, 2.4)
    high = estimate_gsd_from_pixel_pitch(200, 10.6, 2.4)
    assert high == pytest.approx(2 * low)


def test_longer_focal_length_means_finer_gsd():
    short_focal = estimate_gsd_from_pixel_pitch(175, 8.8, 2.4)
    long_focal = estimate_gsd_from_pixel_pitch(175, 17.6, 2.4)
    assert long_focal == pytest.approx(short_focal / 2)


@pytest.mark.parametrize(
    "flying_height_m,focal_length_mm,pixel_pitch_um",
    [(0, 10.6, 2.4), (-1, 10.6, 2.4), (175, 0, 2.4), (175, 10.6, 0), (175, -10.6, 2.4)],
)
def test_non_positive_inputs_are_rejected(flying_height_m, focal_length_mm, pixel_pitch_um):
    with pytest.raises(ValueError):
        estimate_gsd_from_pixel_pitch(flying_height_m, focal_length_mm, pixel_pitch_um)


def test_estimate_gsd_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        estimate_gsd(0, 10.6, 29.0, 5472)
    with pytest.raises(ValueError):
        estimate_gsd(175, 10.6, 29.0, 0)
