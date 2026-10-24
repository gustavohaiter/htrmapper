"""tools/comparar_ortomosaicos.py: recovers a known shift between two
orthomosaics in cm, with the reference in another CRS (EPSG:32723, RGBA,
as Pix4D writes it) and the scene dominated by repeating cane rows."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "comparar_ortomosaicos.py"
spec = importlib.util.spec_from_file_location("comparar_ortomosaicos", TOOL_PATH)
comparar = importlib.util.module_from_spec(spec)
sys.modules["comparar_ortomosaicos"] = comparar
spec.loader.exec_module(comparar)

RES_M = 0.025
SIZE_PX = 2400
ORIGIN = (260000.0, 7548700.0)


def _scene(shift_east_m: float, shift_north_m: float, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """RGB of the same ground rendered twice: the second with every feature
    moved by (shift_east_m, shift_north_m). Rows every 1.5 m, slightly
    rotated, over blotchy soil/leaf texture."""
    rng = np.random.default_rng(seed)
    noise = ndimage.gaussian_filter(rng.normal(0, 1, (SIZE_PX + 200, SIZE_PX + 200)), 3)
    blotch = ndimage.gaussian_filter(rng.normal(0, 1, (SIZE_PX + 200, SIZE_PX + 200)), 25)

    def render(dx_px: float, dy_px: float) -> np.ndarray:
        rows, cols = np.mgrid[0:SIZE_PX, 0:SIZE_PX].astype(float)
        # A feature at (r, c) in the first is at (r + dy, c + dx) in the second.
        src_r, src_c = rows - dy_px + 100, cols - dx_px + 100
        along = (src_c * np.cos(0.3) + src_r * np.sin(0.3)) * RES_M
        stripes = np.cos(2 * np.pi * along / 1.5)
        texture = ndimage.map_coordinates(noise, [src_r, src_c], order=1)
        soil = ndimage.map_coordinates(blotch, [src_r, src_c], order=1)
        gray = 120 + 40 * stripes + 18 * texture * 4 + 30 * soil * 8
        green = np.clip(gray, 0, 255)
        return np.stack([green * 0.8, green, green * 0.6]).astype(np.uint8)

    return render(0.0, 0.0), render(shift_east_m / RES_M, -shift_north_m / RES_M)


def _write_pair(tmp_path: Path, shift_east_m: float, shift_north_m: float) -> tuple[Path, Path]:
    ours_rgb, ref_rgb = _scene(shift_east_m, shift_north_m)
    transform = from_origin(*ORIGIN, RES_M, RES_M)
    ours_path = tmp_path / "ours.tif"
    with (
        rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
        rasterio.open(
            ours_path,
            "w",
            driver="GTiff",
            width=SIZE_PX,
            height=SIZE_PX,
            count=3,
            dtype="uint8",
            crs="EPSG:31983",
            transform=transform,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="JPEG",
            photometric="YCBCR",
            jpeg_quality=90,
        ) as dst,
    ):
        dst.write(ours_rgb)
        mask = np.full((SIZE_PX, SIZE_PX), 255, np.uint8)
        mask[:, :300] = 0
        dst.write_mask(mask)
    ref_path = tmp_path / "ref.tif"
    alpha = np.full((1, SIZE_PX, SIZE_PX), 255, np.uint8)
    alpha[:, -250:, :] = 0
    with rasterio.open(
        ref_path,
        "w",
        driver="GTiff",
        width=SIZE_PX,
        height=SIZE_PX,
        count=4,
        dtype="uint8",
        crs="EPSG:32723",
        transform=transform,
        photometric="RGB",
    ) as dst:
        dst.write(np.concatenate([ref_rgb, alpha]))
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]
    return ours_path, ref_path


@pytest.mark.parametrize("shift", [(0.03, -0.02), (0.0, 0.0), (-0.45, 0.62)])
def test_recovers_known_shift(tmp_path, shift):
    ours_path, ref_path = _write_pair(tmp_path, *shift)
    matches, info = comparar.compare(ours_path, ref_path, spacing_m=8.0)

    assert len(matches) >= 20
    assert info["rejected_peak"] <= 0.1 * info["attempted"]
    de = np.array([m.d_east_m for m in matches])
    dn = np.array([m.d_north_m for m in matches])
    assert np.median(de) == pytest.approx(shift[0], abs=0.004)
    assert np.median(dn) == pytest.approx(shift[1], abs=0.004)
    assert np.percentile(np.abs(de - shift[0]), 95) < 0.005
    assert np.percentile(np.abs(dn - shift[1]), 95) < 0.005
    # WGS 84 -> SIRGAS 2000 is the null transformation in PROJ.
    assert np.hypot(*info["crs_shift_m"]) < 0.001


def test_geojson_output(tmp_path):
    ours_path, ref_path = _write_pair(tmp_path, 0.03, -0.02)
    out = tmp_path / "pontos.geojson"
    assert comparar.main([str(ours_path), str(ref_path), "--espacamento", "8", "--saida", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["crs"]["properties"]["name"] == "urn:ogc:def:crs:EPSG::31983"
    first = data["features"][0]["properties"]
    assert first["dE_cm"] == pytest.approx(3.0, abs=0.5)
    assert first["dN_cm"] == pytest.approx(-2.0, abs=0.5)
