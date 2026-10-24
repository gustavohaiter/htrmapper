"""Mede, em centimetros, o deslocamento entre dois ortomosaicos do mesmo voo.

Uso (no .venv do HTRMapper):

    python tools/comparar_ortomosaicos.py NOSSO.tif REFERENCIA.tif [--saida pontos.geojson]

NOSSO e o ortomosaico do HTRMapper; REFERENCIA e o do Pix4D, Metashape etc.
A referencia e reprojetada (bilinear) na grade do nosso, num VRT, sem gravar
nada; o CRS dela pode ser outro (ex.: EPSG:32723), e a transformacao usada
e informada no relatorio.

Como mede:

1. Deslocamento global numa versao reduzida (~40 cm/px) da area toda, para
   pegar um deslocamento grande (dezenas de cm ou metros) antes de tudo.
2. Numa grade de pontos (padrao: a cada 20 m), recorta uma janela de
   256 x 256 px dos dois mosaicos e acha o deslocamento por correlacao de
   fase com precisao de 1/20 de pixel. A busca fica limitada a +-30 px (~75 cm) em
   torno do global: as linhas de cana se repetem a cada ~1,5 m e uma busca
   maior poderia "pular" para a linha vizinha.
3. Descarta janelas com area vazia, sem textura ou com pico de correlacao
   fraco ou ambiguo, e resume o resto: mediana, media, RMSE, p95 e maximo.

Sinal: dE/dN e quanto um objeto esta deslocado NA REFERENCIA em relacao ao
NOSSO (dE > 0: na referencia ele aparece mais a leste).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

WINDOW_PX = 256
SEARCH_RADIUS_PX = 30
GLOBAL_PIXEL_M = 0.40
GLOBAL_SEARCH_M = 5.0
MIN_VALID_FRACTION = 0.95
MIN_GRAY_STD = 6.0
MIN_PEAK = 0.15
MIN_PEAK_RATIO = 1.5
PEAK_EXCLUSION_PX = 4
UPSAMPLE = 20
LOWPASS_CYCLES_PER_PX: float | None = 0.15


@dataclass
class Match:
    east: float
    north: float
    d_east_m: float
    d_north_m: float
    peak: float
    peak_ratio: float

    @property
    def d_2d_m(self) -> float:
        return float(np.hypot(self.d_east_m, self.d_north_m))


def _gray(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _prepare(gray: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Zero-mean, invalid pixels at the mean, Hann-tapered edges."""
    mean = gray[valid].mean() if valid.any() else 0.0
    out = np.where(valid, gray, mean) - mean
    hann = np.outer(np.hanning(out.shape[0]), np.hanning(out.shape[1]))
    return out * hann


def _upsampled_dft(data: np.ndarray, region: int, factor: int, offsets: np.ndarray) -> np.ndarray:
    """Inverse DFT of `data` on a `factor`-times finer grid, only on a
    `region` x `region` patch starting at `offsets` (Guizar-Sicairos et
    al. 2008, the method scikit-image uses)."""
    for n_items, offset in list(zip(data.shape, offsets))[::-1]:
        kernel = (np.arange(region) - offset)[:, None] * np.fft.fftfreq(n_items, factor)
        data = np.tensordot(np.exp(-2j * np.pi * kernel), data, axes=(1, -1))
    return data


def phase_shift(
    ours: np.ndarray,
    ref: np.ndarray,
    valid: np.ndarray,
    center: tuple[int, int] = (0, 0),
    radius: int | None = None,
) -> tuple[float, float, float, float]:
    """(row shift, col shift, peak, peak / second peak) of `ref` relative to
    `ours`: a feature at (r, c) in `ours` is at (r + dr, c + dc) in `ref`.
    The peak is searched within `radius` of `center` (a shift)."""
    a = _prepare(ours, valid)
    b = _prepare(ref, valid)
    cross = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    cross /= np.maximum(np.abs(cross), 1e-12)
    if LOWPASS_CYCLES_PER_PX is not None:
        # The finest detail differs between the two mosaics (resampling,
        # JPEG); pure phase correlation would weigh that noise as much as
        # the real structure. Normalized so a perfect match still peaks at 1.
        freq = np.hypot(
            *np.meshgrid(
                np.fft.fftfreq(cross.shape[0]),
                np.fft.fftfreq(cross.shape[1]),
                indexing="ij",
            )
        )
        weight = np.exp(-((freq / LOWPASS_CYCLES_PER_PX) ** 2))
        cross *= weight / weight.mean()
    surface = np.fft.fftshift(np.real(np.fft.ifft2(cross)))
    rows, cols = surface.shape
    zero_r, zero_c = rows // 2, cols // 2

    search = np.zeros_like(surface, dtype=bool)
    if radius is None:
        search[:] = True
    else:
        r0, c0 = zero_r + center[0], zero_c + center[1]
        search[max(0, r0 - radius) : r0 + radius + 1, max(0, c0 - radius) : c0 + radius + 1] = True
    masked = np.where(search, surface, -np.inf)
    peak_r, peak_c = np.unravel_index(int(np.argmax(masked)), surface.shape)
    peak = float(surface[peak_r, peak_c])

    rest = masked.copy()
    rest[
        max(0, peak_r - PEAK_EXCLUSION_PX) : peak_r + PEAK_EXCLUSION_PX + 1,
        max(0, peak_c - PEAK_EXCLUSION_PX) : peak_c + PEAK_EXCLUSION_PX + 1,
    ] = -np.inf
    second = float(rest.max()) if np.isfinite(rest).any() else 0.0
    ratio = peak / second if second > 0 else np.inf

    # Refine to 1/UPSAMPLE px around the integer peak.
    coarse = np.array([peak_r - zero_r, peak_c - zero_c], dtype=float)
    region = int(np.ceil(UPSAMPLE * 1.5))
    center_idx = np.trunc(region / 2.0)
    fine = np.real(np.conj(_upsampled_dft(np.conj(cross), region, UPSAMPLE, center_idx - coarse * UPSAMPLE)))
    best = np.array(np.unravel_index(int(np.argmax(fine)), fine.shape), dtype=float)
    shift = coarse + (best - center_idx) / UPSAMPLE
    return float(shift[0]), float(shift[1]), peak, ratio


def _read_ours(src, window: Window, out_shape=None) -> tuple[np.ndarray, np.ndarray]:
    kwargs = {} if out_shape is None else {"out_shape": out_shape, "resampling": Resampling.average}
    rgb = src.read([1, 2, 3], window=window, **kwargs)
    mask_kwargs = {} if out_shape is None else {"out_shape": out_shape[1:]}
    valid = src.dataset_mask(window=window, **mask_kwargs) > 0
    return rgb, valid


def _global_offset(ours, ref_path: Path) -> tuple[float, float, float]:
    """(dE m, dN m, peak) over the whole overlap at ~GLOBAL_PIXEL_M."""
    factor = max(1, round(GLOBAL_PIXEL_M / ours.res[0]))
    width, height = ours.width // factor, ours.height // factor
    window = Window(0, 0, width * factor, height * factor)
    rgb, valid = _read_ours(ours, window, out_shape=(3, height, width))
    coarse_transform = ours.transform @ Affine.scale(factor)
    with (
        rasterio.open(ref_path) as ref_src,
        WarpedVRT(
            ref_src,
            crs=ours.crs,
            transform=coarse_transform,
            width=width,
            height=height,
            resampling=Resampling.average,
        ) as vrt,
    ):
        ref_rgb = vrt.read([1, 2, 3])
        ref_valid = (vrt.dataset_mask() > 0) & (ref_rgb.max(axis=0) > 0)
    both = valid & ref_valid
    if both.sum() < 0.05 * both.size:
        sys.exit("Os dois ortomosaicos quase nao se sobrepoem: confira os arquivos e o CRS.")
    radius = int(GLOBAL_SEARCH_M / (ours.res[0] * factor))
    dr, dc, peak, _ = phase_shift(_gray(rgb), _gray(ref_rgb), both, radius=radius)
    pixel = ours.res[0] * factor
    return dc * pixel, -dr * pixel, peak


def compare(ours_path: Path, ref_path: Path, spacing_m: float) -> tuple[list[Match], dict]:
    with rasterio.open(ours_path) as ours, rasterio.open(ref_path) as ref_src:
        info = {
            "ours_crs": str(ours.crs),
            "ours_epsg": ours.crs.to_epsg(),
            "ours_res_m": ours.res[0],
            "ref_crs": str(ref_src.crs),
            "ref_res": ref_src.res[0],
        }
        cx, cy = ours.xy(ours.height // 2, ours.width // 2)
        if ref_src.crs != ours.crs:
            xs, ys = warp_transform(ours.crs, ref_src.crs, [cx], [cy])
            back_x, back_y = warp_transform(ref_src.crs, ours.crs, xs, ys)
            info["crs_shift_m"] = (xs[0] - cx, ys[0] - cy)
            info["crs_roundtrip_m"] = float(np.hypot(back_x[0] - cx, back_y[0] - cy))

    with rasterio.open(ours_path) as ours:
        g_east, g_north, g_peak = _global_offset(ours, ref_path)
        info["global"] = (g_east, g_north, g_peak)
        res = ours.res[0]
        center = (round(-g_north / res), round(g_east / res))
        step = max(WINDOW_PX, round(spacing_m / res))
        half = WINDOW_PX // 2

        matches: list[Match] = []
        attempted = rejected_empty = rejected_texture = rejected_peak = 0
        with (
            rasterio.open(ref_path) as ref_src,
            WarpedVRT(
                ref_src,
                crs=ours.crs,
                transform=ours.transform,
                width=ours.width,
                height=ours.height,
                resampling=Resampling.bilinear,
            ) as vrt,
        ):
            for row in range(half, ours.height - half, step):
                for col in range(half, ours.width - half, step):
                    window = Window(col - half, row - half, WINDOW_PX, WINDOW_PX)
                    valid = ours.dataset_mask(window=window) > 0
                    if valid.mean() < MIN_VALID_FRACTION:
                        continue
                    ref_rgb = vrt.read([1, 2, 3], window=window)
                    # Black also counts as empty, for a reference without alpha or NoData.
                    ref_valid = (vrt.dataset_mask(window=window) > 0) & (ref_rgb.max(axis=0) > 0)
                    attempted += 1
                    both = valid & ref_valid
                    if both.mean() < MIN_VALID_FRACTION:
                        rejected_empty += 1
                        continue
                    a = _gray(ours.read([1, 2, 3], window=window))
                    b = _gray(ref_rgb)
                    if a[both].std() < MIN_GRAY_STD or b[both].std() < MIN_GRAY_STD:
                        rejected_texture += 1
                        continue
                    dr, dc, peak, ratio = phase_shift(a, b, both, center=center, radius=SEARCH_RADIUS_PX)
                    at_border = max(abs(dr - center[0]), abs(dc - center[1])) >= SEARCH_RADIUS_PX - 0.5
                    if peak < MIN_PEAK or ratio < MIN_PEAK_RATIO or at_border:
                        rejected_peak += 1
                        continue
                    east, north = ours.xy(row, col)
                    matches.append(Match(east, north, dc * res, -dr * res, peak, ratio))

    info.update(
        attempted=attempted,
        rejected_empty=rejected_empty,
        rejected_texture=rejected_texture,
        rejected_peak=rejected_peak,
    )
    return matches, info


def write_geojson(path: Path, matches: list[Match], epsg: int | None) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [m.east, m.north]},
            "properties": {
                "dE_cm": round(m.d_east_m * 100, 2),
                "dN_cm": round(m.d_north_m * 100, 2),
                "d2D_cm": round(m.d_2d_m * 100, 2),
                "pico": round(m.peak, 3),
            },
        }
        for m in matches
    ]
    collection = {
        "type": "FeatureCollection",
        "features": features,
    }
    if epsg is not None:
        collection["crs"] = {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
        }
    path.write_text(json.dumps(collection), encoding="utf-8")


def _summary(matches: list[Match], info: dict) -> str:
    lines = [
        f"Nosso:      {info['ours_crs']}, {info['ours_res_m'] * 100:.2f} cm/px",
        f"Referencia: {info['ref_crs']}, resolucao {info['ref_res']:.4g} (unidades do CRS dela)",
    ]
    if "crs_shift_m" in info:
        sx, sy = info["crs_shift_m"]
        lines.append(
            f"Transformacao entre os CRS no centro: dE {sx * 100:+.2f} cm, dN {sy * 100:+.2f} cm "
            f"(ida e volta {info['crs_roundtrip_m'] * 1000:.2f} mm)"
        )
    g_east, g_north, g_peak = info["global"]
    lines.append(
        f"Deslocamento global (~40 cm/px): dE {g_east * 100:+.1f} cm, dN {g_north * 100:+.1f} cm "
        f"(pico de correlacao {g_peak:.2f})"
    )
    lines.append(
        f"Janelas testadas: {info['attempted']}  aceitas: {len(matches)}  "
        f"descartadas: {info['rejected_empty']} sem sobreposicao, {info['rejected_texture']} sem textura, "
        f"{info['rejected_peak']} correlacao fraca/ambigua"
    )
    if not matches:
        lines.append("Nenhuma janela aceita: nao da para medir.")
        return "\n".join(lines)
    de = np.array([m.d_east_m for m in matches]) * 100
    dn = np.array([m.d_north_m for m in matches]) * 100
    d2 = np.hypot(de, dn)
    lines += [
        "",
        "Deslocamento da REFERENCIA em relacao ao NOSSO (cm):",
        f"  Leste (dE):  mediana {np.median(de):+.1f}   media {de.mean():+.1f}   desvio {de.std():.1f}",
        f"  Norte (dN):  mediana {np.median(dn):+.1f}   media {dn.mean():+.1f}   desvio {dn.std():.1f}",
        (
            f"  2D:          mediana {np.median(d2):.1f}   RMSE {np.sqrt(np.mean(d2**2)):.1f}   "
            f"p95 {np.percentile(d2, 95):.1f}   maximo {d2.max():.1f}"
        ),
        (
            f"  Ate 2,5 cm (1 pixel): {np.mean(d2 <= 2.5) * 100:.0f}%   ate 5 cm: {np.mean(d2 <= 5) * 100:.0f}%   "
            f"ate 10 cm: {np.mean(d2 <= 10) * 100:.0f}%"
        ),
        "",
        "5 pontos com maior diferenca (E, N -> dE, dN, 2D em cm):",
    ]
    for m in sorted(matches, key=lambda m: -m.d_2d_m)[:5]:
        lines.append(
            f"  {m.east:.2f}, {m.north:.2f} -> {m.d_east_m * 100:+.1f}, {m.d_north_m * 100:+.1f}, {m.d_2d_m * 100:.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nosso", type=Path, help="ortomosaico do HTRMapper")
    parser.add_argument("referencia", type=Path, help="ortomosaico do Pix4D/Metashape")
    parser.add_argument(
        "--espacamento",
        type=float,
        default=20.0,
        help="distancia entre pontos, em m (padrao 20)",
    )
    parser.add_argument("--saida", type=Path, help="GeoJSON com os pontos medidos (para abrir no QGIS)")
    args = parser.parse_args(argv)

    matches, info = compare(args.nosso, args.referencia, args.espacamento)
    print(_summary(matches, info))
    if args.saida and matches:
        write_geojson(args.saida, matches, info["ours_epsg"])
        print(f"\nPontos salvos em: {args.saida}")
    return 0 if matches else 1


if __name__ == "__main__":
    sys.exit(main())
