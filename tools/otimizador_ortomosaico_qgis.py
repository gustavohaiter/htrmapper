# -*- coding: utf-8 -*-
"""
===============================================================================
 OTIMIZADOR DE ORTOMOSAICO PARA QGIS
===============================================================================
 Versão : 2.0.0
 Uso    : QGIS 3.x (Qt5 ou Qt6) -> Complementos > Terminal Python >
          "Mostrar editor" > abrir este arquivo > "Executar script".
 Requer : apenas o Python do próprio QGIS (osgeo.gdal, osgeo.osr, numpy, PyQt).

 O QUE O SCRIPT FAZ
 ------------------
 1. Analisa o GeoTIFF de entrada (metadados + amostragem reduzida do conteúdo)
    e ESTIMA o tamanho final de cada compressão, comprimindo de verdade (em
    memória) blocos da própria imagem - com PSNR para as opções com perda.
 2. Reprojeta DE VERDADE os pixels para o CRS de destino (padrão EPSG:31983)
    com gdal.Warp, preservando a resolução original (~4 cm). Se a entrada JÁ
    estiver no CRS de destino, os pixels são copiados 1:1, sem reamostragem.
 3. Grava um GeoTIFF tiled 512x512 em um de dois modos:
      * SEM PERDA: ZSTD / DEFLATE / LZW + PREDICTOR=2 (RGB + NoData=0 ou RGBA);
      * COM PERDA CONTROLADA (o substituto livre do ECW): JPEG (YCbCr) ou WEBP,
        com a transparência numa máscara interna de 1 bit (padrão GDAL; o QGIS
        mostra a máscara como banda Alpha). Tipicamente 5-6x menor que o ZSTD
        com qualidade 90 (PSNR > 40 dB = visualmente idêntico).
 4. Cria overviews internas (pirâmides) com AVERAGE, que respeitam NoData e
    máscara (sem halo preto nas bordas em nenhum nível de zoom).
 5. Opcional: saída COG (Cloud Optimized GeoTIFF), para servidor/nuvem.
 6. Reabre e valida o arquivo final: compara blocos com a imagem reprojetada
    (sem perda: bit a bit; com perda: PSNR), compara pontos com a ORIGINAL e
    gera relatório (salvo também em caso de erro).
 7. Lote: vários arquivos de uma vez, com resumo final.

 ESTRATÉGIA (por que não um único gdal.Warp direto para GeoTIFF?)
 ----------------------------------------------------------------
 * gdal.Warp é feito para um VRT "warpado" (arquivo XML minúsculo, sem pixels).
   Os pixels são reprojetados sob demanda, bloco a bloco, quando lidos. Não
   existe GeoTIFF intermediário gigante (exceto no modo COG, ver abaixo).
 * O VRT produz 4 bandas: RGB + Alpha calculada pelo próprio warp a partir da
   Alpha/NoData/máscara da entrada (marca exatamente onde há cobertura do
   ortomosaico, inclusive os cantos vazios criados pela rotação da reprojeção).
 * Um laço em Python lê janelas alinhadas aos tiles (512 linhas x até 8192
   colunas -> ~16 MB por janela). Uma thread auxiliar reprojeta a PRÓXIMA
   janela enquanto a principal comprime e grava a atual (~30% mais rápido,
   resultado bit a bit idêntico ao sequencial).
 * Modo SEM PERDA + NoData: em pixels sem cobertura grava RGB = 0,0,0
   (NoData); em pixels com cobertura, qualquer canal que valha 0 passa a valer
   1 - "0" fica reservado ao NoData. Sem isso, sombras com R, G ou B = 0
   ficariam transparentes no QGIS. A alteração 0->1 é de 1 nível em 255
   (0,4%): visualmente imperceptível.
 * Modo COM PERDA + máscara: JPEG/WEBP alteram levemente os valores, então
   "NoData=0" não serve (geraria furos e bordas sujas). A área válida vai numa
   máscara interna 1 bit, comprimida sem perda: a borda fica EXATA.
 * Modo COG: o COG exige uma ordem interna específica (pirâmides antes dos
   dados), que só o driver COG do GDAL sabe montar a partir de um arquivo
   pronto. Por isso grava-se primeiro um intermediário sem perda rápido (na
   pasta temporária) e o driver COG faz a única compressão final.
 * Memória: limitada pelas janelas (~50 MB) + cache de blocos do GDAL
   (ajustado temporariamente e restaurado ao final). Nunca se lê o raster
   inteiro com ReadAsArray().

 HISTÓRICO
 ---------
 2.0.0  Compressão com perda JPEG/WEBP + máscara interna; estimativa de
        tamanho e PSNR por amostragem real; cópia 1:1 sem reamostragem quando
        a entrada já está no CRS de destino (antes a imagem inteira era
        reamostrada se a grade não fosse múltipla exata da resolução); saída
        COG; lote; leitura e gravação em paralelo; decodificação multithread
        da entrada; validação bit a bit (sem perda) / PSNR (com perda);
        relatório salvo também em caso de erro; lembra as configurações.
 1.1.0  Versão original (sem perda, NoData/Alpha, overviews, validação).
===============================================================================
"""

import os
import re
import sys
import math
import time
import queue
import shutil
import random
import datetime
import tempfile
import threading
import traceback

import numpy as np
from osgeo import gdal, osr

# --- Imports do QGIS/Qt (qgis.PyQt abstrai PyQt5/PyQt6 automaticamente) -----
try:
    from qgis.PyQt.QtCore import QThread, pyqtSignal, QSettings, QUrl
    from qgis.PyQt.QtGui import QFontDatabase, QTextCursor, QDesktopServices
    from qgis.PyQt.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
        QLineEdit, QPushButton, QFileDialog, QComboBox, QCheckBox,
        QRadioButton, QButtonGroup, QDoubleSpinBox, QSpinBox, QProgressBar,
        QPlainTextEdit, QMessageBox, QWidget,
    )
    from qgis.core import (
        Qgis, QgsProject, QgsCoordinateReferenceSystem, QgsMessageLog,
        QgsRasterLayer,
    )
    from qgis.gui import QgsProjectionSelectionWidget
    HAS_QGIS = True
except ImportError:  # permite importar o núcleo fora do QGIS (testes)
    HAS_QGIS = False

SCRIPT_NAME = "Otimizador de Ortomosaico"
SCRIPT_VERSION = "2.0.0"
DEFAULT_DST_CRS = "EPSG:31983"
SETTINGS_PREFIX = "OtimizadorOrtomosaico/"

# Tile de 512x512: é o padrão do COG/GDAL moderno. Comparado a 256x256:
#  * 4x menos tiles -> índice TIFF menor, menos chamadas de leitura/descompressão
#    por tela renderizada (o QGIS desenha áreas de ~1000-2000 px de lado);
#  * ZSTD/DEFLATE + PREDICTOR comprimem melhor blocos maiores;
#  * 512x512x3 bytes = 768 KB descomprimidos por tile: custo baixo em disco local.
# 256 só compensa para acesso remoto/HTTP com leituras muito pequenas.
DEFAULT_TILE = 512

# Janela de processamento = 1 linha de tiles x até 16 tiles (512 x 8192 px).
WINDOW_TILES_X = 16
# Janelas reprojetadas com antecedência pela thread de leitura (memória ~16 MB cada).
PREFETCH_WINDOWS = 2

# Pixel com Alpha >= 128 (cobertura >= 50%) é considerado válido ao remover a
# Alpha. Alpha parcial só existe na borda do mosaico (1 pixel de antialiasing).
ALPHA_VALID_THRESHOLD = 128

# Amostragem da análise de conteúdo (dimensão máxima do buffer reduzido).
ANALYSIS_MAX_DIM = 2048
# Entrada TILED sem overviews: uma leitura reduzida "normal" obrigaria o GDAL a
# descomprimir TODOS os tiles (cada linha amostrada cruza todos). Em vez disso
# amostramos só algumas linhas de tiles uniformemente espaçadas (~25% do arquivo).
ANALYSIS_TILE_ROWS = 32

# Limite clássico do TIFF = 4 GiB (offsets de 32 bits). Margem de segurança.
CLASSIC_TIFF_LIMIT = 4 * 1024 ** 3
BIGTIFF_SAFETY = 0.75

RESAMPLING_WARP = ["cubic", "bilinear", "lanczos", "near"]
RESAMPLING_OVR = ["AVERAGE", "BILINEAR", "CUBIC", "NEAREST"]

LOSSLESS_CODECS = ("ZSTD", "DEFLATE", "LZW")
LOSSY_CODECS = ("JPEG", "WEBP")
DEFAULT_COMPRESSION = "JPEG"
DEFAULT_QUALITY = 90

# Estimativa de tamanho: blocos reais da imagem comprimidos em memória,
# espalhados pela área toda (só o centro do voo enviesa: lá costuma haver mais
# lavoura e menos borda/estrada que a média).
ESTIMATE_GRID = (6, 8)          # posições candidatas (linhas x colunas)
ESTIMATE_MAX_SAMPLES = 32       # blocos usados (distribuídos pela grade)
# Overviews somam ~1/3 dos pixels do nível 0 (1/4 + 1/16 + ...), mas imagem
# reduzida tem mais detalhe por pixel e comprime pior. Fatores medidos em
# ortomosaicos de teste: ZSTD 1,375-1,376 | JPEG/WEBP 1,407-1,416.
OVR_SIZE_FACTOR = {"lossless": 1.38, "lossy": 1.42}


# =============================================================================
#  UTILITÁRIOS
# =============================================================================
class CancelledError(Exception):
    """Processamento cancelado pelo usuário."""


class ProcessingError(Exception):
    """Erro com mensagem clara para o usuário."""


def gdal_version_num():
    try:
        return int(gdal.VersionInfo("VERSION_NUM"))
    except Exception:
        return 0


def gdal_version_str():
    try:
        return gdal.VersionInfo("RELEASE_NAME")
    except Exception:
        return "?"


def fmt_bytes(n):
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0
    return "%.1f TB" % n


def fmt_duration(s):
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return "%dh %02dmin %02ds" % (h, m, sec)
    if m:
        return "%dmin %02ds" % (m, sec)
    return "%ds" % sec


def fmt_num(v, nd=2):
    if v is None:
        return "-"
    return ("{:,.%df}" % nd).format(v).replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_int(v):
    if v is None:
        return "-"
    return "{:,}".format(int(v)).replace(",", ".")


def norm_path(p):
    return os.path.normcase(os.path.abspath(os.path.realpath(p)))


def same_file(a, b):
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
    except OSError:
        pass
    return norm_path(a) == norm_path(b)


def file_size_with_sidecars(path):
    """Tamanho do arquivo + sidecars (.ovr, .msk, .aux.xml), se existirem."""
    total = os.path.getsize(path) if os.path.exists(path) else 0
    side = []
    for ext in (".ovr", ".msk", ".aux.xml"):
        p = path + ext
        if os.path.exists(p):
            total += os.path.getsize(p)
            side.append(os.path.basename(p))
    return total, side


def detect_free_ram():
    """RAM disponível em bytes (psutil costuma vir com o QGIS; senão None)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
        except Exception:
            pass
    try:  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def free_disk(path):
    try:
        return shutil.disk_usage(path).free
    except Exception:
        return None


def gtiff_supports_compression(name):
    drv = gdal.GetDriverByName("GTiff")
    if drv is None:
        return False
    md = drv.GetMetadataItem("DMD_CREATIONOPTIONLIST") or ""
    return ("<Value>%s</Value>" % name) in md


def cog_driver_available():
    return gdal.GetDriverByName("COG") is not None


def is_lossy(comp):
    return (comp or "").upper() in LOSSY_CODECS


def codec_label(comp, quality=None):
    comp = (comp or "").upper()
    if comp in LOSSY_CODECS:
        return "%s q%d (com perda)" % (comp, int(quality or DEFAULT_QUALITY))
    return "%s (sem perda)" % comp


def psnr_label(db):
    """Interpretação prática do PSNR (8 bits) para ortomosaico."""
    if db is None:
        return "-"
    if db >= 99:
        return "idêntico"
    if db >= 40:
        return "visualmente idêntico"
    if db >= 36:
        return "perda mínima (só em zoom máximo)"
    if db >= 32:
        return "perda perceptível em zoom"
    return "perda visível"


def compute_psnr(a, b, valid=None):
    """PSNR (dB) entre dois arrays (bandas, y, x), só nos pixels válidos."""
    if valid is not None:
        if not np.any(valid):
            return None
        d = a[:, valid].astype(np.float64) - b[:, valid].astype(np.float64)
    else:
        d = a.astype(np.float64) - b.astype(np.float64)
    if d.size == 0:
        return None
    mse = float(np.mean(d * d))
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(255.0 ** 2 / mse)


def gtiff_creation_options(comp, quality, tile, mode, bigtiff, fast=False):
    """Opções do GeoTIFF de saída (ou do intermediário rápido, fast=True)."""
    co = ["TILED=YES", "BLOCKXSIZE=%d" % tile, "BLOCKYSIZE=%d" % tile,
          "COMPRESS=%s" % comp,
          # PIXEL: R,G,B do mesmo tile ficam juntos -> 1 leitura por tile na
          # renderização RGB.
          "INTERLEAVE=PIXEL",
          "BIGTIFF=%s" % bigtiff,
          "NUM_THREADS=ALL_CPUS"]   # compressão multithread (não é gravado no arquivo)
    if comp == "JPEG":
        # YCbCr (luminância + cor subamostrada 4:2:0) é o JPEG "de foto":
        # 2-3x menor que JPEG em RGB com a mesma qualidade visual.
        co += ["PHOTOMETRIC=YCBCR", "JPEG_QUALITY=%d" % quality]
    elif comp == "WEBP":
        co += ["PHOTOMETRIC=RGB", "WEBP_LEVEL=%d" % quality]
    else:
        # PREDICTOR=2 (diferença horizontal) é o adequado para inteiros 8 bits
        # de imagem natural: vizinhos são parecidos, as diferenças comprimem
        # 10-30% melhor. PREDICTOR=3 é só para ponto flutuante.
        co += ["PHOTOMETRIC=RGB", "PREDICTOR=2"]
        if comp == "ZSTD":
            co.append("ZSTD_LEVEL=%d" % (1 if fast else 9))
        elif comp == "DEFLATE":
            co.append("ZLEVEL=%d" % (1 if fast else 6))
    if mode == "alpha":
        co.append("ALPHA=UNASSOCIATED")
    return co


def overview_config(comp, quality, tile, mode):
    """Opções de configuração GDAL para overviews internas no mesmo formato."""
    cfg = {
        "COMPRESS_OVERVIEW": comp,
        "INTERLEAVE_OVERVIEW": "PIXEL",
        "GDAL_TIFF_OVR_BLOCKSIZE": str(tile),
        "GDAL_NUM_THREADS": "ALL_CPUS",
        "BIGTIFF_OVERVIEW": "IF_SAFER",
    }
    if comp == "JPEG":
        cfg["PHOTOMETRIC_OVERVIEW"] = "YCBCR"
        cfg["JPEG_QUALITY_OVERVIEW"] = str(quality)
    elif comp == "WEBP":
        cfg["WEBP_LEVEL_OVERVIEW"] = str(quality)
    else:
        cfg["PREDICTOR_OVERVIEW"] = "2"
        if comp == "ZSTD":
            cfg["ZSTD_LEVEL_OVERVIEW"] = "9"
        elif comp == "DEFLATE":
            cfg["ZLEVEL_OVERVIEW"] = "6"
    if mode == "mask":
        cfg["GDAL_TIFF_INTERNAL_MASK"] = "YES"   # overviews da máscara também internas
    return cfg


def cog_creation_options(comp, quality, tile, ovr_resampling, with_ovr):
    co = ["COMPRESS=%s" % comp, "BLOCKSIZE=%d" % tile, "BIGTIFF=IF_SAFER",
          "NUM_THREADS=ALL_CPUS", "OVERVIEW_RESAMPLING=%s" % ovr_resampling,
          "OVERVIEWS=%s" % ("AUTO" if with_ovr else "NONE")]
    if comp in LOSSY_CODECS:
        co.append("QUALITY=%d" % quality)   # o driver COG usa YCbCr no JPEG RGB sozinho
    else:
        co.append("PREDICTOR=YES")
        if comp == "ZSTD":
            co.append("LEVEL=9")
        elif comp == "DEFLATE":
            co.append("LEVEL=6")
    return co


def set_traditional_axis(srs):
    """GDAL 3+: força ordem (x=lon/leste, y=lat/norte) nas transformações."""
    if srs is not None and hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def srs_from_wkt(wkt):
    if not wkt:
        return None
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(wkt) != 0:
        return None
    return set_traditional_axis(srs)


def srs_epsg(srs):
    if srs is None:
        return None
    try:
        code = srs.GetAuthorityCode(None)
        if code:
            return int(code)
        clone = srs.Clone()
        if clone.AutoIdentifyEPSG() == 0:
            code = clone.GetAuthorityCode(None)
            if code:
                return int(code)
    except Exception:
        pass
    return None


def same_crs(a, b):
    if a is None or b is None:
        return False
    ea, eb = srs_epsg(a), srs_epsg(b)
    if ea is not None and ea == eb:
        return True
    try:
        return bool(a.IsSame(b))
    except Exception:
        return False


def srs_label(srs):
    if srs is None:
        return "SEM CRS"
    try:
        name = srs.GetName() if hasattr(srs, "GetName") else None
    except Exception:
        name = None
    if not name:
        name = srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS") or "?"
    code = srs_epsg(srs)
    return ("EPSG:%d - %s" % (code, name)) if code else name


def srs_units(srs):
    if srs is None:
        return "?"
    if srs.IsGeographic():
        return "graus"
    try:
        name = srs.GetLinearUnitsName()
        f = srs.GetLinearUnits()
        if abs(f - 1.0) < 1e-9:
            return "m"
        return name or "?"
    except Exception:
        return "?"


def make_transform(src, dst):
    try:
        ct = osr.CoordinateTransformation(src, dst)
    except Exception as e:
        raise ProcessingError("Não foi possível criar a transformação de coordenadas: %s" % e)
    if ct is None:
        raise ProcessingError("Não foi possível criar a transformação de coordenadas "
                              "(verifique a instalação do PROJ).")
    return ct


def transform_xy(ct, pts):
    out = ct.TransformPoints([(float(x), float(y)) for x, y in pts])
    return [(p[0], p[1]) for p in out]


def pixel_to_geo(gt, px, py):
    return (gt[0] + px * gt[1] + py * gt[2], gt[3] + px * gt[4] + py * gt[5])


def geo_to_pixel(gt, x, y):
    det = gt[1] * gt[5] - gt[2] * gt[4]
    if det == 0:
        raise ProcessingError("GeoTransform não inversível.")
    dx, dy = x - gt[0], y - gt[3]
    px = (gt[5] * dx - gt[2] * dy) / det
    py = (-gt[4] * dx + gt[1] * dy) / det
    return px, py


def extent_from_gt(gt, w, h):
    corners = [pixel_to_geo(gt, x, y) for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


class GdalConfig(object):
    """Define opções de configuração GDAL temporariamente e restaura depois.
    Usa SetThreadLocalConfigOption quando disponível para não afetar as
    threads de renderização do QGIS."""

    def __init__(self, options):
        self.options = dict(options)
        self.old = {}
        self.thread_local = hasattr(gdal, "SetThreadLocalConfigOption")

    def _get(self, k):
        if self.thread_local and hasattr(gdal, "GetThreadLocalConfigOption"):
            return gdal.GetThreadLocalConfigOption(k, None)
        return gdal.GetConfigOption(k, None)

    def _set(self, k, v):
        if self.thread_local:
            gdal.SetThreadLocalConfigOption(k, v)
        else:
            gdal.SetConfigOption(k, v)

    def __enter__(self):
        for k, v in self.options.items():
            self.old[k] = self._get(k)
            self._set(k, None if v is None else str(v))
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            self._set(k, v)
        return False


class GdalErrorCollector(object):
    """Captura mensagens de erro/aviso do GDAL (funciona com ou sem
    gdal.UseExceptions(); no modo exceções o erro também vira RuntimeError).
    O handler é por thread: cada thread usa o seu."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def handler(self, err_class, err_no, msg):
        msg = (msg or "").strip()
        if err_class >= gdal.CE_Failure:
            self.errors.append(msg)
        elif err_class == gdal.CE_Warning:
            self.warnings.append(msg)

    def __enter__(self):
        gdal.PushErrorHandler(self.handler)
        return self

    def __exit__(self, *exc):
        gdal.PopErrorHandler()
        return False


def close_ds(ds):
    """Fecha um dataset GDAL de forma explícita (Close() no GDAL >= 3.8)."""
    if ds is None:
        return None
    try:
        ds.FlushCache()
    except Exception:
        pass
    try:
        if hasattr(ds, "Close"):
            ds.Close()
    except Exception:
        pass
    return None


def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
        return True
    except OSError:
        return False


def vsimem_unlink(path):
    try:
        gdal.Unlink(path)
    except Exception:
        pass


class WindowReader(threading.Thread):
    """Thread que reprojeta (lê do VRT warpado) as próximas janelas enquanto a
    thread principal comprime/grava a atual. Cada dataset GDAL é usado por uma
    única thread: este lê o VRT, a principal escreve no GeoTIFF."""

    def __init__(self, ds, windows, depth=PREFETCH_WINDOWS):
        threading.Thread.__init__(self, name="orto_leitor_janelas")
        self.daemon = True
        self.ds = ds
        self.windows = windows
        self.q = queue.Queue(maxsize=max(1, depth))
        self.stop_event = threading.Event()

    def _put(self, item):
        while not self.stop_event.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        try:
            for win in self.windows:
                if self.stop_event.is_set():
                    return
                arr = None
                with GdalErrorCollector() as ec:
                    try:
                        arr = self.ds.ReadAsArray(*win)
                    except Exception as e:   # modo exceções do GDAL
                        ec.errors.append(str(e))
                if arr is None or ec.errors:
                    self._put(("erro", win, "; ".join(ec.errors) or "leitura retornou vazio"))
                    return
                if not self._put(("ok", win, arr)):
                    return
            self._put(("fim", None, None))
        except Exception as e:
            self._put(("erro", None, "%s\n%s" % (e, traceback.format_exc())))

    def get(self):
        while True:
            try:
                return self.q.get(timeout=0.5)
            except queue.Empty:
                if not self.is_alive() and self.q.empty():
                    return ("erro", None, "a thread de leitura terminou inesperadamente")

    def shutdown(self):
        self.stop_event.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        if self.is_alive():
            self.join()


# =============================================================================
#  DESCRIÇÃO / ANÁLISE DE RASTER
# =============================================================================
def compute_overview_levels(width, height, tile):
    """Níveis 2,4,8... até a menor overview caber em ~1 tile.
    Ex.: 35000 px e tile 512 -> 2..128 (128 -> 274 px). Mais níveis só
    acrescentariam overviews minúsculas e inúteis."""
    max_dim = max(width, height)
    levels = []
    if max_dim <= tile:
        return levels
    f = 2
    while True:
        levels.append(f)
        if math.ceil(max_dim / float(f)) <= tile or f >= 4096:
            break
        f *= 2
    return levels


def describe_dataset(ds, path):
    """Metadados (rápido, sem ler pixels)."""
    d = {"path": path}
    d["file_size"], d["sidecars"] = file_size_with_sidecars(path)
    d["driver"] = ds.GetDriver().ShortName if ds.GetDriver() else "?"
    d["width"], d["height"], d["bands"] = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
    d["dtypes"] = [gdal.GetDataTypeName(ds.GetRasterBand(i + 1).DataType) for i in range(ds.RasterCount)]
    gt = ds.GetGeoTransform(can_return_null=True)
    d["gt"] = gt
    d["rotated"] = bool(gt and (abs(gt[2]) > 1e-15 or abs(gt[4]) > 1e-15))
    srs = srs_from_wkt(ds.GetProjection())
    d["srs"] = srs
    d["crs_label"] = srs_label(srs)
    d["epsg"] = srs_epsg(srs)
    d["is_geographic"] = bool(srs is not None and srs.IsGeographic())
    d["units"] = srs_units(srs)
    d["res_x"] = abs(gt[1]) if gt else None
    d["res_y"] = abs(gt[5]) if gt else None
    d["extent"] = extent_from_gt(gt, ds.RasterXSize, ds.RasterYSize) if gt else None
    d["compression"] = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") or "NENHUMA"
    d["predictor"] = ds.GetMetadataItem("PREDICTOR", "IMAGE_STRUCTURE")
    d["interleave"] = ds.GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE") or "?"
    d["layout"] = ds.GetMetadataItem("LAYOUT", "IMAGE_STRUCTURE")
    b1 = ds.GetRasterBand(1) if ds.RasterCount else None
    if b1 is not None:
        bx, by = b1.GetBlockSize()
        d["block"] = (bx, by)
        d["tiled"] = (bx != ds.RasterXSize) or (bx == by)
        d["ovr_factors"] = []
        for i in range(b1.GetOverviewCount()):
            o = b1.GetOverview(i)
            if o is not None and o.XSize > 0:
                d["ovr_factors"].append(int(round(ds.RasterXSize / float(o.XSize))))
        d["mask_flags"] = b1.GetMaskFlags()
    else:
        d["block"], d["tiled"], d["ovr_factors"], d["mask_flags"] = None, False, [], 0
    d["ovr_external"] = os.path.exists(path + ".ovr")
    d["nodata"] = [ds.GetRasterBand(i + 1).GetNoDataValue() for i in range(ds.RasterCount)]
    d["color_interp"] = [gdal.GetColorInterpretationName(ds.GetRasterBand(i + 1).GetColorInterpretation())
                         for i in range(ds.RasterCount)]
    alpha_idx = None
    for i in range(ds.RasterCount):
        if ds.GetRasterBand(i + 1).GetColorInterpretation() == gdal.GCI_AlphaBand:
            alpha_idx = i + 1
            break
    d["alpha_assumed"] = False
    if alpha_idx is None and ds.RasterCount == 4 and \
            ds.GetRasterBand(4).GetColorInterpretation() == gdal.GCI_Undefined:
        # Muitos softwares de fotogrametria gravam RGBA sem marcar a 4a banda.
        alpha_idx = 4
        d["alpha_assumed"] = True
    d["alpha_idx"] = alpha_idx
    d["per_dataset_mask"] = bool(b1 is not None and (d["mask_flags"] & gdal.GMF_PER_DATASET)
                                 and not (d["mask_flags"] & gdal.GMF_ALPHA)
                                 and not (d["mask_flags"] & gdal.GMF_NODATA))
    d["mask_ovr_count"] = 0
    if d["per_dataset_mask"]:
        try:
            d["mask_ovr_count"] = b1.GetMaskBand().GetOverviewCount()
        except Exception:
            pass
    d["stats_valid_percent"] = None
    if b1 is not None:
        v = b1.GetMetadataItem("STATISTICS_VALID_PERCENT")
        if v:
            try:
                d["stats_valid_percent"] = float(v)
            except ValueError:
                pass
    return d


def transparency_label(d):
    if d.get("alpha_idx"):
        return "Alpha (banda %d)" % d["alpha_idx"]
    if d.get("per_dataset_mask"):
        return "máscara interna"
    nd = [v for v in d.get("nodata", []) if v is not None]
    if nd:
        return "NoData=%g" % nd[0]
    return "nenhuma"


def estimate_resolution(ds, src_srs, dst_srs):
    """Estima o tamanho do pixel da entrada nas unidades do CRS de destino,
    medindo, numa grade 5x5 de posições, a distância projetada entre o pixel
    (i,j) e seus vizinhos (i+1,j) e (i,j+1). Retorna medianas (rx, ry)."""
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    ct = make_transform(src_srs, dst_srs)
    rx, ry = [], []
    for fy in (0.1, 0.3, 0.5, 0.7, 0.9):
        for fx in (0.1, 0.3, 0.5, 0.7, 0.9):
            px, py = fx * w, fy * h
            pts = [pixel_to_geo(gt, px, py), pixel_to_geo(gt, px + 1, py), pixel_to_geo(gt, px, py + 1)]
            try:
                p0, p1, p2 = transform_xy(ct, pts)
            except Exception:
                continue
            rx.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            ry.append(math.hypot(p2[0] - p0[0], p2[1] - p0[1]))
    if not rx:
        raise ProcessingError("Falha ao estimar a resolução no CRS de destino "
                              "(a área do raster pode estar fora do domínio do CRS).")
    rx.sort()
    ry.sort()
    return rx[len(rx) // 2], ry[len(ry) // 2]


def transformed_extent(ds, src_srs, dst_srs, densify=21):
    """BBOX da entrada no CRS de destino (bordas densificadas)."""
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    pts = []
    for i in range(densify):
        t = i / float(densify - 1)
        pts += [pixel_to_geo(gt, t * w, 0), pixel_to_geo(gt, t * w, h),
                pixel_to_geo(gt, 0, t * h), pixel_to_geo(gt, w, t * h)]
    out = transform_xy(make_transform(src_srs, dst_srs), pts)
    xs = [p[0] for p in out]
    ys = [p[1] for p in out]
    return (min(xs), min(ys), max(xs), max(ys))


def read_source_window(ds, info, x, y, xs, ys, treat_black):
    """Lê uma janela da ENTRADA em resolução total: (rgb, válido)."""
    a_idx = info["alpha_idx"]
    bands = [1, 2, 3] + ([a_idx] if a_idx else [])
    raw = ds.ReadRaster(x, y, xs, ys, xs, ys, gdal.GDT_Byte, bands)
    if raw is None:
        return None, None
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(len(bands), ys, xs)
    rgb = arr[:3].copy()
    nd = info["nodata"][:3]
    if a_idx:
        valid = arr[3] >= ALPHA_VALID_THRESHOLD
    elif all(v is not None for v in nd):
        ndv = np.array([int(v) for v in nd], dtype=np.int32).reshape(3, 1, 1)
        valid = ~np.all(rgb == ndv, axis=0)
    elif info["per_dataset_mask"]:
        m = ds.GetRasterBand(1).GetMaskBand().ReadRaster(x, y, xs, ys)
        valid = np.frombuffer(m, dtype=np.uint8).reshape(ys, xs) > 0
    elif treat_black:
        valid = ~np.all(rgb == 0, axis=0)
    else:
        valid = np.ones((ys, xs), dtype=bool)
    return rgb, valid


# =============================================================================
#  NÚCLEO DE PROCESSAMENTO
# =============================================================================
class OrthoOptimizer(object):
    """Núcleo independente de interface. Recebe callbacks de log/progresso."""

    def __init__(self, params, log=None, progress=None, is_cancelled=None):
        self.p = dict(params)
        self._log = log or (lambda m: sys.stdout.write(m + "\n"))
        self._progress = progress or (lambda v: None)
        self._is_cancelled = is_cancelled or (lambda: False)
        self._last_pct = -1
        self._comp = None
        self.report_lines = []

    # ------------------------------------------------------------- logging --
    def log(self, msg=""):
        for line in str(msg).split("\n"):
            self.report_lines.append(line)
            self._log(line)

    def section(self, title):
        self.log("")
        self.log("=" * 78)
        self.log(" " + title)
        self.log("=" * 78)

    def progress(self, pct):
        pct = int(max(0, min(100, pct)))
        if pct != self._last_pct:
            self._last_pct = pct
            self._progress(pct)

    def check_cancel(self):
        if self._is_cancelled():
            raise CancelledError("Processamento cancelado pelo usuário.")

    def gdal_callback(self, start, end):
        """Callback de progresso para funções GDAL (retornar 0 aborta)."""
        def cb(complete, message, data):
            self.progress(start + (end - start) * complete)
            return 0 if self._is_cancelled() else 1
        return cb

    # ----------------------------------------------------------- ambiente --
    def log_environment(self):
        self.section("AMBIENTE")
        qv = "?"
        if HAS_QGIS:
            try:
                qv = Qgis.QGIS_VERSION
            except Exception:
                pass
        self.log("QGIS               : %s" % qv)
        self.log("GDAL               : %s (%d)" % (gdal_version_str(), gdal_version_num()))
        self.log("Python             : %s" % sys.version.split()[0])
        self.log("NumPy              : %s" % np.__version__)
        self.log("CPUs               : %s" % (os.cpu_count() or "?"))
        ram = detect_free_ram()
        self.log("RAM disponível     : %s" % (fmt_bytes(ram) if ram else "não detectada"))
        self.log("Compressões GTiff  : %s" % ", ".join(c for c in LOSSLESS_CODECS + LOSSY_CODECS
                                                        if gtiff_supports_compression(c)))
        self.log("Driver COG         : %s" % ("sim" if cog_driver_available() else "não"))
        self.log("Cache GDAL atual   : %s" % fmt_bytes(gdal.GetCacheMax()))
        if gdal_version_num() < 2020000:
            raise ProcessingError("GDAL %s é antigo demais (mínimo 2.2)." % gdal_version_str())

    # ----------------------------------------------------- destino (CRS) --
    def get_dst_srs(self):
        user = (self.p.get("dst_crs") or DEFAULT_DST_CRS).strip()
        srs = osr.SpatialReference()
        m = re.match(r"^\s*EPSG\s*:\s*(\d+)\s*$", user, re.I)
        if m:
            err = srs.ImportFromEPSG(int(m.group(1)))
            if err != 0:
                raise ProcessingError("O CRS %s não está disponível no GDAL/PROJ deste QGIS "
                                      "(ImportFromEPSG retornou %s). Verifique o proj.db." % (user, err))
            dst_str = "EPSG:%s" % m.group(1)
        else:
            if srs.SetFromUserInput(user) != 0:
                raise ProcessingError("CRS de destino inválido: %s" % user[:120])
            dst_str = srs.ExportToWkt()
        set_traditional_axis(srs)
        if not srs.IsProjected():
            raise ProcessingError("O CRS de destino (%s) não é projetado. Use um CRS métrico "
                                  "como EPSG:31983." % srs_label(srs))
        return srs, dst_str

    # --------------------------------------------------------- escolhas --
    def choose_compression(self):
        """ZSTD: descompressão ~2-3x mais rápida que DEFLATE e taxa semelhante
        ou melhor -> ideal para navegação. DEFLATE: fallback universal.
        JPEG/WEBP: com perda controlada (qualidade), com máscara interna."""
        if self._comp is not None:
            return self._comp
        req = (self.p.get("compression") or DEFAULT_COMPRESSION).upper()
        if req == "AUTO":   # compatibilidade com a v1: "automático" = melhor sem perda
            req = "ZSTD"
        if req not in LOSSLESS_CODECS + LOSSY_CODECS:
            raise ProcessingError("Compressão desconhecida: %s" % req)
        comp = req
        if not gtiff_supports_compression(req):
            comp = {"ZSTD": "DEFLATE", "WEBP": "JPEG"}.get(req, "DEFLATE")
            self.log("[!] %s não é suportado por este GDAL; usando %s." % (req, comp))
        self._comp = comp
        return comp

    def quality(self):
        try:
            q = int(self.p.get("quality", DEFAULT_QUALITY))
        except (TypeError, ValueError):
            q = DEFAULT_QUALITY
        return max(1, min(100, q))

    def transparency_mode(self, comp):
        """nodata: RGB + NoData=0 | alpha: RGBA | mask: RGB + máscara interna."""
        if is_lossy(comp):
            return "mask"
        return "nodata" if self.p.get("remove_alpha", True) else "alpha"

    def output_format(self):
        fmt = (self.p.get("output_format") or "GTIFF").upper()
        if fmt == "COG" and not cog_driver_available():
            self.log("[!] Driver COG indisponível neste GDAL (< 3.1); gerando GeoTIFF otimizado.")
            fmt = "GTIFF"
        return fmt

    def is_identity(self, info, dst_srs):
        """Entrada já no CRS de destino, sem rotação e resolução automática:
        mantém a MESMA grade (pixels copiados 1:1, zero reamostragem)."""
        return (same_crs(info["srs"], dst_srs) and not info["rotated"]
                and self.p.get("res_mode", "auto") == "auto")

    def decide_bigtiff(self, w, h, nbands, with_ovr):
        """O limite do TIFF clássico é 4 GiB. Estima o pior caso (sem compressão,
        + 1/3 para overviews). Se passar de 75% do limite, força BIGTIFF=YES;
        caso contrário IF_SAFER (o próprio GDAL decide com sua heurística).
        Assim nunca falha por tamanho e não força BigTIFF à toa."""
        raw = float(w) * h * nbands * (4.0 / 3.0 if with_ovr else 1.0)
        if raw > CLASSIC_TIFF_LIMIT * BIGTIFF_SAFETY:
            return "YES", raw
        return "IF_SAFER", raw

    # ------------------------------------------------- análise de conteúdo --
    def analyze_content(self, ds, info, pstart, pend):
        """Lê o raster em resolução REDUZIDA (máx. ANALYSIS_MAX_DIM px), em faixas,
        para estimar transparência/NoData. Usa overviews se existirem; senão o
        GDAL lê somente as linhas necessárias (nearest). Memória ~20 MB."""
        w, h = ds.RasterXSize, ds.RasterYSize
        scale = max(1.0, max(w, h) / float(ANALYSIS_MAX_DIM))
        bw, bh = max(1, int(round(w / scale))), max(1, int(round(h / scale)))
        alpha_idx = info["alpha_idx"]
        bands = [1, 2, 3] + ([alpha_idx] if alpha_idx else [])
        nb = len(bands)
        counts = dict(total=0, alpha0=0, alpha255=0, alpha_part=0, invalid=0,
                      valid=0, valid_with_zero=0, black=0)
        nd3 = info["nodata"][:3]
        all_nodata = all(v is not None for v in nd3)

        # Lista de faixas (y0, y1, linhas_no_buffer) a ler.
        windows = []
        tile_h = info["block"][1] if info["block"] else 1
        n_trows = int(math.ceil(h / float(tile_h)))
        if info["tiled"] and not info["ovr_factors"] and n_trows > ANALYSIS_TILE_ROWS:
            picks = sorted(set(int(round(k * (n_trows - 1) / float(ANALYSIS_TILE_ROWS - 1)))
                               for k in range(ANALYSIS_TILE_ROWS)))
            for r in picks:
                y0, y1 = r * tile_h, min(h, (r + 1) * tile_h)
                windows.append((y0, y1, max(1, int(round((y1 - y0) / scale)))))
            counts["method"] = "%d de %d linhas de tiles" % (len(picks), n_trows)
        else:
            n_chunks = 16
            for c in range(n_chunks):
                by0 = (bh * c) // n_chunks
                by1 = (bh * (c + 1)) // n_chunks
                y0 = int(round(by0 * h / float(bh)))
                y1 = min(h, int(round(by1 * h / float(bh))))
                if by1 > by0 and y1 > y0:
                    windows.append((y0, y1, by1 - by0))
            counts["method"] = "overviews" if info["ovr_factors"] else "leitura reduzida"
        sample_rows = 0
        for c, (y0, y1, nrows) in enumerate(windows):
            self.check_cancel()
            raw = ds.ReadRaster(0, y0, w, y1 - y0, bw, nrows, gdal.GDT_Byte, bands)
            if raw is None:
                raise ProcessingError("Falha de leitura durante a análise (linhas %d-%d)." % (y0, y1))
            sample_rows += nrows
            a = np.frombuffer(raw, dtype=np.uint8).reshape(nb, nrows, bw)
            rgb = a[:3]
            counts["total"] += a.shape[1] * a.shape[2]
            black = np.all(rgb == 0, axis=0)
            counts["black"] += int(np.count_nonzero(black))
            if alpha_idx:
                al = a[3]
                counts["alpha0"] += int(np.count_nonzero(al == 0))
                counts["alpha255"] += int(np.count_nonzero(al == 255))
                valid = al >= ALPHA_VALID_THRESHOLD
            elif all_nodata:
                ndv = np.array([int(v) for v in nd3], dtype=np.int32).reshape(3, 1, 1)
                valid = ~np.all(rgb == ndv, axis=0)
            elif info["per_dataset_mask"]:
                mraw = ds.GetRasterBand(1).GetMaskBand().ReadRaster(0, y0, w, y1 - y0, bw, nrows,
                                                                    gdal.GDT_Byte)
                valid = np.frombuffer(mraw, dtype=np.uint8).reshape(nrows, bw) > 0
            elif self.p.get("treat_black_as_nodata", True):
                valid = ~black
            else:
                valid = np.ones(black.shape, dtype=bool)
            nv = int(np.count_nonzero(valid))
            counts["valid"] += nv
            counts["invalid"] += valid.size - nv
            counts["valid_with_zero"] += int(np.count_nonzero(valid & np.any(rgb == 0, axis=0)))
            self.progress(pstart + (pend - pstart) * (c + 1) / float(len(windows)))
        if alpha_idx:
            counts["alpha_part"] = counts["total"] - counts["alpha0"] - counts["alpha255"]
        counts["sample_size"] = (bw, sample_rows)
        return counts

    # ------------------------------------------- estimativa de tamanho/PSNR --
    def estimate_candidates(self, comp, quality, full):
        """full=True ("Analisar entrada"): tabela comparativa completa.
        full=False (EXECUTAR): só a escolhida (+ intermediário do COG)."""
        key = (comp, quality if is_lossy(comp) else 0)
        if full:
            cands = [(c, 0) for c in ("ZSTD", "DEFLATE") if gtiff_supports_compression(c)]
            cands += [(c, quality) for c in LOSSY_CODECS if gtiff_supports_compression(c)]
        else:
            cands = []
            if (self.p.get("output_format") or "GTIFF").upper() == "COG":
                cands.append(("ZSTD" if gtiff_supports_compression("ZSTD") else "DEFLATE", 0))
        if key not in cands:
            cands.append(key)
        return cands

    def measure_codecs(self, ds, info, candidates, tile):
        """Comprime DE VERDADE, em memória (/vsimem), blocos reais da imagem com
        cada opção e mede bytes por pixel válido (e PSNR nas com perda). Não
        depende da resolução de saída. Retorna None se a imagem for pequena
        demais ou não tiver blocos com imagem suficiente."""
        w, h = ds.RasterXSize, ds.RasterYSize
        ts = tile
        if w < ts or h < ts:
            return None
        rows, cols = ESTIMATE_GRID
        treat_black = self.p.get("treat_black_as_nodata", True)
        # Posições sempre DESALINHADAS da grade de 16 px (resto 5): uma entrada
        # que já foi JPEG comprime artificialmente bem se o bloco cair alinhado
        # com o JPEG antigo. Desalinhado, a estimativa só pode errar PARA MAIS
        # (seguro para planejar disco) e reproduz o caso comum (reprojeção).
        ypos = sorted(set(int(k * (h - ts - 16) / float(max(1, rows - 1))) // 16 * 16 + 5 for k in range(rows)))
        xpos = sorted(set(int(k * (w - ts - 16) / float(max(1, cols - 1))) // 16 * 16 + 5 for k in range(cols)))
        samples = []
        for y in ypos:
            for x in xpos:
                self.check_cancel()
                rgb, valid = read_source_window(ds, info, x, y, ts, ts, treat_black)
                if rgb is None:
                    continue
                frac = float(np.count_nonzero(valid)) / valid.size
                if frac >= 0.5:
                    samples.append((frac, rgb, valid))
        if not samples:
            return None
        if len(samples) > ESTIMATE_MAX_SAMPLES:   # mantém a distribuição espacial
            pick = np.linspace(0, len(samples) - 1, ESTIMATE_MAX_SAMPLES).round().astype(int)
            samples = [samples[i] for i in sorted(set(pick.tolist()))]
        n = len(samples)
        mc = min(6, n)
        mr = int(math.ceil(n / float(mc)))
        mos = np.zeros((3, mr * ts, mc * ts), dtype=np.uint8)
        mval = np.zeros((mr * ts, mc * ts), dtype=bool)
        for i, (_, rgb, valid) in enumerate(samples):
            r, c = divmod(i, mc)
            mos[:, r * ts:(r + 1) * ts, c * ts:(c + 1) * ts] = rgb
            mval[r * ts:(r + 1) * ts, c * ts:(c + 1) * ts] = valid
        mos[:, ~mval] = 0
        nvalid = int(np.count_nonzero(mval))
        MW, MH = mos.shape[2], mos.shape[1]
        buf = np.ascontiguousarray(mos).tobytes()
        drv = gdal.GetDriverByName("GTiff")
        results = {}
        for comp, q in candidates:
            self.check_cancel()
            path = "/vsimem/orto_estimativa_%d_%s_%d.tif" % (id(self), comp, q)
            try:
                co = gtiff_creation_options(comp, q, ts, "nodata", "NO")
                t = drv.Create(path, MW, MH, 3, gdal.GDT_Byte, options=co)
                if t is None:
                    continue
                t.WriteRaster(0, 0, MW, MH, buf, MW, MH, gdal.GDT_Byte, [1, 2, 3])
                t = close_ds(t)
                st = gdal.VSIStatL(path)
                if st is None or not st.size:
                    continue
                db = None
                if comp in LOSSY_CODECS:
                    t = gdal.Open(path)
                    back = np.frombuffer(t.ReadRaster(0, 0, MW, MH, MW, MH, gdal.GDT_Byte, [1, 2, 3]),
                                         dtype=np.uint8).reshape(3, MH, MW)
                    t = close_ds(t)
                    db = compute_psnr(back, mos, mval)
                results[(comp, q)] = dict(bpvp=st.size / float(max(1, nvalid)), psnr=db)
            except Exception as e:
                self.log("[!] Estimativa %s falhou: %s" % (codec_label(comp, q), e))
            finally:
                vsimem_unlink(path)
        if not results:
            return None
        return dict(results=results, n_samples=n, tile=ts,
                    sample_valid=float(nvalid) / mval.size)

    def estimate_sizes(self, info, res_x, res_y, with_ovr):
        """Tamanhos estimados (bytes) para a resolução de saída dada."""
        est = info.get("estimates")
        if not est or info.get("valid_fraction") is None:
            return {}
        rx_m, ry_m = info["res_m"]
        valid_out = info["valid_fraction"] * info["width"] * info["height"] * (rx_m * ry_m) / (res_x * res_y)
        sizes = {}
        for key, r in est["results"].items():
            f = OVR_SIZE_FACTOR["lossy" if is_lossy(key[0]) else "lossless"] if with_ovr else 1.0
            sizes[key] = r["bpvp"] * valid_out * f
        return sizes

    def log_estimates(self, info, res_x, res_y, chosen, with_ovr, only_chosen=False):
        sizes = self.estimate_sizes(info, res_x, res_y, with_ovr)
        if not sizes:
            self.log("Estimativa de tamanho: indisponível (imagem pequena ou sem área válida suficiente).")
            return sizes
        est = info["estimates"]
        shown = [chosen] if only_chosen and chosen in sizes else list(sizes)
        self.log("")
        self.log("ESTIMATIVA DO ARQUIVO FINAL (compressão real de %d blocos %dx%d da própria imagem%s)"
                 % (est["n_samples"], est["tile"], est["tile"], ", com pirâmides" if with_ovr else ""))
        self.log("  %-22s %16s %10s   %s" % ("Compressão", "Tamanho final", "vs. orig.", "Qualidade"))
        base = float(max(1, info["file_size"]))
        for key in sorted(shown, key=lambda k: (is_lossy(k[0]), -sizes[k])):
            r = est["results"][key]
            if r["psnr"] is None:
                qual = "idêntica (sem perda)"
            else:
                qual = "PSNR %s dB - %s" % (fmt_num(r["psnr"], 1), psnr_label(r["psnr"]))
            pct = (sizes[key] - base) / base * 100.0
            mark = "   <= escolhida" if key == chosen else ""
            self.log("  %-22s %16s %9s%%   %s%s" % (codec_label(*key), "~" + fmt_bytes(sizes[key]),
                     ("+" if pct >= 0 else "") + fmt_num(pct, 0), qual, mark))
        self.log("  (PSNR >= 40 dB: diferença invisível a olho nu. Precisão da estimativa: sem perda "
                 "~1%; com perda costuma ficar dentro de ~10%)")
        return sizes

    # ---------------------------------------------------------- análise --
    def analyze(self, pstart=0, pend=10, full_estimate=False):
        """Análise completa da entrada. Retorna (ds_aberto, info, counts)."""
        path = self.p["input"]
        if not path or not os.path.isfile(path):
            raise ProcessingError("Arquivo de entrada não encontrado:\n%s" % path)
        open_opts = []
        try:
            drv = gdal.IdentifyDriver(path)
            if drv is not None and drv.ShortName == "GTiff" and gdal_version_num() >= 3060000:
                open_opts = ["NUM_THREADS=ALL_CPUS"]   # decodificação multithread (~10% mais rápido)
        except Exception:
            pass
        ds = gdal.OpenEx(path, gdal.OF_RASTER | gdal.OF_READONLY, open_options=open_opts)
        if ds is None:
            raise ProcessingError("O GDAL não conseguiu abrir o arquivo de entrada:\n%s" % path)
        info = describe_dataset(ds, path)

        # --- validações bloqueantes ---
        if info["bands"] < 3:
            raise ProcessingError("A entrada tem %d banda(s). É necessário RGB (>= 3 bandas)." % info["bands"])
        if any(t != "Byte" for t in info["dtypes"][:3]):
            raise ProcessingError("Tipo de dado %s não suportado: este otimizador é para "
                                  "ortomosaicos RGB 8 bits (Byte)." % info["dtypes"][0])
        if info["srs"] is None:
            raise ProcessingError("A entrada não possui CRS definido. Não é possível reprojetar.")
        if info["gt"] is None:
            raise ProcessingError("A entrada não possui georreferenciamento (GeoTransform).")

        dst_srs, _ = self.get_dst_srs()
        rx_m, ry_m = estimate_resolution(ds, info["srs"], dst_srs)
        info["res_m"] = (rx_m, ry_m)
        info["dst_units"] = srs_units(dst_srs)
        comp = self.choose_compression()
        quality = self.quality()
        mode = self.transparency_mode(comp)
        identity = self.is_identity(info, dst_srs)

        self.section("ANÁLISE DO ARQUIVO DE ENTRADA")
        self.log("Arquivo            : %s" % path)
        self.log("Tamanho            : %s%s" % (fmt_bytes(info["file_size"]),
                 ("  (inclui %s)" % ", ".join(info["sidecars"])) if info["sidecars"] else ""))
        self.log("Driver             : %s" % info["driver"])
        self.log("Dimensões          : %s x %s px" % (fmt_int(info["width"]), fmt_int(info["height"])))
        self.log("Bandas             : %d  (%s)" % (info["bands"], ", ".join(info["color_interp"])))
        self.log("Tipo               : %s" % ", ".join(sorted(set(info["dtypes"]))))
        self.log("CRS                : %s" % info["crs_label"])
        self.log("Resolução nativa   : %.10g x %.10g %s" % (info["res_x"], info["res_y"], info["units"]))
        self.log("Resolução estimada : %.4f x %.4f %s  (no CRS de destino)" % (rx_m, ry_m, info["dst_units"]))
        e = info["extent"]
        self.log("Extensão           : xmin=%.8f ymin=%.8f xmax=%.8f ymax=%.8f" % e)
        self.log("Compressão         : %s%s" % (info["compression"],
                 (" (PREDICTOR=%s)" % info["predictor"]) if info["predictor"] else ""))
        self.log("Interleave         : %s" % info["interleave"])
        self.log("Organização        : %s, bloco %dx%d" % ("TILED" if info["tiled"] else "STRIPS (faixas)",
                 info["block"][0], info["block"][1]))
        self.log("Overviews          : %s%s" % (", ".join(str(f) for f in info["ovr_factors"]) or "NENHUMA",
                 " (externas .ovr)" if info["ovr_external"] else ""))
        self.log("NoData             : %s" % ", ".join("-" if v is None else ("%g" % v) for v in info["nodata"]))
        if info["alpha_idx"]:
            self.log("Alpha              : banda %d%s" % (info["alpha_idx"],
                     " (assumida: 4a banda sem interpretação de cor)" if info["alpha_assumed"] else ""))
        else:
            self.log("Alpha              : não")
        self.log("Máscara por dataset: %s" % ("sim" if info["per_dataset_mask"] else "não"))
        if info["stats_valid_percent"] is not None:
            self.log("STATISTICS_VALID_PERCENT (metadado): %s%%" % fmt_num(info["stats_valid_percent"], 2))
        if info["rotated"]:
            self.log("[!] GeoTransform com rotação: será removida pela reprojeção.")

        # --- conteúdo (amostragem) ---
        self.log("")
        self.log("Analisando conteúdo (amostra reduzida, sem carregar o raster inteiro)...")
        t0 = time.time()
        st = os.stat(path)
        key = (norm_path(path), st.st_size, int(st.st_mtime),
               bool(self.p.get("treat_black_as_nodata", True)), SCRIPT_VERSION)
        cached = self.p.get("cached_content") or {}
        if cached.get("key") == key:
            counts = cached["counts"]
            reused = True
        else:
            counts = self.analyze_content(ds, info, pstart, pstart + (pend - pstart) * 0.6)
            reused = False
        tot = float(max(1, counts["total"]))
        self.log("Amostra            : %d x %d px, %s (%s)" % (counts["sample_size"][0],
                 counts["sample_size"][1], counts.get("method", "?"),
                 "reaproveitada da análise anterior" if reused else "%.1fs" % (time.time() - t0)))
        if info["alpha_idx"]:
            self.log("Alpha = 0          : %s%%  (transparente / sem cobertura)" % fmt_num(100 * counts["alpha0"] / tot))
            self.log("Alpha = 255        : %s%%  (opaco)" % fmt_num(100 * counts["alpha255"] / tot))
            self.log("Alpha parcial      : %s%%  (1..254, bordas)" % fmt_num(100 * counts["alpha_part"] / tot))
            if counts["alpha0"] + counts["alpha_part"] == 0:
                self.log("=> Alpha é constante 255: não carrega transparência; remoção é trivialmente segura.")
            else:
                self.log("=> Alpha CONTÉM transparência real (áreas sem cobertura do voo).")
                if mode == "nodata":
                    self.log("   Ao remover a Alpha, essas áreas serão codificadas como NoData=0 (não viram RGB válido).")
                elif mode == "mask":
                    self.log("   Essas áreas irão para a máscara interna de 1 bit (borda exata, sem perda).")
                else:
                    self.log("   A Alpha será preservada (modo RGBA).")
        self.log("Pixels válidos     : %s%%" % fmt_num(100 * counts["valid"] / tot))
        self.log("Pixels NoData/vazio: %s%%" % fmt_num(100 * counts["invalid"] / tot))
        self.log("Pixels 0,0,0       : %s%% (total da amostra)" % fmt_num(100 * counts["black"] / tot))
        if mode == "nodata":
            vwz = 100.0 * counts["valid_with_zero"] / max(1, counts["valid"])
            self.log("Válidos c/ canal 0 : %s%% dos válidos (serão ajustados 0->1 no modo RGB+NoData)" % fmt_num(vwz, 3))
        info["valid_fraction"] = counts["valid"] / tot

        # --- estimativa de tamanho (compressão real de blocos) ---
        if self.p.get("estimate", True):
            tile = int(self.p.get("tile", DEFAULT_TILE))
            cands = self.estimate_candidates(comp, quality, full_estimate)
            ekey = (key, tile)
            prev = cached.get("estimates") if cached.get("est_key") == ekey else None
            if prev and all(c in prev["results"] for c in cands):
                info["estimates"] = prev   # reaproveita a análise anterior
            else:
                t1 = time.time()
                info["estimates"] = self.measure_codecs(ds, info, cands, tile)
                if info["estimates"]:
                    self.log("Teste de compressão: %d blocos reais comprimidos em %.1fs"
                             % (info["estimates"]["n_samples"], time.time() - t1))
            self.content_cache = {"key": key, "counts": counts, "est_key": ekey,
                                  "estimates": info.get("estimates")}
        else:
            self.content_cache = {"key": key, "counts": counts}
        self.progress(pend)

        # --- problemas e recomendações ---
        problems, recs = [], []
        if not info["ovr_factors"]:
            problems.append("Raster sem overviews (pirâmides): o QGIS lê a resolução total a cada zoom out")
            recs.append("Criar overviews internas (AVERAGE)")
        if not info["tiled"]:
            problems.append("Raster não tiled (strips de %dx%d): leitura aleatória ineficiente no pan/zoom"
                            % tuple(info["block"]))
            recs.append("Criar GeoTIFF tiled %dx%d" % (self.p.get("tile", DEFAULT_TILE), self.p.get("tile", DEFAULT_TILE)))
        if info["is_geographic"]:
            problems.append("CRS geográfico %s (graus): exige reprojeção em tempo real num projeto métrico"
                            % info["crs_label"])
        if identity:
            recs.append("Entrada já está em %s: pixels copiados 1:1, SEM reamostragem" % srs_label(dst_srs))
        elif same_crs(info["srs"], dst_srs):
            recs.append("Mesmo CRS, mas %s: haverá reamostragem"
                        % ("GeoTransform rotacionado" if info["rotated"] else "resolução personalizada"))
        else:
            recs.append("Reprojetar (warp real dos pixels) para %s" % srs_label(dst_srs))
        if counts["invalid"] / tot > 0.2:
            problems.append("Raster possui grande percentual de NoData/transparência (%s%%)"
                            % fmt_num(100 * counts["invalid"] / tot))
        problems.append("Compressão atual: %s%s" % (info["compression"],
                        "" if info["predictor"] else " (sem PREDICTOR informado)"))
        if info["alpha_idx"]:
            problems.append("%d bandas incluindo Alpha (banda %d)" % (info["bands"], info["alpha_idx"]))
        if info["bands"] > 4 or (info["bands"] == 4 and not info["alpha_idx"]):
            problems.append("Bandas extras além de RGB: somente 1,2,3 serão usadas")
        if mode == "nodata":
            recs.append("Gerar RGB de 3 bandas com NoData=0 (Alpha removida, transparência preservada)")
        elif mode == "alpha":
            recs.append("Preservar Alpha (RGBA 4 bandas)")
        else:
            recs.append("Gerar RGB de 3 bandas + máscara interna 1 bit (transparência exata)")
        if is_lossy(comp):
            recs.append("Compressão %s%s" % (codec_label(comp, quality),
                        " em YCbCr" if comp == "JPEG" else ""))
        else:
            recs.append("Compressão sem perda %s + PREDICTOR=2" % comp)
        self.log("")
        self.log("PROBLEMAS DETECTADOS")
        for pr in problems:
            self.log("  [!] " + pr)
        self.log("")
        self.log("RECOMENDAÇÃO")
        for r in recs:
            self.log("  [+] " + r)
        return ds, info, counts

    def auto_resolution(self, info, dst_srs):
        """(res_x, res_y, nota) da resolução automática."""
        if self.is_identity(info, dst_srs):
            return info["res_x"], info["res_y"], "idêntica à original (cópia 1:1, sem reamostragem)"
        rx_m, ry_m = info["res_m"]
        # Menor dimensão do pixel original, truncada em 0,1 mm (nunca mais
        # grossa que o original -> sem perda de resolução).
        res = math.floor(min(rx_m, ry_m) * 10000.0) / 10000.0
        if res <= 0:
            res = min(rx_m, ry_m)
        return res, res, "automática (preserva a original)"

    # ------------------------------------------------------- execução --
    def run(self):
        """Pipeline completo. Nunca lança exceção: retorna dict de resultado."""
        t_start = time.time()
        result = {"ok": False, "output": None, "error": None, "temp_left": [], "report": None,
                  "in_size": None, "out_size": None, "psnr": None}
        params = self.p
        tmpdir = None
        partial = None
        inter_path = None
        out_path = None
        final_written = False
        old_cache = gdal.GetCacheMax()
        src_ds = src_vrt = warped = out_ds = None
        reader = None
        try:
            self.log("%s v%s - início: %s" % (SCRIPT_NAME, SCRIPT_VERSION,
                                               datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            self.log_environment()
            self.progress(0)

            # ---------------------------------------------- caminhos/segurança
            in_path = os.path.abspath(params["input"])
            out_path = os.path.abspath(params["output"])
            self.p["input"] = in_path
            if same_file(in_path, out_path):
                raise ProcessingError("O arquivo de saída é o próprio arquivo de entrada. "
                                      "O original NUNCA é sobrescrito. Escolha outro nome.")
            out_dir = os.path.dirname(out_path)
            if not os.path.isdir(out_dir):
                raise ProcessingError("Pasta de saída inexistente: %s" % out_dir)
            if not os.access(out_dir, os.W_OK):
                raise ProcessingError("Sem permissão de escrita na pasta de saída: %s" % out_dir)
            if os.path.exists(out_path) and not params.get("overwrite", False):
                raise ProcessingError("O arquivo de saída já existe e a substituição não foi autorizada.")

            # Escreve num arquivo parcial na MESMA pasta (rename atômico ao final).
            stem = os.path.splitext(os.path.basename(out_path))[0]
            partial = os.path.join(out_dir, stem + ".__em_processamento__.tif")
            if same_file(partial, in_path):
                raise ProcessingError("Conflito de nomes com o arquivo de entrada.")
            safe_remove(partial)
            tmp_base = params.get("temp_dir") or None
            if tmp_base and not os.path.isdir(tmp_base):
                raise ProcessingError("Pasta temporária inexistente: %s" % tmp_base)
            tmpdir = tempfile.mkdtemp(prefix="orto_otim_", dir=tmp_base)
            self.log("Pasta temporária   : %s" % tmpdir)
            self.log("Arquivo parcial    : %s" % partial)

            # Cache de blocos do GDAL: aumenta temporariamente (restaurado no finally).
            # Basta caber a faixa de origem de uma linha de tiles (~60 MB para
            # 26.000 px de largura RGBA) + tiles de saída; 15% da RAM livre,
            # entre 256 MB e 1 GB, é folgado sem pressionar máquinas modestas.
            ram = detect_free_ram()
            want = 512 * 1024 ** 2 if not ram else int(min(1024 ** 3, max(256 * 1024 ** 2, ram * 0.15)))
            if want > old_cache:
                gdal.SetCacheMax(want)
                self.log("Cache GDAL         : %s (temporário)" % fmt_bytes(want))

            # ------------------------------------------------------- análise
            src_ds, info, counts = self.analyze(2, 12)
            result["in_size"] = info["file_size"]
            self.check_cancel()
            dst_srs, dst_str = self.get_dst_srs()
            comp = self.choose_compression()
            quality = self.quality()
            mode = self.transparency_mode(comp)
            fmt = self.output_format()
            tile = int(params.get("tile", DEFAULT_TILE))
            with_ovr = params.get("overviews", True)
            ovr_res = params.get("ovr_resampling", "AVERAGE")
            identity = self.is_identity(info, dst_srs)

            # ------------------------------------------------- resolução alvo
            rx_m, ry_m = info["res_m"]
            if params.get("res_mode", "auto") == "auto":
                res_x, res_y, res_note = self.auto_resolution(info, dst_srs)
            else:
                res_x = res_y = float(params["res_value"])
                if res_x <= 0:
                    raise ProcessingError("Resolução personalizada inválida.")
                res_note = "personalizada"
                if res_x > 1.5 * max(rx_m, ry_m):
                    self.log("[!] Resolução escolhida é %.1fx mais grossa que a original." % (res_x / max(rx_m, ry_m)))
                if res_x < 0.5 * min(rx_m, ry_m):
                    self.log("[!] Resolução escolhida é bem mais fina que a original: arquivo muito maior sem ganho real.")
            res = res_x
            sizes = self.log_estimates(info, res_x, res_y, (comp, quality if is_lossy(comp) else 0), with_ovr,
                                       only_chosen=True)

            # ----------------------------------------------- VRT de origem
            # Seleciona exatamente R,G,B (+Alpha como ÚLTIMA banda, exigência do
            # -srcalpha do warp) sem copiar pixels.
            self.section("REPROJEÇÃO (gdal.Warp -> VRT, processado bloco a bloco)")
            alpha_idx = info["alpha_idx"]
            band_list = [1, 2, 3] + ([alpha_idx] if alpha_idx else [])
            src_vrt_path = os.path.join(tmpdir, "origem.vrt")
            src_vrt = gdal.Translate(src_vrt_path, src_ds,
                                     options=gdal.TranslateOptions(format="VRT", bandList=band_list))
            if src_vrt is None:
                raise ProcessingError("Falha ao criar VRT de origem.")
            for i, ci in enumerate((gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand)):
                src_vrt.GetRasterBand(i + 1).SetColorInterpretation(ci)
            if alpha_idx:
                src_vrt.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

            # Como o warp deve saber o que é "sem dado" na ENTRADA:
            warp_opts = ["NUM_THREADS=ALL_CPUS", "INIT_DEST=0"]
            src_nodata = None
            if alpha_idx:
                src_nodata = "None"   # Alpha manda; ignora NoData de banda (evitaria buracos em R=0)
                how = "banda Alpha da entrada (-srcalpha)"
            else:
                nd = info["nodata"][:3]
                if all(v is not None for v in nd):
                    src_nodata = " ".join("%g" % v for v in nd)
                    warp_opts.append("UNIFIED_SRC_NODATA=YES")  # vazio só se R=G=B=NoData
                    how = "NoData da entrada (%s), unificado entre bandas" % src_nodata
                elif info["per_dataset_mask"]:
                    how = "máscara interna/externa da entrada"
                elif params.get("treat_black_as_nodata", True):
                    src_nodata = "0 0 0"
                    warp_opts.append("UNIFIED_SRC_NODATA=YES")
                    how = "pixels 0,0,0 tratados como vazio (opção do usuário)"
                else:
                    how = "nenhuma (somente a área fora do retângulo original será vazia)"
            self.log("Validade da entrada: %s" % how)

            warped_path = os.path.join(tmpdir, "reprojetado.vrt")
            common = dict(format="VRT", dstSRS=dst_str, srcAlpha=bool(alpha_idx), dstAlpha=True,
                          srcNodata=src_nodata, multithread=True, warpOptions=warp_opts,
                          creationOptions=["BLOCKXSIZE=%d" % tile, "BLOCKYSIZE=%d" % tile])
            if identity:
                # Mesma grade da entrada (mesma origem, resolução e tamanho) com
                # vizinho mais próximo: cada pixel de saída É o pixel de entrada.
                wo = gdal.WarpOptions(outputBounds=info["extent"], width=info["width"],
                                      height=info["height"], resampleAlg="near",
                                      errorThreshold=0, **common)
            else:
                wo = gdal.WarpOptions(xRes=res_x, yRes=res_y,
                                      targetAlignedPixels=True,          # grade alinhada a múltiplos da resolução
                                      resampleAlg=params.get("resampling", "cubic"),
                                      errorThreshold=0.125, **common)
            with GdalErrorCollector() as ec:
                warped = gdal.Warp(warped_path, src_vrt, options=wo)
            if warped is None:
                raise ProcessingError("gdal.Warp falhou: %s" % ("; ".join(ec.errors) or "erro desconhecido"))
            W, H = warped.RasterXSize, warped.RasterYSize
            wgt = warped.GetGeoTransform()
            if identity:
                igt = info["gt"]
                if (W, H) != (info["width"], info["height"]) or \
                        max(abs(a - b) for a, b in zip(wgt, igt)) > 1e-6 * max(res_x, res_y):
                    raise ProcessingError("Falha interna: a grade de saída não coincide com a de entrada "
                                          "(%s x %s, %s) x (%s x %s, %s)." % (W, H, wgt, info["width"],
                                                                              info["height"], igt))
            out_ext = extent_from_gt(wgt, W, H)
            self.log("CRS original       : %s" % info["crs_label"])
            self.log("CRS final          : %s" % srs_label(dst_srs))
            self.log("Resolução original : %.10g x %.10g %s" % (info["res_x"], info["res_y"], info["units"]))
            self.log("Resolução estimada : %.4f x %.4f %s" % (rx_m, ry_m, info["dst_units"]))
            self.log("Resolução de saída : %.10g x %.10g %s  [%s]" % (res_x, res_y, info["dst_units"], res_note))
            self.log("Reamostragem warp  : %s" % ("nenhuma (cópia 1:1 da grade original)" if identity
                                                  else params.get("resampling", "cubic")))
            self.log("Dimensão final     : %s x %s px" % (fmt_int(W), fmt_int(H)))
            self.log("Extensão final     : xmin=%.3f ymin=%.3f xmax=%.3f ymax=%.3f" % out_ext)
            px_ratio = (W * H) / float(info["width"] * info["height"])
            self.log("Pixels final/orig. : %.2fx (cantos vazios da rotação/reprojeção incluídos)" % px_ratio)
            if px_ratio > 6:
                raise ProcessingError("A saída teria %.1fx mais pixels que a entrada. Verifique a resolução "
                                      "escolhida e o CRS de destino." % px_ratio)

            # ------------------------------------------- opções do GeoTIFF
            nb_out = 4 if mode == "alpha" else 3
            bigtiff, raw = self.decide_bigtiff(W, H, nb_out, with_ovr)
            fast_codec = "ZSTD" if gtiff_supports_compression("ZSTD") else "DEFLATE"
            if fmt == "COG":
                inter_path = os.path.join(tmpdir, "intermediario_sem_perda.tif")
                target = inter_path
                co = gtiff_creation_options(fast_codec, 0, tile, mode, bigtiff, fast=True)
            else:
                target = partial
                co = gtiff_creation_options(comp, quality, tile, mode, bigtiff)
            self.log("")
            self.log("Formato de saída   : %s" % ("COG (Cloud Optimized GeoTIFF)" if fmt == "COG"
                                                  else "GeoTIFF otimizado (tiled + pirâmides internas)"))
            self.log("Bandas de saída    : %d (%s)" % (nb_out, {"nodata": "R,G,B + NoData=0",
                                                              "alpha": "R,G,B,Alpha",
                                                              "mask": "R,G,B + máscara interna 1 bit"}[mode]))
            self.log("Tiling             : %dx%d" % (tile, tile))
            if is_lossy(comp):
                self.log("Compressão         : %s%s (com perda controlada)" % (
                    codec_label(comp, quality).split(" (")[0], " YCbCr" if comp == "JPEG" else ""))
            else:
                self.log("Compressão         : %s + PREDICTOR=2 (sem perda)" % comp)
            if fmt == "COG":
                self.log("Intermediário      : %s nível 1 (rápido, sem perda) -> driver COG" % fast_codec)
            self.log("BIGTIFF            : %s (pior caso sem compressão: %s)" % (bigtiff, fmt_bytes(raw)))

            # Espaço em disco (usa a estimativa por compressão real quando existe).
            key = (comp, quality if is_lossy(comp) else 0)
            est_final = sizes.get(key)
            if est_final is None:
                est_final = info["file_size"] * px_ratio * (nb_out / float(info["bands"])) * (1.34 if with_ovr else 1.0)
            free = free_disk(out_dir)
            if free is not None:
                self.log("Espaço livre       : %s na saída (estimativa do arquivo final: ~%s)"
                         % (fmt_bytes(free), fmt_bytes(est_final)))
                if free < est_final * 1.2:
                    self.log("[!] ATENÇÃO: pouco espaço livre em disco para a saída estimada.")
            if fmt == "COG":
                inter_key = (fast_codec, 0)
                est_inter = sizes.get(inter_key, est_final * 3.0)
                est_inter = est_inter / OVR_SIZE_FACTOR["lossless"] * 1.15   # sem overviews, nível rápido
                free_t = free_disk(tmpdir)
                if free_t is not None:
                    self.log("Espaço temporário  : %s livres na pasta temporária (intermediário: ~%s)"
                             % (fmt_bytes(free_t), fmt_bytes(est_inter)))
                    if free_t < est_inter * 1.2:
                        self.log("[!] ATENÇÃO: pouco espaço na pasta temporária para o intermediário do COG.")

            # ------------------------------------------ criação + laço blocos
            self.section("GRAVAÇÃO (janelas alinhadas aos tiles, leitura e gravação em paralelo)")
            drv = gdal.GetDriverByName("GTiff")
            mask_cfg = {"GDAL_TIFF_INTERNAL_MASK": "YES"} if mode == "mask" else {}
            stats = dict(valid=0, remapped=0, total=0, empty_windows=0)
            p0, p1 = 12, (72 if with_ovr else 90)
            with GdalConfig(mask_cfg):
                with GdalErrorCollector() as ec:
                    out_ds = drv.Create(target, W, H, nb_out, gdal.GDT_Byte, options=co)
                if out_ds is None:
                    raise ProcessingError("Falha ao criar o GeoTIFF de saída: %s" % "; ".join(ec.errors))
                out_ds.SetGeoTransform(wgt)
                out_ds.SetProjection(warped.GetProjection())
                for i, ci in enumerate((gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand)):
                    out_ds.GetRasterBand(i + 1).SetColorInterpretation(ci)
                    if mode == "nodata":
                        out_ds.GetRasterBand(i + 1).SetNoDataValue(0)
                if mode == "alpha":
                    out_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
                mband = None
                if mode == "mask":
                    if out_ds.CreateMaskBand(gdal.GMF_PER_DATASET) not in (0, None):
                        raise ProcessingError("Falha ao criar a máscara interna.")
                    mband = out_ds.GetRasterBand(1).GetMaskBand()
                out_ds.SetMetadataItem("TIFFTAG_SOFTWARE", "%s %s / GDAL %s" % (SCRIPT_NAME, SCRIPT_VERSION,
                                                                                gdal_version_str()))

                win_w = tile * WINDOW_TILES_X
                windows = [(tx, ty, min(win_w, W - tx), min(tile, H - ty))
                           for ty in range(0, H, tile) for tx in range(0, W, win_w)]
                n_win = len(windows)
                n_rows = int(math.ceil(H / float(tile)))
                self.log("Janelas            : %d (%d linhas de tiles x colunas de até %d px)" % (n_win, n_rows, win_w))
                self.log("Memória            : ~%s (janela atual + %d reprojetadas à frente)"
                         % (fmt_bytes(tile * win_w * 10 * (1 + PREFETCH_WINDOWS)), PREFETCH_WINDOWS))
                t_w = time.time()
                reader = WindowReader(warped, windows)
                reader.start()
                log_every = max(1, n_rows // 10)
                for done in range(1, n_win + 1):
                    self.check_cancel()
                    kind, win, arr = reader.get()
                    if kind == "erro":
                        raise ProcessingError("Erro ao reprojetar a janela %s: %s" % (win, arr))
                    if kind != "ok":
                        raise ProcessingError("Leitura terminou antes do esperado (%d de %d janelas)."
                                              % (done - 1, n_win))
                    tx, ty, xs, ys = win
                    alpha = arr[3]
                    rgb = arr[:3]
                    stats["total"] += xs * ys
                    if mode == "alpha":
                        stats["valid"] += int(np.count_nonzero(alpha))
                        buf = np.ascontiguousarray(arr).tobytes()
                        err = out_ds.WriteRaster(tx, ty, xs, ys, buf, xs, ys, gdal.GDT_Byte, [1, 2, 3, 4])
                    else:
                        valid = alpha >= ALPHA_VALID_THRESHOLD
                        nvalid = int(np.count_nonzero(valid))
                        stats["valid"] += nvalid
                        if nvalid == 0:
                            rgb[...] = 0
                            stats["empty_windows"] += 1
                        else:
                            if mode == "nodata":
                                # Conta e corrige pixels válidos com algum canal = 0.
                                stats["remapped"] += int(np.count_nonzero(valid & np.any(rgb == 0, axis=0)))
                                np.maximum(rgb, 1, out=rgb, where=valid)   # válido: 0 -> 1
                            rgb[:, ~valid] = 0                              # inválido: 0
                        buf = np.ascontiguousarray(rgb).tobytes()
                        err = out_ds.WriteRaster(tx, ty, xs, ys, buf, xs, ys, gdal.GDT_Byte, [1, 2, 3])
                        if mband is not None and err in (None, 0):
                            mbuf = (valid.view(np.uint8) * np.uint8(255)).tobytes()
                            err = mband.WriteRaster(tx, ty, xs, ys, mbuf)
                    if err not in (None, 0):
                        raise ProcessingError("Erro de escrita na janela x=%d y=%d (disco cheio?)." % (tx, ty))
                    del arr, rgb, alpha, buf
                    self.progress(p0 + (p1 - p0) * done / float(n_win))
                    if tx + xs >= W:   # fim de uma linha de tiles
                        row = ty // tile
                        if row % log_every == 0 or row == n_rows - 1:
                            el = time.time() - t_w
                            eta = el / done * (n_win - done)
                            self.log("  linha de tiles %d/%d  (%.0f%%)  decorrido %s  restante ~%s"
                                     % (row + 1, n_rows, 100.0 * done / n_win, fmt_duration(el), fmt_duration(eta)))
                reader.shutdown()
                reader = None
                self.log("Fechando arquivo (compressão dos últimos tiles)...")
                mband = None
                out_ds = close_ds(out_ds)
            # Libera o warp e a origem (a validação reabre o VRT pelo caminho).
            warped = close_ds(warped)
            src_vrt = close_ds(src_vrt)
            tot = float(max(1, stats["total"]))
            self.log("Gravação concluída em %s" % fmt_duration(time.time() - t_w))
            self.log("Pixels válidos     : %s%% da grade final" % fmt_num(100 * stats["valid"] / tot))
            if mode == "nodata":
                self.log("Ajustados 0->1     : %s pixels (%s%% dos válidos) - preservam-se como válidos"
                         % (fmt_int(stats["remapped"]), fmt_num(100.0 * stats["remapped"] / max(1, stats["valid"]), 4)))
            if mode != "alpha":
                self.log("Janelas vazias     : %d" % stats["empty_windows"])
            self.check_cancel()

            # ------------------------------------------------------ overviews
            levels = compute_overview_levels(W, H, tile) if with_ovr else []
            if fmt == "GTIFF" and with_ovr and levels:
                self.section("OVERVIEWS / PIRÂMIDES INTERNAS")
                self.log("Níveis             : %s" % ", ".join(str(lv) for lv in levels))
                for lv in levels:
                    self.log("   1:%-4d -> %s x %s px  (%.3f %s/px)" % (lv, fmt_int(math.ceil(W / float(lv))),
                             fmt_int(math.ceil(H / float(lv))), res * lv, info["dst_units"]))
                self.log("Reamostragem       : %s %s" % (ovr_res,
                         "(média ignorando NoData/Alpha/máscara - recomendado)" if ovr_res == "AVERAGE" else ""))
                if ovr_res in ("CUBIC", "BILINEAR") and mode == "nodata":
                    self.log("[!] %s pode gerar valor 0 em pixel válido junto à borda (pequenos furos "
                             "transparentes em zoom out). AVERAGE é o mais seguro." % ovr_res)
                if mode == "alpha":
                    self.log("RGBA: o GDAL usa a Alpha como máscara ao reduzir R,G,B (bordas sem halo preto).")
                elif mode == "mask":
                    self.log("Máscara: o GDAL reduz R,G,B só com pixels válidos e gera pirâmides da máscara.")
                t_o = time.time()
                with GdalConfig(overview_config(comp, quality, tile, mode)):
                    ods = gdal.OpenEx(partial, gdal.OF_RASTER | gdal.OF_UPDATE, open_options=["NUM_THREADS=ALL_CPUS"])
                    if ods is None:
                        raise ProcessingError("Não foi possível reabrir o arquivo para criar overviews.")
                    try:
                        with GdalErrorCollector() as ec:
                            err = ods.BuildOverviews(ovr_res, levels, callback=self.gdal_callback(p1, 92))
                    finally:
                        ods = close_ds(ods)
                self.check_cancel()
                if err not in (None, 0) or ec.errors:
                    raise ProcessingError("Falha ao criar overviews: %s" % ("; ".join(ec.errors) or "código %s" % err))
                self.log("Overviews criadas em %s" % fmt_duration(time.time() - t_o))
            elif fmt == "GTIFF" and with_ovr:
                self.log("Raster pequeno: overviews desnecessárias.")

            # ------------------------------------------------------------ COG
            if fmt == "COG":
                self.section("CONVERSÃO PARA COG (Cloud Optimized GeoTIFF)")
                cco = cog_creation_options(comp, quality, tile, ovr_res, with_ovr)
                self.log("Opções             : %s" % " ".join(cco))
                t_c = time.time()
                # CPL_TMPDIR: o driver COG grava as pirâmides temporárias na pasta temporária.
                with GdalConfig({"CPL_TMPDIR": tmpdir, "GDAL_NUM_THREADS": "ALL_CPUS"}):
                    with GdalErrorCollector() as ec:
                        try:
                            cog = gdal.Translate(partial, inter_path, options=gdal.TranslateOptions(
                                format="COG", creationOptions=cco, callback=self.gdal_callback(p1, 92)))
                        except RuntimeError as e:
                            cog = None
                            ec.errors.append(str(e))
                self.check_cancel()
                if cog is None:
                    raise ProcessingError("Falha ao gerar o COG: %s" % ("; ".join(ec.errors) or "erro desconhecido"))
                cog = close_ds(cog)
                safe_remove(inter_path)   # libera o espaço temporário já
                self.log("COG gerado em %s" % fmt_duration(time.time() - t_c))
            self.progress(92)

            # ------------------------------------------------------ validação
            out_info = None
            if params.get("validate", True):
                out_info, ok, issues = self.validate(
                    partial, dst_srs, nb_out, mode, tile, levels, info, counts, 92, 99,
                    comp=comp, quality=quality, fmt=fmt, ref_path=warped_path)
                if not ok:
                    raise ProcessingError("Validação reprovou o arquivo gerado:\n  - " + "\n  - ".join(issues))
                result["psnr"] = out_info.get("psnr")
            self.check_cancel()

            # --------------------------------------------- arquivo definitivo
            if os.path.exists(out_path):
                # Substituição autorizada: remove o antigo e sidecars obsoletos
                # (NUNCA o arquivo de entrada - garantido por same_file acima).
                for p in (out_path, out_path + ".ovr", out_path + ".aux.xml", out_path + ".msk"):
                    if os.path.exists(p) and not same_file(p, in_path):
                        if not safe_remove(p):
                            raise ProcessingError("Não foi possível remover %s (arquivo aberto no QGIS?)." % p)
            os.replace(partial, out_path)
            safe_remove(partial + ".aux.xml")
            final_written = True
            chk = gdal.Open(out_path)
            if chk is None:
                raise ProcessingError("O arquivo final não abre após a renomeação.")
            chk = None
            if out_info is None:
                tmp = gdal.Open(out_path)
                out_info = describe_dataset(tmp, out_path)
                tmp = None
            out_info["path"] = out_path
            out_info["file_size"], _ = file_size_with_sidecars(out_path)
            result["out_size"] = out_info["file_size"]

            # ------------------------------------------------------- relatório
            self.progress(99)
            src_ds = close_ds(src_ds)
            self.final_report(info, out_info, dst_srs, res, time.time() - t_start, stats, comp, quality,
                              mode, fmt, identity, sizes.get(key))
            self.qgis_tips(mode, comp)
            self.progress(100)
            result["ok"] = True
            result["output"] = out_path
        except CancelledError as e:
            result["error"] = str(e)
            self.log("")
            self.log("*** CANCELADO: %s" % e)
        except ProcessingError as e:
            if self._is_cancelled():
                result["error"] = "Processamento cancelado pelo usuário."
                self.log("")
                self.log("*** CANCELADO")
            else:
                result["error"] = str(e)
                self.log("")
                self.log("*** ERRO: %s" % e)
        except Exception as e:
            if self._is_cancelled():   # GDAL abortado pelo callback (modo exceções)
                result["error"] = "Processamento cancelado pelo usuário."
                self.log("")
                self.log("*** CANCELADO")
            else:
                result["error"] = "Erro inesperado: %s" % e
                self.log("")
                self.log("*** ERRO INESPERADO: %s" % e)
                self.log(traceback.format_exc())
        finally:
            # A thread de leitura usa o VRT: precisa parar ANTES de fechá-lo.
            if reader is not None:
                reader.shutdown()
                reader = None
            # Fecha tudo na ordem inversa de dependência. Um GeoTIFF JPEG
            # cancelado no meio reclama ao fechar ("too few scanlines": o GDAL
            # tenta preencher os tiles que faltam); o arquivo é apagado logo
            # abaixo. Mensagens vindas das threads internas do GDAL ainda podem
            # aparecer no log do QGIS - são inofensivas.
            with GdalErrorCollector():
                out_ds = close_ds(out_ds)
            warped = close_ds(warped)
            src_vrt = close_ds(src_vrt)
            src_ds = close_ds(src_ds)
            gdal.SetCacheMax(old_cache)
            left = []
            if tmpdir and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
                if os.path.isdir(tmpdir):
                    left.append(tmpdir)
            if not final_written and partial:
                for p in (partial, partial + ".aux.xml", partial + ".ovr", partial + ".msk",
                          partial + ".ovr.tmp", partial + ".msk.ovr.tmp"):
                    if os.path.exists(p) and not safe_remove(p):
                        left.append(p)
            result["temp_left"] = left
            if left:
                self.log("[!] Temporários que NÃO puderam ser removidos (apague manualmente):")
                for p in left:
                    self.log("    " + p)
            elif tmpdir:
                self.log("Temporários removidos.")
            self.log("Tempo total: %s" % fmt_duration(time.time() - t_start))
            # Relatório em texto ao lado do resultado (também em caso de erro,
            # para facilitar o suporte).
            if params.get("save_report", True) and out_path:
                if result["ok"]:
                    rpath = os.path.splitext(result["output"])[0] + "_relatorio.txt"
                else:
                    rpath = os.path.splitext(out_path)[0] + "_relatorio_ERRO.txt"
                try:
                    if os.path.isdir(os.path.dirname(rpath)):
                        with open(rpath, "w", encoding="utf-8") as f:
                            f.write("\n".join(self.report_lines) + "\n")
                        result["report"] = rpath
                        self._log("Relatório salvo em: %s" % rpath)
                except Exception as e:
                    self._log("[!] Não foi possível salvar o relatório: %s" % e)
        return result

    def analyze_only(self):
        result = {"ok": False, "error": None, "sizes": {}}
        ds = None
        try:
            self.log_environment()
            ds, info, counts = self.analyze(0, 90, full_estimate=True)
            dst_srs, _ = self.get_dst_srs()
            comp = self.choose_compression()
            quality = self.quality()
            with_ovr = self.p.get("overviews", True)
            tile = int(self.p.get("tile", DEFAULT_TILE))
            res_x, res_y, note = self.auto_resolution(info, dst_srs)
            if self.is_identity(info, dst_srs):
                w, h = info["width"], info["height"]
            else:
                ext = transformed_extent(ds, info["srs"], dst_srs)
                w = int(math.ceil((ext[2] - ext[0]) / res_x))
                h = int(math.ceil((ext[3] - ext[1]) / res_y))
            self.log("")
            self.log("PREVISÃO DA SAÍDA (%s)" % srs_label(dst_srs))
            self.log("  Resolução automática : %.4f %s  [%s]" % (res_x, info["dst_units"], note))
            self.log("  Dimensão aproximada  : %s x %s px" % (fmt_int(w), fmt_int(h)))
            self.log("  Overviews previstas  : %s" % (", ".join(str(lv) for lv in
                     compute_overview_levels(w, h, tile)) or "nenhuma"))
            sizes = self.log_estimates(info, res_x, res_y, (comp, quality if is_lossy(comp) else 0), with_ovr)
            result.update(ok=True, res=res_x, info=info, sizes=sizes, in_size=info["file_size"])
        except CancelledError as e:
            result["error"] = str(e)
            self.log("*** CANCELADO")
        except ProcessingError as e:
            result["error"] = str(e)
            self.log("*** ERRO: %s" % e)
        except Exception as e:
            result["error"] = "Erro inesperado: %s" % e
            self.log("*** ERRO INESPERADO: %s\n%s" % (e, traceback.format_exc()))
        finally:
            close_ds(ds)
        self.progress(100)
        return result

    # ------------------------------------------------------------ validação --
    def validate(self, path, dst_srs, nb_exp, mode, tile, levels, in_info, in_counts, pstart, pend,
                 comp="ZSTD", quality=DEFAULT_QUALITY, fmt="GTIFF", ref_path=None):
        self.section("VALIDAÇÃO DO RESULTADO (arquivo fechado e reaberto)")
        issues, warns = [], []

        def chk(cond, ok_msg, fail_msg, critical=True):
            if cond:
                self.log("  [OK]   " + ok_msg)
            else:
                self.log("  [FALHA] " + fail_msg if critical else "  [AVISO] " + fail_msg)
                (issues if critical else warns).append(fail_msg)

        chk(os.path.isfile(path), "arquivo existe (%s)" % fmt_bytes(os.path.getsize(path) if os.path.isfile(path) else 0),
            "arquivo não existe")
        if issues:
            return None, False, issues
        with GdalErrorCollector() as ec:
            ds = gdal.OpenEx(path, gdal.OF_RASTER | gdal.OF_READONLY)
        chk(ds is not None, "abre corretamente pelo GDAL", "GDAL não abre o arquivo: %s" % "; ".join(ec.errors))
        if ds is None:
            return None, False, issues
        try:
            o = describe_dataset(ds, path)
            srs = o["srs"]
            chk(srs is not None and srs.IsSame(dst_srs), "CRS = %s" % o["crs_label"],
                "CRS inesperado: %s" % o["crs_label"])
            chk(srs is not None and srs.IsProjected() and o["units"] == "m",
                "coordenadas projetadas em metros", "CRS não está em metros (%s)" % o["units"], critical=False)
            chk(o["bands"] == nb_exp, "%d bandas (%s)" % (o["bands"], ", ".join(o["color_interp"])),
                "número de bandas %d (esperado %d)" % (o["bands"], nb_exp))
            chk(all(t == "Byte" for t in o["dtypes"]), "tipo Byte", "tipo inesperado: %s" % o["dtypes"])
            if mode == "nodata":
                chk(o["alpha_idx"] is None, "sem banda Alpha", "existe banda Alpha")
                chk(all(v == 0 for v in o["nodata"]), "NoData = 0 nas 3 bandas", "NoData incorreto: %s" % o["nodata"])
            elif mode == "alpha":
                chk(o["alpha_idx"] == 4, "banda 4 = Alpha", "Alpha não encontrada na banda 4")
            else:
                chk(o["alpha_idx"] is None, "sem banda Alpha", "existe banda Alpha")
                chk(all(v is None for v in o["nodata"]), "sem NoData (a máscara define a área válida)",
                    "NoData inesperado: %s" % o["nodata"])
                chk(o["per_dataset_mask"], "máscara interna 1 bit presente", "máscara interna ausente")
                chk(not os.path.exists(path + ".msk"), "máscara interna (sem .msk externo)", "existe .msk externo")
            chk(o["color_interp"][:3] == ["Red", "Green", "Blue"], "B1=Red B2=Green B3=Blue",
                "interpretação de cor: %s" % o["color_interp"], critical=False)
            chk(o["tiled"] and o["block"] == (tile, tile), "tiled %dx%d" % o["block"],
                "não está tiled %dx%d (bloco %s)" % (tile, tile, o["block"]))
            chk(comp in o["compression"].upper(), "compressão %s" % o["compression"],
                "compressão %s (esperado %s)" % (o["compression"], comp))
            if fmt == "COG":
                chk(o["layout"] == "COG", "layout COG válido (LAYOUT=COG)",
                    "o arquivo não é um COG (LAYOUT=%s)" % o["layout"])
            chk(abs(abs(o["gt"][1]) - abs(o["gt"][5])) < 1e-9, "pixel quadrado %.4f m" % o["res_x"],
                "pixel não quadrado (%.10g x %.10g)" % (o["res_x"], o["res_y"]), critical=False)
            if levels:
                if fmt == "COG":
                    chk(len(o["ovr_factors"]) > 0, "overviews internas (COG): %s" % ", ".join(map(str, o["ovr_factors"])),
                        "COG sem overviews")
                else:
                    chk(o["ovr_factors"] == levels, "overviews internas: %s" % ", ".join(map(str, o["ovr_factors"])),
                        "overviews %s (esperado %s)" % (o["ovr_factors"], levels))
                chk(not o["ovr_external"], "overviews internas (sem .ovr externo)", "existe .ovr externo",
                    critical=False)
                if mode == "mask":
                    chk(o["mask_ovr_count"] == len(o["ovr_factors"]),
                        "pirâmides da máscara: %d níveis" % o["mask_ovr_count"],
                        "pirâmides da máscara: %d (esperado %d)" % (o["mask_ovr_count"], len(o["ovr_factors"])))
            self.log("  [INFO] Extensão: xmin=%.3f ymin=%.3f xmax=%.3f ymax=%.3f" % o["extent"])
            # Garante que o resultado está realmente em UTM e cobre a mesma área.
            ct = make_transform(srs, in_info["srs"])
            cx = (o["extent"][0] + o["extent"][2]) / 2.0
            cy = (o["extent"][1] + o["extent"][3]) / 2.0
            bx, by = transform_xy(ct, [(cx, cy)])[0]
            ie = in_info["extent"]
            tolx = (ie[2] - ie[0]) * 0.05
            toly = (ie[3] - ie[1]) * 0.05
            chk(ie[0] - tolx <= bx <= ie[2] + tolx and ie[1] - toly <= by <= ie[3] + toly,
                "centro do resultado reprojetado de volta cai dentro da extensão original",
                "centro do resultado não coincide com a área original (reprojeção incorreta?)")

            # --- leitura de amostras de blocos (nível 0 e todas as overviews) ---
            n_read, n_err = 0, 0
            band_list = list(range(1, o["bands"] + 1))
            with GdalErrorCollector() as ec:
                grid = 8
                for gy in range(grid):
                    for gx in range(grid):
                        self.check_cancel()
                        x = min(o["width"] - 1, int(gx * o["width"] / float(grid))) // tile * tile
                        y = min(o["height"] - 1, int(gy * o["height"] / float(grid))) // tile * tile
                        xs, ys = min(tile, o["width"] - x), min(tile, o["height"] - y)
                        try:
                            r = ds.ReadRaster(x, y, xs, ys, xs, ys, gdal.GDT_Byte, band_list)
                        except RuntimeError:
                            r = None
                        n_read += 1
                        if r is None or len(r) != xs * ys * len(band_list):
                            n_err += 1
                    self.progress(pstart + (pend - pstart) * 0.3 * (gy + 1) / grid)
                for b in band_list + (["mask"] if mode == "mask" else []):
                    band = ds.GetRasterBand(1).GetMaskBand() if b == "mask" else ds.GetRasterBand(b)
                    for i in range(band.GetOverviewCount()):
                        ov = band.GetOverview(i)
                        if ov.XSize * ov.YSize <= 4096 * 4096:   # ovr pequenas: lê inteiras
                            try:
                                r = ov.ReadRaster(0, 0, ov.XSize, ov.YSize)
                            except RuntimeError:
                                r = None
                            n_read += 1
                            if r is None:
                                n_err += 1
                        else:                                     # grandes: amostra
                            for k in range(4):
                                x = (k * ov.XSize // 4) // tile * tile
                                xs = min(tile, ov.XSize - x)
                                ys = min(tile, ov.YSize)
                                try:
                                    r = ov.ReadRaster(x, 0, xs, ys)
                                except RuntimeError:
                                    r = None
                                n_read += 1
                                if r is None:
                                    n_err += 1
                self.progress(pstart + (pend - pstart) * 0.45)
            chk(n_err == 0 and not ec.errors, "%d leituras de blocos/overviews sem erros" % n_read,
                "%d de %d leituras falharam: %s" % (n_err, n_read, "; ".join(ec.errors[:3])))

            # --- fidelidade: blocos do resultado x imagem reprojetada (referência) ---
            if ref_path and os.path.exists(ref_path):
                self.compare_reference(ds, o, ref_path, mode, comp, quality, tile, chk)
            self.progress(pstart + (pend - pstart) * 0.75)
            # --- comparação ponto a ponto com a ORIGINAL (prova da reprojeção) ---
            self.compare_points(ds, o, in_info, mode)
            # --- fração válida aproximada (a partir da menor overview útil) ---
            o["valid_fraction"] = self.output_valid_fraction(ds, o, mode)
            self.progress(pend)
        finally:
            ds = close_ds(ds)
        if warns:
            self.log("  Avisos: %d" % len(warns))
        self.log("  RESULTADO: %s" % ("APROVADO" if not issues else "REPROVADO"))
        return o, not issues, issues

    def compare_reference(self, ds, o, ref_path, mode, comp, quality, tile, chk, grid=8):
        """Compara 64 blocos do arquivo final com os MESMOS blocos da imagem
        reprojetada (VRT, antes da compressão). Sem perda: tem de ser bit a bit
        idêntico. Com perda: mede o PSNR e exige máscara idêntica."""
        ref = gdal.Open(ref_path)
        if ref is None:
            self.log("  [AVISO] não foi possível reabrir a referência para comparação")
            return
        try:
            W, H = o["width"], o["height"]
            maxdiff, ndiff, mask_bad, nblk, nvalid = 0, 0, 0, 0, 0
            sse = 0.0
            mband = ds.GetRasterBand(1).GetMaskBand() if mode == "mask" else None
            bands = [1, 2, 3, 4] if mode == "alpha" else [1, 2, 3]
            seen = set()
            for gy in range(grid):
                for gx in range(grid):
                    self.check_cancel()
                    x = min(W - 1, int((gx + 0.5) * W / float(grid))) // tile * tile
                    y = min(H - 1, int((gy + 0.5) * H / float(grid))) // tile * tile
                    if (x, y) in seen:
                        continue
                    seen.add((x, y))
                    xs, ys = min(tile, W - x), min(tile, H - y)
                    r = ref.ReadAsArray(x, y, xs, ys)
                    f = np.frombuffer(ds.ReadRaster(x, y, xs, ys, xs, ys, gdal.GDT_Byte, bands),
                                      dtype=np.uint8).reshape(len(bands), ys, xs)
                    valid = r[3] >= ALPHA_VALID_THRESHOLD
                    nblk += 1
                    if mode == "alpha":
                        d = np.abs(f.astype(np.int16) - r.astype(np.int16))
                    elif mode == "nodata":
                        exp = r[:3].copy()
                        np.maximum(exp, 1, out=exp, where=valid)
                        exp[:, ~valid] = 0
                        d = np.abs(f.astype(np.int16) - exp.astype(np.int16))
                    else:
                        m = np.frombuffer(mband.ReadRaster(x, y, xs, ys), dtype=np.uint8).reshape(ys, xs) > 0
                        mask_bad += int(np.count_nonzero(m != valid))
                        if np.any(valid):
                            dd = f[:, valid].astype(np.float64) - r[:3][:, valid].astype(np.float64)
                            sse += float(np.sum(dd * dd))
                            nvalid += dd.size
                        continue
                    maxdiff = max(maxdiff, int(d.max()) if d.size else 0)
                    ndiff += int(np.count_nonzero(d))
            if mode == "mask":
                chk(mask_bad == 0, "máscara idêntica à área válida em %d blocos" % nblk,
                    "máscara difere da área válida em %d pixels" % mask_bad)
                if nvalid:
                    mse = sse / nvalid
                    db = 99.0 if mse <= 0 else 10.0 * math.log10(255.0 ** 2 / mse)
                    o["psnr"] = db
                    chk(db >= 32, "qualidade %s: PSNR %s dB (%s) em %d blocos"
                        % (codec_label(comp, quality).split(" (")[0], fmt_num(db, 1), psnr_label(db), nblk),
                        "qualidade baixa: PSNR %s dB (%s) - aumente a qualidade"
                        % (fmt_num(db, 1), psnr_label(db)), critical=False)
            else:
                o["psnr"] = 99.0 if maxdiff == 0 else None
                chk(maxdiff == 0, "SEM PERDA confirmado: %d blocos idênticos bit a bit à imagem reprojetada" % nblk,
                    "%d pixels diferem da imagem reprojetada (diferença máx. %d) - deveria ser 0"
                    % (ndiff, maxdiff))
        finally:
            ref = None

    def _validity_downsampled(self, ds, o, mode, bw, bh):
        w, h = o["width"], o["height"]
        if mode == "mask":
            r = ds.GetRasterBand(1).GetMaskBand().ReadRaster(0, 0, w, h, bw, bh)
        else:
            b = 4 if mode == "alpha" else 1
            r = ds.ReadRaster(0, 0, w, h, bw, bh, gdal.GDT_Byte, [b])   # usa overviews automaticamente
        if r is None:
            return None
        return np.frombuffer(r, dtype=np.uint8).reshape(bh, bw)

    def output_valid_fraction(self, ds, o, mode):
        w, h = o["width"], o["height"]
        scale = max(1.0, max(w, h) / 1024.0)
        bw, bh = max(1, int(w / scale)), max(1, int(h / scale))
        a = self._validity_downsampled(ds, o, mode, bw, bh)
        if a is None:
            return None
        return float(np.count_nonzero(a)) / a.size

    def compare_points(self, out_ds, o, in_info, mode, n=150):
        """Sorteia pixels válidos do resultado, leva cada um ao CRS original,
        lê o mesmo local na entrada e compara RGB. Diferença pequena => os pixels
        foram realmente reprojetados para o lugar certo."""
        src = gdal.Open(in_info["path"])
        if src is None:
            self.log("  [AVISO] não foi possível reabrir a entrada para comparação")
            return
        try:
            w, h = o["width"], o["height"]
            scale = max(1.0, max(w, h) / 1024.0)
            bw, bh = max(1, int(w / scale)), max(1, int(h / scale))
            m = self._validity_downsampled(out_ds, o, mode, bw, bh)
            if m is None:
                return
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                self.log("  [AVISO] nenhum pixel válido encontrado para comparação")
                return
            rnd = random.Random(12345)
            idx = rnd.sample(range(len(xs)), min(len(xs), n * 3))
            ct = make_transform(o["srs"], in_info["srs"])
            ogt, sgt = o["gt"], in_info["gt"]
            a_idx = in_info["alpha_idx"]
            sbands = [1, 2, 3] + ([a_idx] if a_idx else [])
            mband = out_ds.GetRasterBand(1).GetMaskBand() if mode == "mask" else None
            diffs = []
            for k in idx:
                if len(diffs) >= n:
                    break
                px = int((xs[k] + 0.5) * w / float(bw))
                py = int((ys[k] + 0.5) * h / float(bh))
                px, py = min(px, w - 1), min(py, h - 1)
                ov = np.frombuffer(out_ds.ReadRaster(px, py, 1, 1, 1, 1, gdal.GDT_Byte, [1, 2, 3]), dtype=np.uint8)
                if mode == "nodata" and np.any(ov == 0):
                    continue
                if mband is not None and np.frombuffer(mband.ReadRaster(px, py, 1, 1), dtype=np.uint8)[0] == 0:
                    continue
                gx, gy = pixel_to_geo(ogt, px + 0.5, py + 0.5)
                sx, sy = transform_xy(ct, [(gx, gy)])[0]
                spx, spy = geo_to_pixel(sgt, sx, sy)
                ix, iy = int(math.floor(spx)), int(math.floor(spy))
                if not (0 <= ix < src.RasterXSize and 0 <= iy < src.RasterYSize):
                    continue
                sv = np.frombuffer(src.ReadRaster(ix, iy, 1, 1, 1, 1, gdal.GDT_Byte, sbands), dtype=np.uint8)
                if a_idx and sv[3] < 255:
                    continue
                diffs.append(float(np.mean(np.abs(ov[:3].astype(np.int16) - sv[:3].astype(np.int16)))))
            if not diffs:
                self.log("  [AVISO] sem pontos comparáveis")
                return
            d = np.array(diffs)
            med = float(np.median(d))
            self.log("  [%s] comparação de %d pontos com a original: diferença RGB média=%.1f mediana=%.1f p90=%.1f (0-255)"
                     % ("OK" if med < 25 else "AVISO", len(d), d.mean(), med, np.percentile(d, 90)))
            self.log("         (diferenças pequenas vêm só da reamostragem/compressão; deslocamento de "
                     "posição geraria valores altos)")
        finally:
            src = None

    # ------------------------------------------------------------ relatório --
    def final_report(self, a, b, dst_srs, res, elapsed, stats, comp, quality, mode, fmt, identity, est_size):
        self.section("COMPARAÇÃO ORIGINAL x FINAL")

        def ovr(d):
            return ", ".join(str(f) for f in d["ovr_factors"]) or "nenhuma"

        def nd(d):
            vals = [v for v in d["nodata"]]
            if all(v is None for v in vals):
                return "-"
            return ", ".join("-" if v is None else "%g" % v for v in vals)

        def resol(d):
            return "%.10g x %.10g %s" % (d["res_x"], d["res_y"], d["units"])

        if is_lossy(comp):
            comp_final = "%s q%d%s" % (comp, quality, " YCbCr" if comp == "JPEG" else "")
        else:
            comp_final = b["compression"] + ("+P%s" % b["predictor"] if b["predictor"] else "+P2")
        rows = [
            ("Arquivo", os.path.basename(a["path"]), os.path.basename(b["path"])),
            ("Tamanho", fmt_bytes(a["file_size"]), fmt_bytes(b["file_size"])),
            ("Formato", "GeoTIFF" if a["driver"] == "GTiff" else a["driver"],
             "COG" if fmt == "COG" else "GeoTIFF otimizado"),
            ("CRS", a["crs_label"], b["crs_label"]),
            ("Dimensões", "%s x %s" % (fmt_int(a["width"]), fmt_int(a["height"])),
             "%s x %s" % (fmt_int(b["width"]), fmt_int(b["height"]))),
            ("Bandas", "%d (%s)" % (a["bands"], ",".join(a["color_interp"])),
             "%d (%s)" % (b["bands"], ",".join(b["color_interp"]))),
            ("Datatype", ",".join(sorted(set(a["dtypes"]))), ",".join(sorted(set(b["dtypes"])))),
            ("Resolução", resol(a), resol(b)),
            ("Resolução (m)", "%.4f x %.4f" % a["res_m"], "%.4f x %.4f" % (b["res_x"], b["res_y"])),
            ("Compressão", a["compression"] + ("+P%s" % a["predictor"] if a["predictor"] else ""), comp_final),
            ("Tiled", "sim" if a["tiled"] else "não (strips)", "sim" if b["tiled"] else "não"),
            ("Block size", "%dx%d" % a["block"], "%dx%d" % b["block"]),
            ("Overviews", ovr(a), ovr(b)),
            ("NoData", nd(a), nd(b)),
            ("Transparência", transparency_label(a), {"nodata": "NoData=0", "alpha": "Alpha (banda 4)",
                                                      "mask": "máscara interna 1 bit"}[mode]),
        ]
        if b.get("psnr") is not None:
            rows.append(("Fidelidade", "-", "idêntica (sem perda)" if b["psnr"] >= 99 and not is_lossy(comp)
                         else "PSNR %s dB (%s)" % (fmt_num(b["psnr"], 1), psnr_label(b["psnr"]))))
        c1 = max(len(r[0]) for r in rows) + 1
        c2 = max(28, max(len(r[1]) for r in rows) + 1)
        c3 = max(28, max(len(r[2]) for r in rows))
        sep = "+-%s-+-%s-+-%s-+" % ("-" * c1, "-" * c2, "-" * c3)
        self.log(sep)
        self.log("| %s | %s | %s |" % ("Propriedade".ljust(c1), "ORIGINAL".ljust(c2), "FINAL".ljust(c3)))
        self.log(sep)
        for r in rows:
            self.log("| %s | %s | %s |" % (r[0].ljust(c1), r[1].ljust(c2), r[2].ljust(c3)))
        self.log(sep)

        ds_ = (b["file_size"] - a["file_size"]) / float(max(1, a["file_size"])) * 100
        self.log("")
        self.log("Tamanho            : %s%% (%s -> %s)" % (("+" if ds_ >= 0 else "") + fmt_num(ds_, 1),
                 fmt_bytes(a["file_size"]), fmt_bytes(b["file_size"])))
        if est_size:
            self.log("Estimativa prévia  : ~%s (erro de %s%%)" % (fmt_bytes(est_size),
                     fmt_num((est_size - b["file_size"]) / float(max(1, b["file_size"])) * 100, 0)))
        orig_m = min(a["res_m"])
        self.log("Resolução          : %.4f m -> %.4f m (%s%%)%s" % (orig_m, res,
                 ("+" if res >= orig_m else "") + fmt_num((res - orig_m) / orig_m * 100, 2),
                 "  [mesma grade, sem reamostragem]" if identity else ""))
        try:
            src = gdal.Open(a["path"])
            te = transformed_extent(src, a["srs"], dst_srs)
            src = None
            be = b["extent"]
            self.log("Extensão original no CRS final: xmin=%.3f ymin=%.3f xmax=%.3f ymax=%.3f" % te)
            self.log("Extensão final                : xmin=%.3f ymin=%.3f xmax=%.3f ymax=%.3f" % be)
            self.log("Diferença (m)      : xmin %+.3f  ymin %+.3f  xmax %+.3f  ymax %+.3f  "
                     "(<= 1 pixel = alinhamento da grade)" % (be[0] - te[0], be[1] - te[1], be[2] - te[2], be[3] - te[3]))
        except Exception as e:
            self.log("Diferença de extensão: não calculada (%s)" % e)
        if a.get("valid_fraction") is not None and stats.get("valid"):
            # Final: contagem EXATA dos pixels válidos gravados (a estimativa por
            # overview reduzida conta a borda como válida e inflava ~1%).
            area_a = a["valid_fraction"] * a["width"] * a["height"] * a["res_m"][0] * a["res_m"][1]
            area_b = stats["valid"] * b["res_x"] * b["res_y"]
            self.log("Área com cobertura : %s ha (original, por amostragem) -> %s ha (final, contagem exata)"
                     % (fmt_num(area_a / 1e4, 2), fmt_num(area_b / 1e4, 2)))
        self.log("Tempo total        : %s" % fmt_duration(elapsed))
        self.log("")
        self.log("PROBLEMAS CORRIGIDOS")
        if not a["ovr_factors"] and b["ovr_factors"]:
            self.log("  [x] sem overviews        -> overviews internas %s" % ovr(b))
        if not a["tiled"]:
            self.log("  [x] strips               -> tiles %dx%d" % b["block"])
        if a["epsg"] != b["epsg"]:
            self.log("  [x] %s -> %s (pixels reprojetados)" % (a["crs_label"], b["crs_label"]))
        if a["alpha_idx"] and not b["alpha_idx"]:
            self.log("  [x] banda Alpha          -> %s" % ("removida; áreas vazias como NoData=0" if mode == "nodata"
                                                         else "máscara interna 1 bit (borda exata)"))
        if is_lossy(comp):
            self.log("  [x] compressão %s -> %s (com perda controlada%s)" % (
                a["compression"], comp_final,
                ", PSNR %s dB" % fmt_num(b["psnr"], 1) if b.get("psnr") else ""))
        else:
            self.log("  [x] compressão %s -> %s + PREDICTOR=2 (sem perda)" % (a["compression"], comp))
        if fmt == "COG":
            self.log("  [x] layout COG (Cloud Optimized GeoTIFF): pronto para servir por HTTP/nuvem")

    def qgis_tips(self, mode, comp):
        self.section("DICAS DE DESEMPENHO NO QGIS (nada foi alterado automaticamente)")
        tips = [
            "Use o CRS do PROJETO = EPSG:31983 (canto inferior direito). Assim o QGIS não "
            "reprojeta o raster em tempo real a cada pan/zoom.",
            "Propriedades da camada > Simbologia: renderizador 'Cor multibanda' com "
            "'Realce de contraste: Sem realce' (8 bits já está em 0-255; evita cálculo de estatísticas).",
            "Propriedades da camada > Simbologia > Reamostragem: 'Vizinho mais próximo' é o mais rápido; "
            "'Bilinear' (aumentado) melhora a aparência em zoom muito próximo com custo baixo.",
            "Propriedades da camada > Pirâmides: as resoluções aparecem com ícone de pirâmide "
            "(internas). Não é preciso gerar novamente.",
            "Configurações > Opções > Renderização: marque 'Renderizar camadas em paralelo' e "
            "'Usar cache de renderização quando possível'.",
            "Configurações > Opções > Sistema > Variáveis de ambiente: GDAL_CACHEMAX (ex.: 1024 = 1 GB) "
            "pode ajudar em mosaicos muito grandes. Opcional; reinicie o QGIS depois.",
            "Mantenha o arquivo em SSD local; evite pastas de rede/sincronizadas (OneDrive/Drive) durante o uso.",
        ]
        if mode == "nodata":
            tips.append("A transparência das bordas vem do NoData=0 nas 3 bandas (aba Transparência "
                        "mostra 'Sem valor de dados'). Não marque valores adicionais de transparência.")
        elif mode == "mask":
            tips.append("A transparência vem da máscara interna: o QGIS a mostra como banda 4 (Alpha). "
                        "Não configure 'Sem valor de dados' nem valores de transparência.")
        if comp == "JPEG":
            tips.append("JPEG dentro de TIFF é padrão e abre em praticamente qualquer SIG. A máscara "
                        "interna é padrão GDAL (QGIS, ArcGIS Pro, Global Mapper); um programa que a ignore "
                        "mostra a área fora do voo em preto - a imagem em si continua correta.")
        elif comp == "WEBP":
            tips.append("WEBP dentro de TIFF só é lido por softwares baseados em GDAL >= 2.4 (QGIS lê). "
                        "Se for abrir em outros programas, prefira JPEG.")
        elif comp == "ZSTD":
            tips.append("ZSTD é lido por qualquer QGIS/GDAL >= 2.3. Softwares antigos não-GDAL podem não ler; "
                        "nesse caso gere novamente com DEFLATE.")
        for i, t in enumerate(tips, 1):
            self.log("%d. %s" % (i, t))


# =============================================================================
#  LOTE
# =============================================================================
def run_batch(jobs, mode, log, progress, is_cancelled):
    """Processa (ou analisa) vários arquivos em sequência, com resumo final."""
    n = len(jobs)
    results = []
    t0 = time.time()
    for i, params in enumerate(jobs):
        if is_cancelled():
            break
        log("")
        log("#" * 78)
        log("# ARQUIVO %d de %d: %s" % (i + 1, n, os.path.basename(params["input"])))
        log("#" * 78)

        def prog(v, i=i):
            progress(int((i * 100 + v) / float(n)))

        core = OrthoOptimizer(params, log=log, progress=prog, is_cancelled=is_cancelled)
        r = core.run() if mode == "run" else core.analyze_only()
        r["input"] = params["input"]
        if r.get("in_size") is None:
            try:
                r["in_size"] = os.path.getsize(params["input"])
            except OSError:
                r["in_size"] = None
        results.append(r)
    cancelled = is_cancelled() or len(results) < n

    lines = ["", "=" * 78, " RESUMO DO LOTE (%d arquivo(s), %s)" % (n, fmt_duration(time.time() - t0)), "=" * 78]
    if mode == "run":
        tin = tout = 0
        lines.append("  %-40s %10s %10s %8s  %s" % ("Arquivo", "Original", "Final", "Redução", "Situação"))
        for r in results:
            name = os.path.basename(r["input"])
            if len(name) > 40:
                name = name[:37] + "..."
            if r.get("ok"):
                tin += r["in_size"] or 0
                tout += r["out_size"] or 0
                red = (1.0 - float(r["out_size"] or 0) / max(1, r["in_size"] or 1)) * 100
                lines.append("  %-40s %10s %10s %7s%%  OK" % (name, fmt_bytes(r["in_size"]), fmt_bytes(r["out_size"]),
                                                             fmt_num(red, 0)))
            else:
                lines.append("  %-40s %10s %10s %8s  %s" % (name, fmt_bytes(r.get("in_size")), "-", "-",
                                                           "ERRO: " + (r.get("error") or "?").split("\n")[0][:60]))
        if tin:
            lines.append("  %-40s %10s %10s %7s%%" % ("TOTAL (arquivos OK)", fmt_bytes(tin), fmt_bytes(tout),
                                                     fmt_num((1.0 - float(tout) / tin) * 100, 0)))
    else:
        totals = {}
        tin = 0
        for r in results:
            if r.get("ok"):
                tin += r.get("in_size") or 0
                for k, v in (r.get("sizes") or {}).items():
                    totals[k] = totals.get(k, 0) + v
        lines.append("  Tamanho atual somado: %s" % fmt_bytes(tin))
        for k in sorted(totals, key=lambda k: (is_lossy(k[0]), -totals[k])):
            lines.append("  %-24s ~%s" % (codec_label(*k), fmt_bytes(totals[k])))
        bad = [r for r in results if not r.get("ok")]
        for r in bad:
            lines.append("  ERRO em %s: %s" % (os.path.basename(r["input"]), (r.get("error") or "?").split("\n")[0]))
    if cancelled:
        lines.append("  *** Lote interrompido: %d de %d arquivo(s) processado(s)." % (len(results), n))
    for line in lines:
        log(line)
    summary_path = None
    if mode == "run" and jobs and jobs[0].get("save_report", True):
        out_dir = os.path.dirname(os.path.abspath(jobs[0]["output"]))
        summary_path = os.path.join(out_dir, "resumo_lote_%s.txt" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            summary_path = None
    failed = [r for r in results if not r.get("ok")]
    err = None
    if cancelled:
        err = "Lote cancelado (%d de %d arquivo(s) processado(s))." % (len(results), n)
    elif failed:
        err = "%d de %d arquivo(s) falharam. Veja o log." % (len(failed), n)
    return {"ok": not cancelled and not failed, "batch": True, "results": results, "error": err,
            "outputs": [r["output"] for r in results if r.get("ok") and r.get("output")],
            "report": summary_path}


# =============================================================================
#  INTERFACE (QGIS)
# =============================================================================
if HAS_QGIS:

    def _exec(obj):
        return obj.exec() if hasattr(obj, "exec") else obj.exec_()

    COMPRESSION_CHOICES = [
        ("JPEG", "JPEG - com perda leve, arquivo MUITO menor (substitui o ECW)"),
        ("WEBP", "WEBP - com perda, alternativa ao JPEG (só softwares GDAL)"),
        ("ZSTD", "ZSTD - sem perda (maior; para medições/reprocessamento)"),
        ("DEFLATE", "DEFLATE - sem perda (compatível com programas antigos)"),
        ("LZW", "LZW - sem perda (compatibilidade máxima)"),
    ]
    FORMAT_CHOICES = [
        ("GTIFF", "GeoTIFF otimizado (tiled + pirâmides) - recomendado"),
        ("COG", "COG - Cloud Optimized GeoTIFF (para servidor/nuvem)"),
    ]

    class OptimizerThread(QThread):
        sig_log = pyqtSignal(str)
        sig_progress = pyqtSignal(int)
        sig_done = pyqtSignal(object)

        def __init__(self, jobs, mode="run", parent=None):
            QThread.__init__(self, parent)
            self.jobs = jobs
            self.mode = mode
            self._cancel = False

        def cancel(self):
            self._cancel = True

        def run(self):
            try:
                if len(self.jobs) == 1:
                    core = OrthoOptimizer(self.jobs[0], log=self.sig_log.emit,
                                          progress=self.sig_progress.emit,
                                          is_cancelled=lambda: self._cancel)
                    res = core.run() if self.mode == "run" else core.analyze_only()
                    res["content_cache"] = getattr(core, "content_cache", None)
                    res["outputs"] = [res["output"]] if res.get("output") else []
                else:
                    res = run_batch(self.jobs, self.mode, self.sig_log.emit, self.sig_progress.emit,
                                    lambda: self._cancel)
            except Exception as e:  # salvaguarda final
                res = {"ok": False, "error": "%s\n%s" % (e, traceback.format_exc())}
            res["mode"] = self.mode
            self.sig_done.emit(res)

    class OrthoOptimizerDialog(QDialog):

        def __init__(self, iface=None, parent=None):
            QDialog.__init__(self, parent)
            self.iface = iface
            self.worker = None
            self._content_cache = None   # análise de conteúdo reaproveitável
            self._inputs = []
            self._last_auto_name = None
            self._settings = QSettings()
            self.setWindowTitle("%s v%s" % (SCRIPT_NAME, SCRIPT_VERSION))
            self.resize(940, 900)
            self._build_ui()
            self._load_settings()
            self._fill_from_active_layer(silent=True)

        # ---------------------------------------------------------------- UI --
        def _build_ui(self):
            lay = QVBoxLayout(self)
            title = QLabel("<b>OTIMIZADOR DE ORTOMOSAICO</b> &nbsp; GeoTIFF -> CRS métrico, tiled, "
                           "com pirâmides, comprimido com ou sem perda (substituto livre do ECW)")
            title.setWordWrap(True)
            lay.addWidget(title)

            # --- Arquivos ---
            g1 = QGroupBox("Arquivos")
            f1 = QFormLayout(g1)
            self.ed_in = QLineEdit()
            self.ed_in.setPlaceholderText("Um ou vários GeoTIFFs (vários = processamento em lote)")
            self.ed_in.textEdited.connect(self._on_input_edited)
            b_in = QPushButton("Selecionar...")
            b_in.setToolTip("Selecione um ou VÁRIOS arquivos (Ctrl/Shift) para processar em lote")
            b_in.clicked.connect(self._pick_input)
            b_layer = QPushButton("Camada ativa")
            b_layer.setToolTip("Usar o raster da camada selecionada no painel de camadas")
            b_layer.clicked.connect(lambda: self._fill_from_active_layer(silent=False))
            f1.addRow("Arquivo(s) de entrada:", self._row(self.ed_in, b_in, b_layer))
            self.ed_out_dir = QLineEdit()
            b_out = QPushButton("Selecionar...")
            b_out.clicked.connect(self._pick_out_dir)
            f1.addRow("Pasta de saída:", self._row(self.ed_out_dir, b_out))
            self.ed_name = QLineEdit()
            f1.addRow("Nome do arquivo:", self.ed_name)
            self.ed_tmp = QLineEdit(tempfile.gettempdir())
            b_tmp = QPushButton("Selecionar...")
            b_tmp.clicked.connect(self._pick_tmp_dir)
            f1.addRow("Pasta temporária:", self._row(self.ed_tmp, b_tmp))
            lay.addWidget(g1)

            # --- Parâmetros ---
            g2 = QGroupBox("Parâmetros")
            f2 = QFormLayout(g2)
            self.crs_widget = QgsProjectionSelectionWidget()
            self.crs_widget.setCrs(QgsCoordinateReferenceSystem(DEFAULT_DST_CRS))
            self.crs_widget.crsChanged.connect(lambda *_: self._refresh_name())
            f2.addRow("CRS destino:", self.crs_widget)

            self.cb_res = QComboBox()
            self.cb_res.addItems(["Automática / preservar original", "Personalizada (metros)"])
            self.sp_res = QDoubleSpinBox()
            self.sp_res.setDecimals(4)
            self.sp_res.setRange(0.001, 1000.0)
            self.sp_res.setSingleStep(0.005)
            self.sp_res.setValue(0.04)
            self.sp_res.setSuffix(" m")
            self.sp_res.setEnabled(False)
            self.cb_res.currentIndexChanged.connect(lambda i: self.sp_res.setEnabled(i == 1))
            f2.addRow("Resolução:", self._row(self.cb_res, self.sp_res))

            self.cb_resamp = QComboBox()
            self.cb_resamp.addItems(RESAMPLING_WARP)
            self.cb_resamp.setToolTip("cubic: nítido e suave (recomendado). bilinear: um pouco mais suave. "
                                      "near: sem interpolação (serrilhado). lanczos: mais lento.\n"
                                      "Se a entrada já estiver no CRS de destino, não há reamostragem.")
            f2.addRow("Reamostragem (reproj.):", self.cb_resamp)

            self.cb_format = QComboBox()
            for code, label in FORMAT_CHOICES:
                if code == "COG" and not cog_driver_available():
                    continue
                self.cb_format.addItem(label, code)
            self.cb_format.setToolTip("GeoTIFF otimizado: ideal para uso local no QGIS (mesmo desempenho do "
                                      "COG, sem arquivo temporário).\nCOG: para publicar em servidor/nuvem; "
                                      "usa um intermediário temporário do tamanho da versão sem perda.")
            self.cb_format.currentIndexChanged.connect(lambda *_: self._refresh_name())
            f2.addRow("Formato:", self.cb_format)

            self.cb_comp = QComboBox()
            for code, label in COMPRESSION_CHOICES:
                if gtiff_supports_compression(code):
                    self.cb_comp.addItem(label, code)
            self.cb_comp.currentIndexChanged.connect(self._on_comp_changed)
            f2.addRow("Compressão:", self.cb_comp)

            self.sp_q = QSpinBox()
            self.sp_q.setRange(50, 100)
            self.sp_q.setValue(DEFAULT_QUALITY)
            self.sp_q.setToolTip("90: visualmente idêntico (recomendado).\n85: mais econômico, diferença só "
                                 "em zoom máximo.\n75: máxima economia, perda visível em zoom.\n"
                                 "Use 'Analisar entrada' para ver tamanho e PSNR estimados.")
            self.sp_q.valueChanged.connect(lambda *_: self._refresh_name())
            self.lb_q = QLabel("90 = visualmente idêntico | 85 = econômico | 75 = máxima economia")
            f2.addRow("Qualidade (JPEG/WEBP):", self._row(self.sp_q, self.lb_q))

            self.cb_tile = QComboBox()
            self.cb_tile.addItems(["512", "256"])
            f2.addRow("Tile (px):", self.cb_tile)

            self.rb_rgb = QRadioButton("Remover Alpha: RGB 3 bandas + NoData=0 (recomendado)")
            self.rb_rgba = QRadioButton("Preservar Alpha: RGBA 4 bandas")
            self.lb_mask = QLabel("Com JPEG/WEBP a transparência vai numa máscara interna de 1 bit "
                                  "(automático; o QGIS mostra como Alpha).")
            self.lb_mask.setWordWrap(True)
            self.rb_rgb.setChecked(True)
            grp = QButtonGroup(self)
            grp.addButton(self.rb_rgb)
            grp.addButton(self.rb_rgba)
            w_alpha = QWidget()
            va = QVBoxLayout(w_alpha)
            va.setContentsMargins(0, 0, 0, 0)
            va.addWidget(self.rb_rgb)
            va.addWidget(self.rb_rgba)
            va.addWidget(self.lb_mask)
            f2.addRow("Transparência:", w_alpha)

            self.ck_black = QCheckBox("Se a entrada não tiver Alpha/NoData/máscara, tratar 0,0,0 como área vazia")
            self.ck_black.setChecked(True)
            f2.addRow("", self.ck_black)

            self.ck_ovr = QCheckBox("Criar overviews internas")
            self.ck_ovr.setChecked(True)
            self.cb_ovr = QComboBox()
            self.cb_ovr.addItems(RESAMPLING_OVR)
            self.ck_ovr.toggled.connect(self.cb_ovr.setEnabled)
            f2.addRow("Overviews:", self._row(self.ck_ovr, self.cb_ovr))

            self.ck_val = QCheckBox("Validar resultado")
            self.ck_val.setChecked(True)
            self.ck_add = QCheckBox("Adicionar resultado ao projeto")
            self.ck_add.setChecked(True)
            self.ck_rep = QCheckBox("Salvar relatório .txt")
            self.ck_rep.setChecked(True)
            f2.addRow("Ao final:", self._row(self.ck_val, self.ck_add, self.ck_rep))
            lay.addWidget(g2)

            # --- Botões ---
            hb = QHBoxLayout()
            self.bt_an = QPushButton("Analisar entrada")
            self.bt_an.setToolTip("Mostra problemas, previsão da saída e o tamanho/qualidade estimados "
                                  "de cada compressão (sem gravar nada)")
            self.bt_an.clicked.connect(lambda: self._start("analyze"))
            self.bt_run = QPushButton("EXECUTAR")
            self.bt_run.setStyleSheet("font-weight: bold; padding: 6px 18px;")
            self.bt_run.clicked.connect(lambda: self._start("run"))
            self.bt_cancel = QPushButton("Cancelar")
            self.bt_cancel.setEnabled(False)
            self.bt_cancel.clicked.connect(self._cancel)
            self.bt_open = QPushButton("Abrir pasta de saída")
            self.bt_open.clicked.connect(self._open_out_dir)
            self.bt_close = QPushButton("Fechar")
            self.bt_close.clicked.connect(self.close)
            hb.addWidget(self.bt_an)
            hb.addStretch(1)
            hb.addWidget(self.bt_run)
            hb.addWidget(self.bt_cancel)
            hb.addWidget(self.bt_open)
            hb.addWidget(self.bt_close)
            lay.addLayout(hb)

            self.pb = QProgressBar()
            self.pb.setRange(0, 100)
            self.pb.setValue(0)
            lay.addWidget(QLabel("Progresso:"))
            lay.addWidget(self.pb)

            lay.addWidget(QLabel("Log:"))
            self.log_box = QPlainTextEdit()
            self.log_box.setReadOnly(True)
            self.log_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self.log_box.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
            self.log_box.setMaximumBlockCount(50000)
            lay.addWidget(self.log_box, 1)
            self._on_comp_changed()

        @staticmethod
        def _row(*widgets):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            for i, x in enumerate(widgets):
                h.addWidget(x, 1 if i == 0 and isinstance(x, QLineEdit) else 0)
            return w

        # ------------------------------------------------------ configurações --
        def _sget(self, key, default):
            v = self._settings.value(SETTINGS_PREFIX + key, None)
            if v is None:
                return default
            try:
                if isinstance(default, bool):
                    return str(v).lower() in ("true", "1", "yes", "sim")
                if isinstance(default, int):
                    return int(v)
                return str(v)
            except (TypeError, ValueError):
                return default

        def _sset(self, key, value):
            self._settings.setValue(SETTINGS_PREFIX + key, value)

        @staticmethod
        def _select_data(combo, code):
            i = combo.findData(code)
            if i >= 0:
                combo.setCurrentIndex(i)

        @staticmethod
        def _select_text(combo, text):
            i = combo.findText(text)
            if i >= 0:
                combo.setCurrentIndex(i)

        def _load_settings(self):
            out_dir = self._sget("out_dir", "")
            if out_dir and os.path.isdir(out_dir):
                self.ed_out_dir.setText(out_dir)
            tmp = self._sget("tmp_dir", "")
            if tmp and os.path.isdir(tmp):
                self.ed_tmp.setText(tmp)
            crs = self._sget("crs", "")
            if crs:
                c = QgsCoordinateReferenceSystem(crs)
                if c.isValid():
                    self.crs_widget.setCrs(c)
            self._select_data(self.cb_comp, self._sget("compression", DEFAULT_COMPRESSION))
            self.sp_q.setValue(self._sget("quality", DEFAULT_QUALITY))
            self._select_data(self.cb_format, self._sget("format", "GTIFF"))
            self._select_text(self.cb_tile, str(self._sget("tile", DEFAULT_TILE)))
            self._select_text(self.cb_resamp, self._sget("resampling", "cubic"))
            self._select_text(self.cb_ovr, self._sget("ovr_resampling", "AVERAGE"))
            self.rb_rgb.setChecked(self._sget("remove_alpha", True))
            self.rb_rgba.setChecked(not self._sget("remove_alpha", True))
            self.ck_black.setChecked(self._sget("treat_black", True))
            self.ck_ovr.setChecked(self._sget("overviews", True))
            self.ck_val.setChecked(self._sget("validate", True))
            self.ck_add.setChecked(self._sget("add_to_project", True))
            self.ck_rep.setChecked(self._sget("save_report", True))
            self._on_comp_changed()

        def _save_settings(self):
            if self.ed_out_dir.text().strip():
                self._sset("out_dir", self.ed_out_dir.text().strip())
            self._sset("tmp_dir", self.ed_tmp.text().strip())
            crs = self.crs_widget.crs()
            if crs.isValid():
                self._sset("crs", crs.authid() or crs.toWkt())
            self._sset("compression", self._comp_code())
            self._sset("quality", int(self.sp_q.value()))
            self._sset("format", self._fmt_code())
            self._sset("tile", int(self.cb_tile.currentText()))
            self._sset("resampling", self.cb_resamp.currentText())
            self._sset("ovr_resampling", self.cb_ovr.currentText())
            self._sset("remove_alpha", self.rb_rgb.isChecked())
            self._sset("treat_black", self.ck_black.isChecked())
            self._sset("overviews", self.ck_ovr.isChecked())
            self._sset("validate", self.ck_val.isChecked())
            self._sset("add_to_project", self.ck_add.isChecked())
            self._sset("save_report", self.ck_rep.isChecked())

        # ----------------------------------------------------------- ações --
        def _comp_code(self):
            return self.cb_comp.currentData() or DEFAULT_COMPRESSION

        def _fmt_code(self):
            return self.cb_format.currentData() or "GTIFF"

        def _on_comp_changed(self, *args):
            lossy = is_lossy(self._comp_code())
            self.sp_q.setEnabled(lossy)
            self.lb_q.setEnabled(lossy)
            self.rb_rgb.setEnabled(not lossy)
            self.rb_rgba.setEnabled(not lossy)
            self.lb_mask.setVisible(lossy)
            self._refresh_name()

        def _name_tag(self):
            crs = self.crs_widget.crs()
            authid = crs.authid() if crs.isValid() else ""
            epsg = authid.split(":")[-1] if authid.upper().startswith("EPSG:") else "crs"
            comp = self._comp_code()
            tag = "%s_%s" % (epsg, ("%s%d" % (comp.lower(), self.sp_q.value())) if is_lossy(comp) else "otimizado")
            if self._fmt_code() == "COG":
                tag += "_cog"
            return tag

        def _auto_name_for(self, path):
            return "%s_%s.tif" % (os.path.splitext(os.path.basename(path))[0], self._name_tag())

        def _refresh_name(self, force=False):
            if len(self._inputs) != 1:
                return
            cur = self.ed_name.text().strip()
            if force or not cur or cur == self._last_auto_name:
                new = self._auto_name_for(self._inputs[0])
                self.ed_name.setText(new)
                self._last_auto_name = new

        def _pick_input(self):
            start = os.path.dirname(self._inputs[0]) if self._inputs else self.ed_out_dir.text()
            paths, _ = QFileDialog.getOpenFileNames(self, "GeoTIFF(s) de entrada", start,
                                                    "GeoTIFF (*.tif *.tiff *.TIF *.TIFF);;Todos (*.*)")
            if paths:
                self._set_inputs(paths)

        def _on_input_edited(self, text):
            # Digitação manual = um único arquivo.
            self._inputs = [text.strip()] if text.strip() else []
            self.ed_name.setEnabled(True)
            self._content_cache = None
            if self._inputs and os.path.isfile(self._inputs[0]):
                self._refresh_name()

        def _set_inputs(self, paths):
            paths = [os.path.abspath(p) for p in paths]
            self._inputs = paths
            self._content_cache = None
            if not self.ed_out_dir.text():
                self.ed_out_dir.setText(os.path.dirname(paths[0]))
            if len(paths) == 1:
                self.ed_in.setReadOnly(False)
                self.ed_in.setText(paths[0])
                self.ed_name.setEnabled(True)
                self.ed_name.setPlaceholderText("")
                self._refresh_name(force=True)
            else:
                self.ed_in.setReadOnly(True)
                self.ed_in.setText("%d arquivos: %s" % (len(paths), "; ".join(os.path.basename(p) for p in paths)))
                self.ed_name.setText("")
                self.ed_name.setEnabled(False)
                self.ed_name.setPlaceholderText("automático: <nome>_%s.tif (lote)" % self._name_tag())
            for p in paths[:10]:
                self._quick_info(p)
            if len(paths) > 10:
                self._append("... e mais %d arquivo(s)." % (len(paths) - 10))
            self._append("Clique em 'Analisar entrada' para a análise completa (com estimativa de tamanho) "
                         "ou em EXECUTAR.")

        def _quick_info(self, path):
            ds = gdal.Open(path)
            if ds is None:
                self._append("[!] GDAL não abre: %s" % path)
                return
            try:
                d = describe_dataset(ds, path)
                self._append("Entrada: %s | %s | %s x %s | %d bandas | %s | %s | Alpha: %s | overviews: %s"
                             % (os.path.basename(path), fmt_bytes(d["file_size"]), fmt_int(d["width"]),
                                fmt_int(d["height"]), d["bands"], d["crs_label"], d["compression"],
                                ("banda %d" % d["alpha_idx"]) if d["alpha_idx"] else "não",
                                ", ".join(map(str, d["ovr_factors"])) or "nenhuma"))
            finally:
                ds = None

        def _fill_from_active_layer(self, silent=True):
            try:
                lyr = self.iface.activeLayer() if self.iface else None
            except Exception:
                lyr = None
            if lyr is not None and isinstance(lyr, QgsRasterLayer):
                src = lyr.source().split("|")[0]
                if os.path.isfile(src):
                    self._set_inputs([src])
                    return
            if not silent:
                QMessageBox.information(self, SCRIPT_NAME, "A camada ativa não é um raster em arquivo.")

        def _pick_out_dir(self):
            d = QFileDialog.getExistingDirectory(self, "Pasta de saída", self.ed_out_dir.text())
            if d:
                self.ed_out_dir.setText(d)

        def _pick_tmp_dir(self):
            d = QFileDialog.getExistingDirectory(self, "Pasta temporária", self.ed_tmp.text())
            if d:
                self.ed_tmp.setText(d)

        def _open_out_dir(self):
            d = self.ed_out_dir.text().strip()
            if d and os.path.isdir(d):
                QDesktopServices.openUrl(QUrl.fromLocalFile(d))

        def _append(self, text):
            self.log_box.appendPlainText(text)
            self.log_box.moveCursor(QTextCursor.MoveOperation.End)

        def _error(self, msg):
            QMessageBox.critical(self, SCRIPT_NAME, msg)

        def _output_path(self):
            name = self.ed_name.text().strip()
            name = re.sub(r'[<>:"/\\|?*]', "_", name)
            if not name:
                return None
            if not name.lower().endswith((".tif", ".tiff")):
                name += ".tif"
            return os.path.join(self.ed_out_dir.text().strip(), name)

        @staticmethod
        def _suggest_name(path, taken=()):
            base, ext = os.path.splitext(path)
            k = 2
            while os.path.exists("%s_v%d%s" % (base, k, ext)) or \
                    norm_path("%s_v%d%s" % (base, k, ext)) in taken:
                k += 1
            return "%s_v%d%s" % (base, k, ext)

        def _layer_using(self, path):
            for lyr in QgsProject.instance().mapLayers().values():
                try:
                    src = lyr.source().split("|")[0]
                    if src and os.path.exists(src) and same_file(src, path):
                        return lyr
                except Exception:
                    pass
            return None

        def _base_params(self):
            crs = self.crs_widget.crs()
            if not crs.isValid():
                self._error("Selecione um CRS de destino válido.")
                return None
            authid = crs.authid()
            dst = authid if authid.upper().startswith("EPSG:") else crs.toWkt()
            return dict(
                dst_crs=dst,
                res_mode="auto" if self.cb_res.currentIndex() == 0 else "custom",
                res_value=self.sp_res.value(),
                resampling=self.cb_resamp.currentText(),
                compression=self._comp_code(),
                quality=int(self.sp_q.value()),
                output_format=self._fmt_code(),
                tile=int(self.cb_tile.currentText()),
                remove_alpha=self.rb_rgb.isChecked(),
                treat_black_as_nodata=self.ck_black.isChecked(),
                overviews=self.ck_ovr.isChecked(),
                ovr_resampling=self.cb_ovr.currentText(),
                validate=self.ck_val.isChecked(),
                save_report=self.ck_rep.isChecked(),
                temp_dir=self.ed_tmp.text().strip() or None,
                overwrite=False,
                estimate=True,
            )

        def _collect_jobs(self, mode):
            inputs = [p for p in self._inputs if p]
            if not inputs:
                self._error("Selecione um ou mais arquivos de entrada.")
                return None
            missing = [p for p in inputs if not os.path.isfile(p)]
            if missing:
                self._error("Arquivo(s) de entrada não encontrado(s):\n" + "\n".join(missing[:10]))
                return None
            base = self._base_params()
            if base is None:
                return None
            if mode == "analyze":
                jobs = [dict(base, input=p) for p in inputs]
                if len(jobs) == 1:
                    jobs[0]["cached_content"] = self._content_cache
                return jobs
            out_dir = self.ed_out_dir.text().strip()
            if not out_dir or not os.path.isdir(out_dir):
                self._error("Selecione uma pasta de saída existente.")
                return None
            if len(inputs) == 1:
                p = dict(base, input=inputs[0], cached_content=self._content_cache)
                out = self._output_path()
                if not out:
                    self._error("Informe o nome do arquivo de saída.")
                    return None
                if same_file(out, inputs[0]):
                    self._error("O arquivo de saída não pode ser o arquivo de entrada.\n"
                                "O original nunca é sobrescrito.")
                    return None
                if os.path.exists(out):
                    box = QMessageBox(self)
                    box.setIcon(QMessageBox.Icon.Warning)
                    box.setWindowTitle(SCRIPT_NAME)
                    sug = self._suggest_name(out)
                    box.setText("O arquivo de saída já existe:\n%s\n\nO que deseja fazer?" % out)
                    b_new = box.addButton("Usar novo nome: %s" % os.path.basename(sug), QMessageBox.ButtonRole.AcceptRole)
                    b_rep = box.addButton("Substituir", QMessageBox.ButtonRole.DestructiveRole)
                    box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
                    _exec(box)
                    clicked = box.clickedButton()
                    if clicked == b_new:
                        out = sug
                        self.ed_name.setText(os.path.basename(sug))
                        self._last_auto_name = None
                    elif clicked == b_rep:
                        lyr = self._layer_using(out)
                        if lyr is not None:
                            self._error("O arquivo a ser substituído está carregado no projeto (camada '%s').\n"
                                        "Remova a camada do projeto e tente novamente." % lyr.name())
                            return None
                        p["overwrite"] = True
                    else:
                        return None
                p["output"] = out
                return [p]
            # Lote: nome automático por arquivo; nunca sobrescreve (gera _v2, _v3...).
            jobs, taken = [], set()
            for inp in inputs:
                out = os.path.join(out_dir, self._auto_name_for(inp))
                if same_file(out, inp) or os.path.exists(out) or norm_path(out) in taken:
                    out = self._suggest_name(out, taken)
                taken.add(norm_path(out))
                jobs.append(dict(base, input=inp, output=out))
            return jobs

        def _set_running(self, running):
            for w in (self.bt_an, self.bt_run, self.bt_close):
                w.setEnabled(not running)
            self.bt_cancel.setEnabled(running)

        def _start(self, mode):
            if self.worker is not None and self.worker.isRunning():
                return
            jobs = self._collect_jobs(mode)
            if not jobs:
                return
            self._save_settings()
            self.log_box.clear()
            self.pb.setValue(0)
            self._set_running(True)
            self.worker = OptimizerThread(jobs, mode)
            self.worker.sig_log.connect(self._append)
            self.worker.sig_progress.connect(self.pb.setValue)
            self.worker.sig_done.connect(self._done)
            self.worker.start()

        def _cancel(self):
            if self.worker is not None and self.worker.isRunning():
                self._append(">>> Cancelamento solicitado; aguardando o ponto seguro...")
                self.worker.cancel()
                self.bt_cancel.setEnabled(False)

        def _done(self, res):
            # Aguarda a thread terminar de fato antes de soltar a referência
            # (destruir um QThread em execução derruba o QGIS).
            if self.worker is not None:
                self.worker.wait()
            self._set_running(False)
            if res.get("content_cache"):
                self._content_cache = res["content_cache"]
            mode = res.get("mode")
            outputs = res.get("outputs") or []
            if mode == "run" and outputs and self.ck_add.isChecked() and self.iface is not None:
                for out in outputs:
                    self.iface.addRasterLayer(out, os.path.splitext(os.path.basename(out))[0])
            if mode == "run" and outputs:
                try:
                    for out in outputs:
                        QgsMessageLog.logMessage("Ortomosaico otimizado: %s" % out, SCRIPT_NAME)
                except Exception:
                    pass
            if res.get("ok"):
                if mode == "run":
                    self.pb.setValue(100)
                    if res.get("batch"):
                        msg = "Lote concluído com sucesso: %d arquivo(s).%s" % (
                            len(outputs), ("\n\nResumo: %s" % res["report"]) if res.get("report") else "")
                    else:
                        out = res.get("output")
                        extra = ""
                        if res.get("in_size") and res.get("out_size"):
                            extra = "\n\n%s -> %s (%s%%)" % (
                                fmt_bytes(res["in_size"]), fmt_bytes(res["out_size"]),
                                fmt_num((float(res["out_size"]) / res["in_size"] - 1) * 100, 0))
                        msg = "Concluído com sucesso.\n\n%s%s%s" % (
                            out, extra, ("\n\nRelatório: %s" % res["report"]) if res.get("report") else "")
                    QMessageBox.information(self, SCRIPT_NAME, msg)
                else:
                    r = res.get("res")
                    if r:
                        self.sp_res.setValue(r)
            else:
                msg = res.get("error") or "Erro desconhecido."
                if res.get("temp_left"):
                    msg += "\n\nArquivos temporários remanescentes:\n" + "\n".join(res["temp_left"])
                if res.get("batch") and outputs:
                    msg += "\n\n%d arquivo(s) foram gerados com sucesso." % len(outputs)
                self._error(msg)
            self.worker = None

        def closeEvent(self, ev):
            if self.worker is not None and self.worker.isRunning():
                r = QMessageBox.question(self, SCRIPT_NAME,
                                         "Há um processamento em andamento. Cancelar e fechar?")
                if r != QMessageBox.StandardButton.Yes:
                    ev.ignore()
                    return
                self.worker.cancel()
                self.worker.wait()
            ev.accept()


def main():
    if not HAS_QGIS:
        print("Este script deve ser executado dentro do QGIS (Terminal Python).")
        return None
    try:
        from qgis.utils import iface as _iface
    except Exception:
        _iface = None
    g = globals()
    old = g.get("_ORTHO_OPT_DIALOG")
    if old is not None:
        try:
            if old.worker is None:
                old.close()
            else:
                old.show()
                old.raise_()
                return old
        except Exception:
            pass
    parent = _iface.mainWindow() if _iface is not None else None
    dlg = OrthoOptimizerDialog(_iface, parent)
    g["_ORTHO_OPT_DIALOG"] = dlg   # mantém referência (evita coleta pelo GC)
    dlg.show()
    return dlg


main()
