"""Hardware detection for the processing report and for GPU/CPU dispatch.

Per the project brief: NVIDIA GPU / CUDA / VRAM must be detected
automatically, and CUDA must never be *required* -- a CPU-only machine is
a supported fallback, not an error. This module only detects and reports
hardware; it does not itself decide which backend to run (that decision
belongs to each processing stage in later phases, which will consult this
module and fall back to CPU when no compatible GPU is found).
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_total_mb: float
    driver_version: str | None = None


@dataclass(frozen=True)
class GpuUsage:
    name: str
    utilization_percent: float
    vram_used_mb: float
    vram_total_mb: float


@dataclass(frozen=True)
class LiveUsage:
    """A point-in-time hardware usage sample, for the Fase 7 GUI's live
    CPU/RAM/GPU/VRAM monitor. Never a static capability descriptor like
    `SystemInfo` above -- this is meant to be polled repeatedly (e.g. by a
    Qt timer) and always reflects hardware actually queried at call time.
    `gpus` is empty on a machine with no NVIDIA GPU, exactly like
    `SystemInfo.gpus` -- the GUI must render that as "sem GPU", never as
    0% usage (which would misrepresent "not measured" as "measured, idle")."""

    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpus: tuple[GpuUsage, ...]


@dataclass(frozen=True)
class SystemInfo:
    os_name: str
    cpu_name: str
    cpu_logical_cores: int
    ram_total_gb: float
    gpus: tuple[GpuInfo, ...]

    @property
    def has_nvidia_gpu(self) -> bool:
        return len(self.gpus) > 0

    def to_dict(self) -> dict:
        return {
            "os_name": self.os_name,
            "cpu_name": self.cpu_name,
            "cpu_logical_cores": self.cpu_logical_cores,
            "ram_total_gb": self.ram_total_gb,
            "gpus": [
                {"name": g.name, "vram_total_mb": g.vram_total_mb, "driver_version": g.driver_version}
                for g in self.gpus
            ],
        }


def _detect_cpu_name() -> str:
    # platform.processor() is often empty on Linux; /proc/cpuinfo has the
    # real model name there. Fall back gracefully if unavailable (e.g. a
    # sandboxed environment, or a non-Linux OS where processor() works).
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _detect_nvidia_gpus() -> tuple[GpuInfo, ...]:
    """Query nvidia-smi for GPU name, VRAM and driver version.

    Returns an empty tuple (not an error) when nvidia-smi is absent or
    fails -- this is the expected, supported case on a CPU-only machine.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ()

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, vram_mb_str, driver_version = parts
        try:
            vram_mb = float(vram_mb_str)
        except ValueError:
            continue
        gpus.append(GpuInfo(name=name, vram_total_mb=vram_mb, driver_version=driver_version))
    return tuple(gpus)


def detect_system_info() -> SystemInfo:
    return SystemInfo(
        os_name=f"{platform.system()} {platform.release()}",
        cpu_name=_detect_cpu_name(),
        cpu_logical_cores=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=psutil.virtual_memory().total / (1024**3),
        gpus=_detect_nvidia_gpus(),
    )


def _sample_nvidia_gpu_usage() -> tuple[GpuUsage, ...]:
    """Query nvidia-smi for a live utilization/VRAM sample. Same
    fail-quiet-to-empty contract as `_detect_nvidia_gpus`: an empty tuple
    means "no NVIDIA GPU here", the expected, supported case."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ()

    usages = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        name, util_str, used_mb_str, total_mb_str = parts
        try:
            usages.append(
                GpuUsage(
                    name=name,
                    utilization_percent=float(util_str),
                    vram_used_mb=float(used_mb_str),
                    vram_total_mb=float(total_mb_str),
                )
            )
        except ValueError:
            continue
    return tuple(usages)


def sample_cpu_ram_usage() -> tuple[float, float, float]:
    """CPU percent + RAM used/total, with no subprocess involved at all
    (`psutil` reads `/proc` directly on Linux) -- safe to call at any time,
    including while other heavily multi-threaded native code (e.g. a
    Fase 2-6 pipeline stage) is running in a background thread. See
    `sample_live_usage`'s docstring for why the GPU sample is NOT part of
    this function."""
    vm = psutil.virtual_memory()
    return psutil.cpu_percent(interval=None), vm.used / (1024**3), vm.total / (1024**3)


def sample_live_usage() -> LiveUsage:
    """A single point-in-time CPU/RAM/GPU/VRAM sample for the GUI's live
    monitor. `psutil.cpu_percent(interval=None)` reports the delta since
    the previous call in this process (0.0 on the very first call) -- the
    GUI polls this repeatedly on a timer, which is exactly the usage
    pattern that makes that delta meaningful.

    Callers running a heavily multi-threaded native pipeline stage (any of
    the Fase 2-6 `run_*` functions) in a background thread should use
    `sample_cpu_ram_usage()` instead of this function for the duration of
    that stage: `_sample_nvidia_gpu_usage()` below spawns `nvidia-smi` via
    `subprocess.run`, which on POSIX forks the whole process. Forking a
    process that has other threads deep inside heavy native (C/C++) code
    risks the classic fork()-in-a-multithreaded-process deadlock -- a
    lock held by one of those other threads at the moment of the fork is
    copied into the child in its "held" state, but the thread that would
    release it does not exist there, so the child (and, waiting on it,
    the caller) can hang forever. This was reproduced empirically while
    testing the Fase 7 GUI (its monitor timer polling this function once a
    second, concurrently with a background SfM worker) before this split
    was introduced."""
    cpu_percent, ram_used_gb, ram_total_gb = sample_cpu_ram_usage()
    return LiveUsage(
        cpu_percent=cpu_percent,
        ram_used_gb=ram_used_gb,
        ram_total_gb=ram_total_gb,
        gpus=_sample_nvidia_gpu_usage(),
    )
