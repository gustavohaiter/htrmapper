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
