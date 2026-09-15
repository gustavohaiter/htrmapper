from __future__ import annotations

from htrmapper.core.system_info import GpuInfo, SystemInfo, detect_system_info


def test_detect_system_info_does_not_raise_and_has_plausible_values():
    info = detect_system_info()

    assert info.os_name
    assert info.cpu_name
    assert info.cpu_logical_cores >= 1
    assert info.ram_total_gb > 0
    assert isinstance(info.gpus, tuple)


def test_has_nvidia_gpu_false_when_no_gpus():
    info = SystemInfo(os_name="Linux", cpu_name="test", cpu_logical_cores=4, ram_total_gb=16.0, gpus=())

    assert info.has_nvidia_gpu is False


def test_has_nvidia_gpu_true_when_gpus_present():
    gpu = GpuInfo(name="RTX 3060 Ti", vram_total_mb=8192.0, driver_version="535.86")
    info = SystemInfo(os_name="Linux", cpu_name="test", cpu_logical_cores=8, ram_total_gb=24.0, gpus=(gpu,))

    assert info.has_nvidia_gpu is True


def test_to_dict_round_trips_gpu_fields():
    gpu = GpuInfo(name="RTX 3060 Ti", vram_total_mb=8192.0, driver_version="535.86")
    info = SystemInfo(os_name="Linux", cpu_name="test", cpu_logical_cores=8, ram_total_gb=24.0, gpus=(gpu,))

    data = info.to_dict()

    assert data["gpus"][0]["name"] == "RTX 3060 Ti"
    assert data["gpus"][0]["vram_total_mb"] == 8192.0
