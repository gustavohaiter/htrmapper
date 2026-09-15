from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from htrmapper.core.project import GnssAccuracyConfig, Project, ProjectCrsConfig
from htrmapper.gnss.accuracy import CameraAccuracy
from htrmapper.io.image_import import import_folder
from htrmapper.mvs.dense import MvsConfig, MvsError, _export_to_las, run_dense_reconstruction
from htrmapper.sfm.pipeline import SfmConfig, run_structure_from_motion
from tests.synthetic_scene import SyntheticFlightSpec, generate_synthetic_flight


def test_mvs_config_rejects_unknown_quality():
    with pytest.raises(ValueError):
        MvsConfig(quality="ultra-mega-hd")


def test_export_to_las_writes_real_coordinates_colors_and_crs(tmp_path: Path):
    import laspy
    import pycolmap

    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_point3D(
        np.array([200000.0, 8200000.0, 740.0]), pycolmap.Track(), np.array([255, 128, 0], dtype=np.uint8)
    )
    reconstruction.add_point3D(
        np.array([200001.5, 8200001.5, 741.25]), pycolmap.Track(), np.array([0, 255, 64], dtype=np.uint8)
    )

    out_path = tmp_path / "cloud.las"
    _export_to_las(reconstruction, 31983, out_path)

    las = laspy.read(out_path)
    assert len(las.points) == 2
    xyz = sorted(zip(las.x, las.y, las.z))
    assert xyz[0] == pytest.approx((200000.0, 8200000.0, 740.0), abs=1e-3)
    assert xyz[1] == pytest.approx((200001.5, 8200001.5, 741.25), abs=1e-3)

    crs = las.header.parse_crs()
    assert crs is not None
    assert crs.to_epsg() == 31983


def test_export_to_las_rejects_empty_point_cloud(tmp_path: Path):
    import pycolmap

    reconstruction = pycolmap.Reconstruction()

    with pytest.raises(MvsError, match="zero points"):
        _export_to_las(reconstruction, 31983, tmp_path / "empty.las")


def _aligned_project(tmp_path: Path):
    generate_synthetic_flight(tmp_path / "images", SyntheticFlightSpec())
    records, report = import_folder(tmp_path / "images")
    assert not report.has_problems
    project = Project(
        name="synthetic_flight",
        crs=ProjectCrsConfig(source_epsg=4326, project_epsg=31983),
        gnss_accuracy=GnssAccuracyConfig(accuracy=CameraAccuracy(xy_sigma_m=0.02, z_sigma_m=0.02)),
        images=records,
    )
    sfm_result = run_structure_from_motion(project, tmp_path / "work", SfmConfig(key_point_limit=8000))
    assert sfm_result.success
    return project, Path(sfm_result.reconstruction_path)


def test_raises_when_reconstruction_missing(tmp_path: Path):
    (tmp_path / "images").mkdir()
    project = Project(name="p")

    with pytest.raises(MvsError, match="reconstruction not found"):
        run_dense_reconstruction(project, tmp_path / "does_not_exist", tmp_path / "images", tmp_path / "work")


def test_raises_when_image_root_missing(tmp_path: Path):
    project, reconstruction_path = _aligned_project(tmp_path)

    with pytest.raises(MvsError, match="image folder not found"):
        run_dense_reconstruction(project, reconstruction_path, tmp_path / "no_such_images", tmp_path / "work")


def test_raises_with_clear_message_when_no_cuda(tmp_path: Path):
    """This sandbox genuinely has no CUDA/HIP GPU (confirmed via
    pycolmap.has_cuda during development) -- this test exercises the real
    fail-fast path, not a mock. Full patch-match stereo + fusion can only
    be exercised end-to-end on a machine with an NVIDIA GPU (see
    ARCHITECTURE.md Fase 4 notes)."""
    import pycolmap

    if pycolmap.has_cuda:
        pytest.skip("this machine has CUDA; the no-GPU fail-fast path doesn't apply here")

    project, reconstruction_path = _aligned_project(tmp_path)

    with pytest.raises(MvsError, match="CUDA"):
        run_dense_reconstruction(project, reconstruction_path, tmp_path / "images", tmp_path / "work")
