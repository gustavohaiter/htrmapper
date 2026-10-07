from __future__ import annotations

import json
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


def test_undistorted_paths_point_at_the_real_images_and_sparse_subfolders(tmp_path: Path, monkeypatch):
    """Regression test for a real bug: `MvsResult.undistorted_image_path`
    used to be set to the whole dense workspace directory (e.g.
    `.../dense`) rather than the `images` subfolder `undistort_images`
    actually writes images into (`.../dense/images`) -- Fase 6's
    orthomosaic step needs the exact subfolder, not its parent, to find
    images by name. This sandbox has no CUDA, so `patch_match_stereo`
    (the very next call) is expected to fail -- but `undistort_images`
    itself runs entirely on CPU (confirmed in Fase 4/6 development), so by
    monkeypatching only the fast-fail CUDA check, undistortion still runs
    for real and we can verify its actual on-disk output layout before the
    (expected, unavoidable without a GPU) failure."""
    import pycolmap

    monkeypatch.setattr(pycolmap, "has_cuda", True)
    project, reconstruction_path = _aligned_project(tmp_path)
    workdir = tmp_path / "work_dense"

    with pytest.raises(ValueError, match="CUDA"):
        run_dense_reconstruction(project, reconstruction_path, tmp_path / "images", workdir)

    dense_workspace = workdir / "dense"
    assert (dense_workspace / "images").is_dir()
    assert (dense_workspace / "sparse").is_dir()
    assert any((dense_workspace / "images").iterdir())
    assert (dense_workspace / "sparse" / "cameras.bin").exists()


def test_experimental_patch_match_knobs_are_passed_through_to_pycolmap(tmp_path: Path, monkeypatch):
    """Regression test for the experimental sparse-cloud-density knobs
    (ARCHITECTURE.md seção 24): `MvsConfig.num_patch_match_src_images` must
    reach the real `pycolmap.undistort_images` call, and
    `filter_min_triangulation_angle`/`filter_min_ncc` must reach the real
    `PatchMatchOptions` object -- not just exist on the dataclass. Uses the
    same monkeypatch-only-the-CUDA-check trick as
    `test_undistorted_paths_point_at_the_real_images_and_sparse_subfolders`:
    `undistort_images` runs for real (CPU-only), `patch_match_stereo` is
    expected to fail fast afterwards since this sandbox has no GPU."""
    import pycolmap

    monkeypatch.setattr(pycolmap, "has_cuda", True)

    captured_undistort_kwargs = {}
    real_undistort_images = pycolmap.undistort_images

    def _spy_undistort_images(*args, **kwargs):
        captured_undistort_kwargs.update(kwargs)
        return real_undistort_images(*args, **kwargs)

    monkeypatch.setattr(pycolmap, "undistort_images", _spy_undistort_images)

    captured_patch_match_options = {}
    real_patch_match_stereo = pycolmap.patch_match_stereo

    def _spy_patch_match_stereo(*args, **kwargs):
        options = kwargs.get("options")
        captured_patch_match_options["filter_min_triangulation_angle"] = options.filter_min_triangulation_angle
        captured_patch_match_options["filter_min_ncc"] = options.filter_min_ncc
        return real_patch_match_stereo(*args, **kwargs)

    monkeypatch.setattr(pycolmap, "patch_match_stereo", _spy_patch_match_stereo)

    project, reconstruction_path = _aligned_project(tmp_path)
    config = MvsConfig(
        quality="baixa",
        num_patch_match_src_images=40,
        filter_min_triangulation_angle=1.5,
        filter_min_ncc=0.05,
    )

    with pytest.raises(ValueError, match="CUDA"):
        run_dense_reconstruction(project, reconstruction_path, tmp_path / "images", tmp_path / "work", config)

    assert captured_undistort_kwargs["num_patch_match_src_images"] == 40
    assert captured_patch_match_options["filter_min_triangulation_angle"] == pytest.approx(1.5)
    assert captured_patch_match_options["filter_min_ncc"] == pytest.approx(0.05)


def test_patch_match_knobs_default_to_pycolmaps_own_defaults_when_unset(tmp_path: Path, monkeypatch):
    """When the user doesn't set the experimental knobs, behavior must be
    byte-for-byte identical to before they existed -- `num_patch_match_src_images`
    passed as pycolmap's own -1 sentinel, and the filter options left
    untouched on a fresh `PatchMatchOptions()`."""
    import pycolmap

    monkeypatch.setattr(pycolmap, "has_cuda", True)

    captured_undistort_kwargs = {}
    real_undistort_images = pycolmap.undistort_images

    def _spy_undistort_images(*args, **kwargs):
        captured_undistort_kwargs.update(kwargs)
        return real_undistort_images(*args, **kwargs)

    monkeypatch.setattr(pycolmap, "undistort_images", _spy_undistort_images)

    project, reconstruction_path = _aligned_project(tmp_path)

    with pytest.raises(ValueError, match="CUDA"):
        run_dense_reconstruction(
            project, reconstruction_path, tmp_path / "images", tmp_path / "work", MvsConfig(quality="baixa")
        )

    assert captured_undistort_kwargs["num_patch_match_src_images"] == -1


def test_mvs_runs_in_a_local_frame_and_exports_absolute_coordinates(tmp_path: Path, monkeypatch):
    """Regression test for the float32-precision root cause found on the
    first real flight (ARCHITECTURE.md seção 25): COLMAP's MVS must never
    see absolute UTM coordinates (the synthetic scene sits at northing
    8.2e6, the same order as the real one, where float32 resolves only
    0.5 m). This sandbox has no GPU, so patch-match is stubbed and fusion
    is faked by handing back the undistorted *local* sparse model's own
    points -- which lets us check the full round trip without a GPU: what
    goes into undistortion is local, and what comes out (LAS, the model
    Phase 6 consumes) is back in the project CRS, exactly."""
    import laspy
    import pycolmap

    project, reconstruction_path = _aligned_project(tmp_path)
    absolute = pycolmap.Reconstruction(str(reconstruction_path))
    absolute_centers = {
        absolute.image(i).name: np.array(absolute.image(i).projection_center()) for i in absolute.reg_image_ids()
    }
    absolute_points = np.array(sorted(tuple(p.xyz) for p in absolute.points3D.values()))
    assert np.abs(absolute_points[:, 1]).min() > 1e6  # really in absolute UTM-like coordinates

    monkeypatch.setattr(pycolmap, "has_cuda", True)
    seen_input_centers = []
    real_undistort_images = pycolmap.undistort_images

    def _spy_undistort_images(*args, **kwargs):
        model = pycolmap.Reconstruction(str(kwargs["input_path"]))
        seen_input_centers.extend(np.array(model.image(i).projection_center()) for i in model.reg_image_ids())
        return real_undistort_images(*args, **kwargs)

    def _fake_fusion(*args, **kwargs):
        return pycolmap.Reconstruction(str(Path(kwargs["workspace_path"]) / "sparse"))

    monkeypatch.setattr(pycolmap, "undistort_images", _spy_undistort_images)
    monkeypatch.setattr(pycolmap, "patch_match_stereo", lambda *a, **k: None)
    monkeypatch.setattr(pycolmap, "stereo_fusion", _fake_fusion)

    workdir = tmp_path / "work_dense"
    result = run_dense_reconstruction(project, reconstruction_path, tmp_path / "images", workdir, MvsConfig())

    # What COLMAP's MVS sees: small, local coordinates only.
    assert seen_input_centers
    assert max(np.abs(c[:2]).max() for c in seen_input_centers) < 1_000.0

    # The origin is persisted, horizontal-only and near the cameras' centroid.
    origin = np.array(result.local_origin_m)
    assert origin[2] == 0.0
    assert np.allclose(origin[:2], np.mean(list(absolute_centers.values()), axis=0)[:2], atol=1.0)
    assert json.loads((workdir / "local_origin.json").read_text())["local_origin_m"] == result.local_origin_m

    # The model Phase 6 consumes is back in the project CRS, exactly.
    georef = pycolmap.Reconstruction(result.undistorted_reconstruction_path)
    for image_id in georef.reg_image_ids():
        image = georef.image(image_id)
        assert np.allclose(np.array(image.projection_center()), absolute_centers[image.name], atol=1e-6)

    # And so is the exported LAS. Matched by nearest neighbor, not by sort
    # order: LAS quantizes to 0.1 mm, which can swap near-tied points.
    from scipy.spatial import cKDTree

    las = laspy.read(result.point_cloud_las_path)
    las_points = np.column_stack([las.x, las.y, las.z])
    assert las_points.shape == absolute_points.shape
    distances, _ = cKDTree(absolute_points).query(las_points)
    assert distances.max() < 1e-3


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
