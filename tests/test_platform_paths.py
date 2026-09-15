"""Windows <-> WSL2 path translation, for the Fase 4-6 GPU workflow.

Pure string-manipulation tests -- this development sandbox is plain
Linux, not WSL, so `is_running_under_wsl()` is monkeypatched directly
(bypassing its `lru_cache`) to deterministically exercise both sides of
the translation without needing a real WSL2 environment, which is not
available here. What genuinely can't be tested in this sandbox -- that
WSL2's own `/mnt/<drive>` mount behaves as NVIDIA/Microsoft document -- is
not this module's concern; see ARCHITECTURE.md for what still needs the
user's own hardware to validate.
"""

from __future__ import annotations

import htrmapper.core.platform_paths as platform_paths
from htrmapper.core.platform_paths import normalize_path_for_current_platform


def test_windows_path_translated_to_wsl_mount_when_running_under_wsl(monkeypatch):
    monkeypatch.setattr(platform_paths, "is_running_under_wsl", lambda: True)

    result = normalize_path_for_current_platform(r"C:\Users\gustavo.haiter\Documents\htrmapper\photo.jpg")

    assert result == "/mnt/c/Users/gustavo.haiter/Documents/htrmapper/photo.jpg"


def test_windows_path_with_forward_slashes_also_translated(monkeypatch):
    monkeypatch.setattr(platform_paths, "is_running_under_wsl", lambda: True)

    result = normalize_path_for_current_platform("D:/drone/imagens/DJI_0001.JPG")

    assert result == "/mnt/d/drone/imagens/DJI_0001.JPG"


def test_uppercase_and_lowercase_drive_letters_both_translated(monkeypatch):
    monkeypatch.setattr(platform_paths, "is_running_under_wsl", lambda: True)

    assert normalize_path_for_current_platform(r"E:\data").startswith("/mnt/e/")
    assert normalize_path_for_current_platform(r"e:\data").startswith("/mnt/e/")


def test_already_wsl_style_path_left_unchanged_when_running_under_wsl(monkeypatch):
    monkeypatch.setattr(platform_paths, "is_running_under_wsl", lambda: True)

    result = normalize_path_for_current_platform("/mnt/c/Users/gustavo.haiter/photo.jpg")

    assert result == "/mnt/c/Users/gustavo.haiter/photo.jpg"


def test_native_linux_path_left_unchanged_when_running_under_wsl(monkeypatch):
    # A project created and re-run entirely within WSL (no Windows side
    # involved) must not have its ordinary Linux paths mangled.
    monkeypatch.setattr(platform_paths, "is_running_under_wsl", lambda: True)

    result = normalize_path_for_current_platform("/home/user/htrmapper/work/sparse")

    assert result == "/home/user/htrmapper/work/sparse"


def test_wsl_mount_path_translated_to_windows_when_running_on_native_windows(monkeypatch):
    monkeypatch.setattr(platform_paths, "is_running_under_wsl", lambda: False)
    monkeypatch.setattr(platform_paths.os, "name", "nt")

    result = normalize_path_for_current_platform("/mnt/c/Users/gustavo.haiter/work/sparse")

    assert result == r"C:\Users\gustavo.haiter\work\sparse"


def test_windows_path_left_unchanged_on_native_windows(monkeypatch):
    monkeypatch.setattr(platform_paths, "is_running_under_wsl", lambda: False)
    monkeypatch.setattr(platform_paths.os, "name", "nt")

    result = normalize_path_for_current_platform(r"C:\Users\gustavo.haiter\photo.jpg")

    assert result == r"C:\Users\gustavo.haiter\photo.jpg"


def test_path_unchanged_on_plain_linux_not_under_wsl(monkeypatch):
    monkeypatch.setattr(platform_paths, "is_running_under_wsl", lambda: False)
    monkeypatch.setattr(platform_paths.os, "name", "posix")

    # A Windows-style path arriving on plain Linux (not WSL) has nowhere
    # sensible to translate to -- left as-is rather than guessing.
    result = normalize_path_for_current_platform(r"C:\Users\gustavo.haiter\photo.jpg")

    assert result == r"C:\Users\gustavo.haiter\photo.jpg"


def test_empty_path_passes_through():
    assert normalize_path_for_current_platform("") == ""


def test_relative_path_never_mistaken_for_a_windows_drive_path(monkeypatch):
    monkeypatch.setattr(platform_paths, "is_running_under_wsl", lambda: True)

    result = normalize_path_for_current_platform("images/DJI_0001.JPG")

    assert result == "images/DJI_0001.JPG"
