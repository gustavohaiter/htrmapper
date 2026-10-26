"""Folder/file choosers (gui.file_dialogs): the Windows dialog through
PowerShell inside WSL, and the QFileDialog fallback."""

from __future__ import annotations

import base64
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from htrmapper.gui import file_dialogs  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _app():
    return QApplication.instance() or QApplication([])


def test_qt_filter_to_windows():
    assert file_dialogs.qt_filter_to_windows("Projeto HTRMapper (*.json)") == "Projeto HTRMapper (*.json)|*.json"
    assert (
        file_dialogs.qt_filter_to_windows("GeoTIFF (*.tif *.tiff);;Todos (*)")
        == "GeoTIFF (*.tif *.tiff)|*.tif;*.tiff|Todos (*)|*"
    )
    assert file_dialogs.qt_filter_to_windows("") == "Todos os arquivos (*.*)|*.*"


def test_parse_output_handles_accents_cancel_and_garbage():
    encoded = base64.b64encode("C:\\Users\\joão\\Área de Trabalho\\Fotos".encode()).decode()
    assert file_dialogs.parse_output(f"aviso qualquer\r\nOK:{encoded}\r\n") == "C:\\Users\\joão\\Área de Trabalho\\Fotos"
    assert file_dialogs.parse_output("CANCEL\r\n") == ""
    assert file_dialogs.parse_output("Add-Type : erro de compilação") is None


def test_script_quotes_arguments_and_embeds_the_helper():
    title = file_dialogs._ps_quote("Pasta d'água")
    call = f"[HTRMapper.Dialogs]::PickFolder({title}, '')"
    script = file_dialogs.build_script(call)
    assert "PickFolder('Pasta d''água', '')" in script
    assert "FOS_PICKFOLDERS" in script and "IFileDialog" in script
    assert f"HTRMapperDialogs_v{file_dialogs.HELPER_VERSION}.dll" in script
    # The here-string must close on its own line, or PowerShell reads the rest as C#.
    assert "\n'@\n" in script


def test_outside_wsl_uses_qfiledialog(monkeypatch, tmp_path):
    monkeypatch.setattr(file_dialogs, "is_running_under_wsl", lambda: False)
    seen = {}

    def fake(parent, title, start):
        seen.update(parent=parent, title=title, start=start)
        return str(tmp_path)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(fake))
    assert file_dialogs.choose_folder(None, "Fotos", str(tmp_path / "nao_existe" / "x")) == str(tmp_path)
    assert seen["start"] == str(tmp_path)  # nearest existing folder
    assert seen["parent"] is None  # no inherited window colors


def _fake_powershell(tmp_path, output: str):
    script = tmp_path / "powershell.exe"
    script.write_text(f"#!/bin/sh\nprintf '%s\\r\\n' '{output}'\n")
    script.chmod(0o755)
    return str(script)


def test_wsl_uses_the_windows_dialog(monkeypatch, tmp_path):
    windows_path = "C:\\Users\\joão\\Fotos"
    encoded = base64.b64encode(windows_path.encode()).decode()
    monkeypatch.setattr(file_dialogs, "is_running_under_wsl", lambda: True)
    monkeypatch.setattr(file_dialogs, "_powershell", lambda: _fake_powershell(tmp_path, f"OK:{encoded}"))
    monkeypatch.setattr(file_dialogs, "_to_windows", lambda p: "C:\\start")
    monkeypatch.setattr(file_dialogs, "_to_linux", lambda p: "/mnt/c/Users/joão/Fotos" if p == windows_path else "?")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a: pytest.fail("fallback used")))
    assert file_dialogs.choose_folder(None, "Fotos", str(tmp_path)) == "/mnt/c/Users/joão/Fotos"


def test_wsl_cancel_does_not_open_a_second_dialog(monkeypatch, tmp_path):
    monkeypatch.setattr(file_dialogs, "is_running_under_wsl", lambda: True)
    monkeypatch.setattr(file_dialogs, "_powershell", lambda: _fake_powershell(tmp_path, "CANCEL"))
    monkeypatch.setattr(file_dialogs, "_to_windows", lambda p: "")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a: pytest.fail("fallback used")))
    assert file_dialogs.choose_folder(None, "Fotos") == ""


def test_wsl_failure_falls_back_to_qfiledialog(monkeypatch, tmp_path):
    monkeypatch.setattr(file_dialogs, "is_running_under_wsl", lambda: True)
    monkeypatch.setattr(file_dialogs, "_powershell", lambda: _fake_powershell(tmp_path, "erro: Add-Type falhou"))
    monkeypatch.setattr(file_dialogs, "_to_windows", lambda p: "")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a: ("/tmp/x.json", "")))
    assert file_dialogs.choose_save_file(None, "Salvar", "projeto.json", "Projeto (*.json)") == "/tmp/x.json"
