"""Folder and file choosers that look like the rest of the system.

Inside WSL (the program runs in Linux, shown on Windows through WSLg) Qt
has no Windows dialog of its own: its fallback is a Qt-drawn dialog that
picks up the window's colors and shows Linux paths. So there we ask
Windows itself, through powershell.exe, for the standard Explorer dialog
(IFileOpenDialog for folders -- the modern one, not the old tree view --
and the Windows Forms open/save dialogs for files), and convert the
answer back to a Linux path with `wslpath`. The small C# helper is
compiled once and kept in %LOCALAPPDATA%\\HTRMapper.

Outside WSL, or if PowerShell fails for any reason, the plain QFileDialog
is used (native on Windows and on Linux desktops).

Every function returns the chosen path, or "" when the user cancels.
"""

from __future__ import annotations

import base64
import logging
import re
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QEventLoop, QProcess, QTimer
from PySide6.QtWidgets import QFileDialog, QWidget

from htrmapper.core.platform_paths import is_running_under_wsl

logger = logging.getLogger(__name__)

POWERSHELL_FALLBACK = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
# Bump when _CSHARP changes, so the cached DLL is rebuilt.
HELPER_VERSION = 1
PICK_TIMEOUT_MS = 30 * 60 * 1000

_CSHARP = r"""
using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace HTRMapper {
public static class Dialogs {
    [ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
    internal class FileOpenDialogCom {}

    [ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IFileDialog {
        [PreserveSig] int Show(IntPtr parent);
        void SetFileTypes(uint count, IntPtr filters);
        void SetFileTypeIndex(uint index);
        void GetFileTypeIndex(out uint index);
        void Advise(IntPtr events, out uint cookie);
        void Unadvise(uint cookie);
        void SetOptions(uint options);
        void GetOptions(out uint options);
        void SetDefaultFolder(IShellItem item);
        void SetFolder(IShellItem item);
        void GetFolder(out IShellItem item);
        void GetCurrentSelection(out IShellItem item);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        void GetResult(out IShellItem item);
        void AddPlace(IShellItem item, int place);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
        void Close(int hr);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(IntPtr filter);
    }

    [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IShellItem {
        void BindToHandler(IntPtr bindContext, ref Guid handler, ref Guid riid, out IntPtr result);
        void GetParent(out IShellItem parent);
        void GetDisplayName(uint form, out IntPtr name);
        void GetAttributes(uint mask, out uint attributes);
        void Compare(IShellItem other, uint hint, out int order);
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHCreateItemFromParsingName(string path, IntPtr bindContext, ref Guid riid, out IShellItem item);

    // Invisible, topmost owner: the dialog opens in front of every window
    // (its real parent lives in WSLg and has no Windows handle).
    static Form Owner() {
        Form owner = new Form();
        owner.TopMost = true;
        owner.ShowInTaskbar = false;
        owner.FormBorderStyle = FormBorderStyle.None;
        owner.StartPosition = FormStartPosition.CenterScreen;
        owner.Size = new System.Drawing.Size(1, 1);
        owner.Opacity = 0;
        owner.Show();
        owner.Activate();
        return owner;
    }

    public static string PickFolder(string title, string start) {
        IFileDialog dialog = (IFileDialog)new FileOpenDialogCom();
        uint options;
        dialog.GetOptions(out options);
        // FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST
        dialog.SetOptions(options | 0x20 | 0x40 | 0x800);
        dialog.SetTitle(title);
        if (!String.IsNullOrEmpty(start)) {
            try {
                Guid riid = typeof(IShellItem).GUID;
                IShellItem folder;
                SHCreateItemFromParsingName(start, IntPtr.Zero, ref riid, out folder);
                dialog.SetFolder(folder);
            } catch (Exception) { }
        }
        int hr;
        using (Form owner = Owner()) { hr = dialog.Show(owner.Handle); }
        if (hr != 0) return "";
        IShellItem result;
        dialog.GetResult(out result);
        IntPtr name;
        result.GetDisplayName(0x80058000, out name);  // SIGDN_FILESYSPATH
        string path = Marshal.PtrToStringUni(name);
        Marshal.FreeCoTaskMem(name);
        return path;
    }

    public static string PickFile(string title, string start, string filter, bool save, string fileName) {
        FileDialog dialog = save ? (FileDialog)new SaveFileDialog() : new OpenFileDialog();
        dialog.Title = title;
        dialog.Filter = filter;
        dialog.RestoreDirectory = true;
        if (!String.IsNullOrEmpty(start)) dialog.InitialDirectory = start;
        if (!String.IsNullOrEmpty(fileName)) dialog.FileName = fileName;
        using (Form owner = Owner()) {
            return dialog.ShowDialog(owner) == DialogResult.OK ? dialog.FileName : "";
        }
    }
}
}
"""


def _ps_quote(text: str) -> str:
    """A PowerShell single-quoted string literal."""
    return "'" + text.replace("'", "''") + "'"


def qt_filter_to_windows(qt_filter: str) -> str:
    """'Projeto (*.json);;GeoTIFF (*.tif *.tiff)' -> 'Projeto (*.json)|*.json|GeoTIFF (*.tif *.tiff)|*.tif;*.tiff'."""
    parts = []
    for entry in (e.strip() for e in qt_filter.split(";;")):
        if not entry:
            continue
        match = re.search(r"\(([^)]*)\)", entry)
        patterns = match.group(1).split() if match else ["*.*"]
        parts += [entry, ";".join(patterns)]
    return "|".join(parts) or "Todos os arquivos (*.*)|*.*"


def build_script(call: str) -> str:
    """PowerShell script that loads the helper (compiling it the first time)
    and prints the chosen path as 'OK:<base64 utf-8>' -- ASCII only, so no
    console code page can garble accented folder names -- or 'CANCEL'."""
    dll = f"$env:LOCALAPPDATA\\HTRMapper\\HTRMapperDialogs_v{HELPER_VERSION}.dll"
    return f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$code = @'
{_CSHARP}
'@
$dll = "{dll}"
$refs = @('System.Windows.Forms', 'System.Drawing')
if (-not ('HTRMapper.Dialogs' -as [type])) {{
    try {{
        if (-not (Test-Path $dll)) {{
            New-Item -ItemType Directory -Force -Path (Split-Path $dll) | Out-Null
            Add-Type -TypeDefinition $code -ReferencedAssemblies $refs -OutputAssembly $dll -OutputType Library
        }}
        if (-not ('HTRMapper.Dialogs' -as [type])) {{ Add-Type -Path $dll }}
    }} catch {{
        if (-not ('HTRMapper.Dialogs' -as [type])) {{ Add-Type -TypeDefinition $code -ReferencedAssemblies $refs }}
    }}
}}
[System.Windows.Forms.Application]::EnableVisualStyles()
$path = {call}
if ($path) {{
    'OK:' + [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($path))
}} else {{
    'CANCEL'
}}
"""


def parse_output(stdout: str) -> str | None:
    """The chosen Windows path, "" if cancelled, None if the output is not
    the helper's (an error)."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line == "CANCEL":
            return ""
        if line.startswith("OK:"):
            return base64.b64decode(line[3:]).decode("utf-8")
    return None


def _powershell() -> str | None:
    found = shutil.which("powershell.exe")
    if found:
        return found
    return POWERSHELL_FALLBACK if Path(POWERSHELL_FALLBACK).is_file() else None


def _to_windows(path: str) -> str:
    if not path:
        return ""
    try:
        return subprocess.run(
            ["wslpath", "-w", path], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _to_linux(path: str) -> str:
    return subprocess.run(["wslpath", "-u", path], capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def _run_windows_dialog(parent: QWidget | None, call: str) -> str | None:
    """Runs the dialog without freezing the Qt window (it keeps repainting
    while the user browses); the window is disabled meanwhile."""
    exe = _powershell()
    if exe is None:
        return None
    encoded = base64.b64encode(build_script(call).encode("utf-16-le")).decode("ascii")
    process = QProcess()
    loop = QEventLoop()
    process.finished.connect(loop.quit)
    process.errorOccurred.connect(lambda _error: loop.quit())
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    window = parent.window() if parent is not None else None
    if window is not None:
        window.setEnabled(False)
    try:
        arguments = ["-NoProfile", "-NonInteractive", "-STA", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded]
        process.start(exe, arguments)
        if not process.waitForStarted(15000):
            logger.warning("powershell.exe não iniciou: %s", process.errorString())
            return None
        timeout.start(PICK_TIMEOUT_MS)
        if process.state() != QProcess.ProcessState.NotRunning:
            loop.exec()
        if process.state() != QProcess.ProcessState.NotRunning:
            process.kill()
            process.waitForFinished(3000)
            return None
        stdout = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        chosen = parse_output(stdout)
        if chosen is None:
            stderr = bytes(process.readAllStandardError()).decode("utf-8", "replace")
            logger.warning("diálogo do Windows falhou: %s", (stderr or stdout).strip()[-2000:])
            return None
        return _to_linux(chosen) if chosen else ""
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("diálogo do Windows falhou: %s", exc)
        return None
    finally:
        if window is not None:
            window.setEnabled(True)
            window.activateWindow()


def _start_dir(start: str) -> str:
    path = Path(start) if start else None
    while path is not None and not path.is_dir():
        path = path.parent if path.parent != path else None
    return str(path) if path is not None else ""


def choose_folder(parent: QWidget | None, title: str, start: str = "") -> str:
    start = _start_dir(start)
    if is_running_under_wsl():
        call = f"[HTRMapper.Dialogs]::PickFolder({_ps_quote(title)}, {_ps_quote(_to_windows(start))})"
        chosen = _run_windows_dialog(parent, call)
        if chosen is not None:
            return chosen
    # parent=None: the fallback dialog does not inherit the window's colors.
    return QFileDialog.getExistingDirectory(None, title, start)


def choose_open_file(parent: QWidget | None, title: str, start: str = "", qt_filter: str = "") -> str:
    folder = _start_dir(start)
    if is_running_under_wsl():
        call = (
            f"[HTRMapper.Dialogs]::PickFile({_ps_quote(title)}, {_ps_quote(_to_windows(folder))}, "
            f"{_ps_quote(qt_filter_to_windows(qt_filter))}, $false, '')"
        )
        chosen = _run_windows_dialog(parent, call)
        if chosen is not None:
            return chosen
    return QFileDialog.getOpenFileName(None, title, start, qt_filter)[0]


def choose_save_file(parent: QWidget | None, title: str, start: str = "", qt_filter: str = "") -> str:
    """`start` may be a full path: its folder opens and its name is suggested."""
    folder = _start_dir(start)
    name = Path(start).name if start and not Path(start).is_dir() else ""
    if is_running_under_wsl():
        call = (
            f"[HTRMapper.Dialogs]::PickFile({_ps_quote(title)}, {_ps_quote(_to_windows(folder))}, "
            f"{_ps_quote(qt_filter_to_windows(qt_filter))}, $true, {_ps_quote(name)})"
        )
        chosen = _run_windows_dialog(parent, call)
        if chosen is not None:
            return chosen
    return QFileDialog.getSaveFileName(None, title, start, qt_filter)[0]
