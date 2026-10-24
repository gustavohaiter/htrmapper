# Cria os atalhos do HTRMapper no Windows (Área de Trabalho e Menu Iniciar).
# Chamado pelo instalar_wsl.sh --criar-atalhos; não precisa de administrador.
param(
    [Parameter(Mandatory = $true)][string]$Distro,
    [Parameter(Mandatory = $true)][string]$WslUser,
    [Parameter(Mandatory = $true)][string]$AppDir
)
$ErrorActionPreference = "Stop"

$dest = Join-Path $env:LOCALAPPDATA "HTRMapper"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -Force (Join-Path $PSScriptRoot "htrmapper.ico") (Join-Path $dest "htrmapper.ico")

$logWin = "\\wsl.localhost\$Distro\home\$WslUser\.local\share\htrmapper\gui.log"
$vbs = @"
' Abre o HTRMapper (que roda dentro do WSL) sem mostrar terminal.
Option Explicit
Dim sh, code
Set sh = CreateObject("WScript.Shell")
code = sh.Run("wsl.exe -d ""$Distro"" -- bash -lc ""~/.local/bin/htrmapper-abrir""", 0, True)
If code <> 0 Then
    MsgBox "O HTRMapper fechou com um erro (código " & code & ")." & vbCrLf & vbCrLf & _
        "Veja o log em:" & vbCrLf & "$logWin", vbExclamation, "HTRMapper"
End If
"@
# VBScript lê ANSI (Windows-1252): acentos certos tanto no PowerShell 5 quanto no 7.
[System.IO.File]::WriteAllText((Join-Path $dest "HTRMapper.vbs"), ($vbs -replace "`r?`n", "`r`n"), [System.Text.Encoding]::GetEncoding(1252))

$update = @"
@echo off
echo Atualizando o HTRMapper (git pull + reinstalacao do ambiente)...
wsl.exe -d "$Distro" -- bash -lc "cd '$AppDir' && git pull && bash windows/instalar_wsl.sh"
echo.
pause
"@
[System.IO.File]::WriteAllText((Join-Path $dest "Atualizar HTRMapper.bat"), ($update -replace "`r?`n", "`r`n"), [System.Text.Encoding]::ASCII)

$shell = New-Object -ComObject WScript.Shell
function New-Shortcut([string]$path, [string]$target, [string]$arguments, [string]$description) {
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $target
    $link.Arguments = $arguments
    $link.WorkingDirectory = $dest
    $link.IconLocation = (Join-Path $dest "htrmapper.ico")
    $link.Description = $description
    $link.Save()
}

$wscript = Join-Path $env:WINDIR "System32\wscript.exe"
$vbsPath = Join-Path $dest "HTRMapper.vbs"
$desktop = [Environment]::GetFolderPath("Desktop")
New-Shortcut (Join-Path $desktop "HTRMapper.lnk") $wscript "`"$vbsPath`"" "HTRMapper - fotogrametria"

$menu = Join-Path ([Environment]::GetFolderPath("Programs")) "HTRMapper"
New-Item -ItemType Directory -Force -Path $menu | Out-Null
New-Shortcut (Join-Path $menu "HTRMapper.lnk") $wscript "`"$vbsPath`"" "HTRMapper - fotogrametria"
New-Shortcut (Join-Path $menu "Atualizar HTRMapper.lnk") (Join-Path $dest "Atualizar HTRMapper.bat") "" "Baixa a versão mais nova do HTRMapper"

Write-Host "Atalhos criados: Área de Trabalho e Menu Iniciar > HTRMapper"
