@echo off
REM Instalador do HTRMapper. Duplo clique: pede permissao de administrador e roda o instalar_htrmapper.ps1.
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)
pushd "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0instalar_htrmapper.ps1"
popd
echo.
pause
