@echo off
REM Abre a GUI do HTRMapper. Basta dar duplo-clique neste arquivo,
REM ou rodar "run_gui.bat" no cmd, dentro da pasta do projeto.
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Ambiente virtual nao encontrado. Rodando configuracao inicial...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -e ".[gui,dev]"
) else (
    call .venv\Scripts\activate.bat
)

htrmapper-gui
