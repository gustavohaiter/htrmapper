@echo off
REM Abre a GUI do HTRMapper. Basta dar duplo-clique neste arquivo,
REM ou rodar "run_gui.bat" no cmd, dentro da pasta do projeto.
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Ambiente virtual nao encontrado. Criando...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

REM Reinstala/atualiza sempre, nao so na primeira vez: cada fase nova pode
REM ter adicionado uma dependencia (ex. pycolmap na Fase 2, rasterio na
REM Fase 5) que um .venv criado antes dela nao tem. pip nao reinstala o
REM que ja esta satisfeito, entao isso e rapido quando nada mudou.
pip install -e ".[gui,dev]"

htrmapper-gui
