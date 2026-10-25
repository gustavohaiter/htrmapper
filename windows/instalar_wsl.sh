#!/usr/bin/env bash
# Instala (ou atualiza) o HTRMapper dentro do WSL (Ubuntu).
#
# Uso, dentro do WSL:
#   bash windows/instalar_wsl.sh                  # a partir da pasta do código
#   bash windows/instalar_wsl.sh --criar-atalhos  # e cria os atalhos no Windows
#
# O que faz (pode rodar quantas vezes quiser; o que já está pronto é mantido):
#   1. Pacotes do Ubuntu: Python, venv, git e as bibliotecas que a janela (Qt) precisa no WSLg.
#   2. Código em ~/htrmapper (copiado desta pasta se ainda não existir lá).
#   3. Ambiente Python (.venv) com o HTRMapper, a interface e o pycolmap COM GPU (pycolmap-cuda12).
#   4. Confere se a GPU NVIDIA é vista pelo pycolmap.
#   5. Cria ~/.local/bin/htrmapper-abrir, que o atalho do Windows usa para abrir a janela.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${HTRMAPPER_DIR:-$HOME/htrmapper}"
CREATE_SHORTCUTS=0
for arg in "$@"; do
    case "$arg" in
        --criar-atalhos) CREATE_SHORTCUTS=1 ;;
        *) echo "Opção desconhecida: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mAVISO: %s\033[0m\n' "$*"; }

APT_MARKER="$HOME/.local/share/htrmapper/pacotes_ubuntu_ok"
if [ -f "$APT_MARKER" ] && python3 -m venv --help >/dev/null 2>&1; then
    say "1/5 Pacotes do Ubuntu: já instalados (apague $APT_MARKER para reinstalar)"
else
say "1/5 Pacotes do Ubuntu (pede a senha do Linux)"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git rsync
# Bibliotecas da janela Qt no WSLg. Instaladas uma a uma: os nomes mudam
# entre versões do Ubuntu, e a falta de uma não deve parar a instalação.
for pkg in libgl1 libegl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 \
    libxcb-xkb1 libfontconfig1 libdbus-1-3 libglib2.0-0t64 libglib2.0-0 fonts-dejavu-core; do
    sudo apt-get install -y "$pkg" >/dev/null 2>&1 || true
done
mkdir -p "$(dirname "$APT_MARKER")" && touch "$APT_MARKER"
fi

say "2/5 Código do HTRMapper em $APP_DIR"
if [ "$SRC_DIR" != "$APP_DIR" ]; then
    if [ -d "$APP_DIR/.git" ] || [ -f "$APP_DIR/pyproject.toml" ]; then
        echo "Já existe em $APP_DIR; mantido como está (para atualizar: cd $APP_DIR && git pull)."
    else
        mkdir -p "$APP_DIR"
        rsync -a --exclude ".venv" --exclude "__pycache__" "$SRC_DIR/" "$APP_DIR/"
        echo "Copiado de $SRC_DIR"
    fi
fi
cd "$APP_DIR"

say "3/5 Ambiente Python (.venv)"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -e ".[gui,cuda]"

say "4/5 Conferindo a GPU"
if ! python -c "import pycolmap, sys; sys.exit(0 if pycolmap.has_cuda else 1)"; then
    warn "o pycolmap instalado não tem CUDA; reinstalando a versão com GPU."
    pip uninstall -y pycolmap pycolmap-cuda12 >/dev/null 2>&1 || true
    pip install "pycolmap-cuda12>=4.0,<5"
fi
if python -c "import pycolmap, sys; sys.exit(0 if pycolmap.has_cuda else 1)"; then
    python -c "import pycolmap; print('pycolmap', pycolmap.__version__, '- GPU CUDA: OK')"
else
    warn "o pycolmap continua sem GPU. Confira se o driver NVIDIA do WINDOWS está atualizado e se 'nvidia-smi' funciona aqui no WSL. Sem GPU a nuvem densa não roda."
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    warn "nvidia-smi não encontrado no WSL (o driver NVIDIA do Windows é que fornece)."
fi

say "5/5 Lançador da janela"
mkdir -p "$HOME/.local/bin" "$HOME/.local/share/htrmapper"
cat > "$HOME/.local/bin/htrmapper-abrir" <<EOF
#!/usr/bin/env bash
# Abre a janela do HTRMapper (usado pelo atalho do Windows). Saída em ~/.local/share/htrmapper/gui.log
LOG="\$HOME/.local/share/htrmapper/gui.log"
{ echo; echo "=== \$(date '+%Y-%m-%d %H:%M:%S') ==="; } >> "\$LOG"
cd "$APP_DIR" || exit 3
source .venv/bin/activate || exit 4
# X11 (XWayland) no WSLg: janela com a barra de título normal do Windows.
export QT_QPA_PLATFORM="\${QT_QPA_PLATFORM:-xcb}"
exec htrmapper-gui >> "\$LOG" 2>&1
EOF
chmod +x "$HOME/.local/bin/htrmapper-abrir"
echo "Criado: $HOME/.local/bin/htrmapper-abrir"

if [ "$CREATE_SHORTCUTS" = "1" ]; then
    say "Atalhos no Windows"
    PS1_WIN="$(wslpath -w "$APP_DIR/windows/criar_atalhos.ps1")"
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PS1_WIN" \
        -Distro "${WSL_DISTRO_NAME:-Ubuntu}" -WslUser "$(whoami)" -AppDir "$APP_DIR" \
        -Launcher "$HOME/.local/bin/htrmapper-abrir"
fi

say "Pronto. Para abrir: atalho 'HTRMapper' na Área de Trabalho do Windows, ou 'htrmapper-gui' aqui no terminal."
