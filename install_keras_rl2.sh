#!/usr/bin/env bash
# Bash installer for a legacy keras-rl2 environment (mirrors install_keras_rl2.ps1).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./install_keras_rl2.sh [--accept-license] [--rom-path PATH]

Options:
  --accept-license   Use AutoROM to download Atari ROMs (accepts license).
  --rom-path PATH    Import ROMs from PATH instead of using AutoROM.
EOF
}

ACCEPT_LICENSE=0
ROM_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-license) ACCEPT_LICENSE=1 ;;
    --rom-path) shift; ROM_PATH="${1:-}";;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

info() { printf '[info] %s\n' "$*"; }
error() { printf '[error] %s\n' "$*" >&2; exit 1; }

info "Checking for python3.8..."
command -v python3.8 >/dev/null 2>&1 || error "Python 3.8 is required (python3.8 not found)."

info "Verifying Python is 64-bit..."
IS_64BIT=$(python3.8 - <<'PY'
import sys
print(int(sys.maxsize > 2**32))
PY
)
[[ "$IS_64BIT" == "1" ]] || error "Python 3.8 x64 is required. Install 64-bit Python 3.8 and retry."

VENV_PATH="$(dirname "$0")/.venv-rl2"
if [[ -d "$VENV_PATH" ]]; then
  info "Removing existing venv at $VENV_PATH"
  rm -rf "$VENV_PATH"
fi

if [[ ! -d "$VENV_PATH" ]]; then
  info "Creating venv at $VENV_PATH"
  python3.8 -m venv "$VENV_PATH"
else
  info "Venv already exists at $VENV_PATH"
fi

info "Activating venv"
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

info "Upgrading pip"
python -m pip install --upgrade pip

info "Installing legacy deps for keras-rl2"
pip install gym==0.17.3
pip install pyglet==1.5.0
pip install Pillow==9.5.0
pip install numpy==1.19.5
pip install h5py==3.1.0

info "Installing TensorFlow 2.5.3 (and removing newer TensorFlow if present)"
pip uninstall -y tensorflow-intel tensorflow || true
pip install tensorflow==2.5.3

info "Installing keras-rl2 (uses tensorflow.keras)"
pip uninstall -y Keras || true
pip install --no-deps keras-rl2==1.0.5

info "Installing atari-py"
pip install atari-py==0.2.9

if [[ "$ACCEPT_LICENSE" == "1" ]]; then
  info "Installing AutoROM to download Atari ROMs"
  pip install autorom 'autorom[accept-rom-license]'
  info "Downloading ROMs (AutoROM)"
  python - <<'PY'
from AutoROM.AutoROM import cli
cli.main(["--accept-license"])
PY
  AUTO_ROM_DIR="$(python - <<'PY'
import os, sys
try:
    import AutoROM
    print(os.path.join(os.path.dirname(AutoROM.__file__), "roms"))
except ImportError:
    sys.exit(1)
PY
)"
  if [[ -d "$AUTO_ROM_DIR" ]]; then
    info "Importing ROMs from $AUTO_ROM_DIR into atari-py"
    python -m atari_py.import_roms "$AUTO_ROM_DIR"
  else
    info "AutoROM ROM directory not found at $AUTO_ROM_DIR"
  fi
elif [[ -n "$ROM_PATH" ]]; then
  if [[ ! -d "$ROM_PATH" ]]; then
    error "RomPath '$ROM_PATH' does not exist."
  fi
  info "Importing ROMs from $ROM_PATH"
  python -m atari_py.import_roms "$ROM_PATH"
else
  info "No RomPath provided; skip ROM import."
  info "Run later with: ./install_keras_rl2.sh --accept-license"
  info "Or import your own ROMs: ./install_keras_rl2.sh --rom-path /path/to/roms"
fi

info "Done. Activate with: source $VENV_PATH/bin/activate"
