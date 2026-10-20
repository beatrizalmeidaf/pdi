#!/usr/bin/env bash
# Funções e variáveis compartilhadas pelos scripts. Uso: source scripts/_common.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Interpretador: $PY > venv do projeto > python do sistema
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
    PY="$ROOT/.venv/Scripts/python.exe"       # Windows (Git Bash)
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"               # Linux / macOS
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    PY="python"
  fi
fi
export PY
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8
CONFIG="${CONFIG:-configs/default.yaml}"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[aviso]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[erro]\033[0m %s\n' "$*" >&2; exit 1; }

pdi() { "$PY" -m pdi_spectral "$@"; }

fetch() {  # fetch URL DESTINO
  local url="$1" dst="$2"
  if [[ -s "$dst" ]]; then log "já existe: $dst"; return 0; fi
  mkdir -p "$(dirname "$dst")"
  log "baixando $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$dst.part" "$url"
  else
    wget -O "$dst.part" "$url"
  fi
  mv "$dst.part" "$dst"
}
