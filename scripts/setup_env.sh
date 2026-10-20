#!/usr/bin/env bash
# Cria o ambiente virtual .venv e instala as dependências.
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 scripts/setup_env.sh   # GPU NVIDIA
#   BASE_PY=python3.12 scripts/setup_env.sh                                       # escolhe o Python base
source "$(dirname "$0")/_common.sh"

BASE_PY="${BASE_PY:-python}"
if [[ ! -d .venv ]]; then
  log "criando .venv com $BASE_PY ($("$BASE_PY" --version))"
  "$BASE_PY" -m venv .venv
fi
unset PY
source "$(dirname "$0")/_common.sh"

"$PY" -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  log "instalando torch de $TORCH_INDEX_URL"
  "$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e . --no-deps
"$PY" - <<'EOF'
import torch, cv2
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| opencv", cv2.__version__,
      "| dnn_superres:", hasattr(cv2, "dnn_superres"))
EOF
log "ambiente pronto: $PY"
