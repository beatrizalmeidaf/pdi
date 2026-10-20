#!/usr/bin/env bash
# E0: testes unitários + tabela de concordância do zero x referência.
#   scripts/verify.sh               # imagens sintéticas
#   VERIFY_DATASET=cifar10 scripts/verify.sh
source "$(dirname "$0")/_common.sh"

log "testes unitários"
"$PY" -m pytest -q tests
log "concordância das implementações (results/verification.csv)"
args=(--config "$CONFIG" --n "${N:-6}")
[[ -n "${VERIFY_DATASET:-}" ]] && args+=(--verify-dataset "$VERIFY_DATASET")
pdi verify "${args[@]}"
