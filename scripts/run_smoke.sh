#!/usr/bin/env bash
# Teste de ponta a ponta com 64 imagens do CIFAR-10 e 2 modelos (poucos minutos em CPU).
source "$(dirname "$0")/_common.sh"

[[ -s data/sr_models/ESPCN_x2.pb ]] || "$ROOT/scripts/download_data.sh" sr
CONFIG=configs/smoke.yaml "$ROOT/scripts/run_experiments.sh" oracle main adapt analyze
CONFIG=configs/smoke.yaml "$ROOT/scripts/run_experiments.sh" examples
log "smoke ok -> results/smoke/analysis/summary.md"
