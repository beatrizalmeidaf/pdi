#!/usr/bin/env bash
# Roda as etapas do estudo em ordem.
#   scripts/run_experiments.sh                         # oracle main adapt analyze, config padrão
#   CONFIG=configs/smoke.yaml scripts/run_experiments.sh main analyze
#   EXTRA="--set backend=reference --set device=cuda" scripts/run_experiments.sh
#   DATASETS="cifar10" scripts/run_experiments.sh main
source "$(dirname "$0")/_common.sh"

stages=("$@")
[[ ${#stages[@]} -eq 0 ]] && stages=(oracle main adapt analyze)
read -r -a extra <<< "${EXTRA:-}"
ds_args=()
[[ -n "${DATASETS:-}" ]] && { read -r -a ds <<< "$DATASETS"; ds_args=(--datasets "${ds[@]}"); }

CFG_NAME="$(basename "$CONFIG" .yaml)"
LOG_DIR="results/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${CFG_NAME}.log"
log "config=$CONFIG etapas=${stages[*]} log=$LOG"

for s in "${stages[@]}"; do
  case "$s" in
    verify|examples|oracle|main|adapt|analyze) ;;
    *) die "etapa desconhecida: $s" ;;
  esac
  log ">>> $s"
  start=$SECONDS
  pdi "$s" --config "$CONFIG" "${extra[@]}" "${ds_args[@]}" 2>&1 | tee -a "$LOG"
  log "<<< $s concluída em $(( (SECONDS - start) / 60 )) min"
done
log "fim. Resultados em results/<run_name>/ (run_name definido em $CONFIG)"
