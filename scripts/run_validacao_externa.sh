#!/usr/bin/env bash
# Validação externa: roda as 6 técnicas novas (techniques/extra.py) nas mesmas imagens e modelos da rodada final.
#   setsid nohup bash scripts/run_validacao_externa.sh &      # retomável: tarefas com .done são puladas
# Depois: python scripts/ire_validacao_externa.py
# GPUs: EXTRA_GPUS (padrão 2); processos simultâneos: EXTRA_JOBS (padrão 32).
set -uo pipefail
source "$(dirname "$0")/_common.sh"

GPUS=(${EXTRA_GPUS:-2})
JOBS=${EXTRA_JOBS:-32}
SEEDS=(0 1 2)
N_IMNET=${EXTRA_IMNET_SHARDS:-32}
N_SMALL=4
L=results/pub_logs/extra; mkdir -p "$L/jobs"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONWARNINGS=ignore

jobs() {
  for s in "${SEEDS[@]}"; do for k in $(seq 0 $((N_SMALL - 1))); do
    echo "cifar10_s${s}_$k|configs/pub_extra_cifar10.yaml|main|run_name=pub_cifar10_s${s}_extra seed=$s shard=$k/$N_SMALL"
    echo "imagenette_s${s}_$k|configs/pub_extra_imagenette.yaml|main|run_name=pub_imagenette_s${s}_extra seed=$s shard=$k/$N_SMALL"
  done; done
  for k in $(seq 0 $((N_IMNET - 1))); do
    echo "imagenet_$k|configs/pub_extra_imagenet.yaml|main|shard=$k/$N_IMNET"
  done
}

run_job() {
  IFS='|' read -r name cfg stages sets <<< "$1"
  local gpus=(${GPUS_STR}) slot=$2
  export CUDA_VISIBLE_DEVICES=${gpus[$(( slot % ${#gpus[@]} ))]}
  local log="$L/jobs/$name.log" args=() st
  [[ -f "$L/jobs/$name.done" ]] && { echo "pulado (já feito): $name"; return 0; }
  for kv in $sets; do args+=(--set "$kv"); done
  echo "[$(date '+%F %T')] início GPU=$CUDA_VISIBLE_DEVICES" >> "$log"
  for st in $stages; do
    "$PY" -m pdi_spectral "$st" --config "$cfg" --set n_workers=4 "${args[@]}" >> "$log" 2>&1 || { echo "FALHOU $name ($st)" >> "$log"; return 1; }
  done
  echo "[$(date '+%F %T')] fim" >> "$log"; touch "$L/jobs/$name.done"
}
export -f run_job; export L PY GPUS_STR="${GPUS[*]}"

log "=== inferência das técnicas novas ==="
echo "inicio $(date +%s)" >> "$L/fases.txt"
jobs > "$L/jobs.txt"
parallel --will-cite -j "$JOBS" --joblog "$L/joblog.tsv" --line-buffer run_job {} {%} < "$L/jobs.txt" || warn "alguma tarefa falhou (ver $L/joblog.tsv)"

log "=== junta as fatias ==="
pdi merge-shards --config configs/pub_extra_imagenet.yaml
for s in "${SEEDS[@]}"; do
  pdi merge-shards --config configs/pub_extra_imagenette.yaml --set run_name=pub_imagenette_s${s}_extra
  pdi merge-shards --config configs/pub_extra_cifar10.yaml --set run_name=pub_cifar10_s${s}_extra
done
echo "fim $(date +%s)" >> "$L/fases.txt"
log "fim. Próximo passo: python scripts/ire_validacao_externa.py"
