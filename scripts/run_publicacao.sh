#!/usr/bin/env bash
# Rodada "para publicação": ImageNet-val completo, 7 modelos em 224 px e 5 no CIFAR, 3 sementes,
# LPIPS e distância entre features como preditores de comparação com o IRE.
#
#   nohup bash scripts/run_publicacao.sh > results/pub_logs/main.log 2>&1 &
#   bash scripts/status_publicacao.sh          # progresso e estimativa de término
#
# Pode ser interrompido e relançado: fatias prontas (meta.json) são puladas.
# Fases:  A) oracle+main em fatias (+ E4 do CIFAR)   B) junta fatias; E4 do Imagenette; análise do ImageNet e CIFAR
#         C) análise do Imagenette   D) agregação entre sementes
source "$(dirname "$0")/_common.sh"

GPUS=(${PUB_GPUS:-0 1})          # GPUs usadas
JOBS=${PUB_JOBS:-44}             # processos simultâneos na fase A
SEEDS=(${PUB_SEEDS:-0 1 2})
N_IMNET=${PUB_IMNET_SHARDS:-80}  # fatias do ImageNet
N_SMALL=4                        # fatias por semente no CIFAR / Imagenette
L=results/pub_logs; mkdir -p "$L/jobs"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONWARNINGS=ignore

# Uma linha por tarefa:  nome | config | etapas | --set extras
jobs_a() {  # tarefas pequenas primeiro; as fatias do ImageNet preenchem a fila depois
  for s in "${SEEDS[@]}"; do
    echo "adapt_cifar10_s$s|configs/pub_cifar10.yaml|adapt|run_name=pub_cifar10_s$s seed=$s"
  done
  for s in "${SEEDS[@]}"; do for k in $(seq 0 $((N_SMALL - 1))); do
    echo "imagenette_s${s}_$k|configs/pub_imagenette.yaml|oracle main|run_name=pub_imagenette_s$s seed=$s shard=$k/$N_SMALL"
    echo "cifar10_s${s}_$k|configs/pub_cifar10.yaml|oracle main|run_name=pub_cifar10_s$s seed=$s shard=$k/$N_SMALL"
  done; done
  for k in $(seq 0 $((N_IMNET - 1))); do
    echo "imagenet_$k|configs/pub_imagenet.yaml|oracle main|shard=$k/$N_IMNET"
  done
}

run_job() {  # run_job "nome|config|etapas|sets" slot
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

phase() { log "=== fase $1 ==="; echo "$1 $(date +%s)" >> "$L/phases.txt"; }

phase A
jobs_a > "$L/jobs_a.txt"
parallel --will-cite -j "$JOBS" --joblog "$L/joblog_a.tsv" --line-buffer run_job {} {%} < "$L/jobs_a.txt" || warn "alguma tarefa da fase A falhou (ver $L/joblog_a.tsv)"

phase B
pdi merge-shards --config configs/pub_imagenet.yaml
for s in "${SEEDS[@]}"; do
  pdi merge-shards --config configs/pub_imagenette.yaml --set run_name=pub_imagenette_s$s
  pdi merge-shards --config configs/pub_cifar10.yaml --set run_name=pub_cifar10_s$s
done
{
  for s in "${SEEDS[@]}"; do echo "adapt_imagenette_s$s|configs/pub_imagenette.yaml|adapt|run_name=pub_imagenette_s$s seed=$s"; done
  echo "analyze_imagenet|configs/pub_imagenet.yaml|analyze|"
  for s in "${SEEDS[@]}"; do echo "analyze_cifar10_s$s|configs/pub_cifar10.yaml|analyze|run_name=pub_cifar10_s$s seed=$s"; done
} > "$L/jobs_b.txt"
parallel --will-cite -j 7 --joblog "$L/joblog_b.tsv" --line-buffer run_job {} {%} < "$L/jobs_b.txt" || warn "alguma tarefa da fase B falhou"

phase C
for s in "${SEEDS[@]}"; do echo "analyze_imagenette_s$s|configs/pub_imagenette.yaml|analyze|run_name=pub_imagenette_s$s seed=$s"; done > "$L/jobs_c.txt"
parallel --will-cite -j 3 --joblog "$L/joblog_c.tsv" --line-buffer run_job {} {%} < "$L/jobs_c.txt" || warn "alguma tarefa da fase C falhou"

phase D
"$PY" scripts/aggregate_seeds.py --runs pub_cifar10_s pub_imagenette_s --seeds "${SEEDS[@]}" --single pub_imagenet --out results/pub_agregado
phase FIM
log "fim. Resumo em results/pub_agregado/summary.md"
