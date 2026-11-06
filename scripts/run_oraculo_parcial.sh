#!/usr/bin/env bash
# Filtros com atenuação parcial de cada anel (E1) em CIFAR-10 s0, Imagenette s0 e 10 mil imagens do ImageNet-val.
#   setsid nohup bash scripts/run_oraculo_parcial.sh &       # depois: python scripts/ire_oraculo_suave.py --parcial
set -uo pipefail
source "$(dirname "$0")/_common.sh"
GPU=${PARCIAL_GPU:-2}; JOBS=${PARCIAL_JOBS:-16}
L=results/pub_logs/oraculo_parcial; mkdir -p "$L"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$GPU
{
  for k in 0 1 2 3; do echo "cifar10_$k configs/pub_oraculo_parcial_cifar10.yaml seed=0 shard=$k/4"; done
  for k in 0 1 2 3; do echo "imagenette_$k configs/pub_oraculo_parcial_imagenette.yaml seed=0 shard=$k/4"; done
  for k in $(seq 0 7); do echo "imagenet_$k configs/pub_oraculo_parcial_imagenet.yaml shard=$k/8"; done
} > "$L/jobs.txt"
job() {
  read -r name cfg sets <<< "$1"; [[ -f "$L/$name.done" ]] && return 0
  args=(); for kv in $sets; do args+=(--set "$kv"); done
  "$PY" -m pdi_spectral oracle --config "$cfg" --set n_workers=4 "${args[@]}" > "$L/$name.log" 2>&1 && touch "$L/$name.done" || echo "FALHOU $name" >> "$L/$name.log"
}
export -f job; export L PY
parallel --will-cite -j "$JOBS" job {} < "$L/jobs.txt"
for b in cifar10 imagenette imagenet; do pdi merge-shards --config configs/pub_oraculo_parcial_$b.yaml; done
log "fim. Próximo passo: python scripts/ire_oraculo_suave.py --parcial"
