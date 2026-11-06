#!/usr/bin/env bash
# Coerência por anel (scripts/espectro_coerencia_anel.py) para técnicas reais e filtros, nas 3 bases. Só CPU.
#   setsid nohup bash scripts/run_coerencia_anel.sh &      # depois: python scripts/ire_coerencia_fina.py
set -uo pipefail
source "$(dirname "$0")/_common.sh"
J=${COH_JOBS:-48}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
run() { [[ -f "$2/coerencia_anel.parquet" ]] && { echo "pulado: $2"; return; }; "$PY" scripts/espectro_coerencia_anel.py --jobs "$J" $1 || echo "FALHOU $1"; }
for b in cifar10 imagenette; do
  r=results/pub_${b}_s0
  run "--config configs/pub_$b.yaml --set run_name=pub_${b}_s0 --set seed=0 --kind main" $r/main/$b
  run "--config configs/pub_$b.yaml --set run_name=pub_${b}_s0 --set seed=0 --kind oracle" $r/oracle/$b
  run "--config configs/pub_oraculo_suave_$b.yaml --kind oracle" results/pub_${b}_s0_oraculo_suave/oracle/$b
  run "--config configs/pub_oraculo_parcial_$b.yaml --kind oracle" results/pub_${b}_s0_oraculo_parcial/oracle/$b
done
run "--config configs/pub_imagenet.yaml --kind main --n-images 10000" results/pub_imagenet/main/imagenet_val
run "--config configs/pub_imagenet.yaml --kind oracle --n-images 10000" results/pub_imagenet/oracle/imagenet_val
run "--config configs/pub_oraculo_suave_imagenet.yaml --kind oracle" results/pub_imagenet_oraculo_suave/oracle/imagenet_val
run "--config configs/pub_oraculo_parcial_imagenet.yaml --kind oracle" results/pub_imagenet_oraculo_parcial/oracle/imagenet_val
log "fim"
