#!/usr/bin/env bash
# Baixa dados e pesos.  Uso: scripts/download_data.sh [sr] [cifar] [imagenette] [models] [cifar10c]
# Sem argumentos: sr cifar imagenette models
source "$(dirname "$0")/_common.sh"

DATA="${DATA_DIR:-data}"
targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(sr cifar imagenette models)

for t in "${targets[@]}"; do
  case "$t" in
    sr)
      # Pesos públicos para cv2.dnn_superres
      for s in 2 3 4; do
        fetch "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x${s}.pb" "$DATA/sr_models/ESPCN_x${s}.pb"
        fetch "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x${s}.pb" "$DATA/sr_models/FSRCNN_x${s}.pb"
      done ;;
    cifar)
      log "CIFAR-10 via torchvision"
      "$PY" - <<EOF
from torchvision.datasets import CIFAR10
for train in (True, False):
    CIFAR10(root="$DATA/cifar10", train=train, download=True)
EOF
      ;;
    imagenette)
      if [[ -d "$DATA/imagenette2-320/val" ]]; then log "Imagenette já extraído"; else
        fetch "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz" "$DATA/imagenette2-320.tgz"
        log "extraindo Imagenette"
        tar -xzf "$DATA/imagenette2-320.tgz" -C "$DATA"
      fi ;;
    models)
      log "pré-baixando pesos dos classificadores (torchvision / torch.hub)"
      pdi download --config "$CONFIG" ;;
    cifar10c)
      # Opcional (QP4: pré-processar entradas realmente degradadas). ~2.9 GB.
      fetch "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar" "$DATA/CIFAR-10-C.tar"
      tar -xf "$DATA/CIFAR-10-C.tar" -C "$DATA" ;;
    *) die "alvo desconhecido: $t" ;;
  esac
done
log "downloads concluídos"
