# Makefile do estudo "pré-processamento perceptual x energia espectral x acurácia".
# Requer bash (Linux/macOS, ou Git Bash/WSL no Windows) e GNU make
# (Windows: `choco install make` ou `conda install -c conda-forge make`).
#
# Variáveis úteis:
#   make main CONFIG=configs/smoke.yaml
#   make main EXTRA="--set backend=reference --set device=cuda"
#   make main DATASETS=cifar10
#   make setup BASE_PY=python3.12 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

SHELL   := bash
CONFIG  ?= configs/default.yaml
EXTRA   ?=
DATASETS ?=
export CONFIG EXTRA DATASETS

.DEFAULT_GOAL := help
.PHONY: help setup download download-sr download-cifar download-imagenette download-models \
        test verify examples smoke offline sweep safe-dose human-export human-analyze oracle main adapt analyze all fast clean clean-results

help:  ## lista os alvos
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## cria .venv e instala dependências
	bash scripts/setup_env.sh

download:  ## baixa pesos de SR, CIFAR-10, Imagenette e pesos dos classificadores
	bash scripts/download_data.sh sr cifar imagenette models

download-sr:  ## só os pesos ESPCN/FSRCNN
	bash scripts/download_data.sh sr
download-cifar:
	bash scripts/download_data.sh cifar
download-imagenette:
	bash scripts/download_data.sh imagenette
download-models:
	bash scripts/download_data.sh models

test:  ## testes unitários (técnicas e quantificador espectral)
	bash -c 'source scripts/_common.sh && "$$PY" -m pytest -q tests'

verify:  ## E0: testes + concordância do zero x OpenCV/PIL/SciPy
	bash scripts/verify.sh

examples:  ## figuras qualitativas e assinatura espectral por técnica
	bash scripts/run_experiments.sh examples

smoke:  ## pipeline completo em miniatura (CIFAR-10, 64 imagens)
	bash scripts/run_smoke.sh

sweep:  ## varredura densa de dose -> limiares (BMD), janela operacional e IRE (CIFAR-10 por padrão)
	CONFIG=configs/sweep.yaml DATASETS="$(or $(DATASETS),cifar10)" bash scripts/run_experiments.sh oracle main adapt analyze

safe-dose:  ## dose segura pela fórmula IRE, sem rodar a rede. Ex.: make safe-dose T=gaussian M=cifar10_resnet20
	bash -c 'source scripts/_common.sh && pdi safe-dose --config $(if $(filter configs/default.yaml,$(CONFIG)),configs/sweep.yaml,$(CONFIG)) --technique $(T) $(if $(M),--model $(M)) $(if $(TAU),--tau $(TAU))'

human-export:  ## estudo com pessoas: gera results/<run>/human_study/index.html (config do sweep)
	bash -c 'source scripts/_common.sh && pdi human-export --config $(if $(filter configs/default.yaml,$(CONFIG)),configs/sweep.yaml,$(CONFIG))'

human-analyze:  ## analisa os CSVs de respostas em results/<run>/human_study/respostas/
	bash -c 'source scripts/_common.sh && pdi human-analyze --config $(if $(filter configs/default.yaml,$(CONFIG)),configs/sweep.yaml,$(CONFIG))'

offline:  ## pipeline completo sem internet (imagens sintéticas + CNN minúscula)
	CONFIG=configs/offline.yaml bash scripts/run_experiments.sh oracle main adapt analyze examples

oracle:  ## E1: sensibilidade por banda com filtros ideais
	bash scripts/run_experiments.sh oracle

main:  ## E2: dose-resposta, inferência congelada
	bash scripts/run_experiments.sh main

adapt:  ## E4: domain shift x perda de informação
	bash scripts/run_experiments.sh adapt

analyze:  ## E3: estatística, previsibilidade, figuras, relatório
	bash scripts/run_experiments.sh analyze

all:  ## E1 + E2 + E4 + E3
	bash scripts/run_experiments.sh oracle main adapt analyze

fast:  ## como `all`, mas com backend OpenCV (validado em `make verify`)
	EXTRA="--set backend=reference $(EXTRA)" bash scripts/run_experiments.sh oracle main adapt analyze

clean:  ## remove caches Python
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache src/*.egg-info

clean-results:  ## apaga results/ (cuidado)
	rm -rf results
