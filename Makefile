# Convenience targets — mirrors scripts/pipeline.py stages
PY ?= python3

.PHONY: setup data train generate evaluate all clean paper

setup:            ## Create venv and install deps
	$(PY) -m venv .venv && .venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Done. Activate with: source .venv/bin/activate"

data:             ## Build tensor cache from SportVU JSONs (or synthetic if none found)
	$(PY) scripts/pipeline.py --stage data --output-dir outputs

train:            ## Train conditional diffusion model
	$(PY) scripts/train.py --config configs/base.yaml

generate:         ## Sample counterfactuals from a checkpoint
	$(PY) scripts/generate.py --ckpt outputs/checkpoints/last.pt

evaluate:         ## Compute FTD / ADE / physics-violation metrics
	$(PY) scripts/evaluate.py --ckpt outputs/checkpoints/last.pt

test:             ## Run unit tests
	$(PY) -m pytest tests/ -q

all: data train evaluate test

clean:
	rm -rf outputs data/processed
