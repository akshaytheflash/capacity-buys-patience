PY ?= python3

.PHONY: install smoke test capacity resources interventions analyse figures paper clean

install:
	pip install -r requirements.txt

# Tiny end-to-end sanity check on CPU (~1 min)
smoke:
	$(PY) -m src.sweep --config configs/smoke.yaml

# Regression test: an interrupted + resumed run must be bit-identical to an
# uninterrupted one. Run this after touching train.py or the data pipeline.
test:
	$(PY) -m tests.test_resume

capacity:
	$(PY) -m src.sweep --config configs/capacity.yaml

resources:
	$(PY) -m src.sweep --config configs/resources.yaml

interventions:
	$(PY) -m src.sweep --config configs/interventions.yaml

analyse:
	$(PY) -m src.analyse.transition --raw results/raw --out results/processed/transitions.csv
	$(PY) -m src.analyse.mediation --raw results/raw --out results/processed/mediation.csv
	$(PY) -m src.analyse.earlywarning --raw results/raw --out results/processed/earlywarning.csv

figures:
	$(PY) -m src.figures --processed results/processed --out results/figures

paper:
	bash scripts/build_paper.sh

clean:
	rm -f results/processed/*.csv results/figures/*.pdf
