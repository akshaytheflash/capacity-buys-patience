.PHONY: install smoke capacity resources interventions analyse figures paper clean

install:
	pip install -r requirements.txt

# Tiny end-to-end sanity check on CPU (~1 min)
smoke:
	python -m src.sweep --config configs/smoke.yaml

capacity:
	python -m src.sweep --config configs/capacity.yaml

resources:
	python -m src.sweep --config configs/resources.yaml

interventions:
	python -m src.sweep --config configs/interventions.yaml

analyse:
	python -m src.analyse.transition --raw results/raw --out results/processed/transitions.csv
	python -m src.analyse.mediation --raw results/raw --out results/processed/mediation.csv
	python -m src.analyse.earlywarning --raw results/raw --out results/processed/earlywarning.csv

figures:
	python -m src.figures --processed results/processed --out results/figures

paper:
	bash scripts/build_paper.sh

clean:
	rm -f results/processed/*.csv results/figures/*.pdf
