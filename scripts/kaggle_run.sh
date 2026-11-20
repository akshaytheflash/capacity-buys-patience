#!/usr/bin/env bash
# One-command Kaggle session. Resumes automatically from results/raw.
set -euo pipefail
ARM="${1:-capacity}"
cd "$(dirname "$0")/.."

echo "[kaggle] GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no GPU)"

python -m src.sweep --config "configs/${ARM}.yaml"

echo "[kaggle] analysis"
python -m src.analyse.transition --raw "results/raw/${ARM}" --out results/processed/transitions.csv || true
python -m src.analyse.mediation  --transitions results/processed/transitions.csv --out results/processed/mediation.csv || true
python -m src.analyse.earlywarning --raw "results/raw/${ARM}" --out results/processed/earlywarning.csv || true
python -m src.figures --raw "results/raw/${ARM}"
echo "[kaggle] done. Results live in results/ ; commit the outputs back to the repo."
