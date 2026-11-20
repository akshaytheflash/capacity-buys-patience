#!/usr/bin/env bash
# Colab entry point: run a sweep arm, upload per-epoch JSONL to GitHub every 60s
# via the REST API (see push_results.py). Checkpoints stay local (they exceed
# GitHub's 100MB limit, which is what broke the first git-based sync).
set -uo pipefail
ARM="${1:-colab_subset}"
cd "$(dirname "$0")/.."
export GH_TOKEN="${GH_TOKEN:-}"

python3 -m src.sweep --config "configs/${ARM}.yaml" &
SWEEP=$!

( while kill -0 $SWEEP 2>/dev/null; do
    python3 scripts/push_results.py --arm "$ARM" 2>&1 | tail -1
    sleep 60
  done ) &
SYNC=$!

wait $SWEEP
kill $SYNC 2>/dev/null

python3 -m src.analyse.transition --raw "results/raw/${ARM}" --out results/processed/transitions.csv || true
python3 -m src.analyse.mediation  --transitions results/processed/transitions.csv --out results/processed/mediation.csv || true
python3 -m src.analyse.earlywarning --raw "results/raw/${ARM}" --out results/processed/earlywarning.csv || true
python3 -m src.analyse.transition --raw "results/raw/${ARM}" --out /dev/null || true
python3 scripts/push_results.py --arm "$ARM" 2>&1 | tail -1
echo "[colab] ${ARM} complete"
