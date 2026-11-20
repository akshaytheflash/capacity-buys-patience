#!/usr/bin/env bash
# Colab entry point. Runs a sweep arm and syncs per-epoch JSONL logs to the
# `results-wip` branch every 30s. Checkpoints (.state.pt, ~128MB) stay local:
# GitHub rejects files >100MB (GH001), which silently broke the first attempt.
set -uo pipefail
ARM="${1:-colab_capacity}"
cd "$(dirname "$0")/.."

WT=/content/results_sync
rm -rf "$WT"
git worktree add -q --detach "$WT" 2>/dev/null || { mkdir -p "$WT"; }

sync_results() {
  cd "$WT" || return 0
  mkdir -p results/raw
  # copy only the small JSONL logs, never the checkpoints
  find /content/capacity-buys-patience/results -name '*.jsonl' -exec cp --parents {} . \; 2>/dev/null
  cp -r /content/capacity-buys-patience/results/processed/*.csv results/ 2>/dev/null
  git add -A results 2>/dev/null
  git diff --cached --quiet 2>/dev/null && { cd - >/dev/null; return 0; }
  git -c user.email=colab@noreply -c user.name=colab commit -q -m "results: ${ARM} $(date -u +%H:%M:%S) [skip ci]"
  git push -q -f origin HEAD:results-wip && echo "[sync $(date -u +%H:%M:%S)] pushed"
  cd - >/dev/null
}

mkdir -p results/raw/"${ARM}"
python3 -m src.sweep --config "configs/${ARM}.yaml" &
SWEEP=$!
( while kill -0 $SWEEP 2>/dev/null; do sync_results; sleep 30; done ) &
SYNC=$!
wait $SWEEP
kill $SYNC 2>/dev/null

python3 -m src.analyse.transition --raw "results/raw/${ARM}" --out results/processed/transitions.csv || true
python3 -m src.analyse.mediation  --transitions results/processed/transitions.csv --out results/processed/mediation.csv || true
python3 -m src.analyse.earlywarning --raw "results/raw/${ARM}" --out results/processed/earlywarning.csv || true
sync_results
echo "[colab] ${ARM} complete"
