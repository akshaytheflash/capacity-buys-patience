#!/usr/bin/env bash
# Colab entry point. Runs a sweep arm and pushes every epoch's results to GitHub,
# so a disconnect loses nothing (Colab VMs are wiped on disconnect).
set -uo pipefail
ARM="${1:-capacity}"
cd "$(dirname "$0")/.."

git config user.email "colab@users.noreply.github.com"
git config user.name  "colab"

# results/raw is gitignored locally; force-add so the VM's output reaches the repo
sync_results() {
  # Only the small per-epoch JSONL logs are synced. The .state.pt checkpoints are
  # 100+ MB and exceed GitHub's file limit, which silently rejected every push.
  git add -f 'results/raw/**/*.jsonl' 'results/raw/*.jsonl' results/processed 2>/dev/null
  git diff --cached --quiet || git commit -q -m "results: ${ARM} @ $(date -u +%H:%M:%S) [skip ci]"
  git pull -q --rebase 2>/dev/null || true
  git push -q origin HEAD:results-wip 2>/dev/null && echo "[sync] pushed"
}

python3 -m src.sweep --config "configs/${ARM}.yaml" &
SWEEP=$!

# background syncer: push every 60s while the sweep runs
( while kill -0 $SWEEP 2>/dev/null; do sync_results; sleep 60; done ) &
SYNC=$!

wait $SWEEP
kill $SYNC 2>/dev/null

python3 -m src.analyse.transition --raw "results/raw/${ARM}" --out results/processed/transitions.csv || true
python3 -m src.analyse.mediation  --transitions results/processed/transitions.csv --out results/processed/mediation.csv || true
python3 -m src.analyse.earlywarning --raw "results/raw/${ARM}" --out results/processed/earlywarning.csv || true
python3 -m src.figures --raw "results/raw/${ARM}" || true
sync_results
echo "[colab] ${ARM} complete"
