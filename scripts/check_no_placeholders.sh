#!/usr/bin/env bash
# Fail loudly if any experimental number made it into the paper without provenance.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=paper/paper.md
# Numbers inside Results/Experiments must either be marked PENDING or be traceable to
# results/processed. This is a *manual review aid*, not a proof.
echo "== PENDING placeholders =="
grep -n "PENDING" "$SRC" || echo "  none (unexpected before experiments finish)"

echo "== Any percentages or accuracies in the text? =="
grep -nE '[0-9]+(\.[0-9]+)?\s?\\?%|accuracy of [0-9]' "$SRC" || echo "  none"

echo "== Figure files referenced =="
grep -nE '!\[.*\]\(' "$SRC" || true
