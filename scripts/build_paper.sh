#!/usr/bin/env bash
# Build the manuscript: markdown -> DOCX (reference style) + PDF (LaTeX).
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=paper/paper.md
DOCX=paper/capacity-buys-patience.docx
PDF=paper/capacity-buys-patience.pdf
TEMPLATE=paper/reference.docx

mkdir -p paper

if [[ ! -f "$TEMPLATE" ]]; then
  pandoc --print-default-data-file reference.docx > "$TEMPLATE"
fi

# Note: this pandoc build (2.5) has no --citeproc and no pandoc-citeproc filter.
# Citations are kept as plain "[@key]" markers in the draft and the bibliography is
# appended as a numbered list below. Swap in a citeproc-capable pandoc (>=2.11) or
# Zotero export for a submission build; see paper/DEVIATIONS.md.
pandoc "$SRC" \
  --from markdown+tex_math_dollars+pipe_tables+footnotes \
  --to docx \
  --reference-doc="$TEMPLATE" \
  -o "$DOCX"

pandoc "$SRC" \
  --from markdown+tex_math_dollars+pipe_tables+footnotes \
  --to latex \
  --pdf-engine=pdflatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V colorlinks=true \
  -V linkcolor=black -V urlcolor=blue -V citecolor=black \
  -V documentclass=article \
  -o "$PDF"

echo "[paper] wrote $DOCX and $PDF"
echo "[paper] placeholder count: $(grep -c 'PENDING' "$SRC" || true)"
