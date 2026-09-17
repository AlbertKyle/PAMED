#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model/dementia}"
FONT_DIR="$PROJECT_DIR/fonts"
mkdir -p "$FONT_DIR"
if command -v fc-match >/dev/null 2>&1; then
  echo "Current sans font: $(fc-match sans | head -n 1)"
  echo "Arial match: $(fc-match Arial | head -n 1)"
fi
echo "For exact Arial rendering, upload Arial.ttf, Arialbd.ttf, Ariali.ttf and Arialbi.ttf to: $FONT_DIR"
echo "The v3 script will otherwise use Liberation Sans/Nimbus Sans and records the actual font in V3_VISUAL_QA.json."
