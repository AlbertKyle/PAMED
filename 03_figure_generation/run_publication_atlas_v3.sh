#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model/dementia}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/Results/v13_final}"
if [[ -d "$RESULT_ROOT/v13_final/02_core_model" ]]; then V13_DIR="$RESULT_ROOT/v13_final"; else V13_DIR="$RESULT_ROOT"; fi
OUT="${OUTPUT_DIR:-$RESULT_ROOT/publication_atlas_v3_revision}"
MODULE="${1:-all}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.matplotlib_cache_v3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
mkdir -p "$OUT/logs" "$MPLCONFIGDIR"
python "$CODE_DIR/publication_atlas_v3.py" \
  --module "$MODULE" --base-dir "$PROJECT_DIR" --v13-dir "$V13_DIR" \
  --descriptive-dir "$PROJECT_DIR/descriptive_nature_results" \
  --external-dir "$PROJECT_DIR/external_validation" --output-dir "$OUT" \
  2>&1 | tee "$OUT/logs/${MODULE}.log"
echo "Completed: $MODULE"
echo "Visual QA: $OUT/V3_VISUAL_QA.json"
