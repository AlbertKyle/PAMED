#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/PA_CR_CNN/Project_new/PA_Foundation_Model/dementia"
CODE="$ROOT/v13_final"
STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  echo "Usage: bash v13_final/run_v13_final.sh {protocol|pool_smoke|pool|core_smoke|core|apoe_smoke|apoe|external_smoke|external|supplement_smoke|stability|paired|subgroups|ablation|interpretability|decision|clinical_smoke|representations|subtypes|temporal_shapley|rhythm_concepts|survival|interactions|clinical_associations|final_paired|clinical_all}"
  exit 2
fi
for var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS; do
  value="${!var:-}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then export "$var"=4; fi
done
mkdir -p "$ROOT/Results/v13_final/logs"
cd "$CODE"
if [[ "$STAGE" == *_smoke || "$STAGE" == "protocol" ]]; then
  python -u v13_final.py "$STAGE"
else
  LOG="$ROOT/Results/v13_final/logs/${STAGE}.log"
  PIDFILE="$ROOT/Results/v13_final/logs/${STAGE}.pid"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Stage $STAGE is already running as PID $(cat "$PIDFILE")"; exit 1
  fi
  nohup python -u v13_final.py "$STAGE" >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "Started v13 $STAGE PID $!"
  echo "Monitor: tail -f $LOG"
fi
