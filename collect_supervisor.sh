#!/bin/zsh
# Restarts the collector after crashes (e.g. goszakup HTTP 500 streaks).
# Checkpoints make every restart resume where it stopped.
# Usage: ./collect_supervisor.sh [stages] [extra collect args], e.g. ./collect_supervisor.sh lot_plans,plans
cd "$(dirname "$0")"
PY=~/.venvs/tender-risk-ml/bin/python
STAGES=${1:-lots,contracts,rnu,trd_app}
for i in $(seq 1 200); do
  echo "[supervisor] attempt $i ($STAGES): $(date)" >> data/raw/_supervisor.log
  $PY -m src.collect --stages $STAGES "${@:2}" >> data/raw/_collect.log 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "[supervisor] collection COMPLETE: $(date)" >> data/raw/_supervisor.log
    exit 0
  fi
  echo "[supervisor] collector exited code=$code, restarting in 120s" >> data/raw/_supervisor.log
  sleep 120
done
