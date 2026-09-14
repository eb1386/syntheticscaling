#!/usr/bin/env bash
# Run the whole study. Two modes:
#   ./run_all.sh c4              serial: one process runs every stage in order (one GPU or CPU micro)
#   ./run_all.sh c4 --fleet      fleet:  build a shared job queue, then drain it with a local worker.
#                                Add more pods by running scripts/worker.sh on each (see docs/26).
# Resumable either way: re-run to continue after an interruption. Fleet mode requeues preempted jobs.
# Usage:  ./run_all.sh [PROFILE] [--fleet] [extra args...]
#   PROFILE:  c4 (main study) | scaling5080_fast (pilot) | smoke (GPU, ~1h) | micro (CPU, minutes)
set -euo pipefail
PROFILE="${1:-c4}"; shift || true
FLEET=0; PASS=()
for a in "$@"; do
  if [ "$a" = "--fleet" ]; then FLEET=1; else PASS+=("$a"); fi
done
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
if [ -d .venv ]; then source .venv/bin/activate; fi
export SYNSCALE_STORE="${SYNSCALE_STORE:-/workspace/synscale/checkpoints/$PROFILE}"  # persistent volume for spot resumes
mkdir -p "results/$PROFILE" logs
echo "==> profile=$PROFILE  store=$SYNSCALE_STORE  fleet=$FLEET  (logging to results/$PROFILE/run.log)"
echo "==> This is long-running. Use tmux/screen so it survives disconnects."

if [ "$FLEET" -eq 1 ]; then
  QDIR="${QUEUE_DIR:-/workspace/synscale/results/$PROFILE/queue}"
  echo "==> Fleet mode. Queue: $QDIR"
  echo "==> Start more workers on other spot pods with: ./scripts/worker.sh $PROFILE spot"
  stdbuf -oL -eL ./scripts/worker.sh "$PROFILE" "${TIER:-spot}" 2>&1 | tee -a "results/$PROFILE/run.log"
  echo "==> queue drained. The finalize job built results/$PROFILE/index.csv and analysis/."
else
  stdbuf -oL -eL python -m synscale.pipeline --profile "$PROFILE" "${PASS[@]}" 2>&1 | tee -a "results/$PROFILE/run.log"
fi
echo "==> done. Tables + figures under results/$PROFILE/analysis/"
