#!/usr/bin/env bash
# Run the whole study on one RTX 5080. Resumable: re-run to continue after an interruption.
# Usage:  ./run_all.sh [PROFILE] [extra args...]
#   PROFILE:  scaling5080 (full, ~3-4 wks) | local5080_fast (~1 week) | smoke (GPU, ~1h) | micro (CPU, minutes)
set -euo pipefail
PROFILE="${1:-c4}"; shift || true
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
if [ -d .venv ]; then source .venv/bin/activate; fi
export SYNSCALE_STORE="${SYNSCALE_STORE:-/workspace/synscale/checkpoints/$PROFILE}"  # persistent volume for spot resumes
mkdir -p "results/$PROFILE" logs
echo "==> profile=$PROFILE  store=$SYNSCALE_STORE  (logging to results/$PROFILE/run.log)"
echo "==> This is long-running. Use tmux/screen so it survives disconnects."
stdbuf -oL -eL python -m synscale.pipeline --profile "$PROFILE" "$@" 2>&1 | tee -a "results/$PROFILE/run.log"
echo "==> done. Tables + figures under results/$PROFILE/analysis/"
