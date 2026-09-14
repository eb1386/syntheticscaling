#!/usr/bin/env bash
# Fleet worker loop: claim and run jobs until the shared queue drains. Run this on each pod.
#   ./scripts/worker.sh c4 spot        # a cheap spot pod
#   ./scripts/worker.sh c4 ondemand    # the one reliable pod (also runs spot jobs)
set -uo pipefail
PROFILE="${1:-c4}"; TIER="${2:-spot}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
[ -d .venv ] && source .venv/bin/activate
export SYNSCALE_STORE="${SYNSCALE_STORE:-/workspace/synscale/checkpoints/$PROFILE}"
QDIR="${QUEUE_DIR:-/workspace/synscale/results/$PROFILE/queue}"
echo "worker on $(hostname): profile=$PROFILE tier=$TIER queue=$QDIR"
while true; do
  python scripts/worker.py --profile "$PROFILE" --tier "$TIER" --queue-dir "$QDIR"
  code=$?
  if [ $code -eq 3 ]; then echo "queue drained; worker exiting"; break; fi
  if [ $code -ne 0 ]; then echo "job failed (code $code); continuing"; fi
  sleep 2
done
