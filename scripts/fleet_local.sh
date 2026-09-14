#!/usr/bin/env bash
# On-box fleet: run one GPU-pinned worker loop per visible GPU, all draining the same local queue.
# This is how the study parallelises on a SINGLE multi-GPU vast.ai instance (local disk = the shared
# volume; file locks work within one filesystem). Each worker owns one GPU via CUDA_VISIBLE_DEVICES.
#   ./scripts/fleet_local.sh c4
# Env: SYNSCALE_STORE (checkpoint dir), QUEUE_DIR (queue dir), GEN_SHARDS (default = #GPUs*2).
set -uo pipefail
PROFILE="${1:-c4}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
[ -d .venv ] && source .venv/bin/activate

# Count GPUs (fall back to 1 if nvidia-smi is absent, e.g. a CPU dry-run).
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '); DEV=cuda
else
  NGPU=1; DEV=cpu   # CPU dry-run (e.g. micro on a box with no GPU)
fi
[ "$NGPU" -ge 1 ] 2>/dev/null || NGPU=1
export SYNSCALE_STORE="${SYNSCALE_STORE:-/workspace/synscale/checkpoints/$PROFILE}"
QDIR="${QUEUE_DIR:-/workspace/synscale/results/$PROFILE/queue}"
SHARDS="${GEN_SHARDS:-$((NGPU * 2))}"
mkdir -p logs "$QDIR"
echo "fleet_local: profile=$PROFILE gpus=$NGPU shards=$SHARDS store=$SYNSCALE_STORE queue=$QDIR"

# Build the queue once (idempotent). --build-only creates queue.json and runs nothing.
python scripts/worker.py --profile "$PROFILE" --queue-dir "$QDIR" --gen-shards "$SHARDS" --build-only

pids=()
for ((g=0; g<NGPU; g++)); do
  (
    export CUDA_VISIBLE_DEVICES="$g"
    while true; do
      python scripts/worker.py --profile "$PROFILE" --tier ondemand --queue-dir "$QDIR" \
        --device "$DEV" --gen-shards "$SHARDS" >> "logs/worker_gpu${g}.log" 2>&1
      code=$?
      if [ $code -eq 3 ]; then echo "[gpu$g] queue drained"; break; fi
      sleep 2
    done
  ) &
  pids+=($!)
  echo "  started worker on GPU $g (pid ${pids[-1]}), log logs/worker_gpu${g}.log"
done

# Wait for all GPU loops to finish (queue drained on every GPU). The DAG's finalize job (claimed by
# whichever worker reaches it) has already built results/$PROFILE/index.csv and analysis/.
for p in "${pids[@]}"; do wait "$p"; done
echo "fleet_local: all workers exited; queue drained."
python scripts/worker.py --profile "$PROFILE" --queue-dir "$QDIR" --build-only   # prints final summary
echo "fleet_local: done. See results/$PROFILE/index.csv and analysis/."
