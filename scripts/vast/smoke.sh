#!/usr/bin/env bash
# MONEY-SAFETY GATE. Rent ONE cheap single GPU, run the `smoke` profile (real vLLM teachers, tiny
# token budget), verify it produces an index, then DESTROY the instance. Expected cost ~$1-5. This
# is the proof that the code runs end to end on real CUDA before any H100 budget is committed.
#   ./scripts/vast/smoke.sh [HF_TOKEN]
# Pass an HF token if the smoke teachers are gated (they are small/open by default, so usually not).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
HF="${1:-${HF_TOKEN:-}}"
export SCRATCH="${SCRATCH:-/tmp}"

vast_require_balance 5 || exit 1
KEY=$(vast_ensure_key); echo "ssh key: $KEY"

echo "== picking a cheap single GPU =="
read -r OFFER DPH GPU NG < <(pick_offer 'gpu_ram>=24 num_gpus=1 verified=true rentable=true dph_total<0.8' on-demand)
[ "$OFFER" = "NONE" ] || [ -z "$OFFER" ] && { echo "no cheap offer found; widen the query"; exit 1; }
echo "  offer $OFFER : \$$DPH/hr $GPU x$NG"

echo "== creating smoke instance =="
ID=$(vast_create "$OFFER" 60 "$HERE/onstart.sh" "synscale-smoke")
[ -z "$ID" ] && { echo "create failed"; exit 1; }
echo "  instance $ID (label synscale-smoke)"
trap 'echo "cleanup: destroying $ID"; vastai destroy instance "$ID" -y >/dev/null 2>&1' EXIT

wait_running "$ID" 900 || { echo "instance never reached running"; exit 1; }
upload_and_go "$ID" smoke "$HF"

echo "== waiting for smoke to finish (timeout 2h) =="
for i in $(seq 1 480); do
  if vast_exec "$ID" "test -f /workspace/DONE && echo yes" 2>/dev/null | grep -q yes; then
    echo "  DONE marker present"; break
  fi
  st=$(vast_exec "$ID" "cat /workspace/status.txt 2>/dev/null" | tail -1)
  echo "  [$((i*15))s] $st"
  sleep 15
done

echo "== verifying smoke output =="
OK=$(vast_exec "$ID" "test -s /workspace/index.csv && wc -l < /workspace/index.csv" 2>/dev/null | tail -1)
if [ -n "$OK" ] && [ "$OK" -gt 1 ] 2>/dev/null; then
  echo "  PASS: index.csv has $OK lines. The code runs on real CUDA."
  vast_exec "$ID" "cat /workspace/DONE 2>/dev/null" | tail -1
  RESULT=0
else
  echo "  FAIL: no valid index.csv. Dumping tails for diagnosis:"
  vast_exec "$ID" "tail -30 /workspace/install.log 2>/dev/null; echo ---; tail -30 /workspace/run.log 2>/dev/null"
  RESULT=1
fi
# trap destroys the instance on exit either way
echo "== smoke result: $([ $RESULT -eq 0 ] && echo PASS || echo FAIL) (instance being destroyed) =="
exit $RESULT
