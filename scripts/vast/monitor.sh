#!/usr/bin/env bash
# One monitoring snapshot for a running study instance: status, phase, queue progress, recent log,
# balance, and whether it is DONE. Prints a machine-parseable last line: MONITOR <state> where state
# is one of RUNNING | DONE | STOPPED | GONE. The driver session calls this on scheduled check-ins;
# it does NOT loop (so it is cheap and safe to run from a woken session).
#   ./scripts/vast/monitor.sh [INSTANCE_ID]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
ID="${1:-$(cat "$REPO_ROOT/.vast_instance" 2>/dev/null || true)}"
[ -z "$ID" ] && { echo "usage: monitor.sh <instance_id>"; exit 1; }

ST=$(vastai show instance "$ID" --raw 2>/dev/null | python3 -c '
import sys,json
try: d=json.load(sys.stdin)
except Exception: print("gone|nan"); sys.exit()
print((d.get("actual_status") or "unknown")+"|"+str(d.get("dph_total","?")))')
STATUS="${ST%%|*}"; DPH="${ST##*|}"
echo "instance $ID  status=$STATUS  \$$DPH/hr  balance=\$$(vast_balance)"

if [ "$STATUS" = "unknown" ] && [ -z "$(vastai show instance "$ID" --raw 2>/dev/null)" ]; then
  echo "MONITOR GONE"; exit 0
fi
if [ "$STATUS" = "stopped" ] || [ "$STATUS" = "exited" ]; then
  echo "instance is $STATUS (outbid or halted). Disk preserved; resume by starting it or re-bidding."
  echo "MONITOR STOPPED"; exit 0
fi

PHASE=$(vast_exec "$ID" "cat /workspace/status.txt 2>/dev/null" | tail -1)
echo "phase: ${PHASE:-<none yet>}"
QSUM=$(vast_exec "$ID" "cd /workspace/synscale-repo && python scripts/worker.py --profile \$(cat /workspace/PROFILE) --queue-dir /workspace/synscale/results/\$(cat /workspace/PROFILE)/queue --build-only 2>/dev/null" | tail -1)
echo "queue: ${QSUM:-<not built yet>}"
echo "recent run.log:"; vast_exec "$ID" "tail -8 /workspace/run.log 2>/dev/null" | sed 's/^/    /'

if vast_exec "$ID" "test -f /workspace/DONE && echo yes" 2>/dev/null | grep -q yes; then
  echo "STUDY COMPLETE:"; vast_exec "$ID" "cat /workspace/DONE" | tail -1
  echo "Pull results:  vastai copy $ID:/workspace/index.csv local:./results/  &&  vastai copy $ID:/workspace/analysis local:./results/"
  echo "Then tear down: ./scripts/vast/teardown.sh $ID"
  echo "MONITOR DONE"; exit 0
fi
echo "MONITOR RUNNING"
