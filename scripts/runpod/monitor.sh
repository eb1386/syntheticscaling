#!/usr/bin/env bash
# One RunPod status snapshot. RunPod exec is SSH-based, so live progress (status.txt/run.log) is
# read either from the console or over SSH; this shows pod state + cost from the API.
#   ./scripts/runpod/monitor.sh [POD_ID]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PID="${1:-$(cat "$REPO_ROOT/.runpod_instance" 2>/dev/null || true)}"
[ -z "$PID" ] && { echo "usage: monitor.sh <pod_id>"; exit 1; }
echo "== pod $PID =="
runpodctl get pod "$PID" -a 2>&1 | sed 's/^/  /'
echo
echo "Live progress lives on the network volume at /workspace/{status.txt,run.log,DONE}. Read it via"
echo "  runpodctl ssh info $PID   (then ssh in and: tail -f /workspace/run.log)"
echo "or the pod's Logs tab in the RunPod console. Tear down: ./scripts/runpod/teardown.sh $PID"
