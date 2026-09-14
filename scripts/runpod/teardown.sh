#!/usr/bin/env bash
# Destroy the RunPod pod so GPU billing stops. The NETWORK VOLUME persists (results survive) until
# you delete it separately in the console.
#   ./scripts/runpod/teardown.sh [POD_ID]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PID="${1:-$(cat "$REPO_ROOT/.runpod_instance" 2>/dev/null || true)}"
[ -z "$PID" ] && { echo "usage: teardown.sh <pod_id>"; exit 1; }
echo "== removing pod $PID (GPU billing stops; network volume + results persist) =="
runpodctl remove pod "$PID" 2>&1 | sed 's/^/  /'
rm -f "$REPO_ROOT/.runpod_instance"
echo "Remaining pods:"; runpodctl get pod 2>&1 | sed 's/^/  /'
echo "Pull results before deleting the volume: attach it to a cheap pod and copy /workspace/{index.csv,analysis}."
