#!/usr/bin/env bash
# Pull results off the instance (if present) and DESTROY it so billing stops. Run this the moment
# the study is DONE, or any time you want to stop spending. Destroy is irreversible.
#   ./scripts/vast/teardown.sh [INSTANCE_ID] [--no-fetch]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
ID="${1:-$(cat "$REPO_ROOT/.vast_instance" 2>/dev/null || true)}"
[ -z "$ID" ] && { echo "usage: teardown.sh <instance_id>"; exit 1; }
FETCH=1; [ "${2:-}" = "--no-fetch" ] && FETCH=0

if [ "$FETCH" -eq 1 ]; then
  echo "== fetching results before destroy =="
  mkdir -p "$REPO_ROOT/results/from_vast"
  vastai copy "$ID:/workspace/index.csv" "local:$REPO_ROOT/results/from_vast/index.csv" 2>/dev/null || echo "  (no index.csv yet)"
  vastai copy "$ID:/workspace/analysis" "local:$REPO_ROOT/results/from_vast/analysis" 2>/dev/null || echo "  (no analysis dir yet)"
  echo "  saved under results/from_vast/ (if present)"
fi

echo "== destroying instance $ID (billing stops) =="
vastai destroy instance "$ID" -y
rm -f "$REPO_ROOT/.vast_instance"
echo "done. Verify nothing else is running:"
vastai show instances --raw 2>/dev/null | python3 -c '
import sys,json
try: rows=json.load(sys.stdin)
except Exception: rows=[]
live=[r for r in rows if (r.get("actual_status") in ("running","loading","frozen"))]
print("  LIVE INSTANCES STILL BILLING:", [r.get("id") for r in live] or "none")'
