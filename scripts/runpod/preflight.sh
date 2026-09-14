#!/usr/bin/env bash
# RunPod preflight: verify the API key works and list your pods/GPU availability.
# Run in the NEW session (where api.runpod.io is reachable), after: runpodctl config --apiKey <KEY>
#   ./scripts/runpod/preflight.sh
set -uo pipefail
echo "== 1. auth =="
if runpodctl get pod >/dev/null 2>&1; then echo "  OK: runpodctl authenticated"; else
  echo "  FAIL: run 'runpodctl config --apiKey <KEY>' (key at https://www.runpod.io/console/user/settings)"; exit 1; fi
echo "== 2. current pods (should be empty before launch) =="
runpodctl get pod 2>&1 | sed 's/^/  /'
echo
echo "== 3. reminders =="
echo "  - Create a NETWORK VOLUME first in the console (Storage -> Network Volumes), ~600GB, in a"
echo "    region that has H100 SXM availability. Note its volume id."
echo "  - H100 SXM gpuType id is usually 'NVIDIA H100 80GB HBM3'. Secure = reliable, Community = cheaper."
echo "  - Then: ./scripts/runpod/launch.sh c4 --volume <VOL_ID> --gh-token <GH> --hf <HF>"
