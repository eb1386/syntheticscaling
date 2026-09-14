#!/usr/bin/env bash
# Launch the study on ONE multi-GPU RunPod pod attached to a persistent NETWORK VOLUME (so a pod
# crash never loses work). The pod self-runs scripts/runpod/bootstrap.sh via its start command.
#   ./scripts/runpod/launch.sh [PROFILE] --volume <NET_VOL_ID> [--gpus 8] [--secure] \
#        [--gh-token <GH_PAT>] [--hf <HF_TOKEN>] [--gpu-type 'NVIDIA H100 80GB HBM3'] [--cost 30]
# --gh-token: a fine-grained, read-only PAT for just this repo (destroyed with the pod). Without it,
#   the pod waits for an uploaded tarball (see the runbook for the manual-upload path).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PROFILE="c4"; GPUS=8; CLOUD="--communityCloud"; VOL=""; GH=""; HF=""; GT="NVIDIA H100 80GB HBM3"; COST=30
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo claude/admiring-fermi-btrw8q)"
args=("$@"); [ "${#args[@]}" -gt 0 ] && [[ "${args[0]}" != --* ]] && { PROFILE="${args[0]}"; args=("${args[@]:1}"); }
while [ "${#args[@]}" -gt 0 ]; do case "${args[0]}" in
  --volume) VOL="${args[1]}"; args=("${args[@]:2}");;
  --gpus) GPUS="${args[1]}"; args=("${args[@]:2}");;
  --secure) CLOUD="--secureCloud"; args=("${args[@]:1}");;
  --gh-token) GH="${args[1]}"; args=("${args[@]:2}");;
  --hf) HF="${args[1]}"; args=("${args[@]:2}");;
  --gpu-type) GT="${args[1]}"; args=("${args[@]:2}");;
  --cost) COST="${args[1]}"; args=("${args[@]:2}");;
  *) echo "unknown arg ${args[0]}"; exit 1;; esac; done
[ -z "$VOL" ] && { echo "ERROR: --volume <NET_VOL_ID> is required (create it in the console first)"; exit 1; }
[ "$PROFILE" = "c4" ] && [ -z "$HF" ] && echo "WARNING: C4 needs --hf <HF_TOKEN> with Llama-3.x access."

# The pod's start command pulls bootstrap.sh from the repo and runs it. Env carries the tokens.
START="bash -lc 'apt-get update -y >/dev/null 2>&1; apt-get install -y git curl >/dev/null 2>&1; \
curl -fsSL -H \"Authorization: token \$GH_TOKEN\" \
https://raw.githubusercontent.com/eb1386/syntheticscaling/$BRANCH/scripts/runpod/bootstrap.sh -o /workspace/bootstrap.sh; \
GH_TOKEN=\$GH_TOKEN HF_TOKEN=\$HF_TOKEN PROFILE=\$PROFILE BRANCH=$BRANCH bash /workspace/bootstrap.sh'"

echo "== creating pod: ${GPUS}x [$GT] $CLOUD, volume $VOL, cost ceiling \$$COST/hr =="
OUT=$(runpodctl create pod --name "synscale-$PROFILE" --gpuType "$GT" --gpuCount "$GPUS" $CLOUD \
  --networkVolumeId "$VOL" --volumePath /workspace --containerDiskSize 60 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --startSSH --ports '22/tcp' --cost "$COST" \
  --env "GH_TOKEN=$GH" --env "HF_TOKEN=$HF" --env "PROFILE=$PROFILE" --args "$START" 2>&1)
echo "$OUT"
PID=$(echo "$OUT" | grep -oE '"id"[: ]+"[^"]+"' | head -1 | grep -oE '[^"]+"$' | tr -d '"')
[ -z "$PID" ] && PID=$(echo "$OUT" | grep -oE 'pod "[^"]+"' | head -1 | sed 's/pod "//;s/"//')
if [ -n "$PID" ]; then echo "$PID" > "$REPO_ROOT/.runpod_instance"; echo "pod id: $PID (saved)"; fi
echo
echo "If create failed or --args did not run, use the CONSOLE path in docs/28 (paste the start"
echo "command as the pod's Container Start Command). Monitor: ./scripts/runpod/monitor.sh"
