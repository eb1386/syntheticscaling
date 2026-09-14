#!/usr/bin/env bash
# RunPod pod-side bootstrap. Set as the pod's Container Start Command (or run once over SSH). It
# self-installs and runs the on-box fleet, unattended, so the study survives the driver session.
# Code delivery, in order of preference:
#   1. GH_TOKEN env set  -> git clone the private repo (fully unattended, recommended for RunPod)
#   2. else wait for the driver to upload /workspace/repo.tgz + touch /workspace/GO
# Progress is written to /workspace/status.txt and /workspace/run.log on the NETWORK VOLUME, so it
# survives a pod crash and can be read from the console or a fresh pod.
exec >>/workspace/bootstrap.log 2>&1
set -x
PROFILE="${PROFILE:-c4}"
REPO="${REPO:-eb1386/syntheticscaling}"
BRANCH="${BRANCH:-claude/admiring-fermi-btrw8q}"
mkdir -p /workspace/synscale
status() { echo "$(date -u +%FT%TZ) $*" | tee /workspace/status.txt; }

cd /workspace
if [ -n "${GH_TOKEN:-}" ]; then
  status "cloning $REPO@$BRANCH"
  rm -rf /workspace/synscale-repo
  git clone --depth 1 --branch "$BRANCH" "https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git" \
    /workspace/synscale-repo || { status "ERROR: clone failed"; exit 1; }
else
  status "no GH_TOKEN: waiting for /workspace/repo.tgz + GO (up to 2h)"
  for i in $(seq 1 720); do [ -f /workspace/GO ] && break; sleep 10; done
  [ -f /workspace/GO ] || { status "ERROR: GO never arrived"; exit 1; }
  mkdir -p /workspace/synscale-repo && tar xzf /workspace/repo.tgz -C /workspace/synscale-repo
fi

cd /workspace/synscale-repo
export SYNSCALE_STORE=/workspace/synscale/checkpoints/$PROFILE
export QUEUE_DIR=/workspace/synscale/results/$PROFILE/queue
export HF_HOME=/workspace/hf-cache
export HF_TOKEN="${HF_TOKEN:-}"
export PROFILE

status "install: deps + tokenizer + data + teacher models (profile=$PROFILE)"
if ! bash install.sh "$PROFILE" >/workspace/install.log 2>&1; then
  status "ERROR: install failed; see /workspace/install.log"; exit 1
fi
status "install done; launching on-box fleet across all GPUs"
setsid bash -c '
  cd /workspace/synscale-repo
  export SYNSCALE_STORE='"$SYNSCALE_STORE"' QUEUE_DIR='"$QUEUE_DIR"' HF_HOME='"$HF_HOME"' HF_TOKEN='"$HF_TOKEN"'
  bash scripts/fleet_local.sh '"$PROFILE"' >/workspace/run.log 2>&1
  echo "$(date -u +%FT%TZ) DONE $(python scripts/worker.py --profile '"$PROFILE"' --queue-dir '"$QUEUE_DIR"' --build-only 2>/dev/null | tail -1)" >/workspace/DONE
  cp -f results/'"$PROFILE"'/index.csv /workspace/index.csv 2>/dev/null || true
  cp -rf results/'"$PROFILE"'/analysis /workspace/analysis 2>/dev/null || true
' >/dev/null 2>&1 &
status "fleet running; watch /workspace/run.log and /workspace/DONE"
