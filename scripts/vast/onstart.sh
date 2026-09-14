#!/usr/bin/env bash
# vast.ai onstart hook (runs once at boot, as root, inside the container). It does NOT clone the
# private repo (no GitHub token on a third-party box). Instead it waits for the driver to upload
# the code tarball and touch /workspace/GO, then installs and runs the on-box fleet unattended so
# the study survives the driver session dying. All phases write /workspace/status.txt for the
# monitor to read via `vastai execute <id> "cat /workspace/status.txt"`.
exec >>/workspace/onstart.log 2>&1
set -x
mkdir -p /workspace/synscale /workspace/synscale-repo
status() { echo "$(date -u +%FT%TZ) $*" | tee /workspace/status.txt; }

status "boot: waiting for code + GO (up to 2h)"
for i in $(seq 1 720); do [ -f /workspace/GO ] && break; sleep 10; done
[ -f /workspace/GO ] || { status "ERROR: GO marker never arrived; giving up"; exit 1; }

PROFILE="$(cat /workspace/PROFILE 2>/dev/null || echo c4)"
cd /workspace/synscale-repo
if [ -f /workspace/repo.tgz ]; then
  status "extracting uploaded code"
  tar xzf /workspace/repo.tgz -C /workspace/synscale-repo
fi

export SYNSCALE_STORE=/workspace/synscale/checkpoints/$PROFILE
export QUEUE_DIR=/workspace/synscale/results/$PROFILE/queue
export HF_TOKEN="${HF_TOKEN:-$(cat /workspace/HF_TOKEN 2>/dev/null || true)}"
export HF_HOME=/workspace/hf-cache          # cache models on the big local disk, not the small root
export PROFILE

command -v tmux >/dev/null 2>&1 || (apt-get update -y && apt-get install -y tmux) >/dev/null 2>&1 || true

status "install: deps + tokenizer + data + teacher models (profile=$PROFILE) — this can take a while"
if ! bash install.sh "$PROFILE" >/workspace/install.log 2>&1; then
  status "ERROR: install.sh failed; see /workspace/install.log"; exit 1
fi
status "install done; launching on-box fleet across all GPUs"

# Detached so it keeps running regardless of the driver session. fleet_local writes logs/ + results/.
setsid bash -c '
  cd /workspace/synscale-repo
  export SYNSCALE_STORE='"$SYNSCALE_STORE"' QUEUE_DIR='"$QUEUE_DIR"' HF_HOME='"$HF_HOME"' HF_TOKEN='"$HF_TOKEN"'
  bash scripts/fleet_local.sh '"$PROFILE"' >/workspace/run.log 2>&1
  echo "$(date -u +%FT%TZ) DONE $(python scripts/worker.py --profile '"$PROFILE"' --queue-dir '"$QUEUE_DIR"' --build-only 2>/dev/null | tail -1)" > /workspace/DONE
  cp -f results/'"$PROFILE"'/index.csv /workspace/index.csv 2>/dev/null || true
  cp -rf results/'"$PROFILE"'/analysis /workspace/analysis 2>/dev/null || true
' >/dev/null 2>&1 &

status "fleet running; monitor /workspace/run.log and /workspace/DONE"
