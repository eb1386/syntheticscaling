#!/usr/bin/env bash
# Shared helpers for the vast.ai launch/monitor/teardown scripts. Source this, do not run it.
# Every function is defensive about money: creates always pair with a wait-and-verify that has a
# timeout and destroys the instance on any terminal status, so a bad rental never bills silently.
set -uo pipefail

vast() { vastai "$@" --raw; }                       # always machine-readable JSON

# --- auth + balance -----------------------------------------------------------------------------
vast_check_auth() {
  local out; out=$(vastai show user --raw 2>&1) || { echo "AUTH FAIL: $out" >&2; return 1; }
  echo "$out"
}
# credit balance in USD (positive = money available). Reads several possible field names.
vast_balance() {
  vastai show user --raw 2>/dev/null | python3 -c '
import sys, json
try: d = json.load(sys.stdin)
except Exception: print("nan"); sys.exit()
for k in ("credit","balance","current_balance"):
    if isinstance(d, dict) and d.get(k) is not None:
        print(float(d[k])); break
else: print("nan")'
}
# Refuse to proceed unless at least $MIN is on the account.
vast_require_balance() {
  local min="$1"; local bal; bal=$(vast_balance)
  echo "account balance: \$$bal (need >= \$$min)"
  python3 -c "import sys; sys.exit(0 if (str('$bal')!='nan' and float('$bal')>=float('$min')) else 1)" \
    || { echo "INSUFFICIENT BALANCE. Add credit before launching." >&2; return 1; }
}

# --- SSH key ------------------------------------------------------------------------------------
vast_ensure_key() {
  local kf="${HOME}/.ssh/id_ed25519"
  [ -f "${kf}.pub" ] || ssh-keygen -t ed25519 -N "" -f "$kf" >/dev/null 2>&1
  vastai create ssh-key "$(cat "${kf}.pub")" --raw >/dev/null 2>&1 || true   # idempotent
  echo "${kf}.pub"
}

# --- offer selection ----------------------------------------------------------------------------
# pick_offer "<query>" <type on-demand|bid> -> prints "OFFER_ID DPH GPU_NAME NUM_GPUS"
pick_offer() {
  local query="$1"; local otype="${2:-on-demand}"
  vastai search offers "$query" --type "$otype" -o 'dph_total' --raw 2>/dev/null | python3 -c '
import sys, json
rows = json.load(sys.stdin)
if not rows: print("NONE"); sys.exit()
r = rows[0]
print(r["id"], round(float(r.get("dph_total",0)),3), r.get("gpu_name","?"), r.get("num_gpus","?"))'
}

# --- wait for running (money-safe) --------------------------------------------------------------
# wait_running <instance_id> <timeout_sec>; destroys and returns 1 on any terminal/failed status.
wait_running() {
  local id="$1"; local timeout="${2:-900}"; local waited=0
  while [ "$waited" -lt "$timeout" ]; do
    local st; st=$(vastai show instance "$id" --raw 2>/dev/null | python3 -c '
import sys,json
try: d=json.load(sys.stdin)
except Exception: print("unknown"); sys.exit()
print((d.get("actual_status") or d.get("cur_state") or "null"))' )
    echo "  instance $id status: $st (${waited}s)"
    case "$st" in
      running) return 0 ;;
      exited|offline|unknown) echo "TERMINAL status '$st' -> destroying $id" >&2; vastai destroy instance "$id" -y >/dev/null 2>&1; return 1 ;;
    esac
    sleep 15; waited=$((waited+15))
  done
  echo "TIMEOUT after ${timeout}s -> destroying $id" >&2; vastai destroy instance "$id" -y >/dev/null 2>&1; return 1
}

# run a short command on the instance via the vast API (HTTPS, proxy-friendly)
vast_exec() { vastai execute "$1" "$2" 2>/dev/null; }

# create instance from an offer id + onstart file; prints new instance id or empty on failure
vast_create() {
  local offer="$1"; local disk="$2"; local onstart="$3"; local label="$4"; shift 4
  local extra=("$@")   # e.g. --bid_price 1.20  or  --env '-e HF_TOKEN=..'
  vastai create instance "$offer" --image "$VAST_IMAGE" --disk "$disk" --ssh --direct \
    --onstart "$onstart" --label "$label" "${extra[@]}" --raw 2>/dev/null | python3 -c '
import sys,json
try: d=json.load(sys.stdin)
except Exception: print(""); sys.exit()
print(d.get("new_contract","") if d.get("success",True) else "")'
}

# --- upload code + release the box --------------------------------------------------------------
# Ships tracked repo files (no secrets, no data/ or results/) to the instance, writes the profile
# and optional HF token, then touches GO so the onstart hook proceeds.
upload_and_go() {
  local id="$1"; local profile="$2"; local hf="${3:-}"
  local tgz="${SCRATCH:-/tmp}/synscale-repo.tgz"
  ( cd "$REPO_ROOT" && git archive --format=tar.gz -o "$tgz" HEAD )
  echo "  uploading code ($(du -h "$tgz" | cut -f1)) to $id"
  vastai copy "local:$tgz" "$id:/workspace/repo.tgz"
  printf '%s' "$profile" > "${SCRATCH:-/tmp}/PROFILE" && vastai copy "local:${SCRATCH:-/tmp}/PROFILE" "$id:/workspace/PROFILE"
  if [ -n "$hf" ]; then
    printf '%s' "$hf" > "${SCRATCH:-/tmp}/HF_TOKEN" && vastai copy "local:${SCRATCH:-/tmp}/HF_TOKEN" "$id:/workspace/HF_TOKEN"
    rm -f "${SCRATCH:-/tmp}/HF_TOKEN"
  fi
  vast_exec "$id" "touch /workspace/GO" || vastai execute "$id" "touch /workspace/GO" >/dev/null 2>&1
  echo "  GO sent to $id"
}

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${VAST_IMAGE:=vastai/pytorch:@vastai-automatic-tag}"
