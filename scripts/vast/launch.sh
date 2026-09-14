#!/usr/bin/env bash
# Launch the MAIN study on ONE multi-GPU vast.ai instance (local disk = the shared volume; one
# GPU-pinned fleet worker per GPU). On-demand by default for reliability; pass --bid <price/hr> for
# a cheaper interruptible box (the whole box stops if outbid, but disk is preserved and the run
# resumes from checkpoints). Does not block: prints the instance id, then use monitor.sh.
#   ./scripts/vast/launch.sh [PROFILE] [--gpus 8] [--bid 12.0] [--hf <HF_TOKEN>] [--disk 600]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"
export SCRATCH="${SCRATCH:-/tmp}"

PROFILE="c4"; GPUS=8; BID=""; HF="${HF_TOKEN:-}"; DISK=600
args=("$@")
[ "${#args[@]}" -gt 0 ] && [[ "${args[0]}" != --* ]] && { PROFILE="${args[0]}"; args=("${args[@]:1}"); }
while [ "${#args[@]}" -gt 0 ]; do
  case "${args[0]}" in
    --gpus) GPUS="${args[1]}"; args=("${args[@]:2}");;
    --bid)  BID="${args[1]}";  args=("${args[@]:2}");;
    --hf)   HF="${args[1]}";   args=("${args[@]:2}");;
    --disk) DISK="${args[1]}"; args=("${args[@]:2}");;
    *) echo "unknown arg ${args[0]}"; exit 1;;
  esac
done

# Llama-3.x teachers are gated on Hugging Face; C4 needs a token with Llama access.
if [ "$PROFILE" = "c4" ] && [ -z "$HF" ]; then
  echo "WARNING: C4 uses gated Llama-3.x teachers. Pass --hf <HF_TOKEN> (with Llama access) or the"
  echo "         Llama generation jobs will fail. Qwen-only profiles do not need it."
fi

MIN_USD="${MIN_USD:-50}"
vast_require_balance "$MIN_USD" || exit 1
echo "NOTE: full C4 hybrid estimate is ~\$1,900 USD (~\$2,650 CAD) + ~25% for preemption. Your finite"
echo "      balance is the hard cap: if it runs out the box stops (disk kept) and the run resumes on top-up."
KEY=$(vast_ensure_key); echo "ssh key: $KEY"

OTYPE="on-demand"; EXTRA=()
[ -n "$BID" ] && { OTYPE="bid"; EXTRA=(--bid_price "$BID"); echo "interruptible bid: \$$BID/hr (whole box)"; }
QUERY="gpu_name=H100_SXM num_gpus>=$GPUS gpu_ram>=80 verified=true rentable=true reliability>0.98"

echo "== picking a ${GPUS}x H100 box ($OTYPE) =="
read -r OFFER DPH GPU NG < <(pick_offer "$QUERY" "$OTYPE")
{ [ "$OFFER" = "NONE" ] || [ -z "$OFFER" ]; } && { echo "no ${GPUS}xH100 offer; try --gpus 4"; exit 1; }
echo "  offer $OFFER : \$$DPH/hr $GPU x$NG"

echo "== creating instance (disk ${DISK}GB) =="
ID=$(vast_create "$OFFER" "$DISK" "$HERE/onstart.sh" "synscale-$PROFILE" "${EXTRA[@]}")
[ -z "$ID" ] && { echo "create failed"; exit 1; }
echo "  instance $ID"

if ! wait_running "$ID" 1200; then echo "instance never reached running (destroyed)"; exit 1; fi
upload_and_go "$ID" "$PROFILE" "$HF"

echo
echo "MAIN RUN LAUNCHED on instance $ID ($GPU x$NG at \$$DPH/hr)."
echo "$ID" > "$REPO_ROOT/.vast_instance"
echo "Monitor with:   ./scripts/vast/monitor.sh $ID"
echo "Tear down with: ./scripts/vast/teardown.sh $ID     (do this the moment it is DONE)"
