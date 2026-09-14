#!/usr/bin/env bash
# Preflight: verify the vast.ai login works, show the balance, register an SSH key, and confirm a
# suitable multi-GPU H100 offer exists. Run this first, right after the user sets the API key.
#   vastai set api-key <KEY>        # user does this once (or it is set in the environment)
#   ./scripts/vast/preflight.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/common.sh"

echo "== 1. auth =="
vast_check_auth >/dev/null && echo "  OK: authenticated" || { echo "  FAIL: run 'vastai set api-key <KEY>'"; exit 1; }

echo "== 2. balance =="
echo "  \$$(vast_balance) USD on account"

echo "== 3. ssh key =="
echo "  registered: $(vast_ensure_key)"

echo "== 4. offers (main run needs a multi-GPU H100 box) =="
for q in 'gpu_name=H100_SXM num_gpus>=8 gpu_ram>=80 verified=true rentable=true reliability>0.98' \
         'gpu_name=H100_SXM num_gpus>=4 gpu_ram>=80 verified=true rentable=true reliability>0.98'; do
  echo "  on-demand [$q]:"
  echo "    $(pick_offer "$q" on-demand)"
  echo "  interruptible/bid:"
  echo "    $(pick_offer "$q" bid)"
done

echo "== 5. smoke offer (cheap single GPU for the \$1-5 sanity run) =="
echo "  $(pick_offer 'gpu_ram>=24 num_gpus=1 verified=true rentable=true dph_total<0.8' on-demand)"

echo
echo "Preflight done. If auth + balance + at least one H100 offer look good, run:"
echo "  ./scripts/vast/smoke.sh          # ~\$1-5, proves the code runs on a real GPU"
echo "  ./scripts/vast/launch.sh c4      # main run, only after smoke passes"
