#!/usr/bin/env bash
# One-command setup for the synthetic-scaling study.
# Usage:  ./install.sh [PROFILE]        (PROFILE default: c4)
# Installs Python deps, downloads the tokenizer + corpora + benchmarks, pre-fetches the
# teacher models (Qwen2.5 + Llama-3.x, int4/AWQ), and builds the frozen prompt pool.
# Re-run safe: each step is skipped if its output already exists.
# On vast.ai the intended path is a single multi-GPU box; local disk under /workspace holds the
# data, checkpoints, and the fleet queue, and the on-box fleet (scripts/fleet_local.sh) runs one
# GPU-pinned worker per card. Gated Llama teachers need HF_TOKEN in the environment.
set -euo pipefail
PROFILE="${1:-c4}"
PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "==> [1/5] Python environment"
if [ ! -d .venv ]; then "$PYTHON" -m venv --system-site-packages .venv; fi  # inherit the image's CUDA torch
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "==> [2/5] Core + training + generation + eval + analysis dependencies"
# Only install torch if the image did not already provide a working CUDA build (the vastai/pytorch
# image does). Reinstalling risks clobbering the image's matched CUDA torch, so we avoid it.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "    no working CUDA torch found; installing one (cu124 build)"
  pip install torch --index-url https://download.pytorch.org/whl/cu124 || pip install torch
else
  echo "    using pre-installed CUDA torch: $(python -c 'import torch;print(torch.__version__)')"
fi
pip install -e ".[train,gen,eval,analysis,dev]" || pip install \
  pydantic pyyaml numpy tokenizers safetensors datasets transformers \
  "vllm>=0.6" "lm-eval>=0.4.5" pandas scipy statsmodels matplotlib scikit-learn pytest

echo "==> [3/5] Sanity: unit tests (CPU; torch/vllm tests skip if unavailable)"
python -m pytest -q || { echo "tests failed — stop and inspect before a multi-day run"; exit 1; }

echo "==> [4/5] Download + tokenise data (tokenizer, FineWeb-Edu, StackExchange, Dolly/OASST, benchmarks)"
python scripts/prepare_data.py --profile "$PROFILE"

echo "==> [5/5] Pre-fetch teacher models and build the frozen prompt pool"
python - "$PROFILE" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
from synscale.config.profiles import get_profile
tok = os.environ.get("HF_TOKEN") or None    # gated Llama-3.x repos require this
p = get_profile(sys.argv[1])
for t in p.teachers:
    print(f"[install] fetching {t.hf_repo} ...")
    try:
        snapshot_download(t.hf_repo, token=tok)
    except Exception as e:
        print(f"[install] WARNING could not prefetch {t.hf_repo}: {e}\n"
              f"          Gated Llama repos need HF_TOKEN with access granted at huggingface.co.\n"
              f"          If a repo 404s, swap the -AWQ repo for -GPTQ-Int4 in configs/teachers or the profile.")
PY
python -c "from synscale.pipeline import build_pool_stage; from synscale.config.profiles import get_profile; import pathlib; print('pool:', build_pool_stage(get_profile('$PROFILE'), pathlib.Path('data')))"

echo "==> install complete. Estimate the run time with:  make budget PROFILE=$PROFILE"
echo "    then start the experiment with:               ./run_all.sh $PROFILE"
