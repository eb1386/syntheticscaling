#!/usr/bin/env python3
"""Train one phase of an experiment.

    python scripts/train.py configs/experiments/main/<id>.yaml --phase synthetic \
        --data data/processed/branches/<id> --base-checkpoint $SYNSCALE_STORE/base/s100m_b1.pt \
        --micro-batch-seqs 16 --eval-set base_heldout=data/processed/heldout_base

``--data`` takes the packed branch stem for the synthetic phase (built by
scripts/build_branch.py) or one or more document-stream stems for the base phase.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synscale.config.loader import load_experiment  # noqa: E402
from synscale.training.cli import add_run_args, inputs_from_args, options_from_args  # noqa: E402
from synscale.training.trainer import run_phase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(parser, "--data", "branch stem (synthetic) or document-stream stem(s) (base)")
    parser.add_argument("--phase", choices=["base", "synthetic"], default="synthetic")
    parser.add_argument("--base-checkpoint", type=Path, default=None, help="θ* checkpoint (required for --phase synthetic)")
    args = parser.parse_args()
    cfg = load_experiment(args.experiment, repo_root=args.repo_root)
    manifest = run_phase(cfg, args.phase, inputs_from_args(args, args.base_checkpoint), options_from_args(args), repo_root=args.repo_root)
    print(f"{manifest.experiment_id}: {manifest.runtime.status} after {manifest.metrics['steps']} steps; "
          f"final loss {manifest.metrics['final_train_loss']:.4f}; checkpoint {manifest.metrics['checkpoint_path']}")
    sys.exit(0 if manifest.runtime.status == "finished" else 1)


if __name__ == "__main__":
    main()
