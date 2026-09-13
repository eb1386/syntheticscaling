#!/usr/bin/env python3
"""Base pretraining to the stable-phase checkpoint θ*(S, seed_base).

    python scripts/pretrain_base.py configs/experiments/main/<base-only id>.yaml \
        --corpus data/processed/fineweb_edu/shard_000 data/processed/fineweb_edu/shard_001 \
        --micro-batch-seqs 16 --out-checkpoint $SYNSCALE_STORE/base/s100m_b1.pt

Shards are consumed as one seeded shuffle over all their sequences (nested prefixes of one
shuffle across sizes are obtained by using the same shard list and seed).
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
    add_run_args(parser, "--corpus", "document-stream stem(s) of the tokenised base corpus")
    args = parser.parse_args()
    cfg = load_experiment(args.experiment, repo_root=args.repo_root)
    manifest = run_phase(cfg, "base", inputs_from_args(args), options_from_args(args), repo_root=args.repo_root)
    print(f"{manifest.experiment_id}: {manifest.runtime.status} after {manifest.metrics['steps']} steps; "
          f"theta* at {manifest.metrics['checkpoint_path']} (sha256 {manifest.metrics['checkpoint_sha256'][:12]})")
    sys.exit(0 if manifest.runtime.status == "finished" else 1)


if __name__ == "__main__":
    main()
