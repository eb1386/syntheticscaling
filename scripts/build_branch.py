#!/usr/bin/env python3
"""Assemble the packed phase-2 data file for one condition (student_training.md §2.11).

    python scripts/build_branch.py --treatment data/processed/t7b.600m.eqtok \
        --replay data/processed/fineweb_edu/shard_000 data/processed/fineweb_edu/shard_001 \
        --size s100m --seed 1 --d2-tokens 800000000 --replay-fraction 0.25 \
        --out data/processed/branches/main.s100m.t7b.d600m.eqtok.s1

The slot pattern is a function of (seed, n_treat, n_rep) only, so every condition of a
size built with the same seed shares it; pass ``--slots`` to reuse a stored pattern.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synscale.training.data import BranchSpec, SequenceSource, ShardedSource, build_branch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--treatment", required=True, help="document-stream stem of the treatment file")
    parser.add_argument("--replay", nargs="+", required=True, help="document-stream stem(s) of the replay shards")
    parser.add_argument("--size", required=True, help="student id, e.g. s100m")
    parser.add_argument("--seed", type=int, required=True, help="phase seed")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--d2-tokens", type=int, default=800_000_000)
    parser.add_argument("--replay-fraction", type=float, default=0.25)
    parser.add_argument("--out", required=True, help="output stem")
    parser.add_argument("--slots", type=Path, default=None, help="existing slots .npy to reuse")
    args = parser.parse_args()

    spec = BranchSpec(size=args.size, seed=args.seed, seq_len=args.seq_len, d2_tokens=args.d2_tokens, replay_fraction=args.replay_fraction)
    treatment = SequenceSource.open(args.treatment, args.seq_len)
    replay = ShardedSource([SequenceSource.open(s, args.seq_len) for s in args.replay])
    slots = np.load(args.slots) if args.slots else None
    branch = build_branch(treatment, replay, spec, args.out, treatment_ref=args.treatment, replay_ref=",".join(args.replay), slots=slots)
    print(json.dumps({k: branch[k] for k in ("n_slots", "n_treatment", "n_replay", "treatment_tokens", "sha256", "slots_sha256")}, indent=2))


if __name__ == "__main__":
    main()
