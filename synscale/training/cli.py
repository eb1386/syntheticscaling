"""Shared argument handling for the training CLIs in ``scripts/``."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from synscale.training.trainer import RunInputs, RuntimeOptions


def add_run_args(parser: argparse.ArgumentParser, data_flag: str, data_help: str) -> None:
    parser.add_argument("experiment", help="experiment YAML (composed by synscale.config.loader)")
    parser.add_argument(data_flag, dest="data", nargs="+", required=True, help=data_help)
    parser.add_argument("--micro-batch-seqs", type=int, required=True, help="sequences per micro-batch; fixed per (size, GPU)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--store-dir", type=Path, default=Path(os.environ.get("SYNSCALE_STORE", "checkpoints")))
    parser.add_argument("--out-checkpoint", type=Path, default=None, help="final checkpoint path (default: <store>/<id>/<phase>_final.pt)")
    parser.add_argument("--eval-set", action="append", default=[], metavar="NAME=STEM", help="held-out set (repeatable)")
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume-every-steps", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--peak-tflops", type=float, default=None, help="dense bf16 peak for the MFU estimate")
    parser.add_argument("--max-steps", type=int, default=None, help="cap the step count (smoke tests only)")
    parser.add_argument("--repo-root", default=None, help="repo root for fragment paths and git info (default: YAML's directory)")


def parse_eval_sets(pairs: list[str]) -> dict[str, str]:
    sets = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--eval-set expects NAME=STEM, got {pair!r}")
        name, stem = pair.split("=", 1)
        sets[name] = stem
    return sets


def inputs_from_args(args: argparse.Namespace, base_checkpoint: Path | None = None) -> RunInputs:
    return RunInputs(
        data=list(args.data), results_dir=args.results_dir, store_dir=args.store_dir, base_checkpoint=base_checkpoint,
        eval_sets=parse_eval_sets(args.eval_set), out_checkpoint=args.out_checkpoint,
    )


def options_from_args(args: argparse.Namespace) -> RuntimeOptions:
    opts = RuntimeOptions(
        micro_batch_seqs=args.micro_batch_seqs, eval_sequences=args.eval_sequences, log_every=args.log_every,
        resume_every_steps=args.resume_every_steps, resume_from=args.resume_from, peak_tflops=args.peak_tflops,
        max_steps=args.max_steps,
    )
    if args.device:
        opts.device = args.device
    return opts
