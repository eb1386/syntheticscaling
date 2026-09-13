#!/usr/bin/env python3
"""Generate the experiment YAMLs for a profile and print the run counts (no training).

    python scripts/make_matrix.py --profile local5080
    python scripts/make_matrix.py --profile local5080 --write
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synscale.config.profiles import get_profile
from synscale.pipeline import REPO_ROOT, enumerate_branches, make_experiment_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="local5080")
    ap.add_argument("--write", action="store_true", help="write experiment YAMLs under configs/experiments/<profile>/")
    a = ap.parse_args()
    p = get_profile(a.profile)
    jobs = enumerate_branches(p)
    out_dir = REPO_ROOT / "configs" / "experiments" / p.name

    # base runs
    n_base = sum(p.base_seeds[s] for s in p.students)
    print(f"profile={p.name}")
    print(f"  students        : {list(p.students)}")
    print(f"  teachers        : {[t.name for t in p.teachers]}")
    print(f"  base runs (theta*): {n_base}")
    print(f"  branch runs     : {len(jobs)}")
    print(f"  total training runs: {n_base + len(jobs)}")
    by_cond = Counter(j.cond for j in jobs)
    print(f"  branches by condition: {dict(sorted(by_cond.items()))}")
    by_student = Counter(j.student for j in jobs)
    print(f"  branches by student : {dict(sorted(by_student.items()))}")

    if a.write:
        n = 0
        for s in p.students:
            for b in range(1, p.base_seeds[s] + 1):
                make_experiment_yaml(p, student=s, arm="base_only", seed_base=b, seed_phase=1,
                                     d_syn=0, variant="-", out_dir=out_dir); n += 1
        for j in jobs:
            make_experiment_yaml(p, student=j.student, arm=j.arm, seed_base=j.seed_base,
                                 seed_phase=j.seed_phase, teacher=j.teacher, d_syn=j.d_syn,
                                 variant=j.variant, out_dir=out_dir); n += 1
        print(f"  wrote {n} experiment YAMLs -> {out_dir}")


if __name__ == "__main__":
    main()
