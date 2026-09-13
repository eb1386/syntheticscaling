#!/usr/bin/env python3
"""Build the index and run the analysis (tables + figures) for a profile's results.

    python scripts/analyze.py --results-dir results/local5080
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synscale.tracking.index import build_index
from synscale.analysis.run import analyze_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--figures", action="store_true", help="also render figures (needs matplotlib)")
    a = ap.parse_args()
    rd = Path(a.results_dir)
    idx = build_index(rd, out_path=rd / "index.csv")
    rep = analyze_index(idx, out_dir=rd / "analysis")
    print("index:", idx)
    print("retained tasks:", rep.get("retained_tasks"))
    print("tstar:", rep.get("tstar"))
    if a.figures:
        try:
            from synscale.tracking.index import load_index
            from synscale.analysis import aggregate as agg, plots
            df = load_index(idx)
            gate = agg.gate_from_index(df); K = list(gate.retained) or list(agg.TIER_A)
            df = agg.add_aggregates(df, K)
            plots.transfer_curves(df, rd / "analysis"); plots.sxt_heatmap(df, rd / "analysis")
            print("figures ->", rd / "analysis")
        except Exception as e:
            print("figures skipped:", e)


if __name__ == "__main__":
    main()
