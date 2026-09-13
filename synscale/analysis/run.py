"""Thin analysis driver: index -> cell table, floor gate, P_cc, transfer curves, T*, STE.

Produces CSV/Markdown tables and (if matplotlib is present) figures under ``out_dir``.
Best-effort: any sub-analysis that fails (e.g. too few teachers in a toy run) is skipped
with a note, so the pipeline never blocks on analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def analyze_index(index_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    import pandas as pd
    from synscale.tracking.index import load_index
    from synscale.analysis import aggregate as agg

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_index(index_path)
    report: dict[str, Any] = {"n_rows": len(df), "notes": []}
    if len(df) == 0:
        report["notes"].append("empty index"); (out_dir / "report.json").write_text(json.dumps(report, indent=2)); return report

    # floor gate on the smallest student's C0 rows
    try:
        gate = agg.gate_from_index(df)
        K = list(gate.retained) or list(agg.TIER_A)
        report["retained_tasks"] = K
        report["gate_status"] = gate.status
        report["smallest_student"] = gate.smallest_student
        gate.details.to_csv(out_dir / "floor_gate.csv", index=False)
    except Exception as e:
        K = list(agg.TIER_A); report["notes"].append(f"floor gate skipped: {e}")

    # aggregates + cell table
    try:
        df = agg.add_aggregates(df, K)
        cell = agg.cell_table(df, outcome="p_cc")
        cell.to_csv(out_dir / "cell_table.csv", index=False)
        report["cell_table"] = str(out_dir / "cell_table.csv")
        bands = agg.control_bands(df, outcome="p_cc")
        bands.to_csv(out_dir / "control_bands.csv", index=False)
    except Exception as e:
        report["notes"].append(f"aggregates skipped: {e}")

    # per-student T* classification + STE (need >=3 teacher levels)
    try:
        from synscale.analysis.tstar import classify_all
        tab, results = classify_all(df, outcome="p_cc")
        tab.to_csv(out_dir / "tstar.csv", index=False)
        report["tstar"] = {s: r.label for s, r in results.items()}
    except Exception as e:
        report["notes"].append(f"tstar skipped: {e}")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
