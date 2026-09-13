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

    # scaling-law fits (loss vs synthetic tokens) — the core investigation (docs/22)
    try:
        from synscale.analysis.scaling import run_scaling_analysis
        results_dir = Path(index_path).parent
        for es in ("instr", "base_heldout"):
            rep = run_scaling_analysis(results_dir, out_dir=out_dir / f"scaling_{es}", eval_set=es)
            report[f"scaling_{es}"] = {k: rep.get(k) for k in ("n_runs_with_curves", "note")}
    except Exception as e:
        report["notes"].append(f"scaling analysis skipped: {e}")

    # NOVELTY analyses (docs/24): screening (q vs identity), predictive law, compute allocation
    try:
        import json as _json
        from synscale.analysis import novelty, scaling as _sc, data_properties as _dp
        results_dir = Path(index_path).parent
        dp_path = results_dir / "data_props.json"
        recs = _sc.collect_curves(results_dir, eval_set="instr")
        fits = _sc.fit_cells(recs) if recs else []
        # attach q from data_props.json if it exists
        if dp_path.exists() and fits:
            props = {(r["student"], r["teacher"]): r for r in _dp.quality_index(_json.loads(dp_path.read_text()))}
            cells = []
            for f in fits:
                if not f.get("t_params"):
                    continue
                pr = props.get((f["student"], f["teacher"]), {})
                cells.append({**f, "family": pr.get("family", "qwen2.5"), "q": pr.get("q")})
            nrep = {"screening": novelty.screening_test(cells, outcome="L_inf_hat")}
        else:
            nrep = {"note": "data_props.json not found or no fits; run with measure_data_properties"}
        # predictive law + allocation from the raw curves / fits (no q needed)
        by_cell = {}
        for r in recs:
            by_cell.setdefault((r["student"], r["cond"]), []).extend(r["curve"])
        nrep["predictive_law"] = novelty.predictive_law({k: v for k, v in by_cell.items() if k[1] and k[1][0] in "tl"})
        fit_by_cell = {(f["student"], f["teacher"]): {**f["_fit"], "t_params": f["t_params"]}
                       for f in fits if f.get("t_params")}
        student_params = {f["student"]: f["s_params"] for f in fits if f.get("s_params")}
        nrep["compute_allocation"] = novelty.compute_allocation_frontier(fit_by_cell, student_params)
        (out_dir / "novelty_report.json").write_text(json.dumps(nrep, indent=2, default=str))
        report["novelty"] = "written to novelty_report.json"
    except Exception as e:
        report["notes"].append(f"novelty analysis skipped: {e}")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
